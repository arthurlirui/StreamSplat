"""
Perspective Gaussian rasterizer with metric (real-world) scale support.

This is a drop-in replacement for `gaussian_renderer_dynamic/__init__.py`
which uses an orthographic projection with a fixed, normalized camera. The
key differences are:

1. The projection matrix is built from real intrinsics (fx, fy, cx, cy) so
   that focal length and principal point are honored.
2. The view matrix is per-camera, allowing a stereo rig (left/right) and a
   moving rig (per-frame pose) to be rendered correctly.
3. `xyz_static` output by the network is interpreted as (du, dv, dZ) pixel /
   metric-depth offsets around the metric depth derived from stereo disparity,
   and `MetricHead` back-projects them into the metric world frame.

The dynamic extensions of the original orthographic rasterizer
(`scales_t`, `rotations_r`, `flow_2d`, time-basis polynomial motion, time-
dependent opacity decay) are preserved verbatim so the existing dynamic
decoder (`SplatModel` / `SplatPredictor`) keeps working.

We intentionally keep the same `render(gaussians, bg_color, ...)` signature
so that callers can switch between orthographic and perspective by changing
the imported module.
"""
from __future__ import annotations
import math
import os
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the orthographic CUDA extension that already ships with StreamSplat.
# Its forward path accepts a generic `viewmatrix` / `projmatrix`, so a
# perspective projection matrix is a valid substitution; the orthographic
# naming in the import is a historical artifact of the original codebase.
from diff_gaussian_rasterization_kiui_orth import (
    GaussianRasterizationSettings as GaussianRasterizationSettingsPersp,
    GaussianRasterizer as GaussianRasterizerPersp,
)

from configs.options import Options


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------
def getWorld2View2(R: np.ndarray, t: np.ndarray,
                   translate: np.ndarray = np.array([0.0, 0.0, 0.0]),
                   scale: float = 1.0) -> np.ndarray:
    """World-to-view matrix from camera rotation R (c2w rows) and translation t."""
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getPerspectiveProjectionMatrix(fx: float, fy: float,
                                   cx: float, cy: float,
                                   H: int, W: int,
                                   znear: float = 0.01,
                                   zfar: float = 100.0) -> np.ndarray:
    """OpenGL-style perspective projection from pixel-space intrinsics.

    The resulting 4x4 matrix maps the camera-space cube [-1, 1]^3 to clip
    space, matching the convention expected by the diff-gaussian-rasterizer
    CUDA kernel.
    """
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = 2.0 * cx / W - 1.0
    P[1, 2] = 2.0 * cy / H - 1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


class CameraPose:
    """A simple pinhole-camera pose bundle (intrinsics + extrinsics)."""

    def __init__(self, K: np.ndarray, R_c2w: np.ndarray, t_c2w: np.ndarray,
                 H: int, W: int, znear: float = 0.01, zfar: float = 100.0):
        self.K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        self.R_c2w = np.asarray(R_c2w, dtype=np.float32).reshape(3, 3)
        self.t_c2w = np.asarray(t_c2w, dtype=np.float32).reshape(3)
        self.H, self.W = int(H), int(W)
        self.znear, self.zfar = float(znear), float(zfar)

        # w2c from c2w
        R_w2c = self.R_c2w.T
        t_w2c = -R_w2c @ self.t_c2w
        self.view_matrix = getWorld2View2(R_w2c, t_w2c)  # 4x4
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        self.proj_matrix = getPerspectiveProjectionMatrix(
            fx, fy, cx, cy, self.H, self.W, self.znear, self.zfar)
        # Camera center in world frame (for the kernel's `campos`).
        self.campos = (self.R_c2w @ self.t_c2w).astype(np.float32)

    def to(self, device):
        return {
            "viewmatrix": torch.as_tensor(self.view_matrix, device=device).float(),
            "projmatrix": torch.as_tensor(self.proj_matrix, device=device).float(),
            "campos": torch.as_tensor(self.campos, device=device).float(),
            "full_proj": (torch.as_tensor(self.view_matrix, device=device).float()
                          @ torch.as_tensor(self.proj_matrix, device=device).float()),
        }


# ---------------------------------------------------------------------------
# Default camera (left camera of a static stereo rig at the origin).
# Used when no per-frame cameras are supplied, to keep the API compatible
# with the orthographic renderer's default-identity camera.
# ---------------------------------------------------------------------------
def _default_left_camera(H: int, W: int, fx: float, fy: float,
                         cx: float, cy: float) -> CameraPose:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    R_c2w = np.eye(3, dtype=np.float32)
    t_c2w = np.zeros(3, dtype=np.float32)
    return CameraPose(K, R_c2w, t_c2w, H, W)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(gaussians: dict,
           bg_color: torch.Tensor,
           timestamps: torch.Tensor = None,
           scaling_modifier: float = 1.0,
           opt: Any = None,
           anchor_time: torch.Tensor = None,
           training: bool = True,
           override_opacity: bool = False,
           cameras: Optional[list] = None,
           camera_index: int = 0):
    """Render a batch of Gaussians with a perspective pinhole camera.

    Parameters
    ----------
    gaussians : dict
        Same layout as the orthographic renderer's input. `xyz` is expected to
        already be in metric world coordinates (use `MetricHead` upstream).
    cameras : list of CameraPose, optional
        One CameraPose per batch element / view. If None, a default left
        camera built from `opt` intrinsics is used (static-rig assumption).
    camera_index : int
        Index into `cameras` to use for this render call. For stereo training
        the caller renders left and right sequentially with camera_index=0/1.
    """
    if training:
        bg_color = torch.rand(3, device=gaussians['xyz'].device)
    else:
        bg_color = torch.tensor([0.5, 0.5, 0.5], device=gaussians['xyz'].device)

    L = 0
    LP = opt.forder

    batch_size, N = gaussians['xyz'].shape[0], gaussians['xyz'].shape[1]

    screenspace_points = torch.zeros_like(
        gaussians['xyz'][:, :, 0, :], dtype=gaussians['xyz'].dtype,
        requires_grad=True, device=gaussians['xyz'].device)
    screenspace_points.retain_grad()

    if len(opt.down_resolution) > 0:
        render_height, render_width = opt.down_resolution
    else:
        render_height, render_width = opt.image_height, opt.image_width

    # Resolve the camera for this render.
    if cameras is None:
        fx = getattr(opt, "fx", 500.0)
        fy = getattr(opt, "fy", 500.0)
        cx = getattr(opt, "cx", render_width / 2.0)
        cy = getattr(opt, "cy", render_height / 2.0)
        cam = _default_left_camera(render_height, render_width, fx, fy, cx, cy)
        cam_tensors = cam.to(gaussians['xyz'].device)
    else:
        cam = cameras[camera_index]
        if isinstance(cam, CameraPose):
            cam_tensors = cam.to(gaussians['xyz'].device)
        else:
            cam_tensors = cam  # already a dict of tensors

    tanfovx = 1.0 / (getattr(opt, "fx", 500.0) / render_width) if not isinstance(cam, CameraPose) \
        else (cam.W / (2.0 * cam.K[0, 0]))
    tanfovy = 1.0 / (getattr(opt, "fy", 500.0) / render_height) if not isinstance(cam, CameraPose) \
        else (cam.H / (2.0 * cam.K[1, 1]))

    raster_settings = GaussianRasterizationSettingsPersp(
        image_height=render_height,
        image_width=render_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=cam_tensors["viewmatrix"],
        projmatrix=cam_tensors["full_proj"],
        sh_degree=0,
        campos=cam_tensors["campos"],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizerPersp(raster_settings=raster_settings)

    render_images, render_depths, render_alphas = [], [], []
    dummy_time = torch.zeros(1, device=gaussians['xyz'].device)
    output_frames = opt.output_frames
    if timestamps is None:
        output_frames = 1
        timestamps = dummy_time.repeat(batch_size, output_frames)
    if anchor_time is None:
        anchor_time = torch.zeros((N, 1), device=gaussians['xyz'].device)
    else:
        anchor_time = anchor_time.repeat_interleave(N // 2).unsqueeze(-1)

    for b in range(batch_size):
        means3D_static = gaussians['xyz'][b, :, 0, :].contiguous().float()
        dynamic_components = gaussians['xyz'][b, :, 1:, :].contiguous().float()
        opacity = gaussians['opacity'][b, :, :1].contiguous().float()
        opacity_dynamic = (gaussians['opacity'][b, :, 1:].contiguous().float()
                           if gaussians['opacity'].shape[2] > 1 else None)
        scales = gaussians['scale'][b, :, :].contiguous().float()
        y_scale = scales.mean(dim=-1, keepdim=True)
        scales = torch.cat([scales[..., 0:1], y_scale, scales[..., 1:2]], dim=-1)
        rotations_static = gaussians['rot'][b, :, 0, :].contiguous().float()
        rotations_dynamic = gaussians['rot'][b, :, 1, :].contiguous().float()
        colors_precomp = gaussians['rgb'][b, :, :].contiguous().float()

        for tidx in range(output_frames):
            actual_time = timestamps[b, tidx]
            time_basis = actual_time - anchor_time
            if LP > 0:
                polynomial_basis = torch.stack(
                    [time_basis ** i for i in range(1, LP + 1)], dim=1)
                dynamic_poly = (dynamic_components[:, 2 * L:, :] * polynomial_basis).sum(dim=1)
            else:
                dynamic_poly = torch.zeros_like(means3D_static)

            dynamic_means = dynamic_poly
            final_means3D = means3D_static + dynamic_means

            # NOTE: no `pred_inverse` rescaling here — `xyz` is already metric.
            final_rotations = rotations_static + rotations_dynamic * time_basis

            if opacity_dynamic is not None:
                opacity_dynamic_coef = torch.sigmoid(
                    -opacity_dynamic[:, 0:1] * (time_basis.abs() - opacity_dynamic[:, 1:])
                ) / torch.sigmoid(opacity_dynamic[:, 0:1] * opacity_dynamic[:, 1:])
            else:
                opacity_dynamic_coef = torch.ones_like(opacity)
            if override_opacity:
                opacity_dynamic_coef = (time_basis.abs() <= 0.5).float().to(
                    opacity_dynamic_coef.device)
            final_opacity = opacity * opacity_dynamic_coef

            rendered_image, radii, depth, alpha = rasterizer(
                means3D=final_means3D,
                means2D=torch.zeros_like(final_means3D, dtype=torch.float32),
                shs=None,
                colors_precomp=colors_precomp,
                opacities=final_opacity,
                scales=scales,
                rotations=final_rotations,
            )
            render_images.append(rendered_image)
            render_depths.append(depth)
            render_alphas.append(alpha)

    render_images = torch.stack(render_images, dim=0).view(
        batch_size, output_frames, 3, render_images[0].shape[-2], render_images[0].shape[-1])
    render_depths = torch.stack(render_depths, dim=0).view(
        batch_size, output_frames, 1, render_depths[0].shape[-2], render_depths[0].shape[-1])
    render_alphas = torch.stack(render_alphas, dim=0).view(
        batch_size, output_frames, 1, render_depths[0].shape[-2], render_depths[0].shape[-1])

    return {
        "render": render_images,
        "depth": render_depths,
        "alpha": render_alphas,
    }
