"""
Metric back-projection head for StereoSplat.

Takes the per-pixel Gaussian attributes produced by the network
(`xyz_static` interpreted as pixel/depth offsets, `scale` in normalized
units) and lifts them into the metric world frame using:

  - the left-camera intrinsics (fx, fy, cx, cy)
  - the stereo-derived metric depth Z (meters)
  - the left-camera pose (R_c2w, t_c2w)

The output `gaussians['xyz']` is in meters in the world frame, and
`gaussians['scale']` is in meters. This makes the downstream perspective
rasterizer (gaussian_renderer_perspective) render at true metric scale.

Pixel grid convention: (u, v) with u along +x (column), v along +y (row).
The image plane follows the standard pinhole convention
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = Z
then transformed to world via (R_c2w, t_c2w).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def make_pixel_grid(H: int, W: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Return a [H*W, 2] tensor of (u, v) pixel centers."""
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    return torch.stack([u + 0.5, v + 0.5], dim=-1).reshape(-1, 2)  # [N, 2]


def metric_backproject(xyz_static: torch.Tensor,
                       scale: torch.Tensor,
                       depth_metric: torch.Tensor,
                       K: torch.Tensor,
                       R_c2w: torch.Tensor,
                       t_c2w: torch.Tensor,
                       pixel_grid: torch.Tensor,
                       scale_range: tuple = (0.001, 0.3)) -> dict:
    """Back-project per-pixel Gaussians into the metric world frame.

    Parameters
    ----------
    xyz_static : [B, N, 3]
        Network output for the `xyz_static` key. The first two channels are
        interpreted as sub-pixel (du, dv) offsets; the third channel is a
        relative depth residual dZ/Z in [-1, 1] applied multiplicatively.
    scale : [B, N, 2]
        Network output for the `scale` key (in-plane scales).
    depth_metric : [B, N, 1]
        Stereo-derived metric depth (meters), one value per pixel.
    K : [B, 3, 3]
        Left-camera intrinsics.
    R_c2w, t_c2w : [B, 3, 3], [B, 3]
        Left-camera pose (camera-to-world).
    pixel_grid : [N, 2]
        (u, v) pixel centers, broadcastable to batch.
    scale_range : tuple
        (min, max) for the in-plane Gaussian scale, in meters.

    Returns
    -------
    dict with keys `means3D` ([B,N,3]) and `scales3D` ([B,N,3]).
    """
    B, N, _ = xyz_static.shape
    device = xyz_static.device
    dtype = xyz_static.dtype

    du = xyz_static[..., 0]   # [B, N]
    dv = xyz_static[..., 1]
    dz_rel = xyz_static[..., 2]  # multiplicative residual

    u = pixel_grid[..., 0].unsqueeze(0).expand(B, -1)  # [B, N]
    v = pixel_grid[..., 1].unsqueeze(0).expand(B, -1)

    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    cx = K[:, 0, 2].view(B, 1)
    cy = K[:, 1, 2].view(B, 1)

    Z = depth_metric[..., 0] * (1.0 + 0.1 * dz_rel)  # [B, N], meters
    Z = Z.clamp(min=0.05)

    X = (u + du - cx) * Z / fx
    Y = (v + dv - cy) * Z / fy
    pts_cam = torch.stack([X, Y, Z], dim=-1)  # [B, N, 3] in left-camera frame

    # camera-to-world
    pts_world = torch.einsum('bij,bnj->bni', R_c2w, pts_cam) + t_c2w.unsqueeze(1)
    # NOTE: einsum over R_c2w treats it as the c2w rotation directly.

    # Scale: convert normalized scale to meters proportional to depth.
    s_min, s_max = scale_range
    s_in = s_min + (s_max - s_min) * torch.sigmoid(scale[..., 0])  # [B, N]
    s_out = s_min + (s_max - s_min) * torch.sigmoid(scale[..., 1])
    # Depth scale ~ proportional to Z (so distant Gaussians are larger).
    s_z = (0.05 * Z).clamp(min=s_min, max=s_max)
    scales3D = torch.stack([s_in, s_z, s_out], dim=-1)  # [B, N, 3]

    return {"means3D": pts_world, "scales3D": scales3D, "depth_metric": Z}
