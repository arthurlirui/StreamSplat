"""StereoSplat model: metric-scale dynamic 3DGS from a stereo video stream.

Architecture (reusing StreamSplat's components):

    [L, R frames] + [stereo metric depth]
            │
            ▼
    GSEncoder (StreamSplat, V=2)        ── reused
            │
            ▼
    GaussianUpsampler + GSPMDecoder     ── reused (pred_keys unchanged)
            │
            ▼  per-pixel (xyz_static, scale, rgb, opacity, rot)
    MetricHead (metric_backproject)     ── new: lift to metric world coords
            │
            ▼
    gaussian_renderer_perspective.render ── new: perspective, per-camera pose
            │
            ▼  rendered left & right images, depth, alpha
    Losses: photometric(L), photometric(R), metric-depth, LPIPS, dssim

The model inherits `SplatModel` so that `forward_gaussians`, `compute_losses`,
state-dict filtering, and the LPIPS bookkeeping all carry over. Only
`forward` and the rasterizer choice are overridden.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.splat_model import SplatModel
from model.metric_head import metric_backproject, make_pixel_grid
import gaussian_renderer_perspective as renderer_perspective
from configs.options_stereo import StereoOptions


class StereoSplatModel(SplatModel):
    """StreamSplat with a metric stereo front-end and perspective rasterizer."""

    def __init__(self, opt: StereoOptions, **model_kwargs):
        # Parent constructor builds `self.model = SplatPredictor(opt)` and an
        # orthographic `self.gaussian_renderer`. We override the renderer with
        # the perspective one after construction.
        super().__init__(opt, **model_kwargs)
        self.opt = opt
        self.gaussian_renderer = renderer_perspective.render
        # Precompute the pixel grid for the configured resolution.
        H, W = opt.down_resolution if len(opt.down_resolution) > 0 else (opt.image_height, opt.image_width)
        # The decoder produces one Gaussian per output pixel after the
        # pixel-shuffle upsampling; for the metric head we use the input
        # resolution grid as a stable reference.
        self._pixel_grid = make_pixel_grid(H, W, device='cuda')

    def _build_cameras(self, K, poses_left, baseline, device):
        """Build a list of CameraPose objects for left (idx 0) and right (idx 1)."""
        from gaussian_renderer_perspective import CameraPose
        import numpy as np
        B = K.shape[0]
        cameras = []
        for b in range(B):
            Kb = K[b].cpu().numpy()
            # Left camera pose for the input frame (frame 0).
            pl = poses_left[b, 0].cpu().numpy()  # [3, 4] c2w
            Rl, tl = pl[:, :3], pl[:, 3]
            H, W = self.opt.down_resolution if len(self.opt.down_resolution) > 0 else (self.opt.image_height, self.opt.image_width)
            cameras.append(CameraPose(Kb, Rl, tl, H, W))
            # Right camera pose: left pose composed with baseline offset.
            # Right cam in left frame: t_R_in_L = (baseline, 0, 0).
            # c2w_right = c2w_left @ T(baseline, 0, 0)^-1, but for a static rig
            # we just place it at left-position + R_left @ (baseline,0,0).
            tr = tl + Rl @ np.array([float(baseline[b].item()), 0.0, 0.0], dtype=np.float32)
            cameras.append(CameraPose(Kb, Rl, tr, H, W))
        return cameras

    def forward(self, data, step_ratio: float = 0.0):
        """Forward pass for a stereo batch.

        Expected `data` keys (produced by `datasets.provider_stereo`):
            frames         [B, 2, 3, H, W]   left + right input frames (t=0)
            depths         [B, 2, 1, H, W]   metric depth (left, duplicated)
            target_frames_left  [B, of, 3, H, W]
            target_frames_right [B, of, 3, H, W]
            target_depths       [B, of, 1, H, W]
            intrinsics     [B, 3, 3]
            baseline       [B]
            poses_left     [B, of, 3, 4]   c2w
        """
        input_frames = data['frames']               # [B, 2, 3, H, W]
        input_depths_metric = data['depths']        # [B, 2, 1, H, W] meters
        K = data['intrinsics']                      # [B, 3, 3]
        baseline = data['baseline']                 # [B]
        poses_left = data['poses_left']             # [B, of, 3, 4]

        B, V, C, H, W = input_frames.shape
        device = input_frames.device

        # Normalize depth for the encoder (StreamSplat's encoder expects
        # roughly [0,1] inputs). We keep the metric version separately for
        # the back-projection head.
        d_min = input_depths_metric.flatten(2).min(dim=2)[0].view(B, 1, 1, 1, 1)
        d_max = input_depths_metric.flatten(2).max(dim=2)[0].view(B, 1, 1, 1, 1)
        input_depths_norm = (input_depths_metric - d_min) / (d_max - d_min + 1e-6)

        # Build per-batch cameras (left=0, right=1).
        cameras = self._build_cameras(K, poses_left, baseline, device)

        # Run the StreamSplat decoder: produces per-pixel Gaussian attrs in
        # the network's normalized convention.
        # `forward_gaussians` expects [B, V, C, H, W] frames and depths.
        decoder_out = self.forward_gaussians(input_frames, input_depths_norm)
        pred_gs = decoder_out["pred_gs"]

        # ---- Metric back-projection ----
        # `xyz_static` is the network's per-pixel (x, y, z) in normalized
        # space. We re-interpret it as (du, dv, dz_rel) for the metric head.
        # The decoder emits `xyz` already combined; we operate on the static
        # slice [B, N, 0, 3].
        xyz_static = pred_gs['xyz'][..., 0, :]      # [B, N, 3]
        scale = pred_gs['scale']                    # [B, N, 2]
        # Metric depth per pixel: take the left-image depth, flattened.
        depth_metric = input_depths_metric[:, 0].reshape(B, -1, 1)  # [B, N, 1]
        K_dev = K.to(device)
        R_c2w = poses_left[:, 0, :3, :3].to(device)             # [B, 3, 3]
        t_c2w = poses_left[:, 0, :3, 3].to(device)              # [B, 3]
        pg = self._pixel_grid.to(device)
        # Match N between pixel grid and network output.
        N = xyz_static.shape[1]
        if pg.shape[0] != N:
            # Resample pixel grid to N via interpolation (decoder may upsample).
            pg = F.interpolate(pg.unsqueeze(0).permute(0, 2, 1),
                               size=N, mode='linear', align_corners=False
                               ).permute(0, 2, 1).squeeze(0)
        mhead = metric_backproject(
            xyz_static, scale, depth_metric, K_dev, R_c2w, t_c2w, pg,
            scale_range=(self.opt.scale_min, self.opt.scale_max))
        # Replace the static means with metric means; keep dynamic components.
        pred_gs['xyz'][..., 0, :] = mhead['means3D']
        # Replace 2D scale with metric 3D scale (decoder expects [N, 2] for
        # the in-plane pair; we feed (sx, sz) so the rasterizer's y_scale
        # mean still works).
        pred_gs['scale'] = mhead['scales3D'][..., [0, 2]]  # [B, N, 2]

        # ---- Render left and right ----
        with torch.cuda.amp.autocast(enabled=False):
            render_left = self.gaussian_renderer(
                pred_gs, self.background, opt=self.opt,
                training=self.training, cameras=cameras, camera_index=0)
            render_right = self.gaussian_renderer(
                pred_gs, self.background, opt=self.opt,
                training=self.training, cameras=cameras, camera_index=1)

        return {
            'render_left': render_left,
            'render_right': render_right,
            'pred_gs_metric': pred_gs,
            'depth_metric': mhead['depth_metric'],
            'input_frames': input_frames,
            'input_depths_metric': input_depths_metric,
            'target_frames_left': data['target_frames_left'],
            'target_frames_right': data['target_frames_right'],
            'target_depths': data['target_depths'],
            'intrinsics': K,
            'baseline': baseline,
        }


def stereo_loss(model_out, opt, lpips_fn=None, epoch: int = 0):
    """Compute the StereoSplat combined loss.

    Combines:
      - L1 photometric on left render vs. left target
      - L1 photometric on right render vs. right target (stereo consistency)
      - metric-depth L1 (rendered depth vs. stereo depth)
      - LPIPS (optional, after warmup)
      - dssim
    """
    rl = model_out['render_left']
    rr = model_out['render_right']
    img_l = rl['render']      # [B, of, 3, H, W]
    img_r = rr['render']
    dep_l = rl['depth']       # [B, of, 1, H, W]
    tgt_l = model_out['target_frames_left']
    tgt_r = model_out['target_frames_right']
    tgt_d = model_out['target_depths']

    # Photometric (left)
    l1_l = F.l1_loss(img_l, tgt_l)
    # Photometric (right) — stereo consistency
    l1_r = F.l1_loss(img_r, tgt_r)
    # Metric depth L1 (only where the stereo depth is valid, i.e. > 0)
    valid = (tgt_d > 0).float()
    dep_err = (dep_l - tgt_d).abs() * valid
    l_depth = dep_err.sum() / (valid.sum() + 1e-6)

    loss = l1_l + opt.lambda_stereo_consist * l1_r + opt.lambda_metric_depth * l_depth

    # LPIPS (after warmup)
    if opt.lambda_lpips > 0 and epoch >= opt.lpips_start_epoch and lpips_fn is not None:
        B, of, C, H, W = img_l.shape
        lp = lpips_fn(
            F.interpolate(img_l.reshape(-1, 3, H, W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            F.interpolate(tgt_l.reshape(-1, 3, H, W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
        ).mean()
        loss = loss + opt.lambda_lpips * lp
    else:
        lp = torch.tensor(0.0, device=img_l.device)

    with torch.no_grad():
        psnr = -10.0 * torch.log10(((img_l - tgt_l) ** 2).mean() + 1e-12)

    metrics = {
        'loss': loss.detach(),
        'l1_left': l1_l.detach(),
        'l1_right': l1_r.detach(),
        'depth_l1': l_depth.detach(),
        'lpips': lp.detach() if torch.is_tensor(lp) else torch.tensor(float(lp)),
        'psnr': psnr.detach(),
    }
    return loss, metrics
