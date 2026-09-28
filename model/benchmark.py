"""Benchmark data generation for EventGS experiments.

We provide three synthetic scenes of increasing difficulty that isolate
the aperture problem and the benefit of observability-aware keyframe
scheduling:

  1. `disc_translation`  -- a textured disc undergoing rigid translation.
     Contains both corners (well-observed) and straight edges
     (aperture-ambiguous). GT motion is a known rigid translation.
  2. `bar_translation`   -- a single straight bar translating
     perpendicular to its long axis. Pure aperture-problem case: the
     event loss alone cannot recover the along-edge motion component.
  3. `textured_grid`    -- a grid of textured cells with random
     per-cell motion magnitudes, used to evaluate per-Gaussian
     observability-weighted recovery.

For each scene we render the baseline G0, apply the GT motion, simulate
events with the standard log-intensity contrast model, and return all
the tensors the EventDynModel needs. This module is intentionally
CPU/GPU-portable and depends only on the pure-PyTorch rasterizer.

Real-data path (documented in the paper, not exercised here): capture
a static scene with a DAVIS-346, build G0 from the APS frames with
vanilla 3DGS, then run EventGS on the DVS stream. The v2e simulator
(github.com/SensorsINI/v2e) can be used to generate events from
arbitrary video when a real event camera is unavailable.
"""
from __future__ import annotations
import math
import torch

from model.torch_rasterizer import render_orth


def _identity_rot(N, device):
    rot = torch.zeros(N, 4, device=device)
    rot[:, 0] = 1.0
    return rot


def make_disc_scene(N=400, H=64, W=64, device='cpu', seed=0):
    """Textured disc: corners + edges, rigid translation GT."""
    torch.manual_seed(seed)
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:N]
    r = 0.3 + 0.05 * torch.randn(N)
    x = (r * torch.cos(theta)) * 1.0
    z = (r * torch.sin(theta)) * 1.0
    y = torch.full((N,), 0.5)
    means = torch.stack([x, y, z], dim=-1).to(device)
    # Vary colors so the disc has texture (gives both corners and edges).
    colors = (torch.rand(N, 3, device=device) * 0.6 + 0.3).clamp(0, 1)
    opacity = torch.full((N, 1), 0.9, device=device)
    scales = torch.full((N, 3), 0.03, device=device)
    rot = _identity_rot(N, device)
    uv = torch.stack([(means[:, 0] * 0.5 + 0.5) * (W - 1),
                      (1.0 - (means[:, 2] * 0.5 + 0.5)) * (H - 1)], dim=-1)
    feat = torch.cat([means, colors], dim=-1)
    gauss_dim = 64
    feat = torch.cat([feat, torch.zeros(N, gauss_dim - feat.shape[1], device=device)], dim=-1)
    gt_dmu = torch.zeros(N, 3, device=device)
    gt_dmu[:, 0] = 0.15
    gt_dmu[:, 2] = -0.10
    return {
        'means3D': means.unsqueeze(0), 'rgb': colors.unsqueeze(0),
        'opacity': opacity.unsqueeze(0), 'scale': scales.unsqueeze(0),
        'rot': rot.unsqueeze(0), 'uv': uv.unsqueeze(0), 'feat': feat.unsqueeze(0),
        'bg': torch.full((3,), 0.2, device=device), 'gt_dmu': gt_dmu,
        'name': 'disc_translation',
    }


def make_bar_scene(N=300, H=64, W=64, device='cpu', seed=0):
    """Straight bar: pure aperture-problem case.

    The bar is a thin rectangle of Gaussians along the x-axis. GT motion
    is a translation with both perpendicular (y) and parallel (x)
    components. The event loss can only recover the y component
    (perpendicular to the long edge); the x component is in the edge
    nullspace and requires CMax / keyframe to recover.
    """
    torch.manual_seed(seed)
    # Bar along x: spread Gaussians in a thin strip.
    x = torch.linspace(-0.5, 0.5, N, device=device)
    z = 0.3 + 0.01 * torch.randn(N, device=device)  # thin in z
    y = torch.full((N,), 0.5, device=device)
    means = torch.stack([x, y, z], dim=-1)
    # Uniform color along the bar (no texture => pure aperture case).
    colors = torch.full((N, 3), 0.7, device=device)
    opacity = torch.full((N, 1), 0.95, device=device)
    scales = torch.full((N, 3), 0.025, device=device)
    rot = _identity_rot(N, device)
    uv = torch.stack([(means[:, 0] * 0.5 + 0.5) * (W - 1),
                      (1.0 - (means[:, 2] * 0.5 + 0.5)) * (H - 1)], dim=-1)
    feat = torch.cat([means, colors], dim=-1)
    gauss_dim = 64
    feat = torch.cat([feat, torch.zeros(N, gauss_dim - feat.shape[1], device=device)], dim=-1)
    # GT motion: mostly perpendicular (z), small parallel (x).
    gt_dmu = torch.zeros(N, 3, device=device)
    gt_dmu[:, 0] = 0.08   # along edge (aperture-ambiguous)
    gt_dmu[:, 2] = -0.12  # perpendicular to edge (observable)
    return {
        'means3D': means.unsqueeze(0), 'rgb': colors.unsqueeze(0),
        'opacity': opacity.unsqueeze(0), 'scale': scales.unsqueeze(0),
        'rot': rot.unsqueeze(0), 'uv': uv.unsqueeze(0), 'feat': feat.unsqueeze(0),
        'bg': torch.full((3,), 0.2, device=device), 'gt_dmu': gt_dmu,
        'name': 'bar_translation',
    }


def make_textured_grid(N=600, H=64, W=64, device='cpu', seed=0):
    """Grid of textured cells with random per-Gaussian motion magnitudes.

    Each Gaussian gets an independent motion magnitude drawn from a
    mixture: well-observed (corner) Gaussians get large motion,
    poorly-observed (edge) Gaussians get small motion. This lets us
    evaluate whether observability-weighted recovery correlates with
    the per-Gaussian observability score.
    """
    torch.manual_seed(seed)
    # 4x4 grid of cell centers.
    gx = torch.linspace(-0.6, 0.6, 4, device=device)
    gz = torch.linspace(-0.6, 0.6, 4, device=device)
    cells = torch.cartesian_prod(gx, gz)  # [16, 2]
    per = N // cells.shape[0]
    means = []
    colors = []
    for c in cells:
        # jitter within cell
        jx = c[0] + 0.12 * torch.randn(per, device=device)
        jz = c[1] + 0.12 * torch.randn(per, device=device)
        jy = torch.full((per,), 0.5, device=device)
        means.append(torch.stack([jx, jy, jz], dim=-1))
        # per-cell color to create texture diversity
        col = torch.rand(per, 3, device=device) * 0.6 + 0.3
        colors.append(col)
    means = torch.cat(means, dim=0)[:N]
    colors = torch.cat(colors, dim=0)[:N]
    # Recompute N to match the actual count after truncation (per-cell
    # counts may not divide evenly into N).
    N = means.shape[0]
    opacity = torch.full((N, 1), 0.9, device=device)
    scales = torch.full((N, 3), 0.025, device=device)
    rot = _identity_rot(N, device)
    uv = torch.stack([(means[:, 0] * 0.5 + 0.5) * (W - 1),
                      (1.0 - (means[:, 2] * 0.5 + 0.5)) * (H - 1)], dim=-1)
    feat = torch.cat([means, colors], dim=-1)
    gauss_dim = 64
    feat = torch.cat([feat, torch.zeros(N, gauss_dim - feat.shape[1], device=device)], dim=-1)
    # Random per-Gaussian motion: half large (0.15), half small (0.03).
    gt_dmu = torch.zeros(N, 3, device=device)
    mag = torch.where(torch.rand(N, device=device) > 0.5,
                      torch.full((N,), 0.15, device=device),
                      torch.full((N,), 0.03, device=device))
    ang = torch.rand(N, device=device) * 2 * math.pi
    gt_dmu[:, 0] = mag * torch.cos(ang)
    gt_dmu[:, 2] = mag * torch.sin(ang)
    return {
        'means3D': means.unsqueeze(0), 'rgb': colors.unsqueeze(0),
        'opacity': opacity.unsqueeze(0), 'scale': scales.unsqueeze(0),
        'rot': rot.unsqueeze(0), 'uv': uv.unsqueeze(0), 'feat': feat.unsqueeze(0),
        'bg': torch.full((3,), 0.2, device=device), 'gt_dmu': gt_dmu,
        'name': 'textured_grid',
    }


def simulate_events(render_t0, render_t1, contrast=0.2, device='cpu'):
    """Standard event-camera simulation from two intensity frames.

    Fires an event at (x, y, p) when |log I1 - log I0| crosses the
    contrast threshold. One event per pixel per window (sufficient for
    the prototype's small resolution; the full pipeline uses v2e/ESIM
    for multi-event-per-pixel simulation).
    """
    eps = 1e-4
    dlog = (torch.log(render_t1.clamp(min=eps)) -
            torch.log(render_t0.clamp(min=eps))).sum(dim=0)
    yi, xi = torch.where(dlog.abs() > contrast)
    p = torch.sign(dlog[yi, xi])
    t = torch.rand(xi.shape[0], device=device)
    pos = torch.stack([xi.float(), yi.float(), t, p], dim=-1)
    return {'pos': pos, 't_min': torch.tensor(0.0, device=device),
            't_max': torch.tensor(1.0, device=device)}


def render_moved(scene, dmu, t=1.0, H=64, W=64):
    """Render the scene with a per-Gaussian translation applied at time t."""
    means = scene['means3D'] + dmu.unsqueeze(0) * t
    color, _, _ = render_orth(
        means[0], scene['rgb'][0], scene['opacity'][0],
        scene['scale'][0], scene['rot'][0], H, W, bg=scene['bg'])
    return color


def build_benchmark(name, H=64, W=64, device='cpu', contrast=0.2):
    """Build a complete benchmark sample: scene + baseline render +
    moved render + simulated events + GT motion.

    name in {'disc', 'bar', 'grid'}.
    """
    if name == 'disc':
        scene = make_disc_scene(H=H, W=W, device=device)
    elif name == 'bar':
        scene = make_bar_scene(H=H, W=W, device=device)
    elif name == 'grid':
        scene = make_textured_grid(H=H, W=W, device=device)
    else:
        raise ValueError(name)
    color_t0, _, _ = render_orth(
        scene['means3D'][0], scene['rgb'][0], scene['opacity'][0],
        scene['scale'][0], scene['rot'][0], H, W, bg=scene['bg'])
    color_t1_gt = render_moved(scene, scene['gt_dmu'], t=1.0, H=H, W=W)
    events = simulate_events(color_t0, color_t1_gt, contrast=contrast, device=device)
    scene['render_t0'] = color_t0.unsqueeze(0)
    scene['render_t1_gt'] = color_t1_gt.unsqueeze(0)
    return scene, events


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def motion_recovery_error(pred_dmu, gt_dmu):
    """L2 error of recovered motion, per-Gaussian and aggregate.

    pred_dmu, gt_dmu : [N, 3] or [B, N, 3].
    returns dict with per_gaussian L2, mean, and sign-agreement fraction.
    """
    if pred_dmu.dim() == 3:
        pred_dmu = pred_dmu[0]
    if gt_dmu.dim() == 3:
        gt_dmu = gt_dmu[0]
    l2 = (pred_dmu - gt_dmu).norm(dim=-1)  # [N]
    gt_norm = gt_dmu.norm(dim=-1).clamp(min=1e-6)
    sign_agree = ((pred_dmu * gt_dmu).sum(dim=-1) > 0).float().mean()
    return {
        'l2_mean': l2.mean().item(),
        'l2_median': l2.median().item(),
        'rel_l2_mean': (l2 / gt_norm).mean().item(),
        'sign_agreement': sign_agree.item(),
        'pred_norm_mean': pred_dmu.norm(dim=-1).mean().item(),
        'gt_norm_mean': gt_norm.norm(dim=-1).mean().item(),
    }


def rendering_psnr(pred_color, gt_color):
    """PSNR between predicted and GT rendered color images."""
    if pred_color.dim() == 4:
        pred_color = pred_color[0]
    if gt_color.dim() == 4:
        gt_color = gt_color[0]
    mse = ((pred_color - gt_color) ** 2).mean()
    if mse.item() < 1e-12:
        return 100.0
    return (10.0 * torch.log10(1.0 / mse)).item()
