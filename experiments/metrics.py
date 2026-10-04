"""Shared metric helpers for StereoSplat experiments.

Kept dependency-free (torch only) so the experiment scripts can be
imported and unit-tested without pulling in the CUDA rasterizer or the
full model. The model-side imports happen inside each script's `main()`.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak SNR between two image tensors (any shape, values in [0,1])."""
    mse = F.mse_loss(pred, gt)
    return float(-10.0 * torch.log10(mse + 1e-12))


def ssim_simple(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """A simple channel-mean SSIM-like score in [-1, 1].

    For the full SSIM used in the paper, call `utils.metrics.compute_ssim`
    from within the training environment; this lightweight version is for
    unit tests and CPU-only harness runs.
    """
    mu_p = pred.mean(dim=(-1, -2, -3), keepdim=True)
    mu_g = gt.mean(dim=(-1, -2, -3), keepdim=True)
    var_p = pred.var(dim=(-1, -2, -3), keepdim=True, unbiased=False)
    var_g = gt.var(dim=(-1, -2, -3), keepdim=True, unbiased=False)
    cov = ((pred - mu_p) * (gt - mu_g)).mean(dim=(-1, -2, -3), keepdim=True)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_p * mu_g + c1) * (2 * cov + c2)) / \
        ((mu_p ** 2 + mu_g ** 2 + c1) * (var_p + var_g + c2) + 1e-12)
    return float(s.mean())


def metrics_dict(pred: torch.Tensor, gt: torch.Tensor, lpips_fn=None) -> dict:
    """Compute PSNR/SSIM/LPIPS. LPIPS is 0.0 if no lpips_fn is given."""
    out = {'psnr': psnr(pred, gt), 'ssim': ssim_simple(pred, gt)}
    if lpips_fn is not None:
        out['lpips'] = float(lpips_fn(pred * 2 - 1, gt * 2 - 1).mean())
    else:
        out['lpips'] = 0.0
    return out
