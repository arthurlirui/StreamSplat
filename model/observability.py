"""
Observability-aware keyframe scheduling for EventGS.

Physical motivation
-------------------
The event-generation model

    log I(x, t) - log I(x, t_ref) = +/- C            (1)

is a *single* scalar constraint per pixel per event. The 3D Gaussian
increment dmu_i lives in R^3, so even with a perfectly differentiable
rasterizer, a single event only constrains the component of dmu along
the *projected image gradient* direction. The tangential component
(parallel to the edge) is in the nullspace --- this is the 3D analogue
of the classic aperture problem, and it is *exactly* the degeneracy we
observed empirically: the event-only loss admits a no-motion minimum.

This module makes the degeneracy *computable*. For each Gaussian i we
estimate a local observability score s_i in [0, 1] from the structure
tensor of the rendered baseline image in a neighborhood of the
Gaussian's projection. The score is large when the local image
neighborhood has two strong, non-degenerate gradient directions
(corners / textured regions) and small along straight edges and in
flat regions. We then aggregate per-pixel observability into a scene
*observedness* map and use it to:

  1. Weight the event-consistency loss so that well-observed Gaussians
     are trusted and poorly-observed ones are regularized toward the
     keyframe prior (preventing the aperture minimum from dominating).
  2. Schedule keyframes adaptively: a new APS / intensity keyframe is
     requested only when the running average observedness drops below a
     threshold, i.e. when the event stream alone is no longer
     informative enough to disambiguate the motion. This gives the
     "intermittent keyframe alignment" of Layer 3 a principled trigger
     instead of a fixed cadence.

The structure-tensor observability measure is the same quantity that
underlies Lucas-Kanade's A^T A matrix and the Harris corner detector,
so it connects directly to the classical aperture-problem literature.
We extend it to the 3D-Gaussian setting by evaluating it on the
*rasterized baseline* at each Gaussian's projected pixel, which is the
natural reference frame for the event-generation model.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def image_structure_tensor(img: torch.Tensor, ksize: int = 5) -> torch.Tensor:
    """Per-pixel 2x2 structure tensor of a grayscale image.

    img : [B, 1, H, W] or [B, 3, H, W] (converted to grayscale).
    returns : [B, H, W, 2, 2] structure tensor S = (g g^T) * window.
    """
    if img.shape[1] == 3:
        gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)
    else:
        gray = img
    # Sobel gradients.
    gx = F.conv2d(gray, torch.tensor([[[[-1., 0., 1.]]]], device=img.device).expand(1, 1, 1, 3),
                  padding=(0, 1))
    gy = F.conv2d(gray, torch.tensor([[[[-1.], [0.], [1.]]]], device=img.device).expand(1, 1, 3, 1),
                  padding=(1, 0))
    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy
    # Box-window aggregation (approximates Gaussian window for speed).
    pad = ksize // 2
    w = torch.ones(1, 1, ksize, ksize, device=img.device) / (ksize * ksize)
    Sxx = F.conv2d(Ixx, w, padding=pad)[:, 0]
    Syy = F.conv2d(Iyy, w, padding=pad)[:, 0]
    Sxy = F.conv2d(Ixy, w, padding=pad)[:, 0]
    S = torch.stack([torch.stack([Sxx, Sxy], dim=-1),
                     torch.stack([Sxy, Syy], dim=-1)], dim=-1)  # [B,H,W,2,2]
    return S


def observability_score(img: torch.Tensor, ksize: int = 5,
                        eps: float = 1e-6) -> torch.Tensor:
    """Per-pixel observability in [0, 1] from the structure tensor.

    We use the Harris-like measure  det(S) / (trace(S) + eps), which is
    small along edges (one eigenvalue ~0) and large at corners (both
    eigenvalues comparable). We normalize to [0, 1] by a softplus + tanh
    so it is differentiable and bounded.
    """
    S = image_structure_tensor(img, ksize)  # [B,H,W,2,2]
    det = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] ** 2
    tr = S[..., 0, 0] + S[..., 1, 1]
    # Harris response with denominator guard; note this is exactly the
    # quantity whose vanishing marks the aperture degeneracy.
    r = det / (tr + eps)
    # Map to [0, 1] smoothly. r can be negative in noisy flat regions;
    # softplus keeps it non-negative, tanh bounds it.
    score = torch.tanh(F.softplus(r))
    return score  # [B, H, W]


def sample_gaussian_observability(obs_map: torch.Tensor,
                                  uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample the per-pixel observability map at each Gaussian's
    projected pixel location.

    obs_map : [B, H, W] from observability_score.
    uv      : [B, N, 2] pixel coords (x, y) for each Gaussian.
    returns : [B, N] per-Gaussian observability in [0, 1].
    """
    B, H, W = obs_map.shape
    feat = obs_map.unsqueeze(1)  # [B,1,H,W]
    uv_n = uv.clone().float()
    uv_n[..., 0] = uv_n[..., 0] / (W - 1) * 2 - 1
    uv_n[..., 1] = uv_n[..., 1] / (H - 1) * 2 - 1
    grid = uv_n.unsqueeze(2)  # [B,N,1,2]
    s = F.grid_sample(feat, grid, mode='bilinear', align_corners=True,
                      padding_mode='border')
    return s.squeeze(-1).squeeze(1)  # [B,N]


def observedness_loss_weight(obs_per_gauss: torch.Tensor,
                             beta: float = 2.0) -> torch.Tensor:
    """Convert per-Gaussian observability to a loss weight in [beta^-1, 1].

    Well-observed Gaussians (corners/textures) get weight ~1, poorly
    observed ones (edges/flats) get weight ~1/beta so the event loss
    does not force them toward the degenerate no-motion solution.
    """
    return 1.0 / (1.0 + beta * (1.0 - obs_per_gauss))


class AdaptiveKeyframeScheduler:
    """Decide when to request an absolute-intensity keyframe.

    The scheduler tracks a running mean of the scene observedness over
    the event windows processed so far. When the running mean drops
    below `threshold` (i.e. the event stream is dominated by
    aperture-ambiguous regions and can no longer disambiguate the
    motion on its own), it requests a new keyframe. This replaces a
    fixed-cadence trigger with a principled, content-adaptive one.

    Usage:
        sched = AdaptiveKeyframeScheduler(threshold=0.25)
        for window in event_windows:
            obs = observability_score(render_baseline(window))
            need_kf = sched.step(obs.mean())
            if need_kf: grab_aps_keyframe(); sched.reset()
    """

    def __init__(self, threshold: float = 0.25, warmup: int = 3,
                 ema_alpha: float = 0.3):
        self.threshold = threshold
        self.warmup = warmup            # always grab the first `warmup` keyframes
        self.ema_alpha = ema_alpha
        self.ema = None
        self.n_calls = 0
        self.n_keyframes = 0

    def reset(self):
        # After grabbing a keyframe we reset the EMA so the next request
        # is driven by post-keyframe observedness, not stale history.
        self.ema = None

    def step(self, observedness: float) -> bool:
        self.n_calls += 1
        if self.n_calls <= self.warmup:
            self.n_keyframes += 1
            self.ema = float(observedness)
            return True
        v = float(observedness)
        if self.ema is None:
            self.ema = v
        else:
            self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * v
        if self.ema < self.threshold:
            self.n_keyframes += 1
            self.ema = v  # reset baseline after request
            return True
        return False
