"""
Contrast-maximization regularizer for EventGS.

Why
---
The event-consistency loss L_event is a *residual* loss: it penalizes
mismatch between predicted and observed log-intensity changes. It has a
known degenerate minimum (the aperture problem): zero motion produces
zero residual on the pixels that *did* fire, and the negative term can
be minimized by shrinking all motion. Photometric keyframe alignment
disambiguates the absolute geometry, but it requires an APS frame and
is therefore intermittent.

As a *continuous*, keyframe-free regularizer we add a Contrast
Maximization (CMax) term on the predicted 2D Gaussian-flow field. CMax
is the standard event-camera motion-estimation framework (Gallego et
al. 2018): the correct motion is the one that, when used to warp the
events in a time window onto a single reference time, maximizes the
sharpness (contrast) of the Image of Warped Events (IWE). Crucially,
CMax only depends on events --- no absolute intensity needed --- so it
is available in every window, unlike the keyframe loss.

The 3D-Gaussian analogue
------------------------
For each Gaussian i with predicted 2D projected displacement
du_i = pi(mu_i + dmu_i) - pi(mu_i) over the window [t_a, t_b], and each
event e_k = (x_k, y_k, t_k, p_k) in the window, we warp the event onto
the reference time t_a by

    x'_k = x_k - (t_k - t_a) / (t_b - t_a) * du_i(k)

where i(k) is the Gaussian whose projection is closest to event k
(precomputed via the static pixel->Gaussian map). The IWE is then

    IWE(x) = sum_k delta(x - x'_k)

and the CMax loss is -Var(IWE) (we minimize the negative contrast).

Because du_i is a differentiable function of the Gaussian increments
dmu through the rasterizer's projection, the CMax gradient flows into
dmu and provides an aperture-breaking signal along the *edge-normal*
direction --- the exact direction the event loss is weak in. This
makes CMax and the event-consistency loss complementary: L_event fixes
the magnitude (via the contrast threshold C), CMax fixes the direction
(via edge alignment).

Implementation note
-------------------
For the prototype (small N, orthographic camera) we implement a
differentiable soft-IWE via 2D histogramming with a Gaussian kernel.
The full CUDA rasterizer would replace this with a splatting-based IWE.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def gaussian_projected_flow(uv_t0: torch.Tensor,
                            uv_t1: torch.Tensor) -> torch.Tensor:
    """2D projected displacement of each Gaussian over the window.
    uv_t0, uv_t1 : [B, N, 2] pixel coords at t_a and t_b.
    returns du : [B, N, 2]."""
    return uv_t1 - uv_t0


def soft_image_of_warped_events(events: dict,
                                du: torch.Tensor,
                                assoc: torch.Tensor,
                                H: int, W: int,
                                sigma: float = 1.0) -> torch.Tensor:
    """Differentiable soft Image of Warped Events (IWE).

    events : {'pos': [M, 4] (x, y, t, p), 't_min', 't_max'}
    du     : [B, N, 2] per-Gaussian 2D flow.
    assoc  : [M] index of the Gaussian associated with each event
             (nearest projected Gaussian at the event pixel; precomputed
             from the static pixel->Gaussian map).
    H, W   : image resolution.
    sigma  : Gaussian kernel std (pixels) for soft histogramming.

    returns : [B, H, W] IWE (sum of per-event Gaussian bumps at warped
               positions, polarity-signed so the contrast is meaningful).
    """
    B = du.shape[0]
    device = du.device
    pos = events['pos']  # [M,4]
    x = pos[:, 0]
    y = pos[:, 1]
    t = pos[:, 2]
    p = pos[:, 3]
    t_min = events.get('t_min', t.min().detach())
    t_max = events.get('t_max', t.max().detach())
    tn = (t - t_min) / (t_max - t_min + 1e-6)  # [M] in [0,1]

    # Per-event flow: look up the associated Gaussian's du and interpolate
    # in time. Linear-in-time flow is the standard CMax assumption.
    flow = du[:, assoc, :]  # [B, M, 2]
    flow_t = tn[None, :, None] * flow  # displacement at event time
    x_warp = x[None, :] - flow_t[..., 0]  # [B, M]
    y_warp = y[None, :] - flow_t[..., 1]

    # Soft histogram onto an HxW grid via a Gaussian kernel. We render
    # each event as a small Gaussian bump; the IWE is the sum. This is
    # differentiable w.r.t. x_warp, y_warp (and hence du).
    iwe = torch.zeros(B, H, W, device=device)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    # Chunk events to bound memory (M x H x W).
    chunk = 4096
    for b in range(B):
        for s in range(0, x_warp.shape[1], chunk):
            sl = slice(s, min(s + chunk, x_warp.shape[1]))
            xw = x_warp[b, sl][:, None, None]  # [m,1,1]
            yw = y_warp[b, sl][:, None, None]
            pol = p[sl][:, None, None]
            # squared distance to every pixel
            dx2 = (xs[None, None, :] - xw) ** 2
            dy2 = (ys[None, :, None] - yw) ** 2
            bump = torch.exp(-0.5 * (dx2 + dy2) / (sigma ** 2 + 1e-6)) * pol
            iwe[b] += bump.sum(dim=0)
    return iwe


def contrast_maximization_loss(events: dict,
                               du: torch.Tensor,
                               assoc: torch.Tensor,
                               H: int, W: int,
                               sigma: float = 1.0) -> torch.Tensor:
    """CMax loss = -Var(IWE). Lower is better (sharper warped events).

    Minimizing this encourages the per-Gaussian flow to align the events
    along their true trajectories, which breaks the aperture ambiguity
    along the edge-normal direction without needing an intensity frame.
    """
    iwe = soft_image_of_warped_events(events, du, assoc, H, W, sigma)
    # Variance over pixels, averaged over batch. We use variance (not
    # just std) so the gradient scale is stable across resolutions.
    var = iwe.var(dim=(1, 2), unbiased=False)
    return -var.mean()


def pixel_to_gaussian_assoc(uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Precompute, for every pixel, the index of the nearest projected
    Gaussian. This is the static-camera pixel->Gaussian map that makes
    the event-to-Gaussian association O(1) per event.

    uv : [B, N, 2] projected pixel coords of all Gaussians.
    returns : [B, H, W] long tensor of Gaussian indices.

    For large N this is O(N*H*W); in the static-camera setting it is
    computed once and cached, so the cost is amortized over all windows.
    """
    B, N, _ = uv.shape
    device = uv.device
    ys = torch.arange(H, device=device, dtype=torch.float32).view(H, 1, 1)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, W, 1)
    px = uv[:, :, 0].unsqueeze(1)  # [B,1,N]
    py = uv[:, :, 1].unsqueeze(1)
    # broadcast: [B, H, W, N]
    d2 = (xs - px.unsqueeze(2)) ** 2 + (ys - py.unsqueeze(2)) ** 2
    return d2.argmin(dim=-1)  # [B,H,W]
