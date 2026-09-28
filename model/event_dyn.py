"""
Event-conditioned dynamic Gaussian decoder (minimal prototype).

Physical motivation
-------------------
An event camera fires an event e_k = (x_k, y_k, t_k, p_k) whenever the
log-intensity at a pixel changes by a contrast threshold C:

    log I(x, y, t_k) - log I(x, y, t_{k-1}) = p_k * C.        (1)

Given a *known, high-fidelity static 3D Gaussian Splatting scene* G0 and a
*static event camera with known pose*, we don't reconstruct the scene from
scratch. Instead, events only describe the *dynamic increment* of the scene:

    G(t) = G0 (+) dG(t),        dG(t) = { dmu, dalpha, dcolor }_i(t).

The decoder maps an event-stream representation (a time-binned voxel grid,
following the E2VID / event-to-video convention) to per-Gaussian increments
dtheta. We keep the same polynomial motion model as StreamSplat
(``forder``, ``dynamic_type="poly"``) so the output plugs straight into the
existing 4D rasterizer.

Two physical connections to event-to-video are made explicit:

  1. E2VID reconstructs intensity from events by inverting the integral of
     eq. (1). Here we keep intensities on the *3D scene side*: we render G(t),
     take log, and require its inter-frame difference to match the events.
     The decoder is thus the "3D analogue" of E2VID — instead of producing a
     2D image per timestamp, it produces the 3D deformation that explains the
     observed 2D events.

  2. The event-consistency loss penalizes *predicted-but-unobserved* events
     (pixels where the rendered log-intensity changed but no event fired).
     This is the negative-supervision signal E2VID lacks because it only ever
     sees positive events; here the 3D scene gives us the full predicted
     intensity field, so we can penalize spurious change too.

This module is intentionally compact (~a few hundred lines) and depends only
on torch so it runs without the CUDA rasterizer. See ``demo_event_dyn.py``.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Event representation
# ---------------------------------------------------------------------------
def events_to_voxel(events, H, W, num_bins=5, polarity_split=True):
    """Convert an event list to a voxel-grid tensor (E2VID-style).

    events: dict with
        'pos'  : [M, 4]  (x, y, t, p)  float tensor on any device
        't_min': scalar, 't_max': scalar (time window bounds)
    Returns: [B=1, C, H, W] tensor.
    With polarity_split, C = 2*num_bins (positive and negative events
    histogrammed separately); otherwise C = num_bins with signed polarity.
    """
    xy = events['pos'][:, :2].long()
    t = events['pos'][:, 2]
    p = events['pos'][:, 3]
    t_min = events.get('t_min', t.min().detach())
    t_max = events.get('t_max', t.max().detach())
    tn = (t - t_min) / (t_max - t_min + 1e-6)  # [0,1]
    bin_idx = (tn * (num_bins - 1)).round().long().clamp(0, num_bins - 1)

    H_ = H
    W_ = W
    device = xy.device
    if polarity_split:
        C = 2 * num_bins
        # channel = 2*bin + (p<0)
        chan = 2 * bin_idx + (p < 0).long()
    else:
        C = num_bins
        chan = bin_idx
        p = p  # signed weight

    voxel = torch.zeros(C, H_, W_, device=device)
    if polarity_split:
        voxel.index_put_(
            (chan, xy[:, 1].clamp(0, H_ - 1), xy[:, 0].clamp(0, W_ - 1)),
            torch.ones(xy.shape[0], device=device),
            accumulate=True,
        )
    else:
        voxel.index_put_(
            (chan, xy[:, 1].clamp(0, H_ - 1), xy[:, 0].clamp(0, W_ - 1)),
            p,
            accumulate=True,
        )
    # Normalize to unit std so the encoder sees a comparable scale across windows.
    voxel = voxel / (voxel.std() + 1e-6)
    return voxel.unsqueeze(0)  # [1, C, H, W]


class EventEncoder(nn.Module):
    """A small ConvNet that maps an event voxel grid to a per-Gaussian feature.

    Because the camera is static and the pixel->Gaussian correspondence is
    precomputed (each pixel maps to a fixed set of Gaussians), we can decode
    per-Gaussian features by sampling the event feature map at each Gaussian's
    projected pixel location. This is the "ray-localized" decoding: an event
    at pixel (x,y) only influences the Gaussians along that pixel's ray.
    """

    def __init__(self, in_ch=10, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_dim, 1),
        )
        self.out_dim = out_dim

    def forward(self, voxel, gauss_uv):
        """voxel: [B, C, H, W]; gauss_uv: [B, N, 2] in pixel coords (x,y).
        Returns per-Gaussian event feature [B, N, out_dim]."""
        feat = self.net(voxel)  # [B, F, h, w]
        B, F_, h, w = feat.shape
        # normalize coords to [-1,1] for grid_sample (y flipped)
        uv = gauss_uv.clone().float()
        uv[..., 0] = uv[..., 0] / (w - 1) * 2 - 1
        uv[..., 1] = uv[..., 1] / (h - 1) * 2 - 1
        grid = uv.unsqueeze(2)  # [B, N, 1, 2]
        sampled = F.grid_sample(feat, grid, mode='bilinear',
                                align_corners=True, padding_mode='zeros')
        # grid_sample returns [B, F, N, 1]; permute to [B, N, F].
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, F]


# ---------------------------------------------------------------------------
# Event-conditioned dynamic decoder
# ---------------------------------------------------------------------------
class EventDynamicDecoder(nn.Module):
    """Maps (static Gaussian features, event features, time) -> dynamic increments.

    The output mirrors StreamSplat's dynamic Gaussian parameterization:
      - dmu_i(t) = sum_m a_{i,m} * t^m   (polynomial translation, forder terms)
      - dalpha_i, dcolor_i                (opacity / color increments)

    All increments are applied additively on top of the frozen baseline scene.
    """

    def __init__(self, gauss_dim=64, event_dim=64, hidden=128, forder=2,
                 use_pm=True):
        super().__init__()
        self.forder = forder
        self.use_pm = use_pm
        in_dim = gauss_dim + event_dim + 1  # +1 for query time

        # Polynomial translation coefficients: forder * 3 (x, y, z) per Gaussian.
        self.dmu_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, forder * 3),
        )
        # Opacity increment (delta logit).
        self.dalpha_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Color increment.
        self.dcolor_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        # Optional probabilistic head (truncated-Gaussian scale for dmu),
        # echoing StreamSplat's Truncated_Gaussian_Model. We predict a per-axis
        # log-scale to model uncertainty from sparse events.
        if use_pm:
            self.dmu_scale_head = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, forder * 3),
            )
        # LayerNorm on the fused feature for stable training.
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, gauss_feat, event_feat, t_query):
        """gauss_feat: [B,N,Dg], event_feat: [B,N,De], t_query: [B,1] in [0,1].
        Returns dict of increments."""
        B, N, _ = gauss_feat.shape
        t_exp = t_query.unsqueeze(1).expand(-1, N, -1)  # [B,N,1]
        x = torch.cat([gauss_feat, event_feat, t_exp], dim=-1)
        x = self.ln(x)

        dmu_coef = self.dmu_head(x)  # [B,N,forder*3]
        dmu_coef = torch.tanh(dmu_coef) * 0.2  # bound motion magnitude
        # form polynomial: dmu(t) = sum_{m=1}^{forder} coef_m * t^m
        powers = torch.stack(
            [t_exp ** m for m in range(1, self.forder + 1)], dim=2
        )  # [B,N,forder,1]
        dmu = (dmu_coef.view(B, N, self.forder, 3) * powers).sum(dim=2)  # [B,N,3]

        dalpha = self.dalpha_head(x)  # [B,N,1] delta logit
        dcolor = torch.tanh(self.dcolor_head(x)) * 0.3  # [B,N,3] bounded

        out = {'dmu': dmu, 'dalpha': dalpha, 'dcolor': dcolor}
        if self.use_pm:
            log_scale = self.dmu_scale_head(x).view(B, N, self.forder, 3)
            log_scale = log_scale.clamp(-5.0, 2.0)
            out['dmu_log_scale'] = (log_scale * powers).sum(dim=2)  # [B,N,3]
        return out


# ---------------------------------------------------------------------------
# Event-consistency loss
# ---------------------------------------------------------------------------
def event_consistency_loss(render_t0, render_t1, events, contrast=0.2,
                           tau=0.05, lambda_neg=0.5):
    """Loss implementing eq. (1) in the paper.

    render_t0 / render_t1: [B,3,H,W] rendered intensity images (in [0,1]) at
                            the two ends of the event window.
    events: dict with 'pos' [M,4] (x,y,t,p), 't_min', 't_max'.
    contrast: C, the event camera contrast threshold.
    tau: soft threshold for the negative term.
    lambda_neg: weight on the "predicted-but-unobserved" penalty.

    Three terms:
      L_pos   : pixels that DID fire an event must show |Dlog I| ~ C with the
                correct sign.
      L_neg   : pixels that did NOT fire any event must show |Dlog I| < tau.
      L_smooth: (optional, applied outside) temporal smoothness of increments.
    """
    B, _, H, W = render_t1.shape
    eps = 1e-4
    log0 = torch.log(render_t0.clamp(min=eps))
    log1 = torch.log(render_t1.clamp(min=eps))
    dlog = (log1 - log0).sum(dim=1)  # [B,H,W] log-luminance change (grayscale)

    xy = events['pos'][:, :2].long()
    p = events['pos'][:, 3]
    # Aggregate signed polarity per pixel over the window (last write wins is
    # fine for the loss; we only need where events happened and the net sign).
    pos_mask = torch.zeros(B, H, W, device=dlog.device)
    neg_mask = torch.zeros(B, H, W, device=dlog.device)
    sign = torch.zeros(B, H, W, device=dlog.device)
    valid = torch.zeros(B, H, W, device=dlog.device)
    for b in range(B):
        xi = xy[:, 0].clamp(0, W - 1)
        yi = xy[:, 1].clamp(0, H - 1)
        pos_mask[b, yi, xi] = (p > 0).float()
        neg_mask[b, yi, xi] = (p < 0).float()
        sign[b, yi, xi] = p.float()
        valid[b, yi, xi] = 1.0

    # Positive term: where events fired, dlog should be ~ p*C (L1).
    target = sign * contrast
    l_pos = (valid * (dlog - target).abs()).sum() / (valid.sum() + 1.0)

    # Negative term: where no event fired, |dlog| should be small.
    no_event = 1.0 - valid
    excess = F.relu(no_event * (dlog.abs() - tau))
    l_neg = excess.sum() / (no_event.sum() + 1.0)

    return l_pos + lambda_neg * l_neg, {'l_pos': l_pos.item(), 'l_neg': l_neg.item()}


# ---------------------------------------------------------------------------
# Full prototype model
# ---------------------------------------------------------------------------
class EventDynModel(nn.Module):
    """End-to-end prototype: baseline scene + event decoder + rasterize + loss.

    The baseline scene is provided as a dict of frozen tensors (the known
    high-fidelity GS scene). The learnable part is the EventEncoder +
    EventDynamicDecoder. Forward returns rendered images and the loss dict,
    so it can be trained with a standard optimizer loop.
    """

    def __init__(self, gauss_dim=64, event_dim=64, hidden=128, forder=2,
                 num_bins=5, polarity_split=True, use_pm=True, contrast=0.2):
        super().__init__()
        self.contrast = contrast
        in_ch = (2 if polarity_split else 1) * num_bins
        self.event_encoder = EventEncoder(in_ch=in_ch, hidden=64, out_dim=event_dim)
        self.dyn_decoder = EventDynamicDecoder(
            gauss_dim=gauss_dim, event_dim=event_dim, hidden=hidden,
            forder=forder, use_pm=use_pm,
        )
        self.num_bins = num_bins
        self.polarity_split = polarity_split
        self.gauss_dim = gauss_dim

    def forward(self, scene, events, t_query, render_fn, H, W):
        """scene: dict with frozen baseline Gaussians + per-Gaussian feature
                  'feat' [B,N,Dg] and projected pixel 'uv' [B,N,2].
        events: dict (see events_to_voxel).
        t_query: [B,1] query time in [0,1] for which to render G(t).
        render_fn: callable(means3D, colors, opacities, scales, rotations, H, W)
                   -> (color[H,W,3], depth, alpha). Use render_orth here.
        Returns dict with 'color', 'loss', 'metrics', 'increments'.
        """
        from model.torch_rasterizer import render_orth

        gauss_feat = scene['feat']      # [B,N,Dg]
        uv = scene['uv']                # [B,N,2]
        base_means = scene['means3D']   # [B,N,3]
        base_color = scene['rgb']       # [B,N,3]
        base_opacity = scene['opacity'] # [B,N,1]
        base_scales = scene['scale']    # [B,N,3]
        base_rot = scene['rot']         # [B,N,4]

        voxel = events_to_voxel(events, H, W, self.num_bins, self.polarity_split)
        event_feat = self.event_encoder(voxel, uv)  # [B,N,De]

        inc = self.dyn_decoder(gauss_feat, event_feat, t_query)

        means3D = base_means + inc['dmu']
        colors = (base_color + inc['dcolor']).clamp(0.0, 1.0)
        opacities = torch.sigmoid(torch.log(base_opacity.clamp(min=1e-4) /
                                            (1 - base_opacity.clamp(max=1-1e-4)) + 1e-6)
                                  + inc['dalpha'])
        scales = base_scales
        rot = base_rot

        B = means3D.shape[0]
        renders = []
        for b in range(B):
            color, depth, alpha = render_orth(
                means3D[b], colors[b], opacities[b], scales[b], rot[b], H, W,
                bg=scene.get('bg', None),
            )
            renders.append(color)
        render_t1 = torch.stack(renders, dim=0)  # [B,3,H,W]

        out = {'color': render_t1, 'increments': inc}

        # Loss against the reference render (t=0 of the window).
        render_t0 = scene['render_t0']  # [B,3,H,W] precomputed
        loss, metrics = event_consistency_loss(
            render_t0, render_t1, events, contrast=self.contrast
        )
        out['loss'] = loss
        out['metrics'] = metrics
        return out


# ---------------------------------------------------------------------------
# V2: observability-weighted event loss + CMax regularizer
# ---------------------------------------------------------------------------
def event_consistency_loss_v2(render_t0, render_t1, events, contrast=0.2,
                              tau=0.05, lambda_neg=0.5, obs_weight=None):
    """Observability-weighted event-consistency loss.

    Same bidirectional supervision as ``event_consistency_loss`` but
    weighted per-event by the per-Gaussian observability score
    ``obs_weight`` [B, N]. Events whose associated Gaussian lies on a
    well-observed corner/texture are trusted; events on straight edges
    (aperture-ambiguous) are down-weighted so they cannot pull the
    solution toward the degenerate no-motion minimum.

    The per-event observability is the per-Gaussian score of the
    Gaussian that fired the event, looked up via the precomputed
    pixel->Gaussian association.
    """
    from model.observability import observability_score, sample_gaussian_observability

    B, _, H, W = render_t1.shape
    eps = 1e-4
    log0 = torch.log(render_t0.clamp(min=eps))
    log1 = torch.log(render_t1.clamp(min=eps))
    dlog = (log1 - log0).sum(dim=1)  # [B,H,W]

    xy = events['pos'][:, :2].long()
    p = events['pos'][:, 3]
    pos_mask = torch.zeros(B, H, W, device=dlog.device)
    neg_mask = torch.zeros(B, H, W, device=dlog.device)
    sign = torch.zeros(B, H, W, device=dlog.device)
    valid = torch.zeros(B, H, W, device=dlog.device)
    for b in range(B):
        xi = xy[:, 0].clamp(0, W - 1)
        yi = xy[:, 1].clamp(0, H - 1)
        pos_mask[b, yi, xi] = (p > 0).float()
        neg_mask[b, yi, xi] = (p < 0).float()
        sign[b, yi, xi] = p.float()
        valid[b, yi, xi] = 1.0

    target = sign * contrast
    # If per-Gaussian observability weights are provided, build a
    # per-pixel weight map by scattering the Gaussian weight to the
    # event pixel. In the static-camera setting each event pixel maps
    # to a fixed Gaussian, so this is a simple scatter.
    if obs_weight is not None:
        # obs_weight may be a per-pixel map [B,H,W] (preferred, passed by
        # V2) or a per-Gaussian vector [B,N] (fallback). For the
        # per-Gaussian case callers scatter it externally; here we accept
        # both shapes.
        if obs_weight.dim() == 3 and obs_weight.shape[1] == H and obs_weight.shape[2] == W:
            w_pos = obs_weight
        else:
            w_pos = torch.ones_like(valid)
    else:
        w_pos = torch.ones_like(valid)

    l_pos = (w_pos * valid * (dlog - target).abs()).sum() / (valid.sum() + 1.0)
    no_event = 1.0 - valid
    excess = F.relu(no_event * (dlog.abs() - tau))
    l_neg = excess.sum() / (no_event.sum() + 1.0)
    return l_pos + lambda_neg * l_neg, {'l_pos': l_pos.item(), 'l_neg': l_neg.item()}


class EventDynModelV2(nn.Module):
    """EventGS three-layer model with observability-aware weighting and
    a Contrast-Maximization regularizer.

    Layers:
      1. Feed-forward event-conditioned decoder (same as V1).
      2. Online analysis-by-synthesis refinement with an
         observability-weighted event loss + CMax regularizer.
      3. Adaptive keyframe alignment scheduled by observedness.

    The CMax term is computed on the predicted 2D projected Gaussian
    flow and the observed events; it provides an aperture-breaking
    signal in every window, so the model is not dependent on a
    keyframe to disambiguate straight-edge motion.
    """

    def __init__(self, gauss_dim=64, event_dim=64, hidden=128, forder=2,
                 num_bins=5, polarity_split=True, use_pm=True, contrast=0.2,
                 lambda_cmax=0.05, cmax_sigma=1.5, obs_beta=2.0,
                 keyframe_threshold=0.25):
        super().__init__()
        self.contrast = contrast
        self.lambda_cmax = lambda_cmax
        self.cmax_sigma = cmax_sigma
        self.obs_beta = obs_beta
        in_ch = (2 if polarity_split else 1) * num_bins
        self.event_encoder = EventEncoder(in_ch=in_ch, hidden=64, out_dim=event_dim)
        self.dyn_decoder = EventDynamicDecoder(
            gauss_dim=gauss_dim, event_dim=event_dim, hidden=hidden,
            forder=forder, use_pm=use_pm,
        )
        self.num_bins = num_bins
        self.polarity_split = polarity_split
        self.gauss_dim = gauss_dim
        from model.observability import AdaptiveKeyframeScheduler
        self.scheduler = AdaptiveKeyframeScheduler(threshold=keyframe_threshold)

    def compute_obs_map(self, render_t0, uv):
        """Per-pixel observability map from the baseline render."""
        from model.observability import observability_score, sample_gaussian_observability
        obs_map = observability_score(render_t0)  # [B,H,W]
        obs_g = sample_gaussian_observability(obs_map, uv)  # [B,N]
        return obs_map, obs_g

    def forward(self, scene, events, t_query, render_fn, H, W,
                assoc=None, use_cmax=True):
        """Full forward with observability weighting + CMax.

        scene : dict with frozen baseline Gaussians; must include
                'render_t0' [B,3,H,W], 'uv' [B,N,2], 'uv_t0' optionally.
        events : dict (see events_to_voxel).
        assoc  : [B,H,W] long, precomputed pixel->Gaussian map. If
                 None, it is computed from scene['uv'] (cached for the
                 static camera).
        """
        from model.torch_rasterizer import render_orth
        from model.contrast_max import contrast_maximization_loss, pixel_to_gaussian_assoc
        from model.observability import observedness_loss_weight

        gauss_feat = scene['feat']
        uv = scene['uv']
        base_means = scene['means3D']
        base_color = scene['rgb']
        base_opacity = scene['opacity']
        base_scales = scene['scale']
        base_rot = scene['rot']
        render_t0 = scene['render_t0']

        # Observability from the baseline render.
        obs_map, obs_g = self.compute_obs_map(render_t0, uv)
        obs_w = observedness_loss_weight(obs_g, beta=self.obs_beta)  # [B,N]

        voxel = events_to_voxel(events, H, W, self.num_bins, self.polarity_split)
        event_feat = self.event_encoder(voxel, uv)
        inc = self.dyn_decoder(gauss_feat, event_feat, t_query)

        means3D = base_means + inc['dmu']
        colors = (base_color + inc['dcolor']).clamp(0.0, 1.0)
        opacities = torch.sigmoid(
            torch.log(base_opacity.clamp(min=1e-4) /
                      (1 - base_opacity.clamp(max=1 - 1e-4)) + 1e-6)
            + inc['dalpha'])

        B = means3D.shape[0]
        renders = []
        uv_t1_list = []
        for b in range(B):
            color, depth, alpha = render_orth(
                means3D[b], colors[b], opacities[b], base_scales[b], base_rot[b],
                H, W, bg=scene.get('bg', None))
            renders.append(color)
            # recompute projected uv at t1 (ortho: x->u, z->v)
            u1 = (means3D[b][:, 0] * 0.5 + 0.5) * (W - 1)
            v1 = (1.0 - (means3D[b][:, 2] * 0.5 + 0.5)) * (H - 1)
            uv_t1_list.append(torch.stack([u1, v1], dim=-1))
        render_t1 = torch.stack(renders, dim=0)
        uv_t1 = torch.stack(uv_t1_list, dim=0)

        # Event loss with observability weighting. We build a per-pixel
        # weight map by scattering obs_w at uv (each event inherits the
        # weight of the Gaussian at its pixel).
        from model.observability import observability_score
        B2, N = obs_w.shape
        pw = torch.ones(B2, H, W, device=obs_w.device)
        for b in range(B2):
            ui = uv[b, :, 0].long().clamp(0, W - 1)
            vi = uv[b, :, 1].long().clamp(0, H - 1)
            pw[b, vi, ui] = obs_w[b]
        loss_event, metrics = event_consistency_loss_v2(
            render_t0, render_t1, events, contrast=self.contrast,
            obs_weight=pw)

        loss = loss_event
        metrics['l_event'] = loss_event.item()

        # CMax regularizer on predicted 2D flow.
        if use_cmax and self.lambda_cmax > 0:
            if assoc is None:
                assoc = pixel_to_gaussian_assoc(uv, H, W)
            du = uv_t1 - uv  # [B,N,2] 2D projected flow
            # Map per-pixel assoc to per-event assoc by indexing at
            # event pixels.
            ev_xy = events['pos'][:, :2].long()
            ev_assoc = assoc[0, ev_xy[:, 1].clamp(0, H - 1),
                               ev_xy[:, 0].clamp(0, W - 1)]
            loss_cmax = contrast_maximization_loss(
                events, du, ev_assoc, H, W, sigma=self.cmax_sigma)
            loss = loss + self.lambda_cmax * loss_cmax
            metrics['l_cmax'] = loss_cmax.item()

        # Adaptive keyframe request (purely informational in the
        # prototype; the training loop uses it to decide whether to add
        # the photometric term).
        need_kf = self.scheduler.step(float(obs_map.mean()))
        metrics['obs_mean'] = float(obs_map.mean())
        metrics['need_keyframe'] = need_kf

        return {
            'color': render_t1,
            'increments': inc,
            'loss': loss,
            'metrics': metrics,
            'uv_t1': uv_t1,
            'obs_map': obs_map,
            'obs_weight': obs_w,
        }

