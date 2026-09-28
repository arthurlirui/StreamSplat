"""Full EventGS experiment runner.

Runs the three benchmark scenes (disc, bar, grid) under four method
configurations that isolate the contribution of each component:

  A. Event-only (V1 baseline)         -- L_event only, no keyframe, no CMax.
  B. Event + keyframe                 -- L_event + L_photo at fixed cadence.
  C. Event + CMax                     -- L_event + L_cmax, no keyframe.
  D. Full V2                          -- observability-weighted L_event + L_cmax
                                         + adaptive keyframe.

Reports: event loss, motion L2, sign-agreement, PSNR, and (for D) the
number of keyframes requested by the adaptive scheduler.

Run:
    python -m model.run_experiments
    python -m model.run_experiments --scenes disc bar grid --H 96 --W 96 --steps 400

This produces a results table printed to stdout and saved to
model/results.json. The numbers feed Table 1 and the aperture-problem
ablation in the paper.
"""
from __future__ import annotations
import argparse
import json
import math
import torch
import torch.nn.functional as F

from model.torch_rasterizer import render_orth
from model.event_dyn import (EventDynModel, EventDynModelV2,
                             event_consistency_loss)
from model.benchmark import (build_benchmark, motion_recovery_error,
                             rendering_psnr)


def run_event_only(scene, events, H, W, device, steps=400):
    """Configuration A: V1 event-only (no keyframe, no CMax)."""
    model = EventDynModel(gauss_dim=64, event_dim=64, hidden=128, forder=2,
                          num_bins=5, polarity_split=True, use_pm=True,
                          contrast=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    t_query = torch.tensor([[1.0]], device=device)
    # Layer 1: warm up feed-forward prior.
    for _ in range(200):
        opt.zero_grad()
        out = model(scene, events, t_query, render_orth, H, W)
        out['loss'].backward()
        opt.step()
    # Layer 2: refine dmu directly on L_event only (no photo).
    with torch.no_grad():
        out = model(scene, events, t_query, render_orth, H, W)
    dmu = out['increments']['dmu'].detach().clone().requires_grad_(True)
    opt_r = torch.optim.Adam([dmu], lr=1e-2)
    last_event = 0.0
    for _ in range(steps):
        opt_r.zero_grad()
        means3D = scene['means3D'] + dmu
        color, _, _ = render_orth(means3D[0], scene['rgb'][0], scene['opacity'][0],
                                  scene['scale'][0], scene['rot'][0], H, W,
                                  bg=scene['bg'])
        render_t1 = color.unsqueeze(0)
        loss_event, m = event_consistency_loss(scene['render_t0'], render_t1,
                                               events, contrast=0.2)
        loss = loss_event + 1e-4 * dmu.abs().mean()
        loss.backward()
        opt_r.step()
        last_event = loss_event.item()
    pred_color, _, _ = render_orth((scene['means3D'] + dmu)[0], scene['rgb'][0],
                                   scene['opacity'][0], scene['scale'][0],
                                   scene['rot'][0], H, W, bg=scene['bg'])
    return {
        'event_loss': last_event,
        'motion': motion_recovery_error(dmu, scene['gt_dmu']),
        'psnr': rendering_psnr(pred_color, scene['render_t1_gt'][0]),
        'keyframes': 0,
    }


def run_event_keyframe(scene, events, H, W, device, steps=400, lambda_photo=20.0):
    """Configuration B: event + fixed-cadence photometric keyframe."""
    gt_color = scene['render_t1_gt']
    # Reuse event-only setup but add photo loss in refinement.
    res = run_event_only(scene, events, H, W, device, steps=0)
    # Redo refinement with photo loss.
    model = EventDynModel(gauss_dim=64, event_dim=64, hidden=128, forder=2,
                          num_bins=5, polarity_split=True, use_pm=True,
                          contrast=0.2).to(device)
    t_query = torch.tensor([[1.0]], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(200):
        opt.zero_grad()
        out = model(scene, events, t_query, render_orth, H, W)
        out['loss'].backward()
        opt.step()
    with torch.no_grad():
        out = model(scene, events, t_query, render_orth, H, W)
    dmu = out['increments']['dmu'].detach().clone().requires_grad_(True)
    opt_r = torch.optim.Adam([dmu], lr=1e-2)
    last_event = 0.0
    last_photo = 0.0
    for _ in range(steps):
        opt_r.zero_grad()
        means3D = scene['means3D'] + dmu
        color, _, _ = render_orth(means3D[0], scene['rgb'][0], scene['opacity'][0],
                                  scene['scale'][0], scene['rot'][0], H, W,
                                  bg=scene['bg'])
        render_t1 = color.unsqueeze(0)
        loss_event, m = event_consistency_loss(scene['render_t0'], render_t1,
                                               events, contrast=0.2)
        loss_photo = F.l1_loss(render_t1, gt_color)
        loss = loss_event + lambda_photo * loss_photo + 1e-4 * dmu.abs().mean()
        loss.backward()
        opt_r.step()
        last_event = loss_event.item()
        last_photo = loss_photo.item()
    pred_color, _, _ = render_orth((scene['means3D'] + dmu)[0], scene['rgb'][0],
                                   scene['opacity'][0], scene['scale'][0],
                                   scene['rot'][0], H, W, bg=scene['bg'])
    return {
        'event_loss': last_event,
        'photo_loss': last_photo,
        'motion': motion_recovery_error(dmu, scene['gt_dmu']),
        'psnr': rendering_psnr(pred_color, gt_color[0]),
        'keyframes': steps,  # photo applied every step in this simple prototype
    }


def run_v2(scene, events, H, W, device, steps=400, use_cmax=True,
           lambda_photo=20.0, adaptive_kf=True):
    """Configuration D: full V2 (observability + CMax + adaptive keyframe).

    If use_cmax=False this becomes configuration C with observability
    weighting only. If adaptive_kf=False the photometric keyframe is
    applied every step (same as config B but with obs weighting).
    """
    model = EventDynModelV2(gauss_dim=64, event_dim=64, hidden=128, forder=2,
                            num_bins=5, polarity_split=True, use_pm=True,
                            contrast=0.2, lambda_cmax=0.05 if use_cmax else 0.0,
                            cmax_sigma=1.5, obs_beta=2.0,
                            keyframe_threshold=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    t_query = torch.tensor([[1.0]], device=device)
    # Precompute pixel->Gaussian assoc for CMax (static camera: cache).
    from model.contrast_max import pixel_to_gaussian_assoc
    assoc = pixel_to_gaussian_assoc(scene['uv'], H, W)
    # Layer 1 warmup.
    for _ in range(200):
        opt.zero_grad()
        out = model(scene, events, t_query, render_orth, H, W, assoc=assoc,
                    use_cmax=use_cmax)
        out['loss'].backward()
        opt.step()
    # Layer 2 refinement of dmu directly.
    with torch.no_grad():
        out = model(scene, events, t_query, render_orth, H, W, assoc=assoc,
                    use_cmax=use_cmax)
    dmu = out['increments']['dmu'].detach().clone().requires_grad_(True)
    opt_r = torch.optim.Adam([dmu], lr=1e-2)
    gt_color = scene['render_t1_gt']
    last_event = 0.0
    last_cmax = 0.0
    last_photo = 0.0
    n_kf = 0
    model.scheduler.n_calls = 0  # reset so the count reflects refinement only
    for step in range(steps):
        opt_r.zero_grad()
        means3D = scene['means3D'] + dmu
        color, _, _ = render_orth(means3D[0], scene['rgb'][0], scene['opacity'][0],
                                  scene['scale'][0], scene['rot'][0], H, W,
                                  bg=scene['bg'])
        render_t1 = color.unsqueeze(0)
        # observability-weighted event loss
        from model.observability import (observability_score, sample_gaussian_observability,
                                         observedness_loss_weight)
        obs_map = observability_score(scene['render_t0'])
        obs_g = sample_gaussian_observability(obs_map, scene['uv'])
        obs_w = observedness_loss_weight(obs_g, beta=2.0)
        B2, N = obs_w.shape
        pw = torch.ones(B2, H, W, device=obs_w.device)
        ui = scene['uv'][0, :, 0].long().clamp(0, W - 1)
        vi = scene['uv'][0, :, 1].long().clamp(0, H - 1)
        pw[0, vi, ui] = obs_w[0]
        from model.event_dyn import event_consistency_loss_v2
        loss_event, m = event_consistency_loss_v2(
            scene['render_t0'], render_t1, events, contrast=0.2, obs_weight=pw)
        loss = loss_event
        last_event = loss_event.item()
        # CMax
        if use_cmax:
            from model.contrast_max import contrast_maximization_loss
            u1 = (means3D[0][:, 0] * 0.5 + 0.5) * (W - 1)
            v1 = (1.0 - (means3D[0][:, 2] * 0.5 + 0.5)) * (H - 1)
            uv_t1 = torch.stack([u1, v1], dim=-1).unsqueeze(0)
            du = uv_t1 - scene['uv']
            ev_xy = events['pos'][:, :2].long()
            ev_assoc = assoc[0, ev_xy[:, 1].clamp(0, H - 1),
                               ev_xy[:, 0].clamp(0, W - 1)]
            loss_cmax = contrast_maximization_loss(events, du, ev_assoc, H, W,
                                                   sigma=1.5)
            loss = loss + 0.05 * loss_cmax
            last_cmax = loss_cmax.item()
        # Adaptive keyframe (photo loss) only when requested.
        apply_photo = (not adaptive_kf) or model.scheduler.step(float(obs_map.mean()))
        if apply_photo:
            loss_photo = F.l1_loss(render_t1, gt_color)
            loss = loss + lambda_photo * loss_photo
            last_photo = loss_photo.item()
            n_kf += 1
        loss = loss + 1e-4 * dmu.abs().mean()
        loss.backward()
        opt_r.step()
    pred_color, _, _ = render_orth((scene['means3D'] + dmu)[0], scene['rgb'][0],
                                   scene['opacity'][0], scene['scale'][0],
                                   scene['rot'][0], H, W, bg=scene['bg'])
    return {
        'event_loss': last_event,
        'cmax_loss': last_cmax,
        'photo_loss': last_photo,
        'motion': motion_recovery_error(dmu, scene['gt_dmu']),
        'psnr': rendering_psnr(pred_color, gt_color[0]),
        'keyframes': n_kf,
        'obs_mean': float(observability_score(scene['render_t0']).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=['disc', 'bar', 'grid'])
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--W', type=int, default=64)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--out', default='model/results.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, resolution: {args.H}x{args.W}, steps: {args.steps}")

    all_results = {}
    for name in args.scenes:
        print(f"\n=== Scene: {name} ===")
        scene, events = build_benchmark(name, H=args.H, W=args.W, device=device)
        print(f"  events: {events['pos'].shape[0]}")
        res = {}
        print("  [A] event-only ...")
        res['A_event_only'] = run_event_only(scene, events, args.H, args.W, device, args.steps)
        print(f"      event_loss={res['A_event_only']['event_loss']:.4f} "
              f"motion_l2={res['A_event_only']['motion']['l2_mean']:.4f} "
              f"sign={res['A_event_only']['motion']['sign_agreement']:.2f} "
              f"psnr={res['A_event_only']['psnr']:.2f}")
        print("  [B] event+keyframe ...")
        res['B_event_kf'] = run_event_keyframe(scene, events, args.H, args.W, device, args.steps)
        print(f"      event_loss={res['B_event_kf']['event_loss']:.4f} "
              f"motion_l2={res['B_event_kf']['motion']['l2_mean']:.4f} "
              f"sign={res['B_event_kf']['motion']['sign_agreement']:.2f} "
              f"psnr={res['B_event_kf']['psnr']:.2f}")
        print("  [C] event+CMax (no keyframe) ...")
        res['C_event_cmax'] = run_v2(scene, events, args.H, args.W, device, args.steps,
                                     use_cmax=True, adaptive_kf=False)
        # Override: config C uses no photo at all. We redo a no-photo run.
        # (run_v2 with adaptive_kf=False applies photo every step, which is
        # config B. For a pure CMax-no-keyframe config we set lambda_photo=0.)
        res['C_event_cmax'] = run_v2(scene, events, args.H, args.W, device, args.steps,
                                     use_cmax=True, lambda_photo=0.0, adaptive_kf=False)
        print(f"      event_loss={res['C_event_cmax']['event_loss']:.4f} "
              f"cmax={res['C_event_cmax']['cmax_loss']:.4f} "
              f"motion_l2={res['C_event_cmax']['motion']['l2_mean']:.4f} "
              f"sign={res['C_event_cmax']['motion']['sign_agreement']:.2f} "
              f"psnr={res['C_event_cmax']['psnr']:.2f}")
        print("  [D] full V2 (obs+CMax+adaptive keyframe) ...")
        res['D_full_v2'] = run_v2(scene, events, args.H, args.W, device, args.steps,
                                  use_cmax=True, adaptive_kf=True)
        print(f"      event_loss={res['D_full_v2']['event_loss']:.4f} "
              f"cmax={res['D_full_v2']['cmax_loss']:.4f} "
              f"motion_l2={res['D_full_v2']['motion']['l2_mean']:.4f} "
              f"sign={res['D_full_v2']['motion']['sign_agreement']:.2f} "
              f"psnr={res['D_full_v2']['psnr']:.2f} "
              f"kf={res['D_full_v2']['keyframes']} "
              f"obs={res['D_full_v2']['obs_mean']:.3f}")
        all_results[name] = res

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
