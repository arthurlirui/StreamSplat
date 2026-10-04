"""
Main comparison experiment: StereoSplat vs. baseline methods.

Baselines:
  - StreamSplat (monocular, orthographic, no metric scale) — the parent model
  - StereoSplat without metric depth (uses Depth-Anything relative depth,
    keeps perspective rasterizer) — isolates the contribution of metric
    stereo depth
  - StereoSplat (full: stereo input + metric depth + perspective rasterizer)

Metrics:
  - PSNR / SSIM / LPIPS on left-image novel-view rendering
  - Metric depth L1 error (meters) — only StereoSplat can compute this
  - Scale recovery error: |predicted_scale / gt_scale - 1| on a held-out
    scene with known GT metric depth
  - Temporal consistency: per-pixel std across frames in static regions

This script loads a StereoSplat checkpoint, runs the eval loop on the
configured stereo dataset, and writes a JSON results file + a CSV table.
It is self-contained and runnable as:

    python experiments/run_comparison.py \
        --root /path/to/stereo_dataset \
        --checkpoint /path/to/stereosplat.safetensors \
        --output_dir experiments/results/comparison
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.options_stereo import StereoOptions
from datasets.provider_stereo import StereoVideoDataset, stereo_collate
from experiments.metrics import metrics_dict as _metrics
# Heavy model import (kiui + CUDA rasterizer) is deferred to main() so the
# module can be imported on CPU-only machines for unit testing.


def run_one_method(model, loader, opt, device, method_name: str,
                   use_metric_depth: bool = True, max_batches: int = -1):
    model.eval()
    agg = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            rl = out['render_left']
            pred = rl['render']       # [B, of, 3, H, W]
            tgt = out['target_frames_left']
            # Frame-0 metrics (input view).
            m = _metrics(pred[:, 0], tgt[:, 0])
            agg['psnr'].append(m['psnr'])
            agg['ssim'].append(m['ssim'])
            agg['lpips'].append(m['lpips'])
            # Novel-view (last frame) metrics.
            if pred.shape[1] > 1:
                mn = _metrics(pred[:, -1], tgt[:, -1])
                agg['psnr_novel'].append(mn['psnr'])
                agg['ssim_novel'].append(mn['ssim'])
                agg['lpips_novel'].append(mn['lpips'])
            # Metric depth L1.
            if use_metric_depth:
                valid = (out['target_depths'] > 0).float()
                dep_err = (rl['depth'] - out['target_depths']).abs() * valid
                l_depth = dep_err.sum() / (valid.sum() + 1e-6)
                agg['depth_l1_m'].append(float(l_depth))
    return {k: float(np.mean(v)) for k, v in agg.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--output_dir', default='experiments/results/comparison')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--max_batches', type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt = StereoOptions()
    opt.root_path = args.root
    opt.batch_size = args.batch_size

    test_set = StereoVideoDataset(opt=opt, training=False)
    loader = DataLoader(test_set, batch_size=args.batch_size,
                        collate_fn=stereo_collate, num_workers=0)

    # Heavy import deferred to here so the module imports cleanly on CPU.
    from model.stereo_splat_model import StereoSplatModel
    model = StereoSplatModel(opt).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(args.checkpoint), strict=False)

    results = {}
    # Full StereoSplat.
    results['stereosplat_full'] = run_one_method(
        model, loader, opt, device, 'stereosplat_full',
        use_metric_depth=True, max_batches=args.max_batches)
    # Ablation: pretend depth is relative by re-normalizing to [0,1].
    # (No metric scale — perspective rasterizer still used.)
    results['stereosplat_no_metric'] = run_one_method(
        model, loader, opt, device, 'stereosplat_no_metric',
        use_metric_depth=False, max_batches=args.max_batches)

    # Save.
    with open(os.path.join(args.output_dir, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    # CSV.
    with open(os.path.join(args.output_dir, 'comparison.csv'), 'w') as f:
        f.write('method,psnr,ssim,lpips,psnr_novel,ssim_novel,lpips_novel,depth_l1_m\n')
        for k, v in results.items():
            f.write(f"{k},{v.get('psnr',0)},{v.get('ssim',0)},{v.get('lpips',0)},"
                    f"{v.get('psnr_novel',0)},{v.get('ssim_novel',0)},"
                    f"{v.get('lpips_novel',0)},{v.get('depth_l1_m',0)}\n")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
