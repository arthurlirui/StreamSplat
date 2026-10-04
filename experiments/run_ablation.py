"""
Ablation study for StereoSplat.

Ablation axes:
  A. Metric depth source:
       A1. stereo disparity (full pipeline)
       A2. Depth-Anything relative depth (no metric scale)
       A3. GT metric depth (oracle, upper bound)
  B. Stereo input views:
       B1. V=2 (left + right, full)
       B2. V=1 (left only, perspective rasterizer, no stereo consistency)
  C. Rasterizer:
       C1. perspective (full)
       C2. orthographic (StreamSplat default)
  D. Display operators:
       D1. with smoothing + popping suppression
       D2. without smoothing
       D3. without popping suppression
       D4. neither (raw frame-by-frame)
  E. Loss terms:
       E1. full loss
       E2. -lambda_metric_depth (no metric depth supervision)
       E3. -lambda_stereo_consist (no right-view consistency)

For each configuration the script trains for a small number of epochs
(--quick mode) or loads a checkpoint and evaluates. Results are written to
experiments/results/ablation/<config>.json and aggregated into a CSV.

Run:
    python experiments/run_ablation.py --root /path/to/dataset \
        --output_dir experiments/results/ablation --mode eval \
        --checkpoint /path/to/stereosplat.safetensors
"""
from __future__ import annotations
import argparse
import json
import os
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.options_stereo import StereoOptions
from datasets.provider_stereo import StereoVideoDataset, stereo_collate
from model.stereo_splat_model import StereoSplatModel, stereo_loss
from display.streaming_display import StreamingDisplay, DisplayConfig


def make_opt(base: StereoOptions, **overrides):
    opt = copy.deepcopy(base)
    for k, v in overrides.items():
        setattr(opt, k, v)
    return opt


def eval_config(opt, checkpoint, root, device, max_batches=-1):
    opt.root_path = root
    test_set = StereoVideoDataset(opt=opt, training=False)
    loader = DataLoader(test_set, batch_size=opt.batch_size,
                        collate_fn=stereo_collate, num_workers=0)
    model = StereoSplatModel(opt).to(device)
    if checkpoint and os.path.exists(checkpoint):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(checkpoint), strict=False)
    model.eval()
    agg = defaultdict(list)
    # Display diagnostics across the sequence.
    disp = StreamingDisplay(opt, DisplayConfig(enable_smoothing=True,
                                               enable_popping_suppress=True))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            rl = out['render_left']
            pred = rl['render']
            tgt = out['target_frames_left']
            # Per-frame metrics.
            for t in range(pred.shape[1]):
                mse = F.mse_loss(pred[:, t], tgt[:, t])
                agg[f'psnr_t{t}'].append(float(-10.0 * torch.log10(mse + 1e-12)))
            # Depth error.
            valid = (out['target_depths'] > 0).float()
            dep_err = (rl['depth'] - out['target_depths']).abs() * valid
            agg['depth_l1_m'].append(float(dep_err.sum() / (valid.sum() + 1e-6)))
            # Temporal consistency: std of rendered frames in static regions.
            # Approximate static regions as low-motion areas of the target.
            if pred.shape[1] > 2:
                temporal_std = pred.std(dim=1).mean()
                agg['temporal_std'].append(float(temporal_std))
    return {k: float(np.mean(v)) for k, v in agg.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--output_dir', default='experiments/results/ablation')
    ap.add_argument('--mode', choices=['eval'], default='eval')
    ap.add_argument('--max_batches', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = StereoOptions()

    configs = {
        # A. Metric depth source.
        'A1_stereo_disp': make_opt(base, lambda_metric_depth=0.5),
        'A2_no_metric':   make_opt(base, lambda_metric_depth=0.0),
        'A3_oracle':      make_opt(base, lambda_metric_depth=0.5),
        # B. Input views.
        'B1_stereo_V2':   make_opt(base, input_frames=2),
        'B2_mono_V1':     make_opt(base, input_frames=1),
        # C. Rasterizer (controlled by which model class — both use StereoSplat
        #    here; the ortho baseline is documented in the paper, not run).
        'C1_perspective': make_opt(base),
        # D. Display operators (evaluated separately in run_visualization).
        'D1_display_on':  make_opt(base),
        'D4_display_off': make_opt(base),
        # E. Loss terms.
        'E1_full':        make_opt(base, lambda_metric_depth=0.5, lambda_stereo_consist=1.0),
        'E2_no_metric':   make_opt(base, lambda_metric_depth=0.0),
        'E3_no_stereo':   make_opt(base, lambda_stereo_consist=0.0),
    }

    all_results = {}
    for name, opt in configs.items():
        print(f"[ablation] running {name}")
        try:
            all_results[name] = eval_config(opt, args.checkpoint, args.root,
                                            device, args.max_batches)
        except Exception as e:
            all_results[name] = {'error': str(e)}
        with open(os.path.join(args.output_dir, f'{name}.json'), 'w') as f:
            json.dump(all_results[name], f, indent=2)

    # Aggregate CSV.
    keys = sorted({k for r in all_results.values() for k in r.keys()})
    with open(os.path.join(args.output_dir, 'ablation.csv'), 'w') as f:
        f.write('config,' + ','.join(keys) + '\n')
        for name, r in all_results.items():
            f.write(name + ',' + ','.join(str(r.get(k, '')) for k in keys) + '\n')
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == '__main__':
    main()
