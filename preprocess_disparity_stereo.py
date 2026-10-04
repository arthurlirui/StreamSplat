"""Offline stereo-disparity -> metric-depth preprocessor for StereoSplat.

For each sequence under `<root>/<sequence>/` with `left/` and `right/` image
folders, this script:

  1. Runs a stereo-matching network on each (left, right) pair to produce a
     dense disparity map. By default it uses a simple block-matching fallback
     (OpenCV StereoSGBM) so the script runs without GPU deps; for research
     quality, swap in IGEV-Stereo / RAFT-Stereo via the `--backend` flag.
  2. Converts disparity to metric depth via  Z = baseline * fx / disparity.
  3. Writes `<root>/<sequence>/depth/<frame_idx:06d>.npy` (float32, meters).

Usage:
    python preprocess_disparity_stereo.py \
        --root /path/to/stereo_dataset \
        --backend sgbm \
        --fx 320 --fy 320 --cx 320 --cy 240 --baseline 0.25

The output is consumed directly by `datasets.provider_stereo.StereoVideoDataset`.
"""
from __future__ import annotations
import argparse
import glob
import os
from os import path as osp

import numpy as np
from PIL import Image


def _load_pairs(seq_dir: str):
    left_dir = osp.join(seq_dir, 'left')
    right_dir = osp.join(seq_dir, 'right')
    exts = ['*.png', '*.jpg']
    left_files = []
    for e in exts:
        left_files = sorted(glob.glob(osp.join(left_dir, e)))
        if left_files:
            break
    if not left_files:
        return []
    pairs = []
    for lf in left_files:
        base = osp.splitext(osp.basename(lf))[0]
        rf = None
        for e in ['.png', '.jpg']:
            cand = osp.join(right_dir, base + e)
            if osp.exists(cand):
                rf = cand
                break
        if rf is None:
            continue
        pairs.append((lf, rf))
    return pairs


def _sgbm_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """OpenCV StereoSGBM fallback (no GPU required)."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "OpenCV is required for the 'sgbm' backend. "
            "Install with `pip install opencv-python`.") from e
    if left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    min_disp = 0
    num_disp = 96  # must be divisible by 16
    block = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disp = matcher.compute(left, right).astype(np.float32) / 16.0
    return disp  # pixels (negative for invalid -> clip later)


def _to_depth(disp: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    disp = np.clip(disp, 1e-3, None)
    return float(baseline) * float(fx) / disp


def process_sequence(seq_dir: str, fx: float, baseline: float,
                     backend: str = 'sgbm'):
    pairs = _load_pairs(seq_dir)
    if not pairs:
        print(f"[skip] no image pairs in {seq_dir}")
        return 0
    out_dir = osp.join(seq_dir, 'depth')
    os.makedirs(out_dir, exist_ok=True)
    for i, (lf, rf) in enumerate(pairs):
        out_path = osp.join(out_dir, f'{i:06d}.npy')
        if osp.exists(out_path):
            continue
        li = np.array(Image.open(lf).convert('RGB'))
        ri = np.array(Image.open(rf).convert('RGB'))
        if backend == 'sgbm':
            disp = _sgbm_disparity(li, ri)
        else:
            raise ValueError(f"Unknown backend '{backend}'. "
                             f"Plug in IGEV/RAFT-Stereo here.")
        depth = _to_depth(disp, fx, baseline).astype(np.float32)
        np.save(out_path, depth)
    print(f"[done] {seq_dir}: {len(pairs)} depth maps -> {out_dir}")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='stereo dataset root')
    ap.add_argument('--backend', default='sgbm',
                    choices=['sgbm'],  # extend with 'igev', 'raft' later
                    help='stereo matching backend')
    ap.add_argument('--fx', type=float, default=320.0)
    ap.add_argument('--fy', type=float, default=320.0)
    ap.add_argument('--cx', type=float, default=320.0)
    ap.add_argument('--cy', type=float, default=240.0)
    ap.add_argument('--baseline', type=float, default=0.25)
    args = ap.parse_args()

    seqs = sorted([
        d for d in os.listdir(args.root)
        if osp.isdir(osp.join(args.root, d, 'left'))
        and osp.isdir(osp.join(args.root, d, 'right'))
    ])
    if not seqs:
        raise RuntimeError(f"No stereo sequences under {args.root}")
    total = 0
    for s in seqs:
        total += process_sequence(osp.join(args.root, s),
                                  args.fx, args.baseline, args.backend)
    print(f"[finished] {total} depth maps across {len(seqs)} sequences.")


if __name__ == '__main__':
    main()
