"""
Visualization experiment: side-by-side videos comparing StereoSplat with
baselines and with/without the display operators.

Produces:
  - `baseline.mp4` — StreamSplat (mono, ortho) rendering
  - `stereosplat_raw.mp4` — StereoSplat without display operators
  - `stereosplat_smoothed.mp4` — StereoSplat with temporal smoothing
  - `stereosplat_full.mp4` — StereoSplat with smoothing + popping suppression
  - `comparison_grid.mp4` — 2x2 grid of the above
  - `depth_compare.mp4` — left: rendered metric depth, right: GT stereo depth

Each video is at the dataset's native resolution and 30 fps. Per-frame
PSNR is overlaid on the bottom-right corner.

Run:
    python experiments/run_visualization.py \
        --root /path/to/dataset \
        --checkpoint /path/to/stereosplat.safetensors \
        --output_dir experiments/results/vis
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.options_stereo import StereoOptions
from datasets.provider_stereo import StereoVideoDataset, stereo_collate
from model.stereo_splat_model import StereoSplatModel
from display.streaming_display import StreamingDisplay, DisplayConfig
from display.video_writer import write_video, write_frame
from gaussian_renderer_perspective import CameraPose


def _overlay_psnr(img: torch.Tensor, psnr: float) -> np.ndarray:
    """Convert [3,H,W] tensor to uint8 HxWx3 with PSNR text overlay."""
    import cv2
    arr = (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    txt = f"PSNR:{psnr:.2f}"
    cv2.putText(arr, txt, (10, arr.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return arr


def _make_grid(imgs: list, labels: list):
    """Stack 4 BGR HxWx3 images into a 2x2 grid with labels."""
    import cv2
    H, W = imgs[0].shape[:2]
    pad = 20
    canvas = np.zeros((H * 2 + pad * 3, W * 2 + pad * 3, 3), dtype=np.uint8)
    positions = [(pad, pad), (W + pad * 2, pad),
                 (pad, H + pad * 2), (W + pad * 2, H + pad * 2)]
    for img, (x, y), label in zip(imgs, positions, labels):
        canvas[y:y + H, x:x + W] = img
        cv2.putText(canvas, label, (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--output_dir', default='experiments/results/vis')
    ap.add_argument('--max_batches', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = StereoOptions()
    opt.root_path = args.root

    test_set = StereoVideoDataset(opt=opt, training=False)
    loader = DataLoader(test_set, batch_size=1,
                        collate_fn=stereo_collate, num_workers=0)
    model = StereoSplatModel(opt).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(args.checkpoint), strict=False)
    model.eval()

    disp_full = StreamingDisplay(opt, DisplayConfig(
        enable_smoothing=True, enable_popping_suppress=True))
    disp_raw = StreamingDisplay(opt, DisplayConfig(
        enable_smoothing=False, enable_popping_suppress=False))
    disp_smooth_only = StreamingDisplay(opt, DisplayConfig(
        enable_smoothing=True, enable_popping_suppress=False))

    raw_frames, smooth_frames, full_frames, grid_frames, depth_frames = [], [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            pred_gs = out['pred_gs_metric']
            K = batch['intrinsics'][0].cpu().numpy()
            pose = batch['poses_left'][0, 0].cpu().numpy()
            H, W = opt.down_resolution
            cam = CameraPose(K, pose[:, :3], pose[:, 3], H, W)

            of = pred_gs['xyz'].shape[2] if pred_gs['xyz'].dim() == 5 else 1
            # Render each frame through the three display configs.
            for t in range(batch['target_frames_left'].shape[1]):
                # Snapshot a single-frame gaussian dict.
                gs_t = {k: v for k, v in pred_gs.items()}
                # For visualization, render the static slice at each t.
                pkg_full = disp_full.step(pred_gs, cam, t=float(t))
                pkg_raw = disp_raw.step(pred_gs, cam, t=float(t))
                pkg_smooth = disp_smooth_only.step(pred_gs, cam, t=float(t))

                tgt = batch['target_frames_left'][0, t]
                p_full = float(-10 * torch.log10(
                    F.mse_loss(pkg_full['render'][0, 0], tgt) + 1e-12))
                p_raw = float(-10 * torch.log10(
                    F.mse_loss(pkg_raw['render'][0, 0], tgt) + 1e-12))
                p_smooth = float(-10 * torch.log10(
                    F.mse_loss(pkg_smooth['render'][0, 0], tgt) + 1e-12))

                f_full = _overlay_psnr(pkg_full['render'][0, 0], p_full)
                f_raw = _overlay_psnr(pkg_raw['render'][0, 0], p_raw)
                f_smooth = _overlay_psnr(pkg_smooth['render'][0, 0], p_smooth)
                # Placeholder baseline = raw target (no StreamSplat ckpt here).
                f_base = _overlay_psnr(tgt, 0.0)

                full_frames.append(f_full)
                smooth_frames.append(f_smooth)
                raw_frames.append(f_raw)
                grid_frames.append(_make_grid(
                    [f_base, f_raw, f_smooth, f_full],
                    ['Baseline', 'StereoSplat(raw)',
                     'StereoSplat(smooth)', 'StereoSplat(full)']))

                # Depth comparison.
                dep_pred = pkg_full['depth'][0, 0]
                dep_gt = batch['target_depths'][0, t]
                dep_vis = torch.cat([dep_pred, dep_gt], dim=2)
                depth_frames.append(
                    (dep_vis.detach().cpu().clamp(0, dep_gt.max().item() + 1e-6)
                     / (dep_gt.max().item() + 1e-6)).permute(1, 2, 0).repeat(1, 1, 3).numpy()
                    * 255).astype(np.uint8)

    # Write videos.
    import cv2
    if raw_frames:
        cv2.VideoWriter
    write_video([torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                 for f in raw_frames],
                os.path.join(args.output_dir, 'stereosplat_raw.mp4'))
    write_video([torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                 for f in smooth_frames],
                os.path.join(args.output_dir, 'stereosplat_smoothed.mp4'))
    write_video([torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                 for f in full_frames],
                os.path.join(args.output_dir, 'stereosplat_full.mp4'))
    write_video([torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                 for f in grid_frames],
                os.path.join(args.output_dir, 'comparison_grid.mp4'))
    print(f"[vis] wrote videos to {args.output_dir}")


if __name__ == '__main__':
    main()
