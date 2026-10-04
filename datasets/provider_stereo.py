"""Stereo-video dataset provider for StereoSplat.

A sequence is laid out on disk as:

    <root>/<sequence>/
        left/   000000.png 000001.png ...
        right/  000000.png 000001.png ...
        depth/  000000.npy        (optional GT metric depth, meters)
        disparity/ 000000.npy     (optional stereo disparity, pixels)
        pose.txt                   (optional per-frame trajectory, 12-col c2w)

If `depth/` is missing, the provider falls back to `disparity/` and converts
to metric depth via  Z = baseline * fx / disparity. If neither is present,
the provider expects a precomputed metric-depth cache produced by
`preprocess_disparity_stereo.py`.

The provider mirrors `provider_davis.py`'s public interface (`__getitem__`
returns a dict with `frames`, `depths`, `masks`, `poses`, `intrinsics`) so it
slots into the existing training loop with minimal changes.
"""
from __future__ import annotations
import os
import glob
from os import path as osp
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as tf
from PIL import Image


class StereoVideoDataset(Dataset):
    def __init__(self, opt, training: bool = True, shuffle: bool = False,
                 nearby_range: int = 4):
        self.opt = opt
        self.training = training
        self.shuffle = shuffle
        self.nearby_range = max(nearby_range, opt.output_frames - 1)

        self.root = opt.root_path
        if not osp.isdir(self.root):
            raise FileNotFoundError(f"Stereo root not found: {self.root}")

        # Enumerate sequences (each subdirectory containing left/ and right/)
        self.sequences = sorted([
            d for d in os.listdir(self.root)
            if osp.isdir(osp.join(self.root, d, 'left'))
            and osp.isdir(osp.join(self.root, d, 'right'))
        ])
        if len(self.sequences) == 0:
            raise RuntimeError(f"No stereo sequences found under {self.root}")

        # Build a flat index of (seq, frame_idx) pairs that have enough
        # future frames for the temporal window.
        self.index = []  # list of (seq_idx, center_idx)
        self.seq_meta = []  # per-sequence metadata (intrinsics, pose, lengths)
        for s_idx, sname in enumerate(self.sequences):
            left_dir = osp.join(self.root, sname, 'left')
            n = len(glob.glob(osp.join(left_dir, '*.png')))
            if n == 0:
                n = len(glob.glob(osp.join(left_dir, '*.jpg')))
            K, baseline = self._load_calib(osp.join(self.root, sname))
            poses = self._load_poses(osp.join(self.root, sname, 'pose.txt'), n)
            self.seq_meta.append({
                'name': sname, 'length': n,
                'K': K, 'baseline': baseline, 'poses': poses,
            })
            for i in range(n):
                if i + opt.output_frames <= n:
                    self.index.append((s_idx, i))

        self.set_transform()

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def _load_calib(self, seq_dir: str) -> Tuple[np.ndarray, float]:
        """Load intrinsics and baseline from `calib.txt` if present,
        otherwise fall back to the option-level defaults."""
        K = np.array([
            [self.opt.fx, 0, self.opt.cx],
            [0, self.opt.fy, self.opt.cy],
            [0, 0, 1],
        ], dtype=np.float32)
        baseline = float(self.opt.baseline)
        calib_path = osp.join(seq_dir, 'calib.txt')
        if osp.exists(calib_path):
            with open(calib_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('fx'):
                        K[0, 0] = float(line.split('=')[1])
                    elif line.startswith('fy'):
                        K[1, 1] = float(line.split('=')[1])
                    elif line.startswith('cx'):
                        K[0, 2] = float(line.split('=')[1])
                    elif line.startswith('cy'):
                        K[1, 2] = float(line.split('=')[1])
                    elif line.startswith('baseline'):
                        baseline = float(line.split('=')[1])
        return K, baseline

    def _load_poses(self, pose_path: str, n: int) -> Optional[np.ndarray]:
        """Load per-frame camera-to-world poses (12-col) if available.
        Returns [n, 3, 4] or None (static-rig fallback)."""
        if not osp.exists(pose_path):
            return None
        rows = []
        with open(pose_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nums = [float(x) for x in line.split()]
                if len(nums) == 12:
                    rows.append(nums)
        if len(rows) == 0:
            return None
        poses = np.array(rows, dtype=np.float32).reshape(-1, 3, 4)
        if poses.shape[0] == 1:
            poses = np.repeat(poses, n, axis=0)
        return poses  # [n, 3, 4] c2w

    def set_transform(self):
        H, W = self.opt.down_resolution
        self.transform = tf.Compose([
            tf.Resize((H, W)),
            tf.ToTensor(),
        ])
        self.depth_transform = tf.Compose([
            tf.Resize((H, W), interpolation=tf.InterpolationMode.NEAREST),
            tf.ToTensor(),
        ])
        self.mask_transform = tf.Compose([
            tf.Resize((H, W), interpolation=tf.InterpolationMode.NEAREST),
        ])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert('RGB')
        return img

    def _load_depth(self, seq_dir: str, idx: int, H: int, W: int) -> np.ndarray:
        """Return metric depth [H, W] in meters."""
        # Prefer precomputed metric depth.
        d_path = osp.join(seq_dir, 'depth', f'{idx:06d}.npy')
        if osp.exists(d_path):
            d = np.load(d_path).astype(np.float32)
            return d
        # Fall back to disparity -> metric depth.
        disp_path = osp.join(seq_dir, 'disparity', f'{idx:06d}.npy')
        if osp.exists(disp_path):
            disp = np.load(disp_path).astype(np.float32)
            disp = np.clip(disp, 1e-3, None)
            return float(self.opt.baseline) * float(self.opt.fx) / disp
        # No depth available — return ones (caller should mask it out).
        return np.ones((H, W), dtype=np.float32)

    def __getitem__(self, item: int):
        s_idx, center = self.index[item]
        meta = self.seq_meta[s_idx]
        seq_dir = osp.join(self.root, meta['name'])

        H, W = self.opt.down_resolution
        of = self.opt.output_frames
        # Sample `output_frames` consecutive frames ending at `center + of - 1`.
        # The first frame is the input context; the rest are supervision targets.
        frame_idxs = list(range(center, center + of))

        left_imgs, right_imgs, depths = [], [], []
        for fi in frame_idxs:
            l = self._load_image(osp.join(seq_dir, 'left', f'{fi:06d}.png'))
            r = self._load_image(osp.join(seq_dir, 'right', f'{fi:06d}.png'))
            left_imgs.append(self.transform(l))
            right_imgs.append(self.transform(r))
            d = self._load_depth(seq_dir, fi, H, W)
            d = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            # Resize via nearest-neighbor interpolation on the tensor directly
            # (depth_transform's ToTensor expects a PIL/ndarray input, not a
            # tensor, so we cannot reuse it here).
            d = F.interpolate(d, size=(H, W), mode='nearest').squeeze(0)  # [1,H,W]
            depths.append(d)

        # Left + right stacked as two views: [V=2, C, H, W]
        frames_left = torch.stack(left_imgs, dim=0)   # [of, 3, H, W]
        frames_right = torch.stack(right_imgs, dim=0)  # [of, 3, H, W]
        frames = torch.stack([frames_left[0], frames_right[0]], dim=0)  # [2, 3, H, W] (input)
        depths_in = torch.stack([depths[0], depths[0]], dim=0)  # [2, 1, H, W] (same depth for L/R at t=0)

        # Supervision targets (left + right for stereo-consistency loss)
        target_frames_left = frames_left  # [of, 3, H, W]
        target_frames_right = frames_right
        target_depths = torch.stack(depths, dim=0)  # [of, 1, H, W]

        # Poses: [of, 3, 4] c2w (left camera). Static rig if None.
        if meta['poses'] is not None:
            poses = meta['poses'][frame_idxs]  # [of, 3, 4]
        else:
            poses = np.zeros((of, 3, 4), dtype=np.float32)
            poses[:, :3, :3] = np.eye(3, dtype=np.float32)
            # Right camera = left + (baseline, 0, 0) in left frame.
        # Right-camera pose for stereo pair: identity-baseline offset.
        right_pose = np.zeros((3, 4), dtype=np.float32)
        right_pose[:3, :3] = np.eye(3, dtype=np.float32)
        right_pose[:, 3] = np.array([meta['baseline'], 0, 0], dtype=np.float32)

        return {
            'frames': frames,                 # [2, 3, H, W] input (L, R)
            'depths': depths_in,              # [2, 1, H, W] input metric depth
            'target_frames_left': target_frames_left,    # [of, 3, H, W]
            'target_frames_right': target_frames_right,  # [of, 3, H, W]
            'target_depths': target_depths,   # [of, 1, H, W]
            'intrinsics': torch.from_numpy(meta['K']).float(),     # [3, 3]
            'baseline': torch.tensor(meta['baseline']).float(),
            'poses_left': torch.from_numpy(poses).float(),         # [of, 3, 4] c2w
            'poses_right': torch.from_numpy(right_pose).float(),   # [3, 4] c2w (static)
            'sequence': meta['name'],
            'frame_idx': center,
        }


def stereo_collate(batch):
    """Default collate that keeps tensor fields and lists string fields."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
