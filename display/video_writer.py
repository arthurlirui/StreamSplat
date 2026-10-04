"""Video / frame dumping helpers for the display module."""
from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """[3, H, W] or [1, 3, H, W] float in [0, 1] → [H, W, 3] uint8."""
    if img.dim() == 4:
        img = img[0]
    img = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def write_frame(img: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(_to_uint8(img)).save(path)


def write_video(frames, path: str, fps: int = 30):
    """Write a list of [3, H, W] tensors to an MP4 file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2
        h, w = frames[0].shape[-2], frames[0].shape[-1]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(_to_uint8(f), cv2.COLOR_RGB2BGR))
        writer.release()
    except ImportError:
        # Fallback: imageio
        import imageio
        imageio.mimsave(path, [_to_uint8(f) for f in frames], fps=fps)
