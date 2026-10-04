"""
Temporal buffer and smoothing operators for dynamic Gaussian display.

The buffer stores a short history of per-frame Gaussian attribute tensors
(means3D, rgb, opacity, scale) so that temporal operators can run on a
sliding window. Two operators are provided:

  - `online_temporal_smooth`: exponential moving average of means / rgb /
    opacity, with a motion-adaptive mixing factor. When the per-Gaussian
    motion energy is high the EMA weight is reduced (less smoothing) to
    avoid introducing motion blur; in static regions the EMA weight is high
    to suppress flicker. This directly targets display problem (1).

  - `popping_suppress`: softens abrupt opacity changes by clamping the
    per-frame opacity delta and applying a short temporal fade. This
    targets display problem (3), the appearance/disappearance popping that
    is especially visible in feed-forward dynamic GS under fast motion.

Both operators preserve the metric (meters) convention of the StereoSplat
pipeline.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class GaussianFrame:
    """A single frame's worth of Gaussians in metric world coordinates."""
    means3D: torch.Tensor      # [N, 3]
    rgb: torch.Tensor          # [N, 3]
    opacity: torch.Tensor      # [N, 1]
    scale: torch.Tensor        # [N, 2] or [N, 3]
    rot: Optional[torch.Tensor] = None  # [N, 4]
    t: float = 0.0


class DynamicGaussianBuffer:
    """Ring buffer of recent GaussianFrame objects.

    The buffer is keyed by attribute and keeps at most `window` frames.
    It is used by the smoothing / popping operators below.
    """

    def __init__(self, window: int = 5):
        self.window = max(2, window)
        self._frames: deque = deque(maxlen=self.window)

    def push(self, frame: GaussianFrame):
        self._frames.append(frame)

    def __len__(self):
        return len(self._frames)

    def previous(self) -> Optional[GaussianFrame]:
        if len(self._frames) < 2:
            return None
        return self._frames[-2]

    def latest(self) -> Optional[GaussianFrame]:
        return self._frames[-1] if self._frames else None

    def history(self):
        return list(self._frames)


def online_temporal_smooth(cur: GaussianFrame,
                           prev: Optional[GaussianFrame],
                           alpha_static: float = 0.8,
                           alpha_dynamic: float = 0.2,
                           motion_thresh: float = 0.05) -> GaussianFrame:
    """EMA smoothing with a motion-adaptive mixing factor.

    Parameters
    ----------
    cur, prev : current and previous GaussianFrame (same N, same order).
    alpha_static : EMA weight in static regions (high → strong smoothing).
    alpha_dynamic : EMA weight in high-motion regions (low → little smoothing
        to avoid motion blur).
    motion_thresh : per-Gaussian motion magnitude (meters) above which the
        dynamic mixing factor is used.

    Returns a new GaussianFrame with smoothed means / rgb / opacity.
    Scale and rotation are kept from `cur` (smoothing them is harmful — it
    blurs geometry). If `prev` is None, `cur` is returned unchanged.
    """
    if prev is None:
        return cur
    # Per-Gaussian motion magnitude (meters).
    motion = (cur.means3D - prev.means3D).norm(dim=-1, keepdim=True)  # [N, 1]
    is_static = (motion < motion_thresh).float()
    alpha = alpha_static * is_static + alpha_dynamic * (1.0 - is_static)

    means = alpha * prev.means3D + (1.0 - alpha) * cur.means3D
    rgb = alpha * prev.rgb + (1.0 - alpha) * cur.rgb
    opacity = alpha * prev.opacity + (1.0 - alpha) * cur.opacity
    return GaussianFrame(
        means3D=means, rgb=rgb, opacity=opacity,
        scale=cur.scale, rot=cur.rot, t=cur.t)


def popping_suppress(cur: GaussianFrame,
                     prev: Optional[GaussianFrame],
                     max_opacity_delta: float = 0.3,
                     fade_alpha: float = 0.5) -> GaussianFrame:
    """Suppress abrupt opacity changes (appearance/disappearance popping).

    The opacity delta between consecutive frames is clamped to
    `max_opacity_delta`, and a soft EMA with weight `fade_alpha` is applied
    on top. This prevents Gaussians from flashing in/out within a single
    frame, which is the most perceptually jarring dynamic-GS display
    artifact.
    """
    if prev is None:
        return cur
    delta = cur.opacity - prev.opacity
    clamped_delta = delta.clamp(-max_opacity_delta, max_opacity_delta)
    faded = prev.opacity + clamped_delta
    opacity = fade_alpha * faded + (1.0 - fade_alpha) * cur.opacity
    opacity = opacity.clamp(0.0, 1.0)
    return GaussianFrame(
        means3D=cur.means3D, rgb=cur.rgb, opacity=opacity,
        scale=cur.scale, rot=cur.rot, t=cur.t)
