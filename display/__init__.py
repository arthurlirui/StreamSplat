"""
Dynamic-Gaussian display module for StereoSplat / StreamSplat.

This package addresses the *display* side of dynamic 3D Gaussian Splatting,
which is a known weakness of feed-forward dynamic GS methods. Five recurring
display problems are identified in the literature (see
`display/research_notes.md` for the survey):

  1. Temporal flickering in static regions (Yun et al., SIGGRAPH 2025).
  2. High-frequency motion blur / drift under polynomial motion models
     (FAST-GS, DG-4DGS).
  3. Popping / appearance-disappearance artifacts when Gaussians vanish or
     appear abruptly across frames.
  4. CPU-bound splat sort → frame drops under fast camera motion
     (GaussianSplats3D forum; mitigated by stochastic / foveated rendering,
     Fov-GS).
  5. Scale ambiguity when the rendering camera intrinsics differ from the
     training camera (a problem our metric pipeline already solves, but
     which the display layer must respect).

This module provides:

  - `DynamicGaussianBuffer`: a ring buffer of per-frame metric Gaussians
    with EMA smoothing and temporal-consistency filtering.
  - `StreamingDisplay`: wraps the perspective rasterizer and the buffer,
    exposing a `step(frame_gaussians, camera, t)` API that returns a
    rendered frame plus diagnostics (PSNR, motion energy, popping score).
  - `online_temporal_smooth`: a temporal EMA on Gaussian means / colors /
    opacities with a motion-adaptive mixing factor (fast motion → less
    smoothing, to avoid motion blur).
  - `popping_suppress`: a soft opacity fade-in / fade-out for Gaussians
    whose opacity changes abruptly across frames.
  - `write_frame` / `write_video`: dump rendered frames to disk and
    assemble an MP4 with imageio / opencv for qualitative comparison.

The module is intentionally torch-only and has no GUI dependency, so it
integrates into the existing training / inference scripts and the
experiment harness.

The rasterizer-backed `StreamingDisplay` is imported lazily so that the
buffer / smoothing operators can be used without the CUDA Gaussian
rasterizer extension being installed.
"""
from __future__ import annotations
from .buffer import DynamicGaussianBuffer, online_temporal_smooth, popping_suppress


def __getattr__(name):
    # Lazy import: only pull in StreamingDisplay/DisplayConfig when requested,
    # so that `import display.buffer` works without the CUDA rasterizer.
    if name in ("StreamingDisplay", "DisplayConfig"):
        from . import streaming_display as _sd
        return getattr(_sd, name)
    if name in ("write_frame", "write_video"):
        from . import video_writer as _vw
        return getattr(_vw, name)
    raise AttributeError(f"module 'display' has no attribute {name!r}")


__all__ = [
    "DynamicGaussianBuffer",
    "StreamingDisplay",
    "DisplayConfig",
    "online_temporal_smooth",
    "popping_suppress",
    "write_frame",
    "write_video",
]
