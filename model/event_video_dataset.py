"""
Physical event-camera simulator for generating synchronized event + video data.

This module converts a standard video into a DAVIS-style synchronized dataset
of (APS grayscale frames, DVS event stream) pairs, using the standard
event-generation model

    log I(x, t_k) - log I(x, t_{k-1}) = p_k * C.

This is the same approach used by v2e (Hu et al., CVPRW 2021) and ESIM
(Rebecq et al., CoRL 2018), and is the standard pipeline used by recent
event-3DGS works (DEGS, ERF-GS, EdMCGS) when a real event camera is
unavailable: a regular video is converted to synthetic events so that an
event+RGB fusion algorithm can be tested on real-world dynamic content.

Two refinements over the toy `simulate_events` in benchmark.py:
  1. Temporal interpolation: consecutive frames are linearly interpolated
     in log-intensity space at a high internal rate so that fast motion
     produces multiple events per pixel per inter-frame interval, matching
     real DVS behavior.
  2. Per-pixel contrast thresholds with noise: each pixel has a slightly
     randomized threshold C +/- noise, and a refractory period, matching
     real sensor non-idealities.

Output format: a single .npz file per sequence with
  'frames'   : [T, H, W]    uint8 grayscale APS frames (the "video stream")
  'events'   : [M, 4]        float32 (x, y, t, p) in [0,W-1]x[0,H-1]x[t0,t1]x{-1,+1}
  'fps'      : float         original video fps
  'C'        : float         nominal contrast threshold used
The events and frames share the same time axis, so they are synchronized.
"""
from __future__ import annotations
import math
import numpy as np
import torch


def video_to_events_and_frames(video_path: str,
                               output_npz: str,
                               contrast: float = 0.15,
                               contrast_noise: float = 0.02,
                               refractory_ms: float = 0.5,
                               internal_rate: int = 8,
                               max_frames: int = 60,
                               target_h: int = 128,
                               target_w: int = 128,
                               seed: int = 0):
    """Convert a video to a synchronized (events, frames) dataset.

    Parameters
    ----------
    video_path : path to input video (mp4/avi).
    output_npz : path to write the .npz dataset.
    contrast : nominal log-intensity contrast threshold C.
    contrast_noise : per-pixel std of threshold noise.
    refractory_ms : minimum time between events at the same pixel (ms).
    internal_rate : number of interpolated sub-steps between consecutive
        frames (higher = more temporal resolution for event generation).
    max_frames : cap on the number of frames to process (keeps the dataset
        small).
    target_h, target_w : resize frames to this resolution (keeps the
        dataset small and matches our prototype rasterizer).
    """
    import imageio.v3 as iio
    import cv2

    rng = np.random.RandomState(seed)
    frames_rgb = iio.imread(video_path, index=None)  # [T, H, W, 3] uint8
    T_full = frames_rgb.shape[0]
    T = min(T_full, max_frames)
    frames_rgb = frames_rgb[:T]

    # Resize + grayscale.
    frames = np.zeros((T, target_h, target_w), dtype=np.uint8)
    for i in range(T):
        fr = cv2.resize(frames_rgb[i], (target_w, target_h),
                        interpolation=cv2.INTER_AREA)
        # ITU-R BT.601 luma.
        gray = (0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2])
        frames[i] = gray.astype(np.uint8)

    # Determine fps from video metadata if available.
    try:
        meta = iio.immeta(video_path, index=0)
        fps = float(meta.get('fps', 30.0))
    except Exception:
        fps = 30.0

    H, W = target_h, target_w
    log_I = np.log(frames.astype(np.float64) + 1.0)  # [T, H, W]
    # Per-pixel thresholds with noise.
    C_map = contrast + contrast_noise * rng.randn(H, W)
    C_map = np.clip(C_map, 0.05, 0.5)
    # Refractory period in frame-time units.
    refractory = max(1e-6, refractory_ms * 1e-3 * fps)
    last_event_t = np.full((H, W), -1e9, dtype=np.float64)

    events = []  # (x, y, t, p)
    # Internal time axis: between frame i and i+1 we insert `internal_rate`
    # sub-steps with linear log-intensity interpolation.
    for i in range(T - 1):
        t0, t1 = float(i), float(i + 1)
        a = log_I[i]
        b = log_I[i + 1]
        for s in range(internal_rate):
            ts = t0 + (s + 1) / internal_rate * (t1 - t0)
            # Interpolated log-intensity at sub-step.
            a_s = a + (b - a) * (s / internal_rate)
            b_s = a + (b - a) * ((s + 1) / internal_rate)
            dlog = b_s - a_s  # [H, W]
            # Pixels whose |dlog| exceeds their threshold fire an event.
            mag = np.abs(dlog)
            fires = np.where(mag > C_map)
            ys, xs = fires
            for yy, xx in zip(ys, xs):
                if (ts - last_event_t[yy, xx]) < refractory:
                    continue
                pol = 1.0 if dlog[yy, xx] > 0 else -1.0
                events.append((float(xx), float(yy), ts, pol))
                last_event_t[yy, xx] = ts

    if len(events) == 0:
        events_arr = np.zeros((0, 4), dtype=np.float32)
    else:
        events_arr = np.array(events, dtype=np.float32)

    np.savez_compressed(
        output_npz,
        frames=frames,           # [T,H,W] uint8 grayscale APS
        events=events_arr,       # [M,4] (x,y,t,p)
        fps=np.float64(fps),
        contrast=np.float64(contrast),
        H=np.int64(H), W=np.int64(W),
    )
    print(f"Saved {output_npz}: {T} frames, {events_arr.shape[0]} events, "
          f"{H}x{W}, fps={fps:.1f}, C={contrast}")
    return output_npz


class EventVideoDataset:
    """Loader for a synchronized event+video .npz dataset.

    Provides access to the APS frames (video stream) and the DVS event
    stream, with helpers to extract event windows between frames for the
    EventGS pipeline.
    """

    def __init__(self, npz_path: str, device: str = 'cpu'):
        d = np.load(npz_path, allow_pickle=False)
        self.frames = torch.from_numpy(d['frames']).float() / 255.0  # [T,H,W]
        self.events = torch.from_numpy(d['events'])  # [M,4]
        self.fps = float(d['fps'])
        self.contrast = float(d['contrast'])
        self.H = int(d['H']); self.W = int(d['W'])
        self.device = device
        self.frames = self.frames.to(device)
        self.events = self.events.to(device)
        self.T = self.frames.shape[0]

    def frame_rgb(self, i: int) -> torch.Tensor:
        """Return APS frame i as a [3,H,W] RGB-like tensor (grayscale
        broadcast to 3 channels so the GS rasterizer's color path works)."""
        g = self.frames[i].unsqueeze(0).expand(3, -1, -1)
        return g  # [3,H,W]

    def events_between(self, i: int, j: int) -> dict:
        """Return events with timestamps in [i, j] (frame indices as time)."""
        t = self.events[:, 2]
        mask = (t >= float(i)) & (t < float(j))
        pos = self.events[mask]
        if pos.shape[0] == 0:
            pos = torch.zeros(0, 4, device=self.device)
        return {
            'pos': pos,
            't_min': torch.tensor(float(i), device=self.device),
            't_max': torch.tensor(float(j), device=self.device),
        }

    def __len__(self):
        return self.T


if __name__ == '__main__':
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/event_video/jellyfish.mp4'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'data/event_video/jellyfish_events.npz'
    video_to_events_and_frames(video_path, out_path,
                               contrast=0.15, max_frames=30,
                               target_h=128, target_w=128)
