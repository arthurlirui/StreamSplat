"""
StreamingDisplay: online display / rendering loop for dynamic metric Gaussians.

Wraps `gaussian_renderer_perspective.render` with the temporal buffer and
smoothing operators in `display.buffer`, exposing a `step()` API that:

  1. Smooths the incoming per-frame Gaussians (motion-adaptive EMA + popping
     suppression).
  2. Renders the smoothed Gaussians with a perspective camera.
  3. Returns the rendered frame plus a small set of display diagnostics
     (mean motion energy, popping score, effective smoothing factor).

This is the integration point between the StereoSplat model output and a
real-time-ish display loop. It is also the artifact used by the experiment
harness to dump comparison videos and to measure temporal-consistency
metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch

from .buffer import (
    DynamicGaussianBuffer, GaussianFrame,
    online_temporal_smooth, popping_suppress,
)
import gaussian_renderer_perspective as renderer_perspective


@dataclass
class DisplayConfig:
    window: int = 5
    alpha_static: float = 0.8
    alpha_dynamic: float = 0.2
    motion_thresh: float = 0.05   # meters
    max_opacity_delta: float = 0.3
    fade_alpha: float = 0.5
    enable_smoothing: bool = True
    enable_popping_suppress: bool = True
    training: bool = False        # passed to rasterizer for bg augmentation


class StreamingDisplay:
    def __init__(self, opt, config: DisplayConfig = DisplayConfig()):
        self.opt = opt
        self.cfg = config
        self.buffer = DynamicGaussianBuffer(window=config.window)

    @torch.no_grad()
    def step(self, gaussians: Dict[str, torch.Tensor],
             camera: Any, t: float,
             background: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process one frame of Gaussians and render it.

        Parameters
        ----------
        gaussians : dict with keys 'xyz' [B,N,T,3], 'rgb' [B,N,3],
            'opacity' [B,N,1or2], 'scale' [B,N,2]. These are the *metric*
            Gaussians produced by `StereoSplatModel` (already back-projected
            into the world frame). We operate on batch index 0.
        camera : CameraPose (or dict) for the rendering viewpoint.
        t : timestamp of this frame (seconds).
        """
        # Take batch 0, static slice (T=0) of the means.
        means3D = gaussians['xyz'][0, :, 0, :]      # [N, 3]
        rgb = gaussians['rgb'][0]                   # [N, 3]
        opacity = gaussians['opacity'][0, :, :1]    # [N, 1]
        scale = gaussians['scale'][0]               # [N, 2]
        rot = gaussians.get('rot', None)
        if rot is not None:
            rot = rot[0, :, 0, :]
        cur = GaussianFrame(means3D=means3D, rgb=rgb, opacity=opacity,
                            scale=scale, rot=rot, t=t)

        prev = self.buffer.previous()
        if self.cfg.enable_smoothing:
            cur = online_temporal_smooth(
                cur, prev,
                alpha_static=self.cfg.alpha_static,
                alpha_dynamic=self.cfg.alpha_dynamic,
                motion_thresh=self.cfg.motion_thresh)
        if self.cfg.enable_popping_suppress:
            cur = popping_suppress(
                cur, prev,
                max_opacity_delta=self.cfg.max_opacity_delta,
                fade_alpha=self.cfg.fade_alpha)
        self.buffer.push(cur)

        # Re-pack into the rasterizer's expected layout (B=1).
        gs = {
            'xyz': cur.means3D.unsqueeze(0).unsqueeze(2),     # [1, N, 1, 3]
            'rgb': cur.rgb.unsqueeze(0),                       # [1, N, 3]
            'opacity': cur.opacity.unsqueeze(0),               # [1, N, 1]
            'scale': cur.scale.unsqueeze(0),                   # [1, N, 2]
        }
        if cur.rot is not None:
            gs['rot'] = cur.rot.unsqueeze(0).unsqueeze(2)      # [1, N, 1, 4]
        else:
            # Default identity rotation.
            N = cur.means3D.shape[0]
            r = torch.zeros(1, N, 2, 4, device=cur.means3D.device)
            r[..., 0, 0] = 1.0
            gs['rot'] = r

        bg = background if background is not None else torch.zeros(
            3, device=cur.means3D.device)
        pkg = renderer_perspective.render(
            gs, bg, opt=self.opt, training=self.cfg.training,
            cameras=[camera], camera_index=0)

        # Diagnostics.
        motion_energy = (cur.means3D - (prev.means3D if prev is not None
                                        else cur.means3D)).norm(dim=-1).mean()
        popping = (cur.opacity - (prev.opacity if prev is not None
                                  else cur.opacity)).abs().mean()
        pkg['motion_energy'] = motion_energy
        pkg['popping_score'] = popping
        return pkg
