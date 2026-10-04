# Dynamic Gaussian Splatting Display — Research Notes

This note summarizes the literature survey on **display problems in dynamic
3D Gaussian Splatting**, conducted via Doubao web search (Oct 2026). It
motivates the design of the `display/` module integrated into the
StereoSplat project.

## Five recurring display problems

1. **Temporal flickering in static regions.**
   Yun et al., *Compensating Spatiotemporally Inconsistent Observations for
   Online Dynamic 3D Gaussian Splatting* (SIGGRAPH 2025), show that
   sensor noise in real-world recordings causes online dynamic GS methods
   to overfit per-frame errors, producing visible flicker in static
   regions. They propose learning a residual error map. Our
   `online_temporal_smooth` operator targets the same problem with a
   motion-adaptive EMA — high smoothing in static regions, low smoothing
   under motion.

2. **High-frequency motion blur and long-term drift under polynomial motion.**
   FAST-GS (arXiv:2608.01958, 2026) identifies that the single-polynomial
   motion model used by 4DGS suppresses high-frequency components (e.g.
   flickering flames) and accumulates trajectory drift over long
   sequences. They replace polynomials with Fourier series. DG-4DGS
   (VRIH 2026) adds a deformation graph for cross-frame geometric
   alignment, suppressing flicker in hair, cloth, and limbs. Our pipeline
   inherits StreamSplat's polynomial motion model but the display layer's
   motion-adaptive smoothing avoids compounding the blur.

3. **Popping (abrupt appearance/disappearance of Gaussians).**
   Feed-forward dynamic GS methods predict per-frame opacities that can
   swing between 0 and 1 across frames, producing a "popping" artifact.
   Instant4D (NeurIPS 2025) notes 4D Gaussians "tend to vanish
   prematurely as they are underconstrained in time." Our
   `popping_suppress` operator clamps the per-frame opacity delta and
   applies a soft EMA fade.

4. **CPU-bound splat sort → frame drops under fast camera motion.**
   The Three.js GaussianSplats3D thread documents that CPU-based splat
   sorting causes artifacts when the camera moves quickly. Gaussian Splat
   Lite (Sept 2026) introduces stochastic rendering to skip sorting during
   motion, and Fov-GS (TVCG 2025) uses foveated rendering for dynamic
   scenes (11.33× speedup). Our display module is offline / research, so
   we rely on the existing CUDA rasterizer's sort; the diagnostic
   `motion_energy` output of `StreamingDisplay.step` quantifies when this
   would be a problem.

5. **Scale ambiguity when rendering with mismatched intrinsics.**
   Standard 3DGS normalizes scene scale, so rendering with a different
   camera produces inconsistent Gaussians. StereoSplat's metric pipeline
   removes this ambiguity — Gaussians are in meters — and the display
   layer's `CameraPose` always carries the true intrinsics.

## Key references (with bib info)

| Ref | Title | Venue | Year |
|-----|-------|-------|------|
| Yun et al. | Compensating Spatiotemporally Inconsistent Observations for Online Dynamic 3D Gaussian Splatting | SIGGRAPH | 2025 |
| Zhang et al. | FAST-GS: Frequency Aware Space-Time Gaussian Splatting | arXiv:2608.01958 | 2026 |
| Chen et al. | DG-4DGS: deformation-graph-constrained 4D Gaussian splatting | VRIH 8(2):250–264 | 2026 |
| Fan et al. | Fov-GS: Foveated 3D Gaussian Splatting for Dynamic Scenes | TVCG 31:2975–2985 | 2025 |
| Wu et al. | 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering | CVPR | 2024 |
| SpeeDe3DGS | SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping | CVPR | 2026 |
| Kellogg | GaussianSplats3D (Three.js viewer) | GitHub | 2023 |
| Liu | Gaussian Splat Lite (WebGPU, streaming LOD) | three.js forum | 2026 |

## Integration into the project

The `display/` module is imported by:

  - `experiments/run_visualization.py` — dumps side-by-side videos of
    baseline vs. StereoSplat, with and without the temporal-smoothing /
    popping-suppression operators, for the qualitative comparison.
  - `experiments/run_ablation.py` — measures temporal-consistency metrics
    (per-pixel temporal variance in static regions, opacity delta
    distribution) as the display operators are toggled.

The display operators are torch-only and add no GUI dependency, so they
run inside the existing training / inference scripts on the same GPU.
