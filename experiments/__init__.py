"""Experiment harness for StereoSplat.

This package contains three runnable experiment scripts:

  - `run_comparison.py`  — main performance comparison against baselines.
  - `run_ablation.py`    — ablation study across 5 axes (depth source,
                            input views, rasterizer, display operators,
                            loss terms).
  - `run_visualization.py` — side-by-side videos and depth comparisons.

Each script writes results under `experiments/results/<exp_name>/` (JSON +
CSV + MP4). They share the StereoSplat model, the stereo data provider,
and the display module.
"""
