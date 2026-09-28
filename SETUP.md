# EventGS — Environment & Run Guide

## Conda environment

The project uses the existing conda env at `D:\conda-envs\gs` (Python 3.11, torch 2.5.1+cu121, CUDA available).

### Verified package state

| Package | Version | Status |
|---|---|---|
| python | 3.11.15 | OK |
| torch | 2.5.1+cu121 | OK, CUDA available |
| numpy | 2.4.6 | OK |
| matplotlib | 3.11.1 | OK |
| scipy | 1.17.1 | OK |
| torchvision | 0.20.1+cu121 | OK |
| lpips | — | OK |
| einops | 0.8.2 | **installed this session** |
| seaborn | 0.13.2 | **installed this session** |
| kiui | 0.3.5 | **installed this session** |
| opencv-python | 5.0.0 | **installed this session** |

Install command (if recreating):
```bash
D:\conda-envs\gs\python.exe -m pip install einops seaborn opencv-python kiui
```

## CUDA rasterizer (optional)

The submodule `submodules/diff-gaussian-rasterization-orth/` is **not built**. The EventGS prototype uses the pure-PyTorch drop-in `model/torch_rasterizer.py` instead, so the CUDA rasterizer is **not required** to run any of the code in `model/`. To build it for production:
```bash
cd submodules/diff-gaussian-rasterization-orth
D:\conda-envs\gs\python.exe -m pip install .
```

## Running the system

All commands run from the repo root `D:\Code\StreamSplat` using `D:\conda-envs\gs\python.exe`.

### 1. Prototype demo (V1, event-only + keyframe)
```bash
D:\conda-envs\gs\python.exe -m model.demo_event_dyn
```
Expected: event loss drops 5.6×, `OK: event-driven dynamic decoder trains and reduces event-consistency loss.`

### 2. Full experiments (V2, 3 scenes × 4 configs)
```bash
D:\conda-envs\gs\python.exe -m model.run_experiments --scenes disc bar grid --H 64 --W 64 --steps 300
```
Expected: prints per-scene table, saves `model/results.json`. Key result: bar scene sign-agreement 0.53 (event-only) → 0.99 (+keyframe) → 0.64 (full V2, 3 keyframes).

### 3. Unit tests (observability, CMax, benchmark)
Inline test suite covering 9 modules — run via the test script in the session log. All pass.

### 4. Paper build
```bash
cd latex
PATH="/c/texlive/2024/bin/windows:$PATH" pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Expected: 8-page PDF, 0 errors, 0 undefined citations.

## Verified run results (2026-09-27)

All three test layers passed:
- **demo_event_dyn.py**: loss 0.83 → 0.15 (5.6×), motion sign correct ✓
- **unit tests**: 9/9 suites passed (observability, CMax, benchmark, metrics) ✓
- **run_experiments.py**: 3 scenes × 4 configs, results match paper Table 1 ✓
- **latex build**: 8 pages, 0 errors ✓
