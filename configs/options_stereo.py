"""Configuration for StereoSplat (metric-scale stereo-video dynamic 3DGS).

Inherits StreamSplat's decoder options and adds:
  - stereo intrinsics / baseline (the source of metric scale)
  - stereo dataset paths
  - metric-depth loss weights
  - per-frame camera-pose source (static rig or external SLAM trajectory)
"""
import tyro
from dataclasses import dataclass, field
from typing import Tuple, Literal, Dict, Optional, List
from configs.options import Options as BaseOptions


# ---- Default intrinsics (TartanAir left camera, 640x480) ----
# Override these per-dataset in the data provider.
default_fx: float = 320.0
default_fy: float = 320.0
default_cx: float = 320.0
default_cy: float = 240.0
default_baseline: float = 0.25  # meters (TartanAir stereo baseline)


@dataclass
class StereoOptions(BaseOptions):
    model_type = "StereoDecoder"
    mixed_precision: str = 'bf16'
    gradient_accumulation_steps: int = 1

    # ---- Stereo geometry (the metric-scale anchor) ----
    fx: float = default_fx
    fy: float = default_fy
    cx: float = default_cx
    cy: float = default_cy
    baseline: float = default_baseline
    stereo_image_height: int = 288
    stereo_image_width: int = 512

    # ---- Dataset paths ----
    root_path: str = ""
    stereo_path: str = ""          # generic stereo-video root (left/right folders)
    tartanair_path: str = ""
    dsec_path: str = ""
    kitti_stereo_path: str = ""

    batch_size: int = 8
    num_workers: int = 4
    resume: Optional[str] = None
    encoder_path: str = ""

    # ---- Optimization ----
    lr: float = 5e-4
    num_epochs: int = 200
    warmup_iters: int = 10000
    lr_decay_epochs: int = 200
    gradient_clip: float = 1.0
    forder: int = 1
    output_frames: int = 6
    input_frames: int = 2  # left + right

    # ---- Loss weights (metric additions) ----
    lambda_lpips: float = 0.05
    lambda_depth: float = 0.01       # metric-depth L1 (meters)
    lambda_metric_depth: float = 0.5
    lambda_stereo_consist: float = 1.0  # render-right vs. real-right
    lambda_dssim: float = 0.2
    lambda_reg: float = 0.0
    lambda_mask: float = 3.0
    ignore_large_loss: float = 0.3
    lpips_start_epoch: int = 50
    depth_start_epoch: int = 0

    workspace: str = './workspace_stereo'

    # ---- Model architecture (kept identical to StreamSplat) ----
    patch_size: int = 8
    decoder_num_layers: int = 10
    decoder_hidden_dim: int = 768
    decoder_ratio: float = 2.0
    opacity_activation: str = "sigmoid"
    use_augmentation: bool = True
    use_dino: bool = True
    drop_path_rate: float = 0.0
    pm_dynamic: bool = True
    skip: bool = False
    fix_opacity: bool = False
    compile: bool = False

    # ---- Probabilistic sampling (same as StreamSplat) ----
    use_pm: bool = True
    fix_keys: List[str] = field(default_factory=lambda: ["rot_static", "rot_dynamic"])
    sample_keys: List[str] = field(default_factory=lambda: ["xyz_dynamic"])
    pred_keys: List[str] = field(default_factory=lambda: ["rgb", "opacity", "scale", "xyz_static"])

    # ---- Metric-head scale range (meters) ----
    scale_min: float = 0.001
    scale_max: float = 0.3

    # ---- Pose source ----
    pose_source: Literal["static", "trajectory"] = "static"


config_defaults: Dict[str, StereoOptions] = {}
config_doc: Dict[str, str] = {}

config_doc['tartanair'] = 'TartanAir stereo (synthetic, GT depth & pose)'
config_defaults['tartanair'] = StereoOptions(root_path="", tartanair_path="PATH_TO_TARTANAIR")

config_doc['dsec'] = 'DSEC driving stereo'
config_defaults['dsec'] = StereoOptions(root_path="", dsec_path="PATH_TO_DSEC")

config_doc['kitti'] = 'KITTI Stereo 2015'
config_defaults['kitti'] = StereoOptions(root_path="", kitti_stereo_path="PATH_TO_KITTI")

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
