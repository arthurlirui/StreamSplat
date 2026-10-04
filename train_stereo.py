"""Training entry for StereoSplat (metric-scale stereo-video dynamic 3DGS).

This mirrors `train_decoder.py` but targets the stereo provider and the
`StereoSplatModel`. It supports two sub-modes:

  * `train_stereo.py tartanair --tartanair_path ...` — supervised training on
    TartanAir stereo (GT metric depth & pose).
  * `train_stereo.py dsec --dsec_path ...` — DSEC driving stereo.

Stage 1 (static encoder) uses `train_stereo.py --stage encoder`; Stage 2
(dynamic decoder) uses `--stage decoder --encoder_path <ckpt>`.
"""
from __future__ import annotations
import os
import datetime
from collections import OrderedDict

import tyro
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from configs.options_stereo import AllConfigs, StereoOptions
from datasets.provider_stereo import StereoVideoDataset, stereo_collate
from model.stereo_splat_model import StereoSplatModel, stereo_loss


def main():
    set_seed(42)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    opt: StereoOptions = tyro.cli(AllConfigs)

    torch.set_float32_matmul_precision('high')

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    # Resolve dataset root from the chosen subcommand.
    if opt.tartanair_path and opt.tartanair_path != "PATH_TO_TARTANAIR":
        opt.root_path = opt.tartanair_path
    elif opt.dsec_path and opt.dsec_path != "PATH_TO_DSEC":
        opt.root_path = opt.dsec_path
    elif opt.kitti_stereo_path and opt.kitti_stereo_path != "PATH_TO_KITTI":
        opt.root_path = opt.kitti_stereo_path
    if not opt.root_path or not os.path.isdir(opt.root_path):
        raise FileNotFoundError(
            f"Stereo dataset root not found. Set tartanair_path/dsec_path/"
            f"kitti_stereo_path in configs/options_stereo.py. Got: {opt.root_path}")

    train_dataset = StereoVideoDataset(opt=opt, training=True, shuffle=True)
    test_dataset = StereoVideoDataset(opt=opt, training=False, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True,
        shuffle=True, drop_last=True, collate_fn=stereo_collate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=max(1, opt.batch_size // 2),
        num_workers=0, pin_memory=True, shuffle=False,
        collate_fn=stereo_collate)

    model = StereoSplatModel(opt)

    # Resume from a full StereoSplat checkpoint (Stage 2 or interrupted run).
    if opt.resume and os.path.exists(opt.resume):
        print(f"[resume] loading {opt.resume}")
        sd = load_file(opt.resume)
        sd = {k: v for k, v in sd.items()
              if k in model.state_dict()
              and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(sd, strict=False)

    # Load a Stage-1 encoder checkpoint to initialize the static branch.
    if opt.encoder_path and os.path.exists(opt.encoder_path):
        print(f"[encoder] loading {opt.encoder_path}")
        sd = load_file(opt.encoder_path)
        new_sd = OrderedDict()
        for k, v in sd.items():
            if "dynamic" in k:
                continue
            k = k.replace('_orig_mod.', '')
            k = k.replace('model.', 'model.gs_predictor.', 1) if k.startswith('model.') else k
            new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)

    # Freeze the static encoder in Stage 2.
    if opt.model_type == "StereoDecoder":
        if hasattr(model.model, "_freeze_predictor"):
            try:
                model.model._freeze_predictor()
            except Exception:
                pass

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train_loader) // opt.gradient_accumulation_steps)
    total_steps = opt.num_epochs * steps_per_epoch
    lr_decay_steps = opt.lr_decay_epochs * steps_per_epoch

    from utils.general_utils import CosineWarmupScheduler
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup_iters=opt.warmup_iters,
        max_iters=lr_decay_steps, min_lr=0.5 * opt.lr)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler)

    if accelerator.is_main_process:
        print(f"[model] trainable params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        run_name = f"stereosplat_{opt.model_type}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        wandb.init(project="stereosplat", name=run_name, config=vars(opt)
                   if hasattr(opt, '__dict__') else str(opt))
        writer = SummaryWriter(os.path.join(opt.workspace, 'tb'))
    else:
        writer = None

    opt.epoch = 0
    global_step = 0
    for epoch in range(opt.num_epochs):
        opt.epoch = epoch
        model.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                out = model(batch)
                loss, metrics = stereo_loss(
                    out, opt,
                    lpips_fn=getattr(model, 'lpips_loss', None),
                    epoch=epoch)
                accelerator.backward(loss)
                if opt.gradient_clip > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), opt.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if accelerator.is_main_process:
                    pbar.set_description(
                        f"ep{epoch} loss={metrics['loss'].item():.4f} "
                        f"psnr={metrics['psnr'].item():.2f} "
                        f"dL1={metrics['depth_l1'].item():.3f}")
                    if global_step % 50 == 0:
                        wandb.log({k: float(v.item()) for k, v in metrics.items()},
                                  step=global_step)
        # Periodic evaluation + checkpoint.
        if (epoch + 1) % 5 == 0 and accelerator.is_main_process:
            _evaluate(model, test_loader, opt, accelerator, writer, epoch)
            _save(model, opt, accelerator)
    if accelerator.is_main_process:
        _save(model, opt, accelerator)
        wandb.finish()


def _evaluate(model, loader, opt, accelerator, writer, epoch):
    model.eval()
    psnrs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            psnrs.append(out['render_left']['render'].mean().item())
    model.train()
    avg = sum(psnrs) / max(1, len(psnrs))
    print(f"[eval] epoch {epoch} avg-left-mean {avg:.4f}")
    if writer is not None:
        writer.add_scalar('eval/left_mean', avg, epoch)


def _save(model, opt, accelerator):
    import os
    os.makedirs(opt.workspace, exist_ok=True)
    path = os.path.join(opt.workspace, 'model.safetensors')
    sd = accelerator.unwrap_model(model).state_dict()
    save_file(sd, path)
    print(f"[save] -> {path}")


if __name__ == '__main__':
    main()
