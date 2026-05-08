# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-student distillation training for Kimodo G1.

Default target:
- Teacher: 16-layer denoiser, pretrained checkpoint
- Student: 8-layer denoiser
- Distillation objective: 7 (teacher) + 7 (GT), teacher-dominant weighting
- Step mapping: train on a reduced student timestep grid (e.g., 20 steps)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.distillation.loss import DistillationKimodoLoss
from kimodo.model.diffusion import Diffusion
from kimodo.model.loading import instantiate_from_dict
from kimodo.sanitize import sanitize_texts
from kimodo.tools import seed_everything
from kimodo.training.dataset import G1CSVTextDataset, LinearKeyframeScheduler, g1_text_collate_fn
from kimodo.training.ema import EMA
from kimodo.training.optimizers import build_optimizer

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill Kimodo G1 student model.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "distill_g1_100_to_20.yaml"),
        help="Path to distillation config yaml.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional student checkpoint path to resume from.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, local_rank, world_size, is_distributed


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def build_text_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


@torch.no_grad()
def encode_text_batch(
    text_encoder: Any,
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = sanitize_texts(texts)
    text_feat, text_lengths = text_encoder(texts)
    text_feat = torch.as_tensor(text_feat, device=device)
    if text_feat.ndim == 2:
        text_feat = text_feat[:, None]

    text_lengths_t = torch.as_tensor(text_lengths, dtype=torch.long, device=device)
    text_pad_mask = build_text_pad_mask(text_lengths_t, text_feat.shape[1])

    empty_text_mask = torch.tensor([len(t.strip()) == 0 for t in texts], dtype=torch.bool, device=device)
    if empty_text_mask.any():
        text_feat[empty_text_mask] = 0
        text_pad_mask[empty_text_mask] = False
    return text_feat, text_pad_mask


def apply_cfg_dropout(
    text_feat: torch.Tensor,
    text_pad_mask: torch.Tensor,
    observed_motion: Optional[torch.Tensor],
    motion_mask: Optional[torch.Tensor],
    text_dropout_prob: float,
    motion_dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    batch_size = text_feat.shape[0]
    device = text_feat.device

    if text_dropout_prob > 0.0:
        drop_text = torch.rand(batch_size, device=device) < text_dropout_prob
        if drop_text.any():
            text_feat = text_feat.clone()
            text_pad_mask = text_pad_mask.clone()
            text_feat[drop_text] = 0
            text_pad_mask[drop_text] = False

    if observed_motion is not None and motion_mask is not None and motion_dropout_prob > 0.0:
        drop_motion = torch.rand(batch_size, device=device) < motion_dropout_prob
        if drop_motion.any():
            observed_motion = observed_motion.clone()
            motion_mask = motion_mask.clone()
            observed_motion[drop_motion] = 0
            motion_mask[drop_motion] = 0

    return text_feat, text_pad_mask, observed_motion, motion_mask


def maybe_to_device(x: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.to(device)


@torch.no_grad()
def compute_first_heading_angle(
    x0: torch.Tensor,
    motion_rep: Any,
    input_is_normalized: bool,
) -> torch.Tensor:
    features = motion_rep.unnormalize(x0) if input_is_normalized else x0
    return motion_rep.get_root_heading_angle(features)[:, 0]


def make_autocast(mixed_precision: str, device: torch.device) -> tuple[Any, bool]:
    mp = str(mixed_precision).lower()
    if device.type != "cuda" or mp not in {"fp16", "bf16"}:
        return nullcontext, False

    dtype = torch.float16 if mp == "fp16" else torch.bfloat16
    use_scaler = mp == "fp16"

    def _ctx():
        return torch.autocast(device_type="cuda", dtype=dtype)

    return _ctx, use_scaler


def reduce_scalar(value: torch.Tensor, world_size: int, is_distributed: bool) -> torch.Tensor:
    if not is_distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= float(world_size)
    return reduced


def load_text_encoder(cfg: DictConfig, device: torch.device) -> Any:
    text_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True)
    text_encoder = instantiate(OmegaConf.create(text_cfg, flags={"allow_objects": True}))
    if hasattr(text_encoder, "to"):
        text_encoder = text_encoder.to(device)
    if isinstance(text_encoder, nn.Module):
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
    elif hasattr(text_encoder, "eval"):
        text_encoder.eval()

    # Keep behavior aligned with kimodo.training.train
    llm2vec_model = getattr(text_encoder, "model", None)
    if isinstance(llm2vec_model, nn.Module) and not getattr(llm2vec_model, "_kimodo_eval_noop_patched", False):
        # LLM2Vec.encode() calls self.eval() every batch. In some torch/transformers
        # environments this recursive train(False) path can throw unexpectedly.
        # The encoder is already frozen and set to eval above, so a no-op eval is safe.
        def _eval_noop():
            return llm2vec_model

        llm2vec_model.eval = _eval_noop
        llm2vec_model._kimodo_eval_noop_patched = True

    encode_fn = getattr(llm2vec_model, "encode", None)
    if callable(encode_fn) and not getattr(llm2vec_model, "_kimodo_encode_numpy_patched", False):
        def _encode_numpy(*args, **kwargs):
            out = encode_fn(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy()
            return out

        llm2vec_model.encode = _encode_numpy
        llm2vec_model._kimodo_encode_numpy_patched = True

    return text_encoder


def build_dataset_motion_rep(cfg: DictConfig) -> Any:
    motion_rep_cfg = OmegaConf.to_container(cfg.model.student_denoiser.motion_rep, resolve=True)
    motion_rep = instantiate_from_dict(motion_rep_cfg)
    if hasattr(motion_rep, "to"):
        motion_rep = motion_rep.to(torch.device("cpu"))
    if isinstance(motion_rep, nn.Module):
        motion_rep.eval()
        for p in motion_rep.parameters():
            p.requires_grad = False
    elif hasattr(motion_rep, "eval"):
        motion_rep.eval()
    return motion_rep


def build_dataset(cfg: DictConfig, motion_rep: Any, rank: int) -> G1CSVTextDataset:
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_cfg["motion_rep"] = motion_rep
    if "_target_" in dataset_cfg:
        dataset = instantiate(OmegaConf.create(dataset_cfg, flags={"allow_objects": True}))
    else:
        dataset = G1CSVTextDataset(**dataset_cfg)
    if is_main_process(rank):
        log.info("Dataset size: %d clips", len(dataset))
    return dataset


def create_dataloader(
    dataset: G1CSVTextDataset,
    cfg: DictConfig,
    is_distributed: bool,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    num_workers = int(cfg.data.num_workers)
    scheduler_enabled = cfg.get("phase2", None) is not None and bool(cfg.phase2.get("enabled", False))
    if scheduler_enabled and num_workers > 0:
        # Keyframe scheduling mutates dataset state every step; workers would hold
        # out-of-sync dataset copies, so force single-process loading.
        log.warning("phase2 keyframe scheduling requires num_workers=0; overriding from %d to 0", num_workers)
        num_workers = 0

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=g1_text_collate_fn,
    )
    return loader, sampler


def create_keyframe_scheduler(cfg: DictConfig) -> Optional[LinearKeyframeScheduler]:
    phase2_cfg = cfg.get("phase2", None)
    if phase2_cfg is None or not bool(phase2_cfg.get("enabled", False)):
        return None
    return LinearKeyframeScheduler(
        start_step=int(phase2_cfg.start_step),
        end_step=int(phase2_cfg.end_step),
        start_keyframes=int(phase2_cfg.start_keyframes),
        end_keyframes=int(phase2_cfg.end_keyframes),
    )


def should_use_phase2(step: int, cfg: DictConfig) -> bool:
    phase2_cfg = cfg.get("phase2", None)
    if phase2_cfg is None or not bool(phase2_cfg.get("enabled", False)):
        return False
    return int(step) >= int(phase2_cfg.start_step)


def maybe_init_wandb(cfg: DictConfig, rank: int, output_dir: Path):
    """Initialize wandb run on main process if enabled in config."""
    wandb_cfg = cfg.get("wandb", None)
    if not is_main_process(rank) or wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is enabled in config, but package 'wandb' is not installed. "
            "Install it with `pip install wandb` or disable cfg.wandb.enabled."
        ) from exc

    tags_cfg = wandb_cfg.get("tags", [])
    tags = list(tags_cfg) if tags_cfg is not None else None

    run = wandb.init(
        project=str(wandb_cfg.get("project", "kimodo-g1-distill")),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("name", None),
        group=wandb_cfg.get("group", None),
        job_type=wandb_cfg.get("job_type", "train"),
        tags=tags,
        mode=str(wandb_cfg.get("mode", "online")),
        resume=wandb_cfg.get("resume", None),
        id=wandb_cfg.get("id", None),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def get_batch(
    iterator: Iterator[Dict[str, Any]],
    loader: DataLoader,
    sampler: Optional[DistributedSampler],
    epoch: int,
    is_distributed: bool,
) -> tuple[Dict[str, Any], Iterator[Dict[str, Any]], int]:
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def make_student_timestep_grid(num_base_steps: int, student_steps: int) -> torch.Tensor:
    if student_steps <= 0:
        raise ValueError(f"student_steps must be > 0, got {student_steps}.")
    grid = torch.linspace(float(num_base_steps - 1), 0.0, steps=student_steps)
    return torch.round(grid).to(torch.long)


def copy_matching_params(dst_model: nn.Module, src_model: nn.Module) -> tuple[int, int]:
    """Copy same-name same-shape params from src to dst.

    This enables a strong warm start even when student has fewer layers.
    """
    dst_state = dst_model.state_dict()
    src_state = src_model.state_dict()
    copied = 0
    for k, v in dst_state.items():
        src_v = src_state.get(k, None)
        if src_v is not None and src_v.shape == v.shape:
            dst_state[k] = src_v.detach().clone()
            copied += 1
    dst_model.load_state_dict(dst_state, strict=True)
    return copied, len(dst_state)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    epoch: int,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: EMA,
    cfg: DictConfig,
) -> None:
    state = {
        "step": int(step),
        "epoch": int(epoch),
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def cleanup_old_checkpoints(ckpt_dir: Path, max_keep: int) -> None:
    if max_keep <= 0:
        return
    paths = sorted(ckpt_dir.glob("step_*.pt"))
    if len(paths) <= max_keep:
        return
    for p in paths[:-max_keep]:
        p.unlink(missing_ok=True)


def load_resume_checkpoint(
    ckpt_path: Path,
    *,
    student: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[EMA],
    device: Optional[torch.device],
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["student"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"], device=device)
    return ckpt


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = build_device(local_rank)

    logging.basicConfig(
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = OmegaConf.load(args.config)
    os.environ.setdefault("KIMODO_DISABLE_ROOT_SMOOTH", "1")
    seed_everything(int(cfg.training.seed) + rank, deterministic=bool(cfg.training.deterministic))

    output_dir = Path(cfg.training.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_path = output_dir / "train_log.jsonl"
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    wandb_run = maybe_init_wandb(cfg, rank=rank, output_dir=output_dir)

    teacher_cfg = OmegaConf.to_container(cfg.model.teacher_denoiser, resolve=True)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)

    teacher = instantiate_from_dict(teacher_cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = instantiate_from_dict(student_cfg).to(device)

    if bool(cfg.distillation.get("student_init_from_teacher", True)):
        copied, total = copy_matching_params(student, teacher)
        if is_main_process(rank):
            log.info("Warm-started student from teacher: copied %d/%d params", copied, total)

    diffusion = Diffusion(num_base_steps=int(cfg.model.num_base_steps)).to(device)
    motion_rep = student.motion_rep

    dataset_motion_rep = build_dataset_motion_rep(cfg)
    text_encoder = load_text_encoder(cfg, device)
    dataset = build_dataset(cfg, motion_rep=dataset_motion_rep, rank=rank)
    scheduler = create_keyframe_scheduler(cfg)
    loader, sampler = create_dataloader(dataset, cfg, is_distributed=is_distributed)

    optimizer = build_optimizer(cfg, student)
    autocast_ctx, use_scaler = make_autocast(cfg.training.mixed_precision, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    loss_fn = DistillationKimodoLoss(
        motion_rep=motion_rep,
        teacher_gammas=cfg.loss.teacher_gammas,
        gt_gammas=cfg.loss.gt_gammas,
        teacher_weight=float(cfg.loss.teacher_weight),
        gt_weight=float(cfg.loss.gt_weight),
        input_is_normalized=bool(cfg.data.dataset.to_normalize),
    )
    ema = EMA(student, decay=float(cfg.training.ema_decay))

    train_model: nn.Module = student
    if is_distributed:
        train_model = DDP(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    total_steps = int(cfg.training.total_steps)
    log_every = int(cfg.training.log_every)
    save_every = int(cfg.training.save_every)
    max_ckpt_keep = int(cfg.training.max_checkpoints_to_keep)
    grad_clip = float(cfg.training.gradient_clip)
    cfg_text_drop = float(cfg.training.cfg_dropout_text_prob)
    cfg_motion_drop = float(cfg.training.cfg_dropout_motion_prob)
    skip_text_encode_errors = bool(cfg.training.get("skip_text_encode_errors", False))
    max_text_encode_retries = int(cfg.training.get("max_text_encode_retries", 8))
    student_steps = int(cfg.distillation.student_steps)
    student_t_grid = make_student_timestep_grid(diffusion.num_base_steps, student_steps).to(device)

    start_step = 0
    epoch = 0
    resume_path = args.resume or cfg.training.get("resume", None)
    if resume_path:
        ckpt = load_resume_checkpoint(
            Path(resume_path),
            student=student,
            optimizer=optimizer,
            scaler=scaler if use_scaler else None,
            ema=ema,
            device=device,
        )
        start_step = int(ckpt.get("step", -1)) + 1
        epoch = int(ckpt.get("epoch", 0))
        if is_main_process(rank):
            log.info("Resumed from %s (start_step=%d, epoch=%d)", resume_path, start_step, epoch)

    iterator = iter(loader)
    if is_distributed and sampler is not None:
        sampler.set_epoch(epoch)

    student.train(True)

    if is_main_process(rank):
        log.info(
            "KIMODO_DISABLE_ROOT_SMOOTH=%s",
            os.environ.get("KIMODO_DISABLE_ROOT_SMOOTH", "0"),
        )
        if scheduler is None:
            log.info(
                "Starting distillation on %s | world_size=%d | total_steps=%d | student_steps=%d",
                device,
                world_size,
                total_steps,
                student_steps,
            )
        else:
            phase2_cfg = cfg.phase2
            log.info(
                "Starting distillation on %s | world_size=%d | total_steps=%d | student_steps=%d "
                "| keyframes schedule: step[%d,%d] %d->%d",
                device,
                world_size,
                total_steps,
                student_steps,
                int(phase2_cfg.start_step),
                int(phase2_cfg.end_step),
                int(phase2_cfg.start_keyframes),
                int(phase2_cfg.end_keyframes),
            )

    for global_step in range(start_step, total_steps):
        if scheduler is not None:
            if should_use_phase2(global_step, cfg):
                dataset.set_phase2_num_keyframes(int(scheduler(global_step)))
            else:
                dataset.set_phase2_num_keyframes(0)
        text_encode_exc: Optional[Exception] = None
        for encode_attempt in range(1, max_text_encode_retries + 2):
            batch, iterator, epoch = get_batch(iterator, loader, sampler, epoch, is_distributed)

            gt_x0 = batch["motion"].to(device=device, dtype=torch.float32)
            pad_mask = batch["pad_mask"].to(device=device, dtype=torch.bool)

            observed_motion = maybe_to_device(batch["observed_motion"], device)
            motion_mask = maybe_to_device(batch["motion_mask"], device)
            if observed_motion is not None:
                observed_motion = observed_motion.to(dtype=gt_x0.dtype)
            if motion_mask is not None:
                motion_mask = motion_mask.to(dtype=gt_x0.dtype)

            try:
                text_feat, text_pad_mask = encode_text_batch(text_encoder, batch["text"], device=device)
            except Exception as exc:  # noqa: BLE001 - robustness fallback for external encoder instability
                text_encode_exc = exc
                if (not skip_text_encode_errors) or (encode_attempt > max_text_encode_retries + 0):
                    raise
                if is_main_process(rank):
                    log.warning(
                        "Skipping batch at step=%d due to text encoder error (%s: %s), retry %d/%d.",
                        global_step,
                        type(exc).__name__,
                        exc,
                        encode_attempt,
                        max_text_encode_retries + 1,
                    )
                continue

            text_feat, text_pad_mask, observed_motion, motion_mask = apply_cfg_dropout(
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                observed_motion=observed_motion,
                motion_mask=motion_mask,
                text_dropout_prob=cfg_text_drop,
                motion_dropout_prob=cfg_motion_drop,
            )
            text_encode_exc = None
            break

        if text_encode_exc is not None:
            raise RuntimeError(
                f"Text encoding failed for step={global_step} after {max_text_encode_retries + 1} attempts."
            ) from text_encode_exc

        bsz = gt_x0.shape[0]
        t_idx = torch.randint(0, student_t_grid.shape[0], (bsz,), device=device, dtype=torch.long)
        timesteps = student_t_grid[t_idx]
        noise = torch.randn_like(gt_x0)
        x_t = diffusion.q_sample(gt_x0, timesteps, noise)

        first_heading_angle = None
        if bool(cfg.model.student_denoiser.get("input_first_heading_angle", False)):
            first_heading_angle = compute_first_heading_angle(
                x0=gt_x0,
                motion_rep=motion_rep,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )

        with torch.no_grad():
            with autocast_ctx():
                teacher_x0 = teacher(
                    x=x_t,
                    x_pad_mask=pad_mask,
                    text_feat=text_feat,
                    text_feat_pad_mask=text_pad_mask,
                    timesteps=timesteps,
                    first_heading_angle=first_heading_angle,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                )

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            student_x0 = train_model(
                x=x_t,
                x_pad_mask=pad_mask,
                text_feat=text_feat,
                text_feat_pad_mask=text_pad_mask,
                timesteps=timesteps,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )
            loss_dict = loss_fn(
                pred_x0=student_x0,
                teacher_x0=teacher_x0,
                gt_x0=gt_x0,
                pad_mask=pad_mask,
            )
            loss = loss_dict["total"]

        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

        ema_every = int(cfg.training.get("ema_every", 1))
        if (global_step + 1) % ema_every == 0:
            ema.update(student)

        if (global_step + 1) % log_every == 0 or global_step == 0:
            reduced_total = reduce_scalar(loss.detach(), world_size, is_distributed)
            reduced_teacher_total = reduce_scalar(loss_dict["loss_teacher_total"].detach(), world_size, is_distributed)
            reduced_gt_total = reduce_scalar(loss_dict["loss_gt_total"].detach(), world_size, is_distributed)

            if is_main_process(rank):
                metric = {
                    "step": global_step,
                    "epoch": epoch,
                    "num_keyframes": int(dataset.get_phase2_num_keyframes()),
                    "loss_total": float(reduced_total.item()),
                    "loss_teacher_total": float(reduced_teacher_total.item()),
                    "loss_gt_total": float(reduced_gt_total.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "time": time.time(),
                }
                log.info(
                    "step=%d kf=%d loss=%.6f teacher=%.6f gt=%.6f lr=%.3e",
                    metric["step"],
                    metric["num_keyframes"],
                    metric["loss_total"],
                    metric["loss_teacher_total"],
                    metric["loss_gt_total"],
                    metric["lr"],
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metric, ensure_ascii=False) + "\n")
                if wandb_run is not None:
                    wandb_run.log(metric, step=global_step)

        if is_main_process(rank) and (((global_step + 1) % save_every == 0) or (global_step + 1 == total_steps)):
            ckpt_path = ckpt_dir / f"step_{global_step + 1:08d}.pt"
            save_checkpoint(
                ckpt_path,
                step=global_step,
                epoch=epoch,
                student=student,
                optimizer=optimizer,
                scaler=scaler if use_scaler else None,
                ema=ema,
                cfg=cfg,
            )
            cleanup_old_checkpoints(ckpt_dir, max_keep=max_ckpt_keep)
            log.info("Saved checkpoint to %s", ckpt_path)

    if is_main_process(rank):
        ema_path = output_dir / "ema_final.pt"
        torch.save({"ema": ema.state_dict(), "step": total_steps - 1}, ema_path)
        log.info("Saved final EMA to %s", ema_path)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
