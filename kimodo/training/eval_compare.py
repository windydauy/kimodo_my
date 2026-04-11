# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare official zero-shot vs finetuned checkpoints on G1 CSV clips.

This script runs text-to-motion generation for the same set of clips with:
- an official model checkpoint (zero-shot baseline)
- a finetuned checkpoint (loaded on top of the same official model config)

Then it computes:
1) full-body position error (meters)
2) end-effector position error (meters)
3) end-effector rotation error (degrees, geodesic)
4) 2D root position error on (x, z) (meters)

Example:
    python -m kimodo.training.eval_compare \
      --model-name g1 \
      --finetuned outputs/g1_finetune-phase1 \
      --csv-root datasets/g1/csv \
      --timelines-path datasets/SEED-Timeline-Annotations/timelines.jsonl \
      --output-json outputs/eval_g1_compare.json \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import torch
from omegaconf import OmegaConf

from kimodo.model.kimodo_model import Kimodo
from kimodo.model import load_model
from kimodo.model.loading import instantiate_from_dict
from kimodo.motion_rep.feature_utils import compute_heading_angle
from kimodo.tools import seed_everything
from kimodo.training.g1_csv import load_g1_csv_motion
from kimodo.training.timeline_annotations import TimelineAnnotationIndex


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare official vs finetuned Kimodo checkpoints on G1 CSV clips.")
    parser.add_argument("--model-name", type=str, default="g1", help="Model key/name accepted by kimodo.model.load_model.")
    parser.add_argument(
        "--official-config",
        type=str,
        default="",
        help=(
            "Optional local YAML for building the official baseline model directly "
            "(expects fields: model.num_base_steps, model.denoiser, text_encoder). "
            "When set, this bypasses load_model() default path resolution."
        ),
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        required=True,
        help=(
            "Path to finetuned weights. Supports: "
            "(1) step_*.pt checkpoint file, "
            "(2) ema_final.pt, "
            "(3) training output dir containing checkpoints/, "
            "(4) wandb run dir under outputs/<run>/wandb/run-*."
        ),
    )
    parser.add_argument("--csv-root", type=str, required=True, help="Root dir containing G1 CSV files.")
    parser.add_argument("--glob", type=str, default="*.csv", help="Glob pattern under csv-root (recursive).")
    parser.add_argument("--max-clips", type=int, default=0, help="Use first N clips after sorting; 0 means all.")
    parser.add_argument(
        "--clip-sampling",
        type=str,
        default="sorted",
        choices=["sorted", "random"],
        help="How to choose clips when --max-clips > 0.",
    )
    parser.add_argument(
        "--clip-sample-seed",
        type=int,
        default=42,
        help="Random seed for clip sampling when --clip-sampling=random.",
    )
    parser.add_argument(
        "--timelines-path",
        type=str,
        default="",
        help="Optional timelines.jsonl path. If omitted, prompt falls back to clip stem.",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default="overview",
        choices=["overview", "event_first", "mixed"],
        help="Prompt selection mode (deterministic).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic per-clip generation.")
    parser.add_argument("--num-denoising-steps", type=int, default=100, help="DDIM denoising steps.")
    parser.add_argument(
        "--num-transition-frames",
        type=int,
        default=5,
        help="Transition frames used in multi-prompt generation (same default as kimodo_gen).",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable post-processing for non-G1 models (G1 is already forced off by kimodo_gen).",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        nargs=2,
        default=[2.0, 2.0],
        metavar=("TEXT_CFG", "MOTION_CFG"),
        help="Classifier-free guidance weights [text, motion].",
    )
    parser.add_argument("--cfg-type", type=str, default="separated", help="CFG type.")
    parser.add_argument(
        "--eval-max-frames",
        type=int,
        default=300,
        help="Max frames per clip for evaluation after downsample; 0 means full length.",
    )
    parser.add_argument("--input-fps", type=int, default=120, help="CSV source FPS.")
    parser.add_argument("--source-coord-system", type=str, default="mujoco", choices=["mujoco", "kimodo"])
    parser.add_argument("--root-euler-order", type=str, default="xyz")
    parser.add_argument("--root-angle-unit", type=str, default="degrees", choices=["degrees", "radians", "auto"])
    parser.add_argument("--joint-angle-unit", type=str, default="degrees", choices=["degrees", "radians", "auto"])
    parser.add_argument(
        "--root-position-unit",
        type=str,
        default="centimeters",
        choices=["auto", "meters", "centimeters"],
    )
    parser.add_argument("--root-position-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON path for per-clip metrics and summary.",
    )
    return parser.parse_args()


def _safe_mean_std(values: Sequence[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def _resolve_finetuned_ckpt(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Finetuned path does not exist: {path}")

    candidates: List[Path] = []

    # case A: output dir
    candidates.extend(sorted((path / "checkpoints").glob("step_*.pt")))
    if (path / "ema_final.pt").is_file():
        candidates.append(path / "ema_final.pt")

    # case B: wandb run dir (outputs/<name>/wandb/run-xxx)
    parents = list(path.parents)
    for p in [path] + parents[:3]:
        candidates.extend(sorted((p / "checkpoints").glob("step_*.pt")))
        if (p / "ema_final.pt").is_file():
            candidates.append(p / "ema_final.pt")

    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve finetuned checkpoint from directory: {path}. "
            "Expected step_*.pt or ema_final.pt."
        )

    # Prefer latest step checkpoint if available; else ema_final.
    step_ckpts = [c for c in candidates if c.name.startswith("step_") and c.suffix == ".pt"]
    if step_ckpts:
        return sorted(step_ckpts)[-1]
    return sorted(candidates)[-1]


def _load_finetuned_weights(model, ckpt_path: Path, use_ema_if_available: bool = True) -> str:
    """Load finetuned weights into a Kimodo model instance.

    Returns:
        str: loading mode: "denoiser", "ema_shadow"
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    denoiser = model.denoiser.model  # unwrap CFG wrapper

    if isinstance(ckpt, dict) and "denoiser" in ckpt and (not use_ema_if_available or "ema" not in ckpt):
        denoiser.load_state_dict(ckpt["denoiser"], strict=True)
        return "denoiser"

    if isinstance(ckpt, dict) and "ema" in ckpt and use_ema_if_available:
        ema_state = ckpt["ema"]
        if not isinstance(ema_state, dict) or "shadow" not in ema_state:
            raise ValueError(f"Invalid EMA checkpoint format: {ckpt_path}")
        shadow = ema_state["shadow"]
        named_params = dict(denoiser.named_parameters())
        missing = []
        loaded = 0
        for name, param in named_params.items():
            # NOTE: evaluation models are often frozen (requires_grad=False),
            # but EMA shadow must still be copied.
            if name not in shadow:
                missing.append(name)
                continue
            tensor = shadow[name].to(device=param.device, dtype=param.dtype)
            param.data.copy_(tensor)
            loaded += 1
        log.info("Loaded EMA shadow params: %d", loaded)
        if missing:
            log.warning("EMA shadow missing %d trainable params. Example: %s", len(missing), missing[0])
        return "ema_shadow"

    if isinstance(ckpt, dict) and "denoiser" in ckpt:
        denoiser.load_state_dict(ckpt["denoiser"], strict=True)
        return "denoiser"

    raise ValueError(
        f"Unsupported checkpoint format for finetuned weights: {ckpt_path}. "
        "Expected keys like 'denoiser' and/or 'ema'."
    )


def _build_prompt(
    clip_path: Path,
    timelines: TimelineAnnotationIndex | None,
    mode: str,
) -> str:
    if timelines is None:
        return clip_path.stem
    try:
        rec = timelines.get_record(clip_path)
    except KeyError:
        return clip_path.stem

    overview = str(rec["overview_description"]).strip()
    events = rec["events"]

    if mode == "overview":
        if overview:
            return overview
        if events:
            return str(events[0]["description"])
        return clip_path.stem

    if mode == "event_first":
        if events:
            return str(events[0]["description"])
        if overview:
            return overview
        return clip_path.stem

    # mixed: deterministic "overview + first event"
    if overview and events:
        return f"{overview} {events[0]['description']}"
    if overview:
        return overview
    if events:
        return str(events[0]["description"])
    return clip_path.stem


def _geodesic_deg(pred_r: torch.Tensor, gt_r: torch.Tensor) -> torch.Tensor:
    rel = pred_r @ gt_r.transpose(-1, -2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos)
    return torch.rad2deg(ang)


def _compute_metrics_for_clip(
    pred_pos: torch.Tensor,  # [T, J, 3]
    gt_pos: torch.Tensor,  # [T, J, 3]
    pred_root: torch.Tensor,  # [T, 3]
    gt_root: torch.Tensor,  # [T, 3]
    pred_global_rot: torch.Tensor,  # [T, J, 3, 3]
    gt_global_rot: torch.Tensor,  # [T, J, 3, 3]
    ee_pos_idx: List[int],
    ee_rot_idx: List[int],
) -> Dict[str, float]:
    full_body_pos = torch.linalg.norm(pred_pos - gt_pos, dim=-1).mean().item()
    ee_pos = torch.linalg.norm(pred_pos[:, ee_pos_idx] - gt_pos[:, ee_pos_idx], dim=-1).mean().item()
    ee_rot = _geodesic_deg(pred_global_rot[:, ee_rot_idx], gt_global_rot[:, ee_rot_idx]).mean().item()
    root_2d = torch.linalg.norm(pred_root[:, [0, 2]] - gt_root[:, [0, 2]], dim=-1).mean().item()
    return {
        "full_body_pos_m": float(full_body_pos),
        "end_effector_pos_m": float(ee_pos),
        "end_effector_rot_deg": float(ee_rot),
        "root_2d_pos_m": float(root_2d),
    }


def _yaw_rotation_matrix(angle: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build a Y-axis rotation matrix for a scalar yaw angle (radians)."""
    c = torch.cos(angle).to(dtype=dtype, device=device)
    s = torch.sin(angle).to(dtype=dtype, device=device)
    z = torch.zeros((), dtype=dtype, device=device)
    o = torch.ones((), dtype=dtype, device=device)
    return torch.stack(
        [
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ],
        dim=0,
    )


def _rotate_points_y(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate points [..., 3] around +Y by a scalar yaw angle (radians)."""
    c = torch.cos(angle).to(dtype=points.dtype, device=points.device)
    s = torch.sin(angle).to(dtype=points.dtype, device=points.device)
    out = points.clone()
    x = out[..., 0].clone()
    z = out[..., 2].clone()
    out[..., 0] = c * x + s * z
    out[..., 2] = -s * x + c * z
    return out


def _align_pred_to_gt_first_frame(
    *,
    pred_pos: torch.Tensor,  # [T, J, 3]
    pred_root: torch.Tensor,  # [T, 3]
    pred_global_rot: torch.Tensor,  # [T, J, 3, 3]
    gt_pos: torch.Tensor,  # [T, J, 3]
    gt_root: torch.Tensor,  # [T, 3]
    skeleton,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align prediction to GT using first-frame heading + first-frame root xz translation."""
    pred_heading0 = compute_heading_angle(pred_pos.unsqueeze(0), skeleton)[0, 0]
    gt_heading0 = compute_heading_angle(gt_pos.unsqueeze(0), skeleton)[0, 0]
    delta_heading = gt_heading0 - pred_heading0

    pred_pos_aligned = _rotate_points_y(pred_pos, delta_heading)
    pred_root_aligned = _rotate_points_y(pred_root, delta_heading)

    rot_align = _yaw_rotation_matrix(
        delta_heading,
        dtype=pred_global_rot.dtype,
        device=pred_global_rot.device,
    )
    pred_global_rot_aligned = rot_align.view(1, 1, 3, 3) @ pred_global_rot

    dx = gt_root[0, 0] - pred_root_aligned[0, 0]
    dz = gt_root[0, 2] - pred_root_aligned[0, 2]
    pred_root_aligned[..., 0] = pred_root_aligned[..., 0] + dx
    pred_root_aligned[..., 2] = pred_root_aligned[..., 2] + dz
    pred_pos_aligned[..., 0] = pred_pos_aligned[..., 0] + dx
    pred_pos_aligned[..., 2] = pred_pos_aligned[..., 2] + dz

    return pred_pos_aligned, pred_root_aligned, pred_global_rot_aligned


def _is_g1_model_instance(model: Kimodo) -> bool:
    skel_name = str(getattr(model.skeleton, "name", "")).lower()
    return "g1" in skel_name


@torch.no_grad()
def _generate_like_kimodo_gen(
    model: Kimodo,
    prompt: str,
    num_frames: int,
    *,
    first_heading_angle: torch.Tensor,
    num_denoising_steps: int,
    cfg_weight: list[float],
    cfg_type: str,
    num_transition_frames: int,
    no_postprocess: bool,
) -> Dict[str, torch.Tensor]:
    """Run generation with the same call style used by `kimodo_gen`.

    Key alignment points:
    - list-based prompt/frame inputs
    - multi_prompt=True
    - num_transition_frames default=5
    - post_processing disabled for G1
    """
    use_postprocess = False if _is_g1_model_instance(model) else (not no_postprocess)
    return model(
        [prompt],
        [int(num_frames)],
        constraint_lst=[],
        num_denoising_steps=int(num_denoising_steps),
        num_samples=1,
        first_heading_angle=first_heading_angle,
        multi_prompt=True,
        num_transition_frames=int(num_transition_frames),
        post_processing=bool(use_postprocess),
        return_numpy=False,
        cfg_weight=cfg_weight,
        cfg_type=cfg_type,
    )


def _collect_csv_paths(
    csv_root: str,
    pattern: str,
    max_clips: int,
    clip_sampling: str,
    clip_sample_seed: int,
) -> List[Path]:
    paths = sorted(Path(csv_root).rglob(pattern))
    if max_clips > 0:
        if clip_sampling == "random":
            rng = random.Random(int(clip_sample_seed))
            if max_clips >= len(paths):
                rng.shuffle(paths)
            else:
                paths = rng.sample(paths, k=max_clips)
                paths = sorted(paths)
        else:
            paths = paths[:max_clips]
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {csv_root!r} with pattern {pattern!r}.")
    return paths


def _build_model_from_local_config(
    config_path: Path,
    *,
    device: torch.device,
    cfg_type: str,
) -> Kimodo:
    cfg = OmegaConf.load(config_path)
    if "model" not in cfg or "denoiser" not in cfg.model or "text_encoder" not in cfg:
        raise ValueError(
            f"Config {config_path} must contain model.denoiser, model.num_base_steps and text_encoder sections."
        )

    denoiser_cfg = OmegaConf.to_container(cfg.model.denoiser, resolve=True)
    text_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True)

    denoiser = instantiate_from_dict(denoiser_cfg).to(device)
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False

    text_encoder = instantiate_from_dict(text_cfg)
    if hasattr(text_encoder, "to"):
        text_encoder = text_encoder.to(device)
    if isinstance(text_encoder, torch.nn.Module):
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
    elif hasattr(text_encoder, "eval"):
        text_encoder.eval()

    model = Kimodo(
        denoiser=denoiser,
        text_encoder=text_encoder,
        num_base_steps=int(cfg.model.num_base_steps),
        device=device,
        cfg_type=cfg_type,
    )
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device(args.device)
    csv_paths = _collect_csv_paths(
        args.csv_root,
        args.glob,
        args.max_clips,
        args.clip_sampling,
        args.clip_sample_seed,
    )
    timelines = TimelineAnnotationIndex.from_jsonl(args.timelines_path) if args.timelines_path else None

    if args.official_config:
        official_cfg_path = Path(args.official_config)
        if not official_cfg_path.is_file():
            raise FileNotFoundError(f"--official-config not found: {official_cfg_path}")
        official = _build_model_from_local_config(official_cfg_path, device=device, cfg_type=args.cfg_type)
        finetuned = _build_model_from_local_config(official_cfg_path, device=device, cfg_type=args.cfg_type)
        log.info("Built official/finetuned base models from local config: %s", official_cfg_path)
    else:
        official = load_model(args.model_name, device=device, eval_mode=True)
        finetuned = load_model(args.model_name, device=device, eval_mode=True)

    finetuned_ckpt = _resolve_finetuned_ckpt(args.finetuned)
    load_mode = _load_finetuned_weights(finetuned, finetuned_ckpt, use_ema_if_available=True)
    log.info("Loaded finetuned weights from %s (mode=%s)", finetuned_ckpt, load_mode)

    skeleton = official.motion_rep.skeleton
    target_fps = int(official.motion_rep.fps)
    if args.input_fps < target_fps or args.input_fps % target_fps != 0:
        raise ValueError(
            f"input-fps must be integer multiple of model fps. got input_fps={args.input_fps}, model fps={target_fps}"
        )
    stride = args.input_fps // target_fps

    loader_kwargs = {
        "source_coord_system": args.source_coord_system,
        "root_euler_order": args.root_euler_order,
        "root_angle_unit": args.root_angle_unit,
        "joint_angle_unit": args.joint_angle_unit,
        "root_position_unit": args.root_position_unit,
        "root_position_scale": args.root_position_scale,
    }

    ee_pos_names = (
        list(skeleton.left_foot_joint_names)
        + list(skeleton.right_foot_joint_names)
        + list(skeleton.left_hand_joint_names)
        + list(skeleton.right_hand_joint_names)
    )
    ee_rot_names = [
        skeleton.left_foot_joint_names[0],
        skeleton.right_foot_joint_names[0],
        skeleton.left_hand_joint_names[0],
        skeleton.right_hand_joint_names[0],
    ]
    ee_pos_idx = [skeleton.bone_order_names.index(name) for name in ee_pos_names]
    ee_rot_idx = [skeleton.bone_order_names.index(name) for name in ee_rot_names]

    per_clip: List[Dict] = []
    official_metrics: Dict[str, List[float]] = {
        "full_body_pos_m": [],
        "end_effector_pos_m": [],
        "end_effector_rot_deg": [],
        "root_2d_pos_m": [],
    }
    finetuned_metrics: Dict[str, List[float]] = {
        "full_body_pos_m": [],
        "end_effector_pos_m": [],
        "end_effector_rot_deg": [],
        "root_2d_pos_m": [],
    }

    for clip_idx, csv_path in enumerate(csv_paths):
        prompt = _build_prompt(csv_path, timelines, mode=args.text_mode)
        data = load_g1_csv_motion(csv_path, **loader_kwargs)
        gt_local = data["local_joint_rots"][::stride].to(device)
        gt_root = data["root_positions"][::stride].to(device)

        if args.eval_max_frames > 0:
            gt_local = gt_local[: args.eval_max_frames]
            gt_root = gt_root[: args.eval_max_frames]
        num_frames = int(gt_local.shape[0])
        if num_frames <= 0:
            log.warning("Skipping empty clip after processing: %s", csv_path)
            continue

        gt_global_rot, gt_pos, _ = skeleton.fk(gt_local, gt_root)
        gt_first_heading = compute_heading_angle(gt_pos.unsqueeze(0), skeleton)[:, 0]

        # Same seed per clip for both models (fair stochastic comparison)
        clip_seed = int(args.seed + clip_idx)

        seed_everything(clip_seed, deterministic=True)
        out_off = _generate_like_kimodo_gen(
            official,
            prompt,
            num_frames,
            first_heading_angle=gt_first_heading,
            num_denoising_steps=args.num_denoising_steps,
            cfg_weight=args.cfg_weight,
            cfg_type=args.cfg_type,
            num_transition_frames=args.num_transition_frames,
            no_postprocess=args.no_postprocess,
        )

        seed_everything(clip_seed, deterministic=True)
        out_ft = _generate_like_kimodo_gen(
            finetuned,
            prompt,
            num_frames,
            first_heading_angle=gt_first_heading,
            num_denoising_steps=args.num_denoising_steps,
            cfg_weight=args.cfg_weight,
            cfg_type=args.cfg_type,
            num_transition_frames=args.num_transition_frames,
            no_postprocess=args.no_postprocess,
        )

        off_pos = out_off["posed_joints"]
        off_root = out_off["root_positions"]
        off_global_rot = out_off["global_rot_mats"]
        ft_pos = out_ft["posed_joints"]
        ft_root = out_ft["root_positions"]
        ft_global_rot = out_ft["global_rot_mats"]

        if off_pos.ndim == 4:
            off_pos = off_pos.squeeze(0)
            off_root = off_root.squeeze(0)
            off_global_rot = off_global_rot.squeeze(0)
        if ft_pos.ndim == 4:
            ft_pos = ft_pos.squeeze(0)
            ft_root = ft_root.squeeze(0)
            ft_global_rot = ft_global_rot.squeeze(0)

        if off_pos.shape[1] != gt_pos.shape[1] or ft_pos.shape[1] != gt_pos.shape[1]:
            raise ValueError(
                f"Joint count mismatch on clip={csv_path.stem}: "
                f"gt={gt_pos.shape[1]}, official={off_pos.shape[1]}, finetuned={ft_pos.shape[1]}. "
                "Use a model/skeleton consistent with your dataset."
            )

        t_common = min(
            int(gt_pos.shape[0]),
            int(off_pos.shape[0]),
            int(ft_pos.shape[0]),
        )
        gt_pos_clip = gt_pos[:t_common]
        gt_root_clip = gt_root[:t_common]
        gt_global_rot_clip = gt_global_rot[:t_common]

        off_pos_eval, off_root_eval, off_global_rot_eval = _align_pred_to_gt_first_frame(
            pred_pos=off_pos[:t_common],
            pred_root=off_root[:t_common],
            pred_global_rot=off_global_rot[:t_common],
            gt_pos=gt_pos_clip,
            gt_root=gt_root_clip,
            skeleton=skeleton,
        )
        ft_pos_eval, ft_root_eval, ft_global_rot_eval = _align_pred_to_gt_first_frame(
            pred_pos=ft_pos[:t_common],
            pred_root=ft_root[:t_common],
            pred_global_rot=ft_global_rot[:t_common],
            gt_pos=gt_pos_clip,
            gt_root=gt_root_clip,
            skeleton=skeleton,
        )

        off_clip = _compute_metrics_for_clip(
            pred_pos=off_pos_eval,
            gt_pos=gt_pos_clip,
            pred_root=off_root_eval,
            gt_root=gt_root_clip,
            pred_global_rot=off_global_rot_eval,
            gt_global_rot=gt_global_rot_clip,
            ee_pos_idx=ee_pos_idx,
            ee_rot_idx=ee_rot_idx,
        )
        ft_clip = _compute_metrics_for_clip(
            pred_pos=ft_pos_eval,
            gt_pos=gt_pos_clip,
            pred_root=ft_root_eval,
            gt_root=gt_root_clip,
            pred_global_rot=ft_global_rot_eval,
            gt_global_rot=gt_global_rot_clip,
            ee_pos_idx=ee_pos_idx,
            ee_rot_idx=ee_rot_idx,
        )

        for k in official_metrics:
            official_metrics[k].append(off_clip[k])
            finetuned_metrics[k].append(ft_clip[k])

        clip_result = {
            "clip": csv_path.stem,
            "csv_path": str(csv_path),
            "num_frames_eval": t_common,
            "prompt": prompt,
            "official": off_clip,
            "finetuned": ft_clip,
            "delta_finetuned_minus_official": {k: float(ft_clip[k] - off_clip[k]) for k in off_clip},
        }
        per_clip.append(clip_result)
        log.info(
            "[%d/%d] %s | off_full=%.4fm ft_full=%.4fm off_ee=%.4fm ft_ee=%.4fm",
            clip_idx + 1,
            len(csv_paths),
            csv_path.stem,
            off_clip["full_body_pos_m"],
            ft_clip["full_body_pos_m"],
            off_clip["end_effector_pos_m"],
            ft_clip["end_effector_pos_m"],
        )

    summary = {
        "official": {k: _safe_mean_std(v) for k, v in official_metrics.items()},
        "finetuned": {k: _safe_mean_std(v) for k, v in finetuned_metrics.items()},
        "delta_finetuned_minus_official": {
            k: _safe_mean_std([f - o for f, o in zip(finetuned_metrics[k], official_metrics[k])])
            for k in official_metrics
        },
    }

    out = {
        "meta": {
            "model_name": args.model_name,
            "finetuned_checkpoint": str(finetuned_ckpt),
            "finetuned_load_mode": load_mode,
            "num_clips": len(per_clip),
            "num_denoising_steps": args.num_denoising_steps,
            "num_transition_frames": args.num_transition_frames,
            "cfg_weight": list(args.cfg_weight),
            "cfg_type": args.cfg_type,
            "postprocess_mode": "g1_forced_off_else_not_no_postprocess",
            "no_postprocess": bool(args.no_postprocess),
            "generation_call_style": "kimodo_gen_aligned_multi_prompt",
            "use_gt_first_heading": True,
            "metric_alignment": "first_frame_heading_plus_root_xz",
            "text_mode": args.text_mode,
            "input_fps": args.input_fps,
            "target_fps": target_fps,
            "downsample_stride": stride,
            "eval_max_frames": args.eval_max_frames,
            "ee_pos_joints": ee_pos_names,
            "ee_rot_joints": ee_rot_names,
        },
        "summary": summary,
        "per_clip": per_clip,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote comparison report to %s", output_json)


if __name__ == "__main__":
    main()
