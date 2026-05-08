#!/usr/bin/env python3
"""Benchmark Kimodo G1 generation latency for prompt + constraints inference only.

Timing scope:
- Included: the `model(...)` generation call.
- Excluded: model loading, prompt extraction, constraint-file building/loading, heading extraction.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo import DEFAULT_MODEL, load_model
from kimodo.constraints import load_constraints_lst
from kimodo.model.cfg import CFG_TYPES
from kimodo.model.kimodo_model import Kimodo
from kimodo.model.loading import instantiate_from_dict
from kimodo.model.registry import get_model_info
from kimodo.tools import seed_everything
from scipy.spatial.transform import Rotation as R


def _noop_progress(it: Iterable):
    return it


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark G1 generation latency (model(...) only).",
    )
    parser.add_argument("--prompt", required=True, type=str, help="Generation text prompt.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name, e.g. Kimodo-G1-RP-v1.",
    )
    parser.add_argument("--duration", type=float, required=True, help="Target duration in seconds.")
    parser.add_argument(
        "--future_frames",
        type=int,
        default=0,
        help="If >0, benchmark only the next N generated frames (overrides frame count from duration/constraints).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="Sliding-window size in frames. If >0, run multi-window replanning benchmark.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Sliding-window stride in frames. Defaults to window_size when window_size > 0.",
    )
    parser.add_argument(
        "--window_start",
        type=int,
        default=0,
        help="Start frame index (in model FPS timeline) for the first window.",
    )
    parser.add_argument(
        "--num_windows",
        type=int,
        default=0,
        help="Max number of windows to benchmark. 0 means run all valid windows.",
    )
    parser.add_argument("--constraints", type=str, required=True, help="Path to constraints JSON.")
    parser.add_argument(
        "--heading_source_npz",
        type=str,
        required=True,
        help="NPZ path used to compute first-frame heading.",
    )
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of denoising steps.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs (not counted).")
    parser.add_argument("--repeats", type=int, default=30, help="Measured runs.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    parser.add_argument(
        "--cfg_type",
        type=str,
        default="separated",
        choices=CFG_TYPES,
        help="Classifier-free guidance type.",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        nargs="*",
        default=[2.0, 2.0],
        help="CFG scale(s). For separated, pass 2 values [text, constraint].",
    )
    parser.add_argument(
        "--hard_project_observed_motion",
        action="store_true",
        help="Enable hard projection during denoising (default: disabled, soft constraints only).",
    )
    parser.add_argument(
        "--hard_project_prefix_frames",
        type=int,
        default=0,
        help="Hard-project only first K frames when hard projection is enabled.",
    )
    parser.add_argument(
        "--hard_project_release_frames",
        type=int,
        default=0,
        help="Release window after hard-projected prefix.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save benchmark metrics as JSON.",
    )
    parser.add_argument(
        "--distill_config",
        type=str,
        default=None,
        help="Optional distillation YAML. If provided with --distill_ckpt, benchmark student architecture directly.",
    )
    parser.add_argument(
        "--distill_ckpt",
        type=str,
        default=None,
        help="Optional distillation checkpoint path (.pt). Supports either {'student': ...} or {'ema': {'shadow': ...}}.",
    )
    return parser.parse_args()


def _resolve_cfg_kwargs(args: argparse.Namespace) -> dict:
    if args.cfg_type == "nocfg":
        return {"cfg_type": "nocfg"}
    if args.cfg_type == "regular":
        if len(args.cfg_weight) != 1:
            raise ValueError("--cfg_type regular requires exactly one --cfg_weight value.")
        return {"cfg_type": "regular", "cfg_weight": float(args.cfg_weight[0])}
    if len(args.cfg_weight) != 2:
        raise ValueError("--cfg_type separated requires exactly two --cfg_weight values.")
    return {"cfg_type": "separated", "cfg_weight": [float(args.cfg_weight[0]), float(args.cfg_weight[1])]}


def resolve_num_frames(duration_sec: float, fps: float, constraints_path: str | None) -> tuple[int, int | None]:
    num_frames = int(float(duration_sec) * float(fps))
    max_constraint_frame = None
    if constraints_path:
        with open(constraints_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        all_indices: list[int] = []
        for item in items:
            all_indices.extend(int(x) for x in item.get("frame_indices", []))
        if all_indices:
            max_constraint_frame = max(all_indices)
            num_frames = max(num_frames, max_constraint_frame + 1)
    return num_frames, max_constraint_frame


def trim_constraints_to_horizon(constraints: list, horizon_frames: int, start_frame: int = 0) -> list:
    """Crop constraints to [start_frame, start_frame + horizon_frames) and shift to window-local time."""
    if horizon_frames <= 0:
        return constraints

    trimmed = []
    end_frame = start_frame + horizon_frames
    for constraint in constraints:
        if not hasattr(constraint, "crop_move"):
            trimmed.append(constraint)
            continue
        c = constraint.crop_move(start_frame, end_frame)
        if hasattr(c, "frame_indices") and len(c.frame_indices) == 0:
            continue
        trimmed.append(c)
    return trimmed


def build_window_starts(
    total_frames: int,
    window_size: int,
    window_stride: int,
    window_start: int,
    num_windows: int,
) -> list[int]:
    if window_size <= 0:
        return [0]
    if window_stride <= 0:
        window_stride = window_size
    if window_start < 0:
        window_start = 0

    max_start = max(0, total_frames - window_size)
    starts: list[int] = []
    cur = window_start
    while cur <= max_start:
        starts.append(cur)
        cur += window_stride
    if not starts:
        starts = [0]
    if num_windows > 0:
        starts = starts[:num_windows]
    return starts


def extract_heading_angle_from_npz_at_frame(
    npz_path: str | os.PathLike[str],
    frame_idx_target_fps: int,
    target_fps: float,
) -> float:
    data = np.load(str(npz_path), allow_pickle=False)
    src_fps = float(data["fps"]) if "fps" in data.files else float(target_fps)
    src_idx = int(round(float(frame_idx_target_fps) * src_fps / float(target_fps)))

    if "qpos" in data.files:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        src_idx = max(0, min(src_idx, qpos.shape[0] - 1))
        if qpos.shape[1] < 7:
            raise ValueError(f"Expected qpos with at least 7 columns, got {tuple(qpos.shape)}.")
        root_rot_m = R.from_quat(qpos[src_idx, 3:7], scalar_first=True).as_matrix()
        forward = (np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]) @ root_rot_m @ np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ).T) @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return float(np.arctan2(forward[0], forward[2]))

    if "root_global_6d" in data.files:
        root_global_6d = np.asarray(data["root_global_6d"], dtype=np.float64)
        if root_global_6d.ndim != 2 or root_global_6d.shape[1] < 6:
            raise ValueError(f"Expected root_global_6d shape [T, 6], got {tuple(root_global_6d.shape)}.")
        src_idx = max(0, min(src_idx, root_global_6d.shape[0] - 1))
        root_rot_m = R.from_euler("xyz", root_global_6d[src_idx, 3:6]).as_matrix()
        forward = (np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]) @ root_rot_m @ np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ).T) @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return float(np.arctan2(forward[0], forward[2]))

    raise ValueError(f"NPZ {npz_path} does not contain qpos or root_global_6d for heading extraction.")


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(np.ceil(q * len(sorted_vals))) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def _load_distill_weights_into_student(denoiser: torch.nn.Module, ckpt_path: str) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported distill checkpoint format: {ckpt_path}")

    if "student" in ckpt:
        denoiser.load_state_dict(ckpt["student"], strict=True)
        return "student"

    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        shadow = ckpt["ema"]["shadow"]
        if not isinstance(shadow, dict):
            raise ValueError(f"Invalid ema.shadow format in checkpoint: {ckpt_path}")
        named_params = dict(denoiser.named_parameters())
        missing = []
        for name, param in named_params.items():
            if not param.requires_grad:
                continue
            if name not in shadow:
                missing.append(name)
                continue
            param.data.copy_(shadow[name].to(device=param.device, dtype=param.dtype))
        if missing:
            raise ValueError(
                f"EMA shadow missing {len(missing)} trainable params (first 10: {missing[:10]}) in {ckpt_path}"
            )
        return "ema_shadow"

    # Last fallback: try plain state dict strict load.
    denoiser.load_state_dict(ckpt, strict=True)
    return "state_dict"


def _build_model_from_distill(distill_config: str, distill_ckpt: str, device: str) -> Kimodo:
    cfg = OmegaConf.load(distill_config)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)
    text_encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True)

    denoiser = instantiate_from_dict(student_cfg).to(device)
    text_encoder = instantiate_from_dict(text_encoder_cfg)
    text_encoder.to(device)
    text_encoder.eval()
    denoiser.eval()

    mode = _load_distill_weights_into_student(denoiser, distill_ckpt)
    print(f"Loaded distill checkpoint: {distill_ckpt} (mode={mode})")

    num_base_steps = int(cfg.model.num_base_steps)
    model = Kimodo(
        denoiser=denoiser,
        text_encoder=text_encoder,
        num_base_steps=num_base_steps,
        device=device,
        cfg_type="separated",
    )
    return model


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be >= 0 and --repeats must be > 0.")

    if args.seed is not None:
        seed_everything(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if bool(args.distill_config) != bool(args.distill_ckpt):
        raise ValueError("--distill_config and --distill_ckpt must be provided together.")

    if args.distill_config and args.distill_ckpt:
        model = _build_model_from_distill(args.distill_config, args.distill_ckpt, device)
        resolved_model = "distill-student"
        print(f"Loaded distill student model from config={args.distill_config}")
    else:
        model, resolved_model = load_model(
            args.model,
            device=device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        info = get_model_info(resolved_model)
        display = info.display_name if info else resolved_model
        print(f"Loaded model: {display} ({resolved_model})")

    base_constraints = load_constraints_lst(args.constraints, model.skeleton)

    total_frames, max_constraint_frame = resolve_num_frames(args.duration, model.fps, args.constraints)
    eval_frames = total_frames
    if args.window_size > 0:
        eval_frames = int(args.window_size)
    elif args.future_frames > 0:
        eval_frames = int(args.future_frames)
    if eval_frames <= 0:
        raise ValueError("Evaluated frames must be > 0.")
    if args.window_size > 0 and args.future_frames > 0:
        print("Both --window_size and --future_frames were set; using --window_size.")

    window_starts = build_window_starts(
        total_frames=total_frames,
        window_size=eval_frames,
        window_stride=args.window_stride,
        window_start=args.window_start,
        num_windows=args.num_windows,
    )

    clip_duration_sec = float(eval_frames) / float(model.fps)
    cfg_kwargs = _resolve_cfg_kwargs(args)

    print(f"Prompt: {args.prompt!r}")
    print(f"Constraints path: {args.constraints}")
    print(f"Base constraints count: {len(base_constraints)}")
    print(f"Duration (requested): {args.duration:.4f}s")
    print(f"Total timeline frames: {total_frames} @ model_fps={model.fps}")
    print(f"Window frames: {eval_frames}")
    print(f"Window starts: {window_starts}")
    if max_constraint_frame is not None:
        print(f"Max constraint frame index: {max_constraint_frame}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Warmup runs: {args.warmup}, measured runs: {args.repeats}")
    print(f"Hard projection: {int(args.hard_project_observed_motion)}")

    all_timings: list[float] = []
    window_summaries: list[dict] = []

    for win_i, start in enumerate(window_starts):
        win_constraints = trim_constraints_to_horizon(base_constraints, eval_frames, start_frame=start)
        heading_angle = extract_heading_angle_from_npz_at_frame(args.heading_source_npz, start, model.fps)
        first_heading = torch.tensor([heading_angle], dtype=torch.float32, device=device)

        if win_i == 0:
            for _ in range(args.warmup):
                _ = model(
                    args.prompt,
                    eval_frames,
                    constraint_lst=win_constraints,
                    num_denoising_steps=args.diffusion_steps,
                    num_samples=1,
                    multi_prompt=False,
                    post_processing=False,
                    return_numpy=False,
                    first_heading_angle=first_heading,
                    hard_project_observed_motion=args.hard_project_observed_motion,
                    hard_project_prefix_frames=args.hard_project_prefix_frames,
                    hard_project_release_frames=args.hard_project_release_frames,
                    progress_bar=_noop_progress,
                    **cfg_kwargs,
                )
                _sync_if_cuda(device)

        timings: list[float] = []
        for _ in range(args.repeats):
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = model(
                args.prompt,
                eval_frames,
                constraint_lst=win_constraints,
                num_denoising_steps=args.diffusion_steps,
                num_samples=1,
                multi_prompt=False,
                post_processing=False,
                return_numpy=False,
                first_heading_angle=first_heading,
                hard_project_observed_motion=args.hard_project_observed_motion,
                hard_project_prefix_frames=args.hard_project_prefix_frames,
                hard_project_release_frames=args.hard_project_release_frames,
                progress_bar=_noop_progress,
                **cfg_kwargs,
            )
            _sync_if_cuda(device)
            timings.append(time.perf_counter() - t0)

        all_timings.extend(timings)
        win_sorted = sorted(timings)
        win_mean = float(statistics.mean(timings))
        win_p50 = float(statistics.median(timings))
        win_p95 = _percentile(win_sorted, 0.95)
        window_summaries.append(
            {
                "window_index": win_i,
                "start_frame": start,
                "end_frame": start + eval_frames - 1,
                "constraints_count": len(win_constraints),
                "heading_angle_rad": heading_angle,
                "latency_sec": {"mean": win_mean, "p50": win_p50, "p95": win_p95},
                "regen_hz": {
                    "mean": (1.0 / win_mean) if win_mean > 0 else float("inf"),
                    "p50": (1.0 / win_p50) if win_p50 > 0 else float("inf"),
                    "p95": (1.0 / win_p95) if win_p95 > 0 else float("inf"),
                },
            }
        )

    s = sorted(all_timings)
    mean_t = float(statistics.mean(all_timings))
    std_t = float(statistics.pstdev(all_timings)) if len(all_timings) > 1 else 0.0
    p50_t = float(statistics.median(all_timings))
    p90_t = _percentile(s, 0.90)
    p95_t = _percentile(s, 0.95)
    min_t = float(s[0])
    max_t = float(s[-1])

    def _safe_inv(x: float) -> float:
        return 1.0 / x if x > 0 else float("inf")

    results = {
        "device": device,
        "model": resolved_model,
        "prompt": args.prompt,
        "constraints_path": args.constraints,
        "frames": eval_frames,
        "total_frames": total_frames,
        "window_starts": window_starts,
        "num_windows": len(window_starts),
        "model_fps": float(model.fps),
        "clip_duration_sec": clip_duration_sec,
        "warmup_runs": args.warmup,
        "measured_runs": args.repeats,
        "total_measured_runs": len(all_timings),
        "diffusion_steps": args.diffusion_steps,
        "latency_sec": {
            "mean": mean_t,
            "std": std_t,
            "p50": p50_t,
            "p90": p90_t,
            "p95": p95_t,
            "min": min_t,
            "max": max_t,
        },
        "regen_hz": {
            "mean": _safe_inv(mean_t),
            "p50": _safe_inv(p50_t),
            "p95": _safe_inv(p95_t),
        },
        "gen_fps": {
            "mean": eval_frames / mean_t if mean_t > 0 else float("inf"),
            "p50": eval_frames / p50_t if p50_t > 0 else float("inf"),
            "p95": eval_frames / p95_t if p95_t > 0 else float("inf"),
        },
        "rtf": {
            "mean": clip_duration_sec / mean_t if mean_t > 0 else float("inf"),
            "p50": clip_duration_sec / p50_t if p50_t > 0 else float("inf"),
            "p95": clip_duration_sec / p95_t if p95_t > 0 else float("inf"),
        },
        "per_window": window_summaries,
    }

    print("")
    print("=== Benchmark Summary (model(...) only) ===")
    print(f"Windows: {len(window_starts)} | runs per window: {args.repeats} | total runs: {len(all_timings)}")
    print(f"Latency mean: {mean_t:.6f}s | p50: {p50_t:.6f}s | p95: {p95_t:.6f}s")
    print(f"Regen Hz mean: {results['regen_hz']['mean']:.4f} | p50: {results['regen_hz']['p50']:.4f} | p95: {results['regen_hz']['p95']:.4f}")
    print(f"Gen FPS mean: {results['gen_fps']['mean']:.3f} | p50: {results['gen_fps']['p50']:.3f} | p95: {results['gen_fps']['p95']:.3f}")
    print(f"RTF mean: {results['rtf']['mean']:.3f} | p50: {results['rtf']['p50']:.3f} | p95: {results['rtf']['p95']:.3f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
