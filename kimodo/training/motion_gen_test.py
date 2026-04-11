"""Generate a motion clip using official config + finetuned checkpoint override.

Example:
    python -m kimodo.training.motion_gen_test \
      --prompt "A person walks forward." \
      --duration 5.0 \
      --config kimodo/training/train_config_phase1.yaml \
      --finetuned outputs/g1_finetune-phase1/checkpoints/step_00030000.pt \
      --output outputs/eval/finetuned_gen_walk_forward.npz \
      --save-csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.tools import seed_everything
from kimodo.training.eval_compare import (
    _build_model_from_local_config,
    _load_finetuned_weights,
    _resolve_finetuned_ckpt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate motion with finetuned checkpoint override.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")
    parser.add_argument(
        "--config",
        type=str,
        default="kimodo/training/train_config_phase1.yaml",
        help="Local model config YAML used to build official model structure.",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        required=True,
        help="Finetuned checkpoint path (or output dir containing checkpoints).",
    )
    parser.add_argument("--output", type=str, default="outputs/eval/motion_gen_test.npz", help="Output .npz path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--diffusion-steps", type=int, default=100, help="Number of denoising steps.")
    parser.add_argument("--num-transition-frames", type=int, default=5, help="Transition frames for multi_prompt mode.")
    parser.add_argument("--cfg-type", type=str, default="separated", help="CFG type.")
    parser.add_argument(
        "--cfg-weight",
        type=float,
        nargs=2,
        default=[2.0, 2.0],
        metavar=("TEXT_CFG", "MOTION_CFG"),
        help="CFG weights [text, motion].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string (e.g., cuda, cuda:0, cpu). Falls back to cpu when unavailable.",
    )
    parser.add_argument("--save-csv", action="store_true", help="Also export MuJoCo qpos CSV next to npz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA unavailable, falling back to CPU.")
    else:
        device = requested_device

    model = _build_model_from_local_config(
        Path(args.config),
        device=device,
        cfg_type=args.cfg_type,
    )
    ckpt_path = _resolve_finetuned_ckpt(args.finetuned)
    load_mode = _load_finetuned_weights(model, ckpt_path, use_ema_if_available=True)

    seed_everything(int(args.seed))
    num_frames = int(round(float(args.duration) * float(model.fps)))
    if num_frames <= 0:
        raise ValueError(f"duration must be > 0, got {args.duration}.")

    output = model(
        [args.prompt],
        [num_frames],
        constraint_lst=[],
        num_denoising_steps=int(args.diffusion_steps),
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=int(args.num_transition_frames),
        post_processing=False,  # G1 path; keep consistent with kimodo_gen behavior
        return_numpy=True,
        cfg_weight=list(args.cfg_weight),
        cfg_type=args.cfg_type,
    )

    output_np = {
        key: (
            value[0]
            if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == 1
            else value
        )
        for key, value in output.items()
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output_np)
    print(f"Saved npz: {output_path}")
    print(f"Loaded finetuned checkpoint: {ckpt_path} (mode={load_mode})")
    print(f"Generated frames: {num_frames} @ {model.fps} fps")

    if args.save_csv:
        csv_path = output_path.with_suffix(".csv")
        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(output_np, device=str(device), numpy=False)
        converter.save_csv(qpos, str(csv_path))
        print(f"Saved csv: {csv_path}")


if __name__ == "__main__":
    main()
