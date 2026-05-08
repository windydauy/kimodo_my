#!/usr/bin/env python3
"""Build dense full-body prefix constraints from raw custom-motion qpos NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo.constraints import FullBodyConstraintSet
from kimodo.skeleton import G1Skeleton34
from kimodo.training.custom_motion_npz import load_g1_npz_motion, resample_motion


def _tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _tensor_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_to_list(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first N frames of full-body ground truth from a raw custom-motion qpos NPZ "
            "and save them as Kimodo fullbody constraints."
        )
    )
    parser.add_argument("--input", required=True, help="Path to raw custom-motion NPZ (expects qpos + fps).")
    parser.add_argument("--output", required=True, help="Output constraints JSON path.")
    parser.add_argument(
        "--prefix-frames",
        type=int,
        required=True,
        help="Number of leading frames (at target FPS) to keep as dense full-body constraints.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target FPS used by generation (default: 30.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.prefix_frames) <= 0:
        raise ValueError("--prefix-frames must be > 0.")

    motion = load_g1_npz_motion(args.input, device="cpu", dtype=torch.float32)
    local_joint_rots = motion["local_joint_rots"]
    root_positions = motion["root_positions"]
    src_fps = float(motion["input_fps"])
    target_fps = float(args.target_fps)

    local_joint_rots, root_positions = resample_motion(
        local_joint_rots,
        root_positions,
        input_fps=src_fps,
        target_fps=target_fps,
    )

    total_frames = int(local_joint_rots.shape[0])
    prefix_frames = min(int(args.prefix_frames), total_frames)
    if prefix_frames <= 0:
        raise ValueError("No frames available after resampling.")

    skeleton = G1Skeleton34()
    frame_indices = torch.arange(prefix_frames, dtype=torch.long)
    global_joints_rots, global_joints_positions, _ = skeleton.fk(
        local_joint_rots[:prefix_frames],
        root_positions[:prefix_frames],
    )

    constraint = FullBodyConstraintSet(
        skeleton=skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_joints_positions,
        global_joints_rots=global_joints_rots,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_tensor_to_list(constraint.get_save_info())]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved fullbody prefix constraints: {output_path}")
    print(f"source_fps={src_fps}, target_fps={target_fps}")
    print(f"resampled_total_frames={total_frames}")
    print(f"prefix_frames_saved={prefix_frames}")


if __name__ == "__main__":
    main()
