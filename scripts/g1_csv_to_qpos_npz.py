#!/usr/bin/env python3
"""Convert a G1 training CSV clip into MuJoCo qpos NPZ for downstream tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton import G1Skeleton34
from kimodo.training.g1_csv import load_g1_csv_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert G1 CSV motion to qpos NPZ.")
    parser.add_argument("--input", required=True, help="Input G1 CSV path.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument("--input-fps", type=float, default=120.0, help="Input CSV FPS (default: 120).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    motion = load_g1_csv_motion(
        in_path,
        source_coord_system="mujoco",
        root_euler_order="xyz",
        root_angle_unit="degrees",
        joint_angle_unit="degrees",
        root_position_unit="centimeters",
        root_position_scale=1.0,
        device="cpu",
        dtype=torch.float32,
    )

    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    qpos = converter.to_qpos(
        motion["local_joint_rots"].unsqueeze(0),
        motion["root_positions"].unsqueeze(0),
        root_quat_w_first=True,
        mujoco_rest_zero=False,
    )
    qpos_np = np.asarray(qpos.squeeze(0).cpu().numpy(), dtype=np.float32)

    np.savez(
        out_path,
        fps=np.asarray(float(args.input_fps), dtype=np.float32),
        qpos=qpos_np,
    )
    print(f"Saved qpos npz: {out_path}")
    print(f"frames={qpos_np.shape[0]}, qpos_dim={qpos_np.shape[1]}, fps={float(args.input_fps)}")


if __name__ == "__main__":
    main()

