#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo.training.custom_motion_ee_pose_npz import save_custom_motion_ee_pose_npz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw custom-motion qpos NPZ into the ee-pose adapter NPZ used by constraint extraction."
    )
    parser.add_argument("--input", required=True, help="Path to raw custom-motion NPZ (expects qpos + fps).")
    parser.add_argument("--output", required=True, help="Path to output adapter NPZ.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = save_custom_motion_ee_pose_npz(args.input, args.output)
    converted = np.load(output_path, allow_pickle=False)
    print(f"Saved adapter NPZ: {output_path}")
    print(f"fps={float(np.asarray(converted['fps']).item()):.4f}")
    print(f"root_global_6d_shape={tuple(converted['root_global_6d'].shape)}")
    print(f"ee_root_relative_6d_shape={tuple(converted['ee_root_relative_6d'].shape)}")
    print(f"ee_names={converted['ee_names'].tolist()}")


if __name__ == "__main__":
    main()
