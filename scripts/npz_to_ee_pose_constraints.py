#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

EE_FIELD_MAP = {
    "left_wrist": "left_hand_pose",
    "right_wrist": "right_hand_pose",
    "left_foot": "left_foot_pose",
    "right_foot": "right_foot_pose",
}


def mujoco_xyz_to_kimodo(xyz_m: np.ndarray) -> np.ndarray:
    # [x_k, y_k, z_k] = [y_m, z_m, x_m]
    return np.asarray([xyz_m[1], xyz_m[2], xyz_m[0]], dtype=np.float64)


def rot_mujoco_to_kimodo(rot_m: np.ndarray) -> np.ndarray:
    # R_k = M * R_m * M^T
    return MUJOCO_TO_KIMODO @ rot_m @ MUJOCO_TO_KIMODO.T


def yaw_from_rot_kimodo(rot_k: np.ndarray) -> float:
    # Kimodo forward axis is +z. Project forward vector to x-z plane and take yaw around +y.
    forward = rot_k @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(np.arctan2(forward[0], forward[2]))


def select_source_indices(num_frames: int, step: int) -> np.ndarray:
    idx = np.arange(0, num_frames, step, dtype=np.int64)
    if len(idx) == 0 or idx[-1] != num_frames - 1:
        idx = np.append(idx, num_frames - 1)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Convert custom 50Hz NPZ to Kimodo ee-pose constraints JSON.")
    parser.add_argument("--input", required=True, help="Input custom npz path.")
    parser.add_argument("--output", required=True, help="Output constraints.json path.")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Kimodo target FPS (default: 30).")
    parser.add_argument(
        "--keyframe-step",
        type=int,
        default=10,
        help="Keyframe stride in source frames (default: 10 at 50Hz ~= every 0.2s).",
    )
    parser.add_argument(
        "--root-constraint-mode",
        type=str,
        default="xyzyaw",
        choices=["xyzyaw", "root2d", "none"],
        help=(
            "Root constraint mode: "
            "'xyzyaw' => embed root_xyzyaw in ee-pose (default), "
            "'root2d' => output ee-pose + separate root2d constraint, "
            "'none' => ee-only."
        ),
    )
    parser.add_argument(
        "--include-root-xyzyaw",
        type=int,
        default=None,
        choices=[0, 1],
        help="Deprecated compatibility flag. If set, overrides --root-constraint-mode (1=>xyzyaw, 0=>none).",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(str(inp), allow_pickle=True)
    required_keys = {"fps", "root_global_6d", "ee_root_relative_6d", "ee_names"}
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"Missing required keys in npz: {sorted(missing)}")

    src_fps = float(data["fps"])
    root_global_6d = np.asarray(data["root_global_6d"], dtype=np.float64)  # [T, 6] => xyz + rpy
    ee_rel_6d = np.asarray(data["ee_root_relative_6d"], dtype=np.float64)  # [T, E, 6]
    ee_names = [str(x) for x in data["ee_names"].tolist()]

    num_frames = root_global_6d.shape[0]
    src_idx = select_source_indices(num_frames=num_frames, step=max(1, args.keyframe_step))
    dst_idx = np.round(src_idx * args.target_fps / src_fps).astype(np.int64)

    # Remove duplicates after FPS remap while preserving order.
    uniq_mask = np.ones(len(dst_idx), dtype=bool)
    uniq_mask[1:] = dst_idx[1:] != dst_idx[:-1]
    src_idx = src_idx[uniq_mask]
    dst_idx = dst_idx[uniq_mask]

    root_mode = args.root_constraint_mode
    if args.include_root_xyzyaw is not None:
        root_mode = "xyzyaw" if int(args.include_root_xyzyaw) == 1 else "none"

    ee_item = {
        "type": "ee-pose",
        "frame_indices": dst_idx.tolist(),
    }
    if root_mode == "xyzyaw":
        ee_item["root_xyzyaw"] = []

    root2d_item = None
    if root_mode == "root2d":
        root2d_item = {
            "type": "root2d",
            "frame_indices": dst_idx.tolist(),
            "smooth_root_2d": [],
        }

    for field in EE_FIELD_MAP.values():
        ee_item[field] = []

    name_to_col = {n: i for i, n in enumerate(ee_names)}

    for t in src_idx:
        root_xyz_m = root_global_6d[t, :3]
        root_rpy_m = root_global_6d[t, 3:]
        root_rot_m = R.from_euler("xyz", root_rpy_m).as_matrix()
        root_rot_k = rot_mujoco_to_kimodo(root_rot_m)
        root_xyz_k = mujoco_xyz_to_kimodo(root_xyz_m)
        root_yaw_k = yaw_from_rot_kimodo(root_rot_k)
        if root_mode == "xyzyaw":
            ee_item["root_xyzyaw"].append([float(root_xyz_k[0]), float(root_xyz_k[1]), float(root_xyz_k[2]), root_yaw_k])
        elif root_mode == "root2d":
            root2d_item["smooth_root_2d"].append([float(root_xyz_k[0]), float(root_xyz_k[2])])

        for ee_name, field in EE_FIELD_MAP.items():
            if ee_name not in name_to_col:
                continue
            c = name_to_col[ee_name]
            rel_xyz_m = ee_rel_6d[t, c, :3]
            rel_rpy_m = ee_rel_6d[t, c, 3:]
            rel_rot_m = R.from_euler("xyz", rel_rpy_m).as_matrix()

            ee_xyz_m = root_xyz_m + root_rot_m @ rel_xyz_m
            ee_rot_m = root_rot_m @ rel_rot_m
            ee_xyz_k = mujoco_xyz_to_kimodo(ee_xyz_m)
            ee_rot_k = rot_mujoco_to_kimodo(ee_rot_m)
            ee_rpy_k = R.from_matrix(ee_rot_k).as_euler("xyz")

            ee_item[field].append(
                [
                    float(ee_xyz_k[0]),
                    float(ee_xyz_k[1]),
                    float(ee_xyz_k[2]),
                    float(ee_rpy_k[0]),
                    float(ee_rpy_k[1]),
                    float(ee_rpy_k[2]),
                ]
            )

    # Drop any empty EE field if input ee_names does not include it.
    for field in list(EE_FIELD_MAP.values()):
        if len(ee_item[field]) == 0:
            ee_item.pop(field)

    output_items = [ee_item]
    if root2d_item is not None:
        output_items.append(root2d_item)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(output_items, f, indent=2)

    duration_sec = num_frames / src_fps
    print(f"Saved constraints: {out}")
    print(f"source_fps={src_fps}, target_fps={args.target_fps}")
    print(f"root_constraint_mode={root_mode}")
    print(f"source_frames={num_frames}, duration_sec={duration_sec:.4f}")
    print(f"selected_keyframes={len(src_idx)}, first_last_src=({int(src_idx[0])},{int(src_idx[-1])})")
    print(f"first_last_dst=({int(dst_idx[0])},{int(dst_idx[-1])})")


if __name__ == "__main__":
    main()
