#!/usr/bin/env python3
"""Check generated G1 qpos CSV against dense HUMI raw EE trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.humi_json_to_ee_pose_npz import canonicalize_humi_positions


FIELD_TO_BODY = {
    "root_pose": "pelvis",
    "left_hand_pose": "left_wrist_yaw_link",
    "right_hand_pose": "right_wrist_yaw_link",
    "left_foot_pose": "left_ankle_roll_link",
    "right_foot_pose": "right_ankle_roll_link",
}


def _infer_fps(timestamps: np.ndarray) -> float:
    dt = np.diff(timestamps)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Cannot infer FPS from timestamps.")
    return float(1.0 / np.mean(dt))


def _load_humi_raw(path: Path, *, canonicalize: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    episode = data.get("episode")
    if not isinstance(episode, list) or not episode:
        raise ValueError(f"Expected non-empty episode in {path}.")

    timestamps = np.asarray([float(frame["timestamp"]) for frame in episode], dtype=np.float64)
    fps = _infer_fps(timestamps)
    field_order = list(FIELD_TO_BODY)
    all_positions = np.asarray(
        [[frame[field]["position"] for field in field_order] for frame in episode],
        dtype=np.float64,
    )
    if canonicalize:
        all_positions, _origin_xy, _floor_z = canonicalize_humi_positions(all_positions)
    positions = {field: all_positions[:, i] for i, field in enumerate(field_order)}
    return timestamps, positions, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated G1 CSV against HUMI raw EE trajectories.")
    parser.add_argument("--csv", required=True, help="Generated MuJoCo qpos CSV.")
    parser.add_argument("--humi-json", required=True, help="HUMI raw_trajectories recording JSON.")
    parser.add_argument("--generated-fps", type=float, default=30.0, help="Generated CSV FPS.")
    parser.add_argument(
        "--xml",
        default="kimodo/assets/skeletons/g1skel34/xml/g1.xml",
        help="Path to generated G1 MuJoCo XML.",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["left_hand_pose", "right_hand_pose", "left_foot_pose", "right_foot_pose"],
        choices=sorted(FIELD_TO_BODY),
        help="HUMI fields to evaluate.",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Do not translate root XY to origin or shift floor height to z=0 before computing errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qpos = np.loadtxt(args.csv, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None, :]

    timestamps, humi_positions, humi_fps = _load_humi_raw(Path(args.humi_json), canonicalize=not args.no_canonicalize)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    body_ids = {
        field: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FIELD_TO_BODY[field])
        for field in args.fields
    }
    for field, bid in body_ids.items():
        if bid < 0:
            raise ValueError(f"Body not found for {field}: {FIELD_TO_BODY[field]}")

    rows = []
    per_field = {field: [] for field in args.fields}
    for gen_idx in range(qpos.shape[0]):
        src_idx = int(round(gen_idx * humi_fps / float(args.generated_fps)))
        if src_idx >= len(timestamps):
            break

        data.qpos[:] = qpos[gen_idx]
        mujoco.mj_forward(model, data)

        for field in args.fields:
            actual = np.asarray(data.xpos[body_ids[field]], dtype=np.float64)
            expected = humi_positions[field][src_idx]
            err = float(np.linalg.norm(actual - expected))
            per_field[field].append(err)
            rows.append(err)

    print(f"CSV: {args.csv}")
    print(f"HUMI: {args.humi_json}")
    print(f"canonicalized={not args.no_canonicalize}")
    print(f"humi_fps={humi_fps:.6f}, generated_fps={float(args.generated_fps):.6f}")
    print(f"evaluated_generated_frames={sum(len(v) for v in per_field.values()) // max(len(args.fields), 1)}")
    print("")
    for field in args.fields:
        vals = np.asarray(per_field[field], dtype=np.float64)
        if vals.size == 0:
            print(f"{field}: no rows")
            continue
        print(
            f"{field}: mean_error_m={float(vals.mean()):.6f}, "
            f"max_error_m={float(vals.max()):.6f}, min_error_m={float(vals.min()):.6f}"
        )
    if rows:
        vals = np.asarray(rows, dtype=np.float64)
        print("")
        print(f"overall_mean_error_m={float(vals.mean()):.6f}")
        print(f"overall_max_error_m={float(vals.max()):.6f}")


if __name__ == "__main__":
    main()
