#!/usr/bin/env python3
"""Convert raw HUMI trajectory JSON into the ee-pose adapter NPZ format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


POSE_FIELDS = {
    "root": "root_pose",
    "left_wrist": "left_hand_pose",
    "right_wrist": "right_hand_pose",
    "left_foot": "left_foot_pose",
    "right_foot": "right_foot_pose",
}
EE_NAMES = ("left_wrist", "right_wrist", "left_foot", "right_foot")
POSE_FIELD_NAMES = tuple(POSE_FIELDS.values())


def canonicalize_humi_positions(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Move HUMI world positions into a local MuJoCo-like frame.

    HUMI raw trajectories keep arbitrary world XY and often have the floor below
    z=0. Kimodo/G1 generation is easiest to compare when the first root is near
    horizontal origin and the lowest foot height is on the ground plane.
    """
    out = np.asarray(positions, dtype=np.float64).copy()
    root_xy0 = out[0, 0, :2].copy()
    floor_z = float(out[:, 3:5, 2].min())
    out[..., 0] -= root_xy0[0]
    out[..., 1] -= root_xy0[1]
    out[..., 2] -= floor_z
    return out, root_xy0, floor_z


def _load_episode(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    episode = data.get("episode")
    if not isinstance(episode, list) or not episode:
        raise ValueError(f"Expected non-empty 'episode' list in {path}.")
    return episode


def _infer_fps(episode: list[dict]) -> float:
    timestamps = np.asarray([float(frame["timestamp"]) for frame in episode], dtype=np.float64)
    if timestamps.shape[0] < 2:
        raise ValueError("Need at least two timestamps to infer FPS.")
    dt = np.diff(timestamps)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Timestamps are not increasing; cannot infer FPS.")
    return float(1.0 / np.mean(dt))


def _pose_xyz_rot(frame: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    pose = frame[field]
    xyz = np.asarray(pose["position"], dtype=np.float64)
    quat_wxyz = np.asarray(pose["quaternion_wxyz"], dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"{field}.position must have shape (3,), got {xyz.shape}.")
    if quat_wxyz.shape != (4,):
        raise ValueError(f"{field}.quaternion_wxyz must have shape (4,), got {quat_wxyz.shape}.")
    rot = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return xyz, rot


def build_humi_ee_pose_npz(
    json_path: str | Path,
    *,
    fps: float | None = None,
    max_source_frames: int | None = None,
) -> dict:
    path = Path(json_path)
    episode = _load_episode(path)
    source_fps = float(fps) if fps is not None else _infer_fps(episode)
    if max_source_frames is not None:
        if max_source_frames <= 0:
            raise ValueError(f"max_source_frames must be positive, got {max_source_frames}.")
        episode = episode[: min(int(max_source_frames), len(episode))]

    num_frames = len(episode)
    positions = np.zeros((num_frames, len(POSE_FIELD_NAMES), 3), dtype=np.float64)
    rotations = np.zeros((num_frames, len(POSE_FIELD_NAMES), 3, 3), dtype=np.float64)
    for t, frame in enumerate(episode):
        for j, field in enumerate(POSE_FIELD_NAMES):
            positions[t, j], rotations[t, j] = _pose_xyz_rot(frame, field)
    positions, origin_xy, floor_z = canonicalize_humi_positions(positions)

    root_global_6d = np.zeros((num_frames, 6), dtype=np.float64)
    ee_root_relative_6d = np.zeros((num_frames, len(EE_NAMES), 6), dtype=np.float64)

    for t in range(num_frames):
        root_xyz = positions[t, 0]
        root_rot = rotations[t, 0]
        root_global_6d[t, :3] = root_xyz
        root_global_6d[t, 3:] = R.from_matrix(root_rot).as_euler("xyz")

        for ee_idx, ee_name in enumerate(EE_NAMES):
            pose_col = POSE_FIELD_NAMES.index(POSE_FIELDS[ee_name])
            ee_xyz = positions[t, pose_col]
            ee_rot = rotations[t, pose_col]
            rel_xyz = root_rot.T @ (ee_xyz - root_xyz)
            rel_rot = root_rot.T @ ee_rot
            ee_root_relative_6d[t, ee_idx, :3] = rel_xyz
            ee_root_relative_6d[t, ee_idx, 3:] = R.from_matrix(rel_rot).as_euler("xyz")

    timestamps = np.asarray([float(frame["timestamp"]) for frame in episode], dtype=np.float64)
    return {
        "fps": np.asarray(source_fps, dtype=np.float64),
        "root_global_6d": root_global_6d,
        "ee_root_relative_6d": ee_root_relative_6d,
        "ee_names": np.asarray(EE_NAMES),
        "source_format": np.asarray("humi-json"),
        "source_coord_system": np.asarray("xyz-z-up"),
        "root_translation_order": np.asarray("xyz"),
        "root_position_unit": np.asarray("meters"),
        "root_rotation_type": np.asarray("quaternion_wxyz"),
        "canonical_origin_xy": origin_xy,
        "canonical_floor_z": np.asarray(floor_z, dtype=np.float64),
        "source_json_path": np.asarray(str(path)),
        "timestamps": timestamps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HUMI raw trajectory JSON into the adapter NPZ used by ee-pose constraints."
    )
    parser.add_argument("--input", required=True, help="Path to HUMI recording JSON.")
    parser.add_argument("--output", required=True, help="Path to output adapter NPZ.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional source FPS override. By default FPS is inferred from timestamps.",
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=None,
        help="Optional cap on source frames kept from the beginning of the HUMI recording.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    converted = build_humi_ee_pose_npz(args.input, fps=args.fps, max_source_frames=args.max_source_frames)
    np.savez(output, **converted)

    print(f"Saved adapter NPZ: {output}")
    print(f"fps={float(converted['fps']):.4f}")
    print(f"root_global_6d_shape={tuple(converted['root_global_6d'].shape)}")
    print(f"ee_root_relative_6d_shape={tuple(converted['ee_root_relative_6d'].shape)}")
    print(f"ee_names={converted['ee_names'].tolist()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
