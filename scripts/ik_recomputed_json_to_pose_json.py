#!/usr/bin/env python3
"""Extract retargeted five-point poses from a HUMI ik_recomputed JSON.

The output intentionally matches the raw HUMI recording shape expected by
``humi_json_to_ee_pose_npz.py``: each frame has timestamp plus root/hand/foot
poses with ``position`` and ``quaternion_wxyz`` fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POSE_FIELDS = (
    "root_pose",
    "left_hand_pose",
    "right_hand_pose",
    "left_foot_pose",
    "right_foot_pose",
)


def _quat_to_wxyz(pose: dict[str, Any]) -> list[float]:
    if "quaternion_wxyz" in pose:
        quat = pose["quaternion_wxyz"]
        if len(quat) != 4:
            raise ValueError(f"quaternion_wxyz must have length 4, got {len(quat)}.")
        return [float(x) for x in quat]
    if "quaternion_xyzw" in pose:
        x, y, z, w = pose["quaternion_xyzw"]
        return [float(w), float(x), float(y), float(z)]
    raise ValueError("Pose is missing quaternion_wxyz/quaternion_xyzw.")


def _extract_pose_frame(frame: dict[str, Any], pose_source: str) -> dict[str, Any]:
    source_frame = frame if pose_source == "top_level" else frame.get(pose_source)
    if not isinstance(source_frame, dict):
        raise ValueError(f"Frame missing pose source '{pose_source}'.")

    out: dict[str, Any] = {"timestamp": float(frame["timestamp"])}
    for field in POSE_FIELDS:
        pose = source_frame.get(field)
        if not isinstance(pose, dict):
            raise ValueError(f"Frame missing {pose_source}.{field}.")
        position = pose.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"{pose_source}.{field}.position must have length 3.")
        out[field] = {
            "position": [float(x) for x in position],
            "quaternion_wxyz": _quat_to_wxyz(pose),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract retargeted HUMI IK poses into raw-HUMI-like JSON.")
    parser.add_argument("--input", required=True, help="Input ik_recomputed recording JSON.")
    parser.add_argument("--output", required=True, help="Output raw-HUMI-like pose JSON.")
    parser.add_argument(
        "--pose-source",
        default="realized_target",
        choices=["realized_target", "transformed_target", "top_level"],
        help=(
            "Which five-point poses to extract. realized_target is the IK-retargeted "
            "robot realization; transformed_target is the retargeted target before IK error; "
            "top_level is the original HUMI pose stored on each IK frame."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    episode = data.get("episode")
    if not isinstance(episode, list) or not episode:
        raise ValueError(f"Expected non-empty episode in {input_path}.")

    converted = [_extract_pose_frame(frame, args.pose_source) for frame in episode]
    output = {
        "episode": converted,
        "metadata": {
            "source_json_path": str(input_path),
            "source_format": "humi-ik-recomputed",
            "pose_source": args.pose_source,
            "mjcf_path": data.get("mjcf_path", ""),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Saved pose JSON: {output_path}")
    print(f"pose_source={args.pose_source}")
    print(f"frames={len(converted)}")


if __name__ == "__main__":
    main()
