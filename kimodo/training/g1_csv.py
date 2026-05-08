# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load BONES-SEED G1 CSV motions into Kimodo training tensors.

This module converts one motion CSV into:
- ``local_joint_rots`` with shape ``[T, 34, 3, 3]``
- ``root_positions`` with shape ``[T, 3]``

Assumptions used here:
- CSV contains root columns ``root_translateX/Y/Z`` and ``root_rotateX/Y/Z``.
- Joint columns end with ``_joint_dof``.
- Joint axes are read from the MuJoCo G1 XML.
- By default, CSV is interpreted in MuJoCo coordinates and converted to Kimodo coordinates.
- ``34`` follows Kimodo's G1 bone order (pelvis + 33 joints), with non-DoF end joints
  left as identity rotations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch


PathLike = Union[str, Path]


# Must match kimodo.skeleton.definitions.G1Skeleton34 bone_order_names_with_parents.
G1_BONE_ORDER_NAMES: Sequence[str] = (
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
)

ROOT_TRANSLATE_COLS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATE_COLS = ("root_rotateX", "root_rotateY", "root_rotateZ")

# From kimodo.exports.mujoco.MujocoQposConverter:
# mujoco_to_kimodo_matrix = [[0,1,0],[0,0,1],[1,0,0]]
MUJOCO_TO_KIMODO_MATRIX = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
KIMODO_TO_MUJOCO_MATRIX = MUJOCO_TO_KIMODO_MATRIX.T


def _default_g1_xml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "skeletons" / "g1skel34" / "xml" / "g1.xml"


def _to_radians(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "radians":
        return values
    if unit == "degrees":
        return np.deg2rad(values)
    if unit == "auto":
        max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
        # Heuristic: anything above ~2*pi is likely in degrees.
        if max_abs > 6.5:
            return np.deg2rad(values)
        return values
    raise ValueError(f"Unknown angle unit: {unit!r}. Expected one of: degrees, radians, auto.")


def _root_position_unit_scale(root_positions: np.ndarray, source_coord_system: str, unit: str) -> float:
    if unit == "meters":
        return 1.0
    if unit == "centimeters":
        return 0.01
    if unit != "auto":
        raise ValueError(f"Unknown root position unit: {unit!r}. Expected one of: meters, centimeters, auto.")

    # Infer from vertical axis magnitude.
    # Typical root height:
    # - meters: around 0.6~1.2
    # - centimeters: around 60~120
    vertical_idx = 2 if source_coord_system == "mujoco" else 1
    vertical = root_positions[:, vertical_idx]
    median_abs = float(np.median(np.abs(vertical)))
    if median_abs > 10.0:
        return 0.01
    return 1.0


def _load_csv_table(csv_path: PathLike) -> tuple[list[str], np.ndarray]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(header):
        raise ValueError(
            f"CSV parsing failed for {path}: got {data.shape[1]} columns in data, "
            f"but header has {len(header)} columns."
        )
    return header, data


def _axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors ``[T, 3]`` to rotation matrices ``[T, 3, 3]``.

    Pure NumPy implementation (Rodrigues), avoiding SciPy native kernels.
    """
    if rotvec.ndim != 2 or rotvec.shape[1] != 3:
        raise ValueError(f"rotvec must have shape [T, 3], got {rotvec.shape}.")

    angles = np.linalg.norm(rotvec, axis=1)  # [T]
    # Safe normalized axis; keep zero-angle rows at axis=0.
    axis = np.zeros_like(rotvec, dtype=np.float64)
    nonzero = angles > 1e-12
    axis[nonzero] = rotvec[nonzero] / angles[nonzero, None]

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]

    K = np.zeros((rotvec.shape[0], 3, 3), dtype=np.float64)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x

    I = np.broadcast_to(np.eye(3, dtype=np.float64), K.shape)
    sin_t = np.sin(angles)[:, None, None]
    one_minus_cos_t = (1.0 - np.cos(angles))[:, None, None]
    KK = K @ K
    R = I + sin_t * K + one_minus_cos_t * KK
    return R.astype(np.float32)


def _single_axis_rotation_matrices(axis_char: str, angles: np.ndarray) -> np.ndarray:
    """Create rotation matrices for one axis and per-frame angles.

    Args:
        axis_char: one of ``x``, ``y``, ``z``.
        angles: shape ``[T]`` radians.
    """
    c = np.cos(angles)
    s = np.sin(angles)
    T = angles.shape[0]
    R = np.broadcast_to(np.eye(3, dtype=np.float64), (T, 3, 3)).copy()
    if axis_char == "x":
        R[:, 1, 1] = c
        R[:, 1, 2] = -s
        R[:, 2, 1] = s
        R[:, 2, 2] = c
    elif axis_char == "y":
        R[:, 0, 0] = c
        R[:, 0, 2] = s
        R[:, 2, 0] = -s
        R[:, 2, 2] = c
    elif axis_char == "z":
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
    else:
        raise ValueError(f"Unsupported axis {axis_char!r}; expected one of x/y/z.")
    return R.astype(np.float32)


def _euler_to_matrix(euler_angles: np.ndarray, order: str) -> np.ndarray:
    """Convert Euler angles ``[T, 3]`` to rotation matrices ``[T, 3, 3]``.

    This matches the previous training usage where ``order='xyz'`` and angles
    are provided as root_rotateX/Y/Z (radians).
    """
    if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
        raise ValueError(f"euler_angles must have shape [T, 3], got {euler_angles.shape}.")
    order = order.lower()
    if len(order) != 3 or any(c not in "xyz" for c in order):
        raise ValueError(f"Unsupported root_euler_order={order!r}; expected a 3-char string in xyz.")

    a0 = _single_axis_rotation_matrices(order[0], euler_angles[:, 0])
    a1 = _single_axis_rotation_matrices(order[1], euler_angles[:, 1])
    a2 = _single_axis_rotation_matrices(order[2], euler_angles[:, 2])
    # Match scipy Rotation.from_euler(order, ...): for order "xyz", this is Rz @ Ry @ Rx.
    return (a2 @ a1 @ a0).astype(np.float32)


@lru_cache(maxsize=8)
def _load_joint_axes_from_xml_cached(xml_path_str: str) -> Dict[str, np.ndarray]:
    root = ET.parse(xml_path_str).getroot()

    class_axis: Dict[str, str] = {}
    for default_tag in root.findall(".//default"):
        cls = default_tag.get("class")
        if not cls:
            continue
        joint_tag = default_tag.find("joint")
        if joint_tag is None:
            continue
        axis = joint_tag.get("axis")
        if axis:
            class_axis[cls] = axis

    joint_axis: Dict[str, np.ndarray] = {}
    for joint_tag in root.find("worldbody").findall(".//joint"):
        joint_name = joint_tag.get("name")
        axis_str = joint_tag.get("axis")
        if axis_str is None:
            cls = joint_tag.get("class")
            axis_str = class_axis.get(cls, None)
        if joint_name is None or axis_str is None:
            continue
        axis = np.array([float(x) for x in axis_str.split()], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            raise ValueError(f"Invalid zero axis in XML for joint: {joint_name}")
        joint_axis[joint_name] = axis / norm

    return joint_axis


def _load_joint_axes_from_xml(xml_path: PathLike) -> Dict[str, np.ndarray]:
    # Cache by normalized absolute path to avoid reparsing XML on every sample.
    return _load_joint_axes_from_xml_cached(str(Path(xml_path).resolve()))


def _csv_col_to_xml_joint_name(csv_col: str) -> str:
    # e.g. left_hip_pitch_joint_dof -> left_hip_pitch_joint
    if not csv_col.endswith("_joint_dof"):
        raise ValueError(f"Invalid DOF column name: {csv_col}")
    return csv_col[: -len("_dof")]


def _xml_joint_name_to_skel_name(xml_joint_name: str) -> str:
    # e.g. left_hip_pitch_joint -> left_hip_pitch_skel
    if not xml_joint_name.endswith("_joint"):
        raise ValueError(f"Invalid XML joint name for conversion: {xml_joint_name}")
    return xml_joint_name.replace("_joint", "_skel")


def load_g1_csv_motion(
    csv_path: PathLike,
    *,
    xml_path: Optional[PathLike] = None,
    source_coord_system: str = "mujoco",
    root_euler_order: str = "xyz",
    root_angle_unit: str = "degrees",
    joint_angle_unit: str = "degrees",
    root_position_unit: str = "auto",
    root_position_scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Load one BONES-SEED G1 CSV and convert it to Kimodo-ready tensors.

    Returns a dict with:
    - ``local_joint_rots``: ``[T, 34, 3, 3]``
    - ``root_positions``: ``[T, 3]``
    - ``frame_numbers``: ``[T]``
    - ``joint_order``: tuple of 34 joint names

    Args:
        source_coord_system:
            - ``"mujoco"``: input root pose + joint axes interpreted in MuJoCo frame,
              then converted to Kimodo frame.
            - ``"kimodo"``: input already in Kimodo frame.
        root_position_unit:
            - ``"auto"``: infer meters vs centimeters from root height.
            - ``"meters"`` or ``"centimeters"``: force unit conversion.
    """
    if source_coord_system not in {"mujoco", "kimodo"}:
        raise ValueError(
            f"Unknown source_coord_system={source_coord_system!r}. "
            "Expected one of: mujoco, kimodo."
        )

    if xml_path is None:
        xml_path = _default_g1_xml_path()

    header, table = _load_csv_table(csv_path)
    col2idx = {name: i for i, name in enumerate(header)}

    required_cols = ("Frame",) + ROOT_TRANSLATE_COLS + ROOT_ROTATE_COLS
    missing_required = [c for c in required_cols if c not in col2idx]
    if missing_required:
        raise ValueError(f"CSV is missing required columns: {missing_required}")

    joint_cols = [c for c in header if c.endswith("_joint_dof")]
    if not joint_cols:
        raise ValueError("No *_joint_dof columns found in CSV.")

    frame_numbers = table[:, col2idx["Frame"]].astype(np.int64)
    root_positions_source = table[:, [col2idx[c] for c in ROOT_TRANSLATE_COLS]].astype(np.float32)
    unit_scale = _root_position_unit_scale(root_positions_source, source_coord_system, root_position_unit)
    total_root_scale = float(root_position_scale) * float(unit_scale)
    root_positions_source = root_positions_source * total_root_scale
    if source_coord_system == "mujoco":
        root_positions = np.einsum("ij,tj->ti", MUJOCO_TO_KIMODO_MATRIX, root_positions_source)
    else:
        root_positions = root_positions_source

    root_euler = table[:, [col2idx[c] for c in ROOT_ROTATE_COLS]].astype(np.float32)
    root_euler = _to_radians(root_euler, root_angle_unit)
    if not np.isfinite(root_euler).all():
        bad_count = int(np.size(root_euler) - np.isfinite(root_euler).sum())
        raise ValueError(f"Non-finite root euler angles in {csv_path}; bad_values={bad_count}.")
    root_rot_source = _euler_to_matrix(root_euler, root_euler_order)
    if source_coord_system == "mujoco":
        root_rot_mats = np.einsum(
            "ij,tjk,kl->til",
            MUJOCO_TO_KIMODO_MATRIX,
            root_rot_source,
            KIMODO_TO_MUJOCO_MATRIX,
        )
    else:
        root_rot_mats = root_rot_source

    num_frames = table.shape[0]
    nb_joints = len(G1_BONE_ORDER_NAMES)
    local_joint_rots = np.broadcast_to(np.eye(3, dtype=np.float32), (num_frames, nb_joints, 3, 3)).copy()
    local_joint_rots[:, 0] = root_rot_mats  # pelvis is root joint

    joint_axis_mujoco = _load_joint_axes_from_xml(xml_path)
    joint_name_to_index = {name: i for i, name in enumerate(G1_BONE_ORDER_NAMES)}

    for col in joint_cols:
        xml_joint_name = _csv_col_to_xml_joint_name(col)
        if xml_joint_name not in joint_axis_mujoco:
            raise KeyError(f"Joint {xml_joint_name!r} (from column {col!r}) not found in XML axis definitions.")

        skel_name = _xml_joint_name_to_skel_name(xml_joint_name)
        if skel_name not in joint_name_to_index:
            # Ignore non-kimodo joints if present; expected for strict G1 CSV this should not happen.
            continue

        joint_idx = joint_name_to_index[skel_name]
        angles = table[:, col2idx[col]].astype(np.float32)
        angles = _to_radians(angles, joint_angle_unit)
        if not np.isfinite(angles).all():
            bad_count = int(np.size(angles) - np.isfinite(angles).sum())
            raise ValueError(
                f"Non-finite joint angles for {xml_joint_name!r} (column {col!r}); "
                f"bad_values={bad_count}."
            )

        axis_from_xml = joint_axis_mujoco[xml_joint_name]
        if source_coord_system == "mujoco":
            axis_target = MUJOCO_TO_KIMODO_MATRIX @ axis_from_xml
        else:
            axis_target = axis_from_xml
        axis_norm = float(np.linalg.norm(axis_target))
        if axis_norm < 1e-8:
            raise ValueError(f"Mapped axis for joint {xml_joint_name!r} is near zero.")
        axis_target = axis_target / axis_norm

        rotvec = angles[:, None] * axis_target[None, :]
        if not np.isfinite(rotvec).all():
            bad_count = int(np.size(rotvec) - np.isfinite(rotvec).sum())
            raise ValueError(
                f"Non-finite rotvec for {xml_joint_name!r} (column {col!r}); bad_values={bad_count}."
            )
        rotmats = _axis_angle_to_matrix(rotvec)
        if not np.isfinite(rotmats).all():
            bad_count = int(np.size(rotmats) - np.isfinite(rotmats).sum())
            raise ValueError(
                f"Non-finite rotmats for {xml_joint_name!r} (column {col!r}); bad_values={bad_count}."
            )
        local_joint_rots[:, joint_idx] = rotmats

    root_positions_t = torch.as_tensor(root_positions, dtype=dtype, device=device)
    local_joint_rots_t = torch.as_tensor(local_joint_rots, dtype=dtype, device=device)
    frame_numbers_t = torch.as_tensor(frame_numbers, dtype=torch.long, device=device)

    return {
        "local_joint_rots": local_joint_rots_t,
        "root_positions": root_positions_t,
        "frame_numbers": frame_numbers_t,
        "joint_order": tuple(G1_BONE_ORDER_NAMES),
        "source_coord_system": source_coord_system,
        "root_position_scale_applied": total_root_scale,
    }
