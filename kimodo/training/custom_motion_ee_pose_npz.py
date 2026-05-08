# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt custom-motion qpos NPZ files into the sparse ee-pose NPZ format."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from kimodo.skeleton import G1Skeleton34

from .custom_motion_npz import load_g1_npz_motion
from .g1_csv import KIMODO_TO_MUJOCO_MATRIX, MUJOCO_TO_KIMODO_MATRIX
from .timeline_annotations import PathLike


EE_NAME_TO_JOINT_NAME: dict[str, str] = {
    "left_wrist": "left_wrist_yaw_skel",
    "right_wrist": "right_wrist_yaw_skel",
    "left_foot": "left_ankle_roll_skel",
    "right_foot": "right_ankle_roll_skel",
}
EE_NAMES: tuple[str, ...] = tuple(EE_NAME_TO_JOINT_NAME.keys())


def kimodo_xyz_to_mujoco(xyz_k: np.ndarray) -> np.ndarray:
    xyz_k = np.asarray(xyz_k, dtype=np.float64)
    return np.einsum("ij,...j->...i", KIMODO_TO_MUJOCO_MATRIX.astype(np.float64), xyz_k)


def rot_kimodo_to_mujoco(rot_k: np.ndarray) -> np.ndarray:
    rot_k = np.asarray(rot_k, dtype=np.float64)
    return np.einsum(
        "ij,...jk,kl->...il",
        KIMODO_TO_MUJOCO_MATRIX.astype(np.float64),
        rot_k,
        MUJOCO_TO_KIMODO_MATRIX.astype(np.float64),
    )


def _rot_mats_to_euler_xyz(rot_mats: np.ndarray) -> np.ndarray:
    flat = np.asarray(rot_mats, dtype=np.float64).reshape(-1, 3, 3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected*")
        eulers = R.from_matrix(flat).as_euler("xyz")
    return eulers.reshape(rot_mats.shape[:-2] + (3,))


def build_custom_motion_ee_pose_npz(npz_path: PathLike) -> dict[str, Any]:
    path = Path(npz_path)
    source = np.load(path, allow_pickle=False)
    if "qpos" not in source.files:
        raise ValueError(f"NPZ file missing qpos: {path}")
    if "fps" not in source.files:
        raise ValueError(f"NPZ file missing fps: {path}")

    qpos = np.asarray(source["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 36:
        raise ValueError(f"Expected qpos shape [T, >=36], got {tuple(qpos.shape)} for {path}.")

    motion = load_g1_npz_motion(path)
    skeleton = G1Skeleton34()
    global_rots_k, posed_joints_k, _ = skeleton.fk(motion["local_joint_rots"], motion["root_positions"])
    global_rots_k_np = global_rots_k.detach().cpu().numpy().astype(np.float64)
    posed_joints_k_np = posed_joints_k.detach().cpu().numpy().astype(np.float64)

    root_xyz_m = qpos[:, :3]
    root_rot_m = R.from_quat(qpos[:, 3:7], scalar_first=True).as_matrix()
    root_rpy_m = _rot_mats_to_euler_xyz(root_rot_m)

    ee_rel_6d = np.zeros((qpos.shape[0], len(EE_NAMES), 6), dtype=np.float64)
    for ee_idx, ee_name in enumerate(EE_NAMES):
        joint_name = EE_NAME_TO_JOINT_NAME[ee_name]
        joint_idx = skeleton.bone_index[joint_name]

        ee_xyz_m = kimodo_xyz_to_mujoco(posed_joints_k_np[:, joint_idx])
        ee_rot_m = rot_kimodo_to_mujoco(global_rots_k_np[:, joint_idx])

        rel_xyz_m = np.einsum("tji,tj->ti", root_rot_m, ee_xyz_m - root_xyz_m)
        rel_rot_m = np.einsum("tji,tjk->tik", root_rot_m, ee_rot_m)
        rel_rpy_m = _rot_mats_to_euler_xyz(rel_rot_m)

        ee_rel_6d[:, ee_idx, :3] = rel_xyz_m
        ee_rel_6d[:, ee_idx, 3:] = rel_rpy_m

    return {
        "fps": float(np.asarray(source["fps"]).item()),
        "root_global_6d": np.concatenate([root_xyz_m, root_rpy_m], axis=-1),
        "ee_root_relative_6d": ee_rel_6d,
        "ee_names": np.asarray(EE_NAMES),
        "source_format": np.asarray("custom-motion-qpos"),
        "source_coord_system": np.asarray("mujoco"),
        "root_translation_order": np.asarray("xyz"),
        "root_position_unit": np.asarray("meters"),
        "source_npz_path": np.asarray(str(path)),
    }


def save_custom_motion_ee_pose_npz(input_path: PathLike, output_path: PathLike) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted = build_custom_motion_ee_pose_npz(input_path)
    np.savez(output, **converted)
    return output
