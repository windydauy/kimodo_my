# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint sets for conditioning motion generation (root 2D, full body, end-effectors)."""

from typing import Optional, Union

import torch
from torch import Tensor

from kimodo.motion_rep.feature_utils import compute_heading_angle
from kimodo.skeleton import SkeletonBase, SOMASkeleton30, SOMASkeleton77
from kimodo.tools import ensure_batched, load_json, save_json

from .geometry import angle_to_Y_rotation_matrix, axis_angle_to_matrix, matrix_to_axis_angle


def _rpy_to_rotation_matrix(rpy: Tensor) -> Tensor:
    """Convert roll/pitch/yaw angles (radians) to rotation matrices.

    Uses the common XYZ convention: apply roll around X, pitch around Y, and yaw around Z,
    which corresponds to the matrix product ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    if rpy.shape[-1] != 3:
        raise ValueError(f"Expected [..., 3] roll/pitch/yaw tensor, got shape {tuple(rpy.shape)}.")

    roll, pitch, yaw = rpy.unbind(dim=-1)
    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)

    return torch.stack(
        (
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
            -sy,
            cy * sx,
            cy * cx,
        ),
        dim=-1,
    ).reshape(rpy.shape[:-1] + (3, 3))


def _convert_constraint_local_rots_to_skeleton(local_rot_mats: Tensor, skeleton: SkeletonBase) -> Tensor:
    """Convert loaded local rotation matrices to match the skeleton's joint count.

    Handles SOMA 30↔77: constraint files may have been saved with 30 or 77 joints while the session
    skeleton (e.g. from the SOMA30 model) uses SOMASkeleton77.
    """
    n_joints = local_rot_mats.shape[-3]
    skeleton_joints = skeleton.nbjoints
    if n_joints == skeleton_joints:
        return local_rot_mats
    if n_joints == 77 and skeleton_joints == 30 and isinstance(skeleton, SOMASkeleton30):
        return skeleton.from_SOMASkeleton77(local_rot_mats)
    if n_joints == 30 and skeleton_joints == 77 and isinstance(skeleton, SOMASkeleton77):
        skel30 = SOMASkeleton30()
        return skel30.to_SOMASkeleton77(local_rot_mats)
    raise ValueError(
        f"Constraint joint count ({n_joints}) does not match skeleton joint count "
        f"({skeleton_joints}). Only SOMA 30↔77 conversion is supported."
    )


def create_pairs(tensor_A: Tensor, tensor_B: Tensor) -> Tensor:
    """Form all (a, b) pairs from two 1D tensors; output shape (len(A)*len(B), 2)."""
    pairs = torch.stack(
        (
            tensor_A[:, None].expand(-1, len(tensor_B)),
            tensor_B.expand(len(tensor_A), -1),
        ),
        dim=-1,
    ).reshape(-1, 2)
    return pairs


def compute_global_heading(global_joints_positions: Tensor, skeleton: SkeletonBase) -> Tensor:
    """Compute global root heading (cos, sin) from global joint positions using skeleton."""
    root_heading_angle = compute_heading_angle(global_joints_positions, skeleton)
    global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)
    return global_root_heading


def _tensor_to(
    t: Tensor,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Move tensor to device and/or dtype.

    Returns same tensor if no args.
    """
    if device is not None and dtype is not None:
        return t.to(device=device, dtype=dtype)
    if device is not None:
        return t.to(device=device)
    if dtype is not None:
        return t.to(dtype=dtype)
    return t


class Root2DConstraintSet:
    """Constraint set fixing root (x, z) trajectory and optionally global heading on given
    frames."""

    name = "root2d"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        smooth_root_2d: Tensor,
        to_crop: bool = False,
        global_root_heading: Optional[Tensor] = None,
    ) -> None:
        self.skeleton = skeleton

        # if we pass the full smooth root 3D as input
        if smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]

        if to_crop:
            smooth_root_2d = smooth_root_2d[frame_indices]
            if global_root_heading is not None:
                global_root_heading = global_root_heading[frame_indices]
        else:
            assert len(smooth_root_2d) == len(
                frame_indices
            ), "The number of smooth root 2d should be match the number of frames"
            if global_root_heading is not None:
                assert len(global_root_heading) == len(
                    frame_indices
                ), "The number of global root heading should be match the number of frames"

        self.smooth_root_2d = smooth_root_2d
        self.global_root_heading = global_root_heading
        self.frame_indices = frame_indices

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append this constraint's smooth_root_2d (and optional global_root_heading) to data/index
        dicts."""
        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

        if self.global_root_heading is not None:
            # constraint the global heading
            data_dict["global_root_heading"].append(self.global_root_heading)
            index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "Root2DConstraintSet":
        """Return a new constraint set for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)

        if self.global_root_heading is not None:
            masked_global_root_heading = self.global_root_heading[mask]
        else:
            masked_global_root_heading = None

        return Root2DConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.smooth_root_2d[mask],
            global_root_heading=masked_global_root_heading,
        )

    def get_save_info(self) -> dict:
        """Return a dict suitable for JSON serialization (frame_indices, smooth_root_2d, optional
        global_root_heading)."""
        out = {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "smooth_root_2d": self.smooth_root_2d,
        }
        if self.global_root_heading is not None:
            out["global_root_heading"] = self.global_root_heading
        return out

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Root2DConstraintSet":
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        if self.global_root_heading is not None:
            self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "Root2DConstraintSet":
        """Build a Root2DConstraintSet from a dict (e.g. loaded from JSON)."""
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"

        if "global_root_heading" in dico:
            global_root_heading = torch.tensor(dico["global_root_heading"], device=device)
        else:
            global_root_heading = None

        return cls(
            skeleton,
            frame_indices=torch.tensor(dico["frame_indices"]),
            smooth_root_2d=torch.tensor(dico["smooth_root_2d"], device=device),
            global_root_heading=global_root_heading,
        )


class FullBodyConstraintSet:
    """Constraint set fixing full-body global positions and rotations on given keyframes."""

    name = "fullbody"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        smooth_root_2d: Optional[Tensor] = None,
        to_crop: bool = False,
    ):
        self.skeleton = skeleton
        self.frame_indices = frame_indices

        # if we pass the full smooth root 3D as input
        if smooth_root_2d is not None and smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]

        if to_crop:
            global_joints_positions = global_joints_positions[frame_indices]
            global_joints_rots = global_joints_rots[frame_indices]
            if smooth_root_2d is not None:
                smooth_root_2d = smooth_root_2d[frame_indices]
        else:
            assert len(global_joints_positions) == len(
                frame_indices
            ), "The number of global positions should be match the number of frames"
            assert len(global_joints_rots) == len(
                frame_indices
            ), "The number of global joint rotations should be match the number of frames"

            if smooth_root_2d is not None:
                assert len(smooth_root_2d) == len(
                    frame_indices
                ), "The number of smooth root 2d (if specified) should be match the number of frames"

        if smooth_root_2d is None:
            # substitute the smooth root 2d with the real root
            smooth_root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]

        # root y: from smooth or pelvis is the same
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]

        self.global_joints_positions = global_joints_positions
        self.global_joints_rots = global_joints_rots
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self.smooth_root_2d = smooth_root_2d

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append global positions, smooth root 2D, root y, and global heading to data/index
        dicts."""
        nbjoints = self.skeleton.nbjoints
        indices_lst = create_pairs(
            self.frame_indices,
            torch.arange(nbjoints, device=self.frame_indices.device),
        )
        data_dict["global_joints_positions"].append(
            self.global_joints_positions.reshape(-1, 3)
        )  # flatten the global positions
        index_dict["global_joints_positions"].append(indices_lst)

        # global rotations are not used here

        # as we use smooth root, also constraint the smooth root to get the same full body
        # maybe keep storing the hips offset, if we smooth it ourselves
        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

        # constraint the y pos of the root
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

        # constraint the global heading
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "FullBodyConstraintSet":
        """Return a new FullBodyConstraintSet for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return FullBodyConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.smooth_root_2d[mask],
        )

    def get_save_info(self) -> dict:
        """Return a dict for JSON save: type, frame_indices, local_joints_rot, root_positions, smooth_root_2d."""
        local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
        if isinstance(self.skeleton, SOMASkeleton30):
            local_joints_rot = self.skeleton.to_SOMASkeleton77(local_joints_rot)
        local_joints_rot = matrix_to_axis_angle(local_joints_rot)

        root_positions = self.global_joints_positions[:, self.skeleton.root_idx]
        return {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "smooth_root_2d": self.smooth_root_2d,
        }

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FullBodyConstraintSet":
        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        self.global_joints_positions = _tensor_to(self.global_joints_positions, device, dtype)
        self.global_joints_rots = _tensor_to(self.global_joints_rots, device, dtype)
        self.root_y_pos = _tensor_to(self.root_y_pos, device, dtype)
        self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "FullBodyConstraintSet":
        """Build a FullBodyConstraintSet from a dict (e.g. loaded from JSON)."""
        frame_indices = torch.tensor(dico["frame_indices"])
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"
        local_rot = torch.tensor(dico["local_joints_rot"], device=device)
        local_rot_mats = axis_angle_to_matrix(local_rot)
        local_rot_mats = _convert_constraint_local_rots_to_skeleton(local_rot_mats, skeleton)
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            local_rot_mats,
            torch.tensor(dico["root_positions"], device=device),
        )
        smooth_root_2d = None
        if "smooth_root_2d" in dico:
            smooth_root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=smooth_root_2d,
        )


class EndEffectorConstraintSet:
    """Constraint set fixing selected end-effector positions and rotations on given frames."""

    name = "end-effector"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        smooth_root_2d: Optional[Tensor],
        *,
        joint_names: list[str],
        to_crop: bool = False,
    ) -> None:
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.joint_names = joint_names

        # joint_names are constant for all the frames
        rot_joint_names, pos_joint_names = self.skeleton.expand_joint_names(self.joint_names)
        # indexing works for motion_rep with smooth root only (contains pelvis index)
        self.pos_indices = torch.tensor([self.skeleton.bone_index[jname] for jname in pos_joint_names])
        self.rot_indices = torch.tensor([self.skeleton.bone_index[jname] for jname in rot_joint_names])

        # if we pass the full smooth root 3D as input
        if smooth_root_2d is not None and smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]

        if to_crop:
            global_joints_positions = global_joints_positions[frame_indices]
            global_joints_rots = global_joints_rots[frame_indices]
            if smooth_root_2d is not None:
                smooth_root_2d = smooth_root_2d[frame_indices]
        else:
            assert len(global_joints_positions) == len(
                frame_indices
            ), "The number of global positions should be match the number of frames"
            assert len(global_joints_rots) == len(
                frame_indices
            ), "The number of global joint rotations should be match the number of frames"
            if smooth_root_2d is not None:
                assert len(smooth_root_2d) == len(
                    frame_indices
                ), "The number of smooth root 2d (if specified) should be match the number of frames"

        if smooth_root_2d is None:
            # substitute the smooth root 2d with the real root
            smooth_root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]

        # root y: from smooth or pelvis is the same
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]

        self.global_joints_positions = global_joints_positions
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self.global_joints_rots = global_joints_rots
        self.smooth_root_2d = smooth_root_2d

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append constrained joint positions/rots, smooth root 2D, root y, and heading to
        data/index dicts."""
        crop_frames_indexing = torch.arange(len(self.frame_indices), device=self.frame_indices.device)

        # constraint positions
        pos_indices_real = create_pairs(
            self.frame_indices,
            self.pos_indices,
        )
        pos_indices_crop = create_pairs(
            crop_frames_indexing,
            self.pos_indices,
        )
        data_dict["global_joints_positions"].append(self.global_joints_positions[tuple(pos_indices_crop.T)])
        index_dict["global_joints_positions"].append(pos_indices_real)

        # constraint rotations
        rot_indices_real = create_pairs(
            self.frame_indices,
            self.rot_indices,
        )
        rot_indices_crop = create_pairs(
            crop_frames_indexing,
            self.rot_indices,
        )
        data_dict["global_joints_rots"].append(self.global_joints_rots[tuple(rot_indices_crop.T)])
        index_dict["global_joints_rots"].append(rot_indices_real)

        # as we use smooth root, also constraint the smooth root to get the same full body
        # maybe keep storing the hips offset, if we smooth it ourselves
        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

        # constraint the y pos of the root
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

        # constraint the global heading
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "EndEffectorConstraintSet":
        """Return a new EndEffectorConstraintSet for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)

        cls = type(self)
        kwargs = {}
        if not hasattr(cls, "joint_names"):
            kwargs["joint_names"] = self.joint_names

        return cls(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.smooth_root_2d[mask],
            **kwargs,
        )

    def get_save_info(self) -> dict:
        """Return a dict for JSON save: type, frame_indices, local_joints_rot, root_positions, smooth_root_2d, joint_names."""
        local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
        if isinstance(self.skeleton, SOMASkeleton30):
            local_joints_rot = self.skeleton.to_SOMASkeleton77(local_joints_rot)
        local_joints_rot = matrix_to_axis_angle(local_joints_rot)

        root_positions = self.global_joints_positions[:, self.skeleton.root_idx]
        output = {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "smooth_root_2d": self.smooth_root_2d,
        }
        if not hasattr(self.__class__, "joint_names"):
            # save the joint_names for this base class
            # but not for children
            output["joint_names"] = self.joint_names
        return output

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "EndEffectorConstraintSet":
        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        self.pos_indices = _tensor_to(self.pos_indices, device, dtype)
        self.rot_indices = _tensor_to(self.rot_indices, device, dtype)
        self.root_y_pos = _tensor_to(self.root_y_pos, device, dtype)
        self.global_joints_positions = _tensor_to(self.global_joints_positions, device, dtype)
        self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        self.global_joints_rots = _tensor_to(self.global_joints_rots, device, dtype)
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "EndEffectorConstraintSet":
        """Build an EndEffectorConstraintSet from a dict (e.g. loaded from JSON)."""
        frame_indices = torch.tensor(dico["frame_indices"])
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"
        local_rot = torch.tensor(dico["local_joints_rot"], device=device)
        local_rot_mats = axis_angle_to_matrix(local_rot)
        local_rot_mats = _convert_constraint_local_rots_to_skeleton(local_rot_mats, skeleton)
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            local_rot_mats,
            torch.tensor(dico["root_positions"], device=device),
        )
        smooth_root_2d = None
        if "smooth_root_2d" in dico:
            smooth_root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

        kwargs = {}
        if not hasattr(cls, "joint_names"):
            kwargs["joint_names"] = dico["joint_names"]

        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=smooth_root_2d,
            **kwargs,
        )


class LeftHandConstraintSet(EndEffectorConstraintSet):
    """End-effector constraint for the left hand only."""

    name = "left-hand"
    joint_names: list[str] = ["LeftHand"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class RightHandConstraintSet(EndEffectorConstraintSet):
    """End-effector constraint for the right hand only."""

    name = "right-hand"
    joint_names: list[str] = ["RightHand"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class LeftFootConstraintSet(EndEffectorConstraintSet):
    """End-effector constraint for the left foot only."""

    name = "left-foot"
    joint_names: list[str] = ["LeftFoot"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class RightFootConstraintSet(EndEffectorConstraintSet):
    """End-effector constraint for the right foot only."""

    name = "right-foot"
    joint_names: list[str] = ["RightFoot"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class EEPoseConstraintSet(EndEffectorConstraintSet):
    """G1-only sparse end-effector pose constraint specified directly in world space.

    The input schema defines world-space base-joint poses for hands/feet plus an optional root
    translation/yaw. The loader expands each EE pose into the sparse position/rotation signals the
    model already consumes:
      - hands: base ``*_wrist_yaw_skel`` rotation + positions for base and ``*_hand_roll_skel``
      - feet: base ``*_ankle_roll_skel`` rotation + positions for base and ``*_toe_base``
    """

    name = "ee-pose"
    pose_field_specs = [
        ("left_hand_pose", "LeftHand", "left_wrist_yaw_skel", "left_hand_roll_skel"),
        ("right_hand_pose", "RightHand", "right_wrist_yaw_skel", "right_hand_roll_skel"),
        ("left_foot_pose", "LeftFoot", "left_ankle_roll_skel", "left_toe_base"),
        ("right_foot_pose", "RightFoot", "right_ankle_roll_skel", "right_toe_base"),
    ]

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        root_xyzyaw: Optional[Tensor] = None,
        *,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        left_foot_pose: Optional[Tensor] = None,
        right_foot_pose: Optional[Tensor] = None,
    ) -> None:
        if "g1" not in skeleton.name.lower():
            raise ValueError("ee-pose constraints currently only support the G1 skeleton.")

        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.root_xyzyaw = root_xyzyaw

        n_frames = len(frame_indices)
        self.has_explicit_root_constraint = root_xyzyaw is not None
        if root_xyzyaw is not None:
            if root_xyzyaw.shape[-1] != 4:
                raise ValueError(
                    f"ee-pose root_xyzyaw must have shape [T, 4] ([x, y, z, yaw]), got {tuple(root_xyzyaw.shape)}."
                )
            if len(root_xyzyaw) != n_frames:
                raise ValueError("ee-pose root_xyzyaw length must match frame_indices.")

        raw_pose_fields = {
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "left_foot_pose": left_foot_pose,
            "right_foot_pose": right_foot_pose,
        }
        self.pose_fields = {}
        for field_name, pose in raw_pose_fields.items():
            if pose is None:
                continue
            if pose.shape[-1] != 6:
                raise ValueError(
                    f"ee-pose {field_name} must have shape [T, 6] ([x, y, z, roll, pitch, yaw]),"
                    f" got {tuple(pose.shape)}."
                )
            if len(pose) != n_frames:
                raise ValueError(f"ee-pose {field_name} length must match frame_indices.")
            self.pose_fields[field_name] = pose

        if not self.pose_fields:
            raise ValueError("ee-pose constraints require at least one hand/foot pose field.")

        first_pose = next(iter(self.pose_fields.values()))
        device = first_pose.device
        dtype = first_pose.dtype
        if root_xyzyaw is not None:
            device = root_xyzyaw.device
            dtype = root_xyzyaw.dtype
            root_positions = root_xyzyaw[:, :3]
            root_yaw = root_xyzyaw[:, 3]
            root_rot = angle_to_Y_rotation_matrix(root_yaw).to(dtype=dtype)
        else:
            root_positions = torch.zeros((n_frames, 3), device=device, dtype=dtype)
            root_yaw = torch.zeros((n_frames,), device=device, dtype=dtype)
            root_rot = torch.eye(3, device=device, dtype=dtype).reshape(1, 3, 3).repeat(n_frames, 1, 1)

        neutral_joints = skeleton.neutral_joints.to(device=device, dtype=dtype)
        global_joints_positions = root_positions[:, None, :] + torch.matmul(
            root_rot[:, None],
            neutral_joints[None, :, :, None],
        ).squeeze(-1)

        global_joints_rots = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3).repeat(
            n_frames,
            skeleton.nbjoints,
            1,
            1,
        )
        global_joints_positions[:, skeleton.root_idx] = root_positions
        global_joints_rots[:, skeleton.root_idx] = root_rot

        pos_indices = []
        rot_indices = []
        joint_names = []
        for field_name, joint_group_name, base_joint_name, distal_joint_name in self.pose_field_specs:
            pose = self.pose_fields.get(field_name)
            if pose is None:
                continue

            joint_names.append(joint_group_name)
            base_idx = skeleton.bone_index[base_joint_name]
            distal_idx = skeleton.bone_index[distal_joint_name]

            base_pos = pose[:, :3]
            base_rot = _rpy_to_rotation_matrix(pose[:, 3:]).to(dtype=dtype)
            distal_offset = (neutral_joints[distal_idx] - neutral_joints[base_idx]).view(1, 3, 1)
            distal_pos = base_pos + torch.matmul(base_rot, distal_offset).squeeze(-1)

            global_joints_positions[:, base_idx] = base_pos
            global_joints_positions[:, distal_idx] = distal_pos
            global_joints_rots[:, base_idx] = base_rot
            global_joints_rots[:, distal_idx] = base_rot

            pos_indices.extend([base_idx, distal_idx])
            rot_indices.append(base_idx)

        self.joint_names = joint_names
        self.pos_indices = torch.tensor(pos_indices, dtype=torch.long, device=frame_indices.device)
        self.rot_indices = torch.tensor(rot_indices, dtype=torch.long, device=frame_indices.device)
        self.global_joints_positions = global_joints_positions
        self.global_joints_rots = global_joints_rots
        # Global EE position constraints in Kimodo representation require smooth-root features to be set.
        # If root_xyzyaw is not provided, fall back to a neutral zero-root 2D reference only.
        self.smooth_root_2d = root_positions[:, [0, 2]]
        if self.has_explicit_root_constraint:
            self.root_y_pos = root_positions[:, 1]
            self.global_root_heading = torch.stack([torch.cos(root_yaw), torch.sin(root_yaw)], dim=-1)
        else:
            self.root_y_pos = None
            self.global_root_heading = None

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append sparse EE pose-derived positions/rots plus root signals."""
        crop_frames_indexing = torch.arange(len(self.frame_indices), device=self.frame_indices.device)

        pos_indices_real = create_pairs(self.frame_indices, self.pos_indices)
        pos_indices_crop = create_pairs(crop_frames_indexing, self.pos_indices)
        data_dict.setdefault("global_joints_positions", []).append(
            self.global_joints_positions[tuple(pos_indices_crop.T)]
        )
        index_dict.setdefault("global_joints_positions", []).append(pos_indices_real)

        rot_indices_real = create_pairs(self.frame_indices, self.rot_indices)
        rot_indices_crop = create_pairs(crop_frames_indexing, self.rot_indices)
        data_dict.setdefault("global_joints_rots", []).append(self.global_joints_rots[tuple(rot_indices_crop.T)])
        index_dict.setdefault("global_joints_rots", []).append(rot_indices_real)

        data_dict.setdefault("smooth_root_2d", []).append(self.smooth_root_2d)
        index_dict.setdefault("smooth_root_2d", []).append(self.frame_indices)

        if self.root_y_pos is not None:
            data_dict.setdefault("root_y_pos", []).append(self.root_y_pos)
            index_dict.setdefault("root_y_pos", []).append(self.frame_indices)

        if self.global_root_heading is not None:
            data_dict.setdefault("global_root_heading", []).append(self.global_root_heading)
            index_dict.setdefault("global_root_heading", []).append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "EEPoseConstraintSet":
        """Return a cropped ee-pose constraint set for the frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        pose_fields = {field_name: pose[mask] for field_name, pose in self.pose_fields.items()}
        return EEPoseConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.root_xyzyaw[mask] if self.root_xyzyaw is not None else None,
            **pose_fields,
        )

    def get_save_info(self) -> dict:
        """Return the original ee-pose JSON schema for round-tripping."""
        out = {
            "type": self.name,
            "frame_indices": self.frame_indices,
        }
        if self.root_xyzyaw is not None:
            out["root_xyzyaw"] = self.root_xyzyaw
        out.update(self.pose_fields)
        return out

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "EEPoseConstraintSet":
        self.frame_indices = self.frame_indices.to(device=device) if device is not None else self.frame_indices
        self.pos_indices = self.pos_indices.to(device=device) if device is not None else self.pos_indices
        self.rot_indices = self.rot_indices.to(device=device) if device is not None else self.rot_indices
        self.root_xyzyaw = _tensor_to(self.root_xyzyaw, device, dtype) if self.root_xyzyaw is not None else None
        self.root_y_pos = _tensor_to(self.root_y_pos, device, dtype) if self.root_y_pos is not None else None
        self.global_joints_positions = _tensor_to(self.global_joints_positions, device, dtype)
        self.global_joints_rots = _tensor_to(self.global_joints_rots, device, dtype)
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype) if self.smooth_root_2d is not None else None
        self.global_root_heading = (
            _tensor_to(self.global_root_heading, device, dtype) if self.global_root_heading is not None else None
        )
        self.pose_fields = {field_name: _tensor_to(pose, device, dtype) for field_name, pose in self.pose_fields.items()}
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "EEPoseConstraintSet":
        """Build an EEPoseConstraintSet from a dict (e.g. loaded from JSON)."""
        frame_indices = torch.tensor(dico["frame_indices"], dtype=torch.long)
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"

        pose_fields = {}
        for field_name, *_ in cls.pose_field_specs:
            if field_name in dico:
                pose_fields[field_name] = torch.tensor(dico[field_name], device=device)

        return cls(
            skeleton,
            frame_indices=frame_indices,
            root_xyzyaw=torch.tensor(dico["root_xyzyaw"], device=device) if "root_xyzyaw" in dico else None,
            **pose_fields,
        )


class FootContactConstraintSet:
    """Constraint set fixing foot contact states on given frames.

    The contact vector follows Kimodo's feature layout:
    ``[LeftToe, LeftHeel, RightToe, RightHeel]`` (4 values per frame).
    """

    name = "foot-contact"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        foot_contacts: Tensor,
        to_crop: bool = False,
    ) -> None:
        self.skeleton = skeleton
        self.frame_indices = frame_indices.to(dtype=torch.long)

        if to_crop:
            foot_contacts = foot_contacts[self.frame_indices]
        else:
            assert len(foot_contacts) == len(
                frame_indices
            ), "The number of foot contact rows should match the number of frame indices."

        if foot_contacts.shape[-1] != 4:
            raise ValueError(f"foot_contacts must have last dim = 4, got {tuple(foot_contacts.shape)}.")

        self.foot_contacts = foot_contacts.to(dtype=torch.float32)

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        """Append foot contact values to data/index dicts."""
        data_dict["foot_contacts"].append(self.foot_contacts)
        index_dict["foot_contacts"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "FootContactConstraintSet":
        """Return a new FootContactConstraintSet for the cropped frame range [start, end)."""
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return FootContactConstraintSet(
            self.skeleton,
            frame_indices=self.frame_indices[mask] - start,
            foot_contacts=self.foot_contacts[mask],
        )

    def get_save_info(self) -> dict:
        """Return a dict suitable for JSON serialization."""
        return {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "foot_contacts": self.foot_contacts,
        }

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FootContactConstraintSet":
        self.frame_indices = _tensor_to(self.frame_indices, device, torch.long)
        self.foot_contacts = _tensor_to(self.foot_contacts, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict) -> "FootContactConstraintSet":
        """Build a FootContactConstraintSet from a dict (e.g. loaded from JSON)."""
        device = skeleton.device if hasattr(skeleton, "device") else "cpu"
        return cls(
            skeleton=skeleton,
            frame_indices=torch.tensor(dico["frame_indices"], dtype=torch.long, device=device),
            foot_contacts=torch.tensor(dico["foot_contacts"], dtype=torch.float32, device=device),
        )


TYPE_TO_CLASS = {
    "root2d": Root2DConstraintSet,
    "fullbody": FullBodyConstraintSet,
    "left-hand": LeftHandConstraintSet,
    "right-hand": RightHandConstraintSet,
    "left-foot": LeftFootConstraintSet,
    "right-foot": RightFootConstraintSet,
    "end-effector": EndEffectorConstraintSet,
    "ee-pose": EEPoseConstraintSet,
    "foot-contact": FootContactConstraintSet,
}


def load_constraints_lst(
    path_or_data: str | list,
    skeleton: SkeletonBase,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Load a list of constraints from JSON path or list of dicts.

    Args:
        path_or_data: Path to constraints.json or list of constraint dicts.
        skeleton: Skeleton instance (used for from_dict).
        device: If set, move all constraint tensors and skeleton to this device.
        dtype: If set, cast constraint tensors to this dtype.
    """
    if isinstance(path_or_data, str):
        saved = load_json(path_or_data)
    else:
        saved = path_or_data

    constraints_lst = []
    for el in saved:
        cls = TYPE_TO_CLASS[el["type"]]
        c = cls.from_dict(skeleton, el)
        if device is not None or dtype is not None:
            c.to(device=device, dtype=dtype)
        constraints_lst.append(c)
    return constraints_lst


def save_constraints_lst(path: str, constraints_lst: list) -> list | None:
    """Save a list of constraint sets to a JSON file.

    Returns None if list is empty.
    """
    if not constraints_lst:
        print("The constraints lst is empty. Skip saving")
        return

    to_save = []

    def tensor_to_list(obj):
        """Recursively convert tensors to lists for JSON serialization."""
        if isinstance(obj, Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(v) for v in obj]
        else:
            return obj

    for constraint in constraints_lst:
        constraint_info = constraint.get_save_info()
        # Convert all tensors to lists for JSON serialization
        constraint_info = tensor_to_list(constraint_info)
        to_save.append(constraint_info)

    save_json(path, to_save)
    print(f"Saved constraints to {path}")
    return to_save
