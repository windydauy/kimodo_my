# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NPZ dataset utilities for custom G1 motion fine-tuning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.motion_rep.feature_utils import length_to_mask
from kimodo.skeleton import G1Skeleton34

from .dataset import _pad_motion_to
from .g1_csv import KIMODO_TO_MUJOCO_MATRIX, MUJOCO_TO_KIMODO_MATRIX
from .timeline_annotations import PathLike, TimelineAnnotationIndex


def _resample_times(num_frames: int, input_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    source_times = np.arange(num_frames, dtype=np.float64) / float(input_fps)
    if num_frames == 1 or float(input_fps) == float(target_fps):
        return source_times, source_times
    duration = source_times[-1]
    target_count = max(1, int(round(duration * float(target_fps))) + 1)
    target_times = np.linspace(0.0, duration, target_count, dtype=np.float64)
    return source_times, target_times


def resample_root_positions(root_positions: torch.Tensor, input_fps: float, target_fps: float) -> torch.Tensor:
    if root_positions.shape[0] <= 1 or float(input_fps) == float(target_fps):
        return root_positions
    source_times, target_times = _resample_times(root_positions.shape[0], input_fps, target_fps)
    interp = interp1d(source_times, root_positions.detach().cpu().numpy(), axis=0)
    out = interp(target_times).astype(np.float32)
    return torch.as_tensor(out, dtype=root_positions.dtype, device=root_positions.device)


def resample_local_joint_rots(local_joint_rots: torch.Tensor, input_fps: float, target_fps: float) -> torch.Tensor:
    if local_joint_rots.shape[0] <= 1 or float(input_fps) == float(target_fps):
        return local_joint_rots
    source_times, target_times = _resample_times(local_joint_rots.shape[0], input_fps, target_fps)
    rots_np = local_joint_rots.detach().cpu().numpy()
    num_joints = rots_np.shape[1]
    out = np.zeros((len(target_times), num_joints, 3, 3), dtype=np.float32)
    for joint_idx in range(num_joints):
        joint_rots = Rotation.from_matrix(rots_np[:, joint_idx])
        slerp = Slerp(source_times, joint_rots)
        out[:, joint_idx] = slerp(target_times).as_matrix().astype(np.float32)
    return torch.as_tensor(out, dtype=local_joint_rots.dtype, device=local_joint_rots.device)


def resample_motion(
    local_joint_rots: torch.Tensor,
    root_positions: torch.Tensor,
    input_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(input_fps) == float(target_fps):
        return local_joint_rots, root_positions
    return (
        resample_local_joint_rots(local_joint_rots, input_fps=input_fps, target_fps=target_fps),
        resample_root_positions(root_positions, input_fps=input_fps, target_fps=target_fps),
    )


def load_g1_npz_motion(
    npz_path: PathLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Load one custom-motion NPZ and convert robot qpos to Kimodo-ready tensors.

    Assumptions validated on the current dataset:
    - ``qpos[:, :36]`` is the G1 robot state in MuJoCo order
    - ``qpos[:, 36:43]`` is object pose (xyz + quaternion)
    - root translation is MuJoCo ``xyz`` in meters
    - root quaternion is stored scalar-first ``(w, x, y, z)``
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    data = np.load(path, allow_pickle=False)
    if "qpos" not in data.files:
        raise ValueError(f"NPZ file missing qpos: {path}")
    if "fps" not in data.files:
        raise ValueError(f"NPZ file missing fps: {path}")

    qpos = np.asarray(data["qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < 36:
        raise ValueError(f"Expected qpos shape [T, >=36], got {tuple(qpos.shape)} for {path}.")

    qpos_robot = qpos[:, :36]
    qpos_object = qpos[:, 36:43] if qpos.shape[1] >= 43 else np.zeros((qpos.shape[0], 0), dtype=np.float32)

    root_positions_mujoco = qpos_robot[:, :3]
    root_positions = np.einsum("ij,tj->ti", MUJOCO_TO_KIMODO_MATRIX, root_positions_mujoco)

    root_quat_wxyz = qpos_robot[:, 3:7]
    root_rot_source = Rotation.from_quat(root_quat_wxyz, scalar_first=True).as_matrix().astype(np.float32)
    root_rot_mats = np.einsum(
        "ij,tjk,kl->til",
        MUJOCO_TO_KIMODO_MATRIX,
        root_rot_source,
        KIMODO_TO_MUJOCO_MATRIX,
    )

    device_t = torch.device(device) if device is not None else torch.device("cpu")
    qpos_robot_t = torch.as_tensor(qpos_robot, dtype=dtype, device=device_t)
    qpos_object_t = torch.as_tensor(qpos_object, dtype=dtype, device=device_t)
    root_positions_t = torch.as_tensor(root_positions, dtype=dtype, device=device_t)
    root_rot_mats_t = torch.as_tensor(root_rot_mats, dtype=dtype, device=device_t)

    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    num_frames = qpos_robot.shape[0]
    local_joint_rots = torch.eye(3, dtype=dtype, device=device_t).view(1, 1, 1, 3, 3).repeat(
        1, num_frames, skeleton.nbjoints, 1, 1
    )
    joint_dofs = qpos_robot_t[:, 7:36].unsqueeze(0)
    local_joint_rots = converter._joint_dofs_to_local_rot_mats(
        joint_dofs,
        original_local_rot_mats=local_joint_rots,
        device=device_t,
        dtype=dtype,
        use_relative=True,
    )
    local_joint_rots[0, :, 0] = root_rot_mats_t

    return {
        "local_joint_rots": local_joint_rots.squeeze(0),
        "root_positions": root_positions_t,
        "frame_numbers": torch.arange(num_frames, dtype=torch.long, device=device_t),
        "joint_order": tuple(skeleton.bone_order_names),
        "qpos_robot": qpos_robot_t,
        "qpos_object": qpos_object_t,
        "input_fps": int(round(float(np.asarray(data["fps"]).item()))),
        "source_coord_system": "mujoco",
        "root_translation_order": "xyz",
        "root_position_unit": "meters",
    }


class G1NPZTextDataset(Dataset):
    """G1 motion-text dataset backed by MuJoCo-style custom-motion NPZ files."""

    MIN_EVENT_FRAMES: int = 15

    def __init__(
        self,
        npz_root: PathLike,
        motion_rep: Any,
        *,
        max_frames: int,
        to_normalize: bool = True,
        random_crop: bool = True,
        rotate_to_zero_prob: float = 0.0,
        randomize_first_heading_prob: float = 0.0,
        translate_to_zero_prob: float = 0.0,
        timelines_path: Optional[PathLike] = None,
        text_mode: str = "mixed",
        npz_paths: Optional[Sequence[PathLike]] = None,
        annotated_only: bool = True,
        rng_seed: Optional[int] = None,
        input_fps: Optional[int] = None,
        phase2_enabled: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if phase2_enabled:
            raise NotImplementedError("Phase-2 constraints are not supported yet for G1NPZTextDataset.")

        self.npz_root = Path(npz_root)
        self.motion_rep = motion_rep
        self.max_frames = int(max_frames)
        self.to_normalize = bool(to_normalize)
        self.random_crop = bool(random_crop)
        self.rotate_to_zero_prob = float(rotate_to_zero_prob)
        self.randomize_first_heading_prob = float(randomize_first_heading_prob)
        self.translate_to_zero_prob = float(translate_to_zero_prob)
        self.text_mode = str(text_mode)
        self.annotated_only = bool(annotated_only)
        self.rng = random.Random(rng_seed)
        self.input_fps = int(input_fps) if input_fps is not None else None
        self.target_fps = int(getattr(self.motion_rep, "fps", 0)) or None

        self.timeline_index: Optional[TimelineAnnotationIndex] = None
        if timelines_path is not None:
            self.timeline_index = TimelineAnnotationIndex.from_jsonl(timelines_path)

        if npz_paths is None:
            paths = sorted(self.npz_root.rglob("*.npz"))
        else:
            paths = [Path(p) for p in npz_paths]
        if self.timeline_index is not None and self.annotated_only:
            paths = [p for p in paths if self.timeline_index.has(p)]
        if not paths:
            raise FileNotFoundError(f"No NPZ files found under {self.npz_root}")
        self.npz_paths = paths

    def __len__(self) -> int:
        return len(self.npz_paths)

    def _sample_text_and_time_range(self, npz_path: Path) -> Tuple[str, Optional[float], Optional[float]]:
        if self.timeline_index is None:
            return npz_path.stem, None, None
        if self.text_mode == "overview":
            text = self.timeline_index.sample_text(npz_path, mode="overview", rng=self.rng)
            return text, None, None
        if self.text_mode == "event":
            return self._sample_event_with_range(npz_path)
        if self.rng.random() < 0.5:
            text = self.timeline_index.sample_text(npz_path, mode="overview", rng=self.rng)
            return text, None, None
        return self._sample_event_with_range(npz_path)

    def _sample_event_with_range(self, npz_path: Path) -> Tuple[str, Optional[float], Optional[float]]:
        try:
            rec = self.timeline_index.get_record(npz_path)  # type: ignore[union-attr]
        except KeyError:
            return npz_path.stem, None, None

        events = rec["events"]
        if not events:
            overview = rec["overview_description"].strip()
            return (overview if overview else npz_path.stem), None, None

        event = self.rng.choice(events)
        start_t = float(event["start_time"])
        end_t = float(event["end_time"])
        current_fps = self.input_fps if self.input_fps is not None else float(self.target_fps or 30)
        event_frames = int((end_t - start_t) * float(current_fps))
        if event_frames < self.MIN_EVENT_FRAMES:
            overview = rec["overview_description"].strip()
            return (overview if overview else str(event["description"])), None, None
        return str(event["description"]), start_t, end_t

    def _crop_motion_by_time(
        self,
        local_joint_rots: torch.Tensor,
        root_positions: torch.Tensor,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_frame = max(0, int(np.floor(start_time * float(fps))))
        end_frame = min(local_joint_rots.shape[0], int(np.ceil(end_time * float(fps))))
        if end_frame <= start_frame:
            return local_joint_rots, root_positions
        return local_joint_rots[start_frame:end_frame], root_positions[start_frame:end_frame]

    def _encode_motion(self, local_joint_rots: torch.Tensor, root_positions: torch.Tensor) -> torch.Tensor:
        t = int(local_joint_rots.shape[0])
        lengths = torch.tensor([t], dtype=torch.long, device=local_joint_rots.device)
        features = self.motion_rep(
            local_joint_rots.unsqueeze(0),
            root_positions.unsqueeze(0),
            to_normalize=False,
            lengths=lengths,
        )
        return features.squeeze(0)

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(0)
        if self.randomize_first_heading_prob > 0.0 and self.rng.random() < self.randomize_first_heading_prob:
            x = self.motion_rep.randomize_first_heading(x)
        elif self.rotate_to_zero_prob > 0.0 and self.rng.random() < self.rotate_to_zero_prob:
            x = self.motion_rep.rotate_to_zero(x)
        if self.translate_to_zero_prob > 0.0 and self.rng.random() < self.translate_to_zero_prob:
            x = self.motion_rep.translate_2d_to_zero(x)
        return x.squeeze(0)

    def _crop_or_pad(self, features: torch.Tensor, start: Optional[int] = None) -> tuple[torch.Tensor, int, int]:
        t = int(features.shape[0])
        if t > self.max_frames:
            if start is None:
                start = self.rng.randint(0, t - self.max_frames) if self.random_crop else 0
            end = start + self.max_frames
            return features[start:end], self.max_frames, start
        return _pad_motion_to(features, self.max_frames), t, 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        npz_path = self.npz_paths[idx]
        motion_data = load_g1_npz_motion(npz_path)
        local_joint_rots = motion_data["local_joint_rots"]
        root_positions = motion_data["root_positions"]
        clip_fps = float(self.input_fps or motion_data["input_fps"])

        text, event_start, event_end = self._sample_text_and_time_range(npz_path)
        if event_start is not None and event_end is not None:
            local_joint_rots, root_positions = self._crop_motion_by_time(
                local_joint_rots,
                root_positions,
                event_start,
                event_end,
                clip_fps,
            )

        if self.target_fps is not None and clip_fps != float(self.target_fps):
            local_joint_rots, root_positions = resample_motion(
                local_joint_rots,
                root_positions,
                input_fps=clip_fps,
                target_fps=float(self.target_fps),
            )

        features_unnorm = self._encode_motion(local_joint_rots, root_positions)
        features_unnorm = self._augment_features(features_unnorm)
        features = features_unnorm
        if self.to_normalize:
            features = self.motion_rep.normalize(features.unsqueeze(0)).squeeze(0)

        motion, length, frame_start = self._crop_or_pad(features)
        pad_mask = length_to_mask([length], max_len=self.max_frames, device=motion.device)[0]

        return {
            "motion": motion,
            "length": torch.tensor(length, dtype=torch.long),
            "pad_mask": pad_mask,
            "text": text,
            "csv_path": str(npz_path),
            "clip_name": npz_path.stem,
            "frame_start": torch.tensor(frame_start, dtype=torch.long),
            "frame_end": torch.tensor(frame_start + length, dtype=torch.long),
            "observed_motion": None,
            "motion_mask": None,
            "keyframe_indices": None,
            "constraint_mode": None,
            "constraints": None,
        }
