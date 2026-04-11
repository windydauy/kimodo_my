# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Losses for Kimodo G1 fine-tuning.

Implements the 7-term objective described in the training plan:
1. smooth_root_pos
2. global_root_heading
3. local_joints_positions
4. velocities
5. global_rot_data
6. foot_contacts
7. FK consistency
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from kimodo.geometry import cont6d_to_matrix
from kimodo.skeleton.kinematics import fk
from kimodo.skeleton.transforms import global_rots_to_local_rots

__all__ = [
    "DEFAULT_GAMMAS",
    "LOSS_NAMES",
    "KimodoLoss",
    "compute_kimodo_loss",
    "masked_l1_loss",
]


DEFAULT_GAMMAS: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.5, 1.0, 0.1, 1.0)
LOSS_NAMES: Tuple[str, ...] = (
    "smooth_root_pos",
    "global_root_heading",
    "local_joints_positions",
    "velocities",
    "global_rot_data",
    "foot_contacts",
    "fk_consistency",
)


def _expand_mask_to(x: Tensor, pad_mask: Tensor) -> Tensor:
    """Expand ``[B, T]`` pad mask to match ``x`` shape."""
    mask = pad_mask
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(x)


def masked_l1_loss(pred: Tensor, target: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
    """L1 loss with optional sequence mask.

    Args:
        pred: Predicted tensor.
        target: Ground-truth tensor with same shape as ``pred``.
        pad_mask: Optional bool mask with shape ``[B, T]`` where ``True`` means
            valid timestep and ``False`` means padded timestep.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have same shape. Got {pred.shape} vs {target.shape}.")

    diff = torch.abs(pred - target)
    if pad_mask is None:
        return diff.mean()

    if pad_mask.ndim != 2:
        raise ValueError(f"pad_mask must have shape [B, T], got ndim={pad_mask.ndim}.")
    if pad_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"pad_mask shape must match first two dims of pred. "
            f"Got pad_mask={pad_mask.shape}, pred[:2]={pred.shape[:2]}."
        )

    mask = _expand_mask_to(diff, pad_mask.to(device=diff.device, dtype=torch.bool))
    mask_f = mask.to(dtype=diff.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (diff * mask_f).sum() / denom


def _validate_gammas(gammas: Sequence[float]) -> Tuple[float, ...]:
    gammas = tuple(float(g) for g in gammas)
    if len(gammas) != 7:
        raise ValueError(f"gammas must have length 7, got {len(gammas)}.")
    return gammas


def compute_kimodo_loss(
    pred_x0: Tensor,
    gt_x0: Tensor,
    motion_rep,
    *,
    pad_mask: Optional[Tensor] = None,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    input_is_normalized: bool = False,
) -> Dict[str, Tensor]:
    """Compute the 7-term Kimodo training loss.

    Args:
        pred_x0: Predicted clean features ``[B, T, D]``.
        gt_x0: Ground-truth clean features ``[B, T, D]``.
        motion_rep: Motion representation object with ``slice_dict``, ``nbjoints``,
            ``skeleton``, ``get_root_pos()``, and optional ``unnormalize()``.
        pad_mask: Optional validity mask ``[B, T]`` (True = valid).
        gammas: 7 scalar weights for the seven terms.
        input_is_normalized: Whether ``pred_x0`` and ``gt_x0`` are normalized.
            If True, both tensors are unnormalized before loss computation.

    Returns:
        Dict with total loss and per-term losses (raw + weighted).
    """
    if pred_x0.shape != gt_x0.shape:
        raise ValueError(f"pred_x0 and gt_x0 must have same shape. Got {pred_x0.shape} vs {gt_x0.shape}.")
    if pred_x0.ndim != 3:
        raise ValueError(f"Expected [B, T, D] tensors, got ndim={pred_x0.ndim}.")

    gammas = _validate_gammas(gammas)

    pred = pred_x0
    gt = gt_x0
    if input_is_normalized:
        if not hasattr(motion_rep, "unnormalize"):
            raise TypeError("motion_rep must provide unnormalize() when input_is_normalized=True.")
        pred = motion_rep.unnormalize(pred)
        gt = motion_rep.unnormalize(gt)

    slice_dict = motion_rep.slice_dict

    # First 6 feature-space terms.
    l_smooth_root = masked_l1_loss(
        pred[..., slice_dict["smooth_root_pos"]],
        gt[..., slice_dict["smooth_root_pos"]],
        pad_mask,
    )
    l_heading = masked_l1_loss(
        pred[..., slice_dict["global_root_heading"]],
        gt[..., slice_dict["global_root_heading"]],
        pad_mask,
    )
    l_local_pos = masked_l1_loss(
        pred[..., slice_dict["local_joints_positions"]],
        gt[..., slice_dict["local_joints_positions"]],
        pad_mask,
    )
    l_vel = masked_l1_loss(
        pred[..., slice_dict["velocities"]],
        gt[..., slice_dict["velocities"]],
        pad_mask,
    )
    l_global_rot = masked_l1_loss(
        pred[..., slice_dict["global_rot_data"]],
        gt[..., slice_dict["global_rot_data"]],
        pad_mask,
    )
    l_foot = masked_l1_loss(
        pred[..., slice_dict["foot_contacts"]],
        gt[..., slice_dict["foot_contacts"]],
        pad_mask,
    )

    # 7th term: FK consistency.
    # Following the training description:
    # - use predicted global joint rotations -> FK -> predicted joint positions
    # - compare against GT joint positions directly
    bsz, nframes, _ = pred.shape
    nbjoints = motion_rep.nbjoints
    skel = motion_rep.skeleton

    pred_rot6d = pred[..., slice_dict["global_rot_data"]].reshape(bsz, nframes, nbjoints, 6)
    pred_global_rots = cont6d_to_matrix(pred_rot6d)
    pred_local_rots = global_rots_to_local_rots(pred_global_rots, skel)

    # FK needs root translation. Use GT root so this term focuses on angle->position consistency.
    gt_root_pos = motion_rep.get_root_pos(gt, fallback_to_smooth=False)
    _, pred_fk_pos, _ = fk(pred_local_rots, gt_root_pos, skel)

    # GT joint position target from feature-space positions:
    # local_joints_positions stores x/z relative to smooth_root and absolute y.
    gt_local_pos = gt[..., slice_dict["local_joints_positions"]].reshape(bsz, nframes, nbjoints, 3)
    gt_smooth_root = gt[..., slice_dict["smooth_root_pos"]]
    gt_joint_pos = gt_local_pos.clone()
    gt_joint_pos[..., 0] += gt_smooth_root[..., None, 0]
    gt_joint_pos[..., 2] += gt_smooth_root[..., None, 2]

    l_fk = masked_l1_loss(pred_fk_pos, gt_joint_pos, pad_mask)

    raw_terms = {
        "smooth_root_pos": l_smooth_root,
        "global_root_heading": l_heading,
        "local_joints_positions": l_local_pos,
        "velocities": l_vel,
        "global_rot_data": l_global_rot,
        "foot_contacts": l_foot,
        "fk_consistency": l_fk,
    }

    weighted_terms = {
        f"weighted_{name}": gamma * raw_terms[name]
        for gamma, name in zip(gammas, LOSS_NAMES)
    }
    total = sum(weighted_terms.values())

    return {
        "total": total,
        **raw_terms,
        **weighted_terms,
    }


class KimodoLoss(nn.Module):
    """Module wrapper for ``compute_kimodo_loss``."""

    def __init__(
        self,
        motion_rep,
        gammas: Sequence[float] = DEFAULT_GAMMAS,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()
        self.motion_rep = motion_rep
        self.gammas = _validate_gammas(gammas)
        self.input_is_normalized = bool(input_is_normalized)

    def forward(
        self,
        pred_x0: Tensor,
        gt_x0: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return compute_kimodo_loss(
            pred_x0=pred_x0,
            gt_x0=gt_x0,
            motion_rep=self.motion_rep,
            pad_mask=pad_mask,
            gammas=self.gammas,
            input_is_normalized=self.input_is_normalized,
        )
