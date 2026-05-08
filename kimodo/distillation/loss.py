# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distillation losses for Kimodo student training.

This module implements a teacher-guided + GT-regularized objective:
- 7-term Kimodo loss against teacher prediction (teacher-dominant)
- 7-term Kimodo loss against dataset GT (GT-auxiliary)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from kimodo.training.loss import DEFAULT_GAMMAS, LOSS_NAMES, compute_kimodo_loss

__all__ = ["DistillationKimodoLoss"]


def _validate_weight(name: str, x: float) -> float:
    x = float(x)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}.")
    return x


class DistillationKimodoLoss(nn.Module):
    """Weighted 7+7 distillation loss.

    The "7+7" means:
    - 7 Kimodo terms: student vs teacher (distill branch)
    - 7 Kimodo terms: student vs GT (aux branch)

    Final scalar:
        total = teacher_weight * L_distill + gt_weight * L_gt

    where each branch L_* already aggregates its own 7 weighted terms.
    """

    def __init__(
        self,
        motion_rep,
        *,
        teacher_gammas: Sequence[float] = DEFAULT_GAMMAS,
        gt_gammas: Sequence[float] = DEFAULT_GAMMAS,
        teacher_weight: float = 0.8,
        gt_weight: float = 0.2,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()
        self.motion_rep = motion_rep
        self.teacher_gammas = tuple(float(x) for x in teacher_gammas)
        self.gt_gammas = tuple(float(x) for x in gt_gammas)
        self.teacher_weight = _validate_weight("teacher_weight", teacher_weight)
        self.gt_weight = _validate_weight("gt_weight", gt_weight)
        self.input_is_normalized = bool(input_is_normalized)
        if self.teacher_weight == 0.0 and self.gt_weight == 0.0:
            raise ValueError("teacher_weight and gt_weight cannot both be zero.")

    def forward(
        self,
        *,
        pred_x0: Tensor,
        teacher_x0: Tensor,
        gt_x0: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if pred_x0.shape != teacher_x0.shape or pred_x0.shape != gt_x0.shape:
            raise ValueError(
                "pred_x0, teacher_x0, gt_x0 must have the same shape. "
                f"Got pred={pred_x0.shape}, teacher={teacher_x0.shape}, gt={gt_x0.shape}."
            )

        teacher_terms = compute_kimodo_loss(
            pred_x0=pred_x0,
            gt_x0=teacher_x0,
            motion_rep=self.motion_rep,
            pad_mask=pad_mask,
            gammas=self.teacher_gammas,
            input_is_normalized=self.input_is_normalized,
        )
        gt_terms = compute_kimodo_loss(
            pred_x0=pred_x0,
            gt_x0=gt_x0,
            motion_rep=self.motion_rep,
            pad_mask=pad_mask,
            gammas=self.gt_gammas,
            input_is_normalized=self.input_is_normalized,
        )

        total = (self.teacher_weight * teacher_terms["total"]) + (self.gt_weight * gt_terms["total"])

        out: Dict[str, Tensor] = {
            "total": total,
            "loss_teacher_total": teacher_terms["total"],
            "loss_gt_total": gt_terms["total"],
        }

        for name in LOSS_NAMES:
            out[f"teacher_{name}"] = teacher_terms[name]
            out[f"teacher_weighted_{name}"] = teacher_terms[f"weighted_{name}"]
            out[f"gt_{name}"] = gt_terms[name]
            out[f"gt_weighted_{name}"] = gt_terms[f"weighted_{name}"]

        return out
