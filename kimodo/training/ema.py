# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exponential moving average (EMA) utility for model parameters."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class EMA:
    """Track an exponential moving average over trainable model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module) -> None:
        self.shadow = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow params from current model params."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model params with EMA shadow params."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow:
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Backup current model params."""
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore model params from backup."""
        if not self.backup:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, object], device: torch.device | str | None = None) -> None:
        self.decay = float(state_dict["decay"])
        loaded_shadow = state_dict["shadow"]
        if device is None:
            self.shadow = {name: tensor.clone() for name, tensor in loaded_shadow.items()}
        else:
            self.shadow = {name: tensor.to(device=device) for name, tensor in loaded_shadow.items()}
        self.backup = {}
