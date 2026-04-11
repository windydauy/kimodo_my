# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer helpers for Kimodo training."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

__all__ = ["AdamAtan2", "build_optimizer"]


class AdamAtan2(torch.optim.Optimizer):
    """Adam-atan2 optimizer.

    This variant keeps Adam's first/second moment tracking but uses:
        update = atan2(m_hat, sqrt(v_hat))
    instead of:
        update = m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=float(lr),
            betas=(float(beta1), float(beta2)),
            weight_decay=float(weight_decay),
            decoupled_weight_decay=bool(decoupled_weight_decay),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            decoupled_wd = group["decoupled_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamAtan2 does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                if wd != 0.0:
                    if decoupled_wd:
                        p.mul_(1.0 - lr * wd)
                    else:
                        grad = grad.add(p, alpha=wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                denom = v_hat.sqrt()

                update = torch.atan2(m_hat, denom)
                p.add_(update, alpha=-lr)

        return loss


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from training config."""
    opt_name = str(cfg.training.get("optimizer", "adam_atan2")).lower().replace("-", "_")
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    betas_cfg = cfg.training.get("betas", [0.9, 0.999])
    betas = (float(betas_cfg[0]), float(betas_cfg[1]))

    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if opt_name in {"adam_atan2", "atan2"}:
        return AdamAtan2(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            decoupled_weight_decay=True,
        )

    raise ValueError(f"Unsupported optimizer: {cfg.training.get('optimizer')!r}")
