"""Tiny text encoder used for training smoke tests."""

from __future__ import annotations

import torch


class DummyTextEncoder:
    """Return zero text features with the same shape as LLM2Vec pooled output."""

    def __init__(self, llm_dim: int = 4096) -> None:
        self.llm_dim = int(llm_dim)
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, text: list[str] | str):
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)
        features = torch.zeros((batch_size, 1, self.llm_dim), device=self.device, dtype=torch.float32)
        lengths = [1] * batch_size
        return features, lengths
