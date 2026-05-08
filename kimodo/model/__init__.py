# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo model package: main model class, text encoders, and loading utilities."""

from importlib import import_module

_LAZY_ATTRS = {
    "resolve_target": (".common", "resolve_target"),
    "Kimodo": (".kimodo_model", "Kimodo"),
    "LLM2VecEncoder": (".llm2vec", "LLM2VecEncoder"),
    "load_model": (".load_model", "load_model"),
    "AVAILABLE_MODELS": (".loading", "AVAILABLE_MODELS"),
    "DEFAULT_MODEL": (".loading", "DEFAULT_MODEL"),
    "DEFAULT_TEXT_ENCODER_URL": (".loading", "DEFAULT_TEXT_ENCODER_URL"),
    "MODEL_NAMES": (".loading", "MODEL_NAMES"),
    "load_checkpoint_state_dict": (".loading", "load_checkpoint_state_dict"),
    "TMR": (".tmr", "TMR"),
    "TwostageDenoiser": (".twostage_denoiser", "TwostageDenoiser"),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))

__all__ = [
    "Kimodo",
    "LLM2VecEncoder",
    "TMR",
    "TwostageDenoiser",
    "load_model",
    "load_checkpoint_state_dict",
    "resolve_target",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_ENCODER_URL",
    "MODEL_NAMES",
]
