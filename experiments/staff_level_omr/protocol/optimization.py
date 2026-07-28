"""Fixed optimizer construction for the staff OMR v2 protocol."""

from __future__ import annotations

import math

import torch
from torch import nn

from .errors import ProtocolError


def optimizer_contract(learning_rate: float) -> dict[str, object]:
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0
    ):
        raise ProtocolError("learning_rate must be a finite positive number")
    return {
        "type": "torch.optim.Adam",
        "learning_rate": float(learning_rate),
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "scheduler": "none",
        "parameter_groups": 1,
        "parameter_order": "full_name_utf8_ascending",
        "requires_grad_only": True,
    }


def optimizer_parameter_names(model: nn.Module) -> list[str]:
    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names.sort(key=lambda value: value.encode("utf-8"))
    if not names:
        raise ProtocolError("model has no trainable parameters")
    return names


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> tuple[torch.optim.Adam, list[str]]:
    contract = optimizer_contract(learning_rate)
    named = dict(model.named_parameters())
    names = optimizer_parameter_names(model)
    parameters = [named[name] for name in names]
    optimizer = torch.optim.Adam(
        [
            {
                "params": parameters,
                "lr": contract["learning_rate"],
                "betas": tuple(contract["betas"]),
                "eps": contract["eps"],
                "weight_decay": contract["weight_decay"],
            }
        ]
    )
    return optimizer, names
