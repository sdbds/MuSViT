from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers.optimization import get_wsd_schedule


OPTIMIZER_PROTOCOL = "adamw_wsd_v1"


@dataclass(frozen=True)
class AdamWWSDConfig:
    protocol: str = OPTIMIZER_PROTOCOL
    task_learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    max_steps: int = 4_000_000
    warmup_steps: int = 10_000
    decay_steps: int = 400_000
    warmup_type: str = "linear"
    decay_type: str = "cosine"
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5

    def __post_init__(self):
        if self.protocol != OPTIMIZER_PROTOCOL:
            raise ValueError(f"protocol must be {OPTIMIZER_PROTOCOL}")
        if self.amsgrad is not False:
            raise ValueError("amsgrad must be False for the locked protocol")
        if (
            isinstance(self.num_cycles, bool)
            or not isinstance(self.num_cycles, (int, float))
            or self.num_cycles != 0.5
        ):
            raise ValueError("num_cycles must be 0.5 for the locked protocol")
        for name in ("task_learning_rate", "encoder_learning_rate", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        for name in ("max_steps", "warmup_steps", "decay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_steps <= 0 or self.warmup_steps + self.decay_steps >= self.max_steps:
            raise ValueError("WSD phases must leave a positive stable interval")
        if self.warmup_type not in {"linear", "cosine", "1-sqrt"}:
            raise ValueError("unsupported WSD warmup_type")
        if self.decay_type not in {"linear", "cosine", "1-sqrt"}:
            raise ValueError("unsupported WSD decay_type")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between 0 and 1")

    @property
    def stable_steps(self) -> int:
        return self.max_steps - self.warmup_steps - self.decay_steps

    def to_metadata(self) -> dict:
        metadata = asdict(self)
        metadata["betas"] = list(self.betas)
        metadata["stable_steps"] = self.stable_steps
        return metadata


def _no_decay_parameter_ids(model: nn.Module) -> set[int]:
    result = set()
    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if "bias" in parameter_name.lower() or isinstance(module, nn.LayerNorm):
                result.add(id(parameter))
    return result


def build_adamw_parameter_groups(model: nn.Module, config: AdamWWSDConfig) -> list[dict]:
    if not hasattr(model, "encoder"):
        raise ValueError("full-page OMR model must expose an encoder submodule")
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    no_decay_ids = _no_decay_parameter_ids(model)
    buckets = {
        name: []
        for name in (
            "encoder_decay",
            "encoder_no_decay",
            "task_decay",
            "task_no_decay",
        )
    }
    for parameter in model.parameters():
        owner = "encoder" if id(parameter) in encoder_ids else "task"
        decay = "no_decay" if id(parameter) in no_decay_ids else "decay"
        buckets[f"{owner}_{decay}"].append(parameter)
    empty = [name for name, parameters in buckets.items() if not parameters]
    if empty:
        raise ValueError(f"empty AdamW parameter groups: {empty}")
    groups = []
    for name, parameters in buckets.items():
        groups.append(
            {
                "name": name,
                "params": parameters,
                "lr": (
                    config.encoder_learning_rate
                    if name.startswith("encoder_")
                    else config.task_learning_rate
                ),
                "weight_decay": 0.0 if name.endswith("_no_decay") else config.weight_decay,
            }
        )
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    model_ids = [id(parameter) for parameter in model.parameters()]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(model_ids):
        raise ValueError("AdamW parameter groups must partition model parameters exactly once")
    return groups


def build_adamw(model: nn.Module, config: AdamWWSDConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        build_adamw_parameter_groups(model, config),
        betas=config.betas,
        eps=config.eps,
        amsgrad=config.amsgrad,
    )


def build_wsd_scheduler(optimizer, config: AdamWWSDConfig):
    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_decay_steps=config.decay_steps,
        num_training_steps=config.max_steps,
        warmup_type=config.warmup_type,
        decay_type=config.decay_type,
        min_lr_ratio=config.min_lr_ratio,
        num_cycles=config.num_cycles,
    )


def optimizer_protocol_metadata(model: nn.Module, config: AdamWWSDConfig) -> dict:
    groups = build_adamw_parameter_groups(model, config)
    return {
        "optimizer": "AdamW",
        "optimizer_protocol": config.protocol,
        "optimizer_betas": list(config.betas),
        "optimizer_eps": config.eps,
        "optimizer_amsgrad": config.amsgrad,
        "optimizer_groups": [
            {
                "name": group["name"],
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
                "learning_rate": group["lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in groups
        ],
        "scheduler": "transformers.get_wsd_schedule",
        "scheduler_interval": "step",
        "scheduler_frequency": 1,
        "wsd_max_steps": config.max_steps,
        "wsd_warmup_steps": config.warmup_steps,
        "wsd_stable_steps": config.stable_steps,
        "wsd_decay_steps": config.decay_steps,
        "wsd_warmup_type": config.warmup_type,
        "wsd_decay_type": config.decay_type,
        "wsd_min_lr_ratio": config.min_lr_ratio,
        "wsd_num_cycles": config.num_cycles,
        "task_learning_rate": config.task_learning_rate,
        "encoder_learning_rate": config.encoder_learning_rate,
        "weight_decay": config.weight_decay,
    }
