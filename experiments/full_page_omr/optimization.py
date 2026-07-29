import copy
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from transformers.optimization import get_wsd_schedule


OPTIMIZER_PROTOCOL = "adamw_wsd_v1"
SAMPLES_SEEN_CHECKPOINT_KEY = "full_page_omr_samples_seen"


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
        "optimizer_implementation": "torch.optim.AdamW",
        "torch_version": str(torch.__version__),
        "optimizer_protocol": config.protocol,
        "optimizer_betas": list(config.betas),
        "optimizer_eps": config.eps,
        "optimizer_amsgrad": config.amsgrad,
        "optimizer_groups": [
            {
                "name": group["name"],
                "parameter_tensor_count": len(group["params"]),
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


def same_typed_value(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_typed_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            same_typed_value(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


_MUTABLE_RESUME_SNAPSHOT_FIELDS = (
    ("scheduler", "max_steps"),
    ("scheduler", "stable_steps"),
    ("scheduler", "decay_steps"),
    ("scheduler", "min_lr_ratio"),
    ("trainer", "max_steps"),
    ("validation", "every_n_epochs"),
    ("validation", "first_epoch"),
    ("validation", "expected_count"),
    ("checkpointing", "every_n_epochs"),
)


def _without_mutable_resume_fields(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict):
        raise ValueError("protocol_snapshot must be a dictionary")
    normalized = copy.deepcopy(snapshot)
    for section_name, field_name in _MUTABLE_RESUME_SNAPSHOT_FIELDS:
        section = normalized.get(section_name)
        if not isinstance(section, dict) or field_name not in section:
            raise ValueError(
                "checkpoint protocol snapshot is missing "
                f"{section_name}.{field_name}"
            )
        del section[field_name]
    return normalized


def validate_resume_protocol_snapshots(saved_snapshot: dict,
                                       expected_snapshot: dict) -> None:
    """Allow optimizer-independent schedule and cadence fields to differ."""
    saved_invariants = _without_mutable_resume_fields(saved_snapshot)
    expected_invariants = _without_mutable_resume_fields(expected_snapshot)
    if not same_typed_value(saved_invariants, expected_invariants):
        raise ValueError(
            "checkpoint protocol snapshot mismatch outside mutable resume fields: "
            f"saved={saved_snapshot!r}, expected={expected_snapshot!r}"
        )


def _source_wsd_config(saved_snapshot: dict,
                       target_config: AdamWWSDConfig) -> AdamWWSDConfig:
    scheduler = saved_snapshot.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("checkpoint protocol snapshot is missing scheduler metadata")
    try:
        return replace(
            target_config,
            max_steps=scheduler["max_steps"],
            warmup_steps=scheduler["warmup_steps"],
            decay_steps=scheduler["decay_steps"],
            warmup_type=scheduler["warmup_type"],
            decay_type=scheduler["decay_type"],
            min_lr_ratio=scheduler["min_lr_ratio"],
            num_cycles=scheduler["num_cycles"],
        )
    except KeyError as exc:
        raise ValueError(
            f"checkpoint protocol snapshot is missing scheduler.{exc.args[0]}"
        ) from exc


def prepare_adamw_wsd_resume_state(checkpoint: dict, model: nn.Module,
                                   target_config: AdamWWSDConfig,
                                   expected_protocol_snapshot: dict,
                                   *, mutate: bool) -> None:
    """Validate the saved schedule, then optionally retarget it in place."""
    if not isinstance(checkpoint, dict):
        raise ValueError("full resume checkpoint must be a dictionary")
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyper_parameters, dict):
        raise ValueError("checkpoint hyper_parameters must be a dictionary")
    saved_snapshot = hyper_parameters.get("protocol_snapshot")
    if saved_snapshot is None:
        raise ValueError(
            "checkpoint is missing protocol_snapshot evidence required for full resume"
        )
    validate_resume_protocol_snapshots(
        saved_snapshot,
        expected_protocol_snapshot,
    )

    source_config = _source_wsd_config(saved_snapshot, target_config)
    validate_adamw_wsd_resume_state(checkpoint, model, source_config)
    if not mutate:
        return

    global_step = checkpoint["global_step"]
    if global_step >= target_config.max_steps:
        raise ValueError(
            f"max_steps ({target_config.max_steps}) must exceed resumed checkpoint "
            f"global_step ({global_step})"
        )

    target_optimizer = build_adamw(model, target_config)
    target_scheduler = build_wsd_scheduler(target_optimizer, target_config)
    target_base_lrs = target_scheduler.state_dict()["base_lrs"]
    target_multiplier = target_scheduler.lr_lambdas[0](global_step)
    target_current_lrs = [
        base_lr * target_multiplier for base_lr in target_base_lrs
    ]
    saved_groups = checkpoint["optimizer_states"][0]["param_groups"]
    for saved_group, current_lr in zip(saved_groups, target_current_lrs):
        saved_group["lr"] = current_lr

    target_scheduler_state = target_scheduler.state_dict()
    target_scheduler_state.update({
        "base_lrs": target_base_lrs,
        "last_epoch": global_step,
        "_step_count": global_step + 1,
        "_last_lr": target_current_lrs,
    })
    saved_scheduler_state = checkpoint["lr_schedulers"][0]
    saved_scheduler_state.clear()
    saved_scheduler_state.update(target_scheduler_state)


def validate_adamw_wsd_resume_state(checkpoint: dict, model: nn.Module,
                                    config: AdamWWSDConfig) -> None:
    """Fail before Lightning can restore inconsistent AdamW/WSD state."""
    if not isinstance(checkpoint, dict):
        raise ValueError("full resume checkpoint must be a dictionary")
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("full resume requires a non-negative checkpoint global_step")
    samples_seen = checkpoint.get(SAMPLES_SEEN_CHECKPOINT_KEY)
    if isinstance(samples_seen, bool) or not isinstance(samples_seen, int) or samples_seen < 0:
        raise ValueError(
            f"full resume requires a non-negative {SAMPLES_SEEN_CHECKPOINT_KEY}"
        )

    optimizer_states = checkpoint.get("optimizer_states")
    scheduler_states = checkpoint.get("lr_schedulers")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise ValueError("full resume requires exactly one AdamW optimizer state")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1:
        raise ValueError("full resume requires exactly one WSD scheduler state")
    optimizer_state = optimizer_states[0]
    scheduler_state = scheduler_states[0]
    if not isinstance(optimizer_state, dict):
        raise ValueError("full resume requires exactly one AdamW optimizer state")
    if not isinstance(scheduler_state, dict):
        raise ValueError("full resume requires exactly one WSD scheduler state")

    expected_optimizer = build_adamw(model, config)
    expected_scheduler = build_wsd_scheduler(expected_optimizer, config)
    expected_optimizer_state = expected_optimizer.state_dict()
    expected_groups = expected_optimizer_state["param_groups"]
    saved_groups = optimizer_state.get("param_groups")
    if (
        not isinstance(saved_groups, list)
        or len(saved_groups) != len(expected_groups)
        or not all(isinstance(group, dict) for group in saved_groups)
    ):
        raise ValueError(
            "checkpoint AdamW parameter-group mismatch: "
            f"saved={saved_groups!r}"
        )

    expected_base_lrs = expected_scheduler.state_dict()["base_lrs"]
    expected_multiplier = expected_scheduler.lr_lambdas[0](global_step)
    expected_current_lrs = [
        base_lr * expected_multiplier for base_lr in expected_base_lrs
    ]
    for index, (saved_group, expected_group, expected_lr) in enumerate(
        zip(saved_groups, expected_groups, expected_current_lrs)
    ):
        if saved_group.keys() != expected_group.keys():
            raise ValueError(
                "checkpoint AdamW parameter-group mismatch: "
                f"group {index} keys differ"
            )
        if not same_typed_value(saved_group.get("params"), expected_group["params"]):
            raise ValueError(
                "checkpoint AdamW parameter-group mismatch: "
                f"group {index} parameter order/count differs"
            )
        for key, expected_value in expected_group.items():
            if key == "params":
                continue
            if key == "lr":
                expected_value = expected_lr
            if not same_typed_value(saved_group[key], expected_value):
                raise ValueError(
                    "checkpoint AdamW parameter-group mismatch: "
                    f"group {index} field {key!r} is {saved_group[key]!r}, "
                    f"expected {expected_value!r}"
                )

    expected_scheduler_state = expected_scheduler.state_dict()
    if scheduler_state.keys() != expected_scheduler_state.keys():
        raise ValueError("checkpoint WSD scheduler state mismatch: keys differ")
    dynamic_expected = {
        "base_lrs": expected_base_lrs,
        "last_epoch": global_step,
        "_step_count": global_step + 1,
        "_last_lr": expected_current_lrs,
    }
    for key, expected_value in expected_scheduler_state.items():
        if key in dynamic_expected:
            expected_value = dynamic_expected[key]
        if not same_typed_value(scheduler_state[key], expected_value):
            raise ValueError(
                "checkpoint WSD scheduler state mismatch: "
                f"{key}={scheduler_state[key]!r}, expected={expected_value!r} "
                f"at global_step={global_step}"
            )
