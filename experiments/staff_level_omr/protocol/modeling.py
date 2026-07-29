"""Auditable staff-level OMR task head and PEFT construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as torch_functional

from .backbone import BackboneMetadata
from .errors import ProtocolError
from .geometry import GeometryPlan, build_geometry_plan, extract_spatial_grid
from .seeding import reset_initialization_rng


TASK_HEAD_SCHEMA = "staff_omr_task_head_v1"
LORA_CONTRACT: dict[str, object] = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.1,
    "bias": "none",
    "target_modules": ["query", "key", "value"],
    "use_rslora": True,
}


def task_head_contract(
    metadata: BackboneMetadata,
    num_classes: int,
) -> dict[str, object]:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ProtocolError("num_classes must be an integer")
    if num_classes < 2:
        raise ProtocolError("num_classes must include blank and at least one token")
    return {
        "schema_version": TASK_HEAD_SCHEMA,
        "spatial_order": "rows_then_columns",
        "input_dropout": {
            "p": 0.25,
            "position": "before_projection",
        },
        "projection": {
            "in_features": metadata.hidden_size,
            "out_features": 256,
            "bias": False,
        },
        "row_pool": {
            "operation": "mean",
            "axis": "rows",
            "position": "after_projection",
        },
        "rnn": {
            "type": "LSTM",
            "input_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "bias": True,
            "batch_first": True,
            "dropout": 0.5,
            "bidirectional": True,
            "proj_size": 0,
            "initial_state": "zeros_same_dtype_and_device_as_input",
        },
        "classifier": {
            "in_features": 512,
            "out_features": num_classes,
            "bias": True,
        },
        "output": {"log_softmax_dim": -1},
        "decoder": {
            "type": "greedy_ctc",
            "collapse_repeats": True,
            "remove_blank_id": 0,
        },
    }


class StaffOMRModel(nn.Module):
    """ViT spatial tokens followed by the frozen v2 CTC task-head schema."""

    def __init__(
        self,
        backbone: nn.Module,
        plan: GeometryPlan,
        *,
        num_classes: int,
    ):
        super().__init__()
        task_head_contract(plan.metadata, num_classes)
        self.backbone = backbone
        self._backbone_is_frozen = not any(
            parameter.requires_grad for parameter in backbone.parameters()
        )
        self.plan = plan
        self.input_dropout = nn.Dropout(p=0.25)
        self.projection = nn.Linear(
            plan.metadata.hidden_size,
            256,
            bias=False,
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
            proj_size=0,
        )
        self.classifier_ctc = nn.Linear(512, num_classes, bias=True)

    def train(self, mode: bool = True) -> "StaffOMRModel":
        super().train(mode)
        if self._backbone_is_frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone(
            pixel_values=images,
            interpolate_pos_encoding=self.plan.interpolate_pos_encoding,
        )
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        else:
            raise ProtocolError(
                "backbone output does not expose last_hidden_state"
            )
        grid = extract_spatial_grid(hidden, self.plan)
        projected = self.projection(self.input_dropout(grid))
        sequence = projected.mean(dim=1)
        directions = 2 if self.rnn.bidirectional else 1
        state_shape = (
            self.rnn.num_layers * directions,
            sequence.shape[0],
            self.rnn.hidden_size,
        )
        initial_hidden = sequence.new_zeros(state_shape)
        initial_cell = sequence.new_zeros(state_shape)
        recurrent, _ = self.rnn(
            sequence,
            (initial_hidden, initial_cell),
        )
        logits = self.classifier_ctc(recurrent)
        return torch_functional.log_softmax(logits, dim=-1)


def _normalized_method_config(config: Any) -> tuple[str, str, int, int, int]:
    fields = (
        "method",
        "input_geometry",
        "patch_rows",
        "patch_cols",
        "seed",
    )
    missing = [field for field in fields if not hasattr(config, field)]
    if missing:
        raise ProtocolError(f"model config is missing fields: {missing!r}")
    return (
        config.method,
        config.input_geometry,
        config.patch_rows,
        config.patch_cols,
        config.seed,
    )


def build_model(
    backbone: nn.Module,
    metadata: BackboneMetadata,
    config: Any,
    num_classes: int,
) -> StaffOMRModel:
    method, input_geometry, rows, cols, base_seed = _normalized_method_config(
        config
    )
    plan = build_geometry_plan(metadata, method, rows, cols)
    if input_geometry != plan.geometry:
        raise ProtocolError(
            f"input_geometry {input_geometry!r} disagrees with method "
            f"{method!r}; expected {plan.geometry!r}"
        )

    reset_initialization_rng(base_seed)
    if method == "linear_probe":
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        adapted_backbone = backbone
    else:
        lora_config = LoraConfig(
            r=int(LORA_CONTRACT["rank"]),
            lora_alpha=int(LORA_CONTRACT["alpha"]),
            lora_dropout=float(LORA_CONTRACT["dropout"]),
            bias=str(LORA_CONTRACT["bias"]),
            target_modules=list(LORA_CONTRACT["target_modules"]),
            use_rslora=bool(LORA_CONTRACT["use_rslora"]),
        )
        adapted_backbone = get_peft_model(backbone, lora_config)
    return StaffOMRModel(
        adapted_backbone,
        plan,
        num_classes=num_classes,
    )


def task_head_state_keys(model: StaffOMRModel) -> set[str]:
    return {
        name
        for name in model.state_dict()
        if not name.startswith("backbone.")
        and not name.startswith("input_dropout.")
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(trainable.difference(state), key=lambda item: item.encode("utf-8"))
    if missing:
        raise ProtocolError(
            f"trainable parameters are absent from state_dict: {missing!r}"
        )
    return {
        name: state[name].detach().cpu().clone()
        for name in sorted(trainable, key=lambda item: item.encode("utf-8"))
    }


def load_trainable_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ProtocolError(
            "trainable state key mismatch: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    current = model.state_dict()
    for name in sorted(expected, key=lambda item: item.encode("utf-8")):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ProtocolError(f"trainable state {name!r} is not a tensor")
        if tensor.shape != current[name].shape:
            raise ProtocolError(
                f"trainable state {name!r} shape mismatch: expected "
                f"{tuple(current[name].shape)}, actual {tuple(tensor.shape)}"
            )
        current[name].copy_(
            tensor.to(
                device=current[name].device,
                dtype=current[name].dtype,
            )
        )


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    *,
    blank_id: int = 0,
) -> list[list[int]]:
    if log_probs.ndim != 3:
        raise ProtocolError("CTC decoder expects [batch, time, classes]")
    best = log_probs.argmax(dim=-1).detach().cpu().tolist()
    decoded: list[list[int]] = []
    for sequence in best:
        output: list[int] = []
        previous: int | None = None
        for value in sequence:
            token = int(value)
            if token != previous and token != blank_id:
                output.append(token)
            previous = token
        decoded.append(output)
    return decoded
