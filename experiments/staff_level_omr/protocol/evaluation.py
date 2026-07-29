"""Finite CTC training and capacity-layered evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as torch_functional

from .batching import (
    CTCBatch,
    input_lengths_for,
    split_concatenated_targets,
)
from .ctc import minimum_ctc_frames
from .errors import ProtocolError
from .metrics import layered_metrics
from .modeling import greedy_ctc_decode


@dataclass(frozen=True, slots=True)
class TrainEpochResult:
    loss: float
    samples: int
    batches: int
    global_steps: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train_loss": self.loss,
            "train_samples": self.samples,
            "train_batches": self.batches,
            "global_step_delta": self.global_steps,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    split: str
    metrics: dict[str, object]
    capacity: dict[str, object]
    sample_ids: tuple[str, ...]
    predictions: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    feasible: tuple[bool, ...]
    ctc_losses: tuple[float | None, ...]

    def to_dict(self) -> dict[str, object]:
        value = dict(self.metrics)
        value[f"{self.split}_capacity"] = self.capacity
        return value


def _finite_scalar(value: torch.Tensor, context: str) -> float:
    if value.numel() != 1:
        raise ProtocolError(f"{context} must be a scalar")
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise ProtocolError(f"{context} is non-finite: {result!r}")
    return result


def _validate_model_output(log_probs: torch.Tensor, batch: CTCBatch) -> None:
    if not isinstance(log_probs, torch.Tensor) or log_probs.ndim != 3:
        raise ProtocolError("model output must have [batch, time, classes] layout")
    if log_probs.shape[0] != batch.images.shape[0]:
        raise ProtocolError(
            "model output batch dimension differs from input batch"
        )
    if log_probs.shape[1] <= 0 or log_probs.shape[2] <= 1:
        raise ProtocolError("model output time/classes dimensions are invalid")


def train_epoch(
    model: nn.Module,
    loader: Iterable[CTCBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> TrainEpochResult:
    model.train()
    weighted_loss = 0.0
    sample_count = 0
    batch_count = 0
    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, CTCBatch):
            raise ProtocolError(
                f"train batch {batch_index} does not use CTCBatch"
            )
        images = batch.images.to(device)
        targets = batch.targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        log_probs = model(images)
        _validate_model_output(log_probs, batch)
        time_major = log_probs.transpose(0, 1)
        loss = torch_functional.ctc_loss(
            time_major,
            targets,
            input_lengths_for(time_major),
            batch.target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=False,
        )
        loss_value = _finite_scalar(loss, f"train batch {batch_index} loss")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        try:
            torch.nn.utils.get_total_norm(
                gradients,
                error_if_nonfinite=True,
                foreach=None,
            )
        except RuntimeError as exc:
            raise ProtocolError(
                f"train batch {batch_index} produced non-finite gradients"
            ) from exc
        optimizer.step()
        size = int(batch.images.shape[0])
        weighted_loss += loss_value * size
        sample_count += size
        batch_count += 1
    if sample_count == 0:
        raise ProtocolError("training epoch produced no batches")
    result = weighted_loss / sample_count
    if not math.isfinite(result):
        raise ProtocolError("aggregated train loss is non-finite")
    return TrainEpochResult(
        loss=result,
        samples=sample_count,
        batches=batch_count,
        global_steps=batch_count,
    )


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _capacity(
    targets: list[list[int]],
    feasible: list[bool],
) -> dict[str, object]:
    required = [minimum_ctc_frames(target) for target in targets]

    def population(indices: list[int]) -> dict[str, object]:
        return {
            "samples": len(indices),
            "target_length": _distribution(
                [len(targets[index]) for index in indices]
            ),
            "required_frames": _distribution(
                [required[index] for index in indices]
            ),
        }

    all_indices = list(range(len(targets)))
    feasible_indices = [
        index for index, value in enumerate(feasible) if value
    ]
    infeasible_indices = [
        index for index, value in enumerate(feasible) if not value
    ]
    return {
        "all": population(all_indices),
        "feasible": population(feasible_indices),
        "infeasible": population(infeasible_indices),
    }


def _feasible_batch_losses(
    log_probs: torch.Tensor,
    targets: list[list[int]],
    feasible: list[bool],
    *,
    batch_index: int,
) -> list[float | None]:
    indices = [index for index, value in enumerate(feasible) if value]
    result: list[float | None] = [None] * len(targets)
    if not indices:
        return result
    selected = log_probs[indices].transpose(0, 1)
    selected_targets = torch.tensor(
        [token for index in indices for token in targets[index]],
        dtype=torch.long,
        device=log_probs.device,
    )
    lengths = torch.tensor(
        [len(targets[index]) for index in indices],
        dtype=torch.long,
        device="cpu",
    )
    losses = torch_functional.ctc_loss(
        selected,
        selected_targets,
        input_lengths_for(selected),
        lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    if losses.ndim != 1 or losses.numel() != len(indices):
        raise ProtocolError("validation CTC loss returned an unexpected shape")
    for index, loss in zip(indices, losses, strict=True):
        result[index] = _finite_scalar(
            loss,
            f"validation batch {batch_index} sample {index} loss",
        )
    return result


def evaluate_split(
    model: nn.Module,
    loader: Iterable[CTCBatch],
    *,
    split: str,
    device: torch.device,
) -> EvaluationResult:
    if split not in {"val", "test"}:
        raise ProtocolError("evaluation split must be 'val' or 'test'")
    model.eval()
    sample_ids: list[str] = []
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    feasible: list[bool] = []
    ctc_losses: list[float | None] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if not isinstance(batch, CTCBatch):
                raise ProtocolError(
                    f"{split} batch {batch_index} does not use CTCBatch"
                )
            log_probs = model(batch.images.to(device))
            _validate_model_output(log_probs, batch)
            batch_targets = split_concatenated_targets(
                batch.targets,
                batch.target_lengths,
            )
            batch_feasible = [
                bool(value) for value in batch.ctc_feasible.tolist()
            ]
            batch_predictions = greedy_ctc_decode(log_probs)
            if split == "val":
                batch_losses = _feasible_batch_losses(
                    log_probs,
                    batch_targets,
                    batch_feasible,
                    batch_index=batch_index,
                )
            else:
                batch_losses = [None] * len(batch_targets)
            sample_ids.extend(batch.sample_ids)
            targets.extend(batch_targets)
            predictions.extend(batch_predictions)
            feasible.extend(batch_feasible)
            ctc_losses.extend(batch_losses)
    if not targets:
        raise ProtocolError(f"{split} evaluation produced no samples")
    if len(set(sample_ids)) != len(sample_ids):
        raise ProtocolError(f"{split} evaluation contains duplicate sample ids")
    layered = layered_metrics(
        predictions,
        targets,
        feasible,
        feasible_ctc_losses=ctc_losses if split == "val" else None,
    )
    metrics = layered.to_dict(split)
    for name, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ProtocolError(f"{name} is non-finite")
    return EvaluationResult(
        split=split,
        metrics=metrics,
        capacity=_capacity(targets, feasible),
        sample_ids=tuple(sample_ids),
        predictions=tuple(tuple(value) for value in predictions),
        targets=tuple(tuple(value) for value in targets),
        feasible=tuple(feasible),
        ctc_losses=tuple(ctc_losses),
    )
