"""Micro CER and capacity-layered evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import editdistance

from .errors import ProtocolError


def _validate_sequences(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
) -> None:
    if len(predictions) != len(targets):
        raise ProtocolError(
            "predictions and targets must contain the same number of samples"
        )
    if not targets:
        raise ProtocolError("metrics require at least one sample")
    empty = [index for index, target in enumerate(targets) if not target]
    if empty:
        raise ProtocolError(
            f"metric targets must be non-empty; empty indices={empty}"
        )


def micro_cer(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
) -> float:
    """Compute total edit distance divided by total target length."""
    _validate_sequences(predictions, targets)
    total_distance = sum(
        editdistance.distance(prediction, target)
        for prediction, target in zip(predictions, targets)
    )
    total_target_length = sum(len(target) for target in targets)
    if total_target_length <= 0:
        raise ProtocolError("micro CER denominator must be positive")
    result = total_distance / total_target_length
    if not math.isfinite(result):
        raise ProtocolError("micro CER must be finite")
    return result


@dataclass(frozen=True, slots=True)
class LayeredMetrics:
    cer_all: float
    cer_feasible: float | None
    ctc_loss_feasible: float | None
    feasible_samples: int
    infeasible_samples: int
    infeasible_ratio: float

    def to_dict(self, prefix: str) -> dict[str, Any]:
        if not isinstance(prefix, str) or not prefix:
            raise ProtocolError("metric prefix must be a non-empty string")
        result: dict[str, Any] = {
            f"{prefix}_CER_all": self.cer_all,
            f"{prefix}_CER_feasible": self.cer_feasible,
            f"{prefix}_feasible_samples": self.feasible_samples,
            f"{prefix}_infeasible_samples": self.infeasible_samples,
            f"{prefix}_infeasible_ratio": self.infeasible_ratio,
        }
        if prefix == "val" or self.ctc_loss_feasible is not None:
            result[f"{prefix}_CTC_loss_feasible"] = self.ctc_loss_feasible
            if prefix == "val":
                ordered = {
                    f"{prefix}_CER_all": result[f"{prefix}_CER_all"],
                    f"{prefix}_CER_feasible": result[
                        f"{prefix}_CER_feasible"
                    ],
                    f"{prefix}_CTC_loss_feasible": result[
                        f"{prefix}_CTC_loss_feasible"
                    ],
                    f"{prefix}_feasible_samples": result[
                        f"{prefix}_feasible_samples"
                    ],
                    f"{prefix}_infeasible_samples": result[
                        f"{prefix}_infeasible_samples"
                    ],
                    f"{prefix}_infeasible_ratio": result[
                        f"{prefix}_infeasible_ratio"
                    ],
                }
                return ordered
        return result


def layered_metrics(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    feasible: Sequence[bool],
    *,
    feasible_ctc_losses: Sequence[float | None] | None = None,
) -> LayeredMetrics:
    """Report full-set CER separately from the capacity-feasible subset."""
    _validate_sequences(predictions, targets)
    if len(feasible) != len(targets):
        raise ProtocolError(
            "feasible flags and targets must contain the same number of samples"
        )
    if any(not isinstance(value, bool) for value in feasible):
        raise ProtocolError("feasible flags must be booleans")
    if (
        feasible_ctc_losses is not None
        and len(feasible_ctc_losses) != len(targets)
    ):
        raise ProtocolError(
            "feasible CTC losses and targets must contain the same number "
            "of samples"
        )

    feasible_indices = [
        index for index, is_feasible in enumerate(feasible) if is_feasible
    ]
    feasible_count = len(feasible_indices)
    infeasible_count = len(targets) - feasible_count
    cer_all = micro_cer(predictions, targets)
    if feasible_indices:
        cer_feasible = micro_cer(
            [predictions[index] for index in feasible_indices],
            [targets[index] for index in feasible_indices],
        )
    else:
        cer_feasible = None

    normalized_losses: list[float] = []
    if feasible_ctc_losses is not None:
        for index, (is_feasible, loss) in enumerate(
            zip(feasible, feasible_ctc_losses)
        ):
            if not is_feasible:
                if loss is not None:
                    raise ProtocolError(
                        f"infeasible sample {index} must not have a CTC loss"
                    )
                continue
            if (
                loss is None
                or isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            ):
                raise ProtocolError(
                    f"feasible CTC loss at sample {index} must be finite"
                )
            normalized_losses.append(float(loss) / len(targets[index]))
    ctc_loss_feasible = (
        sum(normalized_losses) / len(normalized_losses)
        if normalized_losses
        else None
    )
    return LayeredMetrics(
        cer_all=cer_all,
        cer_feasible=cer_feasible,
        ctc_loss_feasible=ctc_loss_feasible,
        feasible_samples=feasible_count,
        infeasible_samples=infeasible_count,
        infeasible_ratio=infeasible_count / len(targets),
    )
