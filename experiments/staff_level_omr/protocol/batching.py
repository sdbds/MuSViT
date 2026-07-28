"""Variable-length target collation for staff-level OMR CTC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch

from .errors import ProtocolError


class CTCBatch(NamedTuple):
    images: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    sample_ids: tuple[str, ...]
    ctc_feasible: torch.Tensor


def ctc_collate(
    samples: Sequence[
        tuple[torch.Tensor, torch.Tensor, int, str, bool]
    ],
) -> CTCBatch:
    """Collate targets by concatenation instead of dataset-wide padding."""
    if not samples:
        raise ProtocolError("cannot collate an empty CTC batch")

    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    lengths: list[int] = []
    sample_ids: list[str] = []
    feasibility: list[bool] = []

    for index, item in enumerate(samples):
        if not isinstance(item, (tuple, list)) or len(item) != 5:
            raise ProtocolError(
                f"batch item {index} must contain five protocol fields"
            )
        image, target, target_length, sample_id, feasible = item
        if not isinstance(image, torch.Tensor):
            raise ProtocolError(f"batch item {index} image must be a tensor")
        if not isinstance(target, torch.Tensor) or target.ndim != 1:
            raise ProtocolError(
                f"batch item {index} target must be a 1-D tensor"
            )
        if target.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ProtocolError(
                f"batch item {index} target must use an integer dtype"
            )
        if (
            isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or target_length <= 0
        ):
            raise ProtocolError(
                f"batch item {index} target_length must be a positive integer"
            )
        if target.numel() != target_length:
            raise ProtocolError(
                f"batch item {index} target_length={target_length} "
                f"does not match target elements={target.numel()}"
            )
        normalized_target = target.to(dtype=torch.long, device="cpu")
        if torch.any(normalized_target <= 0):
            raise ProtocolError(
                f"batch item {index} target contains blank/non-positive id"
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ProtocolError(
                f"batch item {index} sample_id must be a non-empty string"
            )
        if not isinstance(feasible, bool):
            raise ProtocolError(
                f"batch item {index} ctc_feasible must be boolean"
            )
        images.append(image)
        targets.append(normalized_target)
        lengths.append(target_length)
        sample_ids.append(sample_id)
        feasibility.append(feasible)

    if len(set(sample_ids)) != len(sample_ids):
        raise ProtocolError("CTC batch contains duplicate sample_ids")
    try:
        image_batch = torch.stack(images)
    except RuntimeError as exc:
        raise ProtocolError(
            f"batch images cannot be stacked into one tensor: {exc}"
        ) from exc
    target_batch = torch.cat(targets)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    if int(length_tensor.sum().item()) != target_batch.numel():
        raise ProtocolError(
            "sum of target lengths does not match concatenated target size"
        )
    return CTCBatch(
        images=image_batch,
        targets=target_batch,
        target_lengths=length_tensor,
        sample_ids=tuple(sample_ids),
        ctc_feasible=torch.tensor(feasibility, dtype=torch.bool),
    )


def split_concatenated_targets(
    targets: torch.Tensor,
    lengths: Sequence[int] | torch.Tensor,
) -> list[list[int]]:
    """Recover target sequences strictly from explicit lengths."""
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
        raise ProtocolError("concatenated targets must be a 1-D tensor")
    if isinstance(lengths, torch.Tensor):
        if lengths.ndim != 1:
            raise ProtocolError("target lengths must be a 1-D tensor")
        raw_lengths = lengths.detach().cpu().tolist()
    else:
        raw_lengths = list(lengths)
    if not raw_lengths:
        raise ProtocolError("target lengths must be non-empty")
    if any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        for length in raw_lengths
    ):
        raise ProtocolError("target lengths must be positive integers")
    if sum(raw_lengths) != targets.numel():
        raise ProtocolError(
            "sum of target lengths does not match concatenated target size"
        )

    values = targets.detach().cpu().tolist()
    result: list[list[int]] = []
    offset = 0
    for length in raw_lengths:
        result.append(values[offset : offset + length])
        offset += length
    return result


def input_lengths_for(log_probs: torch.Tensor) -> torch.Tensor:
    """Build CPU CTC input lengths from runtime ``T,N,C`` dimensions."""
    if not isinstance(log_probs, torch.Tensor) or log_probs.ndim != 3:
        raise ProtocolError("log_probs must have T,N,C layout")
    time_steps, batch_size, classes = log_probs.shape
    if time_steps <= 0 or batch_size <= 0 or classes <= 0:
        raise ProtocolError(
            f"log_probs dimensions must be positive, got {tuple(log_probs.shape)}"
        )
    return torch.full(
        (batch_size,),
        fill_value=time_steps,
        dtype=torch.long,
        device="cpu",
    )
