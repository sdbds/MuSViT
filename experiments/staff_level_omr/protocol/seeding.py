"""Domain-separated deterministic seeds for the staff OMR v2 protocol."""

from __future__ import annotations

import hashlib
import random
from typing import TypeAlias

import numpy as np
import torch

from .errors import ProtocolError


PROTOCOL_VERSION = "staff_omr_v2"
SEED_SCHEDULE_VERSION = "staff_omr_sample_epoch_sha256_v1"
INIT_SEED_LABEL = "init"

SeedPart: TypeAlias = str | int


def _encode_part(part: object) -> bytes:
    if isinstance(part, bool):
        raise ProtocolError("seed parts cannot be booleans")
    if isinstance(part, int):
        if part < 0:
            raise ProtocolError("integer seed parts must be non-negative")
        return str(part).encode("ascii")
    if isinstance(part, str):
        if "\0" in part:
            raise ProtocolError("seed string parts cannot contain NUL")
        return part.encode("utf-8")
    raise ProtocolError("seed parts must be non-negative integers or strings")


def seed_digest(label: str, *parts: SeedPart) -> bytes:
    """Return the full domain-separated SHA-256 digest for a seed domain."""
    if not isinstance(label, str) or not label or "\0" in label:
        raise ProtocolError("seed label must be a non-empty NUL-free string")
    payload = b"\0".join(_encode_part(part) for part in parts)
    return hashlib.sha256(
        PROTOCOL_VERSION.encode("utf-8")
        + b"\0"
        + label.encode("utf-8")
        + b"\0"
        + payload
    ).digest()


def seed32(label: str, *parts: SeedPart) -> int:
    """Return the first four digest bytes as an unsigned big-endian integer."""
    return int.from_bytes(seed_digest(label, *parts)[:4], "big", signed=False)


def seed256(label: str, *parts: SeedPart) -> int:
    """Return the complete digest as an unsigned big-endian integer."""
    return int.from_bytes(seed_digest(label, *parts), "big", signed=False)


def initialization_seed(base_seed: int) -> int:
    return seed32(INIT_SEED_LABEL, base_seed)


def epoch_seed(base_seed: int, epoch: int) -> int:
    return seed32("epoch", base_seed, epoch)


def worker_base_seed(base_seed: int, epoch: int) -> int:
    return seed32("worker", base_seed, epoch)


def sample_augment_seed(base_seed: int, epoch: int, sample_id: str) -> int:
    return seed256("augment", base_seed, epoch, sample_id)


def sample_order_key(
    base_seed: int,
    epoch: int,
    sample_id: str,
) -> tuple[bytes, bytes]:
    return (
        seed_digest("order", base_seed, epoch, sample_id),
        sample_id.encode("utf-8"),
    )


def _reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_initialization_rng(base_seed: int) -> int:
    value = initialization_seed(base_seed)
    _reset_rng(value)
    return value


def reset_epoch_rng(base_seed: int, epoch: int) -> int:
    value = epoch_seed(base_seed, epoch)
    _reset_rng(value)
    return value
