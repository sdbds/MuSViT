"""CPU-only foundations for the staff-level OMR trusted training protocol."""

from .canonical import (
    canonical_json_bytes,
    canonical_sha256,
    read_json,
    write_canonical_json,
)
from .config import StaffOMRConfig
from .errors import ProtocolError

__all__ = [
    "ProtocolError",
    "StaffOMRConfig",
    "canonical_json_bytes",
    "canonical_sha256",
    "read_json",
    "write_canonical_json",
]
