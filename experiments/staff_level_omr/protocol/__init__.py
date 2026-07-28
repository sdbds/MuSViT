"""CPU-only foundations for the staff-level OMR trusted training protocol."""

from .canonical import (
    canonical_json_bytes,
    canonical_sha256,
    read_json,
    write_canonical_json,
)
from .config import StaffOMRConfig
from .data_bundle import (
    BundleSample,
    ImageVerificationStats,
    PrepareDataReport,
    ValidatedDatasetBundle,
    load_dataset_bundle,
    prepare_dataset_bundle,
)
from .errors import ProtocolError
from .vocabulary import Vocabulary

__all__ = [
    "ProtocolError",
    "StaffOMRConfig",
    "BundleSample",
    "ImageVerificationStats",
    "PrepareDataReport",
    "ValidatedDatasetBundle",
    "Vocabulary",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_dataset_bundle",
    "prepare_dataset_bundle",
    "read_json",
    "write_canonical_json",
]
