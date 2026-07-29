"""Auditable CTC feasibility analysis and train exclusion policy."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from .canonical import canonical_sha256, read_json, write_canonical_json
from .data_bundle import BundleSample, SPLITS
from .errors import ProtocolError
from .vocabulary import Vocabulary


EXCLUSIONS_SCHEMA = "staff_omr_train_exclusions_v1"
REQUIRED_FRAMES_ALGORITHM = "ctc_minimum_frames_v1"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def minimum_ctc_frames(target_ids: Sequence[int]) -> int:
    """Return target length plus one blank frame per adjacent repeated id."""
    if not target_ids:
        raise ProtocolError("CTC target_ids must be non-empty")
    for token_id in target_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id <= 0
        ):
            raise ProtocolError(
                f"CTC target ids must be positive integers, got {token_id!r}"
            )
    adjacent_repeats = sum(
        current == previous
        for previous, current in zip(target_ids, target_ids[1:])
    )
    return len(target_ids) + adjacent_repeats


def _distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


@dataclass(frozen=True, slots=True)
class CTCFeasibilityRecord:
    sample_id: str
    split: str
    target_length: int
    adjacent_repeats: int
    required_frames: int
    available_frames: int
    feasible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "target_length": self.target_length,
            "adjacent_repeats": self.adjacent_repeats,
            "required_frames": self.required_frames,
            "available_frames": self.available_frames,
            "feasible": self.feasible,
        }


@dataclass(frozen=True, slots=True)
class CTCPreflight:
    patch_cols: int
    records: tuple[CTCFeasibilityRecord, ...]
    split_summaries: Mapping[str, Mapping[str, object]]

    @property
    def train_records(self) -> tuple[CTCFeasibilityRecord, ...]:
        return tuple(record for record in self.records if record.split == "train")

    @property
    def infeasible_train_records(self) -> tuple[CTCFeasibilityRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.split == "train" and not record.feasible
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "required_frames_algorithm": REQUIRED_FRAMES_ALGORITHM,
            "patch_cols": self.patch_cols,
            "splits": {
                split: dict(self.split_summaries[split]) for split in SPLITS
            },
            "infeasible_samples": [
                record.to_dict() for record in self.records if not record.feasible
            ],
        }


@dataclass(frozen=True, slots=True)
class CTCPolicyResult:
    retained_train_sample_ids: tuple[str, ...]
    excluded_train_sample_ids: tuple[str, ...]
    exclusions_sha256: str | None
    exclusion_count: int


class CTCInfeasibleError(ProtocolError):
    """Raised by the default fail policy with a reusable candidate artifact."""

    def __init__(
        self,
        message: str,
        candidate_document: dict[str, object],
        candidate_path: Path | None,
    ) -> None:
        super().__init__(message)
        self.candidate_document = candidate_document
        self.candidate_path = candidate_path


def _population_summary(
    records: Sequence[CTCFeasibilityRecord],
) -> dict[str, object]:
    return {
        "samples": len(records),
        "target_length": _distribution(
            [record.target_length for record in records]
        ),
        "adjacent_repeats": _distribution(
            [record.adjacent_repeats for record in records]
        ),
        "required_frames": _distribution(
            [record.required_frames for record in records]
        ),
    }


def _split_summary(
    records: Sequence[CTCFeasibilityRecord],
) -> dict[str, object]:
    feasible = [record for record in records if record.feasible]
    infeasible = [record for record in records if not record.feasible]
    return {
        "samples": len(records),
        "feasible_samples": len(feasible),
        "infeasible_samples": len(infeasible),
        "infeasible_ratio": len(infeasible) / len(records) if records else 0.0,
        "max_required_frames": (
            max(record.required_frames for record in records)
            if records
            else None
        ),
        "all": _population_summary(records),
        "feasible": _population_summary(feasible),
        "infeasible": _population_summary(infeasible),
    }


def analyze_ctc_feasibility(
    samples: Sequence[BundleSample],
    vocabulary: Vocabulary,
    *,
    patch_cols: int,
) -> CTCPreflight:
    """Analyze every split without removing any sample."""
    if (
        isinstance(patch_cols, bool)
        or not isinstance(patch_cols, int)
        or patch_cols <= 0
    ):
        raise ProtocolError("patch_cols must be a positive integer")
    if not samples:
        raise ProtocolError("CTC preflight requires at least one sample")

    seen: set[str] = set()
    records: list[CTCFeasibilityRecord] = []
    for sample in sorted(samples, key=lambda item: _utf8_key(item.sample_id)):
        if sample.sample_id in seen:
            raise ProtocolError(
                f"CTC preflight received duplicate sample_id {sample.sample_id!r}"
            )
        seen.add(sample.sample_id)
        if sample.split not in SPLITS:
            raise ProtocolError(
                f"sample {sample.sample_id!r} has invalid split {sample.split!r}"
            )
        target_ids = vocabulary.encode(sample.target_tokens)
        required_frames = minimum_ctc_frames(target_ids)
        target_length = len(target_ids)
        adjacent_repeats = required_frames - target_length
        records.append(
            CTCFeasibilityRecord(
                sample_id=sample.sample_id,
                split=sample.split,
                target_length=target_length,
                adjacent_repeats=adjacent_repeats,
                required_frames=required_frames,
                available_frames=patch_cols,
                feasible=required_frames <= patch_cols,
            )
        )

    summaries = {
        split: _split_summary(
            [record for record in records if record.split == split]
        )
        for split in SPLITS
    }
    return CTCPreflight(
        patch_cols=patch_cols,
        records=tuple(records),
        split_summaries=summaries,
    )


def _candidate_document(
    preflight: CTCPreflight, manifest_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": EXCLUSIONS_SCHEMA,
        "source_manifest_sha256": manifest_sha256,
        "patch_cols": preflight.patch_cols,
        "required_frames_algorithm": REQUIRED_FRAMES_ALGORITHM,
        "sample_ids": sorted(
            (
                record.sample_id
                for record in preflight.infeasible_train_records
            ),
            key=_utf8_key,
        ),
    }


def _validate_manifest_hash(value: str) -> None:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise ProtocolError(
            "manifest_sha256 must be 64 lowercase hexadecimal characters"
        )


def _load_exclusions(
    path: str | Path,
    *,
    manifest_sha256: str,
    patch_cols: int,
) -> tuple[dict[str, object], tuple[str, ...]]:
    source = Path(path)
    document = read_json(source)
    if not isinstance(document, dict):
        raise ProtocolError("train exclusions must be a JSON object")
    expected_fields = {
        "schema_version",
        "source_manifest_sha256",
        "patch_cols",
        "required_frames_algorithm",
        "sample_ids",
    }
    actual_fields = set(document)
    if actual_fields != expected_fields:
        raise ProtocolError(
            "train exclusions fields mismatch; "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"extra={sorted(actual_fields - expected_fields)}"
        )
    if document["schema_version"] != EXCLUSIONS_SCHEMA:
        raise ProtocolError(
            f"unsupported train exclusions schema {document['schema_version']!r}"
        )
    if document["source_manifest_sha256"] != manifest_sha256:
        raise ProtocolError(
            "train exclusions source_manifest_sha256 does not match manifest"
        )
    if (
        isinstance(document["patch_cols"], bool)
        or document["patch_cols"] != patch_cols
    ):
        raise ProtocolError(
            "train exclusions patch_cols does not match current patch_cols"
        )
    if document["required_frames_algorithm"] != REQUIRED_FRAMES_ALGORITHM:
        raise ProtocolError(
            "train exclusions required_frames_algorithm mismatch"
        )
    sample_ids = document["sample_ids"]
    if not isinstance(sample_ids, list) or any(
        not isinstance(sample_id, str) or not sample_id
        for sample_id in sample_ids
    ):
        raise ProtocolError(
            "train exclusions sample_ids must be an array of non-empty strings"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ProtocolError("train exclusions sample_ids contain duplicate ids")
    if sample_ids != sorted(sample_ids, key=_utf8_key):
        raise ProtocolError(
            "train exclusions sample_ids must be sorted by UTF-8 bytes"
        )
    return document, tuple(sample_ids)


def apply_train_policy(
    preflight: CTCPreflight,
    *,
    manifest_sha256: str,
    policy: str,
    exclusions_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
) -> CTCPolicyResult:
    """Apply fail/exclude policy while leaving manifest and eval splits intact."""
    _validate_manifest_hash(manifest_sha256)
    train_records = preflight.train_records
    if not train_records:
        raise ProtocolError("CTC preflight has no train samples")
    infeasible_records = preflight.infeasible_train_records
    infeasible_ids = {
        record.sample_id for record in infeasible_records
    }

    if policy == "fail":
        if exclusions_path is not None:
            raise ProtocolError(
                "exclusions_path must be empty when train policy is fail"
            )
        if infeasible_records:
            candidate = _candidate_document(preflight, manifest_sha256)
            written_path = (
                Path(candidate_path).resolve(strict=False)
                if candidate_path is not None
                else None
            )
            if written_path is not None:
                write_canonical_json(written_path, candidate)
            details = ", ".join(
                f"{record.sample_id}(required={record.required_frames},"
                f"available={record.available_frames})"
                for record in infeasible_records[:20]
            )
            location = (
                f"; candidate={written_path}" if written_path is not None else ""
            )
            raise CTCInfeasibleError(
                "train contains CTC-infeasible samples: "
                f"count={len(infeasible_records)}; first 20={details}{location}",
                candidate,
                written_path,
            )
        retained = tuple(
            sorted(
                (record.sample_id for record in train_records),
                key=_utf8_key,
            )
        )
        return CTCPolicyResult(
            retained_train_sample_ids=retained,
            excluded_train_sample_ids=(),
            exclusions_sha256=None,
            exclusion_count=0,
        )

    if policy != "exclude_listed":
        raise ProtocolError(
            "train infeasible policy must be 'fail' or 'exclude_listed'"
        )
    if exclusions_path is None:
        raise ProtocolError(
            "exclusions_path is required when policy is exclude_listed"
        )
    document, listed_ids = _load_exclusions(
        exclusions_path,
        manifest_sha256=manifest_sha256,
        patch_cols=preflight.patch_cols,
    )
    records_by_id = {record.sample_id: record for record in preflight.records}
    unknown = sorted(
        (sample_id for sample_id in listed_ids if sample_id not in records_by_id),
        key=_utf8_key,
    )
    if unknown:
        raise ProtocolError(f"train exclusions contain unknown ids: {unknown}")
    val_or_test = sorted(
        (
            sample_id
            for sample_id in listed_ids
            if records_by_id[sample_id].split != "train"
        ),
        key=_utf8_key,
    )
    if val_or_test:
        raise ProtocolError(
            f"train exclusions contain val/test ids: {val_or_test}"
        )
    feasible = sorted(
        (
            sample_id
            for sample_id in listed_ids
            if records_by_id[sample_id].feasible
        ),
        key=_utf8_key,
    )
    if feasible:
        raise ProtocolError(
            f"train exclusions contain feasible ids: {feasible}"
        )
    listed_set = set(listed_ids)
    missing = sorted(infeasible_ids - listed_set, key=_utf8_key)
    extra = sorted(listed_set - infeasible_ids, key=_utf8_key)
    if missing or extra:
        raise ProtocolError(
            "train exclusions do not exactly match infeasible train ids; "
            f"missing={missing}, extra={extra}"
        )

    retained = tuple(
        sorted(
            (
                record.sample_id
                for record in train_records
                if record.sample_id not in listed_set
            ),
            key=_utf8_key,
        )
    )
    if not retained:
        raise ProtocolError(
            "retained_train_samples must be greater than zero after exclusions"
        )
    excluded = tuple(sorted(listed_set, key=_utf8_key))
    return CTCPolicyResult(
        retained_train_sample_ids=retained,
        excluded_train_sample_ids=excluded,
        exclusions_sha256=canonical_sha256(document),
        exclusion_count=len(excluded),
    )
