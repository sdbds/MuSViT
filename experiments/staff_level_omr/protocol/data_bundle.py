"""Deterministic dataset preparation and validation for staff-level OMR."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .canonical import (
    canonical_sha256,
    read_json,
    write_canonical_json,
)
from .errors import ProtocolError
from .vocabulary import TARGET_PARSER, TOKEN_SORT, Vocabulary


BUNDLE_SCHEMA = "staff_omr_dataset_bundle_v1"
MANIFEST_SCHEMA = "staff_omr_split_v1"
IMAGE_INDEX_SCHEMA = "staff_omr_image_verification_v1"
PREPARE_PROTOCOL = "staff_omr_prepare_v1"
PAIR_RULE = "replace_final_suffix_v1"
SPLIT_ALGORITHM = "sha256_group_largest_remainder_v1"
IMAGE_SUFFIX = "_region.png"
TARGET_SUFFIX = "_gt.txt"
BUNDLE_FILE = "bundle.json"
MANIFEST_FILE = "split_manifest.json"
VOCABULARY_FILE = "vocabulary.json"
IMAGE_INDEX_FILE = "image_verification_index.json"
SPLITS = ("train", "val", "test")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:$")


@dataclass(frozen=True, slots=True)
class BundleSample:
    sample_id: str
    group_id: str
    image_path: str
    image_size_bytes: int
    image_sha256: str
    target_path: str
    target_sha256: str
    split: str
    target_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImageVerificationStats:
    mode: str
    hits: int
    recomputed: int
    status: str
    trusted_baseline: bool


@dataclass(frozen=True, slots=True)
class ValidatedDatasetBundle:
    bundle_path: Path
    data_path: Path
    dataset_id: str
    dataset_bundle_sha256: str
    manifest_sha256: str
    vocabulary_sha256: str
    vocabulary: Vocabulary
    samples: tuple[BundleSample, ...]
    image_verification: ImageVerificationStats

    def split_samples(self, split: str) -> tuple[BundleSample, ...]:
        if split not in SPLITS:
            raise ProtocolError(f"unknown split {split!r}")
        return tuple(sample for sample in self.samples if sample.split == split)


@dataclass(frozen=True, slots=True)
class PrepareDataReport:
    output_path: Path
    dataset_id: str
    bundle_sha256: str
    manifest_sha256: str
    vocabulary_sha256: str
    split_group_counts: Mapping[str, int]
    split_sample_counts: Mapping[str, int]
    target_length_summaries: Mapping[str, Mapping[str, int | float | None]]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "dataset_id": self.dataset_id,
            "bundle_sha256": self.bundle_sha256,
            "manifest_sha256": self.manifest_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "split_group_counts": dict(self.split_group_counts),
            "split_sample_counts": dict(self.split_sample_counts),
            "target_length_summaries": {
                split: dict(summary)
                for split, summary in self.target_length_summaries.items()
            },
        }


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ProtocolError(f"{context} object keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ProtocolError(
            f"{context} fields mismatch; missing={missing}, extra={extra}"
        )


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProtocolError(f"{context} must be a non-empty string without NUL")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise ProtocolError(f"{context} must be 64 lowercase hex characters")
    return value


def _require_non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{context} must be a non-negative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _read_target(path: Path, sample_id: str) -> tuple[bytes, tuple[str, ...]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(
            f"cannot read target for sample {sample_id!r}: {path}: {exc}"
        ) from exc
    try:
        tokens = tuple(payload.decode("utf-8").split())
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            f"target for sample {sample_id!r} is not valid UTF-8: {path}"
        ) from exc
    if not tokens:
        raise ProtocolError(
            f"empty target after Unicode-whitespace parsing: sample={sample_id!r}"
        )
    return payload, tokens


def _normalized_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _discover_pairs(data_path: Path) -> list[tuple[str, str]]:
    image_paths: dict[str, str] = {}
    target_paths: dict[str, str] = {}
    casefolded: dict[str, str] = {}

    try:
        files = sorted(
            (path for path in data_path.rglob("*") if path.is_file()),
            key=lambda path: _utf8_key(_normalized_relative(path, data_path)),
        )
    except OSError as exc:
        raise ProtocolError(f"cannot scan data_path {data_path}: {exc}") from exc

    for path in files:
        relative = _normalized_relative(path, data_path)
        folded = relative.casefold()
        is_candidate = folded.endswith(IMAGE_SUFFIX) or folded.endswith(
            TARGET_SUFFIX
        )
        if not is_candidate:
            continue
        previous = casefolded.get(folded)
        if previous is not None and previous != relative:
            raise ProtocolError(
                "case-insensitive path collision: "
                f"{previous!r} and {relative!r}"
            )
        casefolded[folded] = relative
        if folded.endswith(IMAGE_SUFFIX):
            if not relative.endswith(IMAGE_SUFFIX):
                raise ProtocolError(
                    f"image suffix casing must be exactly {IMAGE_SUFFIX!r}: "
                    f"{relative!r}"
                )
            image_paths[relative] = relative
        else:
            if not relative.endswith(TARGET_SUFFIX):
                raise ProtocolError(
                    f"target suffix casing must be exactly {TARGET_SUFFIX!r}: "
                    f"{relative!r}"
                )
            target_paths[relative] = relative

    orphan_images = sorted(
        (
            image
            for image in image_paths
            if image[: -len(IMAGE_SUFFIX)] + TARGET_SUFFIX not in target_paths
        ),
        key=_utf8_key,
    )
    orphan_targets = sorted(
        (
            target
            for target in target_paths
            if target[: -len(TARGET_SUFFIX)] + IMAGE_SUFFIX not in image_paths
        ),
        key=_utf8_key,
    )
    if orphan_images:
        raise ProtocolError(
            f"orphan image(s): count={len(orphan_images)}; "
            f"first 20={orphan_images[:20]}"
        )
    if orphan_targets:
        raise ProtocolError(
            f"orphan target(s): count={len(orphan_targets)}; "
            f"first 20={orphan_targets[:20]}"
        )
    if not image_paths:
        raise ProtocolError(
            f"data_path contains no paired *{IMAGE_SUFFIX} / *{TARGET_SUFFIX} files"
        )

    return [
        (image, image[: -len(IMAGE_SUFFIX)] + TARGET_SUFFIX)
        for image in sorted(image_paths, key=_utf8_key)
    ]


def _normalize_split_weights(values: Sequence[Any]) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ProtocolError("split_ratios must contain exactly three values")
    fractions: list[Fraction] = []
    for index, value in enumerate(values):
        try:
            decimal = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ProtocolError(
                f"split_ratios[{index}] is not a decimal number: {value!r}"
            ) from exc
        if not decimal.is_finite() or decimal <= 0:
            raise ProtocolError(
                f"split_ratios[{index}] must be finite and positive"
            )
        fractions.append(Fraction(decimal))

    denominator_lcm = math.lcm(
        *(fraction.denominator for fraction in fractions)
    )
    integers = [
        fraction.numerator * (denominator_lcm // fraction.denominator)
        for fraction in fractions
    ]
    common = reduce(gcd, integers)
    normalized = tuple(value // common for value in integers)
    return normalized  # type: ignore[return-value]


def _allocate_split_counts(
    group_count: int, weights: tuple[int, int, int]
) -> tuple[int, int, int]:
    if group_count < 3:
        raise ProtocolError(
            f"prepare-data requires at least three groups, found {group_count}"
        )
    remaining = group_count - 3
    total_weight = sum(weights)
    exact = [
        Fraction(remaining * weight, total_weight) for weight in weights
    ]
    extras = [value.numerator // value.denominator for value in exact]
    unassigned = remaining - sum(extras)
    remainders = [value - floor for value, floor in zip(exact, extras)]
    priority = sorted(
        range(3), key=lambda index: (-remainders[index], index)
    )
    for index in priority[:unassigned]:
        extras[index] += 1
    return tuple(1 + extra for extra in extras)  # type: ignore[return-value]


def _assign_groups(
    dataset_id: str,
    seed: int,
    groups: Iterable[str],
    weights: tuple[int, int, int],
) -> dict[str, str]:
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            hashlib.sha256(
                (
                    dataset_id
                    + "\0"
                    + str(seed)
                    + "\0"
                    + group_id
                ).encode("utf-8")
            ).digest(),
            _utf8_key(group_id),
        ),
    )
    counts = _allocate_split_counts(len(ordered_groups), weights)
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        for group_id in ordered_groups[offset : offset + count]:
            assignments[group_id] = split
        offset += count
    return assignments


def _distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _safe_member_path(bundle_path: Path, value: Any, expected: str) -> Path:
    if value != expected:
        raise ProtocolError(
            f"bundle member must be fixed filename {expected!r}, got {value!r}"
        )
    if not isinstance(value, str) or "/" in value or "\\" in value or ".." in value:
        raise ProtocolError(f"unsafe bundle member filename {value!r}")
    resolved = (bundle_path / value).resolve(strict=False)
    try:
        resolved.relative_to(bundle_path)
    except ValueError as exc:
        raise ProtocolError(
            f"bundle member escapes bundle directory: {value!r}"
        ) from exc
    return resolved


def _safe_data_file(
    data_path: Path, relative: Any, context: str
) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative:
        raise ProtocolError(f"{context} must be a non-empty relative path")
    if "\\" in relative:
        raise ProtocolError(
            f"{context} must use '/' separators and cannot escape data_path: "
            f"{relative!r}"
        )
    pure = PurePosixPath(relative)
    parts = pure.parts
    if (
        pure.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or _WINDOWS_DRIVE.fullmatch(parts[0])
        or "/".join(parts) != relative
    ):
        raise ProtocolError(
            f"{context} cannot escape data_path and must be normalized: "
            f"{relative!r}"
        )
    candidate = data_path.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"{context} is missing: {relative!r}") from exc
    try:
        resolved.relative_to(data_path)
    except ValueError as exc:
        raise ProtocolError(
            f"{context} resolves outside data_path (escape): {relative!r}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise ProtocolError(
            f"{context} must resolve to a readable file: {relative!r}"
        )
    return relative, resolved


def _validate_generation_contract(value: Any) -> dict[str, Any]:
    contract = _require_object(value, "generation_contract")
    _require_exact_keys(
        contract,
        {
            "pair_rule",
            "image_suffix",
            "target_suffix",
            "group_regex",
            "split_weights",
            "seed",
            "split_algorithm",
            "target_parser",
            "token_sort",
        },
        "generation_contract",
    )
    expected = {
        "pair_rule": PAIR_RULE,
        "image_suffix": IMAGE_SUFFIX,
        "target_suffix": TARGET_SUFFIX,
        "split_algorithm": SPLIT_ALGORITHM,
        "target_parser": TARGET_PARSER,
        "token_sort": TOKEN_SORT,
    }
    for field, expected_value in expected.items():
        if contract[field] != expected_value:
            raise ProtocolError(
                f"generation_contract.{field} must be {expected_value!r}"
            )
    _require_nonempty_string(contract["group_regex"], "group_regex")
    _require_non_negative_int(contract["seed"], "generation_contract.seed")
    weights = contract["split_weights"]
    if (
        not isinstance(weights, list)
        or len(weights) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in weights
        )
        or reduce(gcd, weights) != 1
    ):
        raise ProtocolError(
            "generation_contract.split_weights must be three simplest "
            "positive integers"
        )
    return contract


def prepare_dataset_bundle(
    *,
    data_path: str | Path,
    dataset_id: str,
    group_regex: str,
    split_ratios: Sequence[Any] = ("0.8", "0.1", "0.1"),
    seed: int = 7,
    out: str | Path,
) -> PrepareDataReport:
    """Create and atomically publish a deterministic dataset bundle."""
    dataset_id = _require_nonempty_string(dataset_id, "dataset_id")
    group_regex = _require_nonempty_string(group_regex, "group_regex")
    seed = _require_non_negative_int(seed, "seed")
    source = Path(data_path).expanduser()
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"data_path does not exist: {source}") from exc
    if not source.is_dir() or not os.access(source, os.R_OK):
        raise ProtocolError(f"data_path must be a readable directory: {source}")

    output = Path(out).expanduser().resolve(strict=False)
    if output.exists():
        raise ProtocolError(f"output bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        compiled_regex = re.compile(group_regex)
    except re.error as exc:
        raise ProtocolError(f"invalid group_regex: {exc}") from exc
    if "group_id" not in compiled_regex.groupindex:
        raise ProtocolError(
            "group_regex must define named capture group 'group_id'"
        )

    weights = _normalize_split_weights(split_ratios)
    pairs = _discover_pairs(source)
    unmatched: list[str] = []
    pair_groups: list[tuple[str, str, str]] = []
    for image_relative, target_relative in pairs:
        match = compiled_regex.fullmatch(image_relative)
        if match is None or not match.group("group_id"):
            unmatched.append(image_relative)
            continue
        pair_groups.append(
            (image_relative, target_relative, match.group("group_id"))
        )
    if unmatched:
        raise ProtocolError(
            f"group_regex unmatched image(s): count={len(unmatched)}; "
            f"first 20={unmatched[:20]}"
        )

    assignments = _assign_groups(
        dataset_id,
        seed,
        {group_id for _, _, group_id in pair_groups},
        weights,
    )
    samples: list[dict[str, object]] = []
    index_entries: list[dict[str, object]] = []
    all_tokens: set[str] = set()
    target_lengths: dict[str, list[int]] = {split: [] for split in SPLITS}

    for image_relative, target_relative, group_id in pair_groups:
        sample_id = image_relative[: -len(IMAGE_SUFFIX)]
        image_path = source.joinpath(*PurePosixPath(image_relative).parts)
        target_path = source.joinpath(*PurePosixPath(target_relative).parts)
        image_stat_before = image_path.stat()
        image_sha256 = _sha256_file(image_path)
        image_stat_after = image_path.stat()
        if (
            image_stat_before.st_size != image_stat_after.st_size
            or image_stat_before.st_mtime_ns != image_stat_after.st_mtime_ns
        ):
            raise ProtocolError(
                f"image changed while hashing: sample={sample_id!r}"
            )
        target_payload, tokens = _read_target(target_path, sample_id)
        target_sha256 = hashlib.sha256(target_payload).hexdigest()
        split = assignments[group_id]
        all_tokens.update(tokens)
        target_lengths[split].append(len(tokens))
        samples.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "image_path": image_relative,
                "image_size_bytes": image_stat_after.st_size,
                "image_sha256": image_sha256,
                "target_path": target_relative,
                "target_sha256": target_sha256,
                "split": split,
            }
        )
        index_entries.append(
            {
                "image_path": image_relative,
                "image_size_bytes": image_stat_after.st_size,
                "image_mtime_ns": image_stat_after.st_mtime_ns,
                "image_sha256": image_sha256,
            }
        )

    samples.sort(key=lambda sample: _utf8_key(str(sample["sample_id"])))
    index_entries.sort(
        key=lambda entry: _utf8_key(str(entry["image_path"]))
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "group_semantics": "user_declared_regex",
        "samples": samples,
    }
    manifest_sha256 = canonical_sha256(manifest)
    vocabulary_document = {
        "schema_version": "staff_omr_vocab_v1",
        "dataset_id": dataset_id,
        "vocabulary_scope": "closed_corpus",
        "source_manifest_sha256": manifest_sha256,
        "target_parser": TARGET_PARSER,
        "token_sort": TOKEN_SORT,
        "unicode_normalization": "none",
        "blank_id": 0,
        "tokens": sorted(all_tokens, key=_utf8_key),
    }
    vocabulary_sha256 = canonical_sha256(vocabulary_document)
    image_index = {
        "schema_version": IMAGE_INDEX_SCHEMA,
        "source_manifest_sha256": manifest_sha256,
        "entries": index_entries,
    }
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "dataset_id": dataset_id,
        "prepare_protocol_version": PREPARE_PROTOCOL,
        "manifest_file": MANIFEST_FILE,
        "manifest_sha256": manifest_sha256,
        "vocabulary_file": VOCABULARY_FILE,
        "vocabulary_sha256": vocabulary_sha256,
        "image_verification_index_file": IMAGE_INDEX_FILE,
        "generation_contract": {
            "pair_rule": PAIR_RULE,
            "image_suffix": IMAGE_SUFFIX,
            "target_suffix": TARGET_SUFFIX,
            "group_regex": group_regex,
            "split_weights": list(weights),
            "seed": seed,
            "split_algorithm": SPLIT_ALGORITHM,
            "target_parser": TARGET_PARSER,
            "token_sort": TOKEN_SORT,
        },
    }
    bundle_sha256 = canonical_sha256(bundle)

    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    try:
        temporary.mkdir()
        write_canonical_json(temporary / MANIFEST_FILE, manifest)
        write_canonical_json(temporary / VOCABULARY_FILE, vocabulary_document)
        write_canonical_json(temporary / IMAGE_INDEX_FILE, image_index)
        write_canonical_json(temporary / BUNDLE_FILE, bundle)
        load_dataset_bundle(
            temporary,
            source,
            verify_image_hashes="always",
        )
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    split_group_counts = {
        split: sum(assigned == split for assigned in assignments.values())
        for split in SPLITS
    }
    split_sample_counts = {
        split: sum(sample["split"] == split for sample in samples)
        for split in SPLITS
    }
    return PrepareDataReport(
        output_path=output,
        dataset_id=dataset_id,
        bundle_sha256=bundle_sha256,
        manifest_sha256=manifest_sha256,
        vocabulary_sha256=vocabulary_sha256,
        split_group_counts=split_group_counts,
        split_sample_counts=split_sample_counts,
        target_length_summaries={
            split: _distribution(target_lengths[split]) for split in SPLITS
        },
    )


def load_dataset_bundle(
    bundle_path: str | Path,
    data_path: str | Path,
    *,
    verify_image_hashes: str = "always",
) -> ValidatedDatasetBundle:
    """Load and cross-validate a complete dataset bundle and its source data."""
    if verify_image_hashes not in {"always", "cached"}:
        raise ProtocolError(
            "verify_image_hashes must be 'always' or 'cached'"
        )
    bundle_root = Path(bundle_path).expanduser()
    source_root = Path(data_path).expanduser()
    try:
        bundle_root = bundle_root.resolve(strict=True)
        source_root = source_root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"bundle_path/data_path does not exist: {exc}") from exc
    if not bundle_root.is_dir():
        raise ProtocolError(f"bundle_path must be a directory: {bundle_root}")
    if not source_root.is_dir():
        raise ProtocolError(f"data_path must be a directory: {source_root}")

    bundle = _require_object(read_json(bundle_root / BUNDLE_FILE), "bundle")
    _require_exact_keys(
        bundle,
        {
            "schema_version",
            "dataset_id",
            "prepare_protocol_version",
            "manifest_file",
            "manifest_sha256",
            "vocabulary_file",
            "vocabulary_sha256",
            "image_verification_index_file",
            "generation_contract",
        },
        "bundle",
    )
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ProtocolError(
            f"unsupported bundle schema {bundle['schema_version']!r}"
        )
    if bundle["prepare_protocol_version"] != PREPARE_PROTOCOL:
        raise ProtocolError(
            "unsupported prepare_protocol_version "
            f"{bundle['prepare_protocol_version']!r}"
        )
    dataset_id = _require_nonempty_string(
        bundle["dataset_id"], "bundle.dataset_id"
    )
    generation_contract = _validate_generation_contract(
        bundle["generation_contract"]
    )
    try:
        group_regex = re.compile(generation_contract["group_regex"])
    except re.error as exc:
        raise ProtocolError(f"invalid generation group_regex: {exc}") from exc
    if "group_id" not in group_regex.groupindex:
        raise ProtocolError(
            "generation group_regex must define named capture group 'group_id'"
        )

    manifest_path = _safe_member_path(
        bundle_root, bundle["manifest_file"], MANIFEST_FILE
    )
    vocabulary_path = _safe_member_path(
        bundle_root, bundle["vocabulary_file"], VOCABULARY_FILE
    )
    index_path = _safe_member_path(
        bundle_root,
        bundle["image_verification_index_file"],
        IMAGE_INDEX_FILE,
    )
    manifest = _require_object(read_json(manifest_path), "manifest")
    vocabulary_document = _require_object(
        read_json(vocabulary_path), "vocabulary"
    )
    index = _require_object(read_json(index_path), "image verification index")

    expected_manifest_hash = _require_sha256(
        bundle["manifest_sha256"], "bundle.manifest_sha256"
    )
    expected_vocabulary_hash = _require_sha256(
        bundle["vocabulary_sha256"], "bundle.vocabulary_sha256"
    )
    actual_manifest_hash = canonical_sha256(manifest)
    actual_vocabulary_hash = canonical_sha256(vocabulary_document)
    if actual_manifest_hash != expected_manifest_hash:
        raise ProtocolError(
            "manifest canonical SHA-256 does not match bundle: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    if actual_vocabulary_hash != expected_vocabulary_hash:
        raise ProtocolError(
            "vocabulary canonical SHA-256 does not match bundle: "
            f"expected={expected_vocabulary_hash}, actual={actual_vocabulary_hash}"
        )

    _require_exact_keys(
        manifest,
        {"schema_version", "dataset_id", "group_semantics", "samples"},
        "manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ProtocolError(
            f"unsupported manifest schema {manifest['schema_version']!r}"
        )
    if manifest["dataset_id"] != dataset_id:
        raise ProtocolError(
            "bundle and manifest dataset_id values do not match"
        )
    if manifest["group_semantics"] != "user_declared_regex":
        raise ProtocolError(
            "manifest group_semantics must be 'user_declared_regex'"
        )

    vocabulary = Vocabulary.from_document(
        vocabulary_document,
        manifest_sha256=actual_manifest_hash,
        dataset_id=dataset_id,
    )
    if vocabulary_document["dataset_id"] != dataset_id:
        raise ProtocolError("vocabulary dataset_id mismatch")

    _require_exact_keys(
        index,
        {"schema_version", "source_manifest_sha256", "entries"},
        "image verification index",
    )
    if index["schema_version"] != IMAGE_INDEX_SCHEMA:
        raise ProtocolError(
            f"unsupported image verification schema {index['schema_version']!r}"
        )
    if index["source_manifest_sha256"] != actual_manifest_hash:
        raise ProtocolError(
            "image verification index source_manifest_sha256 mismatch"
        )
    raw_entries = index["entries"]
    if not isinstance(raw_entries, list):
        raise ProtocolError("image verification index entries must be an array")
    index_by_path: dict[str, dict[str, Any]] = {}
    for position, raw_entry in enumerate(raw_entries):
        entry = _require_object(
            raw_entry, f"image verification index entry {position}"
        )
        _require_exact_keys(
            entry,
            {
                "image_path",
                "image_size_bytes",
                "image_mtime_ns",
                "image_sha256",
            },
            f"image verification index entry {position}",
        )
        image_relative = _require_nonempty_string(
            entry["image_path"], f"image verification entry {position}.image_path"
        )
        if image_relative in index_by_path:
            raise ProtocolError(
                f"verification index contains duplicate image_path {image_relative!r}"
            )
        _require_non_negative_int(
            entry["image_size_bytes"],
            f"verification entry {image_relative!r}.image_size_bytes",
        )
        _require_non_negative_int(
            entry["image_mtime_ns"],
            f"verification entry {image_relative!r}.image_mtime_ns",
        )
        _require_sha256(
            entry["image_sha256"],
            f"verification entry {image_relative!r}.image_sha256",
        )
        index_by_path[image_relative] = entry
    ordered_index_paths = sorted(index_by_path, key=_utf8_key)
    if [entry["image_path"] for entry in raw_entries] != ordered_index_paths:
        raise ProtocolError(
            "image verification index entries are not sorted by image_path"
        )

    raw_samples = manifest["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ProtocolError("manifest samples must be a non-empty array")
    seen_sample_ids: set[str] = set()
    seen_files: set[str] = set()
    group_splits: dict[str, str] = {}
    split_counts = Counter()
    samples: list[BundleSample] = []
    manifest_image_paths: set[str] = set()

    for position, raw_sample in enumerate(raw_samples):
        sample = _require_object(raw_sample, f"manifest sample {position}")
        _require_exact_keys(
            sample,
            {
                "sample_id",
                "group_id",
                "image_path",
                "image_size_bytes",
                "image_sha256",
                "target_path",
                "target_sha256",
                "split",
            },
            f"manifest sample {position}",
        )
        sample_id = _require_nonempty_string(
            sample["sample_id"], f"manifest sample {position}.sample_id"
        )
        if sample_id in seen_sample_ids:
            raise ProtocolError(f"duplicate sample_id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        group_id = _require_nonempty_string(
            sample["group_id"], f"sample {sample_id!r}.group_id"
        )
        split = sample["split"]
        if split not in SPLITS:
            raise ProtocolError(
                f"sample {sample_id!r} has invalid split {split!r}"
            )
        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            raise ProtocolError(
                f"group_id {group_id!r} appears in multiple splits: "
                f"{previous_split!r}, {split!r}"
            )
        split_counts[split] += 1

        image_relative, image_path = _safe_data_file(
            source_root,
            sample["image_path"],
            f"sample {sample_id!r} image_path escape check",
        )
        target_relative, target_path = _safe_data_file(
            source_root,
            sample["target_path"],
            f"sample {sample_id!r} target_path escape check",
        )
        if not image_relative.endswith(IMAGE_SUFFIX):
            raise ProtocolError(
                f"sample {sample_id!r} image_path must end with {IMAGE_SUFFIX!r}"
            )
        expected_sample_id = image_relative[: -len(IMAGE_SUFFIX)]
        expected_target = expected_sample_id + TARGET_SUFFIX
        if sample_id != expected_sample_id:
            raise ProtocolError(
                f"sample_id {sample_id!r} does not match image_path "
                f"{image_relative!r}"
            )
        if target_relative != expected_target:
            raise ProtocolError(
                f"sample {sample_id!r} target_path does not follow pair rule"
            )
        match = group_regex.fullmatch(image_relative)
        if match is None or match.group("group_id") != group_id:
            raise ProtocolError(
                f"sample {sample_id!r} group_id does not match generation regex"
            )

        for resolved in (image_path, target_path):
            file_identity = os.path.normcase(str(resolved))
            if file_identity in seen_files:
                raise ProtocolError(
                    f"data file is referenced by multiple samples: {resolved}"
                )
            seen_files.add(file_identity)

        image_size = _require_non_negative_int(
            sample["image_size_bytes"],
            f"sample {sample_id!r}.image_size_bytes",
        )
        image_sha256 = _require_sha256(
            sample["image_sha256"],
            f"sample {sample_id!r}.image_sha256",
        )
        target_sha256 = _require_sha256(
            sample["target_sha256"],
            f"sample {sample_id!r}.target_sha256",
        )
        actual_image_size = image_path.stat().st_size
        if actual_image_size != image_size:
            raise ProtocolError(
                f"image size mismatch for sample {sample_id!r}: "
                f"expected={image_size}, actual={actual_image_size}"
            )
        target_payload, target_tokens = _read_target(target_path, sample_id)
        actual_target_hash = hashlib.sha256(target_payload).hexdigest()
        if actual_target_hash != target_sha256:
            raise ProtocolError(
                f"target SHA-256 mismatch for sample {sample_id!r}: "
                f"expected={target_sha256}, actual={actual_target_hash}"
            )
        try:
            vocabulary.encode(target_tokens)
        except ProtocolError as exc:
            raise ProtocolError(
                f"OOV in split={split!r}, sample_id={sample_id!r}: {exc}"
            ) from exc

        manifest_image_paths.add(image_relative)
        samples.append(
            BundleSample(
                sample_id=sample_id,
                group_id=group_id,
                image_path=image_relative,
                image_size_bytes=image_size,
                image_sha256=image_sha256,
                target_path=target_relative,
                target_sha256=target_sha256,
                split=split,
                target_tokens=target_tokens,
            )
        )

    for split in SPLITS:
        if split_counts[split] == 0:
            raise ProtocolError(f"manifest split {split!r} is empty")
    expected_sample_order = sorted(seen_sample_ids, key=_utf8_key)
    if [sample["sample_id"] for sample in raw_samples] != expected_sample_order:
        raise ProtocolError("manifest samples are not sorted by sample_id")

    index_paths = set(index_by_path)
    if index_paths != manifest_image_paths:
        missing = sorted(manifest_image_paths - index_paths, key=_utf8_key)
        extra = sorted(index_paths - manifest_image_paths, key=_utf8_key)
        raise ProtocolError(
            "verification index does not exactly cover manifest images; "
            f"missing={missing}, extra={extra}"
        )

    hits = 0
    recomputed = 0
    for sample in samples:
        entry = index_by_path[sample.image_path]
        if (
            entry["image_size_bytes"] != sample.image_size_bytes
            or entry["image_sha256"] != sample.image_sha256
        ):
            raise ProtocolError(
                f"verification index metadata mismatch for {sample.image_path!r}"
            )
        image_path = source_root.joinpath(
            *PurePosixPath(sample.image_path).parts
        )
        current_stat = image_path.stat()
        cache_hit = (
            verify_image_hashes == "cached"
            and current_stat.st_size == entry["image_size_bytes"]
            and current_stat.st_mtime_ns == entry["image_mtime_ns"]
        )
        if cache_hit:
            hits += 1
            continue
        recomputed += 1
        actual_image_hash = _sha256_file(image_path)
        if actual_image_hash != sample.image_sha256:
            raise ProtocolError(
                f"image SHA-256 mismatch for sample {sample.sample_id!r}: "
                f"expected={sample.image_sha256}, actual={actual_image_hash}"
            )

    status = (
        "content_verified"
        if verify_image_hashes == "always"
        else "cached_metadata"
    )
    return ValidatedDatasetBundle(
        bundle_path=bundle_root,
        data_path=source_root,
        dataset_id=dataset_id,
        dataset_bundle_sha256=canonical_sha256(bundle),
        manifest_sha256=actual_manifest_hash,
        vocabulary_sha256=actual_vocabulary_hash,
        vocabulary=vocabulary,
        samples=tuple(samples),
        image_verification=ImageVerificationStats(
            mode=verify_image_hashes,
            hits=hits,
            recomputed=recomputed,
            status=status,
            trusted_baseline=verify_image_hashes == "always",
        ),
    )
