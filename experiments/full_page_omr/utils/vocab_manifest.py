from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


POLISH_PREFIX_SHA256 = (
    "3821e7f0d5defd55fe73ce8b7f229f48f688bc490bdae854bc824a6145dac49b"
)
_PROJECT_SEED_EXTRAS = (
    "*M6/16",
    "*staff1",
    "*staff2",
    "88",
    "=:|!;",
    "==;",
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_token_sha256(tokens: Sequence[str]) -> str:
    return canonical_json_sha256(list(tokens))


def _load_numpy_mapping(path: Path, field_name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} file does not exist: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must contain a dictionary")
    return value


def _normalize_id(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} contains a boolean id")
    if not isinstance(value, Integral):
        raise TypeError(f"{field_name} ids must be integers")
    return int(value)


def load_legacy_ordered_tokens(
    w2i_path: str | Path,
    i2w_path: str | Path,
) -> tuple[str, ...]:
    w2i_raw = _load_numpy_mapping(Path(w2i_path), "w2i")
    i2w_raw = _load_numpy_mapping(Path(i2w_path), "i2w")

    w2i: dict[str, int] = {}
    for token, token_id in w2i_raw.items():
        if not isinstance(token, str):
            raise TypeError("w2i tokens must be strings")
        w2i[token] = _normalize_id(token_id, "w2i")

    i2w: dict[int, str] = {}
    for token_id, token in i2w_raw.items():
        normalized_id = _normalize_id(token_id, "i2w")
        if not isinstance(token, str):
            raise TypeError("i2w tokens must be strings")
        i2w[normalized_id] = token

    expected_ids = set(range(len(w2i)))
    if set(w2i.values()) != expected_ids or set(i2w) != expected_ids:
        raise ValueError("legacy vocabulary ids must be contiguous from zero")
    if len(set(i2w.values())) != len(i2w):
        raise ValueError("legacy vocabulary contains duplicate tokens")
    if len(w2i) != len(i2w):
        raise ValueError("w2i and i2w are not strict inverses")
    if any(i2w[token_id] != token for token, token_id in w2i.items()):
        raise ValueError("w2i and i2w are not strict inverses")

    return tuple(i2w[token_id] for token_id in range(len(i2w)))


def _legacy_pair(vocab_dir: Path, name: str) -> tuple[Path, Path]:
    return (
        vocab_dir / f"{name}w2i.npy",
        vocab_dir / f"{name}i2w.npy",
    )


def build_project_seed_tokens(vocab_dir: str | Path) -> tuple[str, ...]:
    root = Path(vocab_dir)
    polish = load_legacy_ordered_tokens(
        *_legacy_pair(root, "Polish_Scores_BeKern")
    )
    if len(polish) != 215:
        raise ValueError(f"Polish vocabulary must contain 215 tokens, got {len(polish)}")
    actual_digest = ordered_token_sha256(polish)
    if actual_digest != POLISH_PREFIX_SHA256:
        raise ValueError(
            "Polish vocabulary digest mismatch: "
            f"expected={POLISH_PREFIX_SHA256}, actual={actual_digest}"
        )

    mozarteum = load_legacy_ordered_tokens(
        *_legacy_pair(root, "Mozarteum_BeKern")
    )
    fp_grandstaff = load_legacy_ordered_tokens(
        *_legacy_pair(root, "FP_GrandStaff_BeKern")
    )
    extras = tuple(
        sorted(
            (set(mozarteum) | set(fp_grandstaff)) - set(polish),
            key=lambda token: token.encode("utf-8"),
        )
    )
    if extras != _PROJECT_SEED_EXTRAS:
        raise ValueError(
            "existing vocabulary compatibility tokens changed: "
            f"expected={_PROJECT_SEED_EXTRAS!r}, actual={extras!r}"
        )
    return (*polish, *extras)


def _validate_ordered_tokens(tokens: Sequence[str], field_name: str) -> tuple[str, ...]:
    ordered = tuple(tokens)
    if any(not isinstance(token, str) for token in ordered):
        raise TypeError(f"{field_name} must contain only strings")
    if any(token == "" for token in ordered):
        raise ValueError(f"{field_name} cannot contain empty tokens")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{field_name} contains duplicate tokens")
    return ordered


def extend_ordered_tokens(
    base_tokens: Sequence[str],
    token_sequences: Iterable[Iterable[str]],
) -> tuple[str, ...]:
    base = _validate_ordered_tokens(base_tokens, "base_tokens")
    base_set = set(base)
    additions: set[str] = set()
    for sequence in token_sequences:
        for token in sequence:
            if not isinstance(token, str):
                raise TypeError("token sequences must contain only strings")
            if token == "":
                raise ValueError("token sequences cannot contain empty tokens")
            if token not in base_set:
                additions.add(token)
    return (
        *base,
        *sorted(additions, key=lambda token: token.encode("utf-8")),
    )


@dataclass(frozen=True)
class VocabularyManifest:
    schema_version: int
    name: str
    tokenization_mode: str
    base_name: str
    base_size: int
    base_digest: str
    ordered_tokens: tuple[str, ...]
    token_provenance: dict[str, str]
    source_dataset_manifests: tuple[str, ...]
    vocab_sha256: str

    @property
    def w2i(self) -> dict[str, int]:
        return {token: token_id for token_id, token in enumerate(self.ordered_tokens)}

    @property
    def i2w(self) -> dict[int, str]:
        return {token_id: token for token_id, token in enumerate(self.ordered_tokens)}


_MANIFEST_FIELDS = {
    "schema_version",
    "name",
    "tokenization_mode",
    "base_name",
    "base_size",
    "base_digest",
    "ordered_tokens",
    "token_provenance",
    "source_dataset_manifests",
    "vocab_sha256",
}


def _validate_manifest(manifest: VocabularyManifest) -> None:
    if manifest.schema_version != 1:
        raise ValueError(
            f"unsupported vocabulary schema version: {manifest.schema_version}"
        )
    for field_name in ("name", "base_name", "base_digest", "vocab_sha256"):
        value = getattr(manifest, field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")
    if manifest.tokenization_mode != "bekern":
        raise ValueError("vocabulary tokenization_mode must be 'bekern'")

    tokens = _validate_ordered_tokens(manifest.ordered_tokens, "ordered_tokens")
    if (
        isinstance(manifest.base_size, bool)
        or not isinstance(manifest.base_size, int)
        or not 0 <= manifest.base_size <= len(tokens)
    ):
        raise ValueError("base_size must be within ordered_tokens")
    actual_base_digest = ordered_token_sha256(tokens[: manifest.base_size])
    if actual_base_digest != manifest.base_digest:
        raise ValueError(
            "base vocabulary SHA-256 mismatch: "
            f"expected={manifest.base_digest}, actual={actual_base_digest}"
        )
    actual_digest = ordered_token_sha256(tokens)
    if actual_digest != manifest.vocab_sha256:
        raise ValueError(
            "vocabulary SHA-256 mismatch: "
            f"expected={manifest.vocab_sha256}, actual={actual_digest}"
        )
    if any(
        not isinstance(token, str) or not isinstance(source, str)
        for token, source in manifest.token_provenance.items()
    ):
        raise TypeError("token_provenance must map strings to strings")
    if any(
        not isinstance(digest, str) or not digest
        for digest in manifest.source_dataset_manifests
    ):
        raise TypeError("source_dataset_manifests must contain non-empty strings")


def _manifest_payload(manifest: VocabularyManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["ordered_tokens"] = list(manifest.ordered_tokens)
    payload["source_dataset_manifests"] = list(manifest.source_dataset_manifests)
    return payload


def load_vocabulary_manifest(path: str | Path) -> VocabularyManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("vocabulary manifest must contain a JSON object")
    if set(payload) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(payload))
        extra = sorted(set(payload) - _MANIFEST_FIELDS)
        raise ValueError(
            f"vocabulary manifest fields mismatch: missing={missing}, extra={extra}"
        )
    payload["ordered_tokens"] = tuple(payload["ordered_tokens"])
    payload["source_dataset_manifests"] = tuple(
        payload["source_dataset_manifests"]
    )
    manifest = VocabularyManifest(**payload)
    _validate_manifest(manifest)
    return manifest


def write_vocabulary_manifest(
    manifest: VocabularyManifest,
    path: str | Path,
) -> Path:
    _validate_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        _manifest_payload(manifest),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    new_bytes = f"{serialized}\n".encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != new_bytes:
        raise FileExistsError(
            f"refusing to overwrite a different vocabulary manifest: {output_path}"
        )
    output_path.write_bytes(new_bytes)
    return output_path


def write_legacy_numpy_pair(
    manifest: VocabularyManifest,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    _validate_manifest(manifest)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    w2i_path = root / f"{manifest.name}w2i.npy"
    i2w_path = root / f"{manifest.name}i2w.npy"

    if w2i_path.exists() or i2w_path.exists():
        if not (w2i_path.exists() and i2w_path.exists()):
            raise FileExistsError("legacy vocabulary pair is incomplete")
        existing = load_legacy_ordered_tokens(w2i_path, i2w_path)
        if existing != manifest.ordered_tokens:
            raise FileExistsError(
                "refusing to overwrite a different legacy vocabulary pair"
            )
        return w2i_path, i2w_path

    np.save(w2i_path, manifest.w2i)
    np.save(i2w_path, manifest.i2w)
    round_trip = load_legacy_ordered_tokens(w2i_path, i2w_path)
    if round_trip != manifest.ordered_tokens:
        raise RuntimeError("legacy vocabulary round trip changed token order")
    return w2i_path, i2w_path
