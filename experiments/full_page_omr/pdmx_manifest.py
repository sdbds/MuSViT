from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
from typing import Any, Mapping, Sequence

from fire import Fire
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError

from .tokenization import parse_kern_file
from .utils.vocab_manifest import (
    VocabularyManifest,
    build_project_seed_tokens,
    canonical_json_sha256,
    extend_ordered_tokens,
    load_legacy_ordered_tokens,
    load_vocabulary_manifest,
    ordered_token_sha256,
    write_legacy_numpy_pair,
    write_vocabulary_manifest,
)


PDMX_DATASET_ID = "tobiashornbogen/page-omr-pdmx-renders"
PDMX_DATASET_REVISION = "7da3ae5237963e57a8fe1c6ee375b1f10af34a09"
PDMX_LICENSE = "CC-BY-4.0"
PDMX_RENDERER_WEIGHTS = {"mscore": 0.5, "verovio": 0.5}

_TRAIN_ROOTS = {
    "verovio": "shards_pdmx_a1_broadened_518x728",
    "mscore": "shards_pdmx_a1_mscore_518x728",
}
_VALIDATION_PATH = "curated_validation_pdmx.tar"
_REQUIRED_SUFFIXES = frozenset({"image.png", "kern.txt", "source.txt"})
_OPTIONAL_SUFFIXES = frozenset({"fill.txt"})
_KNOWN_SUFFIXES = tuple(
    sorted(
        _REQUIRED_SUFFIXES | _OPTIONAL_SUFFIXES,
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class PDMXSampleRecord:
    key: str
    source_id: str
    fill: str | None
    sequence_length: int
    tokens: tuple[str, ...]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ShardScanResult:
    renderer: str
    logical_path: str
    sha256: str
    bytes: int
    sample_count: int
    records: tuple[PDMXSampleRecord, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("source id must be a string")
    normalized = value.strip().replace("\\", "/")
    components = []
    for component in normalized.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            raise ValueError(f"source id contains parent traversal: {value!r}")
        components.append(component)
    if not components:
        raise ValueError("source id cannot be empty")
    return PurePosixPath(*components).as_posix()


def _split_member_name(name: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        raise ValueError(f"unsafe PDMX tar member: {name!r}")
    normalized = name.replace("\\", "/")
    components = normalized.split("/")
    windows_path = PureWindowsPath(name)
    if (
        PurePosixPath(normalized).is_absolute()
        or bool(windows_path.drive)
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise ValueError(f"unsafe PDMX tar member: {name!r}")
    for suffix in _KNOWN_SUFFIXES:
        marker = f".{suffix}"
        if normalized.endswith(marker):
            key = normalized[: -len(marker)]
            if not key:
                break
            return key, suffix
    raise ValueError(f"unrecognized PDMX tar member: {name}")


def _decode_utf8(
    payload: bytes,
    *,
    logical_path: str,
    key: str,
    suffix: str,
) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{logical_path}: sample {key!r} has invalid UTF-8 in {suffix}"
        ) from error


def _read_tar_records(
    tar_path: Path,
    *,
    logical_path: str,
) -> dict[str, dict[str, bytes]]:
    records: dict[str, dict[str, bytes]] = {}
    try:
        archive = tarfile.open(tar_path, "r:*")
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot open PDMX tar {logical_path}: {error}") from error
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            try:
                key, suffix = _split_member_name(member.name)
            except ValueError as error:
                raise ValueError(f"{logical_path}: {error}") from error
            fields = records.setdefault(key, {})
            if suffix in fields:
                raise ValueError(
                    f"{logical_path}: sample {key!r} has duplicate {suffix}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(
                    f"{logical_path}: cannot read sample {key!r} field {suffix}"
                )
            payload = handle.read()
            if len(payload) != member.size:
                raise ValueError(
                    f"{logical_path}: sample {key!r} field {suffix} is truncated"
                )
            fields[suffix] = payload
    return records


def scan_pdmx_tar(
    tar_path: str | Path,
    *,
    renderer: str,
    logical_path: str,
    max_sequence_length: int = 7512,
) -> ShardScanResult:
    path = Path(tar_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDMX shard does not exist: {path}")
    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length < 2
    ):
        raise ValueError("max_sequence_length must be an integer of at least 2")
    if not isinstance(renderer, str) or not renderer:
        raise ValueError("renderer must be a non-empty string")

    grouped = _read_tar_records(path, logical_path=logical_path)
    if not grouped:
        raise ValueError(f"{logical_path}: tar contains no PDMX samples")

    records = []
    for key in sorted(grouped):
        fields = grouped[key]
        missing = sorted(_REQUIRED_SUFFIXES - set(fields))
        if missing:
            raise ValueError(
                f"{logical_path}: sample {key!r} is missing {', '.join(missing)}"
            )

        kern = _decode_utf8(
            fields["kern.txt"],
            logical_path=logical_path,
            key=key,
            suffix="kern.txt",
        )
        content_tokens = tuple(
            token
            for token in parse_kern_file(kern, tokenization_mode="bekern")
            if token != ""
        )
        if not content_tokens:
            raise ValueError(
                f"{logical_path}: sample {key!r} has an empty tokenized target"
            )
        tokens = ("<bos>", *content_tokens, "<eos>")
        if len(tokens) > max_sequence_length:
            raise ValueError(
                f"{logical_path}: sample {key!r} sequence length {len(tokens)} "
                f"exceeds {max_sequence_length}"
            )

        source_id = normalize_source_id(
            _decode_utf8(
                fields["source.txt"],
                logical_path=logical_path,
                key=key,
                suffix="source.txt",
            )
        )
        fill = None
        if "fill.txt" in fields:
            fill = _decode_utf8(
                fields["fill.txt"],
                logical_path=logical_path,
                key=key,
                suffix="fill.txt",
            ).strip()
            if not fill:
                fill = None

        try:
            with Image.open(BytesIO(fields["image.png"])) as image:
                if image.format != "PNG":
                    raise ValueError(
                        f"{logical_path}: sample {key!r} image is not PNG"
                    )
                image.load()
                image_size = tuple(image.size)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(
                f"{logical_path}: sample {key!r} has an invalid image.png"
            ) from error

        records.append(
            PDMXSampleRecord(
                key=key,
                source_id=source_id,
                fill=fill,
                sequence_length=len(tokens),
                tokens=tokens,
                image_size=image_size,
            )
        )

    return ShardScanResult(
        renderer=renderer,
        logical_path=PurePosixPath(logical_path).as_posix(),
        sha256=_sha256_file(path),
        bytes=path.stat().st_size,
        sample_count=len(records),
        records=tuple(records),
    )


def discover_official_pdmx_shards(
    snapshot_root: str | Path,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    root = Path(snapshot_root)
    train = []
    for renderer, logical_root in _TRAIN_ROOTS.items():
        renderer_root = root / logical_root
        if not renderer_root.is_dir():
            raise FileNotFoundError(
                f"missing {renderer} shard root: {renderer_root}"
            )
        for path in renderer_root.glob("**/pdmx-*.tar"):
            if path.is_file():
                train.append((renderer, path.relative_to(root).as_posix()))
    if not train:
        raise FileNotFoundError(f"no official PDMX train shards found under {root}")

    validation_path = root / _VALIDATION_PATH
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"missing curated Kern validation shard: {validation_path}"
        )
    return (
        tuple(sorted(train, key=lambda item: item[1])),
        (_VALIDATION_PATH,),
    )


def _validate_renderer_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if set(weights) != set(PDMX_RENDERER_WEIGHTS):
        raise ValueError(
            f"renderer_weights must have keys {sorted(PDMX_RENDERER_WEIGHTS)}"
        )
    normalized = {}
    for renderer in sorted(weights):
        value = weights[renderer]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("renderer weights must be numbers")
        normalized[renderer] = float(value)
    if normalized != PDMX_RENDERER_WEIGHTS:
        raise ValueError(
            f"PDMX v1 renderer_weights must be {PDMX_RENDERER_WEIGHTS}"
        )
    return normalized


def _resolve_logical_path(root: Path, logical_path: str) -> Path:
    pure_path = PurePosixPath(logical_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"invalid logical shard path: {logical_path!r}")
    candidate = (root / Path(*pure_path.parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"logical shard path escapes snapshot: {logical_path!r}")
    return candidate


def _bucket_fields(logical_path: str) -> tuple[str | None, str | None]:
    parent = PurePosixPath(logical_path).parent.name
    if "_" not in parent:
        return None, None
    bucket, density = parent.rsplit("_", 1)
    if density not in {"short", "medium", "full"}:
        return parent, None
    return bucket, density


def _shard_payload(scan: ShardScanResult) -> dict[str, Any]:
    bucket, density = _bucket_fields(scan.logical_path)
    return {
        "renderer": scan.renderer,
        "logical_path": scan.logical_path,
        "sha256": scan.sha256,
        "bytes": scan.bytes,
        "sample_count": scan.sample_count,
        "voice_bucket": bucket,
        "density_bucket": density,
    }


def _length_summary(lengths: Sequence[int]) -> dict[str, int | float]:
    if not lengths:
        raise ValueError("cannot summarize an empty sequence collection")
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        token: counter[token]
        for token in sorted(counter, key=lambda item: item.encode("utf-8"))
    }


def _manifest_digest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }


def build_pdmx_dataset_manifest(
    *,
    snapshot_root: str | Path,
    dataset_id: str,
    dataset_revision: str,
    train_shards: Sequence[tuple[str, str]],
    validation_shards: Sequence[str],
    renderer_weights: Mapping[str, float],
    max_sequence_length: int = 7512,
) -> dict[str, Any]:
    if dataset_id != PDMX_DATASET_ID:
        raise ValueError(f"dataset_id must be {PDMX_DATASET_ID!r}")
    if dataset_revision != PDMX_DATASET_REVISION:
        raise ValueError(f"dataset_revision must be {PDMX_DATASET_REVISION}")
    weights = _validate_renderer_weights(renderer_weights)
    root = Path(snapshot_root)

    normalized_train = sorted(
        ((renderer, PurePosixPath(path).as_posix()) for renderer, path in train_shards),
        key=lambda item: item[1],
    )
    if not normalized_train:
        raise ValueError("at least one train shard is required")
    logical_paths = [logical_path for _, logical_path in normalized_train]
    if len(set(logical_paths)) != len(logical_paths):
        raise ValueError("duplicate train shard logical path")
    if {renderer for renderer, _ in normalized_train} != set(weights):
        raise ValueError("train shards must include both configured renderers")
    normalized_validation = sorted(
        PurePosixPath(path).as_posix() for path in validation_shards
    )
    if normalized_validation != [_VALIDATION_PATH]:
        raise ValueError(
            f"validation_shards must contain only {_VALIDATION_PATH!r}"
        )

    train_scans = [
        scan_pdmx_tar(
            _resolve_logical_path(root, logical_path),
            renderer=renderer,
            logical_path=logical_path,
            max_sequence_length=max_sequence_length,
        )
        for renderer, logical_path in normalized_train
    ]
    validation_scans = [
        scan_pdmx_tar(
            _resolve_logical_path(root, logical_path),
            renderer="validation",
            logical_path=logical_path,
            max_sequence_length=max_sequence_length,
        )
        for logical_path in normalized_validation
    ]

    seen_train_keys: set[tuple[str, str]] = set()
    for scan in train_scans:
        for record in scan.records:
            identity = (scan.renderer, record.key)
            if identity in seen_train_keys:
                raise ValueError(
                    f"duplicate sample key for {scan.renderer}: {record.key}"
                )
            seen_train_keys.add(identity)

    train_records = [
        record
        for scan in train_scans
        for record in scan.records
    ]
    validation_records = [
        record
        for scan in validation_scans
        for record in scan.records
    ]
    train_sources = {record.source_id for record in train_records}
    validation_sources = {record.source_id for record in validation_records}
    overlap = sorted(train_sources & validation_sources)
    if overlap:
        preview = overlap[:10]
        raise ValueError(
            "train/validation source overlap detected: "
            f"count={len(overlap)}, examples={preview}"
        )

    train_tokens = Counter(
        token
        for record in train_records
        for token in record.tokens
    )
    validation_tokens = Counter(
        token
        for record in validation_records
        for token in record.tokens
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "license": PDMX_LICENSE,
        "selected_renderers": sorted(weights),
        "renderer_weights": weights,
        "tokenization_mode": "bekern",
        "max_sequence_length": max_sequence_length,
        "train_shards": [_shard_payload(scan) for scan in train_scans],
        "validation_shards": [
            _shard_payload(scan) for scan in validation_scans
        ],
        "train_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "train_token_frequencies": _sorted_counter(train_tokens),
        "validation_token_frequencies": _sorted_counter(validation_tokens),
        "train_sequence_lengths": _length_summary(
            [record.sequence_length for record in train_records]
        ),
        "validation_sequence_lengths": _length_summary(
            [record.sequence_length for record in validation_records]
        ),
        "excluded_samples": [],
        "train_validation_source_overlap": 0,
        "scan_tool_version": 1,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        _manifest_digest_payload(manifest)
    )
    return manifest


def _verify_manifest_self_digest(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError("dataset manifest is missing manifest_sha256")
    actual = canonical_json_sha256(_manifest_digest_payload(manifest))
    if actual != expected:
        raise ValueError(
            "dataset manifest SHA-256 mismatch: "
            f"expected={expected}, actual={actual}"
        )


def write_pdmx_dataset_manifest(
    manifest: Mapping[str, Any],
    path: str | Path,
) -> Path:
    _verify_manifest_self_digest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(manifest),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    new_bytes = f"{serialized}\n".encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != new_bytes:
        raise FileExistsError(
            f"refusing to overwrite a different dataset manifest: {output_path}"
        )
    output_path.write_bytes(new_bytes)
    return output_path


def load_pdmx_dataset_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("dataset manifest must contain a JSON object")
    _verify_manifest_self_digest(payload)
    if payload.get("dataset_id") != PDMX_DATASET_ID:
        raise ValueError("dataset manifest id mismatch")
    if payload.get("dataset_revision") != PDMX_DATASET_REVISION:
        raise ValueError("dataset manifest revision mismatch")
    _validate_renderer_weights(payload.get("renderer_weights", {}))
    return payload


def verify_pdmx_dataset_manifest(
    manifest: Mapping[str, Any],
    snapshot_root: str | Path,
) -> None:
    _verify_manifest_self_digest(manifest)
    if manifest.get("dataset_id") != PDMX_DATASET_ID:
        raise ValueError("dataset manifest id mismatch")
    if manifest.get("dataset_revision") != PDMX_DATASET_REVISION:
        raise ValueError("dataset manifest revision mismatch")
    _validate_renderer_weights(manifest.get("renderer_weights", {}))

    root = Path(snapshot_root)
    for split_name in ("train_shards", "validation_shards"):
        shards = manifest.get(split_name)
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"dataset manifest {split_name} must be a non-empty list")
        for shard in shards:
            logical_path = shard["logical_path"]
            path = _resolve_logical_path(root, logical_path)
            if not path.is_file():
                raise FileNotFoundError(f"manifest shard is missing: {logical_path}")
            actual_size = path.stat().st_size
            if actual_size != shard["bytes"]:
                raise ValueError(
                    f"{logical_path} size mismatch: "
                    f"expected={shard['bytes']}, actual={actual_size}"
                )
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != shard["sha256"]:
                raise ValueError(
                    f"{logical_path} SHA-256 mismatch: "
                    f"expected={shard['sha256']}, actual={actual_sha256}"
                )


def resolve_local_snapshot(
    dataset_id: str = PDMX_DATASET_ID,
    revision: str = PDMX_DATASET_REVISION,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    try:
        resolved = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=True,
        )
    except Exception as error:
        raise FileNotFoundError(
            "fixed PDMX snapshot is not available locally; run "
            "`python -m experiments.full_page_omr.pdmx_manifest prepare`"
        ) from error
    return Path(resolved)


def prepare(cache_dir: str | None = None) -> str:
    resolved = snapshot_download(
        repo_id=PDMX_DATASET_ID,
        repo_type="dataset",
        revision=PDMX_DATASET_REVISION,
        cache_dir=cache_dir,
        allow_patterns=[
            f"{_TRAIN_ROOTS['verovio']}/**/*.tar",
            f"{_TRAIN_ROOTS['mscore']}/**/*.tar",
            _VALIDATION_PATH,
            "README.md",
        ],
        local_files_only=False,
    )
    return str(Path(resolved))


def scan(
    output: str,
    snapshot_root: str | None = None,
    cache_dir: str | None = None,
) -> str:
    root = (
        Path(snapshot_root)
        if snapshot_root is not None
        else resolve_local_snapshot(cache_dir=cache_dir)
    )
    train_shards, validation_shards = discover_official_pdmx_shards(root)
    manifest = build_pdmx_dataset_manifest(
        snapshot_root=root,
        dataset_id=PDMX_DATASET_ID,
        dataset_revision=PDMX_DATASET_REVISION,
        train_shards=train_shards,
        validation_shards=validation_shards,
        renderer_weights=PDMX_RENDERER_WEIGHTS,
    )
    return str(write_pdmx_dataset_manifest(manifest, output))


def _frequency_tokens(
    manifest: Mapping[str, Any],
    field_name: str,
) -> tuple[str, ...]:
    frequencies = manifest.get(field_name)
    if not isinstance(frequencies, dict) or not frequencies:
        raise ValueError(f"dataset manifest {field_name} must be non-empty")
    for token, count in frequencies.items():
        if not isinstance(token, str) or not token:
            raise ValueError(f"dataset manifest {field_name} has invalid token")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"dataset manifest {field_name} has invalid count for {token!r}"
            )
    return tuple(frequencies)


def build_vocabulary(
    dataset_manifest: str,
    output: str,
    snapshot_root: str | None = None,
    cache_dir: str | None = None,
    vocab_dir: str | None = None,
) -> str:
    manifest = load_pdmx_dataset_manifest(dataset_manifest)
    root = (
        Path(snapshot_root)
        if snapshot_root is not None
        else resolve_local_snapshot(cache_dir=cache_dir)
    )
    verify_pdmx_dataset_manifest(manifest, root)

    project_vocab_dir = (
        Path(vocab_dir)
        if vocab_dir is not None
        else Path(__file__).resolve().parent / "vocab"
    )
    seed = build_project_seed_tokens(project_vocab_dir)
    train_tokens = _frequency_tokens(
        manifest,
        "train_token_frequencies",
    )
    validation_tokens = set(
        _frequency_tokens(
            manifest,
            "validation_token_frequencies",
        )
    )
    ordered_tokens = extend_ordered_tokens(seed, [train_tokens])
    validation_oov = sorted(
        validation_tokens - set(ordered_tokens),
        key=lambda token: token.encode("utf-8"),
    )
    if validation_oov:
        raise ValueError(
            f"validation contains OOV tokens not seen in train: {validation_oov}"
        )

    output_path = Path(output)
    if output_path.suffix.lower() != ".json" or not output_path.stem:
        raise ValueError("vocabulary output must be a named .json file")
    name = output_path.stem
    additions = ordered_tokens[len(seed) :]
    vocabulary = VocabularyManifest(
        schema_version=1,
        name=name,
        tokenization_mode="bekern",
        base_name="FullPageOMRProjectSeed",
        base_size=len(seed),
        base_digest=ordered_token_sha256(seed),
        ordered_tokens=ordered_tokens,
        token_provenance={
            token: f"pdmx-train:{manifest['manifest_sha256']}"
            for token in additions
        },
        source_dataset_manifests=(manifest["manifest_sha256"],),
        vocab_sha256=ordered_token_sha256(ordered_tokens),
    )

    if output_path.exists():
        existing = load_vocabulary_manifest(output_path)
        if existing.vocab_sha256 != vocabulary.vocab_sha256:
            raise FileExistsError(
                "refusing to overwrite a vocabulary with a different digest; "
                "choose a new output version"
            )
    w2i_path = output_path.parent / f"{name}w2i.npy"
    i2w_path = output_path.parent / f"{name}i2w.npy"
    if w2i_path.exists() or i2w_path.exists():
        if not (w2i_path.exists() and i2w_path.exists()):
            raise FileExistsError("legacy vocabulary pair is incomplete")
        existing_tokens = load_legacy_ordered_tokens(w2i_path, i2w_path)
        if existing_tokens != ordered_tokens:
            raise FileExistsError(
                "refusing to overwrite a legacy pair with a different digest; "
                "choose a new output version"
            )

    write_vocabulary_manifest(vocabulary, output_path)
    write_legacy_numpy_pair(vocabulary, output_path.parent)
    return str(output_path)


def main() -> None:
    Fire(
        {
            "prepare": prepare,
            "scan": scan,
            "build-vocabulary": build_vocabulary,
        }
    )


if __name__ == "__main__":
    main()
