from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .utils.vocab_manifest import (
    canonical_json_sha256,
    load_legacy_ordered_tokens,
    load_vocabulary_manifest,
    ordered_token_sha256,
)


TOKEN_AXIS_KEYS = (
    "model.decoder.embedding.weight",
    "model.decoder.out_layer.weight",
    "model.decoder.out_layer.bias",
)
LEGACY_SOURCE_MANIFEST_TYPE = "legacy_vocabulary_source"
_LEGACY_SOURCE_FIELDS = {
    "schema_version",
    "manifest_type",
    "name",
    "tokenization_mode",
    "ordered_tokens",
    "vocab_sha256",
    "source_files",
    "manifest_sha256",
}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_legacy_source_vocabulary_manifest(
    w2i_path: str | Path,
    i2w_path: str | Path,
    *,
    output_path: str | Path,
    name: str,
    tokenization_mode: str = "bekern",
) -> Path:
    w2i = Path(w2i_path).resolve()
    i2w = Path(i2w_path).resolve()
    output = Path(output_path).resolve()
    if output.parent != w2i.parent or output.parent != i2w.parent:
        raise ValueError(
            "legacy source manifest and both NumPy files must share a directory"
        )
    if not isinstance(name, str) or not name:
        raise ValueError("legacy vocabulary name must be a non-empty string")
    if tokenization_mode != "bekern":
        raise ValueError("legacy vocabulary tokenization_mode must be 'bekern'")

    tokens = load_legacy_ordered_tokens(w2i, i2w)
    payload = {
        "schema_version": 1,
        "manifest_type": LEGACY_SOURCE_MANIFEST_TYPE,
        "name": name,
        "tokenization_mode": tokenization_mode,
        "ordered_tokens": list(tokens),
        "vocab_sha256": ordered_token_sha256(tokens),
        "source_files": {
            "w2i": {
                "filename": w2i.name,
                "bytes": w2i.stat().st_size,
                "sha256": _sha256_file(w2i),
            },
            "i2w": {
                "filename": i2w.name,
                "bytes": i2w.stat().st_size,
                "sha256": _sha256_file(i2w),
            },
        },
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(
            f"refusing to overwrite a different source manifest: {output}"
        )
    output.write_text(serialized, encoding="utf-8")
    return output


def _load_legacy_source_vocabulary_tokens(
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    if set(payload) != _LEGACY_SOURCE_FIELDS:
        raise ValueError("legacy source vocabulary manifest fields differ")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported legacy source vocabulary schema")
    if payload.get("manifest_type") != LEGACY_SOURCE_MANIFEST_TYPE:
        raise ValueError("invalid legacy source vocabulary manifest type")
    if payload.get("tokenization_mode") != "bekern":
        raise ValueError("legacy source vocabulary must use 'bekern'")
    declared_manifest_digest = payload.get("manifest_sha256")
    unsigned = dict(payload)
    del unsigned["manifest_sha256"]
    actual_manifest_digest = canonical_json_sha256(unsigned)
    if declared_manifest_digest != actual_manifest_digest:
        raise ValueError(
            "legacy source manifest SHA-256 mismatch: "
            f"expected={declared_manifest_digest}, "
            f"actual={actual_manifest_digest}"
        )

    source_files = payload.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != {"w2i", "i2w"}:
        raise ValueError("legacy source manifest must pin w2i and i2w files")
    resolved: dict[str, Path] = {}
    for role in ("w2i", "i2w"):
        descriptor = source_files[role]
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"filename", "bytes", "sha256"}
        ):
            raise ValueError(f"legacy source {role} descriptor is invalid")
        filename = descriptor["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"legacy source {role} filename is invalid")
        path = manifest_path.parent / filename
        if not path.is_file():
            raise FileNotFoundError(f"legacy source {role} file is missing: {path}")
        actual_digest = _sha256_file(path)
        actual_bytes = path.stat().st_size
        if (
            actual_bytes != descriptor["bytes"]
            or actual_digest != descriptor["sha256"]
        ):
            raise ValueError(
                f"legacy source {role} byte size/SHA-256 mismatch: "
                f"expected_bytes={descriptor['bytes']}, "
                f"actual_bytes={actual_bytes}, "
                f"expected_sha256={descriptor['sha256']}, "
                f"actual_sha256={actual_digest}"
            )
        resolved[role] = path

    tokens = load_legacy_ordered_tokens(resolved["w2i"], resolved["i2w"])
    declared_tokens = _validate_tokens(
        payload.get("ordered_tokens", ()),
        "legacy ordered_tokens",
    )
    if tokens != declared_tokens:
        raise ValueError("legacy source NumPy order differs from manifest")
    if payload.get("vocab_sha256") != ordered_token_sha256(tokens):
        raise ValueError("legacy source vocabulary SHA-256 mismatch")
    return tokens


def load_source_vocabulary_tokens(
    manifest_path: str | Path,
) -> tuple[str, ...]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("source vocabulary manifest must contain a JSON object")
    if payload.get("manifest_type") == LEGACY_SOURCE_MANIFEST_TYPE:
        return _load_legacy_source_vocabulary_tokens(path, payload)
    return load_vocabulary_manifest(path).ordered_tokens


def _validate_tokens(tokens: Sequence[str], field_name: str) -> tuple[str, ...]:
    ordered = tuple(tokens)
    if any(not isinstance(token, str) or not token for token in ordered):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{field_name} contains duplicate tokens")
    return ordered


def _validate_tensor_state(
    state: Mapping[str, torch.Tensor],
    field_name: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"{field_name} must be a tensor mapping")
    normalized = dict(state)
    if any(not isinstance(key, str) for key in normalized):
        raise TypeError(f"{field_name} keys must be strings")
    non_tensors = [
        key
        for key, value in normalized.items()
        if not isinstance(value, torch.Tensor)
    ]
    if non_tensors:
        raise TypeError(
            f"{field_name} contains non-tensor values: {sorted(non_tensors)}"
        )
    return normalized


def remap_vocabulary_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source = _validate_tensor_state(source_state, "source_state")
    target = _validate_tensor_state(target_state, "target_state")
    source_order = _validate_tokens(source_tokens, "source_tokens")
    target_order = _validate_tokens(target_tokens, "target_tokens")

    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(
            "source and target state_dict key sets differ: "
            f"missing={missing}, extra={extra}"
        )
    missing_token_axes = [
        key for key in TOKEN_AXIS_KEYS if key not in source
    ]
    if missing_token_axes:
        raise ValueError(
            f"state_dict is missing token-axis tensors: {missing_token_axes}"
        )

    source_ids = {token: index for index, token in enumerate(source_order)}
    target_ids = {token: index for index, token in enumerate(target_order)}
    source_only = [
        token for token in source_order if token not in target_ids
    ]
    if source_only:
        raise ValueError(
            f"source tokens are missing from target vocabulary: {source_only}"
        )

    migrated: dict[str, torch.Tensor] = {}
    for key in sorted(source):
        source_tensor = source[key]
        target_tensor = target[key]
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"state tensor {key!r} dtype differs: "
                f"source={source_tensor.dtype}, target={target_tensor.dtype}"
            )
        if key in TOKEN_AXIS_KEYS:
            if source_tensor.ndim < 1 or target_tensor.ndim < 1:
                raise ValueError(f"token-axis tensor {key!r} must have rank >= 1")
            if source_tensor.shape[0] != len(source_order):
                raise ValueError(
                    f"source token axis for {key!r} has length "
                    f"{source_tensor.shape[0]}, expected {len(source_order)}"
                )
            if target_tensor.shape[0] != len(target_order):
                raise ValueError(
                    f"target token axis for {key!r} has length "
                    f"{target_tensor.shape[0]}, expected {len(target_order)}"
                )
            if source_tensor.shape[1:] != target_tensor.shape[1:]:
                raise ValueError(
                    f"token-axis tensor {key!r} has incompatible non-token "
                    f"dimensions: source={tuple(source_tensor.shape[1:])}, "
                    f"target={tuple(target_tensor.shape[1:])}"
                )
            migrated_tensor = target_tensor.detach().clone()
            for token, source_id in source_ids.items():
                target_id = target_ids[token]
                migrated_tensor[target_id].copy_(
                    source_tensor[source_id].to(
                        device=migrated_tensor.device,
                    )
                )
            migrated[key] = migrated_tensor
            continue

        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"non-token tensor {key!r} shape differs: "
                f"source={tuple(source_tensor.shape)}, "
                f"target={tuple(target_tensor.shape)}"
            )
        migrated[key] = source_tensor.detach().clone().to(
            device=target_tensor.device,
        )

    common_tokens = [
        token for token in target_order if token in source_ids
    ]
    new_tokens = [
        token for token in target_order if token not in source_ids
    ]
    report = {
        "schema_version": 1,
        "source_vocab_sha256": ordered_token_sha256(source_order),
        "target_vocab_sha256": ordered_token_sha256(target_order),
        "source_vocab_size": len(source_order),
        "target_vocab_size": len(target_order),
        "common_token_count": len(common_tokens),
        "common_tokens": common_tokens,
        "new_tokens": new_tokens,
        "dropped_tokens": [],
        "copied_token_axis_tensor_count": len(TOKEN_AXIS_KEYS),
        "copied_non_token_tensor_count": len(source) - len(TOKEN_AXIS_KEYS),
    }
    return migrated, report


def load_vocabulary_aware_weights(
    model_wrapper: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    source_vocab_manifest: str | Path,
    target_vocab_manifest: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    source_tokens = load_source_vocabulary_tokens(source_vocab_manifest)
    target_vocabulary = load_vocabulary_manifest(target_vocab_manifest)
    model = getattr(model_wrapper, "model", None)
    model_i2w = getattr(model, "i2w", None)
    if not isinstance(model_i2w, Mapping):
        raise ValueError("target model vocabulary i2w mapping is unavailable")
    try:
        model_tokens = tuple(
            model_i2w[token_id]
            for token_id in range(len(model_i2w))
        )
    except KeyError as error:
        raise ValueError(
            "target model vocabulary ids must be contiguous from zero"
        ) from error
    if model_tokens != target_vocabulary.ordered_tokens:
        raise ValueError(
            "target model vocabulary differs from target manifest"
        )

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("checkpoint is missing a state_dict tensor mapping")

    migrated, report = remap_vocabulary_state_dict(
        source_state,
        model_wrapper.state_dict(),
        source_tokens=source_tokens,
        target_tokens=target_vocabulary.ordered_tokens,
    )
    model_wrapper.load_state_dict(migrated, strict=True)

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    output_path.write_text(payload, encoding="utf-8")
    return report


def convert_legacy(
    w2i_path: str,
    i2w_path: str,
    output_path: str,
    name: str,
) -> str:
    return str(
        write_legacy_source_vocabulary_manifest(
            w2i_path,
            i2w_path,
            output_path=output_path,
            name=name,
        )
    )


if __name__ == "__main__":
    from fire import Fire

    Fire({"convert-legacy": convert_legacy})
