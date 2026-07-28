"""Pinned MuSViT backbone registry and verified encoder loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import ViTModel

from .errors import ProtocolError


BACKBONE_LOADER = "transformers.ViTModel"
WEIGHTS_FILENAME = "model.safetensors"
REVIEWED_INPUT_CONTRACT = "staff_omr_input_v2"


@dataclass(frozen=True, slots=True)
class BlobEvidence:
    filename: str
    git_blob_oid: str
    size: int


@dataclass(frozen=True, slots=True)
class WeightEvidence:
    filename: str
    pointer_blob_oid: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class BackboneRegistryEntry:
    alias: str
    model_id: str
    revision: str
    readme: BlobEvidence
    config: BlobEvidence
    weights: WeightEvidence
    preprocessor_config_absent: bool
    reviewed_input_contract: str
    loader_class: str = BACKBONE_LOADER
    prefix_tokens: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackboneMetadata:
    model_id: str
    revision: str
    image_height: int
    image_width: int
    patch_height: int
    patch_width: int
    hidden_size: int
    num_channels: int
    prefix_tokens: int
    model_type: str
    architectures: tuple[str, ...]

    @property
    def native_rows(self) -> int:
        return self.image_height // self.patch_height

    @property
    def native_cols(self) -> int:
        return self.image_width // self.patch_width

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["architectures"] = list(self.architectures)
        return value


@dataclass(frozen=True, slots=True)
class BackboneInspection:
    entry: BackboneRegistryEntry
    metadata: BackboneMetadata
    raw_config: dict[str, object]
    registry_evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class BackboneLoadResult:
    model: ViTModel
    metadata: BackboneMetadata
    loading_info: dict[str, object]
    weight_verification: dict[str, object] | None = None
    registry_evidence: dict[str, object] | None = None


APPROVED_BACKBONES: dict[str, BackboneRegistryEntry] = {
    "musvit": BackboneRegistryEntry(
        alias="musvit",
        model_id="PRAIG/musvit",
        revision="0e91c7b223b4da30f259198c92045d0cb90e3f2e",
        readme=BlobEvidence(
            filename="README.md",
            git_blob_oid="8789d81c7e698c92b57746ab5dc090fe893069e2",
            size=6105,
        ),
        config=BlobEvidence(
            filename="config.json",
            git_blob_oid="dc2f7bc9bf1aab858aaeefaf1db7858c427f79f2",
            size=665,
        ),
        weights=WeightEvidence(
            filename=WEIGHTS_FILENAME,
            pointer_blob_oid="e935a37bc0a6ca051a091a2ba5b5425547f16af9",
            sha256=(
                "109bbaf31d9f2184df1b841579e06d25bc58ed6a42a10dd5f4a5d27d01889db2"
            ),
            size=467638680,
        ),
        preprocessor_config_absent=True,
        reviewed_input_contract=REVIEWED_INPUT_CONTRACT,
    ),
    "musvit_light": BackboneRegistryEntry(
        alias="musvit_light",
        model_id="PRAIG/musvit-light",
        revision="adf40fd3eaf157e20aaa8603ffea06517e467c7f",
        readme=BlobEvidence(
            filename="README.md",
            git_blob_oid="3155fc2c6beac85bcfd0146de779a65dc872cc8c",
            size=6144,
        ),
        config=BlobEvidence(
            filename="config.json",
            git_blob_oid="b9f746f5b2ecdd858c7dfbf4c724ae1de258b1a7",
            size=663,
        ),
        weights=WeightEvidence(
            filename=WEIGHTS_FILENAME,
            pointer_blob_oid="5c03c2357110e0060f3687d8d79dc5230c43019e",
            sha256=(
                "f2c278f2762a88bfcc7ee4cf846d8eef2f31c04a73e7775124db69e7afd0528f"
            ),
            size=157546736,
        ),
        preprocessor_config_absent=True,
        reviewed_input_contract=REVIEWED_INPUT_CONTRACT,
    ),
}


def approved_revisions() -> dict[str, set[str]]:
    return {
        alias: {entry.revision}
        for alias, entry in APPROVED_BACKBONES.items()
    }


def default_revisions() -> dict[str, str]:
    return {
        alias: entry.revision
        for alias, entry in APPROVED_BACKBONES.items()
    }


def registry_entry(model_name: str, revision: str) -> BackboneRegistryEntry:
    try:
        entry = APPROVED_BACKBONES[model_name]
    except KeyError as exc:
        raise ProtocolError(
            f"model_name {model_name!r} has no approved backbone registry"
        ) from exc
    if revision != entry.revision:
        raise ProtocolError(
            f"revision {revision!r} is not approved for {model_name!r}"
        )
    return entry


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"backbone config {field} must be a positive integer")
    return value


def _dimension_pair(value: object, field: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        item = _positive_int(value, field)
        return item, item
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise ProtocolError(
            f"backbone config {field} must be an integer or two-item array"
        )
    return (
        _positive_int(value[0], f"{field}[0]"),
        _positive_int(value[1], f"{field}[1]"),
    )


def metadata_from_raw_config(
    raw_config: dict[str, object],
    entry: BackboneRegistryEntry | Any,
) -> BackboneMetadata:
    """Validate reviewed ViT-MAE config data without loading model weights."""
    if not isinstance(raw_config, dict):
        raise ProtocolError("backbone config must be a JSON object")
    if raw_config.get("model_type") != "vit_mae":
        raise ProtocolError(
            "backbone config model_type must be exactly 'vit_mae'"
        )
    architectures = raw_config.get("architectures")
    if (
        not isinstance(architectures, list)
        or "ViTMAEForPreTraining" not in architectures
        or not all(isinstance(item, str) for item in architectures)
    ):
        raise ProtocolError(
            "backbone config architectures must include "
            "'ViTMAEForPreTraining'"
        )
    channels = _positive_int(raw_config.get("num_channels"), "num_channels")
    if channels != 3:
        raise ProtocolError("backbone config num_channels must be 3")
    image_height, image_width = _dimension_pair(
        raw_config.get("image_size"),
        "image_size",
    )
    patch_height, patch_width = _dimension_pair(
        raw_config.get("patch_size"),
        "patch_size",
    )
    hidden_size = _positive_int(raw_config.get("hidden_size"), "hidden_size")
    if image_height % patch_height or image_width % patch_width:
        raise ProtocolError(
            "backbone image dimensions must be divisible by patch dimensions"
        )
    prefix_tokens = _positive_int(
        getattr(entry, "prefix_tokens", None),
        "prefix_tokens",
    )
    return BackboneMetadata(
        model_id=str(entry.model_id),
        revision=str(entry.revision),
        image_height=image_height,
        image_width=image_width,
        patch_height=patch_height,
        patch_width=patch_width,
        hidden_size=hidden_size,
        num_channels=channels,
        prefix_tokens=prefix_tokens,
        model_type="vit_mae",
        architectures=tuple(architectures),
    )


def git_blob_oid(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def _metadata_value(value: object, field: str) -> object:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def validate_registry_tree(
    entry: BackboneRegistryEntry,
    model_info: object,
) -> None:
    if _metadata_value(model_info, "sha") != entry.revision:
        raise ProtocolError("Hugging Face resolved revision differs from registry")
    siblings = _metadata_value(model_info, "siblings")
    if not isinstance(siblings, (list, tuple)):
        raise ProtocolError("Hugging Face model metadata has no sibling list")
    files: dict[str, object] = {}
    for sibling in siblings:
        filename = _metadata_value(sibling, "rfilename")
        if isinstance(filename, str):
            files[filename] = sibling
    if "preprocessor_config.json" in files:
        raise ProtocolError(
            "registry requires preprocessor_config.json to be absent"
        )

    for evidence in (entry.readme, entry.config):
        sibling = files.get(evidence.filename)
        if sibling is None:
            raise ProtocolError(
                f"registry evidence file is missing: {evidence.filename}"
            )
        if _metadata_value(sibling, "size") != evidence.size:
            raise ProtocolError(
                f"{evidence.filename} size differs from registry evidence"
            )
        if _metadata_value(sibling, "blob_id") != evidence.git_blob_oid:
            raise ProtocolError(
                f"{evidence.filename} git blob oid differs from registry evidence"
            )

    sibling = files.get(entry.weights.filename)
    if sibling is None:
        raise ProtocolError(
            f"registry weight file is missing: {entry.weights.filename}"
        )
    if _metadata_value(sibling, "size") != entry.weights.size:
        raise ProtocolError("weight size differs from registry evidence")
    if _metadata_value(sibling, "blob_id") != entry.weights.pointer_blob_oid:
        raise ProtocolError(
            "weight Git LFS pointer blob oid differs from registry evidence"
        )
    lfs = _metadata_value(sibling, "lfs")
    if lfs is None:
        raise ProtocolError("weight metadata is missing Git LFS evidence")
    if _metadata_value(lfs, "sha256") != entry.weights.sha256:
        raise ProtocolError("weight LFS SHA-256 differs from registry evidence")
    if _metadata_value(lfs, "size") != entry.weights.size:
        raise ProtocolError("weight LFS size differs from registry evidence")


def _verify_blob_file(path: Path, evidence: BlobEvidence) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"cannot read registry evidence file {path}") from exc
    if len(content) != evidence.size:
        raise ProtocolError(f"{evidence.filename} content size mismatch")
    if git_blob_oid(content) != evidence.git_blob_oid:
        raise ProtocolError(f"{evidence.filename} content git blob mismatch")
    return content


def verify_weight_file(
    path: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, object]:
    resolved = Path(path)
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ProtocolError(f"cannot stat base weight file {resolved}") from exc
    if not resolved.is_file():
        raise ProtocolError(f"base weight path is not a file: {resolved}")
    if stat.st_size != expected_size:
        raise ProtocolError(
            f"base weight size mismatch: expected {expected_size}, "
            f"actual {stat.st_size}"
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"cannot read base weight file {resolved}") from exc
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha256:
        raise ProtocolError(
            "base weight SHA-256 mismatch: "
            f"expected {expected_sha256}, actual {actual_sha}"
        )
    return {
        "filename": resolved.name,
        "sha256": actual_sha,
        "size": stat.st_size,
    }


def validate_loading_info(loading_info: dict[str, object]) -> None:
    for field in (
        "missing_keys",
        "mismatched_keys",
        "unexpected_keys",
        "error_msgs",
    ):
        value = loading_info.get(field)
        if not isinstance(value, list):
            raise ProtocolError(f"loading info {field} must be an array")
        if field == "unexpected_keys":
            invalid = [
                item
                for item in value
                if not isinstance(item, str) or not item.startswith("decoder.")
            ]
            if invalid:
                raise ProtocolError(
                    f"loading info unexpected_keys contains non-decoder keys: "
                    f"{invalid!r}"
                )
        elif value:
            raise ProtocolError(f"loading info {field} must be empty: {value!r}")


def _loaded_metadata(model: ViTModel, expected: BackboneMetadata) -> BackboneMetadata:
    entry = type(
        "_LoadedEntry",
        (),
        {
            "model_id": expected.model_id,
            "revision": expected.revision,
            "prefix_tokens": expected.prefix_tokens,
        },
    )()
    raw_config = {
        "model_type": model.config.model_type,
        "architectures": model.config.architectures,
        "image_size": model.config.image_size,
        "patch_size": model.config.patch_size,
        "hidden_size": model.config.hidden_size,
        "num_channels": model.config.num_channels,
    }
    return metadata_from_raw_config(raw_config, entry)


def _assert_metadata_equal(
    actual: BackboneMetadata,
    expected: BackboneMetadata,
) -> None:
    for field in (
        "image_height",
        "image_width",
        "patch_height",
        "patch_width",
        "hidden_size",
        "num_channels",
        "prefix_tokens",
        "model_type",
        "architectures",
    ):
        if getattr(actual, field) != getattr(expected, field):
            raise ProtocolError(
                f"loaded backbone {field} differs from inspected config: "
                f"expected {getattr(expected, field)!r}, "
                f"actual {getattr(actual, field)!r}"
            )


def load_encoder_from_pretrained(
    source: str | Path,
    *,
    revision: str | None,
    expected_metadata: BackboneMetadata,
) -> BackboneLoadResult:
    """Load only the ordered ViT encoder from a reviewed ViT-MAE checkpoint."""
    model, loading_info = ViTModel.from_pretrained(
        source,
        revision=revision,
        trust_remote_code=False,
        add_pooling_layer=False,
        output_loading_info=True,
    )
    validate_loading_info(loading_info)
    loaded = _loaded_metadata(model, expected_metadata)
    _assert_metadata_equal(loaded, expected_metadata)
    return BackboneLoadResult(
        model=model,
        metadata=expected_metadata,
        loading_info=loading_info,
    )


class ProductionBackboneProvider:
    """Hugging Face backed implementation split into preflight and weight load."""

    def __init__(self, *, api: object | None = None, download=None):
        if api is None or download is None:
            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi() if api is None else api
            download = hf_hub_download if download is None else download
        self._api = api
        self._download = download

    def resolve_revision(self, model_name: str) -> str:
        try:
            entry = APPROVED_BACKBONES[model_name]
        except KeyError as exc:
            raise ProtocolError(f"unknown model_name {model_name!r}") from exc
        info = self._api.model_info(entry.model_id)
        revision = _metadata_value(info, "sha")
        if not isinstance(revision, str):
            raise ProtocolError("Hugging Face did not resolve a commit revision")
        registry_entry(model_name, revision)
        return revision

    def inspect(
        self,
        model_name: str,
        revision: str,
    ) -> BackboneInspection:
        entry = registry_entry(model_name, revision)
        try:
            info = self._api.model_info(
                entry.model_id,
                revision=entry.revision,
                files_metadata=True,
            )
            validate_registry_tree(entry, info)
            readme_path = Path(
                self._download(
                    repo_id=entry.model_id,
                    filename=entry.readme.filename,
                    revision=entry.revision,
                )
            )
            config_path = Path(
                self._download(
                    repo_id=entry.model_id,
                    filename=entry.config.filename,
                    revision=entry.revision,
                )
            )
        except ProtocolError:
            raise
        except Exception as exc:
            raise ProtocolError(
                f"failed to inspect approved backbone {entry.model_id}@"
                f"{entry.revision}"
            ) from exc

        _verify_blob_file(readme_path, entry.readme)
        config_bytes = _verify_blob_file(config_path, entry.config)
        try:
            raw_config = json.loads(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("approved backbone config is not valid UTF-8 JSON") from exc
        metadata = metadata_from_raw_config(raw_config, entry)
        return BackboneInspection(
            entry=entry,
            metadata=metadata,
            raw_config=raw_config,
            registry_evidence={
                "registry_entry": entry.to_dict(),
                "resolved_revision": entry.revision,
                "tree_verified": True,
            },
        )

    def load(self, inspection: BackboneInspection) -> BackboneLoadResult:
        entry = inspection.entry
        try:
            weight_path = Path(
                self._download(
                    repo_id=entry.model_id,
                    filename=entry.weights.filename,
                    revision=entry.revision,
                )
            )
        except Exception as exc:
            raise ProtocolError(
                f"failed to download approved base weights for {entry.model_id}"
            ) from exc
        verification = verify_weight_file(
            weight_path,
            expected_size=entry.weights.size,
            expected_sha256=entry.weights.sha256,
        )
        loaded = load_encoder_from_pretrained(
            entry.model_id,
            revision=entry.revision,
            expected_metadata=inspection.metadata,
        )
        return BackboneLoadResult(
            model=loaded.model,
            metadata=loaded.metadata,
            loading_info=loaded.loading_info,
            weight_verification=verification,
            registry_evidence=inspection.registry_evidence,
        )
