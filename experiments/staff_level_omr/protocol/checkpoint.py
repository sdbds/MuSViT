"""Strict transactional checkpoint schema for staff OMR v2."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import torch
from torch import nn

from .canonical import canonical_sha256, read_json
from .errors import ProtocolError
from .modeling import load_trainable_state_dict, trainable_state_dict
from .seeding import SEED_SCHEDULE_VERSION, initialization_seed
from .vocabulary import Vocabulary


CHECKPOINT_SCHEMA = "staff_omr_checkpoint_v2"
PROTOCOL_VERSION = "staff_omr_v2"
STATE_DICT_SCOPE = "trainable_only"
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "run_id",
        "checkpoint_role",
        "state_dict_scope",
        "trainable_state_dict",
        "trainable_parameter_names",
        "optimizer_state_dict",
        "optimizer_parameter_names",
        "epoch",
        "global_step",
        "early_stopping_state",
        "best_metric_name",
        "best_metric_value",
        "best_epoch",
        "best_updated",
        "stop_reason",
        "epoch_record",
        "training_contract",
        "training_contract_sha256",
        "initial_launch_config",
        "initial_launch_config_sha256",
        "dataset_bundle_relpath",
        "dataset_bundle_sha256",
        "split_manifest_relpath",
        "split_manifest_sha256",
        "vocabulary",
        "vocabulary_sha256",
        "base_model_id",
        "base_model_revision",
        "base_model_weights_filename",
        "base_model_weights_sha256",
        "base_model_registry_evidence",
        "backbone_config",
        "input_contract",
        "augmentation_contract_sha256",
        "train_exclusions_relpath",
        "train_exclusions_sha256",
        "train_exclusions_count",
        "base_seed",
        "init_seed",
        "seed_schedule_version",
        "next_epoch",
        "package_versions",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[0-9a-f]{12}$")


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProtocolError(f"{field} must be 64 lowercase hex characters")
    return value


def _safe_relpath(value: object, field: str, expected: str) -> str:
    if value != expected:
        raise ProtocolError(f"{field} must be exactly {expected!r}")
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ProtocolError(f"{field} is not a safe run-relative path")
    return str(value)


@dataclass(frozen=True, slots=True)
class CheckpointStatic:
    training_contract: dict[str, object]
    training_contract_sha256: str
    initial_launch_config: dict[str, object]
    initial_launch_config_sha256: str
    dataset_bundle_relpath: str
    dataset_bundle_sha256: str
    split_manifest_relpath: str
    split_manifest_sha256: str
    vocabulary: dict[str, object]
    vocabulary_sha256: str
    base_model_id: str
    base_model_revision: str
    base_model_weights_filename: str
    base_model_weights_sha256: str
    base_model_registry_evidence: dict[str, object]
    backbone_config: dict[str, object]
    input_contract: dict[str, object]
    augmentation_contract_sha256: str
    train_exclusions_relpath: str | None
    train_exclusions_sha256: str | None
    train_exclusions_count: int
    base_seed: int
    init_seed: int
    package_versions: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        training_contract: dict[str, object],
        initial_launch_config: dict[str, object],
        dataset_bundle_sha256: str,
        split_manifest_sha256: str,
        vocabulary: dict[str, object],
        base_model_id: str,
        base_model_revision: str,
        base_model_weights_filename: str,
        base_model_weights_sha256: str,
        base_model_registry_evidence: dict[str, object],
        backbone_config: dict[str, object],
        input_contract: dict[str, object],
        augmentation_contract_sha256: str,
        train_exclusions_relpath: str | None,
        train_exclusions_sha256: str | None,
        train_exclusions_count: int,
        base_seed: int,
        package_versions: dict[str, str],
    ) -> "CheckpointStatic":
        if not isinstance(training_contract, dict):
            raise ProtocolError("training_contract must be an object")
        if not isinstance(initial_launch_config, dict):
            raise ProtocolError("initial_launch_config must be an object")
        bundle_hash = _sha256(
            dataset_bundle_sha256, "dataset_bundle_sha256"
        )
        manifest_hash = _sha256(
            split_manifest_sha256, "split_manifest_sha256"
        )
        if not isinstance(vocabulary, dict):
            raise ProtocolError("vocabulary must be an object")
        dataset_id = vocabulary.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ProtocolError("vocabulary dataset_id must be non-empty")
        Vocabulary.from_document(
            vocabulary,
            manifest_sha256=manifest_hash,
            dataset_id=dataset_id,
        )
        if not isinstance(base_model_id, str) or not base_model_id:
            raise ProtocolError("base_model_id must be non-empty")
        if (
            not isinstance(base_model_revision, str)
            or not _REVISION.fullmatch(base_model_revision)
        ):
            raise ProtocolError(
                "base_model_revision must be an immutable commit SHA"
            )
        if (
            not isinstance(base_model_weights_filename, str)
            or PurePosixPath(base_model_weights_filename).name
            != base_model_weights_filename
        ):
            raise ProtocolError(
                "base_model_weights_filename must be a plain filename"
            )
        weight_hash = _sha256(
            base_model_weights_sha256,
            "base_model_weights_sha256",
        )
        augmentation_hash = _sha256(
            augmentation_contract_sha256,
            "augmentation_contract_sha256",
        )
        if (
            isinstance(train_exclusions_count, bool)
            or not isinstance(train_exclusions_count, int)
            or train_exclusions_count < 0
        ):
            raise ProtocolError(
                "train_exclusions_count must be non-negative"
            )
        if train_exclusions_count == 0:
            if (
                train_exclusions_relpath is not None
                or train_exclusions_sha256 is not None
            ):
                raise ProtocolError(
                    "zero exclusions require null exclusions path and hash"
                )
            exclusion_path = None
            exclusion_hash = None
        else:
            exclusion_path = _safe_relpath(
                train_exclusions_relpath,
                "train_exclusions_relpath",
                "train_exclusions.json",
            )
            exclusion_hash = _sha256(
                train_exclusions_sha256,
                "train_exclusions_sha256",
            )
        if (
            isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or base_seed < 0
        ):
            raise ProtocolError("base_seed must be non-negative")
        if not isinstance(package_versions, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in package_versions.items()
        ):
            raise ProtocolError("package_versions must map strings to strings")
        return cls(
            training_contract=dict(training_contract),
            training_contract_sha256=canonical_sha256(training_contract),
            initial_launch_config=dict(initial_launch_config),
            initial_launch_config_sha256=canonical_sha256(
                initial_launch_config
            ),
            dataset_bundle_relpath="dataset_bundle.json",
            dataset_bundle_sha256=bundle_hash,
            split_manifest_relpath="split_manifest.json",
            split_manifest_sha256=manifest_hash,
            vocabulary=dict(vocabulary),
            vocabulary_sha256=canonical_sha256(vocabulary),
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
            base_model_weights_filename=base_model_weights_filename,
            base_model_weights_sha256=weight_hash,
            base_model_registry_evidence=dict(
                base_model_registry_evidence
            ),
            backbone_config=dict(backbone_config),
            input_contract=dict(input_contract),
            augmentation_contract_sha256=augmentation_hash,
            train_exclusions_relpath=exclusion_path,
            train_exclusions_sha256=exclusion_hash,
            train_exclusions_count=train_exclusions_count,
            base_seed=base_seed,
            init_seed=initialization_seed(base_seed),
            package_versions=dict(package_versions),
        )

    def payload_fields(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _trainable_names(model: nn.Module) -> list[str]:
    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return sorted(names, key=lambda value: value.encode("utf-8"))


def build_checkpoint(
    *,
    static: CheckpointStatic,
    run_id: str,
    role: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_parameter_names: list[str],
    epoch: int,
    global_step: int,
    early_stopping_state: dict[str, object],
    best_metric_name: str,
    best_metric_value: float | None,
    best_epoch: int | None,
    best_updated: bool,
    stop_reason: str | None,
    epoch_record: dict[str, object],
) -> dict[str, object]:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ProtocolError("run_id must contain exactly 12 lowercase hex chars")
    if role not in {"last", "best"}:
        raise ProtocolError("checkpoint role must be 'last' or 'best'")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ProtocolError("checkpoint epoch must be positive")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ProtocolError("checkpoint global_step must be non-negative")
    if not isinstance(early_stopping_state, dict):
        raise ProtocolError("early_stopping_state must be an object")
    if not isinstance(best_metric_name, str) or not best_metric_name:
        raise ProtocolError("best_metric_name must be non-empty")
    if best_metric_value is not None and (
        isinstance(best_metric_value, bool)
        or not isinstance(best_metric_value, (int, float))
        or not math.isfinite(float(best_metric_value))
    ):
        raise ProtocolError("best_metric_value must be finite or null")
    if best_epoch is not None and (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch <= 0
        or best_epoch > epoch
    ):
        raise ProtocolError("best_epoch must be null or within committed epochs")
    if not isinstance(best_updated, bool):
        raise ProtocolError("best_updated must be boolean")
    if stop_reason not in {None, "max_epochs", "early_stopping"}:
        raise ProtocolError("invalid checkpoint stop_reason")
    if not isinstance(epoch_record, dict) or epoch_record.get("epoch") != epoch:
        raise ProtocolError("epoch_record must match checkpoint epoch")
    actual_names = _trainable_names(model)
    if optimizer_parameter_names != actual_names:
        raise ProtocolError(
            "optimizer_parameter_names must exactly match sorted trainable "
            "parameter names"
        )
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "checkpoint_role": role,
        "state_dict_scope": STATE_DICT_SCOPE,
        "trainable_state_dict": trainable_state_dict(model),
        "trainable_parameter_names": actual_names,
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_parameter_names": list(optimizer_parameter_names),
        "epoch": epoch,
        "global_step": global_step,
        "early_stopping_state": dict(early_stopping_state),
        "best_metric_name": best_metric_name,
        "best_metric_value": (
            float(best_metric_value)
            if best_metric_value is not None
            else None
        ),
        "best_epoch": best_epoch,
        "best_updated": best_updated,
        "stop_reason": stop_reason,
        "epoch_record": dict(epoch_record),
        "seed_schedule_version": SEED_SCHEDULE_VERSION,
        "next_epoch": epoch + 1,
    }
    payload.update(static.payload_fields())
    validate_checkpoint_payload(payload)
    return payload


def validate_checkpoint_payload(checkpoint: object) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        raise ProtocolError("checkpoint must be a dictionary")
    if "schema_version" not in checkpoint:
        raise ProtocolError(
            "legacy state-dict checkpoint has no v2 schema_version"
        )
    actual_fields = set(checkpoint)
    if actual_fields != CHECKPOINT_FIELDS:
        raise ProtocolError(
            "checkpoint fields mismatch; "
            f"missing={sorted(CHECKPOINT_FIELDS - actual_fields)!r}, "
            f"extra={sorted(actual_fields - CHECKPOINT_FIELDS)!r}"
        )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA:
        raise ProtocolError("unsupported checkpoint schema_version")
    if checkpoint["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("checkpoint protocol_version mismatch")
    if checkpoint["checkpoint_role"] not in {"last", "best"}:
        raise ProtocolError("checkpoint_role must be 'last' or 'best'")
    if checkpoint["state_dict_scope"] != STATE_DICT_SCOPE:
        raise ProtocolError("checkpoint state_dict_scope must be trainable_only")
    if not isinstance(checkpoint["run_id"], str) or not _RUN_ID.fullmatch(
        checkpoint["run_id"]
    ):
        raise ProtocolError("checkpoint run_id is invalid")
    if canonical_sha256(checkpoint["training_contract"]) != checkpoint[
        "training_contract_sha256"
    ]:
        raise ProtocolError("checkpoint training_contract SHA-256 mismatch")
    if canonical_sha256(checkpoint["initial_launch_config"]) != checkpoint[
        "initial_launch_config_sha256"
    ]:
        raise ProtocolError("checkpoint initial_launch_config SHA-256 mismatch")
    for field in (
        "training_contract_sha256",
        "initial_launch_config_sha256",
        "dataset_bundle_sha256",
        "split_manifest_sha256",
        "vocabulary_sha256",
        "base_model_weights_sha256",
        "augmentation_contract_sha256",
    ):
        _sha256(checkpoint[field], field)
    if canonical_sha256(checkpoint["vocabulary"]) != checkpoint[
        "vocabulary_sha256"
    ]:
        raise ProtocolError("checkpoint vocabulary SHA-256 mismatch")
    vocabulary = checkpoint["vocabulary"]
    if not isinstance(vocabulary, dict):
        raise ProtocolError("checkpoint vocabulary must be an object")
    Vocabulary.from_document(
        vocabulary,
        manifest_sha256=checkpoint["split_manifest_sha256"],
        dataset_id=vocabulary.get("dataset_id"),
    )
    _safe_relpath(
        checkpoint["dataset_bundle_relpath"],
        "dataset_bundle_relpath",
        "dataset_bundle.json",
    )
    _safe_relpath(
        checkpoint["split_manifest_relpath"],
        "split_manifest_relpath",
        "split_manifest.json",
    )
    count = checkpoint["train_exclusions_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ProtocolError("checkpoint train_exclusions_count is invalid")
    if count == 0:
        if (
            checkpoint["train_exclusions_relpath"] is not None
            or checkpoint["train_exclusions_sha256"] is not None
        ):
            raise ProtocolError("checkpoint exclusions metadata is inconsistent")
    else:
        _safe_relpath(
            checkpoint["train_exclusions_relpath"],
            "train_exclusions_relpath",
            "train_exclusions.json",
        )
        _sha256(
            checkpoint["train_exclusions_sha256"],
            "train_exclusions_sha256",
        )
    if (
        checkpoint["seed_schedule_version"] != SEED_SCHEDULE_VERSION
        or checkpoint["init_seed"]
        != initialization_seed(checkpoint["base_seed"])
    ):
        raise ProtocolError("checkpoint seed schedule metadata mismatch")
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ProtocolError("checkpoint epoch is invalid")
    if checkpoint["next_epoch"] != epoch + 1:
        raise ProtocolError("checkpoint next_epoch must equal epoch + 1")
    if (
        not isinstance(checkpoint["epoch_record"], dict)
        or checkpoint["epoch_record"].get("epoch") != epoch
    ):
        raise ProtocolError("checkpoint epoch_record is invalid")
    names = checkpoint["trainable_parameter_names"]
    optimizer_names = checkpoint["optimizer_parameter_names"]
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or names != sorted(names, key=lambda value: value.encode("utf-8"))
        or len(set(names)) != len(names)
    ):
        raise ProtocolError("checkpoint trainable_parameter_names is invalid")
    if optimizer_names != names:
        raise ProtocolError(
            "checkpoint optimizer_parameter_names must equal trainable names"
        )
    state = checkpoint["trainable_state_dict"]
    if not isinstance(state, dict) or set(state) != set(names):
        raise ProtocolError("checkpoint trainable_state_dict keys mismatch")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ProtocolError("checkpoint trainable state values must be tensors")
    if not isinstance(checkpoint["optimizer_state_dict"], dict):
        raise ProtocolError("checkpoint optimizer_state_dict must be a dictionary")
    return checkpoint


def save_checkpoint(path: str | Path, checkpoint: dict[str, object]) -> None:
    validate_checkpoint_payload(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid4().hex}"
    )
    try:
        with temporary.open("xb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(
            f"cannot atomically save checkpoint {destination}"
        ) from exc


def load_checkpoint(
    path: str | Path,
    *,
    expected_role: str | None = None,
) -> dict[str, object]:
    source = Path(path)
    try:
        checkpoint = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ProtocolError(f"cannot load checkpoint {source}") from exc
    validated = validate_checkpoint_payload(checkpoint)
    if (
        expected_role is not None
        and validated["checkpoint_role"] != expected_role
    ):
        raise ProtocolError(
            f"checkpoint role must be {expected_role!r}, got "
            f"{validated['checkpoint_role']!r}"
        )
    return validated


def _run_document_hash(run_dir: Path, relpath: str, field: str) -> str:
    path = (run_dir / relpath).resolve(strict=True)
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ProtocolError(f"{field} escapes the run directory") from exc
    return canonical_sha256(read_json(path))


def validate_resume_checkpoint(
    checkpoint: dict[str, object],
    *,
    run_dir: str | Path,
    expected_training_contract_sha256: str,
    expected_optimizer_parameter_names: list[str],
    expected_trainable_parameter_names: list[str],
) -> None:
    validate_checkpoint_payload(checkpoint)
    if checkpoint["checkpoint_role"] != "last":
        raise ProtocolError("resume requires checkpoint role 'last'")
    if checkpoint["training_contract_sha256"] != (
        expected_training_contract_sha256
    ):
        raise ProtocolError("resume training contract identity mismatch")
    if checkpoint["optimizer_parameter_names"] != (
        expected_optimizer_parameter_names
    ):
        raise ProtocolError("resume optimizer_parameter_names mismatch")
    if checkpoint["trainable_parameter_names"] != (
        expected_trainable_parameter_names
    ):
        raise ProtocolError("resume trainable parameter names mismatch")
    root = Path(run_dir).resolve(strict=True)
    bundle_hash = _run_document_hash(
        root,
        checkpoint["dataset_bundle_relpath"],
        "dataset bundle",
    )
    if bundle_hash != checkpoint["dataset_bundle_sha256"]:
        raise ProtocolError("run-local dataset bundle hash mismatch")
    manifest_hash = _run_document_hash(
        root,
        checkpoint["split_manifest_relpath"],
        "split manifest",
    )
    if manifest_hash != checkpoint["split_manifest_sha256"]:
        raise ProtocolError("run-local split manifest hash mismatch")
    vocabulary_hash = _run_document_hash(
        root,
        "vocabulary.json",
        "vocabulary",
    )
    if vocabulary_hash != checkpoint["vocabulary_sha256"]:
        raise ProtocolError("run-local vocabulary hash mismatch")
    if read_json(root / "vocabulary.json") != checkpoint["vocabulary"]:
        raise ProtocolError(
            "embedded vocabulary differs from run-local vocabulary"
        )
    exclusion_path = checkpoint["train_exclusions_relpath"]
    if exclusion_path is not None:
        exclusion_hash = _run_document_hash(
            root,
            exclusion_path,
            "train exclusions",
        )
        if exclusion_hash != checkpoint["train_exclusions_sha256"]:
            raise ProtocolError("run-local train exclusions hash mismatch")


def restore_checkpoint(
    checkpoint: dict[str, object],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_parameter_names: list[str],
) -> None:
    validate_checkpoint_payload(checkpoint)
    current_names = _trainable_names(model)
    if optimizer_parameter_names != current_names:
        raise ProtocolError(
            "current optimizer_parameter_names do not match trainable model"
        )
    if checkpoint["optimizer_parameter_names"] != optimizer_parameter_names:
        raise ProtocolError(
            "checkpoint optimizer_parameter_names mismatch before "
            "optimizer.load_state_dict"
        )
    if checkpoint["trainable_parameter_names"] != current_names:
        raise ProtocolError("checkpoint trainable parameter names mismatch")
    load_trainable_state_dict(model, checkpoint["trainable_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
