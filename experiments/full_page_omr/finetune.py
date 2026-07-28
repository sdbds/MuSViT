import json
import hashlib
import re
import uuid
import copy
from dataclasses import dataclass, fields
from pathlib import Path

from fire import Fire
from loguru import logger
import numpy as np
from PIL import Image
import torch

from . import _globals
from .config.ExperimentConfigWrapper import ExperimentConfig, experiment_config_from_dict
from .data import SyntheticGrandStaffDataset, CLFinetuningDataset, SynthRealFinetuningDataset
from .pdmx_data import (
    PDMXConsumptionAuditCallback,
    PDMXPretrainingDataModule,
    PDMXVirtualEpochCallback,
)
from .migrate_vocabulary_checkpoint import load_vocabulary_aware_weights
from .smt_foundation import SMTFoundationConfig, SMTFoundationModelForCausalLM
from .optimization import (
    AdamWWSDConfig,
    optimizer_protocol_metadata,
    prepare_adamw_wsd_resume_state,
)
from .smt_trainer import (
    ENCODER_TRAINING_MODES,
    SAMPLES_SEEN_CHECKPOINT_KEY,
    SMTPP_Trainer,
)
from .data_augmentation.data_augmentation import set_up_processor

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


DATASETS_TYPE = {
    "CL": CLFinetuningDataset,
    "SR": SynthRealFinetuningDataset,
    "CL1": SyntheticGrandStaffDataset,
    "PDMX": PDMXPretrainingDataModule,
    "R": None
}

PROTOCOL_VERSION = "full_page_omr_adamw_wsd_4m_v1"
METRIC_VERSION = "canonical_v2"
CHECKPOINT_MONITOR = "val_SER_v2"
PRECISION = "16-mixed"
ACCUMULATE_GRAD_BATCHES = 1
MAX_EPOCHS = 100_000
CANONICAL_VALIDATION_EVERY_N_EPOCHS = 2_000


@dataclass(frozen=True)
class CheckpointRunState:
    path: str
    global_step: int | None
    curriculum_step: int | None
    curriculum_step_offset: int
    curriculum_step_source: str
    sha256: str | None
    protocol_snapshot: dict | None


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
def _validate_checkpoint_every_n_epochs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("checkpoint_every_n_epochs must be a positive integer")
    return value


def _validate_validation_every_n_epochs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("validation_every_n_epochs must be a positive integer")
    return value


def _normalize_task_learning_rate(task_learning_rate, learning_rate):
    if task_learning_rate is not None and learning_rate is not None:
        raise ValueError("task_learning_rate and learning_rate are mutually exclusive")
    value = task_learning_rate if task_learning_rate is not None else learning_rate
    if value is None:
        value = 1e-4
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("task_learning_rate must be positive")
    return float(value)


def _validate_canonical_protocol_contract(
        protocol_version: str,
        optimizer_config: AdamWWSDConfig,
        *,
        validation_every_n_epochs: int) -> None:
    if protocol_version != PROTOCOL_VERSION:
        return
    locked = AdamWWSDConfig()
    mismatches = {
        field.name: (
            getattr(optimizer_config, field.name),
            getattr(locked, field.name),
        )
        for field in fields(locked)
        if field.name not in {"max_steps", "decay_steps", "min_lr_ratio"}
        if getattr(optimizer_config, field.name) != getattr(locked, field.name)
    }
    if validation_every_n_epochs != CANONICAL_VALIDATION_EVERY_N_EPOCHS:
        mismatches["validation_every_n_epochs"] = (
            validation_every_n_epochs,
            CANONICAL_VALIDATION_EVERY_N_EPOCHS,
        )
    if mismatches:
        raise ValueError(
            f"{PROTOCOL_VERSION} has locked non-schedule optimizer and validation values; "
            f"use a new protocol_version for overrides: {mismatches}"
        )


def _validate_max_steps(value: int, *, train: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_steps must be an integer")
    if train and value < 1:
        raise ValueError("production training requires max_steps to be a positive integer")
    if not train and value != -1 and value < 1:
        raise ValueError("max_steps must be -1 or a positive integer when train=False")
    return value


def _validate_encoder_training_mode(value: str) -> str:
    if not isinstance(value, str) or value not in ENCODER_TRAINING_MODES:
        raise ValueError(
            f"encoder_training_mode must be one of {sorted(ENCODER_TRAINING_MODES)}, got {value!r}"
        )
    return value


def _validate_checkpoint_sources(from_checkpoint, starting_weights):
    normalized = []
    for field_name, value in (
        ("from_checkpoint", from_checkpoint),
        ("starting_weights", starting_weights),
    ):
        if value is None or (isinstance(value, str) and not value.strip()):
            normalized.append(None)
        elif not isinstance(value, str):
            raise TypeError(f"{field_name} must be a path string or None")
        else:
            normalized.append(value)

    if all(value is not None for value in normalized):
        raise ValueError("from_checkpoint and starting_weights are mutually exclusive")
    return tuple(normalized)


def _validate_source_vocab_manifest(
    starting_weights,
    source_vocab_manifest,
) -> str | None:
    if source_vocab_manifest is None or (
        isinstance(source_vocab_manifest, str)
        and not source_vocab_manifest.strip()
    ):
        return None
    if not isinstance(source_vocab_manifest, str):
        raise TypeError("source_vocab_manifest must be a path string or None")
    if starting_weights is None:
        raise ValueError(
            "source_vocab_manifest is valid only with starting_weights"
        )
    return source_vocab_manifest


def _validate_non_negative_integer(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_stream_resume_boundary(
    *,
    samples_seen: int,
    steps_per_epoch: int,
    mode: str,
) -> None:
    if mode != "virtual_epoch_boundary":
        raise ValueError(f"unsupported stream resume mode: {mode!r}")
    samples_seen = _validate_non_negative_integer(
        samples_seen,
        "stream samples_seen",
    )
    if (
        isinstance(steps_per_epoch, bool)
        or not isinstance(steps_per_epoch, int)
        or steps_per_epoch < 1
    ):
        raise ValueError("stream steps_per_epoch must be a positive integer")
    if samples_seen % steps_per_epoch != 0:
        raise ValueError(
            "PDMX full resume requires a virtual epoch boundary: "
            f"samples_seen={samples_seen}, steps_per_epoch={steps_per_epoch}"
        )


def _data_protocol_metadata(data) -> dict:
    provider = getattr(data, "protocol_metadata", None)
    if provider is None:
        return {}
    if not callable(provider):
        raise TypeError("data protocol_metadata must be callable")
    metadata = provider()
    if not isinstance(metadata, dict):
        raise TypeError("data protocol_metadata() must return a dictionary")
    return copy.deepcopy(metadata)


def _validate_protocol_version(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("protocol_version must be a non-empty string")
    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint_run_state(checkpoint_path: str,
                               *, expected_sha256: str | None = None,
                               allow_missing_global_step: bool = False,
                               require_samples_seen: bool = False,
                               resume_state_validator=None) -> CheckpointRunState:
    path = Path(checkpoint_path).expanduser().resolve()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError("source_checkpoint_sha256 must be a lowercase SHA-256 digest")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None:
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "source checkpoint SHA-256 mismatch: "
                f"declared={expected_sha256}, actual={actual_sha256}"
            )
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary payload: {path}")

    if allow_missing_global_step and "global_step" not in payload:
        global_step = None
    else:
        global_step = _validate_non_negative_integer(
            payload.get("global_step"),
            "checkpoint global_step",
        )
    hyper_parameters = payload.get("hyper_parameters", {})
    if not isinstance(hyper_parameters, dict):
        raise ValueError("checkpoint hyper_parameters must be a dictionary")
    protocol_snapshot = hyper_parameters.get("protocol_snapshot")
    if protocol_snapshot is not None and not isinstance(protocol_snapshot, dict):
        raise ValueError("checkpoint protocol_snapshot must be a dictionary")
    if SAMPLES_SEEN_CHECKPOINT_KEY in payload:
        samples_seen = _validate_non_negative_integer(
            payload[SAMPLES_SEEN_CHECKPOINT_KEY],
            f"checkpoint {SAMPLES_SEEN_CHECKPOINT_KEY}",
        )
        offset = _validate_non_negative_integer(
            hyper_parameters.get("curriculum_step_offset", 0),
            "checkpoint curriculum_step_offset",
        )
        curriculum_step = offset + samples_seen
        curriculum_step_source = "checkpoint_samples_seen"
    else:
        if require_samples_seen:
            raise ValueError(
                f"full resume requires checkpoint {SAMPLES_SEEN_CHECKPOINT_KEY}"
            )
        offset = 0
        curriculum_step = global_step
        curriculum_step_source = (
            "legacy_global_step" if global_step is not None else "unavailable"
        )

    if resume_state_validator is not None:
        resume_state_validator(payload)
    del payload
    return CheckpointRunState(
        path=str(path),
        global_step=global_step,
        curriculum_step=curriculum_step,
        curriculum_step_offset=offset,
        curriculum_step_source=curriculum_step_source,
        sha256=actual_sha256,
        protocol_snapshot=copy.deepcopy(protocol_snapshot),
    )


def _validate_run_contract(*, config, from_checkpoint, starting_weights,
                           max_steps: int, train: bool, protocol_version: str,
                           source_curriculum_step: int | None,
                           source_checkpoint_sha256: str | None = None,
                           expected_protocol_snapshot: dict | None = None,
                           expected_model=None,
                           optimizer_config: AdamWWSDConfig | None = None):
    protocol_version = _validate_protocol_version(protocol_version)
    if from_checkpoint is not None:
        if source_curriculum_step is not None or source_checkpoint_sha256 is not None:
            raise ValueError(
                "source_curriculum_step and source_checkpoint_sha256 are only valid "
                "with starting_weights"
            )
        if not train:
            return _read_checkpoint_run_state(
                from_checkpoint,
                allow_missing_global_step=True,
            )
        if expected_model is None or optimizer_config is None:
            raise ValueError(
                "full resume requires the expected model and optimizer config "
                "for state validation"
            )
        if expected_protocol_snapshot is None or not isinstance(
            expected_protocol_snapshot,
            dict,
        ):
            raise ValueError(
                "full resume requires a complete expected protocol_snapshot"
            )
        if expected_protocol_snapshot.get("protocol_version") != protocol_version:
            raise ValueError(
                "expected protocol snapshot mismatch: "
                f"protocol_version={expected_protocol_snapshot.get('protocol_version')!r}, "
                f"requested={protocol_version!r}"
            )
        state = _read_checkpoint_run_state(
            from_checkpoint,
            require_samples_seen=True,
            resume_state_validator=lambda payload: prepare_adamw_wsd_resume_state(
                payload,
                expected_model,
                optimizer_config,
                expected_protocol_snapshot,
                mutate=False,
            ),
        )
        requested_offset = config.data.skip_steps
        if state.curriculum_step_source == "legacy_global_step" and requested_offset != 0:
            raise ValueError(
                "legacy full resume requires config data.skip_steps=0 because the "
                "checkpoint has no curriculum_step_offset evidence"
            )
        if requested_offset != state.curriculum_step_offset:
            raise ValueError(
                "config data.skip_steps must match checkpoint curriculum_step_offset: "
                f"{requested_offset} != {state.curriculum_step_offset}"
            )
        if train and max_steps <= state.global_step:
            raise ValueError(
                f"max_steps ({max_steps}) must exceed resumed checkpoint "
                f"global_step ({state.global_step})"
            )
        if state.protocol_snapshot is None:
            raise ValueError(
                "checkpoint is missing protocol_snapshot evidence required for full resume"
            )
        if state.protocol_snapshot.get("protocol_version") != protocol_version:
            raise ValueError(
                "checkpoint protocol snapshot mismatch: "
                f"protocol_version="
                f"{state.protocol_snapshot.get('protocol_version')!r}, "
                f"expected={protocol_version!r}"
            )
        return state

    if starting_weights is not None:
        if source_curriculum_step is None:
            raise ValueError(
                "source_curriculum_step is required with starting_weights"
            )
        source_curriculum_step = _validate_non_negative_integer(
            source_curriculum_step,
            "source_curriculum_step",
        )
        if protocol_version == PROTOCOL_VERSION:
            raise ValueError(
                "a weights-only experiment fork must use its own protocol_version"
            )
        if config.data.skip_steps != source_curriculum_step:
            raise ValueError(
                "config data.skip_steps must equal source_curriculum_step: "
                f"{config.data.skip_steps} != {source_curriculum_step}"
            )
        if source_checkpoint_sha256 is None:
            raise ValueError(
                "source_checkpoint_sha256 is required with starting_weights"
            )
        state = _read_checkpoint_run_state(
            starting_weights,
            expected_sha256=source_checkpoint_sha256,
        )
        if state.curriculum_step != source_curriculum_step:
            raise ValueError(
                "source_curriculum_step does not match checkpoint curriculum_step: "
                f"{source_curriculum_step} != {state.curriculum_step}"
            )
        return state

    if source_curriculum_step is not None or source_checkpoint_sha256 is not None:
        raise ValueError(
            "source_curriculum_step and source_checkpoint_sha256 require starting_weights"
        )
    if config.data.skip_steps != 0:
        raise ValueError("fresh training requires data.skip_steps=0")
    return None


def _feature_grid_for_resolution(resolution: int, patch_size) -> tuple[int, int]:
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")

    if isinstance(patch_size, int) and not isinstance(patch_size, bool):
        patch_h = patch_w = patch_size
    elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        patch_h, patch_w = patch_size
    else:
        raise ValueError(f"Unsupported encoder patch_size: {patch_size!r}")

    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (patch_h, patch_w)):
        raise ValueError(f"Unsupported encoder patch_size: {patch_size!r}")
    if patch_h != patch_w:
        raise ValueError(f"Full-page OMR requires square encoder patches, got {patch_size!r}")
    if resolution % patch_h != 0 or resolution % patch_w != 0:
        raise ValueError(
            f"resolution {resolution} must be divisible by encoder patch_size {patch_size!r}"
        )
    return resolution // patch_h, resolution // patch_w


def _build_epoch_checkpointer(experiment_name: str, finetuning_technique: str,
                              checkpoint_every_n_epochs: int) -> ModelCheckpoint:
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
    return ModelCheckpoint(
        dirpath="weights/",
        filename=f"{experiment_name}_{finetuning_technique}-epoch",
        every_n_epochs=checkpoint_every_n_epochs,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
        save_top_k=1,
        verbose=True,
    )


def _build_metric_checkpointer(experiment_name: str,
                               finetuning_technique: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath="weights/",
        filename=(
            f"{experiment_name}_{finetuning_technique}"
            "-step={step}-val_SER_v2={val_SER_v2:.4f}"
        ),
        monitor=CHECKPOINT_MONITOR,
        mode="min",
        save_top_k=2,
        save_weights_only=True,
        auto_insert_metric_name=False,
        verbose=True,
    )


def _build_trainer_kwargs(*, max_steps: int, validation_every_n_epochs: int,
                          callbacks, logger):
    return {
        "max_epochs": MAX_EPOCHS,
        "max_steps": max_steps,
        "check_val_every_n_epoch": validation_every_n_epochs,
        "val_check_interval": 1.0,
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
        "logger": logger,
        "precision": PRECISION,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
    }


def _build_protocol_snapshot(*, max_steps, validation_every_n_epochs,
                             encoder_training_mode, encoder_unfreeze_step,
                             curriculum_step_offset, resolution, reduce_ratio,
                             batch_size, protocol_version,
                             expected_training_batches_per_epoch,
                             curriculum_steady_mixture_step,
                             finetuning_technique, attention_backend,
                             tokenization_mode, num_workers,
                             checkpoint_every_n_epochs, optimizer_metadata,
                             foundation_architecture=None,
                             foundation_weights=None,
                             data_protocol=None):
    required_optimizer_fields = {
        "optimizer",
        "optimizer_implementation",
        "torch_version",
        "optimizer_protocol",
        "optimizer_betas",
        "optimizer_eps",
        "optimizer_amsgrad",
        "optimizer_groups",
        "scheduler",
        "scheduler_interval",
        "scheduler_frequency",
        "wsd_max_steps",
        "wsd_warmup_steps",
        "wsd_stable_steps",
        "wsd_decay_steps",
        "wsd_warmup_type",
        "wsd_decay_type",
        "wsd_min_lr_ratio",
        "wsd_num_cycles",
    }
    missing = required_optimizer_fields - set(optimizer_metadata)
    if missing:
        raise ValueError(
            "optimizer metadata is incomplete for protocol_snapshot: "
            f"{sorted(missing)}"
        )
    expected_validation_count = None
    if expected_training_batches_per_epoch:
        expected_validation_count = max_steps // (
            expected_training_batches_per_epoch * validation_every_n_epochs
        )
    if data_protocol is None:
        data_protocol = {}
    if not isinstance(data_protocol, dict):
        raise TypeError("data_protocol must be a dictionary")
    data_snapshot = {"num_workers": num_workers}
    collisions = set(data_snapshot) & set(data_protocol)
    if collisions:
        raise ValueError(
            f"data protocol metadata collides with base fields: {sorted(collisions)}"
        )
    data_snapshot.update(copy.deepcopy(data_protocol))

    snapshot = {
        "protocol_version": protocol_version,
        "optimizer": {
            "class": optimizer_metadata["optimizer"],
            "implementation": optimizer_metadata["optimizer_implementation"],
            "torch_version": optimizer_metadata["torch_version"],
            "protocol": optimizer_metadata["optimizer_protocol"],
            "betas": optimizer_metadata["optimizer_betas"],
            "eps": optimizer_metadata["optimizer_eps"],
            "amsgrad": optimizer_metadata["optimizer_amsgrad"],
            "groups": optimizer_metadata["optimizer_groups"],
        },
        "scheduler": {
            "helper": optimizer_metadata["scheduler"],
            "interval": optimizer_metadata["scheduler_interval"],
            "frequency": optimizer_metadata["scheduler_frequency"],
            "max_steps": optimizer_metadata["wsd_max_steps"],
            "warmup_steps": optimizer_metadata["wsd_warmup_steps"],
            "stable_steps": optimizer_metadata["wsd_stable_steps"],
            "decay_steps": optimizer_metadata["wsd_decay_steps"],
            "warmup_type": optimizer_metadata["wsd_warmup_type"],
            "decay_type": optimizer_metadata["wsd_decay_type"],
            "min_lr_ratio": optimizer_metadata["wsd_min_lr_ratio"],
            "num_cycles": optimizer_metadata["wsd_num_cycles"],
        },
        "trainer": {
            "max_steps": max_steps,
            "max_epochs": MAX_EPOCHS,
            "precision": PRECISION,
            "batch_size": batch_size,
            "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        },
        "validation": {
            "every_n_epochs": validation_every_n_epochs,
            "first_epoch": validation_every_n_epochs,
            "expected_count": expected_validation_count,
            "expected_training_batches_per_epoch": expected_training_batches_per_epoch,
            "val_check_interval": 1.0,
            "num_sanity_val_steps": 0,
        },
        "curriculum": {
            "encoder_training_mode": encoder_training_mode,
            "encoder_unfreeze_step": encoder_unfreeze_step,
            "steady_mixture_step": curriculum_steady_mixture_step,
            "step_offset": curriculum_step_offset,
        },
        "metrics": {
            "version": METRIC_VERSION,
            "monitor": CHECKPOINT_MONITOR,
        },
        "input": {
            "resolution": resolution,
            "reduce_ratio": reduce_ratio,
            "tokenization_mode": tokenization_mode,
        },
        "model": {
            "foundation_architecture": foundation_architecture,
            "foundation_weights": foundation_weights,
            "finetuning_technique": finetuning_technique,
            "attention_backend": attention_backend,
        },
        "data": data_snapshot,
        "checkpointing": {
            "every_n_epochs": checkpoint_every_n_epochs,
            "metric_save_top_k": 2,
            "metric_save_weights_only": True,
        },
    }
    return copy.deepcopy(snapshot)


def _build_protocol_metadata(*, max_steps, validation_every_n_epochs,
                             from_checkpoint, starting_weights,
                             encoder_training_mode, encoder_unfreeze_step,
                             resolution, reduce_ratio, batch_size,
                             protocol_version=PROTOCOL_VERSION,
                             source_curriculum_step=None,
                             checkpoint_state=None,
                             curriculum_step_offset=0,
                             finetuning_technique=None,
                             attention_backend=None,
                             tokenization_mode=None,
                             num_workers=None,
                             checkpoint_every_n_epochs=None,
                             expected_training_batches_per_epoch=None,
                             curriculum_steady_mixture_step=320_000,
                             optimizer_metadata=None,
                             trainer_max_steps=None,
                             train=True,
                             foundation_architecture=None,
                             foundation_weights=None,
                             data_protocol=None):
    if from_checkpoint is not None:
        checkpoint_source = from_checkpoint
        checkpoint_load_mode = "full" if train else "evaluation_weights_only"
    elif starting_weights is not None:
        checkpoint_source = starting_weights
        checkpoint_load_mode = "weights_only"
    else:
        checkpoint_source = "foundation"
        checkpoint_load_mode = "fresh"
    if checkpoint_state is not None:
        checkpoint_source = checkpoint_state.path
    recorded_source_curriculum_step = source_curriculum_step
    if recorded_source_curriculum_step is None and checkpoint_state is not None:
        recorded_source_curriculum_step = checkpoint_state.curriculum_step
    validation_expected_count = None
    if expected_training_batches_per_epoch:
        validation_expected_count = max_steps // (
            expected_training_batches_per_epoch * validation_every_n_epochs
        )
    metadata = {
        "protocol_version": protocol_version,
        "metric_version": METRIC_VERSION,
        "max_steps": max_steps,
        "trainer_max_steps": max_steps if trainer_max_steps is None else trainer_max_steps,
        "max_epochs": MAX_EPOCHS,
        "validation_every_n_epochs": validation_every_n_epochs,
        "validation_first_epoch": validation_every_n_epochs,
        "validation_expected_count": validation_expected_count,
        "expected_training_batches_per_epoch": expected_training_batches_per_epoch,
        "checkpoint_monitor": CHECKPOINT_MONITOR,
        "checkpoint_source": checkpoint_source,
        "checkpoint_load_mode": checkpoint_load_mode,
        "checkpoint_global_step": (
            checkpoint_state.global_step if checkpoint_state is not None else None
        ),
        "checkpoint_sha256": (
            checkpoint_state.sha256 if checkpoint_state is not None else None
        ),
        "source_curriculum_step": recorded_source_curriculum_step,
        "curriculum_step_offset": curriculum_step_offset,
        "source_curriculum_step_evidence": (
            checkpoint_state.curriculum_step_source
            if checkpoint_state is not None else None
        ),
        "encoder_training_mode": encoder_training_mode,
        "encoder_unfreeze_step": encoder_unfreeze_step,
        "curriculum_steady_mixture_step": curriculum_steady_mixture_step,
        "finetuning_technique": finetuning_technique,
        "attention_backend": attention_backend,
        "tokenization_mode": tokenization_mode,
        "num_workers": num_workers,
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
        "resolution": resolution,
        "reduce_ratio": reduce_ratio,
        "precision": PRECISION,
        "batch_size": batch_size,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
    }
    if data_protocol is None:
        data_protocol = {}
    if not isinstance(data_protocol, dict):
        raise TypeError("data_protocol must be a dictionary")
    data_collisions = set(metadata) & set(data_protocol)
    if data_collisions:
        raise ValueError(
            "data protocol metadata collides with run metadata: "
            f"{sorted(data_collisions)}"
        )
    metadata.update(copy.deepcopy(data_protocol))
    if optimizer_metadata is None:
        raise ValueError("optimizer_metadata is required for complete protocol metadata")
    collisions = set(metadata) & set(optimizer_metadata)
    if collisions:
        raise ValueError(
            f"optimizer metadata collides with run metadata: {sorted(collisions)}"
        )
    metadata.update(optimizer_metadata)
    metadata["protocol_snapshot"] = _build_protocol_snapshot(
        max_steps=max_steps,
        validation_every_n_epochs=validation_every_n_epochs,
        encoder_training_mode=encoder_training_mode,
        encoder_unfreeze_step=encoder_unfreeze_step,
        curriculum_step_offset=curriculum_step_offset,
        resolution=resolution,
        reduce_ratio=reduce_ratio,
        batch_size=batch_size,
        protocol_version=protocol_version,
        expected_training_batches_per_epoch=expected_training_batches_per_epoch,
        curriculum_steady_mixture_step=curriculum_steady_mixture_step,
        finetuning_technique=finetuning_technique,
        attention_backend=attention_backend,
        tokenization_mode=tokenization_mode,
        num_workers=num_workers,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        optimizer_metadata=optimizer_metadata,
        foundation_architecture=foundation_architecture,
        foundation_weights=foundation_weights,
        data_protocol=data_protocol,
    )
    return metadata


def _run_record_directory(experiment_name: str, protocol_version: str,
                          output_root: Path) -> Path:
    for field_name, value in (
        ("experiment_name", experiment_name),
        ("protocol_version", protocol_version),
    ):
        if not isinstance(value, str) or not value.strip() or Path(value).name != value:
            raise ValueError(f"{field_name} must be a single non-empty path component")
    directory = Path(output_root) / experiment_name / protocol_version
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_json(path: Path, payload) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path.resolve()


def _write_protocol_metadata(experiment_name: str, metadata,
                             *, output_root=Path("logs")) -> Path:
    protocol_version = _validate_protocol_version(metadata.get("protocol_version"))
    directory = _run_record_directory(
        experiment_name,
        protocol_version,
        Path(output_root),
    )
    return _write_json(directory / "protocol.json", metadata)


def _audit_image_layout(image) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        return "HW"
    if array.ndim == 3 and array.shape[2] in (1, 3, 4):
        return "HWC"
    raise ValueError(f"audit image must be HW or HWC, got shape {array.shape}")


def _save_audit_image(path: Path, image) -> None:
    array = np.asarray(image)
    _audit_image_layout(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)


def _final_tensor_as_hwc(final: torch.Tensor) -> np.ndarray:
    if not isinstance(final, torch.Tensor) or final.ndim != 4 or final.shape[0] != 1:
        raise ValueError("final audit image must be a rank-4 tensor with batch size 1")
    if final.shape[1] not in (1, 3, 4):
        raise ValueError(f"final audit tensor has unsupported channels: {final.shape[1]}")
    return (
        final.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def _write_resize_audit(data, *, experiment_name: str, protocol_version: str,
                        reduce_ratio: float, resolution: int,
                        output_root=Path("logs")) -> Path | None:
    source = getattr(getattr(data, "train_dataset", None), "real_source", None)
    source_split = "train"
    if source is None:
        source = getattr(getattr(data, "val_dataset", None), "real_source", None)
        source_split = "val"
    if source is None or not hasattr(source, "resize_audit"):
        return None

    stages = source.resize_audit(0)
    final_shape = list(stages.final.shape)
    expected_shape = [1, 3, resolution, resolution]
    if final_shape != expected_shape:
        raise ValueError(
            f"final resize audit shape must be {expected_shape}, got {final_shape}"
        )

    directory = _run_record_directory(
        experiment_name,
        protocol_version,
        Path(output_root),
    ) / "input_audit"
    directory.mkdir(parents=True, exist_ok=True)
    _save_audit_image(directory / "raw.png", stages.raw)
    _save_audit_image(directory / "intermediate.png", stages.intermediate)
    _save_audit_image(directory / "final.png", _final_tensor_as_hwc(stages.final))
    return _write_json(
        directory / "resize.json",
        {
            "audit_scope": f"fixed_real_{source_split}_row_0",
            "row_index": 0,
            "batch_size": data.batch_size,
            "reduce_ratio": reduce_ratio,
            "raw_shape_hwc": list(stages.raw.shape),
            "intermediate_shape_hwc": list(stages.intermediate.shape),
            "raw_layout": _audit_image_layout(stages.raw),
            "intermediate_layout": _audit_image_layout(stages.intermediate),
            "final_shape_nchw": final_shape,
            "images": {
                "raw": "raw.png",
                "intermediate": "intermediate.png",
                "final": "final.png",
            },
        },
    )


class _FirstTrainBatchInputAudit(Callback):
    def __init__(self, output_path) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self._written = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        del trainer, pl_module
        if self._written:
            return
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise ValueError("first CL training batch is missing input audit metadata")
        metadata = batch[3]
        if not isinstance(metadata, dict):
            raise ValueError("first CL training batch audit metadata must be a dictionary")
        actual_shape = list(batch[0].shape)
        if metadata.get("final_shape_nchw") != actual_shape:
            raise ValueError(
                "first CL training batch final shape does not match its audit metadata"
            )
        _write_json(
            self.output_path,
            {
                **metadata,
                "batch_index": int(batch_idx),
                "actual_batch_shape_nchw": actual_shape,
            },
        )
        self._written = True


def _select_test_checkpoint(trainer, checkpointer, experiment_name: str,
                            finetuning_technique: str) -> str:
    if checkpointer.best_model_path:
        return checkpointer.best_model_path

    checkpoint_path = f"weights/{experiment_name}_{finetuning_technique}-end.ckpt"
    trainer.save_checkpoint(checkpoint_path, weights_only=True)
    return checkpoint_path


def _run_test(trainer, model_wrapper, data, checkpoint_path):
    try:
        trainer.test(
            model_wrapper,
            datamodule=data,
            ckpt_path=checkpoint_path,
            weights_only=False,
        )
    except Exception:
        logger.exception("Failed to test checkpoint: {}", checkpoint_path)
        raise


def _run_test_if_available(
    trainer,
    model_wrapper,
    data,
    checkpoint_path,
) -> bool:
    if getattr(data, "has_test_split", True) is False:
        logger.info("Skipping test: data module explicitly declares no test split")
        return False
    _run_test(trainer, model_wrapper, data, checkpoint_path)
    return True


def _data_callbacks(data, run_instance_directory: str | Path) -> list[Callback]:
    if not callable(getattr(data, "set_train_epoch", None)):
        return []
    callbacks: list[Callback] = [PDMXVirtualEpochCallback()]
    data_protocol = _data_protocol_metadata(data)
    if data_protocol.get("stream_resume_mode") == "virtual_epoch_boundary":
        callbacks.append(
            PDMXConsumptionAuditCallback(run_instance_directory)
        )
    return callbacks


def _target_vocab_manifest_path(data) -> Path:
    explicit = getattr(data, "vocab_manifest_path", None)
    if explicit is not None:
        return Path(explicit)
    vocab_name = getattr(data, "vocab_name", None)
    if not isinstance(vocab_name, str) or not vocab_name:
        raise ValueError(
            "vocabulary migration requires a target vocabulary manifest"
        )
    return Path(__file__).resolve().parent / "vocab" / f"{vocab_name}.json"


def main(config: ExperimentConfig, experiment_name,
         foundation_architecture="ViTMAEBase", foundation_weights="carlospm12/LSMT-MAE-Base-1024-16",
         finetuning_technique="CL", from_checkpoint: str | None = None, resolution: int | None = None,
         max_steps: int = 4_000_000, train: bool = True, starting_weights: str | None = None,
         task_learning_rate: float | None = None,
         learning_rate: float | None = None,
         encoder_learning_rate: float = 1e-5, weight_decay: float = 0.01,
         wsd_warmup_steps: int = 10_000, wsd_decay_steps: int = 400_000,
         wsd_warmup_type: str = "linear", wsd_decay_type: str = "cosine",
         wsd_min_lr_ratio: float = 0.0,
         attention_backend: str = "auto", checkpoint_every_n_epochs: int = 100,
         encoder_training_mode: str = "fine_tune",
         validation_every_n_epochs: int = 2_000,
         protocol_version: str = PROTOCOL_VERSION,
         source_curriculum_step: int | None = None,
         source_checkpoint_sha256: str | None = None,
         source_vocab_manifest: str | None = None):
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
    validation_every_n_epochs = _validate_validation_every_n_epochs(
        validation_every_n_epochs
    )
    trainer_max_steps = _validate_max_steps(max_steps, train=train)
    protocol_max_steps = (
        AdamWWSDConfig().max_steps
        if not train and trainer_max_steps == -1
        else trainer_max_steps
    )
    encoder_training_mode = _validate_encoder_training_mode(encoder_training_mode)
    protocol_version = _validate_protocol_version(protocol_version)
    task_learning_rate = _normalize_task_learning_rate(
        task_learning_rate,
        learning_rate,
    )
    optimizer_config = AdamWWSDConfig(
        task_learning_rate=task_learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        weight_decay=weight_decay,
        max_steps=protocol_max_steps,
        warmup_steps=wsd_warmup_steps,
        decay_steps=wsd_decay_steps,
        warmup_type=wsd_warmup_type,
        decay_type=wsd_decay_type,
        min_lr_ratio=wsd_min_lr_ratio,
    )
    _validate_canonical_protocol_contract(
        protocol_version,
        optimizer_config,
        validation_every_n_epochs=validation_every_n_epochs,
    )
    from_checkpoint, starting_weights = _validate_checkpoint_sources(
        from_checkpoint,
        starting_weights,
    )
    source_vocab_manifest = _validate_source_vocab_manifest(
        starting_weights,
        source_vocab_manifest,
    )
    if resolution is None:
        _globals.resolution = 1024
    else:
        _globals.resolution = resolution
    resolution = _globals.resolution

    # Credentials (WANDB_API_KEY / HUGGINGFACE_KEY) are loaded once by the `musvit`
    # launcher (see musvit/env.py) before this function runs.

    logger.info(f"Using {finetuning_technique} technique, implementing {DATASETS_TYPE[finetuning_technique]}")

    data = DATASETS_TYPE[finetuning_technique](config)
    data_protocol = _data_protocol_metadata(data)
    encoder_unfreeze_step = data.encoder_unfreeze_step
    curriculum_step_offset = data.curriculum_step_offset
    print("data_module_type:", type(data))

    set_up_processor(model=foundation_weights)

    logger.info(f"Creating MuSViT ({foundation_architecture}) from the weights: {foundation_weights}")
    logger.info(f"Decoder attention backend policy: {attention_backend}")
    smt_config = SMTFoundationConfig(
        foundation_architecture=foundation_architecture,
        foundation_weights=foundation_weights,
        maxlen=7512,
        out_categories=len(data.train_dataset.w2i),
        padding_token=0,
        in_channels=3,
        w2i=data.train_dataset.w2i,
        i2w=data.train_dataset.i2w,
        d_model=256,
        dim_ff=256,
        num_dec_layers=8,
        attention_backend=attention_backend,
    )
    model = SMTFoundationModelForCausalLM(smt_config)
    feature_grid = _feature_grid_for_resolution(resolution, model.encoder.config.patch_size)
    logger.info(
        "Training contract: mode={}, unfreeze_step={}, curriculum_step_offset={}, "
        "input_resolution={}x{}, "
        "feature_grid={}x{}, tokenization={}, batch_size={}, num_workers={}",
        encoder_training_mode,
        encoder_unfreeze_step,
        curriculum_step_offset,
        resolution,
        resolution,
        feature_grid[0],
        feature_grid[1],
        data.tokenization_mode,
        data.batch_size,
        data.num_workers,
    )

    optimizer_metadata = optimizer_protocol_metadata(model, optimizer_config)
    protocol_metadata = _build_protocol_metadata(
        max_steps=protocol_max_steps,
        trainer_max_steps=trainer_max_steps,
        validation_every_n_epochs=validation_every_n_epochs,
        from_checkpoint=from_checkpoint,
        starting_weights=starting_weights,
        encoder_training_mode=encoder_training_mode,
        encoder_unfreeze_step=encoder_unfreeze_step,
        resolution=resolution,
        reduce_ratio=config.data.reduce_ratio,
        batch_size=data.batch_size,
        protocol_version=protocol_version,
        source_curriculum_step=source_curriculum_step,
        curriculum_step_offset=curriculum_step_offset,
        finetuning_technique=finetuning_technique,
        attention_backend=attention_backend,
        tokenization_mode=data.tokenization_mode,
        num_workers=data.num_workers,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        expected_training_batches_per_epoch=len(data.train_dataset),
        curriculum_steady_mixture_step=320_000,
        optimizer_metadata=optimizer_metadata,
        train=train,
        foundation_architecture=foundation_architecture,
        foundation_weights=foundation_weights,
        data_protocol=data_protocol,
    )
    protocol_snapshot = protocol_metadata["protocol_snapshot"]
    checkpoint_state = _validate_run_contract(
        config=config,
        from_checkpoint=from_checkpoint,
        starting_weights=starting_weights,
        max_steps=protocol_max_steps,
        train=train,
        protocol_version=protocol_version,
        source_curriculum_step=source_curriculum_step,
        source_checkpoint_sha256=source_checkpoint_sha256,
        expected_protocol_snapshot=protocol_snapshot,
        expected_model=model,
        optimizer_config=optimizer_config,
    )
    if (
        checkpoint_state is not None
        and from_checkpoint is not None
        and train
        and data_protocol.get("stream_resume_mode") is not None
    ):
        _validate_stream_resume_boundary(
            samples_seen=(
                checkpoint_state.curriculum_step
                - checkpoint_state.curriculum_step_offset
            ),
            steps_per_epoch=data_protocol.get("steps_per_epoch"),
            mode=data_protocol["stream_resume_mode"],
        )
    if checkpoint_state is not None:
        protocol_metadata.update({
            "checkpoint_source": checkpoint_state.path,
            "checkpoint_global_step": checkpoint_state.global_step,
            "checkpoint_sha256": checkpoint_state.sha256,
            "source_curriculum_step": (
                source_curriculum_step
                if source_curriculum_step is not None
                else checkpoint_state.curriculum_step
            ),
            "source_curriculum_step_evidence": (
                checkpoint_state.curriculum_step_source
            ),
        })

    run_record_id = uuid.uuid4().hex
    run_record_root = Path("logs") / "run_instances" / run_record_id
    run_instance_directory = (
        run_record_root / experiment_name / protocol_version
    )
    protocol_metadata["run_record_id"] = run_record_id

    optimizer_wrapper_kwargs = {
        "run_protocol_version": protocol_version,
        "optimizer_protocol": optimizer_config.protocol,
        "task_learning_rate": optimizer_config.task_learning_rate,
        "encoder_learning_rate": optimizer_config.encoder_learning_rate,
        "weight_decay": optimizer_config.weight_decay,
        "max_steps": optimizer_config.max_steps,
        "wsd_warmup_steps": optimizer_config.warmup_steps,
        "wsd_decay_steps": optimizer_config.decay_steps,
        "wsd_warmup_type": optimizer_config.warmup_type,
        "wsd_decay_type": optimizer_config.decay_type,
        "wsd_min_lr_ratio": optimizer_config.min_lr_ratio,
        "protocol_snapshot": protocol_snapshot,
    }
    if starting_weights is None or source_vocab_manifest is not None:
        model_wrapper = SMTPP_Trainer(
            smt_config,
            model,
            encoder_training_mode=encoder_training_mode,
            encoder_unfreeze_step=encoder_unfreeze_step,
            curriculum_step_offset=curriculum_step_offset,
            batch_size=data.batch_size,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            **optimizer_wrapper_kwargs,
        )
        if source_vocab_manifest is not None:
            migration_report_path = (
                run_instance_directory / "vocabulary_migration.json"
            )
            migration_report = load_vocabulary_aware_weights(
                model_wrapper,
                starting_weights,
                source_vocab_manifest=source_vocab_manifest,
                target_vocab_manifest=_target_vocab_manifest_path(data),
                report_path=migration_report_path,
            )
            protocol_metadata.update(
                {
                    "source_vocab_manifest": source_vocab_manifest,
                    "vocabulary_migration_report_path": str(
                        migration_report_path.resolve()
                    ),
                    "vocabulary_migration": migration_report,
                }
            )
    else:
        model_wrapper = SMTPP_Trainer.load_from_checkpoint(
            starting_weights,
            smt_config=smt_config,
            smt_model=model,
            encoder_training_mode=encoder_training_mode,
            encoder_unfreeze_step=encoder_unfreeze_step,
            curriculum_step_offset=curriculum_step_offset,
            batch_size=data.batch_size,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            enforce_checkpoint_protocol=False,
            weights_only=False,
            **optimizer_wrapper_kwargs,
        )
        model_wrapper.enforce_checkpoint_protocol = True

    print(f"Checkpoints will be saved to \"{experiment_name}_{finetuning_technique}\"")
    epoch_checkpointer = _build_epoch_checkpointer(
        experiment_name,
        finetuning_technique,
        checkpoint_every_n_epochs,
    )
    checkpointer = _build_metric_checkpointer(
        experiment_name,
        finetuning_technique,
    )

    resize_audit_path = _write_resize_audit(
        data,
        experiment_name=experiment_name,
        protocol_version=protocol_version,
        reduce_ratio=config.data.reduce_ratio,
        resolution=resolution,
        output_root=run_record_root,
    )
    protocol_metadata["resize_audit_path"] = (
        str(resize_audit_path) if resize_audit_path is not None else None
    )
    protocol_metadata["resize_audit_status"] = (
        "archived" if resize_audit_path is not None else "not-applicable-synthetic-only"
    )
    trainer_callbacks = [epoch_checkpointer, checkpointer]
    if train:
        trainer_callbacks.extend(
            _data_callbacks(
                data,
                run_record_root / experiment_name / protocol_version,
            )
        )
    if train and finetuning_technique == "CL" and resize_audit_path is not None:
        first_batch_audit_path = resize_audit_path.parent / "first_train_batch.json"
        trainer_callbacks.append(_FirstTrainBatchInputAudit(first_batch_audit_path))
        protocol_metadata["first_train_batch_audit_path"] = str(
            first_batch_audit_path.resolve()
        )
        protocol_metadata["first_train_batch_audit_status"] = "pending"
    else:
        protocol_metadata["first_train_batch_audit_path"] = None
        protocol_metadata["first_train_batch_audit_status"] = "not-applicable"
    protocol_path = _write_protocol_metadata(
        experiment_name,
        protocol_metadata,
        output_root=run_record_root,
    )
    logger.info("Local run protocol: {}", protocol_path)
    logger.info("{}", json.dumps(protocol_metadata, sort_keys=True))

    wandb_logger = WandbLogger(project='Foundation_SMT',
                               name=f"{experiment_name}",
                               log_model=False, save_dir="wandb_logs/")
    wandb_logger.log_hyperparams(protocol_metadata)

    trainer = Trainer(**_build_trainer_kwargs(
        max_steps=trainer_max_steps,
        validation_every_n_epochs=validation_every_n_epochs,
        callbacks=trainer_callbacks,
        logger=wandb_logger,
    ))

    if train:
        trainer.fit(
            model_wrapper,
            datamodule=data,
            ckpt_path=from_checkpoint,
            weights_only=False,
        )

        from_checkpoint = _select_test_checkpoint(
            trainer,
            checkpointer,
            experiment_name,
            finetuning_technique,
        )

    _run_test_if_available(trainer, model_wrapper, data, from_checkpoint)


def launch(config_path: str, experiment_name: str,
           foundation_architecture="ViTMAEBase", foundation_weights="carlospm12/LSMT-MAE-Base-1024-16",
           finetuning: str = "CL", from_checkpoint: str | None = None, resolution: int | None = None,
           max_steps: int = 4_000_000, train: bool = True, starting_weights: str | None = None,
           task_learning_rate: float | None = None,
           learning_rate: float | None = None,
           encoder_learning_rate: float = 1e-5, weight_decay: float = 0.01,
           wsd_warmup_steps: int = 10_000, wsd_decay_steps: int = 400_000,
           wsd_warmup_type: str = "linear", wsd_decay_type: str = "cosine",
           wsd_min_lr_ratio: float = 0.0, attention_backend: str = "auto",
           checkpoint_every_n_epochs: int = 100,
           encoder_training_mode: str = "fine_tune",
           validation_every_n_epochs: int = 2_000,
           protocol_version: str = PROTOCOL_VERSION,
           source_curriculum_step: int | None = None,
           source_checkpoint_sha256: str | None = None,
           source_vocab_manifest: str | None = None):
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
    validation_every_n_epochs = _validate_validation_every_n_epochs(
        validation_every_n_epochs
    )
    max_steps = _validate_max_steps(max_steps, train=train)
    encoder_training_mode = _validate_encoder_training_mode(encoder_training_mode)
    protocol_version = _validate_protocol_version(protocol_version)
    task_learning_rate = _normalize_task_learning_rate(
        task_learning_rate,
        learning_rate,
    )
    from_checkpoint, starting_weights = _validate_checkpoint_sources(
        from_checkpoint,
        starting_weights,
    )
    source_vocab_manifest = _validate_source_vocab_manifest(
        starting_weights,
        source_vocab_manifest,
    )
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    main(config=config, experiment_name=experiment_name,
         foundation_architecture=foundation_architecture,
         foundation_weights=foundation_weights, finetuning_technique=finetuning, from_checkpoint=from_checkpoint,
         resolution=resolution, max_steps=max_steps, train=train, starting_weights=starting_weights,
         task_learning_rate=task_learning_rate,
         encoder_learning_rate=encoder_learning_rate, weight_decay=weight_decay,
         wsd_warmup_steps=wsd_warmup_steps, wsd_decay_steps=wsd_decay_steps,
         wsd_warmup_type=wsd_warmup_type, wsd_decay_type=wsd_decay_type,
         wsd_min_lr_ratio=wsd_min_lr_ratio,
         attention_backend=attention_backend, checkpoint_every_n_epochs=checkpoint_every_n_epochs,
         encoder_training_mode=encoder_training_mode,
         validation_every_n_epochs=validation_every_n_epochs,
         protocol_version=protocol_version,
         source_curriculum_step=source_curriculum_step,
         source_checkpoint_sha256=source_checkpoint_sha256,
         source_vocab_manifest=source_vocab_manifest)


if __name__ == "__main__":
    # When run directly (instead of through `musvit`), load the root .env and log in.
    try:
        from musvit.env import setup as _setup_env

        _setup_env()
    except Exception as exc:
        logger.warning(
            "Optional environment setup failed: {}: {}",
            type(exc).__name__,
            exc,
        )
    Fire(launch)
