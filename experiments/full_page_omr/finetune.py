import json
from dataclasses import dataclass
from pathlib import Path

from fire import Fire
from loguru import logger
import torch

from . import _globals
from .config.ExperimentConfigWrapper import ExperimentConfig, experiment_config_from_dict
from .data import SyntheticGrandStaffDataset, CLFinetuningDataset, SynthRealFinetuningDataset
from .smt_foundation import SMTFoundationConfig, SMTFoundationModelForCausalLM
from .smt_trainer import (
    ENCODER_TRAINING_MODES,
    SAMPLES_SEEN_CHECKPOINT_KEY,
    SMTPP_Trainer,
)
from .data_augmentation.data_augmentation import set_up_processor

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


DATASETS_TYPE = {
    "CL": CLFinetuningDataset,
    "SR": SynthRealFinetuningDataset,
    "CL1": SyntheticGrandStaffDataset,
    "R": None
}

PROTOCOL_VERSION = "full_page_omr_eval_v2"
METRIC_VERSION = "canonical_v2"
CHECKPOINT_MONITOR = "val_SER_v2"
PRECISION = "16-mixed"
ACCUMULATE_GRAD_BATCHES = 1


@dataclass(frozen=True)
class CheckpointRunState:
    path: str
    global_step: int
    curriculum_step: int
    curriculum_step_source: str


def _validate_checkpoint_every_n_epochs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("checkpoint_every_n_epochs must be a positive integer")
    return value


def _validate_validation_every_n_batches(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("validation_every_n_batches must be a positive integer")
    return value


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


def _validate_non_negative_integer(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_protocol_version(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("protocol_version must be a non-empty string")
    return value.strip()


def _read_checkpoint_run_state(checkpoint_path: str) -> CheckpointRunState:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary payload: {path}")

    global_step = _validate_non_negative_integer(
        payload.get("global_step"),
        "checkpoint global_step",
    )
    if SAMPLES_SEEN_CHECKPOINT_KEY in payload:
        samples_seen = _validate_non_negative_integer(
            payload[SAMPLES_SEEN_CHECKPOINT_KEY],
            f"checkpoint {SAMPLES_SEEN_CHECKPOINT_KEY}",
        )
        hyper_parameters = payload.get("hyper_parameters", {})
        if not isinstance(hyper_parameters, dict):
            raise ValueError("checkpoint hyper_parameters must be a dictionary")
        offset = _validate_non_negative_integer(
            hyper_parameters.get("curriculum_step_offset", 0),
            "checkpoint curriculum_step_offset",
        )
        curriculum_step = offset + samples_seen
        curriculum_step_source = "checkpoint_samples_seen"
    else:
        curriculum_step = global_step
        curriculum_step_source = "legacy_global_step"

    del payload
    return CheckpointRunState(
        path=str(path),
        global_step=global_step,
        curriculum_step=curriculum_step,
        curriculum_step_source=curriculum_step_source,
    )


def _validate_run_contract(*, config, from_checkpoint, starting_weights,
                           max_steps: int, train: bool, protocol_version: str,
                           source_curriculum_step: int | None):
    protocol_version = _validate_protocol_version(protocol_version)
    if from_checkpoint is not None:
        if source_curriculum_step is not None:
            raise ValueError(
                "source_curriculum_step is only valid with starting_weights"
            )
        state = _read_checkpoint_run_state(from_checkpoint)
        if train and max_steps <= state.global_step:
            raise ValueError(
                f"max_steps ({max_steps}) must exceed resumed checkpoint "
                f"global_step ({state.global_step})"
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
        state = _read_checkpoint_run_state(starting_weights)
        if state.curriculum_step != source_curriculum_step:
            raise ValueError(
                "source_curriculum_step does not match checkpoint curriculum_step: "
                f"{source_curriculum_step} != {state.curriculum_step}"
            )
        return state

    if source_curriculum_step is not None:
        raise ValueError("source_curriculum_step requires starting_weights")
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


def _build_trainer_kwargs(*, max_steps: int, validation_every_n_batches: int,
                          callbacks, logger):
    return {
        "max_epochs": 100000,
        "max_steps": max_steps,
        "check_val_every_n_epoch": None,
        "val_check_interval": validation_every_n_batches,
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
        "logger": logger,
        "precision": PRECISION,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
    }


def _build_protocol_metadata(*, max_steps, validation_every_n_batches,
                             from_checkpoint, starting_weights,
                             encoder_training_mode, encoder_unfreeze_step,
                             resolution, reduce_ratio, batch_size,
                             protocol_version=PROTOCOL_VERSION,
                             source_curriculum_step=None,
                             checkpoint_state=None):
    if from_checkpoint is not None:
        checkpoint_source = from_checkpoint
        checkpoint_load_mode = "full"
    elif starting_weights is not None:
        checkpoint_source = starting_weights
        checkpoint_load_mode = "weights_only"
    else:
        checkpoint_source = "foundation"
        checkpoint_load_mode = "fresh"
    return {
        "protocol_version": protocol_version,
        "metric_version": METRIC_VERSION,
        "max_steps": max_steps,
        "validation_every_n_batches": validation_every_n_batches,
        "checkpoint_monitor": CHECKPOINT_MONITOR,
        "checkpoint_source": checkpoint_source,
        "checkpoint_load_mode": checkpoint_load_mode,
        "checkpoint_global_step": (
            checkpoint_state.global_step if checkpoint_state is not None else None
        ),
        "source_curriculum_step": source_curriculum_step,
        "source_curriculum_step_evidence": (
            checkpoint_state.curriculum_step_source
            if checkpoint_state is not None else None
        ),
        "encoder_training_mode": encoder_training_mode,
        "encoder_unfreeze_step": encoder_unfreeze_step,
        "resolution": resolution,
        "reduce_ratio": reduce_ratio,
        "optimizer": "Adam",
        "learning_rate": _globals.learning_rate,
        "precision": PRECISION,
        "batch_size": batch_size,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
    }


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


def main(config: ExperimentConfig, experiment_name,
         foundation_architecture="ViTMAEBase", foundation_weights="carlospm12/LSMT-MAE-Base-1024-16",
         finetuning_technique="CL", from_checkpoint: str | None = None, resolution: int | None = None,
         max_steps: int = 320000, train: bool = True, starting_weights: str | None = None,
         attention_backend: str = "auto", checkpoint_every_n_epochs: int = 100,
         encoder_training_mode: str = "fine_tune",
         validation_every_n_batches: int = 10000,
         protocol_version: str = PROTOCOL_VERSION,
         source_curriculum_step: int | None = None):
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
    validation_every_n_batches = _validate_validation_every_n_batches(
        validation_every_n_batches
    )
    max_steps = _validate_max_steps(max_steps, train=train)
    encoder_training_mode = _validate_encoder_training_mode(encoder_training_mode)
    from_checkpoint, starting_weights = _validate_checkpoint_sources(
        from_checkpoint,
        starting_weights,
    )
    checkpoint_state = _validate_run_contract(
        config=config,
        from_checkpoint=from_checkpoint,
        starting_weights=starting_weights,
        max_steps=max_steps,
        train=train,
        protocol_version=protocol_version,
        source_curriculum_step=source_curriculum_step,
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

    if starting_weights is None:
        model_wrapper = SMTPP_Trainer(
            smt_config,
            model,
            encoder_training_mode=encoder_training_mode,
            encoder_unfreeze_step=encoder_unfreeze_step,
            curriculum_step_offset=curriculum_step_offset,
            batch_size=data.batch_size,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
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

    wandb_logger = WandbLogger(project='Foundation_SMT',
                               name=f"{experiment_name}",
                               log_model=False, save_dir="wandb_logs/")
    wandb_logger.log_hyperparams(
        _build_protocol_metadata(
            max_steps=max_steps,
            validation_every_n_batches=validation_every_n_batches,
            from_checkpoint=from_checkpoint,
            starting_weights=starting_weights,
            encoder_training_mode=encoder_training_mode,
            encoder_unfreeze_step=encoder_unfreeze_step,
            resolution=resolution,
            reduce_ratio=config.data.reduce_ratio,
            batch_size=data.batch_size,
            protocol_version=protocol_version,
            source_curriculum_step=source_curriculum_step,
            checkpoint_state=checkpoint_state,
        )
    )

    trainer = Trainer(**_build_trainer_kwargs(
        max_steps=max_steps,
        validation_every_n_batches=validation_every_n_batches,
        callbacks=[epoch_checkpointer, checkpointer],
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

    _run_test(trainer, model_wrapper, data, from_checkpoint)


def launch(config_path: str, experiment_name: str,
           foundation_architecture="ViTMAEBase", foundation_weights="carlospm12/LSMT-MAE-Base-1024-16",
           finetuning: str = "CL", from_checkpoint: str | None = None, resolution: int | None = None,
           max_steps: int = 320000, train: bool = True, starting_weights: str | None = None,
           learning_rate: float | None = None, attention_backend: str = "auto",
           checkpoint_every_n_epochs: int = 100,
           encoder_training_mode: str = "fine_tune",
           validation_every_n_batches: int = 10000,
           protocol_version: str = PROTOCOL_VERSION,
           source_curriculum_step: int | None = None):
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
    validation_every_n_batches = _validate_validation_every_n_batches(
        validation_every_n_batches
    )
    max_steps = _validate_max_steps(max_steps, train=train)
    encoder_training_mode = _validate_encoder_training_mode(encoder_training_mode)
    from_checkpoint, starting_weights = _validate_checkpoint_sources(
        from_checkpoint,
        starting_weights,
    )
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    if learning_rate is not None:
        _globals.learning_rate = learning_rate

    main(config=config, experiment_name=experiment_name,
         foundation_architecture=foundation_architecture,
         foundation_weights=foundation_weights, finetuning_technique=finetuning, from_checkpoint=from_checkpoint,
         resolution=resolution, max_steps=max_steps, train=train, starting_weights=starting_weights,
         attention_backend=attention_backend, checkpoint_every_n_epochs=checkpoint_every_n_epochs,
         encoder_training_mode=encoder_training_mode,
         validation_every_n_batches=validation_every_n_batches,
         protocol_version=protocol_version,
         source_curriculum_step=source_curriculum_step)


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
