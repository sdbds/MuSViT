import json

from fire import Fire
from loguru import logger

from . import _globals
from .config.ExperimentConfigWrapper import ExperimentConfig, experiment_config_from_dict
from .data import SyntheticGrandStaffDataset, CLFinetuningDataset, SynthRealFinetuningDataset
from .smt_foundation import SMTFoundationConfig, SMTFoundationModelForCausalLM
from .smt_trainer import ENCODER_TRAINING_MODES, SMTPP_Trainer
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
    }


def _build_protocol_metadata(*, max_steps, validation_every_n_batches,
                             from_checkpoint, starting_weights,
                             encoder_training_mode, encoder_unfreeze_step,
                             resolution, reduce_ratio, batch_size):
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
        "protocol_version": PROTOCOL_VERSION,
        "metric_version": METRIC_VERSION,
        "max_steps": max_steps,
        "validation_every_n_batches": validation_every_n_batches,
        "checkpoint_monitor": CHECKPOINT_MONITOR,
        "checkpoint_source": checkpoint_source,
        "checkpoint_load_mode": checkpoint_load_mode,
        "encoder_training_mode": encoder_training_mode,
        "encoder_unfreeze_step": encoder_unfreeze_step,
        "resolution": resolution,
        "reduce_ratio": reduce_ratio,
        "optimizer": "Adam",
        "learning_rate": _globals.learning_rate,
        "precision": PRECISION,
        "batch_size": batch_size,
        "accumulate_grad_batches": 1,
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
         validation_every_n_batches: int = 10000):
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
        )
    else:
        model_wrapper = SMTPP_Trainer.load_from_checkpoint(
            starting_weights,
            smt_config=smt_config,
            smt_model=model,
            encoder_training_mode=encoder_training_mode,
            encoder_unfreeze_step=encoder_unfreeze_step,
            curriculum_step_offset=curriculum_step_offset,
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
           validation_every_n_batches: int = 10000):
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
         validation_every_n_batches=validation_every_n_batches)


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
