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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


DATASETS_TYPE = {
    "CL": CLFinetuningDataset,
    "SR": SynthRealFinetuningDataset,
    "CL1": SyntheticGrandStaffDataset,
    "R": None
}


def _validate_checkpoint_every_n_epochs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("checkpoint_every_n_epochs must be a positive integer")
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
         max_steps: int = -1, train: bool = True, starting_weights: str | None = None,
         attention_backend: str = "auto", checkpoint_every_n_epochs: int = 100,
         encoder_training_mode: str = "fine_tune"):
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
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

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=3, mode="min", verbose=True)

    print(f"Checkpoints will be saved to \"{experiment_name}_{finetuning_technique}\"")
    epoch_checkpointer = _build_epoch_checkpointer(
        experiment_name,
        finetuning_technique,
        checkpoint_every_n_epochs,
    )
    checkpointer = ModelCheckpoint(dirpath="weights/", filename=f"{experiment_name}_{finetuning_technique}",
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    wandb_logger = WandbLogger(project='Foundation_SMT',
                               name=f"{experiment_name}",
                               log_model=False, save_dir="wandb_logs/")

    trainer = Trainer(max_epochs=100000, max_steps=max_steps,
                      check_val_every_n_epoch=3500,
                      num_sanity_val_steps=0,
                      callbacks=[epoch_checkpointer, checkpointer, early_stopping], logger=wandb_logger,
                      precision='16-mixed')

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
           max_steps: int = -1, train: bool = True, starting_weights: str | None = None,
           learning_rate: float | None = None, attention_backend: str = "auto",
           checkpoint_every_n_epochs: int = 100,
           encoder_training_mode: str = "fine_tune"):
    checkpoint_every_n_epochs = _validate_checkpoint_every_n_epochs(checkpoint_every_n_epochs)
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
         encoder_training_mode=encoder_training_mode)


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
