"""Public entrypoint for the full-page OMR experiment.

The ``musvit`` launcher imports :func:`run` and calls it in-process. Output
artifacts (``weights/``, ``wandb_logs/``) are written inside this experiment
folder regardless of the directory ``musvit`` was invoked from.
"""

from contextlib import chdir
from pathlib import Path

from .finetune import launch as _launch

PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_config(config_path: str) -> str:
    """Accept a config path relative to the repo root, the cwd, or this package."""
    p = Path(config_path)
    if p.is_file():
        return str(p.resolve())
    alt = PACKAGE_DIR / config_path
    if alt.is_file():
        return str(alt.resolve())
    # Let the downstream open() raise a clear FileNotFoundError.
    return str(p.resolve())


def run(config_path: str, experiment_name: str,
        foundation_architecture: str = "ViTMAEBase",
        foundation_weights: str = "carlospm12/LSMT-MAE-Base-1024-16",
        finetuning: str = "CL", from_checkpoint: str | None = None,
        resolution: int | None = None, max_steps: int = 4_000_000, train: bool = True,
        starting_weights: str | None = None,
        task_learning_rate: float | None = None,
        learning_rate: float | None = None,
        encoder_learning_rate: float = 1e-5, weight_decay: float = 0.01,
        wsd_warmup_steps: int = 10_000, wsd_decay_steps: int = 400_000,
        wsd_warmup_type: str = "linear", wsd_decay_type: str = "cosine",
        wsd_min_lr_ratio: float = 0.0,
        checkpoint_every_n_epochs: int = 100, attention_backend: str = "auto",
        encoder_training_mode: str = "fine_tune",
        validation_every_n_epochs: int = 2_000,
        protocol_version: str = "full_page_omr_adamw_wsd_4m_v1",
        source_curriculum_step: int | None = None,
        source_checkpoint_sha256: str | None = None):
    """Fine-tune MuSViT for full-page Optical Music Recognition.

    Args:
        config_path: JSON config (e.g. config/Polish_Scores/finetuning.json).
        experiment_name: Name used for checkpoints and the W&B run.
        finetuning: Finetuning regime: "CL", "SR", "CL1" or "R".
        See experiments/full_page_omr/finetune.py:launch for the rest.
    """
    config_path = _resolve_config(config_path)  # resolve before chdir
    with chdir(PACKAGE_DIR):
        _launch(
            config_path=config_path,
            experiment_name=experiment_name,
            foundation_architecture=foundation_architecture,
            foundation_weights=foundation_weights,
            finetuning=finetuning,
            from_checkpoint=from_checkpoint,
            resolution=resolution,
            max_steps=max_steps,
            train=train,
            starting_weights=starting_weights,
            task_learning_rate=task_learning_rate,
            learning_rate=learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            weight_decay=weight_decay,
            wsd_warmup_steps=wsd_warmup_steps,
            wsd_decay_steps=wsd_decay_steps,
            wsd_warmup_type=wsd_warmup_type,
            wsd_decay_type=wsd_decay_type,
            wsd_min_lr_ratio=wsd_min_lr_ratio,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            attention_backend=attention_backend,
            encoder_training_mode=encoder_training_mode,
            validation_every_n_epochs=validation_every_n_epochs,
            protocol_version=protocol_version,
            source_curriculum_step=source_curriculum_step,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
