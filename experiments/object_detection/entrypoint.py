"""Public entrypoint for the object-detection experiment.

Wraps the Faster R-CNN training driver in a Fire-friendly ``run(...)`` so the
experiment is launchable as ``musvit object-detection``. Checkpoints
(``ckpt-best.pt`` / ``ckpt-latest.pt``) are written inside ``--out_dir``,
anchored in this folder. Requires a CUDA device and a COCO-format dataset.
"""

import argparse
from contextlib import chdir
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def run(train_images: str, train_ann: str, model: str = "musvit_base",
        val_images: str = None, val_ann: str = None,
        vit_base_model_path: str = "PRAIG/musvit",
        vit_small_model_path: str = "PRAIG/musvit-light",
        hf_token: str = None, finetuning: str = "lora",
        epochs: int = 50, batch_size: int = 2,
        lr_backbone: float = 5e-5, lr_head: float = 2.5e-4,
        weight_decay: float = 1e-2, fixed_size: int = 1024,
        target_tokens: int = 4096, out_channels: int = 256,
        num_workers: int = 4, out_dir: str = "weights",
        from_checkpoint: str = None, use_wandb: bool = False,
        experiment_name: str = None, device: int = 0):
    """Fine-tune a MuSViT-backbone Faster R-CNN on a COCO-format dataset.

    Args:
        train_images: directory holding the training images.
        train_ann: COCO-format JSON with the training annotations.
        model: "musvit_base" or "musvit_small" (selects the backbone checkpoint).
        val_images / val_ann: optional validation split; enables best-checkpointing.
        vit_base_model_path / vit_small_model_path: MuSViT Hub ids per model size.
        hf_token: Hugging Face token; ``None`` uses the ambient login / ``HF_TOKEN``.
        finetuning: "full", "frozen" (freeze the ViT) or "lora" (adapters on q/k/v).
        epochs, batch_size, lr_backbone, lr_head, weight_decay: optimisation knobs.
        fixed_size, target_tokens: enforce MuSViT's square input / token grid.
        out_channels: FPN channel width.
        out_dir: checkpoint directory (relative to this experiment folder).
        from_checkpoint: resume from a ``ckpt-*.pt`` file.
        use_wandb, experiment_name: optional Weights & Biases logging.
        device: CUDA device index.
    """
    from .train import train

    args = argparse.Namespace(
        model=model, train_images=train_images, train_ann=train_ann,
        val_images=val_images, val_ann=val_ann,
        vit_base_model_path=vit_base_model_path,
        vit_small_model_path=vit_small_model_path, hf_token=hf_token,
        finetuning=finetuning, epochs=epochs, batch_size=batch_size,
        lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay,
        fixed_size=fixed_size, target_tokens=target_tokens,
        out_channels=out_channels, num_workers=num_workers, out_dir=out_dir,
        from_checkpoint=from_checkpoint, use_wandb=use_wandb,
        experiment_name=experiment_name, device=device,
    )
    with chdir(PACKAGE_DIR):
        train(args)
