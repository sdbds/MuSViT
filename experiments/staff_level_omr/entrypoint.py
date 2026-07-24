"""Public entrypoint for the staff-level OMR experiment.

Wraps ``train.py``'s training loop in a Fire-friendly ``run(...)`` so the
experiment is launchable as ``musvit staff-level-omr``. Checkpoints
(``<model>_<ds>_<method>_<cols>_ctc.pt``) are written inside this folder.

Requires a CUDA device (the training loop calls ``.cuda()`` unconditionally)
and that ``config.data_paths`` points at your local dataset folders.
"""

import argparse
from contextlib import chdir
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def run(ds_name: str = "catedrales", model_name: str = "musvit",
        method: str = "lora", shape_patches: tuple[int, int] = (8, 64),
        batch_size: int = 8, start_eval: int = 20, lr: float = 3e-4):
    """Train a staff-level OMR model (MuSViT backbone + BiLSTM/CTC head).

    Args:
        ds_name: dataset key from ``config.data_paths``
            (capitan, catedrales, fmt, guatemala, seils).
        model_name: backbone key from ``config.data_models``
            ("musvit" or "musvit_light").
        method: "linear_prob" (freeze the backbone, train only the head) or
            "lora" (inject LoRA adapters into the attention q/k/v projections).
        shape_patches: (rows, cols) patch grid; ``cols`` is the CTC time axis.
            Pass it on the CLI as ``--shape_patches '[8,128]'``.
        batch_size: mini-batch size for every dataloader.
        start_eval: first epoch at which validation / checkpointing begins.
        lr: Adam learning rate.
    """
    from .train import train

    args = argparse.Namespace(
        ds_name=ds_name,
        model_name=model_name,
        method=method,
        shape_patches=list(shape_patches),
        batch_size=batch_size,
        start_eval=start_eval,
        lr=lr,
    )
    with chdir(PACKAGE_DIR):
        train(args)
