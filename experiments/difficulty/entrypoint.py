"""Public entrypoints for the score-difficulty experiment (3 stages).

- ``prepare_images``: rasterize score PDFs into per-page PNGs.
- ``embeddings``:     extract frozen-MuSViT page embeddings (one ``.npy`` per page).
- ``run``:            train/test the difficulty classifier head (RNN/Transformer/MLP).

Called in-process by the ``musvit`` launcher (``musvit difficulty {prepare-images,
embeddings, run}``). Every stage ``chdir``-s into this folder, so ``PDFdifficulty/``,
``difficulty_embeddings/``, ``weights/`` and ``output/`` are created here. Stages
``embeddings`` and ``run`` must receive the *same* ``--architecture`` value so
stage 3 finds the embeddings written by stage 2. Stages 2-3 require a CUDA device.
"""

import argparse
from contextlib import chdir
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def prepare_images(dataset_name: str, fold_idx: str = "0",
                   dpi: int = 300, workers: int = 8):
    """Stage 1: rasterize score PDFs into per-page PNGs.

    Reads ``PDFdifficulty/<dataset_name>/{splits.json,pdf/}`` and writes the
    per-page images to ``PDFdifficulty/<dataset_name>/images_per_score/``.

    Args:
        dataset_name: one of "cipi", "fs", "ps".
        fold_idx: fold whose splits are read (every fold covers the whole dataset).
        dpi: rasterization resolution.
        workers: number of PDFs converted in parallel.
    """
    from .scripts.prepare_images import main

    args = argparse.Namespace(
        dataset_name=dataset_name,
        fold_idx=str(fold_idx),
        dpi=dpi,
        workers=workers,
    )
    with chdir(PACKAGE_DIR):
        main(args)


def embeddings(model_name: str, dataset_name: str, architecture: str):
    """Stage 2: extract frozen-MuSViT page embeddings (one ``.npy`` per page).

    Args:
        model_name: MuSViT checkpoint ("PRAIG/musvit" or "PRAIG/musvit-light").
        dataset_name: one of "cipi", "fs", "ps".
        architecture: rnn|transformer|mlp. Selects the output subfolder and must
            match the ``--architecture`` given to stage 3 (``run``).
    """
    from .scripts.get_embeddings import main

    args = argparse.Namespace(
        model_name=model_name,
        dataset_name=dataset_name,
        architecture_name=architecture,
    )
    with chdir(PACKAGE_DIR):
        main(args)


def run(model_name: str, dataset_name: str, architecture: str,
        fold_idx: str = "0", epochs: int = 10, no_log: bool = False,
        test: bool = False, timestamp: str | None = None):
    """Stage 3: train and test the difficulty classifier head.

    Args:
        model_name: MuSViT checkpoint used for the embeddings in stage 2.
        dataset_name: one of "cipi", "fs", "ps".
        architecture: rnn|transformer|mlp; must match stage 2's ``--architecture``.
        fold_idx: split fold to use.
        epochs: number of training epochs.
        no_log: disable Weights & Biases logging.
        test: only evaluate (skip training), loading the best checkpoint.
        timestamp: W&B run-group timestamp (defaults to the current time).
    """
    from .scripts.run_model import main

    args = argparse.Namespace(
        model_name=model_name,
        dataset_name=dataset_name,
        architecture_type=architecture,
        fold_idx=str(fold_idx),
        epochs=epochs,
        no_log=no_log,
        test=test,
        timestamp=timestamp or datetime.now().isoformat("-", "seconds"),
    )
    with chdir(PACKAGE_DIR):
        main(args)
