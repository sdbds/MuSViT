"""Root entrypoint for the MuSViT monorepo CLI (``musvit``).

Dispatches each subcommand to the in-process entrypoint of the corresponding
experiment under ``experiments/``. Powered by Fire, so CLI arguments map
directly onto each entrypoint's parameters and ``--help`` is generated
automatically.

A single ``EXPERIMENTS`` registry is the source of truth for both ``musvit
list`` (static metadata, no heavy imports) and command dispatch. Each entry
carries a *lazy* ``loader`` that imports only its own experiment, so invoking
one experiment never pulls in another's dependencies.
"""

import sys
from dataclasses import dataclass
from typing import Any, Callable

from . import env


@dataclass(frozen=True)
class Experiment:
    """One CLI command backed by an experiment entrypoint.

    ``loader`` is called lazily (only when the command runs) and returns the
    Fire target: a function for a flat command, or a dict for subcommands.
    ``help_rows`` are ``(usage, description)`` pairs shown by ``musvit list``
    without importing torch/transformers.
    """

    command: str
    location: str
    loader: Callable[[], Any]
    help_rows: tuple[tuple[str, str], ...]
    default_subcommand: str | None = None


def _load_full_page_omr():
    from experiments.full_page_omr.entrypoint import run
    return run


def _load_embeddings():
    from experiments.embeddings_test.entrypoint import run, sweep
    return {"run": run, "sweep": sweep}


def _load_staff_level_omr():
    from experiments.staff_level_omr.entrypoint import run
    from experiments.staff_level_omr.prepare_data import prepare_data

    return {"train": run, "prepare-data": prepare_data}


def _load_object_detection():
    from experiments.object_detection.entrypoint import run
    return run


def _load_difficulty():
    from experiments.difficulty.entrypoint import embeddings, prepare_images, run
    return {"prepare-images": prepare_images, "embeddings": embeddings, "run": run}


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "full-page-omr", "experiments/full_page_omr", _load_full_page_omr,
        (("full-page-omr <config_path> <experiment_name>",
          "Fine-tune MuSViT for full-page OMR."),),
    ),
    Experiment(
        "embeddings", "experiments/embeddings_test", _load_embeddings,
        (("embeddings run --model <id>",
          "Embedding vs transcription distance analysis (single MuSViT encoder)."),
         ("embeddings sweep --models_set <light|default|full>",
          "Same analysis over a preset list of MuSViT variants.")),
    ),
    Experiment(
        "staff-level-omr", "experiments/staff_level_omr", _load_staff_level_omr,
        (("staff-level-omr prepare-data --data_path <dir> --dataset_id <id>",
          "Build a deterministic staff-level dataset bundle."),
         ("staff-level-omr train --ds_name <name> --model_name <musvit|musvit_light>",
          "Run the current staff-level trainer (explicit train form).")),
        default_subcommand="train",
    ),
    Experiment(
        "object-detection", "experiments/object_detection", _load_object_detection,
        (("object-detection --train_images <dir> --train_ann <coco.json>",
          "Fine-tune MuSViT + Faster R-CNN for object detection (full / frozen / LoRA)."),),
    ),
    Experiment(
        "difficulty", "experiments/difficulty", _load_difficulty,
        (("difficulty prepare-images --dataset_name <cipi|fs|ps>",
          "Stage 1: rasterize score PDFs into per-page PNGs."),
         ("difficulty embeddings --model_name <id> --dataset_name <..> --architecture <..>",
          "Stage 2: extract frozen-MuSViT page embeddings."),
         ("difficulty run --model_name <id> --dataset_name <..> --architecture <..>",
          "Stage 3: train/test the difficulty classifier head.")),
    ),
)

_REGISTRY = {exp.command: exp for exp in EXPERIMENTS}


def _select_fire_target(
    experiment: Experiment,
    target: Any,
    arguments: list[str],
) -> tuple[Any, list[str]]:
    """Apply an omitted-subcommand compatibility alias when configured."""
    default = experiment.default_subcommand
    if default is None:
        return target, arguments
    if not isinstance(target, dict) or default not in target:
        raise RuntimeError(
            f"experiment {experiment.command!r} declares missing default "
            f"subcommand {default!r}"
        )
    if not arguments or arguments[0].startswith("-"):
        return target[default], arguments
    return target, arguments


def list_experiments():
    """List the experiments runnable from this CLI (no heavy imports)."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="MuSViT experiments")
        table.add_column("command", style="bold cyan", no_wrap=True)
        table.add_column("location", style="dim")
        table.add_column("description")
        for exp in EXPERIMENTS:
            for usage, desc in exp.help_rows:
                table.add_row(f"musvit {usage}", exp.location, desc)
        Console().print(table)
    except Exception:
        # Fallback if rich is unavailable for any reason.
        print("MuSViT experiments:")
        for exp in EXPERIMENTS:
            for usage, desc in exp.help_rows:
                print(f"  musvit {usage}\n      {desc}  ({exp.location})")


def main():
    """Console-script entrypoint (``musvit``)."""
    env.setup()

    argv = sys.argv[1:]
    # Fast path: discovery/help must not pay the cost of importing the heavy
    # experiment modules (torch, transformers, ...).
    if not argv or argv[0] in ("list", "help", "-h", "--help"):
        return list_experiments()

    command = argv[0]
    if command not in _REGISTRY:
        print(f"Unknown command: {command!r}\n", file=sys.stderr)
        list_experiments()
        raise SystemExit(2)

    from fire import Fire

    # Import ONLY the requested experiment, then let Fire parse the rest.
    experiment = _REGISTRY[command]
    target = experiment.loader()
    target, fire_arguments = _select_fire_target(
        experiment, target, argv[1:]
    )
    Fire(target, command=fire_arguments, name=f"musvit {command}")


if __name__ == "__main__":
    main()
