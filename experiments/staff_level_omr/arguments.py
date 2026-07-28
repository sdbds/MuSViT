"""Argparse adapters for direct module use of staff-level OMR v2."""

from __future__ import annotations

import argparse
import json

from .entrypoint import build_train_config
from .protocol.config import StaffOMRConfig


def _parse_shape_patches(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "shape_patches must be a JSON array such as [8,64]"
        ) from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in parsed)
    ):
        raise argparse.ArgumentTypeError(
            "shape_patches must be a JSON array of two integers"
        )
    return parsed


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a trusted staff-level OMR v2 training run"
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_bundle_path", required=True)
    parser.add_argument(
        "--model_name",
        choices=("musvit", "musvit_light"),
        default="musvit",
    )
    parser.add_argument("--model_revision")
    parser.add_argument("--resolve_model_revision", action="store_true")
    parser.add_argument(
        "--method",
        choices=("linear_probe", "linear_prob", "lora"),
        default="lora",
    )
    parser.add_argument("--patch_rows", type=int)
    parser.add_argument("--patch_cols", type=int)
    parser.add_argument("--shape_patches", type=_parse_shape_patches)
    parser.add_argument(
        "--augmentation_profile",
        choices=("staff_omr_train_v1", "none"),
        default="staff_omr_train_v1",
    )
    parser.add_argument(
        "--train_infeasible_policy",
        choices=("fail", "exclude_listed"),
        default="fail",
    )
    parser.add_argument("--train_exclusions_path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--start_eval", type=int, default=20)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output_root",
        default="experiments/staff_level_omr/runs",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cuda",
    )
    parser.add_argument(
        "--verify_image_hashes",
        choices=("always", "cached"),
        default="always",
    )
    return parser


def _build_resume_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume a trusted staff-level OMR v2 run"
    )
    parser.add_argument("run_dir")
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--data_path")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument(
        "--verify_image_hashes",
        choices=("always", "cached"),
    )
    parser.add_argument("--allow_env_drift", action="store_true")
    return parser


parser_train = _build_train_parser()
parser_resume = _build_resume_parser()


def config_from_namespace(namespace: argparse.Namespace) -> StaffOMRConfig:
    """Build the same normalized config used by the Fire entrypoint."""
    return build_train_config(**vars(namespace))
