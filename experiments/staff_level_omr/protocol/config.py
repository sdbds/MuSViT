"""Validated, immutable configuration primitives for staff-level OMR v2."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ProtocolError


_UNSET = object()
_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return value


def _readable_directory(name: str, value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"{name} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ProtocolError(f"{name} must be a directory: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise ProtocolError(f"{name} is not readable: {resolved}")
    return resolved


def _readable_file(name: str, value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"{name} does not exist: {path}") from exc
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise ProtocolError(f"{name} must be a readable file: {resolved}")
    return resolved


def _creatable_directory(name: str, value: str | Path) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.exists():
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise ProtocolError(f"{name} must be a writable directory: {path}")
        return path

    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise ProtocolError(
            f"{name} cannot be created below non-writable path {ancestor}"
        )
    return path


def _normalize_experiment_name(value: Any) -> str:
    if not isinstance(value, str) or not _EXPERIMENT_NAME.fullmatch(value):
        raise ProtocolError(
            "experiment_name must contain 1-64 ASCII letters, digits, '.', "
            "'_', or '-'"
        )
    if value in {".", ".."}:
        raise ProtocolError("experiment_name cannot be '.' or '..'")
    device_stem = value.split(".", 1)[0].upper()
    if device_stem in _WINDOWS_DEVICE_NAMES:
        raise ProtocolError(
            f"experiment_name cannot use Windows device name {device_stem!r}"
        )
    return value


def _normalize_revision(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{field_name} must be an immutable commit SHA")
    normalized = value.lower()
    if not _COMMIT_SHA.fullmatch(normalized):
        raise ProtocolError(
            f"{field_name} must be exactly 40 hexadecimal characters"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class StaffOMRConfig:
    """Fully normalized user configuration, before model/data preflight."""

    experiment_name: str
    data_path: Path
    dataset_bundle_path: Path
    model_name: str
    model_revision: str
    method: str
    input_geometry: str
    patch_rows: int
    patch_cols: int
    augmentation_profile: str
    train_infeasible_policy: str
    train_exclusions_path: Path | None
    batch_size: int
    num_workers: int
    learning_rate: float
    max_epochs: int
    start_eval: int
    patience: int
    seed: int
    output_root: Path
    device: str
    verify_image_hashes: str

    @classmethod
    def create(
        cls,
        *,
        experiment_name: str,
        data_path: str | Path,
        dataset_bundle_path: str | Path,
        approved_revisions: Mapping[str, Collection[str]],
        default_revisions: Mapping[str, str],
        model_name: str = "musvit",
        model_revision: str | None = None,
        resolve_model_revision: bool = False,
        revision_resolver: Callable[[str], str] | None = None,
        method: str = "lora",
        patch_rows: int | None = None,
        patch_cols: int | None = None,
        shape_patches: Sequence[int] | None = None,
        input_geometry: object = _UNSET,
        augmentation_profile: str = "staff_omr_train_v1",
        train_infeasible_policy: str = "fail",
        train_exclusions_path: str | Path | None = None,
        batch_size: int = 8,
        num_workers: int = 6,
        learning_rate: float = 3e-4,
        max_epochs: int = 1000,
        start_eval: int = 20,
        patience: int = 30,
        seed: int = 7,
        output_root: str | Path = Path(
            "experiments/staff_level_omr/runs"
        ),
        device: str = "cuda",
        verify_image_hashes: str = "always",
    ) -> "StaffOMRConfig":
        if input_geometry is not _UNSET:
            raise ProtocolError(
                "input_geometry is derived from method and cannot be supplied"
            )

        normalized_method = {
            "linear_prob": "linear_probe",
            "linear_probe": "linear_probe",
            "lora": "lora",
        }.get(method)
        if normalized_method is None:
            raise ProtocolError(
                "method must be 'linear_probe', legacy 'linear_prob', or 'lora'"
            )
        derived_geometry = (
            "native_pad" if normalized_method == "linear_probe" else "exact_grid"
        )

        if shape_patches is not None:
            if patch_rows is not None or patch_cols is not None:
                raise ProtocolError(
                    "shape_patches cannot be combined with patch_rows or patch_cols"
                )
            if (
                isinstance(shape_patches, (str, bytes))
                or len(shape_patches) != 2
            ):
                raise ProtocolError(
                    "shape_patches must contain exactly two integers"
                )
            patch_rows, patch_cols = shape_patches
        rows = _positive_int(
            "patch_rows", 8 if patch_rows is None else patch_rows
        )
        cols = _positive_int(
            "patch_cols", 64 if patch_cols is None else patch_cols
        )

        if not isinstance(model_name, str) or model_name not in approved_revisions:
            raise ProtocolError(
                f"model_name {model_name!r} has no approved revision registry"
            )
        if model_revision is not None and resolve_model_revision:
            raise ProtocolError(
                "model_revision and resolve_model_revision are mutually exclusive"
            )
        if resolve_model_revision:
            if revision_resolver is None:
                raise ProtocolError(
                    "resolve_model_revision requires a revision_resolver"
                )
            selected_revision = revision_resolver(model_name)
        elif model_revision is not None:
            selected_revision = model_revision
        else:
            if model_name not in default_revisions:
                raise ProtocolError(
                    f"model_name {model_name!r} has no default approved revision"
                )
            selected_revision = default_revisions[model_name]

        revision = _normalize_revision(selected_revision, "model_revision")
        approved = {
            _normalize_revision(item, "approved revision")
            for item in approved_revisions[model_name]
        }
        if revision not in approved:
            raise ProtocolError(
                f"model_revision {revision!r} is not approved for {model_name!r}"
            )

        if augmentation_profile not in {"staff_omr_train_v1", "none"}:
            raise ProtocolError(
                "augmentation_profile must be 'staff_omr_train_v1' or 'none'"
            )
        if train_infeasible_policy not in {"fail", "exclude_listed"}:
            raise ProtocolError(
                "train_infeasible_policy must be 'fail' or 'exclude_listed'"
            )
        if train_infeasible_policy == "exclude_listed":
            if train_exclusions_path is None:
                raise ProtocolError(
                    "train_exclusions_path is required for exclude_listed"
                )
            exclusions_path = _readable_file(
                "train_exclusions_path", train_exclusions_path
            )
        else:
            if train_exclusions_path is not None:
                raise ProtocolError(
                    "train_exclusions_path must be empty when policy is fail"
                )
            exclusions_path = None

        normalized_batch_size = _positive_int("batch_size", batch_size)
        normalized_workers = _non_negative_int("num_workers", num_workers)
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or learning_rate <= 0
        ):
            raise ProtocolError("learning_rate must be a finite positive number")
        normalized_max_epochs = _positive_int("max_epochs", max_epochs)
        normalized_start_eval = _positive_int("start_eval", start_eval)
        if normalized_start_eval > normalized_max_epochs:
            raise ProtocolError("start_eval must be <= max_epochs")
        normalized_patience = _positive_int("patience", patience)
        normalized_seed = _non_negative_int("seed", seed)

        if device not in {"cpu", "cuda", "auto"}:
            raise ProtocolError("device must be 'cpu', 'cuda', or 'auto'")
        if verify_image_hashes not in {"always", "cached"}:
            raise ProtocolError(
                "verify_image_hashes must be 'always' or 'cached'"
            )

        return cls(
            experiment_name=_normalize_experiment_name(experiment_name),
            data_path=_readable_directory("data_path", data_path),
            dataset_bundle_path=_readable_directory(
                "dataset_bundle_path", dataset_bundle_path
            ),
            model_name=model_name,
            model_revision=revision,
            method=normalized_method,
            input_geometry=derived_geometry,
            patch_rows=rows,
            patch_cols=cols,
            augmentation_profile=augmentation_profile,
            train_infeasible_policy=train_infeasible_policy,
            train_exclusions_path=exclusions_path,
            batch_size=normalized_batch_size,
            num_workers=normalized_workers,
            learning_rate=float(learning_rate),
            max_epochs=normalized_max_epochs,
            start_eval=normalized_start_eval,
            patience=normalized_patience,
            seed=normalized_seed,
            output_root=_creatable_directory("output_root", output_root),
            device=device,
            verify_image_hashes=verify_image_hashes,
        )

    def to_launch_config(self) -> dict[str, object]:
        """Return the normalized, JSON-native launch configuration."""
        return {
            "augmentation_profile": self.augmentation_profile,
            "batch_size": self.batch_size,
            "data_path": str(self.data_path),
            "dataset_bundle_path": str(self.dataset_bundle_path),
            "device": self.device,
            "experiment_name": self.experiment_name,
            "input_geometry": self.input_geometry,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "method": self.method,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "num_workers": self.num_workers,
            "output_root": str(self.output_root),
            "patch_cols": self.patch_cols,
            "patch_rows": self.patch_rows,
            "patience": self.patience,
            "seed": self.seed,
            "start_eval": self.start_eval,
            "train_exclusions_path": (
                str(self.train_exclusions_path)
                if self.train_exclusions_path is not None
                else None
            ),
            "train_infeasible_policy": self.train_infeasible_policy,
            "verify_image_hashes": self.verify_image_hashes,
        }
