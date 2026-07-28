"""Fire-friendly public entrypoints for staff-level OMR protocol v2."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .protocol.backbone import (
    ProductionBackboneProvider,
    approved_revisions,
    default_revisions,
)
from .protocol.config import StaffOMRConfig
from .protocol.runtime import resume as resume_training
from .protocol.runtime import train as run_training


def build_train_config(
    *,
    experiment_name: str,
    data_path: str | Path,
    dataset_bundle_path: str | Path,
    model_name: str = "musvit",
    model_revision: str | None = None,
    resolve_model_revision: bool = False,
    method: str = "lora",
    patch_rows: int | None = None,
    patch_cols: int | None = None,
    shape_patches: Sequence[int] | None = None,
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
    output_root: str | Path = Path("experiments/staff_level_omr/runs"),
    device: str = "cuda",
    verify_image_hashes: str = "always",
) -> StaffOMRConfig:
    """Validate and normalize a public launch into the protocol config."""
    resolver = None
    if resolve_model_revision:
        resolver = ProductionBackboneProvider().resolve_revision
    return StaffOMRConfig.create(
        experiment_name=experiment_name,
        data_path=data_path,
        dataset_bundle_path=dataset_bundle_path,
        approved_revisions=approved_revisions(),
        default_revisions=default_revisions(),
        model_name=model_name,
        model_revision=model_revision,
        resolve_model_revision=resolve_model_revision,
        revision_resolver=resolver,
        method=method,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        shape_patches=shape_patches,
        augmentation_profile=augmentation_profile,
        train_infeasible_policy=train_infeasible_policy,
        train_exclusions_path=train_exclusions_path,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        start_eval=start_eval,
        patience=patience,
        seed=seed,
        output_root=output_root,
        device=device,
        verify_image_hashes=verify_image_hashes,
    )


def train(
    experiment_name: str,
    data_path: str,
    dataset_bundle_path: str,
    model_name: str = "musvit",
    model_revision: str | None = None,
    resolve_model_revision: bool = False,
    method: str = "lora",
    patch_rows: int | None = None,
    patch_cols: int | None = None,
    shape_patches: Sequence[int] | None = None,
    augmentation_profile: str = "staff_omr_train_v1",
    train_infeasible_policy: str = "fail",
    train_exclusions_path: str | None = None,
    batch_size: int = 8,
    num_workers: int = 6,
    learning_rate: float = 3e-4,
    max_epochs: int = 1000,
    start_eval: int = 20,
    patience: int = 30,
    seed: int = 7,
    output_root: str = "experiments/staff_level_omr/runs",
    device: str = "cuda",
    verify_image_hashes: str = "always",
) -> str:
    """Start a new trusted staff-level OMR v2 run.

    Args:
        experiment_name: Stable run label; 1-64 safe ASCII characters.
        data_path: Directory containing the image/target files in the bundle.
        dataset_bundle_path: Directory produced by ``prepare-data``.
        model_name: Approved ``musvit`` or ``musvit_light`` alias.
        model_revision: Optional approved immutable commit SHA.
        resolve_model_revision: Resolve the current Hub ref, then require it
            to be present in the approved registry.
        method: ``linear_probe``, compatibility alias ``linear_prob``, or
            ``lora``.
        patch_rows: Canonical vertical patch count.
        patch_cols: Canonical CTC time-axis patch count.
        shape_patches: Compatibility JSON array such as ``[8,64]``; cannot be
            combined with canonical patch arguments.
        augmentation_profile: ``staff_omr_train_v1`` or ``none``.
        train_infeasible_policy: ``fail`` or ``exclude_listed``.
        train_exclusions_path: Reviewed candidate JSON for ``exclude_listed``.
        batch_size: Positive training and evaluation batch size.
        num_workers: Non-negative DataLoader worker count.
        learning_rate: Finite positive Adam learning rate.
        max_epochs: Positive epoch budget.
        start_eval: First validation epoch, from 1 through ``max_epochs``.
        patience: Positive count of non-improving validations before stopping.
        seed: Non-negative protocol seed.
        output_root: Parent directory for unique run directories.
        device: ``cpu``, ``cuda``, or ``auto``.
        verify_image_hashes: ``always`` or ``cached``.
    """
    config = build_train_config(
        experiment_name=experiment_name,
        data_path=data_path,
        dataset_bundle_path=dataset_bundle_path,
        model_name=model_name,
        model_revision=model_revision,
        resolve_model_revision=resolve_model_revision,
        method=method,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        shape_patches=shape_patches,
        augmentation_profile=augmentation_profile,
        train_infeasible_policy=train_infeasible_policy,
        train_exclusions_path=train_exclusions_path,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        start_eval=start_eval,
        patience=patience,
        seed=seed,
        output_root=output_root,
        device=device,
        verify_image_hashes=verify_image_hashes,
    )
    return str(run_training(config))


def resume(
    run_dir: str,
    max_epochs: int | None = None,
    num_workers: int | None = None,
    data_path: str | None = None,
    device: str | None = None,
    verify_image_hashes: str | None = None,
    allow_env_drift: bool = False,
) -> str:
    """Resume the transaction committed in one run's ``last.pt``.

    Args:
        run_dir: Existing v2 run directory.
        max_epochs: Unchanged or increased epoch budget.
        num_workers: Optional non-negative operational worker count.
        data_path: Optional relocated directory with identical source content.
        device: Optional ``cpu``, ``cuda``, or ``auto`` override.
        verify_image_hashes: Optional ``always`` or ``cached`` override.
        allow_env_drift: Explicitly accept recorded soft environment drift.
    """
    return str(
        resume_training(
            run_dir,
            max_epochs=max_epochs,
            num_workers=num_workers,
            data_path=data_path,
            device=device,
            verify_image_hashes=verify_image_hashes,
            allow_env_drift=allow_env_drift,
        )
    )


# One release-cycle compatibility for ``musvit staff-level-omr --...``.
run = train
