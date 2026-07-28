"""End-to-end staff-level OMR v2 training and epoch-boundary resume."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import albumentations
import cv2
import numpy
import peft
import torch
import torchvision
import transformers

from .artifacts import RunArtifacts, file_sha256
from .augmentation import preflight_augmentation
from .backbone import (
    BackboneInspection,
    BackboneLoadResult,
    ProductionBackboneProvider,
)
from .canonical import canonical_sha256, read_json, write_canonical_json
from .checkpoint import (
    CheckpointStatic,
    build_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    validate_resume_artifact_identity,
    validate_resume_checkpoint,
)
from .config import StaffOMRConfig
from .contracts import PROTOCOL_VERSION, ProtocolContracts, build_protocol_contracts
from .ctc import (
    CTCPolicyResult,
    CTCPreflight,
    analyze_ctc_feasibility,
    apply_train_policy,
)
from .data_bundle import (
    RUN_BUNDLE_FILE,
    ValidatedDatasetBundle,
    load_dataset_bundle,
)
from .data_pipeline import (
    DatasetSplits,
    build_data_loaders,
    build_datasets,
)
from .errors import ProtocolError
from .evaluation import EvaluationResult, evaluate_split, train_epoch
from .geometry import (
    extract_spatial_grid,
    verify_transformers_position_interpolation,
)
from .modeling import (
    StaffOMRModel,
    build_model,
    load_trainable_state_dict,
)
from .optimization import build_optimizer
from .seeding import reset_epoch_rng, reset_initialization_rng


RUN_SCHEMA = "staff_omr_run_v2"
SUMMARY_SCHEMA = "staff_omr_summary_v2"
TEST_SCHEMA = "staff_omr_test_v2"
CORE_PACKAGES = (
    "python",
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "numpy",
    "albumentations",
    "opencv",
)
RESUME_MUTABLE_LAUNCH_FIELDS = frozenset(
    {
        "data_path",
        "dataset_bundle_path",
        "device",
        "max_epochs",
        "num_workers",
        "output_root",
        "train_exclusions_path",
        "verify_image_hashes",
    }
)


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if isinstance(name, str) and name:
            versions[name.lower().replace("_", "-")] = distribution.version
    versions.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "numpy": numpy.__version__,
            "albumentations": albumentations.__version__,
            "opencv": cv2.__version__,
        }
    )
    return dict(sorted(versions.items(), key=lambda item: item[0].encode("utf-8")))


def collect_git_metadata() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError("cannot collect git provenance") from exc
    return {
        "commit": commit,
        "dirty": bool(status),
    }


def collect_runtime_environment(device: torch.device) -> dict[str, object]:
    return {
        "device": str(device),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


@dataclass(slots=True)
class RuntimeDependencies:
    backbone_provider: Any
    now_factory: Callable[[], datetime]
    uuid_factory: Callable[[], str]
    monotonic: Callable[[], float]
    package_versions_factory: Callable[[], dict[str, str]]
    git_metadata_factory: Callable[[], dict[str, object]]
    runtime_environment_factory: Callable[[torch.device], dict[str, object]]

    @classmethod
    def production(cls) -> "RuntimeDependencies":
        return cls(
            backbone_provider=ProductionBackboneProvider(),
            now_factory=lambda: datetime.now(timezone.utc),
            uuid_factory=lambda: uuid4().hex,
            monotonic=time.perf_counter,
            package_versions_factory=collect_package_versions,
            git_metadata_factory=collect_git_metadata,
            runtime_environment_factory=collect_runtime_environment,
        )


@dataclass(slots=True)
class TrainingState:
    next_epoch: int = 1
    global_step: int = 0
    bad_epochs: int = 0
    evaluations: int = 0
    best_metric_value: float | None = None
    best_epoch: int | None = None
    stop_reason: str | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, object]) -> "TrainingState":
        early = checkpoint["early_stopping_state"]
        if not isinstance(early, dict):
            raise ProtocolError("checkpoint early_stopping_state is invalid")
        return cls(
            next_epoch=int(checkpoint["next_epoch"]),
            global_step=int(checkpoint["global_step"]),
            bad_epochs=int(early.get("bad_epochs", 0)),
            evaluations=int(early.get("evaluations", 0)),
            best_metric_value=checkpoint["best_metric_value"],
            best_epoch=checkpoint["best_epoch"],
            stop_reason=checkpoint["stop_reason"],
        )


@dataclass(slots=True)
class PreparedRuntime:
    config: StaffOMRConfig
    artifacts: RunArtifacts
    bundle: ValidatedDatasetBundle
    inspection: BackboneInspection
    contracts: ProtocolContracts
    ctc_preflight: CTCPreflight
    ctc_policy: CTCPolicyResult
    load_result: BackboneLoadResult
    model: StaffOMRModel
    optimizer: torch.optim.Optimizer
    optimizer_parameter_names: list[str]
    datasets: DatasetSplits
    checkpoint_static: CheckpointStatic
    device: torch.device
    runtime_environment: dict[str, object]
    package_versions: dict[str, str]
    reproducibility_status: str


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ProtocolError("runtime timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ProtocolError("device='cuda' requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ProtocolError(f"unsupported device {requested!r}")


def _sample_counts(bundle: ValidatedDatasetBundle) -> dict[str, int]:
    return {
        split: len(bundle.split_samples(split))
        for split in ("train", "val", "test")
    }


def _initial_run_document(
    *,
    config: StaffOMRConfig,
    bundle: ValidatedDatasetBundle,
    contracts: ProtocolContracts,
    inspection: BackboneInspection,
    package_versions: dict[str, str],
    git_metadata: dict[str, object],
    created_at: datetime,
) -> dict[str, object]:
    launch_event = {
        "kind": "initial",
        "timestamp": _utc_text(created_at),
        "launch_config": contracts.launch_config,
        "launch_config_sha256": contracts.launch_config_sha256,
    }
    return {
        "schema_version": RUN_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "status": "preflighting",
        "stage": "preflight",
        "created_at": _utc_text(created_at),
        "updated_at": _utc_text(created_at),
        "training_contract": contracts.training_contract,
        "training_contract_sha256": contracts.training_contract_sha256,
        "identity_hashes": contracts.identity_hashes,
        "initial_launch_config": contracts.launch_config,
        "initial_launch_config_sha256": contracts.launch_config_sha256,
        "current_launch_config": contracts.launch_config,
        "current_launch_config_sha256": contracts.launch_config_sha256,
        "launch_history": [launch_event],
        "resume_history": [],
        "dataset": {
            "dataset_id": bundle.dataset_id,
            "source_dataset_bundle_path": str(config.dataset_bundle_path),
            "dataset_bundle_sha256": bundle.dataset_bundle_sha256,
            "split_manifest_sha256": bundle.manifest_sha256,
            "vocabulary_sha256": bundle.vocabulary_sha256,
            "sample_counts": _sample_counts(bundle),
        },
        "image_verification": {
            "mode": bundle.image_verification.mode,
            "hits": bundle.image_verification.hits,
            "recomputed": bundle.image_verification.recomputed,
            "status": bundle.image_verification.status,
            "trusted_baseline": bundle.image_verification.trusted_baseline,
        },
        "base_model_registry_evidence": inspection.registry_evidence,
        "backbone_config": inspection.raw_config,
        "input_contract": contracts.input_contract,
        "augmentation_contract_sha256": (
            contracts.augmentation_contract_sha256
        ),
        "preflight": None,
        "committed_epoch": 0,
        "global_step": 0,
        "best_metric_name": "val_CER_all",
        "best_metric_value": None,
        "best_epoch": None,
        "stop_reason": None,
        "package_versions": package_versions,
        "git": git_metadata,
        "runtime_environment": None,
        "reproducibility_status": "exact",
        "failure": None,
    }


def _record_failure(
    artifacts: RunArtifacts,
    *,
    stage: str,
    error: BaseException,
    dependencies: RuntimeDependencies,
) -> None:
    try:
        artifacts.update_run(
            status="failed",
            stage=stage,
            updated_at=_utc_text(dependencies.now_factory()),
            failure={
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
        )
    except Exception:
        pass


def _verify_loaded_backbone(
    load_result: BackboneLoadResult,
    inspection: BackboneInspection,
    contracts: ProtocolContracts,
    device: torch.device,
) -> dict[str, object]:
    if load_result.metadata != inspection.metadata:
        raise ProtocolError("loaded backbone metadata differs from inspection")
    verification = load_result.weight_verification
    if not isinstance(verification, dict):
        raise ProtocolError("backbone loader omitted weight verification")
    if verification.get("sha256") != inspection.entry.weights.sha256:
        raise ProtocolError("loaded base weight SHA-256 differs from registry")
    if verification.get("size") != inspection.entry.weights.size:
        raise ProtocolError("loaded base weight size differs from registry")
    backbone = load_result.model.to(device)
    backbone.eval()
    plan = contracts.input_plan
    position_report: dict[str, object] | None = None
    if plan.interpolate_pos_encoding:
        embeddings = getattr(backbone, "embeddings", None)
        if embeddings is None:
            raise ProtocolError(
                "exact_grid backbone has no ViT embeddings for interpolation "
                "preflight"
            )
        position_report = verify_transformers_position_interpolation(
            embeddings,
            inspection.metadata,
            height=plan.input_height,
            width=plan.input_width,
        )
    with torch.no_grad():
        probe = torch.zeros(
            (
                1,
                inspection.metadata.num_channels,
                plan.input_height,
                plan.input_width,
            ),
            dtype=torch.float32,
            device=device,
        )
        output = backbone(
            pixel_values=probe,
            interpolate_pos_encoding=plan.interpolate_pos_encoding,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise ProtocolError(
                "backbone geometry probe has no last_hidden_state"
            )
        grid = extract_spatial_grid(hidden, plan)
    return {
        "status": "passed",
        "output_grid": [
            int(grid.shape[1]),
            int(grid.shape[2]),
            int(grid.shape[3]),
        ],
        "position_interpolation": position_report,
        "weight_verification": verification,
    }


def _checkpoint_static(
    *,
    initial_launch_config: dict[str, object],
    bundle: ValidatedDatasetBundle,
    inspection: BackboneInspection,
    contracts: ProtocolContracts,
    load_result: BackboneLoadResult,
    config: StaffOMRConfig,
    package_versions: dict[str, str],
) -> CheckpointStatic:
    evidence = {
        "registry": inspection.registry_evidence,
        "weight_verification": load_result.weight_verification,
    }
    exclusion_relpath = (
        "train_exclusions.json"
        if config.train_infeasible_policy == "exclude_listed"
        else None
    )
    return CheckpointStatic.create(
        training_contract=contracts.training_contract,
        initial_launch_config=initial_launch_config,
        dataset_bundle_sha256=bundle.dataset_bundle_sha256,
        split_manifest_sha256=bundle.manifest_sha256,
        vocabulary=bundle.vocabulary.to_document(),
        base_model_id=inspection.entry.model_id,
        base_model_revision=inspection.entry.revision,
        base_model_weights_filename=inspection.entry.weights.filename,
        base_model_weights_sha256=inspection.entry.weights.sha256,
        base_model_registry_evidence=evidence,
        backbone_config=inspection.raw_config,
        input_contract=contracts.input_contract,
        augmentation_contract_sha256=(
            contracts.augmentation_contract_sha256
        ),
        train_exclusions_relpath=exclusion_relpath,
        train_exclusions_sha256=contracts.exclusion_sha256,
        train_exclusions_count=contracts.exclusion_count,
        base_seed=config.seed,
        package_versions=package_versions,
    )


def _construct_trainables(
    *,
    config: StaffOMRConfig,
    bundle: ValidatedDatasetBundle,
    inspection: BackboneInspection,
    contracts: ProtocolContracts,
    ctc_preflight: CTCPreflight,
    ctc_policy: CTCPolicyResult,
    load_result: BackboneLoadResult,
    device: torch.device,
) -> tuple[
    StaffOMRModel,
    torch.optim.Optimizer,
    list[str],
    DatasetSplits,
]:
    model = build_model(
        load_result.model,
        inspection.metadata,
        config,
        bundle.vocabulary.num_classes,
    ).to(device)
    optimizer, optimizer_names = build_optimizer(
        model,
        config.learning_rate,
    )
    datasets = build_datasets(
        samples=bundle.samples,
        data_path=bundle.data_path,
        vocabulary=bundle.vocabulary,
        preflight=ctc_preflight,
        plan=contracts.input_plan,
        augmentation_profile=config.augmentation_profile,
        base_seed=config.seed,
        retained_train_sample_ids=ctc_policy.retained_train_sample_ids,
    )
    return model, optimizer, optimizer_names, datasets


def _prepare_after_preflight(
    *,
    config: StaffOMRConfig,
    artifacts: RunArtifacts,
    bundle: ValidatedDatasetBundle,
    inspection: BackboneInspection,
    contracts: ProtocolContracts,
    ctc_preflight: CTCPreflight,
    ctc_policy: CTCPolicyResult,
    package_versions: dict[str, str],
    dependencies: RuntimeDependencies,
    initial_launch_config: dict[str, object],
    reproducibility_status: str,
    resolved_device: torch.device | None = None,
    runtime_environment: dict[str, object] | None = None,
) -> PreparedRuntime:
    device = resolved_device or _resolve_device(config.device)
    environment = (
        runtime_environment
        if runtime_environment is not None
        else dependencies.runtime_environment_factory(device)
    )
    reset_initialization_rng(config.seed)
    load_result = dependencies.backbone_provider.load(inspection)
    model_preflight = _verify_loaded_backbone(
        load_result,
        inspection,
        contracts,
        device,
    )
    model, optimizer, optimizer_names, datasets = _construct_trainables(
        config=config,
        bundle=bundle,
        inspection=inspection,
        contracts=contracts,
        ctc_preflight=ctc_preflight,
        ctc_policy=ctc_policy,
        load_result=load_result,
        device=device,
    )
    static = _checkpoint_static(
        initial_launch_config=initial_launch_config,
        bundle=bundle,
        inspection=inspection,
        contracts=contracts,
        load_result=load_result,
        config=config,
        package_versions=package_versions,
    )
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    artifacts.update_run(
        status="running",
        stage="training",
        updated_at=_utc_text(dependencies.now_factory()),
        runtime_environment=environment,
        model_preflight=model_preflight,
        base_model_registry_evidence=static.base_model_registry_evidence,
        trainable_parameters={
            "count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "names": sorted(
                trainable_names,
                key=lambda value: value.encode("utf-8"),
            ),
        },
        reproducibility_status=reproducibility_status,
        failure=None,
    )
    return PreparedRuntime(
        config=config,
        artifacts=artifacts,
        bundle=bundle,
        inspection=inspection,
        contracts=contracts,
        ctc_preflight=ctc_preflight,
        ctc_policy=ctc_policy,
        load_result=load_result,
        model=model,
        optimizer=optimizer,
        optimizer_parameter_names=optimizer_names,
        datasets=datasets,
        checkpoint_static=static,
        device=device,
        runtime_environment=environment,
        package_versions=package_versions,
        reproducibility_status=reproducibility_status,
    )


def _validation_defaults() -> dict[str, object]:
    return {
        "val_CER_all": None,
        "val_CER_feasible": None,
        "val_CTC_loss_feasible": None,
        "val_feasible_samples": None,
        "val_infeasible_samples": None,
        "val_infeasible_ratio": None,
        "val_capacity": None,
    }


def _run_epochs(
    prepared: PreparedRuntime,
    state: TrainingState,
    dependencies: RuntimeDependencies,
) -> TrainingState:
    config = prepared.config
    if state.next_epoch > config.max_epochs:
        return state
    state.stop_reason = None
    for epoch in range(state.next_epoch, config.max_epochs + 1):
        reset_epoch_rng(config.seed, epoch)
        started = dependencies.monotonic()
        loaders = build_data_loaders(
            prepared.datasets,
            epoch=epoch,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        train_result = train_epoch(
            prepared.model,
            loaders.train,
            prepared.optimizer,
            device=prepared.device,
        )
        state.global_step += train_result.global_steps
        validation_performed = epoch >= config.start_eval
        validation: EvaluationResult | None = None
        best_updated = False
        if validation_performed:
            validation = evaluate_split(
                prepared.model,
                loaders.validation,
                split="val",
                device=prepared.device,
            )
            state.evaluations += 1
            metric = validation.metrics["val_CER_all"]
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise ProtocolError("val_CER_all must be finite")
            if (
                state.best_metric_value is None
                or float(metric) < state.best_metric_value
            ):
                state.best_metric_value = float(metric)
                state.best_epoch = epoch
                state.bad_epochs = 0
                best_updated = True
            else:
                state.bad_epochs += 1

        if validation_performed and state.bad_epochs >= config.patience:
            state.stop_reason = "early_stopping"
        elif epoch >= config.max_epochs:
            state.stop_reason = "max_epochs"
        else:
            state.stop_reason = None
        wall_time = dependencies.monotonic() - started
        if not math.isfinite(wall_time) or wall_time < 0:
            raise ProtocolError("epoch wall time must be finite and non-negative")
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "global_step": state.global_step,
            "train_samples": train_result.samples,
            "train_batches": train_result.batches,
            "train_loss": train_result.loss,
            "learning_rate": config.learning_rate,
            "validation_performed": validation_performed,
            **(
                validation.to_dict()
                if validation is not None
                else _validation_defaults()
            ),
            "best_checkpoint_updated": best_updated,
            "best_metric_name": "val_CER_all",
            "best_metric_value": state.best_metric_value,
            "best_epoch": state.best_epoch,
            "early_stopping_bad_epochs": state.bad_epochs,
            "early_stopping_evaluations": state.evaluations,
            "stop_reason": state.stop_reason,
            "epoch_wall_time_seconds": wall_time,
        }
        early_state = {
            "bad_epochs": state.bad_epochs,
            "evaluations": state.evaluations,
            "patience": config.patience,
        }
        last = build_checkpoint(
            static=prepared.checkpoint_static,
            run_id=prepared.artifacts.run_id,
            role="last",
            model=prepared.model,
            optimizer=prepared.optimizer,
            optimizer_parameter_names=prepared.optimizer_parameter_names,
            epoch=epoch,
            global_step=state.global_step,
            early_stopping_state=early_state,
            best_metric_name="val_CER_all",
            best_metric_value=state.best_metric_value,
            best_epoch=state.best_epoch,
            best_updated=best_updated,
            stop_reason=state.stop_reason,
            epoch_record=epoch_record,
        )
        save_checkpoint(prepared.artifacts.last_checkpoint, last)
        if best_updated:
            best = dict(last)
            best["checkpoint_role"] = "best"
            save_checkpoint(prepared.artifacts.best_checkpoint, best)
        prepared.artifacts.append_epoch_metrics(epoch_record)
        prepared.artifacts.update_run(
            status="running",
            stage="training",
            updated_at=_utc_text(dependencies.now_factory()),
            committed_epoch=epoch,
            global_step=state.global_step,
            best_metric_value=state.best_metric_value,
            best_epoch=state.best_epoch,
            stop_reason=state.stop_reason,
            early_stopping_state=early_state,
        )
        state.next_epoch = epoch + 1
        if state.stop_reason is not None:
            break
    return state


def _test_identity(
    prepared: PreparedRuntime,
    best_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": TEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "training_contract_sha256": (
            prepared.contracts.training_contract_sha256
        ),
        "split_manifest_sha256": prepared.bundle.manifest_sha256,
        "vocabulary_sha256": prepared.bundle.vocabulary_sha256,
        "best_checkpoint_sha256": best_sha256,
        "input_contract_sha256": canonical_sha256(
            prepared.contracts.input_contract
        ),
        "decoder_contract": prepared.contracts.task_head_contract["decoder"],
    }


def _valid_distribution(value: object, expected_count: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "count",
        "min",
        "max",
        "mean",
    }:
        return False
    if value["count"] != expected_count:
        return False
    if expected_count == 0:
        return all(value[field] is None for field in ("min", "max", "mean"))
    minimum = value["min"]
    maximum = value["max"]
    mean = value["mean"]
    return (
        isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and isinstance(mean, (int, float))
        and not isinstance(mean, bool)
        and math.isfinite(float(mean))
        and 0 < minimum <= maximum
    )


def _valid_capacity_population(value: object, expected_count: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "samples",
        "target_length",
        "required_frames",
    }:
        return False
    return (
        value["samples"] == expected_count
        and _valid_distribution(value["target_length"], expected_count)
        and _valid_distribution(value["required_frames"], expected_count)
    )


def _valid_test_document(
    value: object,
    identity: dict[str, object],
    *,
    expected_best_epoch: int,
    expected_sample_count: int,
) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) != set(identity) | {"best_epoch", "metrics", "sample_count"}:
        return False
    if any(value.get(key) != expected for key, expected in identity.items()):
        return False
    best_epoch = value["best_epoch"]
    sample_count = value["sample_count"]
    if (
        best_epoch != expected_best_epoch
        or sample_count != expected_sample_count
    ):
        return False
    metrics = value["metrics"]
    expected_metric_fields = {
        "test_CER_all",
        "test_CER_feasible",
        "test_feasible_samples",
        "test_infeasible_samples",
        "test_infeasible_ratio",
        "test_capacity",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected_metric_fields:
        return False
    feasible = metrics["test_feasible_samples"]
    infeasible = metrics["test_infeasible_samples"]
    if (
        isinstance(feasible, bool)
        or not isinstance(feasible, int)
        or feasible < 0
        or isinstance(infeasible, bool)
        or not isinstance(infeasible, int)
        or infeasible < 0
        or feasible + infeasible != sample_count
    ):
        return False
    cer_all = metrics["test_CER_all"]
    cer_feasible = metrics["test_CER_feasible"]
    ratio = metrics["test_infeasible_ratio"]
    if (
        isinstance(cer_all, bool)
        or not isinstance(cer_all, (int, float))
        or not math.isfinite(float(cer_all))
        or float(cer_all) < 0
        or isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or float(ratio) != infeasible / sample_count
    ):
        return False
    if feasible == 0:
        if cer_feasible is not None:
            return False
    elif (
        isinstance(cer_feasible, bool)
        or not isinstance(cer_feasible, (int, float))
        or not math.isfinite(float(cer_feasible))
        or float(cer_feasible) < 0
    ):
        return False
    capacity = metrics["test_capacity"]
    if not isinstance(capacity, dict) or set(capacity) != {
        "all",
        "feasible",
        "infeasible",
    }:
        return False
    return (
        _valid_capacity_population(capacity["all"], sample_count)
        and _valid_capacity_population(capacity["feasible"], feasible)
        and _valid_capacity_population(capacity["infeasible"], infeasible)
    )


def _finalize(
    prepared: PreparedRuntime,
    state: TrainingState,
    dependencies: RuntimeDependencies,
) -> None:
    if state.stop_reason not in {"max_epochs", "early_stopping"}:
        raise ProtocolError("cannot finalize a non-terminal training state")
    if not prepared.artifacts.best_checkpoint.is_file():
        raise ProtocolError("terminal run has no best checkpoint")
    best = load_checkpoint(
        prepared.artifacts.best_checkpoint,
        expected_role="best",
    )
    if best["run_id"] != prepared.artifacts.run_id:
        raise ProtocolError("best checkpoint belongs to another run")
    if best["training_contract_sha256"] != (
        prepared.contracts.training_contract_sha256
    ):
        raise ProtocolError("best checkpoint training identity mismatch")
    if (
        best["best_epoch"] != state.best_epoch
        or best["best_metric_value"] != state.best_metric_value
    ):
        raise ProtocolError(
            "best checkpoint conflicts with terminal state"
        )
    load_trainable_state_dict(
        prepared.model,
        best["trainable_state_dict"],
    )
    best_sha = file_sha256(prepared.artifacts.best_checkpoint)
    identity = _test_identity(prepared, best_sha)
    test_document: dict[str, object] | None = None
    if prepared.artifacts.test_path.exists():
        try:
            candidate = read_json(prepared.artifacts.test_path)
        except ProtocolError:
            candidate = None
        if _valid_test_document(
            candidate,
            identity,
            expected_best_epoch=best["best_epoch"],
            expected_sample_count=len(
                prepared.bundle.split_samples("test")
            ),
        ):
            test_document = candidate
    if test_document is None:
        loaders = build_data_loaders(
            prepared.datasets,
            epoch=max(1, state.next_epoch - 1),
            batch_size=prepared.config.batch_size,
            num_workers=prepared.config.num_workers,
        )
        result = evaluate_split(
            prepared.model,
            loaders.test,
            split="test",
            device=prepared.device,
        )
        test_document = {
            **identity,
            "best_epoch": best["best_epoch"],
            "metrics": result.to_dict(),
            "sample_count": len(result.sample_ids),
        }
        prepared.artifacts.write_test(test_document)
    test_sha = file_sha256(prepared.artifacts.test_path)
    summary = prepared.artifacts.write_summary(
        {
            "schema_version": SUMMARY_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": prepared.artifacts.run_id,
            "training_contract_sha256": (
                prepared.contracts.training_contract_sha256
            ),
            "split_manifest_sha256": prepared.bundle.manifest_sha256,
            "vocabulary_sha256": prepared.bundle.vocabulary_sha256,
            "stop_reason": state.stop_reason,
            "committed_epoch": state.next_epoch - 1,
            "global_step": state.global_step,
            "best_metric_name": "val_CER_all",
            "best_metric_value": state.best_metric_value,
            "best_epoch": state.best_epoch,
            "test_sha256": test_sha,
            "test_metrics": test_document["metrics"],
            "image_verification_status": (
                prepared.bundle.image_verification.status
            ),
            "trusted_baseline": (
                prepared.bundle.image_verification.trusted_baseline
            ),
            "reproducibility_status": prepared.reproducibility_status,
        }
    )
    prepared.artifacts.update_run(
        status="completed",
        stage="completed",
        updated_at=_utc_text(dependencies.now_factory()),
        stop_reason=state.stop_reason,
        committed_epoch=state.next_epoch - 1,
        global_step=state.global_step,
        best_metric_value=state.best_metric_value,
        best_epoch=state.best_epoch,
        summary_sha256=canonical_sha256(summary),
        failure=None,
    )


def train(
    config: StaffOMRConfig,
    *,
    dependencies: RuntimeDependencies | None = None,
) -> Path:
    deps = dependencies or RuntimeDependencies.production()
    if not isinstance(config, StaffOMRConfig):
        raise ProtocolError("train requires a normalized StaffOMRConfig")
    artifacts: RunArtifacts | None = None
    stage = "registry_preflight"
    try:
        inspection = deps.backbone_provider.inspect(
            config.model_name,
            config.model_revision,
        )
        stage = "dataset_validation"
        bundle = load_dataset_bundle(
            config.dataset_bundle_path,
            config.data_path,
            verify_image_hashes=config.verify_image_hashes,
        )
        contracts = build_protocol_contracts(config, bundle, inspection)
        package_versions = deps.package_versions_factory()
        git_metadata = deps.git_metadata_factory()
        created_at = deps.now_factory()
        run_document = _initial_run_document(
            config=config,
            bundle=bundle,
            contracts=contracts,
            inspection=inspection,
            package_versions=package_versions,
            git_metadata=git_metadata,
            created_at=created_at,
        )
        artifacts = RunArtifacts.create(
            output_root=config.output_root,
            experiment_name=config.experiment_name,
            training_contract_sha256=contracts.training_contract_sha256,
            source_bundle_path=config.dataset_bundle_path,
            run_document=run_document,
            now=created_at,
            run_uuid=deps.uuid_factory(),
        )
        stage = "augmentation_preflight"
        # The combined routine changes the stage label before the CTC policy.
        augmentation_report = preflight_augmentation(
            config.augmentation_profile
        )
        artifacts.update_run(
            preflight={
                "status": "running",
                "augmentation": augmentation_report,
            }
        )
        stage = "ctc_preflight"
        ctc_preflight = analyze_ctc_feasibility(
            bundle.samples,
            bundle.vocabulary,
            patch_cols=config.patch_cols,
        )
        ctc_policy = apply_train_policy(
            ctc_preflight,
            manifest_sha256=bundle.manifest_sha256,
            policy=config.train_infeasible_policy,
            exclusions_path=config.train_exclusions_path,
            candidate_path=(
                artifacts.run_dir / "train_exclusions.candidate.json"
                if config.train_infeasible_policy == "fail"
                else None
            ),
        )
        if (
            ctc_policy.exclusions_sha256 != contracts.exclusion_sha256
            or ctc_policy.exclusion_count != contracts.exclusion_count
        ):
            raise ProtocolError(
                "CTC exclusion result differs from training contract"
            )
        if config.train_infeasible_policy == "exclude_listed":
            write_canonical_json(
                artifacts.run_dir / "train_exclusions.json",
                contracts.exclusion_document,
            )
        artifacts.update_run(
            preflight={
                "status": "passed",
                "augmentation": augmentation_report,
                "ctc": ctc_preflight.to_dict(),
                "retained_train_samples": len(
                    ctc_policy.retained_train_sample_ids
                ),
                "excluded_train_samples": ctc_policy.exclusion_count,
                "train_exclusions_sha256": (
                    ctc_policy.exclusions_sha256
                ),
            }
        )
        stage = "backbone_load"
        prepared = _prepare_after_preflight(
            config=config,
            artifacts=artifacts,
            bundle=bundle,
            inspection=inspection,
            contracts=contracts,
            ctc_preflight=ctc_preflight,
            ctc_policy=ctc_policy,
            package_versions=package_versions,
            dependencies=deps,
            initial_launch_config=contracts.launch_config,
            reproducibility_status="exact",
        )
        stage = "training"
        state = _run_epochs(prepared, TrainingState(), deps)
        stage = "finalization"
        _finalize(prepared, state, deps)
        return artifacts.run_dir
    except Exception as exc:
        if artifacts is not None:
            _record_failure(
                artifacts,
                stage=stage,
                error=exc,
                dependencies=deps,
            )
        raise


def _version_pair(value: str, field: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise ProtocolError(f"cannot parse {field} version {value!r}")
    return int(match.group(1)), int(match.group(2))


def _environment_differences(
    saved: dict[str, str],
    current: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hard: list[dict[str, object]] = []
    soft: list[dict[str, object]] = []
    for name in CORE_PACKAGES:
        if name not in saved or name not in current:
            hard.append(
                {
                    "field": name,
                    "saved": saved.get(name),
                    "current": current.get(name),
                    "reason": "missing_core_package",
                }
            )
            continue
        if _version_pair(saved[name], name) != _version_pair(
            current[name], name
        ):
            hard.append(
                {
                    "field": name,
                    "saved": saved[name],
                    "current": current[name],
                    "reason": "major.minor",
                }
            )
        elif saved[name] != current[name]:
            soft.append(
                {
                    "field": name,
                    "saved": saved[name],
                    "current": current[name],
                    "reason": "patch_or_build",
                }
            )
    for name in sorted(
        (set(saved) | set(current)) - set(CORE_PACKAGES),
        key=lambda value: value.encode("utf-8"),
    ):
        if saved.get(name) != current.get(name):
            soft.append(
                {
                    "field": name,
                    "saved": saved.get(name),
                    "current": current.get(name),
                    "reason": "non_core_package",
                }
            )
    return hard, soft


def _runtime_environment_differences(
    saved: object,
    current: dict[str, object],
) -> list[dict[str, object]]:
    if not isinstance(saved, dict):
        raise ProtocolError("run runtime_environment is missing or invalid")
    differences: list[dict[str, object]] = []
    for field in sorted(
        set(saved) | set(current),
        key=lambda value: value.encode("utf-8"),
    ):
        if saved.get(field) != current.get(field):
            differences.append(
                {
                    "field": f"runtime_environment.{field}",
                    "saved": saved.get(field),
                    "current": current.get(field),
                    "reason": "runtime_environment",
                }
            )
    return differences


def _repair_epoch_sidecars(
    artifacts: RunArtifacts,
    last: dict[str, object],
) -> None:
    artifacts.repair_metrics_jsonl(
        committed_epoch=last["epoch"],
        committed_epoch_record=last["epoch_record"],
    )
    if last["best_updated"]:
        repair = dict(last)
        repair["checkpoint_role"] = "best"
        needs_write = True
        if artifacts.best_checkpoint.exists():
            try:
                existing = load_checkpoint(
                    artifacts.best_checkpoint,
                    expected_role="best",
                )
                needs_write = not (
                    existing["epoch"] == last["epoch"]
                    and existing["best_epoch"] == last["best_epoch"]
                    and existing["best_metric_value"]
                    == last["best_metric_value"]
                    and existing["training_contract_sha256"]
                    == last["training_contract_sha256"]
                )
            except ProtocolError:
                needs_write = True
        if needs_write:
            save_checkpoint(artifacts.best_checkpoint, repair)
    elif last["best_epoch"] is not None:
        if not artifacts.best_checkpoint.is_file():
            raise ProtocolError(
                "best checkpoint is missing and cannot be reconstructed "
                "from a non-best last epoch"
            )
        best = load_checkpoint(
            artifacts.best_checkpoint,
            expected_role="best",
        )
        if (
            best["best_epoch"] != last["best_epoch"]
            or best["best_metric_value"] != last["best_metric_value"]
            or best["training_contract_sha256"]
            != last["training_contract_sha256"]
        ):
            raise ProtocolError("best checkpoint conflicts with last checkpoint")
    artifacts.update_run(
        committed_epoch=last["epoch"],
        global_step=last["global_step"],
        best_metric_value=last["best_metric_value"],
        best_epoch=last["best_epoch"],
        stop_reason=last["stop_reason"],
        early_stopping_state=last["early_stopping_state"],
    )


def _validate_resume_run_document(
    artifacts: RunArtifacts,
    run_document: dict[str, object],
    last: dict[str, object],
) -> None:
    """Cross-check immutable run ownership before sidecar repair."""
    if run_document.get("schema_version") != RUN_SCHEMA:
        raise ProtocolError("run.json schema_version mismatch")
    if run_document.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("run.json protocol_version mismatch")
    if run_document.get("run_id") != last["run_id"]:
        raise ProtocolError("run.json run_id conflicts with last checkpoint")
    if run_document.get("run_name") != artifacts.run_dir.name:
        raise ProtocolError("run.json run_name conflicts with its directory")

    immutable_pairs = (
        (
            "training_contract",
            last["training_contract"],
        ),
        (
            "training_contract_sha256",
            last["training_contract_sha256"],
        ),
        (
            "initial_launch_config",
            last["initial_launch_config"],
        ),
        (
            "initial_launch_config_sha256",
            last["initial_launch_config_sha256"],
        ),
        ("input_contract", last["input_contract"]),
        (
            "augmentation_contract_sha256",
            last["augmentation_contract_sha256"],
        ),
        ("backbone_config", last["backbone_config"]),
    )
    for field, expected in immutable_pairs:
        if run_document.get(field) != expected:
            raise ProtocolError(
                f"run.json {field} conflicts with last checkpoint"
            )

    identity_hashes = run_document.get("identity_hashes")
    expected_identity = {
        "split_manifest_sha256": last["split_manifest_sha256"],
        "training_contract_sha256": last["training_contract_sha256"],
        "vocabulary_sha256": last["vocabulary_sha256"],
    }
    if identity_hashes != expected_identity:
        raise ProtocolError(
            "run.json identity_hashes conflict with last checkpoint"
        )
    dataset = run_document.get("dataset")
    if not isinstance(dataset, dict):
        raise ProtocolError("run.json dataset must be an object")
    expected_dataset_hashes = {
        "dataset_bundle_sha256": last["dataset_bundle_sha256"],
        "split_manifest_sha256": last["split_manifest_sha256"],
        "vocabulary_sha256": last["vocabulary_sha256"],
    }
    for field, expected in expected_dataset_hashes.items():
        if dataset.get(field) != expected:
            raise ProtocolError(
                f"run.json dataset.{field} conflicts with last checkpoint"
            )

    registry_evidence = last["base_model_registry_evidence"]
    if (
        not isinstance(registry_evidence, dict)
        or run_document.get("base_model_registry_evidence")
        != registry_evidence
    ):
        raise ProtocolError(
            "run.json base_model_registry_evidence conflicts with "
            "last checkpoint"
        )
    current_launch = run_document.get("current_launch_config")
    current_launch_hash = run_document.get("current_launch_config_sha256")
    if not isinstance(current_launch, dict):
        raise ProtocolError("run.json current_launch_config must be an object")
    if canonical_sha256(current_launch) != current_launch_hash:
        raise ProtocolError("run.json current_launch_config SHA-256 mismatch")
    initial_launch = last["initial_launch_config"]
    if set(current_launch) != set(initial_launch):
        raise ProtocolError("run.json current_launch_config fields mismatch")
    for field in sorted(
        set(initial_launch) - RESUME_MUTABLE_LAUNCH_FIELDS,
        key=lambda value: value.encode("utf-8"),
    ):
        if current_launch[field] != initial_launch[field]:
            raise ProtocolError(
                f"run.json current_launch_config.{field} changed a "
                "training-semantic field"
            )
    if not isinstance(run_document.get("launch_history"), list):
        raise ProtocolError("run.json launch_history must be an array")
    if not isinstance(run_document.get("resume_history"), list):
        raise ProtocolError("run.json resume_history must be an array")


def _resume_config(
    *,
    run_dir: Path,
    run_document: dict[str, object],
    last: dict[str, object],
    max_epochs: int | None,
    num_workers: int | None,
    data_path: str | Path | None,
    device: str | None,
    verify_image_hashes: str | None,
) -> tuple[StaffOMRConfig, dict[str, object], int]:
    current_launch = run_document.get(
        "current_launch_config",
        last["initial_launch_config"],
    )
    if not isinstance(current_launch, dict):
        raise ProtocolError("run current_launch_config is invalid")
    old_max = current_launch["max_epochs"]
    selected_max = old_max if max_epochs is None else max_epochs
    selected_workers = (
        current_launch["num_workers"]
        if num_workers is None
        else num_workers
    )
    selected_data = (
        current_launch["data_path"] if data_path is None else data_path
    )
    selected_device = (
        current_launch["device"] if device is None else device
    )
    selected_verification = (
        current_launch["verify_image_hashes"]
        if verify_image_hashes is None
        else verify_image_hashes
    )
    exclusions = last["train_exclusions_relpath"]
    config = StaffOMRConfig.create(
        experiment_name=current_launch["experiment_name"],
        data_path=selected_data,
        dataset_bundle_path=run_dir,
        approved_revisions={
            current_launch["model_name"]: {
                last["base_model_revision"],
            }
        },
        default_revisions={
            current_launch["model_name"]: last["base_model_revision"],
        },
        model_name=current_launch["model_name"],
        model_revision=last["base_model_revision"],
        method=current_launch["method"],
        patch_rows=current_launch["patch_rows"],
        patch_cols=current_launch["patch_cols"],
        augmentation_profile=current_launch["augmentation_profile"],
        train_infeasible_policy=current_launch["train_infeasible_policy"],
        train_exclusions_path=(
            run_dir / exclusions if exclusions is not None else None
        ),
        batch_size=current_launch["batch_size"],
        num_workers=selected_workers,
        learning_rate=current_launch["learning_rate"],
        max_epochs=selected_max,
        start_eval=current_launch["start_eval"],
        patience=current_launch["patience"],
        seed=current_launch["seed"],
        output_root=current_launch["output_root"],
        device=selected_device,
        verify_image_hashes=selected_verification,
    )
    return config, current_launch, old_max


def resume(
    run_dir: str | Path,
    *,
    max_epochs: int | None = None,
    num_workers: int | None = None,
    data_path: str | Path | None = None,
    device: str | None = None,
    verify_image_hashes: str | None = None,
    allow_env_drift: bool = False,
    dependencies: RuntimeDependencies | None = None,
) -> Path:
    deps = dependencies or RuntimeDependencies.production()
    artifacts = RunArtifacts.from_existing(run_dir)
    stage = "resume_validation"
    try:
        last = load_checkpoint(
            artifacts.last_checkpoint,
            expected_role="last",
        )
        if last["run_id"] != artifacts.run_id:
            raise ProtocolError("last checkpoint belongs to another run")
        run_document = artifacts.load_run()
        _validate_resume_run_document(artifacts, run_document, last)
        validate_resume_artifact_identity(
            last,
            run_dir=artifacts.run_dir,
        )
        _repair_epoch_sidecars(artifacts, last)
        run_document = artifacts.load_run()
        config, previous_launch, old_max = _resume_config(
            run_dir=artifacts.run_dir,
            run_document=run_document,
            last=last,
            max_epochs=max_epochs,
            num_workers=num_workers,
            data_path=data_path,
            device=device,
            verify_image_hashes=verify_image_hashes,
        )
        if config.max_epochs < old_max:
            raise ProtocolError("resume max_epochs cannot decrease")
        budget_extended = config.max_epochs > old_max
        if budget_extended and last["stop_reason"] == "early_stopping":
            raise ProtocolError(
                "early-stopped run is terminal and cannot extend max_epochs"
            )
        if budget_extended and config.max_epochs < last["next_epoch"]:
            raise ProtocolError(
                "extended max_epochs must be at least checkpoint next_epoch"
            )
        current_packages = deps.package_versions_factory()
        hard_drift, soft_drift = _environment_differences(
            last["package_versions"],
            current_packages,
        )
        if hard_drift:
            raise ProtocolError(
                f"core package major.minor drift is not resumable: {hard_drift!r}"
            )
        if soft_drift and not allow_env_drift:
            raise ProtocolError(
                "environment drift requires --allow_env_drift: "
                f"{soft_drift!r}"
            )
        resolved_device = _resolve_device(config.device)
        current_environment = deps.runtime_environment_factory(resolved_device)
        runtime_drift = _runtime_environment_differences(
            run_document.get("runtime_environment"),
            current_environment,
        )
        combined_soft_drift = [*soft_drift, *runtime_drift]
        if runtime_drift and not allow_env_drift:
            raise ProtocolError(
                "runtime environment drift requires --allow_env_drift: "
                f"{runtime_drift!r}"
            )
        reproducibility = (
            "environment_drift" if combined_soft_drift else "exact"
        )
        inspection = deps.backbone_provider.inspect(
            config.model_name,
            config.model_revision,
        )
        bundle = load_dataset_bundle(
            artifacts.run_dir,
            config.data_path,
            verify_image_hashes=config.verify_image_hashes,
            bundle_filename=RUN_BUNDLE_FILE,
        )
        contracts = build_protocol_contracts(config, bundle, inspection)
        if contracts.training_contract_sha256 != last[
            "training_contract_sha256"
        ]:
            raise ProtocolError("resume training contract identity mismatch")
        augmentation_report = preflight_augmentation(
            config.augmentation_profile
        )
        ctc_preflight = analyze_ctc_feasibility(
            bundle.samples,
            bundle.vocabulary,
            patch_cols=config.patch_cols,
        )
        ctc_policy = apply_train_policy(
            ctc_preflight,
            manifest_sha256=bundle.manifest_sha256,
            policy=config.train_infeasible_policy,
            exclusions_path=config.train_exclusions_path,
        )
        stage = "backbone_load"
        prepared = _prepare_after_preflight(
            config=config,
            artifacts=artifacts,
            bundle=bundle,
            inspection=inspection,
            contracts=contracts,
            ctc_preflight=ctc_preflight,
            ctc_policy=ctc_policy,
            package_versions=current_packages,
            dependencies=deps,
            initial_launch_config=last["initial_launch_config"],
            reproducibility_status=reproducibility,
            resolved_device=resolved_device,
            runtime_environment=current_environment,
        )
        validate_resume_checkpoint(
            last,
            run_dir=artifacts.run_dir,
            expected_training_contract_sha256=(
                contracts.training_contract_sha256
            ),
            expected_optimizer_parameter_names=(
                prepared.optimizer_parameter_names
            ),
            expected_trainable_parameter_names=(
                prepared.optimizer_parameter_names
            ),
        )
        restore_checkpoint(
            last,
            model=prepared.model,
            optimizer=prepared.optimizer,
            optimizer_parameter_names=prepared.optimizer_parameter_names,
        )
        launch = contracts.launch_config
        launch_hash = contracts.launch_config_sha256
        timestamp = _utc_text(deps.now_factory())
        event: dict[str, object] = {
            "timestamp": timestamp,
            "allow_env_drift": allow_env_drift,
            "environment_differences": combined_soft_drift,
            "launch_config_sha256": launch_hash,
            "max_epochs": [old_max, config.max_epochs],
            "num_workers": [
                previous_launch["num_workers"],
                config.num_workers,
            ],
            "data_path": [
                previous_launch["data_path"],
                str(config.data_path),
            ],
            "device": [
                previous_launch["device"],
                config.device,
            ],
            "previous_summary_sha256": (
                file_sha256(artifacts.summary_path)
                if artifacts.summary_path.exists()
                else None
            ),
        }
        artifacts.append_resume_event(event)
        refreshed = artifacts.load_run()
        launch_history = refreshed.get("launch_history")
        if not isinstance(launch_history, list):
            raise ProtocolError("run launch_history must be an array")
        launch_history.append(
            {
                "kind": "resume",
                "timestamp": timestamp,
                "launch_config": launch,
                "launch_config_sha256": launch_hash,
            }
        )
        artifacts.update_run(
            current_launch_config=launch,
            current_launch_config_sha256=launch_hash,
            launch_history=launch_history,
            package_versions=current_packages,
            image_verification={
                "mode": bundle.image_verification.mode,
                "hits": bundle.image_verification.hits,
                "recomputed": bundle.image_verification.recomputed,
                "status": bundle.image_verification.status,
                "trusted_baseline": (
                    bundle.image_verification.trusted_baseline
                ),
            },
            preflight={
                "status": "passed",
                "augmentation": augmentation_report,
                "ctc": ctc_preflight.to_dict(),
                "retained_train_samples": len(
                    ctc_policy.retained_train_sample_ids
                ),
                "excluded_train_samples": ctc_policy.exclusion_count,
            },
            reproducibility_status=reproducibility,
            failure=None,
        )
        state = TrainingState.from_checkpoint(last)
        if budget_extended:
            state.stop_reason = None
        should_train = (
            state.stop_reason is None
            and state.next_epoch <= config.max_epochs
        )
        if should_train:
            stage = "training"
            artifacts.update_run(status="running", stage="training")
            state = _run_epochs(prepared, state, deps)
        stage = "finalization"
        _finalize(prepared, state, deps)
        return artifacts.run_dir
    except Exception as exc:
        _record_failure(
            artifacts,
            stage=stage,
            error=exc,
            dependencies=deps,
        )
        raise
