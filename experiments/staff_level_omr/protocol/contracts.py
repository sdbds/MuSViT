"""Canonical launch and training identities for staff OMR v2."""

from __future__ import annotations

from dataclasses import dataclass

from .augmentation import augmentation_contract
from .backbone import BackboneInspection
from .canonical import canonical_sha256, read_json
from .config import StaffOMRConfig
from .ctc import EXCLUSIONS_SCHEMA, REQUIRED_FRAMES_ALGORITHM
from .data_bundle import ValidatedDatasetBundle
from .data_pipeline import loader_contract
from .errors import ProtocolError
from .geometry import GeometryPlan, build_geometry_plan
from .modeling import LORA_CONTRACT, task_head_contract
from .optimization import optimizer_contract
from .seeding import SEED_SCHEDULE_VERSION


PROTOCOL_VERSION = "staff_omr_v2"
TRAINING_CONTRACT_SCHEMA = "staff_omr_training_contract_v2"
LAUNCH_CONFIG_SCHEMA = "staff_omr_launch_config_v2"


@dataclass(frozen=True, slots=True)
class ProtocolContracts:
    input_plan: GeometryPlan
    input_contract: dict[str, object]
    augmentation_contract: dict[str, object]
    augmentation_contract_sha256: str
    task_head_contract: dict[str, object]
    exclusion_document: dict[str, object] | None
    exclusion_sha256: str | None
    exclusion_count: int
    training_contract: dict[str, object]
    training_contract_sha256: str
    launch_config: dict[str, object]
    launch_config_sha256: str
    identity_hashes: dict[str, str]


def _exclusion_identity(
    config: StaffOMRConfig,
    bundle: ValidatedDatasetBundle,
) -> tuple[dict[str, object] | None, str | None, int]:
    if config.train_infeasible_policy == "fail":
        return None, None, 0
    if config.train_exclusions_path is None:
        raise ProtocolError("exclude_listed requires train_exclusions_path")
    document = read_json(config.train_exclusions_path)
    if not isinstance(document, dict):
        raise ProtocolError("train exclusions must be a JSON object")
    expected_fields = {
        "schema_version",
        "source_manifest_sha256",
        "patch_cols",
        "required_frames_algorithm",
        "sample_ids",
    }
    if set(document) != expected_fields:
        raise ProtocolError(
            "train exclusions fields mismatch; "
            f"missing={sorted(expected_fields - set(document))!r}, "
            f"extra={sorted(set(document) - expected_fields)!r}"
        )
    if document["schema_version"] != EXCLUSIONS_SCHEMA:
        raise ProtocolError("train exclusions schema_version mismatch")
    if document["source_manifest_sha256"] != bundle.manifest_sha256:
        raise ProtocolError(
            "train exclusions source_manifest_sha256 mismatch"
        )
    if document["patch_cols"] != config.patch_cols:
        raise ProtocolError("train exclusions patch_cols mismatch")
    if document["required_frames_algorithm"] != REQUIRED_FRAMES_ALGORITHM:
        raise ProtocolError(
            "train exclusions required_frames_algorithm mismatch"
        )
    sample_ids = document["sample_ids"]
    if (
        not isinstance(sample_ids, list)
        or any(not isinstance(item, str) or not item for item in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or sample_ids
        != sorted(sample_ids, key=lambda value: value.encode("utf-8"))
    ):
        raise ProtocolError(
            "train exclusions sample_ids must be unique non-empty strings "
            "sorted by UTF-8 bytes"
        )
    return document, canonical_sha256(document), len(sample_ids)


def build_protocol_contracts(
    config: StaffOMRConfig,
    bundle: ValidatedDatasetBundle,
    inspection: BackboneInspection,
) -> ProtocolContracts:
    if config.model_revision != inspection.entry.revision:
        raise ProtocolError(
            "normalized model revision differs from backbone inspection"
        )
    if inspection.metadata.model_id != inspection.entry.model_id:
        raise ProtocolError(
            "backbone metadata model id differs from registry entry"
        )
    plan = build_geometry_plan(
        inspection.metadata,
        config.method,
        config.patch_rows,
        config.patch_cols,
    )
    if plan.geometry != config.input_geometry:
        raise ProtocolError(
            "derived input geometry differs from normalized configuration"
        )
    input_document = plan.to_contract()
    augmentation_document = augmentation_contract(
        config.augmentation_profile
    )
    augmentation_hash = canonical_sha256(augmentation_document)
    head_document = task_head_contract(
        inspection.metadata,
        bundle.vocabulary.num_classes,
    )
    exclusion_document, exclusion_hash, exclusion_count = _exclusion_identity(
        config,
        bundle,
    )
    training: dict[str, object] = {
        "schema_version": TRAINING_CONTRACT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": {
            "dataset_id": bundle.dataset_id,
            "dataset_bundle_sha256": bundle.dataset_bundle_sha256,
            "split_manifest_sha256": bundle.manifest_sha256,
            "vocabulary_sha256": bundle.vocabulary_sha256,
            "vocabulary_scope": "closed_corpus",
        },
        "base_model": {
            "alias": inspection.entry.alias,
            "model_id": inspection.entry.model_id,
            "revision": inspection.entry.revision,
            "loader_class": inspection.entry.loader_class,
            "weights_filename": inspection.entry.weights.filename,
            "weights_sha256": inspection.entry.weights.sha256,
            "weights_size": inspection.entry.weights.size,
            "prefix_tokens": inspection.entry.prefix_tokens,
            "reviewed_input_contract": (
                inspection.entry.reviewed_input_contract
            ),
        },
        "method": config.method,
        "input_geometry": config.input_geometry,
        "patch_grid": {
            "rows": config.patch_rows,
            "cols": config.patch_cols,
        },
        "input_contract": input_document,
        "augmentation_contract": augmentation_document,
        "augmentation_contract_sha256": augmentation_hash,
        "train_ctc_feasibility": {
            "required_frames_algorithm": REQUIRED_FRAMES_ALGORITHM,
            "policy": config.train_infeasible_policy,
            "exclusions_sha256": exclusion_hash,
            "exclusions_count": exclusion_count,
        },
        "ctc_loss": {
            "blank_id": 0,
            "reduction": "mean",
            "type": "torch.nn.functional.ctc_loss",
            "zero_infinity": False,
        },
        "num_classes": bundle.vocabulary.num_classes,
        "task_head": head_document,
        "batching": loader_contract(config.batch_size),
        "seed": {
            "base_seed": config.seed,
            "schedule_version": SEED_SCHEDULE_VERSION,
            "init": "seed32_init",
            "epoch": "seed32_epoch",
            "sample_order": "sha256_epoch_order_v1",
            "sample_augmentation": "seed256_sample_epoch",
            "worker_assignment_semantic": False,
        },
        "optimizer": optimizer_contract(config.learning_rate),
        "backbone_train_mode": (
            "eval" if config.method == "linear_probe" else "train"
        ),
        "lora": dict(LORA_CONTRACT) if config.method == "lora" else None,
        "validation": {
            "start_eval": config.start_eval,
            "patience": config.patience,
            "checkpoint_metric": "val_CER_all",
            "strict_improvement": True,
        },
    }
    launch = {
        "schema_version": LAUNCH_CONFIG_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        **config.to_launch_config(),
    }
    training_hash = canonical_sha256(training)
    launch_hash = canonical_sha256(launch)
    return ProtocolContracts(
        input_plan=plan,
        input_contract=input_document,
        augmentation_contract=augmentation_document,
        augmentation_contract_sha256=augmentation_hash,
        task_head_contract=head_document,
        exclusion_document=exclusion_document,
        exclusion_sha256=exclusion_hash,
        exclusion_count=exclusion_count,
        training_contract=training,
        training_contract_sha256=training_hash,
        launch_config=launch,
        launch_config_sha256=launch_hash,
        identity_hashes={
            "split_manifest_sha256": bundle.manifest_sha256,
            "training_contract_sha256": training_hash,
            "vocabulary_sha256": bundle.vocabulary_sha256,
        },
    )
