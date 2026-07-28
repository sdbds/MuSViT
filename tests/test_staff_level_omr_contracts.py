from pathlib import Path

from experiments.staff_level_omr.protocol.backbone import (
    BackboneInspection,
    BackboneMetadata,
    BackboneRegistryEntry,
    BlobEvidence,
    WeightEvidence,
)
from experiments.staff_level_omr.protocol.canonical import canonical_sha256
from experiments.staff_level_omr.protocol.config import StaffOMRConfig
from experiments.staff_level_omr.protocol.contracts import (
    build_protocol_contracts,
)
from experiments.staff_level_omr.protocol.data_bundle import (
    ImageVerificationStats,
    ValidatedDatasetBundle,
)
from experiments.staff_level_omr.protocol.vocabulary import (
    TARGET_PARSER,
    TOKEN_SORT,
    VOCAB_SCHEMA,
    Vocabulary,
)


REVISION = "a" * 40


def _vocabulary() -> Vocabulary:
    return Vocabulary.from_document(
        {
            "schema_version": VOCAB_SCHEMA,
            "dataset_id": "fixture",
            "vocabulary_scope": "closed_corpus",
            "source_manifest_sha256": "b" * 64,
            "target_parser": TARGET_PARSER,
            "token_sort": TOKEN_SORT,
            "unicode_normalization": "none",
            "blank_id": 0,
            "tokens": ["bar", "note"],
        },
        manifest_sha256="b" * 64,
        dataset_id="fixture",
    )


def _inspection() -> BackboneInspection:
    entry = BackboneRegistryEntry(
        alias="fixture",
        model_id="fixture/vit",
        revision=REVISION,
        readme=BlobEvidence("README.md", "1" * 40, 10),
        config=BlobEvidence("config.json", "2" * 40, 20),
        weights=WeightEvidence("model.safetensors", "3" * 40, "c" * 64, 30),
        preprocessor_config_absent=True,
        reviewed_input_contract="staff_omr_input_v2",
    )
    metadata = BackboneMetadata(
        model_id=entry.model_id,
        revision=entry.revision,
        image_height=16,
        image_width=16,
        patch_height=8,
        patch_width=8,
        hidden_size=8,
        num_channels=3,
        prefix_tokens=1,
        model_type="vit_mae",
        architectures=("ViTMAEForPreTraining",),
    )
    return BackboneInspection(
        entry=entry,
        metadata=metadata,
        raw_config={
            "model_type": "vit_mae",
            "architectures": ["ViTMAEForPreTraining"],
            "image_size": 16,
            "patch_size": 8,
            "hidden_size": 8,
            "num_channels": 3,
        },
        registry_evidence={"tree_verified": True},
    )


def _bundle(tmp_path: Path) -> ValidatedDatasetBundle:
    return ValidatedDatasetBundle(
        bundle_path=tmp_path / "bundle",
        data_path=tmp_path / "data",
        dataset_id="fixture",
        dataset_bundle_sha256="d" * 64,
        manifest_sha256="b" * 64,
        vocabulary_sha256=canonical_sha256(_vocabulary().to_document()),
        vocabulary=_vocabulary(),
        samples=(),
        image_verification=ImageVerificationStats(
            mode="always",
            hits=0,
            recomputed=6,
            status="verified",
            trusted_baseline=True,
        ),
    )


def _config(tmp_path: Path, **overrides) -> StaffOMRConfig:
    data = tmp_path / "data"
    bundle = tmp_path / "bundle"
    data.mkdir(exist_ok=True)
    bundle.mkdir(exist_ok=True)
    values = {
        "experiment_name": "fixture",
        "data_path": data,
        "dataset_bundle_path": bundle,
        "approved_revisions": {"fixture": {REVISION}},
        "default_revisions": {"fixture": REVISION},
        "model_name": "fixture",
        "method": "linear_probe",
        "patch_rows": 2,
        "patch_cols": 2,
        "max_epochs": 2,
        "start_eval": 1,
        "output_root": tmp_path / "runs",
        "device": "cpu",
    }
    values.update(overrides)
    return StaffOMRConfig.create(**values)


def test_training_identity_excludes_workers_paths_and_epoch_budget(tmp_path):
    inspection = _inspection()
    bundle = _bundle(tmp_path)
    first = build_protocol_contracts(
        _config(tmp_path, num_workers=0, max_epochs=2),
        bundle,
        inspection,
    )
    second = build_protocol_contracts(
        _config(
            tmp_path,
            num_workers=4,
            max_epochs=5,
            output_root=tmp_path / "elsewhere",
        ),
        bundle,
        inspection,
    )

    assert first.training_contract == second.training_contract
    assert first.training_contract_sha256 == second.training_contract_sha256
    assert first.launch_config_sha256 != second.launch_config_sha256
    assert "num_workers" not in first.training_contract
    assert "max_epochs" not in first.training_contract


def test_method_geometry_and_augmentation_change_training_identity(tmp_path):
    bundle = _bundle(tmp_path)
    inspection = _inspection()
    linear = build_protocol_contracts(
        _config(tmp_path, method="linear_probe"),
        bundle,
        inspection,
    )
    lora = build_protocol_contracts(
        _config(tmp_path, method="lora"),
        bundle,
        inspection,
    )
    no_augmentation = build_protocol_contracts(
        _config(tmp_path, augmentation_profile="none"),
        bundle,
        inspection,
    )

    assert linear.input_plan.geometry == "native_pad"
    assert lora.input_plan.geometry == "exact_grid"
    assert linear.training_contract_sha256 != lora.training_contract_sha256
    assert (
        linear.training_contract_sha256
        != no_augmentation.training_contract_sha256
    )


def test_training_contract_contains_optimizer_seed_ctc_and_head_contracts(
    tmp_path,
):
    contracts = build_protocol_contracts(
        _config(tmp_path),
        _bundle(tmp_path),
        _inspection(),
    )
    training = contracts.training_contract

    assert training["protocol_version"] == "staff_omr_v2"
    assert training["optimizer"]["type"] == "torch.optim.Adam"
    assert training["optimizer"]["scheduler"] == "none"
    assert training["ctc_loss"] == {
        "blank_id": 0,
        "reduction": "mean",
        "type": "torch.nn.functional.ctc_loss",
        "zero_infinity": False,
    }
    assert training["seed"]["schedule_version"] == (
        "staff_omr_sample_epoch_sha256_v1"
    )
    assert training["task_head"]["schema_version"] == (
        "staff_omr_task_head_v1"
    )
    assert training["num_classes"] == len(_vocabulary().tokens) + 1
    assert contracts.identity_hashes == {
        "split_manifest_sha256": "b" * 64,
        "training_contract_sha256": contracts.training_contract_sha256,
        "vocabulary_sha256": bundle_hash
        if (bundle_hash := _bundle(tmp_path).vocabulary_sha256)
        else "",
    }


def test_same_exclusion_content_at_different_paths_has_same_identity(tmp_path):
    exclusion = {
        "schema_version": "staff_omr_train_exclusions_v1",
        "source_manifest_sha256": "b" * 64,
        "patch_cols": 2,
        "required_frames_algorithm": "ctc_minimum_frames_v1",
        "sample_ids": ["sample-01"],
    }
    from experiments.staff_level_omr.protocol.canonical import (
        write_canonical_json,
    )

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "nested" / "second.json"
    write_canonical_json(first_path, exclusion)
    write_canonical_json(second_path, exclusion)
    common = {
        "train_infeasible_policy": "exclude_listed",
        "method": "lora",
    }
    first = build_protocol_contracts(
        _config(tmp_path, train_exclusions_path=first_path, **common),
        _bundle(tmp_path),
        _inspection(),
    )
    second = build_protocol_contracts(
        _config(tmp_path, train_exclusions_path=second_path, **common),
        _bundle(tmp_path),
        _inspection(),
    )

    assert first.training_contract_sha256 == second.training_contract_sha256
    assert first.exclusion_document == exclusion
