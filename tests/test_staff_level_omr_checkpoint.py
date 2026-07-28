from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.staff_level_omr.protocol.canonical import (
    canonical_sha256,
    write_canonical_json,
)
from experiments.staff_level_omr.protocol.checkpoint import (
    CHECKPOINT_FIELDS,
    CHECKPOINT_SCHEMA,
    CheckpointStatic,
    build_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
    validate_resume_checkpoint,
)
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol.optimization import build_optimizer
from experiments.staff_level_omr.protocol.seeding import (
    SEED_SCHEDULE_VERSION,
    initialization_seed,
)
from experiments.staff_level_omr.protocol.vocabulary import (
    TARGET_PARSER,
    TOKEN_SORT,
    VOCAB_SCHEMA,
    Vocabulary,
)


class TinyTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.backbone.requires_grad_(False)
        self.projection = nn.Linear(3, 2, bias=False)
        self.classifier_ctc = nn.Linear(2, 4)


def _vocabulary_document():
    return {
        "schema_version": VOCAB_SCHEMA,
        "dataset_id": "fixture",
        "vocabulary_scope": "closed_corpus",
        "source_manifest_sha256": "b" * 64,
        "target_parser": TARGET_PARSER,
        "token_sort": TOKEN_SORT,
        "unicode_normalization": "none",
        "blank_id": 0,
        "tokens": ["bar", "clef", "note"],
    }


def _static(**overrides) -> CheckpointStatic:
    training_contract = {
        "protocol_version": "staff_omr_v2",
        "method": "linear_probe",
    }
    launch = {
        "max_epochs": 2,
        "num_workers": 0,
        "device": "cpu",
    }
    vocabulary = _vocabulary_document()
    values = {
        "training_contract": training_contract,
        "initial_launch_config": launch,
        "dataset_bundle_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "vocabulary": vocabulary,
        "base_model_id": "fixture/vit",
        "base_model_revision": "c" * 40,
        "base_model_weights_filename": "model.safetensors",
        "base_model_weights_sha256": "d" * 64,
        "base_model_registry_evidence": {"tree_verified": True},
        "backbone_config": {"hidden_size": 3},
        "input_contract": {"schema_version": "staff_omr_input_v2"},
        "augmentation_contract_sha256": "e" * 64,
        "train_exclusions_relpath": None,
        "train_exclusions_sha256": None,
        "train_exclusions_count": 0,
        "base_seed": 7,
        "package_versions": {
            "python": "3.11.11",
            "torch": "2.13.0+cu130",
        },
    }
    values.update(overrides)
    return CheckpointStatic.create(**values)


def _checkpoint(
    role: str = "last",
    *,
    static: CheckpointStatic | None = None,
):
    model = TinyTrainableModel()
    optimizer, names = build_optimizer(model, 3e-4)
    best_epoch = 2 if role == "best" else 1
    best_updated = role == "best"
    early_stopping_state = {
        "bad_epochs": 1,
        "evaluations": 2,
        "patience": 3,
    }
    epoch_record = {
        "epoch": 2,
        "global_step": 6,
        "train_loss": 0.5,
        "best_checkpoint_updated": best_updated,
        "best_metric_name": "val_CER_all",
        "best_metric_value": 0.25,
        "best_epoch": best_epoch,
        "early_stopping_bad_epochs": 1,
        "early_stopping_evaluations": 2,
        "stop_reason": "max_epochs",
    }
    payload = build_checkpoint(
        static=static or _static(),
        run_id="1" * 12,
        role=role,
        model=model,
        optimizer=optimizer,
        optimizer_parameter_names=names,
        epoch=2,
        global_step=6,
        early_stopping_state=early_stopping_state,
        best_metric_name="val_CER_all",
        best_metric_value=0.25,
        best_epoch=best_epoch,
        best_updated=best_updated,
        stop_reason="max_epochs",
        epoch_record=epoch_record,
    )
    return model, optimizer, names, payload


def _run_files(root: Path, static: CheckpointStatic):
    run = root / "run"
    run.mkdir()
    write_canonical_json(
        run / "dataset_bundle.json",
        {"identity": "bundle"},
    )
    write_canonical_json(
        run / "split_manifest.json",
        {"identity": "manifest"},
    )
    write_canonical_json(run / "vocabulary.json", static.vocabulary)
    return run


def test_checkpoint_schema_is_exact_and_does_not_embed_manifest():
    model, _, names, checkpoint = _checkpoint()

    assert set(checkpoint) == CHECKPOINT_FIELDS
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA
    assert checkpoint["protocol_version"] == "staff_omr_v2"
    assert checkpoint["checkpoint_role"] == "last"
    assert checkpoint["state_dict_scope"] == "trainable_only"
    assert checkpoint["vocabulary"]["tokens"] == ["bar", "clef", "note"]
    assert checkpoint["split_manifest_sha256"] == "b" * 64
    assert "split_manifest" not in checkpoint
    assert checkpoint["base_model_weights_sha256"] == "d" * 64
    assert checkpoint["init_seed"] == initialization_seed(7)
    assert checkpoint["seed_schedule_version"] == SEED_SCHEDULE_VERSION
    assert checkpoint["next_epoch"] == 3
    assert checkpoint["trainable_parameter_names"] == names
    assert set(checkpoint["trainable_state_dict"]) == set(names)
    assert not any(name.startswith("backbone.") for name in names)
    assert checkpoint["training_contract_sha256"] == canonical_sha256(
        checkpoint["training_contract"]
    )


def test_checkpoint_vocabulary_decodes_without_target_files():
    _, _, _, checkpoint = _checkpoint()

    vocabulary = Vocabulary.from_document(
        checkpoint["vocabulary"],
        manifest_sha256=checkpoint["split_manifest_sha256"],
        dataset_id="fixture",
    )

    assert vocabulary.decode([1, 3]) == ("bar", "note")


def test_atomic_save_and_strict_load_round_trip(tmp_path):
    _, _, _, checkpoint = _checkpoint()
    path = tmp_path / "last.pt"

    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path, expected_role="last")

    assert loaded.keys() == checkpoint.keys()
    assert loaded["epoch"] == 2
    for name, tensor in checkpoint["trainable_state_dict"].items():
        torch.testing.assert_close(
            loaded["trainable_state_dict"][name],
            tensor,
            rtol=0,
            atol=0,
        )
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_loader_rejects_legacy_state_dict_and_best_for_resume(tmp_path):
    legacy = tmp_path / "legacy.pt"
    torch.save(TinyTrainableModel().state_dict(), legacy)
    with pytest.raises(ProtocolError, match="legacy|schema"):
        load_checkpoint(legacy)

    _, _, _, best = _checkpoint(role="best")
    path = tmp_path / "best.pt"
    save_checkpoint(path, best)
    with pytest.raises(ProtocolError, match="role"):
        load_checkpoint(path, expected_role="last")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("global_step", -1, "global_step"),
        ("stop_reason", "unknown", "stop_reason"),
        ("package_versions", [], "package_versions"),
        ("base_model_revision", "main", "base_model_revision"),
        ("base_model_weights_filename", "", "weights_filename"),
    ],
)
def test_checkpoint_loader_rejects_invalid_dynamic_metadata(
    field,
    value,
    match,
):
    _, _, _, checkpoint = _checkpoint()
    checkpoint[field] = value

    with pytest.raises(ProtocolError, match=match):
        validate_checkpoint_payload(checkpoint)


def test_checkpoint_loader_rejects_incoherent_best_and_early_stop_state():
    _, _, _, checkpoint = _checkpoint()
    checkpoint["early_stopping_state"] = {
        "bad_epochs": -1,
        "evaluations": 1,
        "patience": 3,
    }
    with pytest.raises(ProtocolError, match="bad_epochs"):
        validate_checkpoint_payload(checkpoint)

    _, _, _, checkpoint = _checkpoint()
    checkpoint["best_epoch"] = None
    with pytest.raises(ProtocolError, match="best_metric|best_epoch"):
        validate_checkpoint_payload(checkpoint)

    _, _, _, checkpoint = _checkpoint()
    checkpoint["best_updated"] = True
    with pytest.raises(ProtocolError, match="best_updated"):
        validate_checkpoint_payload(checkpoint)


def test_static_metadata_rejects_hash_and_exclusion_inconsistency():
    values = _static().__dict__ if hasattr(_static(), "__dict__") else None
    assert values is None

    with pytest.raises(ProtocolError, match="exclusions"):
        CheckpointStatic.create(
            training_contract={"protocol_version": "staff_omr_v2"},
            initial_launch_config={"max_epochs": 1},
            dataset_bundle_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
            vocabulary=_vocabulary_document(),
            base_model_id="fixture/vit",
            base_model_revision="c" * 40,
            base_model_weights_filename="model.safetensors",
            base_model_weights_sha256="d" * 64,
            base_model_registry_evidence={},
            backbone_config={},
            input_contract={},
            augmentation_contract_sha256="e" * 64,
            train_exclusions_relpath=None,
            train_exclusions_sha256="f" * 64,
            train_exclusions_count=1,
            base_seed=7,
            package_versions={"python": "3.11.11"},
        )


@pytest.mark.parametrize("filename", ["", ".", "..", "dir/model.safetensors"])
def test_static_metadata_requires_a_safe_plain_weight_filename(filename):
    with pytest.raises(ProtocolError, match="weights_filename"):
        _static(base_model_weights_filename=filename)


def test_optimizer_name_mismatch_is_rejected_before_optimizer_load(monkeypatch):
    model, optimizer, names, checkpoint = _checkpoint()
    called = False

    def forbidden_load(_):
        nonlocal called
        called = True
        raise AssertionError("optimizer state must not be applied")

    monkeypatch.setattr(optimizer, "load_state_dict", forbidden_load)
    checkpoint["optimizer_parameter_names"] = list(reversed(names))

    with pytest.raises(ProtocolError, match="optimizer_parameter_names"):
        restore_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            optimizer_parameter_names=names,
        )
    assert called is False


def test_restore_applies_trainable_and_optimizer_state():
    original, _, _, checkpoint = _checkpoint()
    restored = TinyTrainableModel()
    optimizer, names = build_optimizer(restored, 3e-4)
    with torch.no_grad():
        restored.projection.weight.add_(10)

    restore_checkpoint(
        checkpoint,
        model=restored,
        optimizer=optimizer,
        optimizer_parameter_names=names,
    )

    original_state = original.state_dict()
    restored_state = restored.state_dict()
    for name in checkpoint["trainable_parameter_names"]:
        torch.testing.assert_close(
            restored_state[name],
            original_state[name],
            rtol=0,
            atol=0,
        )


def test_resume_validation_checks_run_local_document_hashes(tmp_path):
    bundle_hash = canonical_sha256({"identity": "bundle"})
    manifest_hash = canonical_sha256({"identity": "manifest"})
    vocabulary = _vocabulary_document()
    vocabulary["source_manifest_sha256"] = manifest_hash
    static = _static(
        dataset_bundle_sha256=bundle_hash,
        split_manifest_sha256=manifest_hash,
        vocabulary=vocabulary,
    )
    _, _, names, checkpoint = _checkpoint(static=static)
    run = _run_files(tmp_path, static)

    validate_resume_checkpoint(
        checkpoint,
        run_dir=run,
        expected_training_contract_sha256=(
            checkpoint["training_contract_sha256"]
        ),
        expected_optimizer_parameter_names=names,
        expected_trainable_parameter_names=(
            checkpoint["trainable_parameter_names"]
        ),
    )

    write_canonical_json(run / "split_manifest.json", {"identity": "changed"})
    with pytest.raises(ProtocolError, match="manifest"):
        validate_resume_checkpoint(
            checkpoint,
            run_dir=run,
            expected_training_contract_sha256=(
                checkpoint["training_contract_sha256"]
            ),
            expected_optimizer_parameter_names=names,
            expected_trainable_parameter_names=(
                checkpoint["trainable_parameter_names"]
            ),
        )


def test_resume_validation_wraps_missing_run_document_as_protocol_error(
    tmp_path,
):
    bundle_hash = canonical_sha256({"identity": "bundle"})
    manifest_hash = canonical_sha256({"identity": "manifest"})
    vocabulary = _vocabulary_document()
    vocabulary["source_manifest_sha256"] = manifest_hash
    static = _static(
        dataset_bundle_sha256=bundle_hash,
        split_manifest_sha256=manifest_hash,
        vocabulary=vocabulary,
    )
    _, _, names, checkpoint = _checkpoint(static=static)
    run = _run_files(tmp_path, static)
    (run / "split_manifest.json").unlink()

    with pytest.raises(ProtocolError, match="split manifest.*missing|missing.*split"):
        validate_resume_checkpoint(
            checkpoint,
            run_dir=run,
            expected_training_contract_sha256=(
                checkpoint["training_contract_sha256"]
            ),
            expected_optimizer_parameter_names=names,
            expected_trainable_parameter_names=(
                checkpoint["trainable_parameter_names"]
            ),
        )
