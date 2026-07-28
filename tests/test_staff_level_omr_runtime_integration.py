from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.staff_level_omr.protocol.backbone import (
    BackboneInspection,
    BackboneLoadResult,
    BackboneMetadata,
    BackboneRegistryEntry,
    BlobEvidence,
    WeightEvidence,
)
from experiments.staff_level_omr.protocol import runtime as runtime_module
from experiments.staff_level_omr.protocol.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from experiments.staff_level_omr.protocol.config import StaffOMRConfig
from experiments.staff_level_omr.protocol.ctc import CTCInfeasibleError
from experiments.staff_level_omr.protocol.data_bundle import (
    prepare_dataset_bundle,
)
from experiments.staff_level_omr.protocol.runtime import (
    RuntimeDependencies,
    resume,
    train,
)
from experiments.staff_level_omr.protocol.canonical import (
    canonical_sha256,
    read_json,
    write_canonical_json,
)


REVISION = "a" * 40
WEIGHT_SHA = "c" * 64
PACKAGE_VERSIONS = {
    "python": "3.11.11",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "transformers": "4.57.5",
    "peft": "0.19.1",
    "numpy": "2.3.4",
    "albumentations": "2.0.8",
    "opencv": "4.13.0",
}


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, kernel_size=8, stride=8, bias=False)
        with torch.no_grad():
            values = torch.arange(self.patch.weight.numel(), dtype=torch.float32)
            self.patch.weight.copy_(
                values.reshape_as(self.patch.weight)
                / self.patch.weight.numel()
            )

    def forward(self, *, pixel_values, interpolate_pos_encoding):
        spatial = self.patch(pixel_values).flatten(2).transpose(1, 2)
        prefix = spatial.new_zeros((spatial.shape[0], 1, spatial.shape[2]))
        return SimpleNamespace(
            last_hidden_state=torch.cat((prefix, spatial), dim=1)
        )


class DummyBackboneProvider:
    def __init__(self):
        self.inspect_calls = 0
        self.load_calls = 0
        entry = BackboneRegistryEntry(
            alias="fixture",
            model_id="fixture/vit",
            revision=REVISION,
            readme=BlobEvidence("README.md", "1" * 40, 10),
            config=BlobEvidence("config.json", "2" * 40, 20),
            weights=WeightEvidence(
                "model.safetensors",
                "3" * 40,
                WEIGHT_SHA,
                30,
            ),
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
        self.inspection = BackboneInspection(
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

    def inspect(self, model_name, revision):
        self.inspect_calls += 1
        assert model_name == "fixture"
        assert revision == REVISION
        return self.inspection

    def load(self, inspection):
        self.load_calls += 1
        assert inspection == self.inspection
        return BackboneLoadResult(
            model=DummyBackbone(),
            metadata=inspection.metadata,
            loading_info={
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": [],
                "error_msgs": [],
            },
            weight_verification={
                "filename": "model.safetensors",
                "sha256": WEIGHT_SHA,
                "size": 30,
            },
            registry_evidence=inspection.registry_evidence,
        )


class StepClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 1.0
        return self.value


def _dependencies(provider=None, *, run_uuid="1" * 32):
    return RuntimeDependencies(
        backbone_provider=provider or DummyBackboneProvider(),
        now_factory=lambda: datetime(
            2026,
            7,
            29,
            3,
            4,
            5,
            tzinfo=timezone.utc,
        ),
        uuid_factory=lambda: run_uuid,
        monotonic=StepClock(),
        package_versions_factory=lambda: dict(PACKAGE_VERSIONS),
        git_metadata_factory=lambda: {
            "commit": "d" * 40,
            "dirty": False,
        },
        runtime_environment_factory=lambda device: {
            "device": str(device),
            "platform": "test",
        },
    )


def _prepare(tmp_path: Path, *, infeasible: bool = False):
    data = tmp_path / "data"
    for index in range(6):
        group = data / f"score{index:02d}"
        group.mkdir(parents=True)
        Image.new(
            "RGB",
            (16, 16),
            (20 + index * 15, 80, 140),
        ).save(group / "staff00_region.png")
        target = "note note" if infeasible else ("note" if index % 2 else "bar")
        (group / "staff00_gt.txt").write_text(target, encoding="utf-8")
    bundle = tmp_path / "bundle"
    prepare_dataset_bundle(
        data_path=data,
        dataset_id="fixture",
        group_regex=(
            r"(?P<group_id>score[0-9]+)/staff[0-9]+_region[.]png"
        ),
        split_ratios=("4", "1", "1"),
        seed=7,
        out=bundle,
    )
    return data, bundle


def _config(
    tmp_path: Path,
    *,
    max_epochs: int,
    augmentation_profile: str = "none",
    infeasible: bool = False,
    device: str = "cpu",
):
    data, bundle = _prepare(tmp_path, infeasible=infeasible)
    return StaffOMRConfig.create(
        experiment_name="fixture",
        data_path=data,
        dataset_bundle_path=bundle,
        approved_revisions={"fixture": {REVISION}},
        default_revisions={"fixture": REVISION},
        model_name="fixture",
        method="linear_probe",
        patch_rows=2,
        patch_cols=2,
        augmentation_profile=augmentation_profile,
        batch_size=2,
        num_workers=0,
        learning_rate=1e-3,
        max_epochs=max_epochs,
        start_eval=1,
        patience=3,
        seed=11,
        output_root=tmp_path / "runs",
        device=device,
        verify_image_hashes="always",
    )


def _assert_complete_run(run_dir: Path, epoch: int):
    assert (run_dir / "dataset_bundle.json").is_file()
    assert (run_dir / "split_manifest.json").is_file()
    assert (run_dir / "vocabulary.json").is_file()
    assert (run_dir / "image_verification_index.json").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "test.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert read_json(run_dir / "run.json")["status"] == "completed"
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == epoch


def test_cpu_prepare_train_finalize_and_resume_budget_extension(tmp_path):
    config = _config(tmp_path, max_epochs=1)
    dependencies = _dependencies()

    run_dir = train(config, dependencies=dependencies)

    _assert_complete_run(run_dir, 1)
    assert dependencies.backbone_provider.load_calls == 1
    metrics = [
        read_json_line
        for read_json_line in (
            __import__("json").loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        )
    ]
    assert [record["epoch"] for record in metrics] == [1]
    assert metrics[0]["validation_performed"] is True
    assert metrics[0]["val_CER_all"] is not None
    assert metrics[0]["val_CTC_loss_feasible"] is not None

    resumed = resume(
        run_dir,
        max_epochs=2,
        dependencies=_dependencies(run_uuid="9" * 32),
    )

    assert resumed == run_dir
    _assert_complete_run(run_dir, 2)
    run = read_json(run_dir / "run.json")
    assert len(run["resume_history"]) == 1
    assert run["resume_history"][0]["max_epochs"] == [1, 2]
    assert [record["epoch"] for record in (
        __import__("json").loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    )] == [1, 2]


def test_epoch_boundary_resume_matches_uninterrupted_trainable_state(tmp_path):
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted = train(
        _config(
            uninterrupted_root,
            max_epochs=2,
            augmentation_profile="staff_omr_train_v1",
        ),
        dependencies=_dependencies(run_uuid="1" * 32),
    )
    staged = train(
        _config(
            resumed_root,
            max_epochs=1,
            augmentation_profile="staff_omr_train_v1",
        ),
        dependencies=_dependencies(run_uuid="2" * 32),
    )
    resume(
        staged,
        max_epochs=2,
        num_workers=2,
        dependencies=_dependencies(run_uuid="3" * 32),
    )

    expected = load_checkpoint(
        uninterrupted / "checkpoints" / "last.pt"
    )
    actual = load_checkpoint(staged / "checkpoints" / "last.pt")

    assert expected["epoch_record"] == actual["epoch_record"]
    assert expected["global_step"] == actual["global_step"]
    for name in expected["trainable_parameter_names"]:
        torch.testing.assert_close(
            expected["trainable_state_dict"][name],
            actual["trainable_state_dict"][name],
            rtol=0,
            atol=0,
        )
    assert read_json(staged / "run.json")["resume_history"][0][
        "num_workers"
    ] == [0, 2]


def test_infeasible_preflight_writes_failed_run_without_loading_backbone(
    tmp_path,
):
    provider = DummyBackboneProvider()
    config = _config(tmp_path, max_epochs=1, infeasible=True)

    with pytest.raises(CTCInfeasibleError):
        train(config, dependencies=_dependencies(provider))

    assert provider.inspect_calls == 1
    assert provider.load_calls == 0
    runs = list((tmp_path / "runs" / "fixture").iterdir())
    assert len(runs) == 1
    run = read_json(runs[0] / "run.json")
    assert run["status"] == "failed"
    assert run["failure"]["stage"] == "ctc_preflight"
    assert (runs[0] / "train_exclusions.candidate.json").is_file()


def test_resume_repairs_best_metrics_and_terminal_finalization(tmp_path):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    (run_dir / "checkpoints" / "best.pt").unlink()
    (run_dir / "metrics.jsonl").write_bytes(b'{"epoch":1')
    (run_dir / "test.json").unlink()
    (run_dir / "summary.json").unlink()
    run = read_json(run_dir / "run.json")
    run["status"] = "failed"
    write_canonical_json(run_dir / "run.json", run)

    resume(run_dir, dependencies=_dependencies(run_uuid="8" * 32))

    _assert_complete_run(run_dir, 1)
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 1


def test_finalization_rejects_best_checkpoint_conflicting_with_terminal_state(
    tmp_path,
    monkeypatch,
):
    real_run_epochs = runtime_module._run_epochs

    def run_then_corrupt_best(prepared, state, dependencies):
        terminal = real_run_epochs(prepared, state, dependencies)
        best = load_checkpoint(prepared.artifacts.best_checkpoint)
        best["best_metric_value"] += 1.0
        best["epoch_record"]["best_metric_value"] = best[
            "best_metric_value"
        ]
        save_checkpoint(prepared.artifacts.best_checkpoint, best)
        return terminal

    monkeypatch.setattr(
        runtime_module,
        "_run_epochs",
        run_then_corrupt_best,
    )

    with pytest.raises(Exception, match="best checkpoint.*terminal state"):
        train(
            _config(tmp_path, max_epochs=1),
            dependencies=_dependencies(),
        )

    run_dir = next((tmp_path / "runs" / "fixture").iterdir())
    run = read_json(run_dir / "run.json")
    assert run["status"] == "failed"
    assert run["failure"]["stage"] == "finalization"


def test_finalization_recomputes_invalid_matching_test_sidecar(tmp_path):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    test_document = read_json(run_dir / "test.json")
    test_document["metrics"] = {"fake": 1}
    write_canonical_json(run_dir / "test.json", test_document)
    (run_dir / "summary.json").unlink()

    resume(run_dir, dependencies=_dependencies(run_uuid="5" * 32))

    repaired = read_json(run_dir / "test.json")
    assert "fake" not in repaired["metrics"]
    assert repaired["metrics"]["test_CER_all"] is not None
    assert repaired["metrics"]["test_capacity"]["all"]["samples"] == (
        repaired["sample_count"]
    )


def test_finalization_recomputes_test_sidecar_with_wrong_terminal_metadata(
    tmp_path,
):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    test_document = read_json(run_dir / "test.json")
    test_document["best_epoch"] += 100
    test_document["sample_count"] = 2
    metrics = test_document["metrics"]
    metrics["test_feasible_samples"] = 2
    for population in ("all", "feasible"):
        capacity = metrics["test_capacity"][population]
        capacity["samples"] = 2
        capacity["target_length"]["count"] = 2
        capacity["required_frames"]["count"] = 2
    write_canonical_json(run_dir / "test.json", test_document)
    (run_dir / "summary.json").unlink()

    resume(run_dir, dependencies=_dependencies(run_uuid="4" * 32))

    repaired = read_json(run_dir / "test.json")
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    assert repaired["best_epoch"] == best["best_epoch"]
    assert repaired["sample_count"] == 1


def test_finalization_recomputes_malformed_test_sidecar(tmp_path):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    (run_dir / "test.json").write_bytes(b'{"schema_version":')
    (run_dir / "summary.json").unlink()

    resume(run_dir, dependencies=_dependencies(run_uuid="3" * 32))

    repaired = read_json(run_dir / "test.json")
    assert repaired["schema_version"] == "staff_omr_test_v2"
    assert repaired["sample_count"] == 1


def test_resume_rejects_corrupt_run_identity_before_backbone_inspection(
    tmp_path,
):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    run = read_json(run_dir / "run.json")
    run["protocol_version"] = "staff_omr_v3"
    write_canonical_json(run_dir / "run.json", run)
    provider = DummyBackboneProvider()

    with pytest.raises(Exception, match="run.json protocol_version"):
        resume(
            run_dir,
            max_epochs=2,
            dependencies=_dependencies(provider),
        )

    assert provider.inspect_calls == 0
    assert provider.load_calls == 0


def test_resume_rejects_semantic_current_launch_drift_before_inspection(
    tmp_path,
):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    run = read_json(run_dir / "run.json")
    run["current_launch_config"]["method"] = "lora"
    run["current_launch_config_sha256"] = canonical_sha256(
        run["current_launch_config"]
    )
    write_canonical_json(run_dir / "run.json", run)
    provider = DummyBackboneProvider()

    with pytest.raises(Exception, match="current_launch_config.method"):
        resume(
            run_dir,
            max_epochs=2,
            dependencies=_dependencies(provider),
        )

    assert provider.inspect_calls == 0
    assert provider.load_calls == 0


def test_resume_validates_run_bundle_before_repairing_sidecars(tmp_path):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    manifest = read_json(run_dir / "split_manifest.json")
    manifest["dataset_id"] = "corrupt"
    write_canonical_json(run_dir / "split_manifest.json", manifest)
    partial_metrics = b'{"epoch":1'
    (run_dir / "metrics.jsonl").write_bytes(partial_metrics)
    provider = DummyBackboneProvider()

    with pytest.raises(Exception, match="manifest.*(SHA|hash)"):
        resume(
            run_dir,
            max_epochs=2,
            dependencies=_dependencies(provider),
        )

    assert (run_dir / "metrics.jsonl").read_bytes() == partial_metrics
    assert provider.inspect_calls == 0
    assert provider.load_calls == 0


def test_resume_rejects_core_major_minor_drift_before_state_restore(tmp_path):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    drifted = dict(PACKAGE_VERSIONS)
    drifted["torch"] = "3.0.0"
    deps = _dependencies(run_uuid="7" * 32)
    deps.package_versions_factory = lambda: drifted

    with pytest.raises(Exception, match="major.minor|torch"):
        resume(
            run_dir,
            max_epochs=2,
            allow_env_drift=True,
            dependencies=deps,
        )


def test_resume_requires_explicit_acceptance_for_patch_environment_drift(
    tmp_path,
):
    run_dir = train(
        _config(tmp_path, max_epochs=1),
        dependencies=_dependencies(),
    )
    drifted = dict(PACKAGE_VERSIONS)
    drifted["transformers"] = "4.57.6"
    deps = _dependencies(run_uuid="6" * 32)
    deps.package_versions_factory = lambda: drifted

    with pytest.raises(Exception, match="allow_env_drift"):
        resume(
            run_dir,
            max_epochs=2,
            dependencies=deps,
        )

    resume(
        run_dir,
        max_epochs=2,
        allow_env_drift=True,
        dependencies=deps,
    )
    run = read_json(run_dir / "run.json")
    assert run["reproducibility_status"] == "environment_drift"
    assert run["resume_history"][-1]["environment_differences"] == [
        {
            "field": "transformers",
            "saved": "4.57.5",
            "current": "4.57.6",
            "reason": "patch_or_build",
        }
    ]


def test_unavailable_cuda_fails_after_data_preflight_before_weight_load(
    tmp_path,
    monkeypatch,
):
    provider = DummyBackboneProvider()
    config = _config(tmp_path, max_epochs=1, device="cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(Exception, match="CUDA is unavailable"):
        train(config, dependencies=_dependencies(provider))

    assert provider.inspect_calls == 1
    assert provider.load_calls == 0
    run_dir = next((tmp_path / "runs" / "fixture").iterdir())
    assert read_json(run_dir / "run.json")["failure"]["stage"] == (
        "backbone_load"
    )
