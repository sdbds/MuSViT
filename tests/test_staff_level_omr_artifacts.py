from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.staff_level_omr.protocol.artifacts import (
    RunArtifacts,
    file_sha256,
)
from experiments.staff_level_omr.protocol.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    read_json,
    write_canonical_json,
)
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol import artifacts as artifacts_module
from experiments.staff_level_omr.protocol import canonical as canonical_module


CONTRACT_HASH = "a" * 64
NOW = datetime(2026, 7, 29, 3, 4, 5, tzinfo=timezone.utc)


def _source_bundle(root: Path) -> Path:
    source = root / "source-bundle"
    source.mkdir(exist_ok=True)
    documents = {
        "bundle.json": {
            "schema_version": "staff_omr_dataset_bundle_v1",
            "manifest_file": "split_manifest.json",
            "vocabulary_file": "vocabulary.json",
            "image_verification_index_file": "image_verification_index.json",
        },
        "split_manifest.json": {
            "schema_version": "staff_omr_split_v1",
            "samples": [{"sample_id": "fixture"}],
        },
        "vocabulary.json": {
            "schema_version": "staff_omr_vocab_v1",
            "tokens": ["note"],
        },
        "image_verification_index.json": {
            "schema_version": "staff_omr_image_verification_v1",
            "entries": [],
        },
    }
    for filename, document in documents.items():
        write_canonical_json(source / filename, document)
    return source


def _create(tmp_path: Path, *, run_uuid: str = "1" * 32) -> RunArtifacts:
    return RunArtifacts.create(
        output_root=tmp_path / "runs",
        experiment_name="fixture",
        training_contract_sha256=CONTRACT_HASH,
        source_bundle_path=_source_bundle(tmp_path),
        run_document={
            "protocol_version": "staff_omr_v2",
            "status": "preflighting",
            "resume_history": [],
        },
        now=NOW,
        run_uuid=run_uuid,
    )


def test_canonical_atomic_replace_syncs_parent_directory(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(
        canonical_module,
        "fsync_parent_directory",
        lambda path: synced.append(Path(path)),
        raising=False,
    )
    destination = tmp_path / "artifact.json"

    write_canonical_json(destination, {"value": 1})

    assert synced == [tmp_path]


def test_run_directory_is_short_unique_and_copies_canonical_inputs(tmp_path):
    artifacts = _create(tmp_path)

    assert artifacts.run_dir.name == (
        "20260729T030405Z-aaaaaaaaaaaa-111111111111"
    )
    assert artifacts.run_id == "111111111111"
    assert artifacts.checkpoints_dir.is_dir()
    assert read_json(artifacts.run_json)["run_id"] == artifacts.run_id
    assert read_json(artifacts.run_json)["run_name"] == artifacts.run_dir.name
    assert (artifacts.run_dir / "dataset_bundle.json").read_bytes() == (
        artifacts.source_bundle_path / "bundle.json"
    ).read_bytes()
    for filename in (
        "split_manifest.json",
        "vocabulary.json",
        "image_verification_index.json",
    ):
        assert (artifacts.run_dir / filename).read_bytes() == (
            artifacts.source_bundle_path / filename
        ).read_bytes()


def test_same_contract_creates_distinct_runs_without_overwrite(tmp_path):
    source = _source_bundle(tmp_path)
    values = []
    for run_uuid in ("1" * 32, "2" * 32):
        values.append(
            RunArtifacts.create(
                output_root=tmp_path / "runs",
                experiment_name="fixture",
                training_contract_sha256=CONTRACT_HASH,
                source_bundle_path=source,
                run_document={"status": "preflighting", "resume_history": []},
                now=NOW,
                run_uuid=run_uuid,
            )
        )

    assert values[0].run_dir != values[1].run_dir
    assert values[0].run_dir.is_dir()
    assert values[1].run_dir.is_dir()


def test_run_updates_reject_undeclared_schema_fields(tmp_path):
    artifacts = _create(tmp_path)

    with pytest.raises(ProtocolError, match="undeclared|field"):
        artifacts.update_run(model_prefight={"typo": True})


def test_run_name_collision_fails_instead_of_reusing_directory(tmp_path):
    source = _source_bundle(tmp_path)
    _create(tmp_path)

    with pytest.raises(ProtocolError, match="already exists|collision"):
        RunArtifacts.create(
            output_root=tmp_path / "runs",
            experiment_name="fixture",
            training_contract_sha256=CONTRACT_HASH,
            source_bundle_path=source,
            run_document={"status": "preflighting", "resume_history": []},
            now=NOW,
            run_uuid="1" * 32,
        )


def test_opening_missing_run_directory_uses_protocol_error(tmp_path):
    with pytest.raises(ProtocolError, match="run_dir.*does not exist"):
        RunArtifacts.from_existing(tmp_path / "missing-run")


def test_run_creation_records_failure_after_directory_ownership(
    tmp_path,
    monkeypatch,
):
    source = _source_bundle(tmp_path)
    real_write = artifacts_module.write_canonical_json

    def fail_manifest(path, document):
        if Path(path).name == "split_manifest.json":
            raise ProtocolError("simulated bundle copy failure")
        return real_write(path, document)

    monkeypatch.setattr(
        artifacts_module,
        "write_canonical_json",
        fail_manifest,
    )

    with pytest.raises(ProtocolError, match="simulated"):
        RunArtifacts.create(
            output_root=tmp_path / "runs",
            experiment_name="fixture",
            training_contract_sha256=CONTRACT_HASH,
            source_bundle_path=source,
            run_document={
                "status": "preflighting",
                "resume_history": [],
            },
            now=NOW,
            run_uuid="3" * 32,
        )

    run_dir = next((tmp_path / "runs" / "fixture").iterdir())
    run = read_json(run_dir / "run.json")
    assert run["status"] == "failed"
    assert run["stage"] == "artifact_initialization"
    assert run["failure"]["type"] == "ProtocolError"


def test_metrics_append_is_canonical_and_repair_adds_committed_record(tmp_path):
    artifacts = _create(tmp_path)
    epoch1 = {"epoch": 1, "train_loss": 1.25}
    epoch2 = {"epoch": 2, "train_loss": 0.75}
    artifacts.append_epoch_metrics(epoch1)

    report = artifacts.repair_metrics_jsonl(
        committed_epoch=2,
        committed_epoch_record=epoch2,
    )

    assert report == {
        "appended_committed_epoch": True,
        "committed_epoch": 2,
        "partial_tail_removed": False,
        "records": 2,
    }
    assert artifacts.metrics_path.read_bytes() == (
        canonical_json_bytes(epoch1)
        + b"\n"
        + canonical_json_bytes(epoch2)
        + b"\n"
    )


def test_metrics_repair_removes_only_partial_tail(tmp_path):
    artifacts = _create(tmp_path)
    epoch1 = {"epoch": 1, "train_loss": 1.25}
    epoch2 = {"epoch": 2, "train_loss": 0.75}
    artifacts.metrics_path.write_bytes(
        canonical_json_bytes(epoch1) + b"\n" + b'{"epoch":2,"train'
    )

    report = artifacts.repair_metrics_jsonl(
        committed_epoch=2,
        committed_epoch_record=epoch2,
    )

    assert report["partial_tail_removed"] is True
    assert report["appended_committed_epoch"] is True
    assert artifacts.read_metrics() == [epoch1, epoch2]


def test_metrics_repair_replaces_a_file_containing_only_partial_tail(tmp_path):
    artifacts = _create(tmp_path)
    epoch1 = {"epoch": 1, "train_loss": 1.25}
    artifacts.metrics_path.write_bytes(b'{"epoch":1,"train')

    report = artifacts.repair_metrics_jsonl(
        committed_epoch=1,
        committed_epoch_record=epoch1,
    )

    assert report["partial_tail_removed"] is True
    assert report["appended_committed_epoch"] is True
    assert artifacts.read_metrics() == [epoch1]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"epoch":1,"value":"left"}\n{"epoch":1,"value":"right"}\n',
        b'{"epoch":1}\n{"epoch":3}\n',
        b'{"epoch":1}\n{"epoch":2}\n{"epoch":3}\n',
        b'{ "epoch":1 }\n',
    ],
)
def test_metrics_repair_rejects_duplicates_gaps_future_and_noncanonical_lines(
    tmp_path,
    payload,
):
    artifacts = _create(tmp_path)
    artifacts.metrics_path.write_bytes(payload)

    with pytest.raises(ProtocolError):
        artifacts.repair_metrics_jsonl(
            committed_epoch=2,
            committed_epoch_record={"epoch": 2},
        )


def test_metrics_repair_rejects_conflict_with_checkpoint_record(tmp_path):
    artifacts = _create(tmp_path)
    artifacts.append_epoch_metrics({"epoch": 1, "train_loss": 1.0})
    artifacts.append_epoch_metrics({"epoch": 2, "train_loss": 0.9})

    with pytest.raises(ProtocolError, match="conflict"):
        artifacts.repair_metrics_jsonl(
            committed_epoch=2,
            committed_epoch_record={"epoch": 2, "train_loss": 0.8},
        )


def test_resume_history_is_append_only_and_status_updates_atomically(tmp_path):
    artifacts = _create(tmp_path)
    first = {"timestamp": "2026-07-29T03:05:00Z", "max_epochs": [1, 2]}
    second = {"timestamp": "2026-07-29T03:06:00Z", "num_workers": [0, 2]}

    artifacts.append_resume_event(first)
    artifacts.append_resume_event(second)
    artifacts.update_run(status="running", committed_epoch=1)
    document = read_json(artifacts.run_json)

    assert document["resume_history"] == [first, second]
    assert document["status"] == "running"
    assert document["committed_epoch"] == 1


def test_summary_records_actual_checkpoint_hashes(tmp_path):
    artifacts = _create(tmp_path)
    artifacts.last_checkpoint.write_bytes(b"last")
    artifacts.best_checkpoint.write_bytes(b"best")

    summary = artifacts.write_summary(
        {
            "protocol_version": "staff_omr_v2",
            "stop_reason": "max_epochs",
        }
    )

    assert summary["last_checkpoint_sha256"] == file_sha256(
        artifacts.last_checkpoint
    )
    assert summary["best_checkpoint_sha256"] == file_sha256(
        artifacts.best_checkpoint
    )
    assert read_json(artifacts.summary_path) == summary
