"""Run-directory ownership and crash-repairable sidecar artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical import (
    canonical_json_bytes,
    read_json,
    write_canonical_json,
)
from .data_bundle import (
    BUNDLE_FILE,
    IMAGE_INDEX_FILE,
    MANIFEST_FILE,
    RUN_BUNDLE_FILE,
    VOCABULARY_FILE,
)
from .errors import ProtocolError


RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
TEST_FILE = "test.json"
SUMMARY_FILE = "summary.json"
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_UUID = re.compile(r"^[0-9a-f]{12,32}$")


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"cannot hash artifact {source}: {exc}") from exc
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ProtocolError(
            f"cannot atomically write artifact {path}: {exc}"
        ) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"metrics JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_metric_line(raw: bytes, line_number: int) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            f"metrics line {line_number} is invalid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"metrics line {line_number} must be an object")
    if canonical_json_bytes(value) != raw:
        raise ProtocolError(f"metrics line {line_number} is not canonical JSON")
    epoch = value.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ProtocolError(
            f"metrics line {line_number} has invalid epoch {epoch!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    run_dir: Path
    run_id: str
    source_bundle_path: Path

    @property
    def run_json(self) -> Path:
        return self.run_dir / RUN_FILE

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / METRICS_FILE

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / CHECKPOINTS_DIR

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / LAST_CHECKPOINT

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / BEST_CHECKPOINT

    @property
    def test_path(self) -> Path:
        return self.run_dir / TEST_FILE

    @property
    def summary_path(self) -> Path:
        return self.run_dir / SUMMARY_FILE

    @classmethod
    def create(
        cls,
        *,
        output_root: str | Path,
        experiment_name: str,
        training_contract_sha256: str,
        source_bundle_path: str | Path,
        run_document: dict[str, object],
        now: datetime | None = None,
        run_uuid: str | None = None,
    ) -> "RunArtifacts":
        if not _SHA256.fullmatch(training_contract_sha256):
            raise ProtocolError(
                "training_contract_sha256 must be 64 lowercase hex characters"
            )
        if (
            not isinstance(experiment_name, str)
            or not experiment_name
            or "/" in experiment_name
            or "\\" in experiment_name
        ):
            raise ProtocolError("experiment_name is not safe for a run path")
        source = Path(source_bundle_path).resolve(strict=True)
        if not source.is_dir():
            raise ProtocolError("source_bundle_path must be a directory")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ProtocolError("run timestamp must be timezone-aware")
        timestamp = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_uuid = (run_uuid or uuid4().hex).lower()
        if not _RUN_UUID.fullmatch(raw_uuid):
            raise ProtocolError("run_uuid must contain at least 12 hex characters")
        run_id = raw_uuid[:12]
        run_name = (
            f"{timestamp}-{training_contract_sha256[:12]}-{run_id}"
        )
        document = dict(run_document)
        if "run_id" in document or "run_name" in document:
            raise ProtocolError(
                "run_document cannot predefine run_id or run_name"
            )
        document["run_id"] = run_id
        document["run_name"] = run_name
        canonical_json_bytes(document)
        experiment_dir = Path(output_root).resolve(strict=False) / experiment_name
        run_dir = experiment_dir / run_name
        try:
            experiment_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(exist_ok=False)
            (run_dir / CHECKPOINTS_DIR).mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise ProtocolError(
                f"run directory collision; path already exists: {run_dir}"
            ) from exc
        except OSError as exc:
            raise ProtocolError(f"cannot create run directory {run_dir}") from exc

        artifacts = cls(
            run_dir=run_dir,
            run_id=run_id,
            source_bundle_path=source,
        )
        copies = {
            BUNDLE_FILE: RUN_BUNDLE_FILE,
            MANIFEST_FILE: MANIFEST_FILE,
            VOCABULARY_FILE: VOCABULARY_FILE,
            IMAGE_INDEX_FILE: IMAGE_INDEX_FILE,
        }
        try:
            # Publish the ownership/status record first so later initialization
            # failures remain auditable inside the run directory.
            write_canonical_json(artifacts.run_json, document)
            for source_name, destination_name in copies.items():
                source_document = read_json(source / source_name)
                write_canonical_json(
                    run_dir / destination_name,
                    source_document,
                )
        except Exception as exc:
            failed = dict(document)
            failed.update(
                {
                    "status": "failed",
                    "stage": "artifact_initialization",
                    "failure": {
                        "stage": "artifact_initialization",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            try:
                write_canonical_json(artifacts.run_json, failed)
            except Exception:
                pass
            raise
        return artifacts

    @classmethod
    def from_existing(cls, run_dir: str | Path) -> "RunArtifacts":
        source = Path(run_dir)
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError(
                f"run_dir does not exist or is unavailable: {source}"
            ) from exc
        if not resolved.is_dir():
            raise ProtocolError(f"run_dir must be a directory: {resolved}")
        document = read_json(resolved / RUN_FILE)
        if not isinstance(document, dict):
            raise ProtocolError("run.json must be an object")
        run_id = document.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(
            r"[0-9a-f]{12}", run_id
        ):
            raise ProtocolError("run.json has an invalid run_id")
        return cls(
            run_dir=resolved,
            run_id=run_id,
            source_bundle_path=resolved,
        )

    def load_run(self) -> dict[str, object]:
        value = read_json(self.run_json)
        if not isinstance(value, dict):
            raise ProtocolError("run.json must be an object")
        return value

    def update_run(self, **changes: object) -> dict[str, object]:
        if "resume_history" in changes:
            raise ProtocolError(
                "resume_history can only change through append_resume_event"
            )
        document = self.load_run()
        document.update(changes)
        write_canonical_json(self.run_json, document)
        return document

    def append_resume_event(
        self,
        event: dict[str, object],
    ) -> dict[str, object]:
        if not isinstance(event, dict) or not event:
            raise ProtocolError("resume event must be a non-empty object")
        canonical_json_bytes(event)
        document = self.load_run()
        history = document.get("resume_history")
        if not isinstance(history, list):
            raise ProtocolError("run.json resume_history must be an array")
        history.append(event)
        write_canonical_json(self.run_json, document)
        return document

    def read_metrics(self) -> list[dict[str, object]]:
        if not self.metrics_path.exists():
            return []
        try:
            raw = self.metrics_path.read_bytes()
        except OSError as exc:
            raise ProtocolError("cannot read metrics.jsonl") from exc
        if raw and not raw.endswith(b"\n"):
            raise ProtocolError("metrics.jsonl has a partial trailing line")
        lines = raw.splitlines()
        records = [
            _parse_metric_line(line, index)
            for index, line in enumerate(lines, start=1)
        ]
        epochs = [record["epoch"] for record in records]
        if epochs != list(range(1, len(records) + 1)):
            raise ProtocolError(
                f"metrics epochs must be contiguous from 1, got {epochs!r}"
            )
        return records

    def append_epoch_metrics(self, record: dict[str, object]) -> None:
        if not isinstance(record, dict):
            raise ProtocolError("epoch metrics must be an object")
        payload = canonical_json_bytes(record)
        parsed = _parse_metric_line(payload, 1)
        existing = self.read_metrics()
        expected_epoch = len(existing) + 1
        if parsed["epoch"] != expected_epoch:
            raise ProtocolError(
                f"next metrics epoch must be {expected_epoch}, "
                f"got {parsed['epoch']}"
            )
        try:
            with self.metrics_path.open("ab") as stream:
                stream.write(payload + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ProtocolError("cannot append metrics.jsonl") from exc

    def repair_metrics_jsonl(
        self,
        *,
        committed_epoch: int,
        committed_epoch_record: dict[str, object] | None,
    ) -> dict[str, object]:
        if (
            isinstance(committed_epoch, bool)
            or not isinstance(committed_epoch, int)
            or committed_epoch < 0
        ):
            raise ProtocolError("committed_epoch must be non-negative")
        raw = self.metrics_path.read_bytes() if self.metrics_path.exists() else b""
        partial = bool(raw and not raw.endswith(b"\n"))
        if partial:
            last_newline = raw.rfind(b"\n")
            complete = raw[: last_newline + 1] if last_newline >= 0 else b""
        else:
            complete = raw
        lines = complete.splitlines()
        records = [
            _parse_metric_line(line, index)
            for index, line in enumerate(lines, start=1)
        ]
        epochs = [int(record["epoch"]) for record in records]
        expected = list(range(1, len(records) + 1))
        if epochs != expected:
            raise ProtocolError(
                f"metrics epochs contain a duplicate or gap: {epochs!r}"
            )
        if epochs and epochs[-1] > committed_epoch:
            raise ProtocolError(
                "metrics.jsonl is ahead of authoritative last checkpoint"
            )

        appended = False
        if committed_epoch == 0:
            if records:
                raise ProtocolError(
                    "metrics exist without an authoritative committed epoch"
                )
            if committed_epoch_record is not None:
                raise ProtocolError(
                    "committed_epoch_record must be null for epoch zero"
                )
        else:
            if not isinstance(committed_epoch_record, dict):
                raise ProtocolError(
                    "committed checkpoint must contain an epoch_record"
                )
            expected_payload = canonical_json_bytes(committed_epoch_record)
            parsed_expected = _parse_metric_line(expected_payload, 1)
            if parsed_expected["epoch"] != committed_epoch:
                raise ProtocolError(
                    "checkpoint epoch_record epoch differs from committed_epoch"
                )
            if len(records) == committed_epoch:
                if canonical_json_bytes(records[-1]) != expected_payload:
                    raise ProtocolError(
                        "metrics committed epoch conflicts with checkpoint record"
                    )
            elif len(records) == committed_epoch - 1:
                records.append(parsed_expected)
                appended = True
            else:
                raise ProtocolError(
                    "metrics contain an unrecoverable earlier epoch gap"
                )
        if partial or appended:
            payload = b"".join(
                canonical_json_bytes(record) + b"\n" for record in records
            )
            _atomic_write_bytes(self.metrics_path, payload)
        return {
            "appended_committed_epoch": appended,
            "committed_epoch": committed_epoch,
            "partial_tail_removed": partial,
            "records": len(records),
        }

    def write_test(self, document: dict[str, object]) -> None:
        write_canonical_json(self.test_path, document)

    def write_summary(
        self,
        document: dict[str, object],
    ) -> dict[str, object]:
        if not self.last_checkpoint.is_file():
            raise ProtocolError("cannot summarize without checkpoints/last.pt")
        if not self.best_checkpoint.is_file():
            raise ProtocolError("cannot summarize without checkpoints/best.pt")
        summary = dict(document)
        summary["last_checkpoint_sha256"] = file_sha256(
            self.last_checkpoint
        )
        summary["best_checkpoint_sha256"] = file_sha256(
            self.best_checkpoint
        )
        write_canonical_json(self.summary_path, summary)
        return summary
