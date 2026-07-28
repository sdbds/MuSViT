"""Canonical JSON primitives shared by staff-level OMR protocol artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import ProtocolError


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value using the protocol's stable byte representation."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise ProtocolError(
            f"canonical JSON numbers must be finite: {exc}"
        ) from exc
    except TypeError as exc:
        raise ProtocolError(
            f"value is not representable as canonical JSON: {exc}"
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of a value's canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON while rejecting duplicate object keys."""
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ProtocolError(f"invalid UTF-8 JSON in {source}: {exc}") from exc
    except OSError as exc:
        raise ProtocolError(f"cannot read JSON file {source}: {exc}") from exc

    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except ProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid UTF-8 JSON in {source}: {exc}") from exc


def write_canonical_json(path: str | Path, value: Any) -> None:
    """Atomically write canonical JSON without a trailing newline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid4().hex}"
    )
    payload = canonical_json_bytes(value)

    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ProtocolError(
            f"cannot atomically write JSON file {destination}: {exc}"
        ) from exc
