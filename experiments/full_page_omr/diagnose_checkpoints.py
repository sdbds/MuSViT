"""Reproducible checkpoint diagnostics for the Polish Scores validation slice."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import torch


FOUNDATION_MODEL_ID = "carlospm12/LSMT-MAE-Base-1024-16"
FOUNDATION_REVISION = "eecd5b327521225e65e1c2fe38ab99eb667c1609"
DATASET_ID = "antoniorv6/polish-scores"
DATASET_REVISION = "b3170c8b8f322885b566efe9e264af9328b5603f"
ENCODER_UNFREEZE_STEP = 120000
FOUNDATION_EXCLUDED_PARAMETER_PREFIXES = ("pooler.",)
FIXED_VALIDATION_PROTOCOL = {
    "split": "val",
    "rows": list(range(10)),
    "reduce_ratio": 0.5,
    "resolution": 1024,
    "batch_size": 1,
    "decoding": "greedy",
    "precision": "16-mixed",
    "attention_backend": "auto",
    "maxlen": 7512,
}

_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_PATTERN = re.compile(
    r"(?:^|\.)(?:layer|layers|block|blocks)\.(?:\d+)(?=\.|$)"
)
_METRIC_NAMES = ("CER_v2", "SER_v2", "LER_v2")


class DiagnosticError(RuntimeError):
    pass


class ManifestError(DiagnosticError):
    pass


class CheckpointIdentityError(DiagnosticError):
    pass


class EncoderStateError(DiagnosticError):
    pass


def _require_mapping(value, location):
    if not isinstance(value, dict):
        raise ManifestError(f"{location} must be a JSON object")
    return value


def _require_exact_keys(value, expected, location):
    actual = set(value)
    expected = set(expected)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(
            f"{location} keys mismatch; missing={missing}, extra={extra}"
        )


def _require_non_negative_integer(value, location):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{location} must be a non-negative integer")


def _require_non_empty_string(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location} must be a non-empty string")


def _validate_checkpoint_entry(entry, location):
    entry = _require_mapping(entry, location)
    allowed = {"path", "sha256", "epoch", "global_step", "run_id", "source_note"}
    required = {"path", "sha256", "epoch", "global_step"}
    actual = set(entry)
    if not required <= actual or not actual <= allowed:
        raise ManifestError(
            f"{location} keys mismatch; missing={sorted(required - actual)}, "
            f"extra={sorted(actual - allowed)}"
        )

    path = entry["path"]
    _require_non_empty_string(path, f"{location}.path")
    if not Path(path).is_absolute():
        raise ManifestError(f"{location}.path must be absolute")
    if path != str(Path(path).resolve(strict=False)):
        raise ManifestError(f"{location}.path must be normalized")
    sha256 = entry["sha256"]
    if not isinstance(sha256, str) or _HEX_64.fullmatch(sha256) is None:
        raise ManifestError(f"{location}.sha256 must be a lowercase SHA-256 digest")
    _require_non_negative_integer(entry["epoch"], f"{location}.epoch")
    _require_non_negative_integer(entry["global_step"], f"{location}.global_step")

    sources = [key for key in ("run_id", "source_note") if key in entry]
    if not sources:
        raise ManifestError(f"{location} requires run_id or source_note")
    for key in sources:
        _require_non_empty_string(entry[key], f"{location}.{key}")


def _validate_manifest(manifest):
    manifest = _require_mapping(manifest, "manifest")
    _require_exact_keys(
        manifest,
        {"mode", "foundation", "dataset", "validation", "runtime", "checkpoints"},
        "manifest",
    )
    if manifest["mode"] not in {"post-only", "pre-post"}:
        raise ManifestError("manifest.mode must be 'post-only' or 'pre-post'")

    foundation = _require_mapping(manifest["foundation"], "manifest.foundation")
    _require_exact_keys(
        foundation,
        {"model_id", "revision", "encoder_state_sha256"},
        "manifest.foundation",
    )
    if foundation["model_id"] != FOUNDATION_MODEL_ID:
        raise ManifestError(f"foundation model_id must be {FOUNDATION_MODEL_ID!r}")
    if foundation["revision"] != FOUNDATION_REVISION:
        raise ManifestError(f"foundation revision must be {FOUNDATION_REVISION!r}")
    digest = foundation["encoder_state_sha256"]
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        raise ManifestError("foundation encoder_state_sha256 must be a lowercase SHA-256 digest")

    dataset = _require_mapping(manifest["dataset"], "manifest.dataset")
    _require_exact_keys(dataset, {"id", "revision"}, "manifest.dataset")
    if dataset != {"id": DATASET_ID, "revision": DATASET_REVISION}:
        raise ManifestError("dataset identity does not match the locked Polish Scores revision")

    validation = _require_mapping(manifest["validation"], "manifest.validation")
    if validation != FIXED_VALIDATION_PROTOCOL:
        raise ManifestError("validation protocol does not match the locked 10-page protocol")

    runtime = _require_mapping(manifest["runtime"], "manifest.runtime")
    _require_exact_keys(
        runtime,
        {
            "resolved_attention_backend",
            "generation_path",
            "gpu_uuid",
            "software_versions",
        },
        "manifest.runtime",
    )
    if runtime["resolved_attention_backend"] not in {
        "flash_attention_2",
        "sdpa",
        "eager",
    }:
        raise ManifestError("runtime resolved_attention_backend must be concrete")
    if runtime["generation_path"] not in {"full-prefix", "incremental"}:
        raise ManifestError("runtime generation_path is unsupported")
    _require_non_empty_string(runtime["gpu_uuid"], "manifest.runtime.gpu_uuid")
    versions = _require_mapping(
        runtime["software_versions"],
        "manifest.runtime.software_versions",
    )
    if not versions:
        raise ManifestError("manifest.runtime.software_versions must not be empty")
    for name, version in versions.items():
        _require_non_empty_string(name, "software version name")
        _require_non_empty_string(version, f"software version {name!r}")

    checkpoints = _require_mapping(manifest["checkpoints"], "manifest.checkpoints")
    if set(checkpoints) - {"pre", "post"} or "post" not in checkpoints:
        raise ManifestError("manifest.checkpoints must contain post and may contain pre")
    if manifest["mode"] == "pre-post" and "pre" not in checkpoints:
        raise ManifestError("pre-post mode requires a pre checkpoint")
    if manifest["mode"] == "post-only" and "pre" in checkpoints:
        raise ManifestError("post-only mode must omit the pre checkpoint")
    for role, entry in checkpoints.items():
        _validate_checkpoint_entry(entry, f"manifest.checkpoints.{role}")
    if checkpoints["post"]["global_step"] < ENCODER_UNFREEZE_STEP:
        raise ManifestError("post checkpoint must be at or after encoder unfreezing")
    if "pre" in checkpoints:
        pre_step = checkpoints["pre"]["global_step"]
        post_step = checkpoints["post"]["global_step"]
        if pre_step >= ENCODER_UNFREEZE_STEP:
            raise ManifestError("pre checkpoint must precede encoder unfreezing")
        if post_step <= pre_step:
            raise ManifestError("post checkpoint global_step must follow pre")


def load_manifest(manifest_path):
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_manifest(manifest)
    return manifest


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_identity(checkpoint, *, role):
    _validate_checkpoint_entry(checkpoint, f"checkpoint {role}")
    path = Path(checkpoint["path"])
    if not path.is_file():
        raise CheckpointIdentityError(f"{role} checkpoint does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != checkpoint["sha256"]:
        raise CheckpointIdentityError(
            f"{role} checkpoint SHA-256 mismatch: "
            f"declared={checkpoint['sha256']}, actual={actual_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise CheckpointIdentityError(f"{role} checkpoint payload is not a mapping")
    for field in ("epoch", "global_step"):
        actual = payload.get(field)
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise CheckpointIdentityError(
                f"{role} checkpoint has invalid {field}: {actual!r}"
            )
        if actual != checkpoint[field]:
            raise CheckpointIdentityError(
                f"{role} checkpoint {field} mismatch: "
                f"declared={checkpoint[field]}, actual={actual}"
            )

    identity = copy.deepcopy(checkpoint)
    identity["role"] = role
    return identity


def _digest_field(digest, value):
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def snapshot_backed_encoder_state(encoder):
    if not isinstance(encoder, torch.nn.Module):
        raise EncoderStateError("encoder must be a torch module")
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in encoder.named_parameters()
        if not name.startswith(FOUNDATION_EXCLUDED_PARAMETER_PREFIXES)
    }
    if not state:
        raise EncoderStateError("encoder has no snapshot-backed parameters")
    return state


def encoder_state_sha256(state):
    if not isinstance(state, Mapping) or not state:
        raise EncoderStateError("encoder state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise EncoderStateError("encoder state names must be non-empty strings")
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise EncoderStateError(f"encoder state {name!r} is not a tensor")
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise EncoderStateError(f"encoder state {name!r} cannot be serialized")
        tensor = tensor.detach().cpu().contiguous()
        _digest_field(digest, name.encode("utf-8"))
        _digest_field(digest, str(tensor.dtype).encode("ascii"))
        shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        _digest_field(digest, shape)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        _digest_field(digest, raw)
    return digest.hexdigest()


def _block_name(parameter_name):
    match = _BLOCK_PATTERN.search(parameter_name)
    if match is None:
        return None
    return parameter_name[:match.end()].lstrip(".")


def _relative_norm(numerator_squared, denominator_squared, location):
    if denominator_squared == 0.0:
        if numerator_squared == 0.0:
            return 0.0
        raise EncoderStateError(f"{location} has zero reference norm and non-zero drift")
    return math.sqrt(numerator_squared / denominator_squared)


def relative_encoder_drift(reference, candidate):
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        raise EncoderStateError("encoder states must be mappings")
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        raise EncoderStateError(
            "encoder state keys mismatch; "
            f"missing={sorted(reference_keys - candidate_keys)}, "
            f"extra={sorted(candidate_keys - reference_keys)}"
        )
    if not reference_keys:
        raise EncoderStateError("encoder states must not be empty")

    numerator_squared = 0.0
    denominator_squared = 0.0
    block_sums = {}
    for name in sorted(reference_keys):
        reference_tensor = reference[name]
        candidate_tensor = candidate[name]
        if not isinstance(reference_tensor, torch.Tensor) or not isinstance(
            candidate_tensor, torch.Tensor
        ):
            raise EncoderStateError(f"encoder state {name!r} is not tensor-aligned")
        if reference_tensor.shape != candidate_tensor.shape:
            raise EncoderStateError(
                f"encoder state shape mismatch for {name!r}: "
                f"reference={tuple(reference_tensor.shape)}, "
                f"candidate={tuple(candidate_tensor.shape)}"
            )
        if reference_tensor.is_complex() or candidate_tensor.is_complex():
            raise EncoderStateError(f"complex encoder state is unsupported: {name!r}")

        reference_64 = reference_tensor.detach().cpu().to(torch.float64)
        candidate_64 = candidate_tensor.detach().cpu().to(torch.float64)
        if not torch.isfinite(reference_64).all() or not torch.isfinite(candidate_64).all():
            raise EncoderStateError(f"non-finite encoder state value in {name!r}")
        difference = candidate_64 - reference_64
        parameter_numerator = float(torch.sum(difference * difference, dtype=torch.float64))
        parameter_denominator = float(
            torch.sum(reference_64 * reference_64, dtype=torch.float64)
        )
        numerator_squared += parameter_numerator
        denominator_squared += parameter_denominator

        block = _block_name(name)
        if block is not None:
            block_numerator, block_denominator = block_sums.get(block, (0.0, 0.0))
            block_sums[block] = (
                block_numerator + parameter_numerator,
                block_denominator + parameter_denominator,
            )

    return {
        "global": _relative_norm(
            numerator_squared,
            denominator_squared,
            "encoder",
        ),
        "blocks": {
            block: _relative_norm(numerator, denominator, block)
            for block, (numerator, denominator) in sorted(block_sums.items())
        },
    }


def _require_number(value, location, *, non_negative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticError(f"{location} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (non_negative and value < 0):
        raise DiagnosticError(f"{location} must be finite and non-negative")
    return value


def _validate_evaluation(result, manifest):
    if not isinstance(result, dict):
        raise DiagnosticError("checkpoint evaluation must return a mapping")
    required = {
        "pages",
        "aggregate",
        "eos_terminated",
        "truncated",
        "decode_seconds",
        "environment",
    }
    if set(result) != required:
        raise DiagnosticError("checkpoint evaluation fields do not match the report schema")
    if result["environment"] != manifest["runtime"]:
        raise DiagnosticError("checkpoint evaluation environment does not match manifest")

    pages = result["pages"]
    expected_rows = manifest["validation"]["rows"]
    if not isinstance(pages, list) or len(pages) != len(expected_rows):
        raise DiagnosticError("checkpoint evaluation pages do not match validation rows")
    terminated = 0
    truncated = 0
    decode_seconds = 0.0
    for expected_row, page in zip(expected_rows, pages, strict=True):
        if not isinstance(page, dict) or set(page) != {
            "row",
            *_METRIC_NAMES,
            "terminated_by_eos",
            "truncated",
            "decode_seconds",
        }:
            raise DiagnosticError("checkpoint page result fields do not match the schema")
        if page["row"] != expected_row:
            raise DiagnosticError("checkpoint page results are not in locked row order")
        for metric in _METRIC_NAMES:
            _require_number(page[metric], f"row {expected_row} {metric}")
        if not isinstance(page["terminated_by_eos"], bool) or not isinstance(
            page["truncated"], bool
        ):
            raise DiagnosticError("EOS and truncation flags must be booleans")
        if page["terminated_by_eos"] == page["truncated"]:
            raise DiagnosticError("each page must be EOS-terminated or truncated")
        terminated += int(page["terminated_by_eos"])
        truncated += int(page["truncated"])
        decode_seconds += _require_number(
            page["decode_seconds"],
            f"row {expected_row} decode_seconds",
        )

    aggregate = result["aggregate"]
    if not isinstance(aggregate, dict) or set(aggregate) != set(_METRIC_NAMES):
        raise DiagnosticError("aggregate metric fields do not match the schema")
    for metric in _METRIC_NAMES:
        _require_number(aggregate[metric], f"aggregate {metric}")
    if result["eos_terminated"] != terminated or result["truncated"] != truncated:
        raise DiagnosticError("aggregate EOS/truncation counts do not match page results")
    declared_seconds = _require_number(result["decode_seconds"], "decode_seconds")
    if not math.isclose(declared_seconds, decode_seconds, rel_tol=1e-9, abs_tol=1e-9):
        raise DiagnosticError("aggregate decode_seconds does not match page results")
    return copy.deepcopy(result)


def _build_comparison(pre, post):
    pages = []
    for pre_page, post_page in zip(pre["pages"], post["pages"], strict=True):
        if pre_page["row"] != post_page["row"]:
            raise DiagnosticError("pre/post page rows do not align")
        page = {"row": pre_page["row"]}
        for metric in _METRIC_NAMES:
            page[f"{metric}_delta"] = post_page[metric] - pre_page[metric]
        page["decode_seconds_delta"] = (
            post_page["decode_seconds"] - pre_page["decode_seconds"]
        )
        pages.append(page)
    return {
        "aggregate_delta": {
            metric: post["aggregate"][metric] - pre["aggregate"][metric]
            for metric in _METRIC_NAMES
        },
        "eos_terminated_delta": post["eos_terminated"] - pre["eos_terminated"],
        "truncated_delta": post["truncated"] - pre["truncated"],
        "decode_seconds_delta": post["decode_seconds"] - pre["decode_seconds"],
        "pages": pages,
    }


def run_diagnostics(
    manifest,
    *,
    load_foundation_encoder: Callable[[dict], Mapping[str, torch.Tensor]],
    load_checkpoint_encoder: Callable[[dict, dict], Mapping[str, torch.Tensor]],
    evaluate_checkpoint: Callable[[dict, dict], dict],
):
    _validate_manifest(manifest)
    roles = ["post"] if manifest["mode"] == "post-only" else ["pre", "post"]

    identities = {
        role: verify_checkpoint_identity(manifest["checkpoints"][role], role=role)
        for role in roles
    }
    reference = load_foundation_encoder(manifest)
    actual_foundation_digest = encoder_state_sha256(reference)
    declared_foundation_digest = manifest["foundation"]["encoder_state_sha256"]
    if actual_foundation_digest != declared_foundation_digest:
        raise EncoderStateError(
            "foundation encoder state digest mismatch: "
            f"declared={declared_foundation_digest}, actual={actual_foundation_digest}"
        )

    drifts = {}
    evaluations = {}
    for role in roles:
        candidate = load_checkpoint_encoder(identities[role], manifest)
        drifts[role] = relative_encoder_drift(reference, candidate)
        evaluations[role] = _validate_evaluation(
            evaluate_checkpoint(identities[role], manifest),
            manifest,
        )

    report = {
        "status": "partial" if manifest["mode"] == "post-only" else "complete",
        "manifest": copy.deepcopy(manifest),
        "foundation": {
            "model_id": manifest["foundation"]["model_id"],
            "revision": manifest["foundation"]["revision"],
            "encoder_state_sha256": actual_foundation_digest,
            "excluded_parameter_prefixes": list(
                FOUNDATION_EXCLUDED_PARAMETER_PREFIXES
            ),
        },
        "checkpoints": identities,
        "evaluations": evaluations,
        "encoder_drift": drifts,
    }
    if manifest["mode"] == "pre-post":
        report["comparison"] = _build_comparison(
            evaluations["pre"],
            evaluations["post"],
        )
    else:
        report["limitations"] = [
            "pre-unfreeze checkpoint was not supplied; pre/post validation delta is unavailable"
        ]
    return report


def main(manifest_path, output_path):
    manifest = load_manifest(manifest_path)
    runtime = _DefaultDiagnosticRuntime()
    report = run_diagnostics(
        manifest,
        load_foundation_encoder=runtime.load_foundation_encoder,
        load_checkpoint_encoder=runtime.load_checkpoint_encoder,
        evaluate_checkpoint=runtime.evaluate_checkpoint,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


class _DefaultDiagnosticRuntime:
    def __init__(self):
        self._models = {}
        self._snapshot = None
        self._rows = None

    @staticmethod
    def _download_foundation_snapshot(snapshot_download):
        kwargs = {
            "repo_id": FOUNDATION_MODEL_ID,
            "revision": FOUNDATION_REVISION,
            "allow_patterns": ["config.json", "model.safetensors"],
        }
        try:
            return snapshot_download(**kwargs, local_files_only=True)
        except Exception:
            try:
                return snapshot_download(**kwargs)
            except Exception as exc:
                raise DiagnosticError(
                    "locked foundation revision is unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def _snapshot_path(self, manifest):
        if (
            manifest["foundation"]["model_id"] != FOUNDATION_MODEL_ID
            or manifest["foundation"]["revision"] != FOUNDATION_REVISION
        ):
            raise DiagnosticError("manifest foundation identity is not locked")
        if self._snapshot is None:
            from huggingface_hub import snapshot_download

            self._snapshot = self._download_foundation_snapshot(snapshot_download)
        return self._snapshot

    def load_foundation_encoder(self, manifest):
        from transformers import ViTModel

        encoder = ViTModel.from_pretrained(
            self._snapshot_path(manifest),
            mask_ratio=0.0,
            add_pooling_layer=False,
        )
        state = snapshot_backed_encoder_state(encoder)
        del encoder
        return state

    @staticmethod
    def _checkpoint_config(payload, manifest, snapshot_path):
        from .smt_foundation.configuration_smt import SMTFoundationConfig

        hyper_parameters = payload.get("hyper_parameters")
        if not isinstance(hyper_parameters, Mapping) or "smt_config" not in hyper_parameters:
            raise DiagnosticError("checkpoint is missing hyper_parameters.smt_config")
        raw_config = hyper_parameters["smt_config"]
        if isinstance(raw_config, SMTFoundationConfig):
            config = copy.deepcopy(raw_config)
        elif isinstance(raw_config, Mapping):
            config = SMTFoundationConfig(**dict(raw_config))
        elif hasattr(raw_config, "to_dict"):
            config = SMTFoundationConfig(**raw_config.to_dict())
        else:
            raise DiagnosticError("checkpoint smt_config has an unsupported type")

        if config.foundation_weights != manifest["foundation"]["model_id"]:
            raise DiagnosticError(
                "checkpoint foundation_weights does not match the manifest model id"
            )
        if config.maxlen != manifest["validation"]["maxlen"]:
            raise DiagnosticError("checkpoint maxlen does not match the validation protocol")
        if config.padding_token != 0:
            raise DiagnosticError("checkpoint padding token does not match the validation protocol")
        if not isinstance(config.w2i, dict) or not isinstance(config.i2w, dict):
            raise DiagnosticError("checkpoint smt_config is missing its vocabulary")
        config.foundation_weights = snapshot_path
        config.attention_backend = manifest["validation"]["attention_backend"]
        return config

    @staticmethod
    def _model_state(payload):
        state = payload.get("state_dict")
        if not isinstance(state, Mapping) or not state:
            raise DiagnosticError("checkpoint is missing a model state_dict")
        if all(name.startswith("model.") for name in state):
            return {name.removeprefix("model."): tensor for name, tensor in state.items()}
        return dict(state)

    def load_checkpoint_encoder(self, identity, manifest):
        from .smt_foundation.modeling_smt import SMTFoundationModelForCausalLM

        payload = torch.load(
            identity["path"],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        config = self._checkpoint_config(
            payload,
            manifest,
            self._snapshot_path(manifest),
        )
        model = SMTFoundationModelForCausalLM(config)
        model.load_state_dict(self._model_state(payload), strict=True)
        model.eval()
        state = snapshot_backed_encoder_state(model.encoder)
        if not torch.cuda.is_available():
            raise DiagnosticError("checkpoint validation requires CUDA")
        model.to(torch.device("cuda", 0))
        self._models[identity["role"]] = model
        return state

    @staticmethod
    def _software_versions():
        def package_version(name):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return "not-installed"

        try:
            repository = Path(__file__).resolve().parents[2]
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "unavailable"
        return {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "lightning": package_version("lightning"),
            "git_commit": git_commit,
        }

    @staticmethod
    def _visible_gpu_uuid():
        try:
            raw_uuid = torch.cuda.get_device_properties(0).uuid
        except (AssertionError, RuntimeError) as exc:
            raise DiagnosticError("could not query CUDA device 0 UUID") from exc
        if isinstance(raw_uuid, bytes):
            raw_uuid = raw_uuid.decode("ascii")
        elif not isinstance(raw_uuid, str):
            raw_uuid = str(raw_uuid)
        if not raw_uuid.strip():
            raise DiagnosticError("CUDA device 0 returned an invalid UUID")
        raw_uuid = raw_uuid.strip()
        return raw_uuid if raw_uuid.startswith("GPU-") else f"GPU-{raw_uuid}"

    def _runtime_identity(self, manifest, resolved_backend):
        identity = {
            "resolved_attention_backend": resolved_backend,
            "generation_path": manifest["runtime"]["generation_path"],
            "gpu_uuid": self._visible_gpu_uuid(),
            "software_versions": self._software_versions(),
        }
        if identity != manifest["runtime"]:
            raise DiagnosticError(
                "actual runtime identity does not match the diagnostic manifest"
            )
        return identity

    def _dataset_rows(self, manifest):
        if self._rows is None:
            from datasets import load_dataset

            self._rows = load_dataset(
                manifest["dataset"]["id"],
                revision=manifest["dataset"]["revision"],
                split=manifest["validation"]["split"],
                keep_in_memory=False,
            )
        return self._rows

    @staticmethod
    def _resolved_backend(model):
        backends = {
            attention.last_backend
            for layer in model.decoder.decoder.layers
            for attention in (layer.input_attention, layer.cross_attention)
            if attention.last_backend is not None
        }
        if len(backends) != 1:
            raise DiagnosticError(
                f"decoder did not resolve one attention backend: {sorted(backends)}"
            )
        return next(iter(backends))

    def evaluate_checkpoint(self, identity, manifest):
        import cv2
        import numpy as np

        from . import _globals
        from .data import parse_kern_file
        from .data_augmentation.data_augmentation import convert_img_to_tensor
        from .eval.eval_functions import (
            canonicalize_prediction_ids,
            canonicalize_target_ids,
            compute_canonical_metrics,
        )

        role = identity["role"]
        if role not in self._models:
            raise DiagnosticError(f"checkpoint model for {role} was not prepared")
        model = self._models.pop(role)
        rows = self._dataset_rows(manifest)
        protocol = manifest["validation"]
        _globals.resolution = protocol["resolution"]
        use_incremental = manifest["runtime"]["generation_path"] == "incremental"
        device = torch.device("cuda", 0)
        predictions = []
        targets = []
        pages = []

        try:
            actual_gpu_uuid = self._visible_gpu_uuid()
            if actual_gpu_uuid != manifest["runtime"]["gpu_uuid"]:
                raise DiagnosticError("actual GPU UUID does not match the diagnostic manifest")
            if self._software_versions() != manifest["runtime"]["software_versions"]:
                raise DiagnosticError(
                    "actual software versions do not match the diagnostic manifest"
                )

            for row_index in protocol["rows"]:
                sample = rows[row_index]
                image = np.asarray(sample["image"])
                width = int(np.ceil(image.shape[1] * protocol["reduce_ratio"]))
                height = int(np.ceil(image.shape[0] * protocol["reduce_ratio"]))
                image = cv2.resize(image, (width, height))
                input_tensor = convert_img_to_tensor(image).to(device)

                target_tokens = [
                    "<bos>",
                    *parse_kern_file(sample["transcription"], tokenization_mode="bekern"),
                    "<eos>",
                ]
                try:
                    target_ids = [model.w2i[token] for token in target_tokens]
                except KeyError as exc:
                    raise DiagnosticError(
                        f"validation row {row_index} contains unknown token {exc.args[0]!r}"
                    ) from None

                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generation = model.generate_token_ids(
                        input_tensor,
                        use_incremental=use_incremental,
                    )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started

                prediction = canonicalize_prediction_ids(
                    generation.token_ids,
                    model.i2w,
                    maxlen=model.maxlen,
                )
                target = canonicalize_target_ids(target_ids, model.i2w)
                predictions.append(prediction)
                targets.append(target)
                cer, ser, ler = compute_canonical_metrics([prediction], [target])
                pages.append(
                    {
                        "row": row_index,
                        "CER_v2": cer,
                        "SER_v2": ser,
                        "LER_v2": ler,
                        "terminated_by_eos": generation.terminated_by_eos,
                        "truncated": generation.truncated,
                        "decode_seconds": elapsed,
                    }
                )

                resolved_backend = self._resolved_backend(model)
                if resolved_backend != manifest["runtime"]["resolved_attention_backend"]:
                    raise DiagnosticError(
                        "resolved attention backend does not match the diagnostic manifest"
                    )

            cer, ser, ler = compute_canonical_metrics(predictions, targets)
            return {
                "pages": pages,
                "aggregate": {"CER_v2": cer, "SER_v2": ser, "LER_v2": ler},
                "eos_terminated": sum(page["terminated_by_eos"] for page in pages),
                "truncated": sum(page["truncated"] for page in pages),
                "decode_seconds": sum(page["decode_seconds"] for page in pages),
                "environment": self._runtime_identity(
                    manifest,
                    self._resolved_backend(model),
                ),
            }
        finally:
            model.cpu()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
