from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

from .smt_foundation.configuration_smt import SMTFoundationConfig
from .smt_foundation.modeling_smt import MHA, SMTFoundationModelForCausalLM


DEFAULT_OPSET_VERSION = 20
MAX_ONNX_BYTES = 2 * 1024**3
_EXPERIMENT_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _EXPERIMENT_DIR / "weights"
DEFAULT_WEIGHTS_PATH = _WEIGHTS_DIR / "polish_scores_cl_CL.safetensors"
DEFAULT_MODEL_CONFIG_PATH = _WEIGHTS_DIR / "polish_scores_cl_CL.config.json"
DEFAULT_ENCODER_CONFIG_PATH = _WEIGHTS_DIR / "polish_scores_cl_CL.encoder_config.json"
DEFAULT_OUTPUT_DIR = _WEIGHTS_DIR / "polish_scores_cl_CL_onnx"
VALIDATION_DATASET_ID = "antoniorv6/polish-scores"
VALIDATION_DATASET_REVISION = "b3170c8b8f322885b566efe9e264af9328b5603f"
VALIDATION_DATASET_SPLIT = "val"
VALIDATION_DATASET_ROW = 0
VALIDATION_REDUCE_RATIO = 0.5


@dataclass(frozen=True)
class ExportPaths:
    output_dir: Path
    encoder: Path
    decoder: Path
    config: Path
    preprocessor_config: Path
    metadata: Path

    @classmethod
    def from_output_dir(cls, output_dir: str | Path) -> "ExportPaths":
        resolved = Path(output_dir).resolve()
        return cls(
            output_dir=resolved,
            encoder=resolved / "encoder.onnx",
            decoder=resolved / "decoder.onnx",
            config=resolved / "config.json",
            preprocessor_config=resolved / "preprocessor_config.json",
            metadata=resolved / "metadata.json",
        )


def _read_json_object(path: str | Path, *, label: str) -> dict:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {resolved}")
    return value


def load_standalone_model(
    weights_path: str | Path,
    model_config_path: str | Path,
    encoder_config_path: str | Path,
) -> tuple[SMTFoundationModelForCausalLM, dict]:
    model_config = _read_json_object(model_config_path, label="model config")
    encoder_config = _read_json_object(encoder_config_path, label="encoder config")
    merged_config = dict(model_config)
    merged_config["foundation_config"] = encoder_config
    merged_config["attention_backend"] = "eager"

    config = SMTFoundationConfig(**merged_config)
    model = SMTFoundationModelForCausalLM(config)
    state = load_file(str(Path(weights_path).resolve()), device="cpu")
    model.load_state_dict(state, strict=True)
    for module in model.modules():
        if isinstance(module, MHA):
            module.attention_backend = "eager"
    model.eval()
    return model, config.to_dict()


class FullPageOMREncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.model.forward_encoder(pixel_values)
        prepared = self.model._prepare_decoder_features(
            encoder_output.permute(0, 2, 1).contiguous()
        )
        return (
            prepared.raw_features.permute(1, 0, 2).contiguous(),
            prepared.enhanced_features.permute(1, 0, 2).contiguous(),
        )


class FullPageOMRDecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        raw_features: torch.Tensor,
        enhanced_features: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        raw_sequence = raw_features.permute(1, 0, 2).contiguous()
        enhanced_sequence = enhanced_features.permute(1, 0, 2).contiguous()
        positioned = self.decoder.embedding(token_ids).permute(0, 2, 1)
        positioned = self.decoder.positional_1D(positioned, start=0)
        positioned = positioned.permute(2, 0, 1).contiguous()
        output, _, _ = self.decoder.decoder(
            positioned,
            memory_key=enhanced_sequence,
            memory_value=raw_sequence,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            use_cache=False,
            cache=None,
            predict_last_n_only=False,
            keep_all_weights=False,
            self_attention_is_causal=True,
            self_attention_window=(-1, -1),
        )
        projected = self.decoder.dropout(self.decoder.end_relu(output[-1:]))
        return self.decoder.out_layer(
            projected.permute(1, 2, 0).contiguous()
        ).squeeze(-1)


def _validate_onnx_graph(path: Path) -> None:
    import onnx

    onnx.checker.check_model(str(path))


def _export_graph_atomic(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: str | Path,
    *,
    device: torch.device,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    opset_version: int,
) -> Path:
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    module = module.to(device).eval()
    device_args = tuple(value.to(device) for value in args)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                module,
                device_args,
                str(temporary),
                export_params=True,
                opset_version=opset_version,
                dynamo=False,
                external_data=False,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        _validate_onnx_graph(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def export_encoder_graph(
    wrapper: torch.nn.Module,
    output_path: str | Path,
    device: torch.device,
    *,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    pixels = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    return _export_graph_atomic(
        wrapper,
        (pixels,),
        output_path,
        device=device,
        input_names=["pixel_values"],
        output_names=["raw_features", "enhanced_features"],
        dynamic_axes=None,
        opset_version=opset_version,
    )


def export_decoder_graph(
    wrapper: torch.nn.Module,
    output_path: str | Path,
    device: torch.device,
    *,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    raw_features = torch.zeros((1, 4096, 256), dtype=torch.float32)
    enhanced_features = torch.zeros((1, 4096, 256), dtype=torch.float32)
    token_ids = torch.tensor([[100]], dtype=torch.long)
    return _export_graph_atomic(
        wrapper,
        (raw_features, enhanced_features, token_ids),
        output_path,
        device=device,
        input_names=["raw_features", "enhanced_features", "token_ids"],
        output_names=["next_token_logits"],
        dynamic_axes={"token_ids": {1: "sequence_length"}},
        opset_version=opset_version,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_preprocessor_config() -> dict[str, Any]:
    return {
        "color": "RGB",
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_size": [1024, 1024],
        "input_layout": "NCHW",
        "interpolation": "bilinear",
        "rescale_factor": 1 / 255,
    }


def build_bundle_metadata(
    *,
    paths: ExportPaths,
    source: Mapping[str, Any],
    versions: Mapping[str, str],
    providers: list[str],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {}
    for label, path in (
        ("encoder", paths.encoder),
        ("decoder", paths.decoder),
        ("config", paths.config),
        ("preprocessor_config", paths.preprocessor_config),
    ):
        if path.is_file():
            artifacts[label] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "artifacts": artifacts,
        "bundle_status": "complete",
        "format_version": 1,
        "model_type": "full_page_omr",
        "files": {
            "encoder": paths.encoder.name,
            "decoder": paths.decoder.name,
            "config": paths.config.name,
            "preprocessor_config": paths.preprocessor_config.name,
        },
        "graphs": {
            "encoder": {
                "inputs": {"pixel_values": [1, 3, 1024, 1024]},
                "outputs": {
                    "raw_features": [1, 4096, 256],
                    "enhanced_features": [1, 4096, 256],
                },
            },
            "decoder": {
                "inputs": {
                    "raw_features": [1, 4096, 256],
                    "enhanced_features": [1, 4096, 256],
                    "token_ids": [1, "sequence_length"],
                },
                "outputs": {"next_token_logits": [1, 215]},
            },
        },
        "opset_version": DEFAULT_OPSET_VERSION,
        "precision": "float32",
        "providers": list(providers),
        "source": dict(source),
        "tokens": {"bos": 100, "eos": 183, "max_length": 7512},
        "validation": dict(validation),
        "versions": dict(versions),
    }


def validate_graph_files(
    paths: ExportPaths,
    *,
    max_onnx_bytes: int = MAX_ONNX_BYTES,
) -> dict[str, int]:
    sizes = {}
    for label, path in (("encoder", paths.encoder), ("decoder", paths.decoder)):
        if not path.is_file():
            raise RuntimeError(f"{label} ONNX graph is missing: {path}")
        size = path.stat().st_size
        if size == 0:
            raise RuntimeError(f"{label} ONNX graph is empty: {path}")
        if size >= max_onnx_bytes:
            raise RuntimeError(f"{label} ONNX graph exceeds the size limit: {size}")
        sizes[label] = size

    combined = sum(sizes.values())
    if combined >= max_onnx_bytes:
        raise RuntimeError(f"combined ONNX graphs exceed the size limit: {combined}")

    sidecars = sorted(paths.output_dir.glob("*.onnx.data*"))
    if sidecars:
        raise RuntimeError(
            "external tensor data is not permitted: "
            + ", ".join(path.name for path in sidecars)
        )
    temporary_files = sorted(paths.output_dir.glob("*.tmp"))
    if temporary_files:
        raise RuntimeError(
            "temporary export files remain: "
            + ", ".join(path.name for path in temporary_files)
        )
    sizes["combined"] = combined
    return sizes


def compare_arrays(
    label: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, Any]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        raise AssertionError(
            f"{label} shape mismatch: {expected.shape} != {actual.shape}"
        )
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"{label} is outside rtol={rtol}, atol={atol}: {exc}"
        ) from exc
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    relative = absolute / denominator
    return {
        "shape": list(expected.shape),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "max_rel_error": float(relative.max(initial=0.0)),
    }


def _decode_token_ids(
    token_ids: tuple[int, ...],
    *,
    i2w: Mapping[int, str],
    eos_token_id: int,
) -> tuple[str, ...]:
    decoded = []
    for token_id in token_ids[1:]:
        if token_id == eos_token_id:
            break
        try:
            decoded.append(i2w[token_id])
        except KeyError:
            raise KeyError(f"Unknown predicted token id {token_id}") from None
    return tuple(decoded)


def compare_greedy_results(
    reference,
    actual,
    *,
    i2w: Mapping[int, str],
    eos_token_id: int,
) -> dict[str, Any]:
    reference_ids = tuple(int(token_id) for token_id in reference.token_ids)
    actual_ids = tuple(int(token_id) for token_id in actual.token_ids)
    if reference_ids != actual_ids:
        common_length = min(len(reference_ids), len(actual_ids))
        mismatch_index = next(
            (
                index
                for index in range(common_length)
                if reference_ids[index] != actual_ids[index]
            ),
            common_length,
        )
        expected = (
            reference_ids[mismatch_index]
            if mismatch_index < len(reference_ids)
            else None
        )
        observed = (
            actual_ids[mismatch_index]
            if mismatch_index < len(actual_ids)
            else None
        )
        raise AssertionError(
            "greedy token-id sequence mismatch at index "
            f"{mismatch_index}: {observed} != {expected}; "
            f"lengths {len(actual_ids)} != {len(reference_ids)}"
        )

    if actual.terminated_by_eos != reference.terminated_by_eos:
        raise AssertionError("greedy EOS termination state mismatch")
    if actual.truncated != reference.truncated:
        raise AssertionError("greedy truncation state mismatch")

    reference_tokens = _decode_token_ids(
        reference_ids,
        i2w=i2w,
        eos_token_id=eos_token_id,
    )
    actual_tokens = tuple(actual.tokens)
    if actual_tokens != reference_tokens:
        raise AssertionError("greedy decoded token sequence mismatch")

    serialized_ids = json.dumps(
        list(reference_ids),
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "status": "passed",
        "token_count": len(reference_ids),
        "decoded_token_count": len(reference_tokens),
        "token_ids_sha256": hashlib.sha256(serialized_ids).hexdigest(),
        "terminated_by_eos": bool(reference.terminated_by_eos),
        "truncated": bool(reference.truncated),
    }


@contextmanager
def exact_fp32():
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32


def _verification_providers(device: torch.device) -> list[str]:
    if device.type == "cuda":
        return ["CUDAExecutionProvider"]
    return ["CPUExecutionProvider"]


def _require_verification_provider(runtime, provider: str) -> None:
    encoder_providers = list(runtime.encoder_session.get_providers())
    decoder_providers = list(runtime.decoder_session.get_providers())
    if provider not in encoder_providers or provider not in decoder_providers:
        raise RuntimeError(
            f"Required {provider} did not initialize for both ONNX Runtime sessions; "
            f"encoder={encoder_providers}, decoder={decoder_providers}"
        )


def verify_runtime_parity(
    model: SMTFoundationModelForCausalLM,
    paths: ExportPaths,
    device: torch.device,
) -> dict[str, Any]:
    with exact_fp32():
        return _verify_runtime_parity_exact(model, paths, device)


def _verify_runtime_parity_exact(
    model: SMTFoundationModelForCausalLM,
    paths: ExportPaths,
    device: torch.device,
) -> dict[str, Any]:
    from .onnx_runtime import FullPageOMROnnxRuntime

    verification_provider = _verification_providers(device)[0]
    runtime = FullPageOMROnnxRuntime(
        paths.output_dir,
        providers=[verification_provider],
    )
    _require_verification_provider(runtime, verification_provider)
    rng = np.random.default_rng(20260718)
    pixels = rng.random((1, 3, 1024, 1024), dtype=np.float32)
    torch_pixels = torch.from_numpy(pixels).to(device)
    encoder = FullPageOMREncoderWrapper(model).to(device).eval()
    decoder = FullPageOMRDecoderWrapper(model).to(device).eval()
    with torch.inference_mode():
        expected_raw, expected_enhanced = encoder(torch_pixels)
    expected_raw_array = expected_raw.detach().cpu().numpy()
    expected_enhanced_array = expected_enhanced.detach().cpu().numpy()
    actual_raw, actual_enhanced = runtime.encode_pixel_values(pixels)
    comparisons = {
        "raw_features": compare_arrays(
            "raw_features",
            expected_raw_array,
            actual_raw,
        ),
        "enhanced_features": compare_arrays(
            "enhanced_features",
            expected_enhanced_array,
            actual_enhanced,
        ),
    }

    prefixes = (
        (100,),
        (100, 10, 20, 30),
        (100, 1, 5, 9, 13, 17, 21, 25),
    )
    checked_prefixes = []
    for prefix in prefixes:
        token_array = np.asarray([prefix], dtype=np.int64)
        with torch.inference_mode():
            expected_logits = decoder(
                expected_raw,
                expected_enhanced,
                torch.from_numpy(token_array).to(device),
            )
        expected_logits_array = expected_logits.detach().cpu().numpy()
        actual_logits = runtime.next_token_logits(
            expected_raw_array,
            expected_enhanced_array,
            token_array,
        )
        metrics = compare_arrays(
            f"decoder_logits_length_{len(prefix)}",
            expected_logits_array,
            actual_logits,
        )
        expected_argmax = int(np.argmax(expected_logits_array[0]))
        actual_argmax = int(np.argmax(actual_logits[0]))
        if actual_argmax != expected_argmax:
            raise AssertionError(
                f"decoder argmax mismatch at prefix length {len(prefix)}: "
                f"{actual_argmax} != {expected_argmax}"
            )
        checked_prefixes.append(
            {
                "length": len(prefix),
                "argmax": actual_argmax,
                **metrics,
            }
        )

    with torch.inference_mode():
        pipeline_expected = decoder(
            expected_raw,
            expected_enhanced,
            torch.tensor([[100]], dtype=torch.long, device=device),
        )
    pipeline_actual = runtime.next_token_logits(
        actual_raw,
        actual_enhanced,
        np.asarray([[100]], dtype=np.int64),
    )
    comparisons["pipeline_bos_logits"] = compare_arrays(
        "pipeline_bos_logits",
        pipeline_expected.detach().cpu().numpy(),
        pipeline_actual,
    )
    cpu_runtime = FullPageOMROnnxRuntime(
        paths.output_dir,
        providers=["CPUExecutionProvider"],
    )
    _require_verification_provider(cpu_runtime, "CPUExecutionProvider")
    cpu_raw, cpu_enhanced = cpu_runtime.encode_pixel_values(pixels)
    if (
        cpu_raw.shape != (1, 4096, 256)
        or cpu_enhanced.shape != (1, 4096, 256)
        or not np.isfinite(cpu_raw).all()
        or not np.isfinite(cpu_enhanced).all()
    ):
        raise AssertionError("CPU encoder smoke output contract failed")
    cpu_logits = cpu_runtime.next_token_logits(
        cpu_raw,
        cpu_enhanced,
        np.asarray([[100]], dtype=np.int64),
    )
    if cpu_logits.shape != (1, 215) or not np.isfinite(cpu_logits).all():
        raise AssertionError(
            "CPU decoder smoke output must be finite float32 logits shaped [1, 215]"
        )
    return {
        "status": "passed",
        "atol": 1e-4,
        "rtol": 1e-4,
        "providers": runtime.providers,
        "comparisons": comparisons,
        "checked_prefixes": checked_prefixes,
        "cpu_smoke": {
            "status": "passed",
            "encoder_providers": list(cpu_runtime.encoder_session.get_providers()),
            "decoder_providers": list(cpu_runtime.decoder_session.get_providers()),
            "raw_features_shape": list(cpu_raw.shape),
            "enhanced_features_shape": list(cpu_enhanced.shape),
            "decoder_output_shape": list(cpu_logits.shape),
        },
    }


def _load_locked_validation_image() -> np.ndarray:
    import cv2
    from datasets import load_dataset

    try:
        rows = load_dataset(
            VALIDATION_DATASET_ID,
            revision=VALIDATION_DATASET_REVISION,
            split=VALIDATION_DATASET_SPLIT,
            keep_in_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "locked validation dataset is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if len(rows) <= VALIDATION_DATASET_ROW:
        raise RuntimeError(
            f"locked validation dataset does not contain row {VALIDATION_DATASET_ROW}"
        )

    image = np.asarray(rows[VALIDATION_DATASET_ROW]["image"])
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise RuntimeError("locked validation row contains an invalid image")
    width = int(np.ceil(image.shape[1] * VALIDATION_REDUCE_RATIO))
    height = int(np.ceil(image.shape[0] * VALIDATION_REDUCE_RATIO))
    return cv2.resize(image, (width, height))


def verify_dataset_parity(
    model: SMTFoundationModelForCausalLM,
    paths: ExportPaths,
    device: torch.device,
) -> dict[str, Any]:
    from . import _globals
    from .data_augmentation.data_augmentation import convert_img_to_tensor
    from .onnx_runtime import FullPageOMROnnxRuntime, preprocess_page

    image = _load_locked_validation_image()
    pixels = preprocess_page(image, build_preprocessor_config())
    previous_resolution = _globals.resolution
    try:
        _globals.resolution = 1024
        reference_pixels = convert_img_to_tensor(image).numpy()
    finally:
        _globals.resolution = previous_resolution
    if not np.array_equal(pixels, reference_pixels):
        difference = np.abs(
            pixels.astype(np.float64) - reference_pixels.astype(np.float64)
        )
        raise AssertionError(
            "ONNX preprocessing differs from convert_img_to_tensor: "
            f"max_abs_error={float(difference.max(initial=0.0))}"
        )

    verification_provider = _verification_providers(device)[0]
    runtime = FullPageOMROnnxRuntime(
        paths.output_dir,
        providers=[verification_provider],
    )
    _require_verification_provider(runtime, verification_provider)
    model = model.to(device).eval()
    torch_pixels = torch.from_numpy(pixels).to(device)
    with exact_fp32():
        started = time.perf_counter()
        with torch.inference_mode():
            reference = model.generate_token_ids(
                torch_pixels,
                use_incremental=False,
            )
        pytorch_seconds = time.perf_counter() - started

        started = time.perf_counter()
        actual = runtime.generate_pixel_values(pixels)
        onnxruntime_seconds = time.perf_counter() - started

    report = compare_greedy_results(
        reference,
        actual,
        i2w=model.i2w,
        eos_token_id=runtime.eos_token_id,
    )
    report.update(
        {
            "dataset": VALIDATION_DATASET_ID,
            "revision": VALIDATION_DATASET_REVISION,
            "split": VALIDATION_DATASET_SPLIT,
            "row": VALIDATION_DATASET_ROW,
            "reduce_ratio": VALIDATION_REDUCE_RATIO,
            "preprocessing_exact": True,
            "providers": runtime.providers,
            "pytorch_seconds": pytorch_seconds,
            "onnxruntime_seconds": onnxruntime_seconds,
        }
    )
    return report


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_versions_and_providers() -> tuple[dict[str, str], list[str]]:
    versions = {
        "onnx": _package_version("onnx"),
        "onnxruntime": _package_version("onnxruntime-gpu"),
        "safetensors": _package_version("safetensors"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }
    try:
        import onnxruntime as ort
    except ImportError:
        providers = []
    else:
        providers = list(ort.get_available_providers())
    return versions, providers


def _publish_bundle_directory(staged_dir: Path, output_dir: Path) -> None:
    if not staged_dir.is_dir():
        raise RuntimeError(f"staged bundle directory is missing: {staged_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = None
    if output_dir.exists():
        backup_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.backup-",
                dir=output_dir.parent,
            )
        )
        backup_dir.rmdir()
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staged_dir, output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


def export_bundle(
    *,
    weights_path: str | Path,
    model_config_path: str | Path,
    encoder_config_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    opset_version: int = DEFAULT_OPSET_VERSION,
    verify_runtime: bool = False,
    verify_dataset: bool = False,
) -> ExportPaths:
    paths = ExportPaths.from_output_dir(output_dir)
    model, standalone_config = load_standalone_model(
        weights_path,
        model_config_path,
        encoder_config_path,
    )
    versions, providers = _runtime_versions_and_providers()
    source = {
        "weights_path": str(Path(weights_path).resolve()),
        "weights_sha256": sha256_file(weights_path),
        "model_config_path": str(Path(model_config_path).resolve()),
        "model_config_sha256": sha256_file(model_config_path),
        "encoder_config_path": str(Path(encoder_config_path).resolve()),
        "encoder_config_sha256": sha256_file(encoder_config_path),
    }
    paths.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.output_dir.name}.staging-",
            dir=paths.output_dir.parent,
        )
    ).resolve()
    staged_paths = ExportPaths.from_output_dir(staging_dir)
    try:
        export_encoder_graph(
            FullPageOMREncoderWrapper(model),
            staged_paths.encoder,
            device,
            opset_version=opset_version,
        )
        export_decoder_graph(
            FullPageOMRDecoderWrapper(model),
            staged_paths.decoder,
            device,
            opset_version=opset_version,
        )
        validate_graph_files(staged_paths)
        write_json_atomic(staged_paths.config, standalone_config)
        write_json_atomic(
            staged_paths.preprocessor_config,
            build_preprocessor_config(),
        )
        metadata = build_bundle_metadata(
            paths=staged_paths,
            source=source,
            versions=versions,
            providers=providers,
            validation={"status": "pending"},
        )
        metadata["opset_version"] = opset_version
        write_json_atomic(staged_paths.metadata, metadata)

        validation = (
            verify_runtime_parity(model, staged_paths, device)
            if verify_runtime or verify_dataset
            else {"status": "not_run"}
        )
        if verify_dataset:
            validation = {
                **validation,
                "dataset_parity": verify_dataset_parity(
                    model,
                    staged_paths,
                    device,
                ),
            }
        metadata["validation"] = validation
        write_json_atomic(staged_paths.metadata, metadata)
        _publish_bundle_directory(staging_dir, paths.output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return paths


def determine_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the standalone full-page OMR model to two ONNX graphs."
    )
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--model-config-path",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
    )
    parser.add_argument(
        "--encoder-config-path",
        type=Path,
        default=DEFAULT_ENCODER_CONFIG_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_OPSET_VERSION,
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Compare deterministic PyTorch and ONNX Runtime tensors after export.",
    )
    parser.add_argument(
        "--verify-dataset",
        action="store_true",
        help="Require exact greedy parity on the locked Polish Scores validation page.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = determine_device(args.device)
    plan = {
        "weights_path": str(args.weights_path.resolve()),
        "model_config_path": str(args.model_config_path.resolve()),
        "encoder_config_path": str(args.encoder_config_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "opset_version": args.opset_version,
        "verify_runtime": args.verify_runtime or args.verify_dataset,
        "verify_dataset": args.verify_dataset,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    paths = export_bundle(
        weights_path=args.weights_path,
        model_config_path=args.model_config_path,
        encoder_config_path=args.encoder_config_path,
        output_dir=args.output_dir,
        device=device,
        opset_version=args.opset_version,
        verify_runtime=args.verify_runtime,
        verify_dataset=args.verify_dataset,
    )
    print(f"Exported encoder graph: {paths.encoder}", flush=True)
    print(f"Exported decoder graph: {paths.decoder}", flush=True)
    print(f"Export metadata: {paths.metadata}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
