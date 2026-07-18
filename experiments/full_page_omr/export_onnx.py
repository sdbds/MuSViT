from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    for label, path in (("encoder", paths.encoder), ("decoder", paths.decoder)):
        if path.is_file():
            artifacts[label] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "artifacts": artifacts,
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


def export_bundle(
    *,
    weights_path: str | Path,
    model_config_path: str | Path,
    encoder_config_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> ExportPaths:
    paths = ExportPaths.from_output_dir(output_dir)
    model, standalone_config = load_standalone_model(
        weights_path,
        model_config_path,
        encoder_config_path,
    )
    export_encoder_graph(
        FullPageOMREncoderWrapper(model),
        paths.encoder,
        device,
        opset_version=opset_version,
    )
    export_decoder_graph(
        FullPageOMRDecoderWrapper(model),
        paths.decoder,
        device,
        opset_version=opset_version,
    )
    validate_graph_files(paths)

    write_json_atomic(paths.config, standalone_config)
    write_json_atomic(paths.preprocessor_config, build_preprocessor_config())
    versions, providers = _runtime_versions_and_providers()
    metadata = build_bundle_metadata(
        paths=paths,
        source={
            "weights_path": str(Path(weights_path).resolve()),
            "weights_sha256": sha256_file(weights_path),
            "model_config_path": str(Path(model_config_path).resolve()),
            "model_config_sha256": sha256_file(model_config_path),
            "encoder_config_path": str(Path(encoder_config_path).resolve()),
            "encoder_config_sha256": sha256_file(encoder_config_path),
        },
        versions=versions,
        providers=providers,
        validation={"status": "not_run"},
    )
    metadata["opset_version"] = opset_version
    write_json_atomic(paths.metadata, metadata)
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
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    paths = export_bundle(
        weights_path=args.weights_path,
        model_config_path=args.model_config_path,
        encoder_config_path=args.encoder_config_path,
        output_dir=args.output_dir,
        device=device,
        opset_version=args.opset_version,
    )
    print(f"Exported encoder graph: {paths.encoder}", flush=True)
    print(f"Exported decoder graph: {paths.decoder}", flush=True)
    print(f"Export metadata: {paths.metadata}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
