from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


_ENCODER_INPUT_CONTRACT = [
    ("pixel_values", "tensor(float)", [1, 3, 1024, 1024]),
]
_ENCODER_OUTPUT_CONTRACT = [
    ("raw_features", "tensor(float)", [1, 4096, 256]),
    ("enhanced_features", "tensor(float)", [1, 4096, 256]),
]
_DECODER_INPUT_CONTRACT = [
    ("raw_features", "tensor(float)", [1, 4096, 256]),
    ("enhanced_features", "tensor(float)", [1, 4096, 256]),
    ("token_ids", "tensor(int64)", [1, "sequence_length"]),
]
_DECODER_OUTPUT_CONTRACT = [
    ("next_token_logits", "tensor(float)", [1, 215]),
]
_BUNDLE_FILES = {
    "encoder": "encoder.onnx",
    "decoder": "decoder.onnx",
    "config": "config.json",
    "preprocessor_config": "preprocessor_config.json",
}


@dataclass(frozen=True)
class OnnxGenerationResult:
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    terminated_by_eos: bool
    truncated: bool


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bundle_metadata(bundle_dir: Path) -> dict[str, Any]:
    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"bundle metadata is missing: {metadata_path}")
    metadata = _read_json_object(metadata_path, label="bundle metadata")
    if metadata.get("format_version") != 1:
        raise RuntimeError("unsupported bundle metadata format_version")
    if metadata.get("bundle_status") != "complete":
        raise RuntimeError("bundle metadata does not mark the bundle complete")
    if metadata.get("files") != _BUNDLE_FILES:
        raise RuntimeError("bundle file manifest does not match the runtime contract")

    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("bundle metadata artifacts must be an object")
    for label, filename in _BUNDLE_FILES.items():
        artifact = artifacts.get(label)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"{label} artifact metadata is missing")
        path = bundle_dir / filename
        if not path.is_file():
            raise RuntimeError(f"{label} artifact is missing: {path}")
        expected_bytes = artifact.get("bytes")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{label} artifact size mismatch")
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"{label} artifact hash mismatch")
    return metadata


def resolve_providers(
    requested: Sequence[str] | None,
    *,
    available: Sequence[str] | None = None,
) -> list[str]:
    if available is None:
        import onnxruntime as ort

        available = ort.get_available_providers()
    available = list(available)
    if requested is not None:
        resolved = list(requested)
        unavailable = [provider for provider in resolved if provider not in available]
        if unavailable:
            raise RuntimeError(
                "Requested ONNX Runtime providers are unavailable: "
                + ", ".join(unavailable)
            )
        if (
            "CUDAExecutionProvider" in resolved
            and "CPUExecutionProvider" in available
            and "CPUExecutionProvider" not in resolved
        ):
            resolved.append("CPUExecutionProvider")
        return resolved

    if "CUDAExecutionProvider" in available:
        resolved = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            resolved.append("CPUExecutionProvider")
        return resolved
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    if available:
        return [available[0]]
    raise RuntimeError("No ONNX Runtime execution providers are available")


def _coerce_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim not in (2, 3) or 0 in array.shape:
            raise ValueError("NumPy images must be non-empty HxW or HxWxC arrays")
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            raise ValueError("NumPy image channels must be 1, 3, or 4")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("floating-point NumPy images must be finite")
            minimum = float(array.min())
            maximum = float(array.max())
            if minimum < 0 or maximum > 255:
                raise ValueError("floating-point NumPy images must be in [0, 1] or [0, 255]")
            if maximum <= 1:
                array = array * np.float32(255.0)
            array = np.rint(array).astype(np.uint8)
        elif array.dtype == np.bool_:
            array = array.astype(np.uint8) * np.uint8(255)
        elif array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.integer):
                raise TypeError("NumPy images must use integer or floating-point values")
            minimum = int(array.min())
            maximum = int(array.max())
            if minimum < 0 or maximum > 255:
                raise ValueError("integer NumPy images must be in [0, 255]")
            array = array.astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        return Image.fromarray(np.ascontiguousarray(array))
    with Image.open(Path(image)) as opened:
        return opened.copy()


def preprocess_page(
    image: str | Path | Image.Image | np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    size = config.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("preprocessor image_size must be [height, width]")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("preprocessor image dimensions must be positive")
    if config.get("interpolation") != "bilinear":
        raise ValueError("only bilinear preprocessing is supported")

    prepared = _coerce_image(image).convert("RGB")
    prepared = prepared.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32)
    rescale_factor = float(config.get("rescale_factor", 1 / 255))
    if rescale_factor == 1 / 255:
        pixels /= np.float32(255.0)
    else:
        pixels *= np.float32(rescale_factor)
    pixels = pixels.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(pixels, dtype=np.float32)


def _session_contract(session, method_name: str) -> list[tuple[str, str, list[Any]]]:
    return [
        (value.name, value.type, list(value.shape))
        for value in getattr(session, method_name)()
    ]


def _require_contract(
    session,
    *,
    label: str,
    input_contract: list[tuple[str, str, list[Any]]],
    output_contract: list[tuple[str, str, list[Any]]],
) -> None:
    actual_inputs = _session_contract(session, "get_inputs")
    actual_outputs = _session_contract(session, "get_outputs")
    if actual_inputs != input_contract:
        raise ValueError(
            f"{label} input contract mismatch: {actual_inputs!r} != {input_contract!r}"
        )
    if actual_outputs != output_contract:
        raise ValueError(
            f"{label} output contract mismatch: {actual_outputs!r} != {output_contract!r}"
        )


class FullPageOMROnnxRuntime:
    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        providers: Sequence[str] | None = None,
        encoder_session=None,
        decoder_session=None,
    ):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.metadata = _validate_bundle_metadata(self.bundle_dir)
        self.config = _read_json_object(
            self.bundle_dir / "config.json",
            label="model config",
        )
        self.preprocessor_config = _read_json_object(
            self.bundle_dir / "preprocessor_config.json",
            label="preprocessor config",
        )

        if encoder_session is None or decoder_session is None:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            resolved_providers = resolve_providers(
                providers,
                available=available_providers,
            )
            if (
                "CUDAExecutionProvider" in resolved_providers
                and hasattr(ort, "preload_dlls")
            ):
                ort.preload_dlls()
            session_providers = [
                (provider, {"use_tf32": 0})
                if provider == "CUDAExecutionProvider"
                else provider
                for provider in resolved_providers
            ]
            if encoder_session is None:
                encoder_session = ort.InferenceSession(
                    str(self.bundle_dir / "encoder.onnx"),
                    providers=session_providers,
                )
            if decoder_session is None:
                decoder_session = ort.InferenceSession(
                    str(self.bundle_dir / "decoder.onnx"),
                    providers=session_providers,
                )
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session
        _require_contract(
            self.encoder_session,
            label="encoder",
            input_contract=_ENCODER_INPUT_CONTRACT,
            output_contract=_ENCODER_OUTPUT_CONTRACT,
        )
        _require_contract(
            self.decoder_session,
            label="decoder",
            input_contract=_DECODER_INPUT_CONTRACT,
            output_contract=_DECODER_OUTPUT_CONTRACT,
        )

        raw_i2w = self.config.get("i2w")
        raw_w2i = self.config.get("w2i")
        if not isinstance(raw_i2w, dict) or not isinstance(raw_w2i, dict):
            raise ValueError("model config must contain i2w and w2i dictionaries")
        self.i2w = {int(token_id): token for token_id, token in raw_i2w.items()}
        self.bos_token_id = int(raw_w2i["<bos>"])
        self.eos_token_id = int(raw_w2i["<eos>"])
        self.max_length = int(self.config["maxlen"])
        if self.max_length < 2:
            raise ValueError("model maxlen must be at least 2")
        self.providers = list(self.encoder_session.get_providers())

    def encode_pixel_values(self, pixel_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expected_height, expected_width = self.preprocessor_config["image_size"]
        expected_shape = (1, 3, int(expected_height), int(expected_width))
        if pixel_values.shape != expected_shape:
            raise ValueError(
                f"pixel_values shape must be {expected_shape}, got {pixel_values.shape}"
            )
        if pixel_values.dtype != np.float32:
            raise TypeError("pixel_values must use float32")
        raw, enhanced = self.encoder_session.run(
            ["raw_features", "enhanced_features"],
            {"pixel_values": np.ascontiguousarray(pixel_values)},
        )
        return (
            np.ascontiguousarray(raw, dtype=np.float32),
            np.ascontiguousarray(enhanced, dtype=np.float32),
        )

    def next_token_logits(
        self,
        raw_features: np.ndarray,
        enhanced_features: np.ndarray,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        if token_ids.dtype != np.int64 or token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("token_ids must have shape [1, T] and dtype int64")
        (logits,) = self.decoder_session.run(
            ["next_token_logits"],
            {
                "raw_features": np.ascontiguousarray(raw_features, dtype=np.float32),
                "enhanced_features": np.ascontiguousarray(
                    enhanced_features,
                    dtype=np.float32,
                ),
                "token_ids": np.ascontiguousarray(token_ids),
            },
        )
        return np.ascontiguousarray(logits, dtype=np.float32)

    def generate_pixel_values(self, pixel_values: np.ndarray) -> OnnxGenerationResult:
        raw_features, enhanced_features = self.encode_pixel_values(pixel_values)
        token_ids = [self.bos_token_id]
        terminated_by_eos = False
        while len(token_ids) < self.max_length:
            prefix = np.asarray([token_ids], dtype=np.int64)
            logits = self.next_token_logits(
                raw_features,
                enhanced_features,
                prefix,
            )
            next_token = int(np.argmax(logits[0]))
            token_ids.append(next_token)
            if next_token == self.eos_token_id:
                terminated_by_eos = True
                break

        decoded = []
        for token_id in token_ids[1:]:
            if token_id == self.eos_token_id:
                break
            try:
                decoded.append(self.i2w[token_id])
            except KeyError:
                raise KeyError(f"Unknown predicted token id {token_id}") from None
        return OnnxGenerationResult(
            token_ids=tuple(token_ids),
            tokens=tuple(decoded),
            terminated_by_eos=terminated_by_eos,
            truncated=not terminated_by_eos,
        )

    def generate(
        self,
        image: str | Path | Image.Image | np.ndarray,
    ) -> OnnxGenerationResult:
        return self.generate_pixel_values(
            preprocess_page(image, self.preprocessor_config)
        )
