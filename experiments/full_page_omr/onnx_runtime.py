from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


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
        return Image.fromarray(image)
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
    pixels *= np.float32(config.get("rescale_factor", 1 / 255))
    pixels = pixels.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(pixels, dtype=np.float32)


def _session_names(session, method_name: str) -> list[str]:
    return [value.name for value in getattr(session, method_name)()]


def _require_contract(
    session,
    *,
    label: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    actual_inputs = _session_names(session, "get_inputs")
    actual_outputs = _session_names(session, "get_outputs")
    if actual_inputs != input_names:
        raise ValueError(
            f"{label} input contract mismatch: {actual_inputs!r} != {input_names!r}"
        )
    if actual_outputs != output_names:
        raise ValueError(
            f"{label} output contract mismatch: {actual_outputs!r} != {output_names!r}"
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

            resolved_providers = resolve_providers(providers)
            if encoder_session is None:
                encoder_session = ort.InferenceSession(
                    str(self.bundle_dir / "encoder.onnx"),
                    providers=resolved_providers,
                )
            if decoder_session is None:
                decoder_session = ort.InferenceSession(
                    str(self.bundle_dir / "decoder.onnx"),
                    providers=resolved_providers,
                )
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session
        _require_contract(
            self.encoder_session,
            label="encoder",
            input_names=["pixel_values"],
            output_names=["raw_features", "enhanced_features"],
        )
        _require_contract(
            self.decoder_session,
            label="decoder",
            input_names=["raw_features", "enhanced_features", "token_ids"],
            output_names=["next_token_logits"],
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
