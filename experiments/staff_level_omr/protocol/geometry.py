"""Explicit image and spatial-token geometry for staff OMR v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .backbone import BackboneMetadata
from .errors import ProtocolError


INPUT_CONTRACT_SCHEMA = "staff_omr_input_v2"


@dataclass(frozen=True, slots=True)
class GeometryPlan:
    metadata: BackboneMetadata
    method: str
    geometry: str
    patch_rows: int
    patch_cols: int
    native_rows: int
    native_cols: int
    content_height: int
    content_width: int
    input_height: int
    input_width: int
    output_rows: int
    output_cols: int
    pad_bottom: int
    interpolate_pos_encoding: bool

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": INPUT_CONTRACT_SCHEMA,
            "method": self.method,
            "geometry": self.geometry,
            "image_decode": {
                "color_mode": "RGB",
                "channels": 3,
            },
            "backbone": {
                "model_id": self.metadata.model_id,
                "revision": self.metadata.revision,
                "image_height": self.metadata.image_height,
                "image_width": self.metadata.image_width,
                "patch_height": self.metadata.patch_height,
                "patch_width": self.metadata.patch_width,
                "hidden_size": self.metadata.hidden_size,
                "prefix_tokens": self.metadata.prefix_tokens,
                "native_rows": self.native_rows,
                "native_cols": self.native_cols,
            },
            "patch_grid": {
                "rows": self.patch_rows,
                "cols": self.patch_cols,
            },
            "model_input": {
                "height": self.input_height,
                "width": self.input_width,
            },
            "resize": {
                "height": self.content_height,
                "width": self.content_width,
                "interpolation": "bilinear",
                "antialias": True,
                "preserve_aspect_ratio": False,
            },
            "padding": {
                "bottom": self.pad_bottom,
                "left": 0,
                "right": 0,
                "top": 0,
                "fill_rgb": [255, 255, 255],
            },
            "tensor": {
                "channel_order": "CHW",
                "dtype": "float32",
                "range": [0.0, 1.0],
                "mean_std_normalization": None,
            },
            "position_encoding": {
                "interpolate": self.interpolate_pos_encoding,
                "mode": (
                    "bicubic" if self.interpolate_pos_encoding else None
                ),
                "align_corners": (
                    False if self.interpolate_pos_encoding else None
                ),
                "antialias": (
                    False if self.interpolate_pos_encoding else None
                ),
            },
            "spatial_tokens": {
                "prefix_tokens": self.metadata.prefix_tokens,
                "reshape_rows": (
                    self.native_rows
                    if self.geometry == "native_pad"
                    else self.patch_rows
                ),
                "reshape_cols": (
                    self.native_cols
                    if self.geometry == "native_pad"
                    else self.patch_cols
                ),
                "slice_top_rows": (
                    self.patch_rows
                    if self.geometry == "native_pad"
                    else None
                ),
                "output_rows": self.output_rows,
                "output_cols": self.output_cols,
                "order": "rows_then_columns",
            },
            "preprocessing_basis": {
                "reviewed_contract": INPUT_CONTRACT_SCHEMA,
                "runtime_reads_preprocessor_config": False,
            },
        }


def _positive_grid(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{field} must be a positive integer")
    return value


def build_geometry_plan(
    metadata: BackboneMetadata,
    method: str,
    patch_rows: int,
    patch_cols: int,
) -> GeometryPlan:
    rows = _positive_grid(patch_rows, "patch_rows")
    cols = _positive_grid(patch_cols, "patch_cols")
    if method not in {"linear_probe", "lora"}:
        raise ProtocolError(
            "method must be 'linear_probe' or 'lora' for input geometry"
        )
    native_rows = metadata.native_rows
    native_cols = metadata.native_cols
    content_height = rows * metadata.patch_height
    content_width = cols * metadata.patch_width
    if method == "linear_probe":
        if rows > native_rows:
            raise ProtocolError(
                f"linear_probe patch_rows {rows} exceeds native rows "
                f"{native_rows}"
            )
        if cols != native_cols:
            raise ProtocolError(
                f"linear_probe patch_cols must equal native cols "
                f"{native_cols}, got {cols}"
            )
        return GeometryPlan(
            metadata=metadata,
            method=method,
            geometry="native_pad",
            patch_rows=rows,
            patch_cols=cols,
            native_rows=native_rows,
            native_cols=native_cols,
            content_height=content_height,
            content_width=content_width,
            input_height=metadata.image_height,
            input_width=metadata.image_width,
            output_rows=rows,
            output_cols=cols,
            pad_bottom=metadata.image_height - content_height,
            interpolate_pos_encoding=False,
        )
    return GeometryPlan(
        metadata=metadata,
        method=method,
        geometry="exact_grid",
        patch_rows=rows,
        patch_cols=cols,
        native_rows=native_rows,
        native_cols=native_cols,
        content_height=content_height,
        content_width=content_width,
        input_height=content_height,
        input_width=content_width,
        output_rows=rows,
        output_cols=cols,
        pad_bottom=0,
        interpolate_pos_encoding=True,
    )


class StaffImageProcessor:
    """Apply only the resize, pad, and tensor rules in a GeometryPlan."""

    def __init__(self, plan: GeometryPlan):
        self.plan = plan

    def __call__(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ProtocolError("decoded image array must have three channels")
            image = Image.fromarray(image.astype(np.uint8, copy=False), "RGB")
        if not isinstance(image, Image.Image):
            raise ProtocolError("image processor expects a PIL image or RGB array")
        rgb = image.convert("RGB")
        resized = vision_functional.resize(
            rgb,
            [self.plan.content_height, self.plan.content_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.plan.pad_bottom:
            resized = vision_functional.pad(
                resized,
                [0, 0, 0, self.plan.pad_bottom],
                fill=(255, 255, 255),
                padding_mode="constant",
            )
        tensor = vision_functional.pil_to_tensor(resized).to(torch.float32)
        return tensor.div(255.0)


def extract_spatial_grid(
    last_hidden_state: torch.Tensor,
    plan: GeometryPlan,
) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ProtocolError(
            "backbone last_hidden_state must have shape [batch, tokens, hidden]"
        )
    if last_hidden_state.shape[2] != plan.metadata.hidden_size:
        raise ProtocolError(
            f"backbone hidden size mismatch: expected "
            f"{plan.metadata.hidden_size}, actual {last_hidden_state.shape[2]}"
        )
    reshape_rows = (
        plan.native_rows if plan.geometry == "native_pad" else plan.patch_rows
    )
    reshape_cols = (
        plan.native_cols if plan.geometry == "native_pad" else plan.patch_cols
    )
    expected_tokens = (
        plan.metadata.prefix_tokens + reshape_rows * reshape_cols
    )
    actual_tokens = int(last_hidden_state.shape[1])
    if actual_tokens != expected_tokens:
        raise ProtocolError(
            f"{plan.metadata.model_id}@{plan.metadata.revision} "
            f"{plan.geometry} token contract expected {expected_tokens} tokens "
            f"for input {plan.input_height}x{plan.input_width}, patch "
            f"{plan.metadata.patch_height}x{plan.metadata.patch_width}; "
            f"actual {actual_tokens}"
        )
    spatial = last_hidden_state[:, plan.metadata.prefix_tokens :]
    grid = spatial.reshape(
        last_hidden_state.shape[0],
        reshape_rows,
        reshape_cols,
        plan.metadata.hidden_size,
    )
    if plan.geometry == "native_pad":
        grid = grid[:, : plan.patch_rows]
    return grid


def reference_interpolate_pos_encoding(
    position_embeddings: torch.Tensor,
    *,
    height: int,
    width: int,
    metadata: BackboneMetadata,
) -> torch.Tensor:
    """Reviewed bicubic reference for Transformers ViT position embeddings."""
    if height % metadata.patch_height or width % metadata.patch_width:
        raise ProtocolError(
            "position interpolation input must align to the patch size"
        )
    if position_embeddings.ndim != 3:
        raise ProtocolError("position embeddings must be a rank-three tensor")
    expected_positions = (
        metadata.prefix_tokens
        + metadata.native_rows * metadata.native_cols
    )
    if position_embeddings.shape[1] != expected_positions:
        raise ProtocolError(
            f"position embedding count mismatch: expected {expected_positions}, "
            f"actual {position_embeddings.shape[1]}"
        )
    if position_embeddings.shape[2] != metadata.hidden_size:
        raise ProtocolError("position embedding hidden size mismatch")
    target_rows = height // metadata.patch_height
    target_cols = width // metadata.patch_width
    if (
        target_rows == metadata.native_rows
        and target_cols == metadata.native_cols
        and height == width
    ):
        return position_embeddings
    prefix = position_embeddings[:, : metadata.prefix_tokens]
    patch = position_embeddings[:, metadata.prefix_tokens :]
    patch = patch.reshape(
        1,
        metadata.native_rows,
        metadata.native_cols,
        metadata.hidden_size,
    ).permute(0, 3, 1, 2)
    patch = torch_functional.interpolate(
        patch,
        size=(target_rows, target_cols),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    patch = patch.permute(0, 2, 3, 1).reshape(
        1,
        target_rows * target_cols,
        metadata.hidden_size,
    )
    return torch.cat((prefix, patch), dim=1)


def verify_transformers_position_interpolation(
    embeddings_module: Any,
    metadata: BackboneMetadata,
    *,
    height: int,
    width: int,
) -> dict[str, object]:
    target_rows = height // metadata.patch_height
    target_cols = width // metadata.patch_width
    placeholder = embeddings_module.position_embeddings.new_zeros(
        (
            1,
            metadata.prefix_tokens + target_rows * target_cols,
            metadata.hidden_size,
        )
    )
    actual = embeddings_module.interpolate_pos_encoding(
        placeholder,
        height,
        width,
    )
    expected = reference_interpolate_pos_encoding(
        embeddings_module.position_embeddings,
        height=height,
        width=width,
        metadata=metadata,
    )
    if not torch.equal(actual, expected):
        max_error = float((actual - expected).abs().max().item())
        raise ProtocolError(
            "installed Transformers position interpolation differs from "
            f"the reviewed bicubic reference (max abs error {max_error})"
        )
    return {
        "align_corners": False,
        "antialias": False,
        "mode": "bicubic",
        "status": "passed",
        "target_cols": target_cols,
        "target_rows": target_rows,
    }
