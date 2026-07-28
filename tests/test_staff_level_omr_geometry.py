from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings

from experiments.staff_level_omr.protocol.backbone import BackboneMetadata
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol.geometry import (
    INPUT_CONTRACT_SCHEMA,
    StaffImageProcessor,
    build_geometry_plan,
    extract_spatial_grid,
    reference_interpolate_pos_encoding,
    verify_transformers_position_interpolation,
)


META = BackboneMetadata(
    model_id="fixture/vit",
    revision="a" * 40,
    image_height=32,
    image_width=32,
    patch_height=8,
    patch_width=8,
    hidden_size=16,
    num_channels=3,
    prefix_tokens=1,
    model_type="vit_mae",
    architectures=("ViTMAEForPreTraining",),
)


def test_native_pad_derives_all_dimensions_from_backbone_metadata():
    plan = build_geometry_plan(
        META,
        method="linear_probe",
        patch_rows=2,
        patch_cols=4,
    )

    assert plan.geometry == "native_pad"
    assert plan.native_rows == 4
    assert plan.native_cols == 4
    assert plan.content_height == 16
    assert plan.content_width == 32
    assert plan.input_height == 32
    assert plan.input_width == 32
    assert plan.output_rows == 2
    assert plan.output_cols == 4
    assert plan.interpolate_pos_encoding is False


@pytest.mark.parametrize(
    ("rows", "cols", "message"),
    [
        (5, 4, "patch_rows"),
        (2, 3, "patch_cols"),
    ],
)
def test_native_pad_rejects_grid_outside_native_contract(rows, cols, message):
    with pytest.raises(ProtocolError, match=message):
        build_geometry_plan(
            META,
            method="linear_probe",
            patch_rows=rows,
            patch_cols=cols,
        )


def test_exact_grid_supports_non_native_width_without_padding():
    plan = build_geometry_plan(
        META,
        method="lora",
        patch_rows=3,
        patch_cols=7,
    )

    assert plan.geometry == "exact_grid"
    assert plan.content_height == plan.input_height == 24
    assert plan.content_width == plan.input_width == 56
    assert plan.output_rows == 3
    assert plan.output_cols == 7
    assert plan.interpolate_pos_encoding is True


def test_method_and_geometry_are_not_independently_selectable():
    with pytest.raises(ProtocolError, match="method"):
        build_geometry_plan(META, method="full", patch_rows=2, patch_cols=4)


def test_native_pad_processor_only_adds_white_pixels_at_bottom():
    image = Image.fromarray(np.full((10, 20, 3), 17, dtype=np.uint8), "RGB")
    plan = build_geometry_plan(
        META,
        method="linear_probe",
        patch_rows=2,
        patch_cols=4,
    )

    tensor = StaffImageProcessor(plan)(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    torch.testing.assert_close(
        tensor[:, 16:, :],
        torch.ones((3, 16, 32)),
        rtol=0,
        atol=0,
    )
    assert torch.all(tensor[:, :16, :] < 1)


def test_exact_grid_processor_converts_numpy_rgb_without_normalization():
    image = np.zeros((11, 19, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    plan = build_geometry_plan(
        META,
        method="lora",
        patch_rows=2,
        patch_cols=3,
    )

    tensor = StaffImageProcessor(plan)(image)

    assert tensor.shape == (3, 16, 24)
    torch.testing.assert_close(tensor[0], torch.ones((16, 24)))
    torch.testing.assert_close(tensor[1:], torch.zeros((2, 16, 24)))


def test_input_contract_records_resize_padding_and_tensor_semantics():
    plan = build_geometry_plan(
        META,
        method="linear_probe",
        patch_rows=2,
        patch_cols=4,
    )

    contract = plan.to_contract()

    assert contract["schema_version"] == INPUT_CONTRACT_SCHEMA
    assert contract["geometry"] == "native_pad"
    assert contract["resize"] == {
        "height": 16,
        "width": 32,
        "interpolation": "bilinear",
        "antialias": True,
        "preserve_aspect_ratio": False,
    }
    assert contract["padding"] == {
        "bottom": 16,
        "left": 0,
        "right": 0,
        "top": 0,
        "fill_rgb": [255, 255, 255],
    }
    assert contract["tensor"] == {
        "channel_order": "CHW",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "mean_std_normalization": None,
    }


def test_exact_grid_extracts_rows_columns_from_all_spatial_tokens():
    plan = build_geometry_plan(META, "lora", patch_rows=2, patch_cols=3)
    hidden = torch.arange(1 * 7 * 16).reshape(1, 7, 16)

    grid = extract_spatial_grid(hidden, plan)

    assert grid.shape == (1, 2, 3, 16)
    torch.testing.assert_close(grid.flatten(1, 2), hidden[:, 1:])


def test_native_pad_extracts_native_grid_then_slices_top_rows():
    plan = build_geometry_plan(
        META,
        "linear_probe",
        patch_rows=2,
        patch_cols=4,
    )
    hidden = torch.arange(1 * 17 * 16).reshape(1, 17, 16)

    grid = extract_spatial_grid(hidden, plan)

    assert grid.shape == (1, 2, 4, 16)
    expected = hidden[:, 1:].reshape(1, 4, 4, 16)[:, :2]
    torch.testing.assert_close(grid, expected)


def test_token_count_mismatch_has_model_geometry_context():
    plan = build_geometry_plan(META, "lora", patch_rows=2, patch_cols=3)

    with pytest.raises(
        ProtocolError,
        match=r"fixture/vit.*exact_grid.*expected 7.*actual 6",
    ):
        extract_spatial_grid(torch.zeros(1, 6, 16), plan)


def test_reference_matches_installed_transformers_bicubic_behavior():
    config = ViTConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    embeddings = ViTEmbeddings(config)
    input_tokens = torch.zeros((1, 1 + 2 * 5, 16))

    actual = embeddings.interpolate_pos_encoding(
        input_tokens,
        height=16,
        width=40,
    )
    expected = reference_interpolate_pos_encoding(
        embeddings.position_embeddings,
        height=16,
        width=40,
        metadata=META,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    report = verify_transformers_position_interpolation(
        embeddings,
        META,
        height=16,
        width=40,
    )
    assert report == {
        "align_corners": False,
        "antialias": False,
        "mode": "bicubic",
        "status": "passed",
        "target_cols": 5,
        "target_rows": 2,
    }


def test_reference_rejects_position_table_that_disagrees_with_metadata():
    bad_meta = replace(META, image_width=40)
    positions = torch.zeros((1, 17, 16))

    with pytest.raises(ProtocolError, match="position"):
        reference_interpolate_pos_encoding(
            positions,
            height=16,
            width=40,
            metadata=bad_meta,
        )
