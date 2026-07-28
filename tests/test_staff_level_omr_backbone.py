import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from experiments.staff_level_omr.protocol.backbone import (
    APPROVED_BACKBONES,
    BackboneMetadata,
    approved_revisions,
    default_revisions,
    git_blob_oid,
    load_encoder_from_pretrained,
    metadata_from_raw_config,
    registry_entry,
    validate_loading_info,
    validate_registry_tree,
    verify_weight_file,
)
from experiments.staff_level_omr.protocol.errors import ProtocolError


def _raw_config(**overrides):
    values = {
        "model_type": "vit_mae",
        "architectures": ["ViTMAEForPreTraining"],
        "image_size": 32,
        "patch_size": 8,
        "hidden_size": 16,
        "num_channels": 3,
    }
    values.update(overrides)
    return values


def test_registry_pins_both_reviewed_model_identities():
    musvit = APPROVED_BACKBONES["musvit"]
    light = APPROVED_BACKBONES["musvit_light"]

    assert musvit.model_id == "PRAIG/musvit"
    assert musvit.revision == "0e91c7b223b4da30f259198c92045d0cb90e3f2e"
    assert musvit.readme.git_blob_oid == (
        "8789d81c7e698c92b57746ab5dc090fe893069e2"
    )
    assert musvit.config.git_blob_oid == (
        "dc2f7bc9bf1aab858aaeefaf1db7858c427f79f2"
    )
    assert musvit.weights.pointer_blob_oid == (
        "e935a37bc0a6ca051a091a2ba5b5425547f16af9"
    )
    assert musvit.weights.sha256 == (
        "109bbaf31d9f2184df1b841579e06d25bc58ed6a42a10dd5f4a5d27d01889db2"
    )
    assert musvit.weights.size == 467638680
    assert light.model_id == "PRAIG/musvit-light"
    assert light.revision == "adf40fd3eaf157e20aaa8603ffea06517e467c7f"
    assert light.weights.sha256 == (
        "f2c278f2762a88bfcc7ee4cf846d8eef2f31c04a73e7775124db69e7afd0528f"
    )
    assert light.weights.size == 157546736
    assert approved_revisions()["musvit"] == {musvit.revision}
    assert default_revisions()["musvit_light"] == light.revision


def test_registry_lookup_requires_alias_and_exact_approved_revision():
    entry = registry_entry("musvit", APPROVED_BACKBONES["musvit"].revision)
    assert entry is APPROVED_BACKBONES["musvit"]

    with pytest.raises(ProtocolError, match="model_name"):
        registry_entry("auto", "a" * 40)
    with pytest.raises(ProtocolError, match="approved"):
        registry_entry("musvit", "a" * 40)


def test_metadata_normalizes_scalar_and_pair_dimensions():
    entry = APPROVED_BACKBONES["musvit"]
    scalar = metadata_from_raw_config(_raw_config(), entry)
    paired = metadata_from_raw_config(
        _raw_config(image_size=[32, 48], patch_size=[8, 16]),
        entry,
    )

    assert scalar.image_height == scalar.image_width == 32
    assert scalar.patch_height == scalar.patch_width == 8
    assert scalar.native_rows == scalar.native_cols == 4
    assert paired.image_height == 32
    assert paired.image_width == 48
    assert paired.patch_height == 8
    assert paired.patch_width == 16
    assert paired.native_rows == 4
    assert paired.native_cols == 3
    assert paired.prefix_tokens == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_type": "vit"}, "model_type"),
        ({"architectures": ["ViTModel"]}, "architectures"),
        ({"num_channels": 1}, "num_channels"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"image_size": 31}, "divisible"),
        ({"patch_size": [8, 0]}, "patch_size"),
        ({"image_size": [32, 32, 32]}, "image_size"),
    ],
)
def test_metadata_rejects_incompatible_raw_config(overrides, message):
    with pytest.raises(ProtocolError, match=message):
        metadata_from_raw_config(
            _raw_config(**overrides),
            APPROVED_BACKBONES["musvit"],
        )


def test_git_blob_oid_uses_git_object_header():
    payload = b'{"model_type":"vit_mae"}'
    expected = hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()

    assert git_blob_oid(payload) == expected


def test_registry_tree_validates_revision_metadata_and_processor_absence():
    entry = APPROVED_BACKBONES["musvit"]
    siblings = [
        SimpleNamespace(
            rfilename="README.md",
            size=entry.readme.size,
            blob_id=entry.readme.git_blob_oid,
            lfs=None,
        ),
        SimpleNamespace(
            rfilename="config.json",
            size=entry.config.size,
            blob_id=entry.config.git_blob_oid,
            lfs=None,
        ),
        SimpleNamespace(
            rfilename="model.safetensors",
            size=entry.weights.size,
            blob_id=entry.weights.pointer_blob_oid,
            lfs=SimpleNamespace(
                sha256=entry.weights.sha256,
                size=entry.weights.size,
            ),
        ),
    ]

    validate_registry_tree(
        entry,
        SimpleNamespace(sha=entry.revision, siblings=siblings),
    )

    siblings.append(
        SimpleNamespace(
            rfilename="preprocessor_config.json",
            size=2,
            blob_id="a" * 40,
            lfs=None,
        )
    )
    with pytest.raises(ProtocolError, match="preprocessor"):
        validate_registry_tree(
            entry,
            SimpleNamespace(sha=entry.revision, siblings=siblings),
        )


def test_weight_verification_checks_actual_bytes_before_model_loading(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weight fixture")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    assert verify_weight_file(path, expected_size=14, expected_sha256=sha) == {
        "filename": "model.safetensors",
        "sha256": sha,
        "size": 14,
    }

    with pytest.raises(ProtocolError, match="size"):
        verify_weight_file(path, expected_size=13, expected_sha256=sha)
    with pytest.raises(ProtocolError, match="SHA-256"):
        verify_weight_file(path, expected_size=14, expected_sha256="0" * 64)


def test_loading_info_only_allows_decoder_unexpected_keys():
    validate_loading_info(
        {
            "missing_keys": [],
            "mismatched_keys": [],
            "unexpected_keys": ["decoder.mask_token", "decoder.layer.weight"],
            "error_msgs": [],
        }
    )

    for field, value in [
        ("missing_keys", ["encoder.weight"]),
        ("mismatched_keys", [("encoder.weight", (1,), (2,))]),
        ("unexpected_keys", ["classifier.weight"]),
        ("error_msgs", ["load failed"]),
    ]:
        info = {
            "missing_keys": [],
            "mismatched_keys": [],
            "unexpected_keys": [],
            "error_msgs": [],
        }
        info[field] = value
        with pytest.raises(ProtocolError, match=field):
            validate_loading_info(info)


def test_tiny_vitmae_checkpoint_loads_through_fixed_vit_encoder(tmp_path):
    config = ViTMAEConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        decoder_hidden_size=8,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2,
        decoder_intermediate_size=16,
    )
    source = tmp_path / "tiny"
    ViTMAEForPreTraining(config).save_pretrained(source)
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    fixture_entry = SimpleNamespace(
        model_id=str(source),
        revision="b" * 40,
        prefix_tokens=1,
    )
    metadata = metadata_from_raw_config(raw, fixture_entry)

    result = load_encoder_from_pretrained(
        source,
        revision=None,
        expected_metadata=metadata,
    )
    output = result.model(
        pixel_values=torch.zeros((1, 3, 32, 32)),
        interpolate_pos_encoding=False,
    )

    assert result.model.__class__.__name__ == "ViTModel"
    assert output.last_hidden_state.shape == (1, 1 + 4 * 4, 16)
    assert result.loading_info["missing_keys"] == []
    assert all(
        key.startswith("decoder.")
        for key in result.loading_info["unexpected_keys"]
    )


def test_loaded_backbone_config_must_match_inspected_metadata(tmp_path):
    config = ViTMAEConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        decoder_hidden_size=8,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2,
        decoder_intermediate_size=16,
    )
    source = tmp_path / "tiny"
    ViTMAEForPreTraining(config).save_pretrained(source)
    expected = BackboneMetadata(
        model_id=str(source),
        revision="b" * 40,
        image_height=32,
        image_width=32,
        patch_height=8,
        patch_width=8,
        hidden_size=32,
        num_channels=3,
        prefix_tokens=1,
        model_type="vit_mae",
        architectures=("ViTMAEForPreTraining",),
    )

    with pytest.raises(ProtocolError, match="hidden_size"):
        load_encoder_from_pretrained(
            source,
            revision=None,
            expected_metadata=expected,
        )
