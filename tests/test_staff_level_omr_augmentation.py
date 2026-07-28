import hashlib
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch

from experiments.staff_level_omr.protocol.augmentation import (
    STAFF_OMR_TRAIN_V1,
    augmentation_contract,
    build_augmentation,
    preflight_augmentation,
)
from experiments.staff_level_omr.protocol.canonical import canonical_sha256
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol.seeding import (
    INIT_SEED_LABEL,
    SEED_SCHEDULE_VERSION,
    reset_epoch_rng,
    sample_augment_seed,
    sample_order_key,
    seed32,
    seed256,
    seed_digest,
    worker_base_seed,
)


def test_package_disables_albumentations_network_update_check():
    environment = os.environ.copy()
    environment.pop("NO_ALBUMENTATIONS_UPDATE", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import experiments.staff_level_omr; "
                "print(os.environ.get('NO_ALBUMENTATIONS_UPDATE'))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def _manual_digest(label: str, *parts: object) -> bytes:
    return hashlib.sha256(
        b"staff_omr_v2\0"
        + label.encode("utf-8")
        + b"\0"
        + b"\0".join(str(part).encode("utf-8") for part in parts)
    ).digest()


def test_seed_digest_matches_the_protocol_byte_layout():
    expected = _manual_digest("augment", 7, 3, "谱面-01")

    assert seed_digest("augment", 7, 3, "谱面-01") == expected
    assert seed32("augment", 7, 3, "谱面-01") == int.from_bytes(
        expected[:4], "big"
    )
    assert seed256("augment", 7, 3, "谱面-01") == int.from_bytes(
        expected, "big"
    )
    assert INIT_SEED_LABEL == "init"
    assert SEED_SCHEDULE_VERSION == "staff_omr_sample_epoch_sha256_v1"


@pytest.mark.parametrize(
    ("label", "parts"),
    [
        ("bad\0label", (1,)),
        ("order", ("bad\0sample",)),
        ("order", (True,)),
        ("order", (-1,)),
        ("order", (1.5,)),
    ],
)
def test_seed_digest_rejects_ambiguous_parts(label, parts):
    with pytest.raises(ProtocolError):
        seed_digest(label, *parts)


def test_seed_domains_are_distinct_and_order_key_uses_full_digest():
    order = sample_order_key(7, 2, "sample-1")

    assert order == (
        _manual_digest("order", 7, 2, "sample-1"),
        b"sample-1",
    )
    assert sample_augment_seed(7, 2, "sample-1") == int.from_bytes(
        _manual_digest("augment", 7, 2, "sample-1"), "big"
    )
    assert worker_base_seed(7, 2) == int.from_bytes(
        _manual_digest("worker", 7, 2)[:4], "big"
    )
    assert sample_order_key(7, 2, "sample-1") != sample_order_key(
        7, 3, "sample-1"
    )
    assert sample_order_key(7, 2, "sample-1") != sample_order_key(
        7, 2, "sample-2"
    )


def test_reset_epoch_rng_reconstructs_main_process_random_stream():
    epoch_seed = reset_epoch_rng(7, 4)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert reset_epoch_rng(7, 4) == epoch_seed
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_none_profile_has_stable_empty_contract_and_no_pipeline():
    assert augmentation_contract("none") == {
        "schema_version": "staff_omr_augmentation_v1",
        "profile": "none",
        "transforms": [],
    }
    assert build_augmentation("none") is None


def test_staff_profile_declares_normalized_effective_blur_values():
    contract = augmentation_contract(STAFF_OMR_TRAIN_V1)
    blur = contract["transforms"][5]
    gaussian = blur["transforms"][0]["transforms"]
    motion = blur["transforms"][1]

    assert gaussian[0]["blur_limit"] == [3, 3]
    assert gaussian[1]["blur_limit"] == [5, 5]
    assert gaussian[0]["sigma_limit"] == [0.5, 3.0]
    assert gaussian[1]["sigma_limit"] == [0.5, 3.0]
    assert motion["blur_limit"] == [3, 5]
    assert motion["angle_range"] == [0, 360]
    assert motion["direction_range"] == [-1, 1]
    assert all(
        value % 2 == 1
        for leaf in gaussian
        for value in leaf["blur_limit"]
    )
    assert canonical_sha256(contract) == canonical_sha256(
        augmentation_contract(STAFF_OMR_TRAIN_V1)
    )


def test_staff_profile_supplies_every_behavior_parameter_explicitly():
    contract = augmentation_contract(STAFF_OMR_TRAIN_V1)

    assert contract == {
        "schema_version": "staff_omr_augmentation_v1",
        "profile": "staff_omr_train_v1",
        "compose_p": 1.0,
        "transforms": [
            {
                "type": "OneOf",
                "p": 0.6,
                "transforms": [
                    {
                        "type": "Morphological",
                        "scale": [2, 2],
                        "operation": "dilation",
                        "p": 1.0,
                    },
                    {
                        "type": "Morphological",
                        "scale": [2, 2],
                        "operation": "erosion",
                        "p": 1.0,
                    },
                ],
            },
            {
                "type": "Sharpen",
                "alpha": [0.2, 0.5],
                "lightness": [0.5, 1.0],
                "method": "kernel",
                "kernel_size": 5,
                "sigma": 1.0,
                "p": 0.25,
            },
            {
                "type": "Rotate",
                "limit": [-3, 3],
                "interpolation": "linear",
                "border_mode": "replicate",
                "rotate_method": "largest_box",
                "crop_border": False,
                "mask_interpolation": "nearest",
                "fill": 0,
                "fill_mask": 0,
                "p": 0.5,
            },
            {
                "type": "GaussNoise",
                "std_range": [0.01, 0.15],
                "mean_range": [0, 0],
                "per_channel": False,
                "noise_scale_factor": 1,
                "p": 0.3,
            },
            {
                "type": "ColorJitter",
                "brightness": [0.25, 1.75],
                "contrast": [0.25, 1.75],
                "saturation": [0.25, 1.75],
                "hue": [-0.05, 0.05],
                "p": 0.75,
            },
            {
                "type": "OneOf",
                "p": 0.25,
                "transforms": [
                    {
                        "type": "OneOf",
                        "p": 1.0,
                        "transforms": [
                            {
                                "type": "GaussianBlur",
                                "blur_limit": [3, 3],
                                "sigma_limit": [0.5, 3.0],
                                "p": 1.0,
                            },
                            {
                                "type": "GaussianBlur",
                                "blur_limit": [5, 5],
                                "sigma_limit": [0.5, 3.0],
                                "p": 1.0,
                            },
                        ],
                    },
                    {
                        "type": "MotionBlur",
                        "blur_limit": [3, 5],
                        "allow_shifted": True,
                        "angle_range": [0, 360],
                        "direction_range": [-1, 1],
                        "p": 1.0,
                    },
                ],
            },
            {
                "type": "ToGray",
                "num_output_channels": 3,
                "method": "weighted_average",
                "p": 0.1,
            },
        ],
    }


def test_build_profile_does_not_use_to_dict(monkeypatch):
    monkeypatch.setattr(
        A.Compose,
        "to_dict",
        lambda self: (_ for _ in ()).throw(AssertionError("must not be used")),
    )

    pipeline = build_augmentation(STAFF_OMR_TRAIN_V1)

    assert isinstance(pipeline, A.Compose)


def test_preflight_probes_actual_kernels_and_full_256_bit_seed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = preflight_augmentation(STAFF_OMR_TRAIN_V1)

    assert report["status"] == "passed"
    assert report["gaussian_kernel_sizes"] == [3, 5]
    assert report["motion_kernel_sizes"] == [3, 5]
    assert report["full_seed_repeat_equal"] is True
    assert report["high_bit_changes_output"] is True
    assert report["contract_sha256"] == canonical_sha256(
        augmentation_contract(STAFF_OMR_TRAIN_V1)
    )


def test_preflight_rejects_even_blur_declaration():
    contract = augmentation_contract(STAFF_OMR_TRAIN_V1)
    contract["transforms"][5]["transforms"][0]["transforms"][0][
        "blur_limit"
    ] = [3, 4]

    with pytest.raises(ProtocolError, match="odd|kernel"):
        preflight_augmentation(
            STAFF_OMR_TRAIN_V1,
            contract_override=contract,
        )


def test_preflight_rejects_zero_probability_leaf():
    contract = augmentation_contract(STAFF_OMR_TRAIN_V1)
    contract["transforms"][6]["p"] = 0.0

    with pytest.raises(ProtocolError, match="p > 0"):
        preflight_augmentation(
            STAFF_OMR_TRAIN_V1,
            contract_override=contract,
        )
