"""Declared and runtime-verified image augmentation for staff OMR v2."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np

from .canonical import canonical_sha256
from .errors import ProtocolError
from .seeding import seed256


AUGMENTATION_SCHEMA = "staff_omr_augmentation_v1"
STAFF_OMR_TRAIN_V1 = "staff_omr_train_v1"
_BLUR_PROBE_SEEDS = range(256)

_STAFF_OMR_TRAIN_V1_CONTRACT: dict[str, object] = {
    "schema_version": AUGMENTATION_SCHEMA,
    "profile": STAFF_OMR_TRAIN_V1,
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

_NONE_CONTRACT: dict[str, object] = {
    "schema_version": AUGMENTATION_SCHEMA,
    "profile": "none",
    "transforms": [],
}

_SYMBOLIC_VALUES = {
    ("Rotate", "interpolation"): {
        "linear": cv2.INTER_LINEAR,
    },
    ("Rotate", "mask_interpolation"): {
        "nearest": cv2.INTER_NEAREST,
    },
    ("Rotate", "border_mode"): {
        "replicate": cv2.BORDER_REPLICATE,
    },
}


def augmentation_contract(profile: str) -> dict[str, object]:
    """Return a mutable copy of the handwritten canonical contract."""
    if profile == STAFF_OMR_TRAIN_V1:
        return copy.deepcopy(_STAFF_OMR_TRAIN_V1_CONTRACT)
    if profile == "none":
        return copy.deepcopy(_NONE_CONTRACT)
    raise ProtocolError(
        "augmentation profile must be 'staff_omr_train_v1' or 'none'"
    )


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{context} must be an object")
    return value


def _pair(value: object, context: str) -> tuple[Any, Any]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ProtocolError(f"{context} must contain exactly two values")
    return value[0], value[1]


def _build_leaf(spec: Mapping[str, Any]) -> A.BasicTransform:
    kind = spec.get("type")
    try:
        if kind == "Morphological":
            return A.Morphological(
                scale=_pair(spec["scale"], "Morphological.scale"),
                operation=spec["operation"],
                p=spec["p"],
            )
        if kind == "Sharpen":
            return A.Sharpen(
                alpha=_pair(spec["alpha"], "Sharpen.alpha"),
                lightness=_pair(spec["lightness"], "Sharpen.lightness"),
                method=spec["method"],
                kernel_size=spec["kernel_size"],
                sigma=spec["sigma"],
                p=spec["p"],
            )
        if kind == "Rotate":
            return A.Rotate(
                limit=_pair(spec["limit"], "Rotate.limit"),
                interpolation=_SYMBOLIC_VALUES[("Rotate", "interpolation")][
                    spec["interpolation"]
                ],
                border_mode=_SYMBOLIC_VALUES[("Rotate", "border_mode")][
                    spec["border_mode"]
                ],
                rotate_method=spec["rotate_method"],
                crop_border=spec["crop_border"],
                mask_interpolation=_SYMBOLIC_VALUES[
                    ("Rotate", "mask_interpolation")
                ][spec["mask_interpolation"]],
                fill=spec["fill"],
                fill_mask=spec["fill_mask"],
                p=spec["p"],
            )
        if kind == "GaussNoise":
            return A.GaussNoise(
                std_range=_pair(spec["std_range"], "GaussNoise.std_range"),
                mean_range=_pair(spec["mean_range"], "GaussNoise.mean_range"),
                per_channel=spec["per_channel"],
                noise_scale_factor=spec["noise_scale_factor"],
                p=spec["p"],
            )
        if kind == "ColorJitter":
            return A.ColorJitter(
                brightness=_pair(
                    spec["brightness"], "ColorJitter.brightness"
                ),
                contrast=_pair(spec["contrast"], "ColorJitter.contrast"),
                saturation=_pair(
                    spec["saturation"], "ColorJitter.saturation"
                ),
                hue=_pair(spec["hue"], "ColorJitter.hue"),
                p=spec["p"],
            )
        if kind == "GaussianBlur":
            return A.GaussianBlur(
                blur_limit=_pair(
                    spec["blur_limit"], "GaussianBlur.blur_limit"
                ),
                sigma_limit=_pair(
                    spec["sigma_limit"], "GaussianBlur.sigma_limit"
                ),
                p=spec["p"],
            )
        if kind == "MotionBlur":
            return A.MotionBlur(
                blur_limit=_pair(
                    spec["blur_limit"], "MotionBlur.blur_limit"
                ),
                allow_shifted=spec["allow_shifted"],
                angle_range=_pair(
                    spec["angle_range"], "MotionBlur.angle_range"
                ),
                direction_range=_pair(
                    spec["direction_range"], "MotionBlur.direction_range"
                ),
                p=spec["p"],
            )
        if kind == "ToGray":
            return A.ToGray(
                num_output_channels=spec["num_output_channels"],
                method=spec["method"],
                p=spec["p"],
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid {kind!r} augmentation declaration") from exc
    raise ProtocolError(f"unsupported augmentation transform {kind!r}")


def _build_transform(spec: Mapping[str, Any]) -> A.BasicTransform:
    if spec.get("type") != "OneOf":
        return _build_leaf(spec)
    children = spec.get("transforms")
    if not isinstance(children, list) or not children:
        raise ProtocolError("OneOf.transforms must be a non-empty array")
    try:
        return A.OneOf(
            [
                _build_transform(_require_mapping(child, "OneOf child"))
                for child in children
            ],
            p=spec["p"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid OneOf augmentation declaration") from exc


def _build_from_contract(contract: Mapping[str, Any]) -> A.Compose | None:
    if contract.get("schema_version") != AUGMENTATION_SCHEMA:
        raise ProtocolError("invalid augmentation schema_version")
    if contract.get("profile") == "none":
        if contract != _NONE_CONTRACT:
            raise ProtocolError("none augmentation contract must be empty")
        return None
    if contract.get("profile") != STAFF_OMR_TRAIN_V1:
        raise ProtocolError("unsupported augmentation contract profile")
    children = contract.get("transforms")
    if not isinstance(children, list) or not children:
        raise ProtocolError("augmentation transforms must be a non-empty array")
    try:
        return A.Compose(
            [
                _build_transform(_require_mapping(child, "transform"))
                for child in children
            ],
            p=contract["compose_p"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid Compose augmentation declaration") from exc


def build_augmentation(profile: str) -> A.Compose | None:
    return _build_from_contract(augmentation_contract(profile))


def _normalized(value: object) -> object:
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _expected_runtime_value(kind: str, field: str, value: object) -> object:
    mapping = _SYMBOLIC_VALUES.get((kind, field))
    if mapping is not None:
        try:
            return mapping[value]
        except KeyError as exc:
            raise ProtocolError(
                f"unsupported symbolic value for {kind}.{field}: {value!r}"
            ) from exc
    return value


def _assert_declared_attributes(
    spec: Mapping[str, Any],
    transform: A.BasicTransform,
) -> None:
    kind = spec.get("type")
    if kind != type(transform).__name__:
        raise ProtocolError(
            f"declared transform {kind!r} constructed as "
            f"{type(transform).__name__!r}"
        )
    if kind == "OneOf":
        declared_children = spec.get("transforms")
        if not isinstance(declared_children, list):
            raise ProtocolError("OneOf.transforms must be an array")
        if len(declared_children) != len(transform.transforms):
            raise ProtocolError("OneOf child count changed during construction")
        if _normalized(transform.p) != _normalized(spec.get("p")):
            raise ProtocolError("OneOf.p differs from its declared value")
        for child_spec, child in zip(
            declared_children, transform.transforms, strict=True
        ):
            _assert_declared_attributes(
                _require_mapping(child_spec, "OneOf child"),
                child,
            )
        return

    for field, declared in spec.items():
        if field == "type":
            continue
        if not hasattr(transform, field):
            raise ProtocolError(f"{kind}.{field} has no runtime attribute")
        actual = getattr(transform, field)
        expected = _expected_runtime_value(kind, field, declared)
        if _normalized(actual) != _normalized(expected):
            raise ProtocolError(
                f"{kind}.{field} constructed as {_normalized(actual)!r}, "
                f"declared {_normalized(expected)!r}"
            )


def _iter_leaves(
    spec: Mapping[str, Any],
    transform: A.BasicTransform,
):
    if spec.get("type") != "OneOf":
        yield spec, transform
        return
    children = spec["transforms"]
    for child_spec, child in zip(children, transform.transforms, strict=True):
        yield from _iter_leaves(
            _require_mapping(child_spec, "OneOf child"),
            child,
        )


def _tiny_image() -> np.ndarray:
    values = np.arange(24 * 40 * 3, dtype=np.uint32).reshape(24, 40, 3)
    return (values % 251).astype(np.uint8)


def _kernel_sizes(
    kind: str,
    spec: Mapping[str, Any],
    transform: A.BasicTransform,
) -> set[int]:
    observed: set[int] = set()
    image = _tiny_image()
    for seed in _BLUR_PROBE_SEEDS:
        transform.set_random_seed(seed)
        if kind == "GaussianBlur":
            params = transform.get_params_dependent_on_data(
                {"shape": image.shape},
                {"image": image},
            )
        else:
            params = transform.get_params()
        kernel = params["kernel"]
        observed.add(int(kernel.shape[0]))
    low, high = (int(item) for item in spec["blur_limit"])
    declared = (
        {low}
        if low == high
        else {value for value in range(low, high + 1) if value % 2 == 1}
    )
    if low % 2 == 0 or high % 2 == 0 or observed != declared:
        raise ProtocolError(
            f"{kind} actual kernel sizes {sorted(observed)} do not match "
            f"declared odd kernel sizes {sorted(declared)}"
        )
    return observed


def _probe_full_seed_support() -> tuple[bool, bool]:
    probe = A.Compose(
        [
            A.GaussNoise(
                std_range=(0.05, 0.15),
                mean_range=(0.0, 0.0),
                per_channel=False,
                noise_scale_factor=1.0,
                p=1.0,
            )
        ],
        p=1.0,
    )
    image = _tiny_image()
    full_seed = seed256("augmentation-preflight", 1)
    probe.set_random_seed(full_seed)
    first = probe(image=image)["image"]
    probe.set_random_seed(full_seed)
    repeated = probe(image=image)["image"]
    probe.set_random_seed(full_seed ^ (1 << 200))
    high_bit = probe(image=image)["image"]
    return np.array_equal(first, repeated), not np.array_equal(first, high_bit)


def _is_contract_warning(item: warnings.WarningMessage) -> bool:
    if issubclass(
        item.category,
        (DeprecationWarning, PendingDeprecationWarning, FutureWarning),
    ):
        return False
    try:
        source = Path(item.filename).resolve(strict=False)
        albumentations_root = Path(A.__file__).resolve().parent
    except OSError:
        return False
    try:
        source.relative_to(albumentations_root)
        return True
    except ValueError:
        return source == Path(__file__).resolve()


def preflight_augmentation(
    profile: str,
    *,
    contract_override: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Construct and behavior-probe the declared augmentation contract."""
    contract = (
        copy.deepcopy(contract_override)
        if contract_override is not None
        else augmentation_contract(profile)
    )
    if contract.get("profile") != profile:
        raise ProtocolError("augmentation profile and contract disagree")
    if profile == "none":
        _build_from_contract(contract)
        return {
            "status": "passed",
            "contract_sha256": canonical_sha256(contract),
            "gaussian_kernel_sizes": [],
            "motion_kernel_sizes": [],
            "full_seed_repeat_equal": None,
            "high_bit_changes_output": None,
        }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = _build_from_contract(contract)
        if pipeline is None:
            raise ProtocolError("training augmentation unexpectedly disabled")
        if _normalized(pipeline.p) != _normalized(contract.get("compose_p")):
            raise ProtocolError("Compose.p differs from its declared value")
        declared = contract["transforms"]
        for child_spec, child in zip(
            declared, pipeline.transforms, strict=True
        ):
            _assert_declared_attributes(
                _require_mapping(child_spec, "transform"),
                child,
            )

        gaussian_sizes: set[int] = set()
        motion_sizes: set[int] = set()
        image = _tiny_image()
        for top_spec, top_transform in zip(
            declared, pipeline.transforms, strict=True
        ):
            for leaf_spec, leaf in _iter_leaves(
                _require_mapping(top_spec, "transform"),
                top_transform,
            ):
                probability = leaf_spec.get("p")
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or probability <= 0
                ):
                    raise ProtocolError(
                        f"{leaf_spec.get('type')}.p must be p > 0 for probing"
                    )
                leaf.set_random_seed(0)
                output = leaf(image=image, force_apply=True)["image"]
                if not isinstance(output, np.ndarray) or output.shape != image.shape:
                    raise ProtocolError(
                        f"{leaf_spec.get('type')} changed the probe image shape"
                    )
                if leaf_spec.get("type") == "GaussianBlur":
                    gaussian_sizes.update(
                        _kernel_sizes("GaussianBlur", leaf_spec, leaf)
                    )
                elif leaf_spec.get("type") == "MotionBlur":
                    motion_sizes.update(
                        _kernel_sizes("MotionBlur", leaf_spec, leaf)
                    )

        repeat_equal, high_bit_changes = _probe_full_seed_support()
        contract_warnings = [
            item for item in caught if _is_contract_warning(item)
        ]
        if contract_warnings:
            messages = "; ".join(
                str(item.message) for item in contract_warnings
            )
            raise ProtocolError(
                f"augmentation construction or probes emitted warnings: {messages}"
            )
    if not repeat_equal:
        raise ProtocolError(
            "Albumentations did not reproduce a full 256-bit seed"
        )
    if not high_bit_changes:
        raise ProtocolError(
            "Albumentations appears to truncate seed bits above bit 32"
        )
    return {
        "status": "passed",
        "contract_sha256": canonical_sha256(contract),
        "gaussian_kernel_sizes": sorted(gaussian_sizes),
        "motion_kernel_sizes": sorted(motion_sizes),
        "full_seed_repeat_equal": repeat_equal,
        "high_bit_changes_output": high_bit_changes,
    }
