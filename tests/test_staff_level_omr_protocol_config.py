from pathlib import Path

import pytest

from experiments.staff_level_omr.protocol import (
    ProtocolError,
    StaffOMRConfig,
    canonical_json_bytes,
    canonical_sha256,
    read_json,
)


REVISION = "a" * 40


@pytest.fixture
def config_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_path = tmp_path / "data"
    bundle_path = tmp_path / "bundle"
    output_root = tmp_path / "runs"
    data_path.mkdir()
    bundle_path.mkdir()
    return data_path, bundle_path, output_root


def _create_config(config_paths, **overrides):
    data_path, bundle_path, output_root = config_paths
    values = {
        "experiment_name": "fixture-lora",
        "data_path": data_path,
        "dataset_bundle_path": bundle_path,
        "output_root": output_root,
        "model_name": "musvit",
        "approved_revisions": {"musvit": {REVISION}},
        "default_revisions": {"musvit": REVISION},
    }
    values.update(overrides)
    return StaffOMRConfig.create(**values)


def test_canonical_json_is_compact_utf8_and_order_independent():
    left = {"z": "\u8c31", "a": [1, 2]}
    right = {"a": [1, 2], "z": "\u8c31"}

    encoded = canonical_json_bytes(left)

    assert encoded == b'{"a":[1,2],"z":"\xe8\xb0\xb1"}'
    assert not encoded.endswith(b"\n")
    assert canonical_sha256(left) == canonical_sha256(right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_non_finite_numbers(value):
    with pytest.raises(ProtocolError, match="finite"):
        canonical_json_bytes({"learning_rate": value})


def test_json_reader_wraps_invalid_utf8_as_protocol_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_bytes(b'{"value":"\xff"}')

    with pytest.raises(ProtocolError, match="invalid UTF-8 JSON"):
        read_json(path)


def test_config_normalizes_legacy_method_and_derives_geometry(config_paths):
    config = _create_config(
        config_paths,
        method="linear_prob",
        start_eval=1000,
        max_epochs=1000,
    )

    assert config.method == "linear_probe"
    assert config.input_geometry == "native_pad"
    assert config.model_revision == REVISION
    assert config.patch_rows == 8
    assert config.patch_cols == 64


def test_lora_derives_exact_grid(config_paths):
    config = _create_config(config_paths, method="lora")

    assert config.input_geometry == "exact_grid"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_eval": 0}, "start_eval"),
        ({"start_eval": 11, "max_epochs": 10}, "start_eval"),
        ({"method": "full"}, "method"),
        ({"patch_rows": 0}, "patch_rows"),
        ({"patch_cols": -1}, "patch_cols"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"patience": 0}, "patience"),
        ({"seed": -1}, "seed"),
        ({"device": "gpu"}, "device"),
        ({"augmentation_profile": "mystery"}, "augmentation_profile"),
        ({"verify_image_hashes": "never"}, "verify_image_hashes"),
    ],
)
def test_config_rejects_invalid_fields(config_paths, overrides, message):
    with pytest.raises(ProtocolError, match=message):
        _create_config(config_paths, **overrides)


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "bad name", "bad/name", "x" * 65, "CON", "con.txt", "LPT9"],
)
def test_config_rejects_unsafe_experiment_names(config_paths, name):
    with pytest.raises(ProtocolError, match="experiment_name"):
        _create_config(config_paths, experiment_name=name)


def test_config_rejects_explicit_input_geometry(config_paths):
    with pytest.raises(ProtocolError, match="input_geometry"):
        _create_config(config_paths, input_geometry="exact_grid")


def test_config_rejects_new_and_legacy_patch_arguments_together(config_paths):
    with pytest.raises(ProtocolError, match="shape_patches"):
        _create_config(
            config_paths,
            patch_rows=8,
            patch_cols=64,
            shape_patches=(8, 128),
        )


def test_config_accepts_legacy_patch_tuple_when_new_fields_are_absent(config_paths):
    config = _create_config(config_paths, shape_patches=(12, 96))

    assert (config.patch_rows, config.patch_cols) == (12, 96)


def test_config_requires_exclusions_only_for_exclude_listed(config_paths, tmp_path):
    exclusions = tmp_path / "exclude.json"
    exclusions.write_text("{}", encoding="utf-8")

    config = _create_config(
        config_paths,
        train_infeasible_policy="exclude_listed",
        train_exclusions_path=exclusions,
    )
    assert config.train_exclusions_path == exclusions.resolve()

    with pytest.raises(ProtocolError, match="train_exclusions_path"):
        _create_config(
            config_paths,
            train_infeasible_policy="exclude_listed",
            train_exclusions_path=None,
        )

    with pytest.raises(ProtocolError, match="train_exclusions_path"):
        _create_config(
            config_paths,
            train_infeasible_policy="fail",
            train_exclusions_path=exclusions,
        )


def test_config_revision_resolution_must_be_explicit_and_approved(config_paths):
    resolved = _create_config(
        config_paths,
        resolve_model_revision=True,
        revision_resolver=lambda model_name: REVISION,
    )
    assert resolved.model_revision == REVISION

    with pytest.raises(ProtocolError, match="mutually exclusive"):
        _create_config(
            config_paths,
            model_revision=REVISION,
            resolve_model_revision=True,
            revision_resolver=lambda model_name: REVISION,
        )

    with pytest.raises(ProtocolError, match="approved"):
        _create_config(config_paths, model_revision="b" * 40)

    with pytest.raises(ProtocolError, match="revision_resolver"):
        _create_config(config_paths, resolve_model_revision=True)


def test_launch_config_is_json_native_and_tracks_runtime_paths(config_paths):
    config = _create_config(config_paths, num_workers=2)

    launch = config.to_launch_config()

    assert launch["data_path"] == str(config.data_path)
    assert launch["dataset_bundle_path"] == str(config.dataset_bundle_path)
    assert launch["output_root"] == str(config.output_root)
    assert launch["num_workers"] == 2
    assert launch["input_geometry"] == "exact_grid"
    canonical_json_bytes(launch)
