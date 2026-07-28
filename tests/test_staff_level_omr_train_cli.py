from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.staff_level_omr import arguments, entrypoint
from experiments.staff_level_omr.protocol.backbone import (
    APPROVED_BACKBONES,
)
from experiments.staff_level_omr.protocol.errors import ProtocolError


def _paths(tmp_path: Path):
    data = tmp_path / "data"
    bundle = tmp_path / "bundle"
    data.mkdir()
    bundle.mkdir()
    return data, bundle


def test_fire_train_requires_three_user_owned_identity_paths():
    signature = inspect.signature(entrypoint.train)

    assert signature.parameters["experiment_name"].default is inspect.Parameter.empty
    assert signature.parameters["data_path"].default is inspect.Parameter.empty
    assert (
        signature.parameters["dataset_bundle_path"].default
        is inspect.Parameter.empty
    )


def test_build_config_normalizes_transitional_method_and_patch_aliases(tmp_path):
    data, bundle = _paths(tmp_path)

    config = entrypoint.build_train_config(
        experiment_name="fixture",
        data_path=data,
        dataset_bundle_path=bundle,
        method="linear_prob",
        shape_patches=(8, 64),
        start_eval=1,
        max_epochs=1,
        device="cpu",
    )

    assert config.method == "linear_probe"
    assert config.input_geometry == "native_pad"
    assert config.patch_rows == 8
    assert config.patch_cols == 64
    assert config.model_revision == APPROVED_BACKBONES["musvit"].revision


def test_build_config_rejects_mixed_old_and_new_patch_arguments(tmp_path):
    data, bundle = _paths(tmp_path)

    with pytest.raises(ProtocolError, match="shape_patches"):
        entrypoint.build_train_config(
            experiment_name="fixture",
            data_path=data,
            dataset_bundle_path=bundle,
            patch_rows=8,
            patch_cols=64,
            shape_patches=(8, 64),
            start_eval=1,
            max_epochs=1,
            device="cpu",
        )


def test_build_config_rejects_non_sequence_patch_alias_as_protocol_error(
    tmp_path,
):
    data, bundle = _paths(tmp_path)

    with pytest.raises(ProtocolError, match="shape_patches"):
        entrypoint.build_train_config(
            experiment_name="fixture",
            data_path=data,
            dataset_bundle_path=bundle,
            shape_patches=8,
            start_eval=1,
            max_epochs=1,
            device="cpu",
        )


def test_entrypoint_calls_only_v2_runtime(monkeypatch, tmp_path):
    data, bundle = _paths(tmp_path)
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return tmp_path / "run"

    monkeypatch.setattr(entrypoint, "run_training", fake_train)

    result = entrypoint.train(
        experiment_name="fixture",
        data_path=str(data),
        dataset_bundle_path=str(bundle),
        start_eval=1,
        max_epochs=1,
        device="cpu",
    )

    assert result == str(tmp_path / "run")
    assert captured["config"].experiment_name == "fixture"
    assert captured["config"].data_path == data.resolve()


def test_argparse_uses_same_json_array_patch_alias_as_fire():
    args = arguments.parser_train.parse_args(
        [
            "--experiment_name",
            "fixture",
            "--data_path",
            r"D:\scores",
            "--dataset_bundle_path",
            r"D:\bundle",
            "--method",
            "linear_prob",
            "--shape_patches",
            "[8,64]",
            "--learning_rate",
            "0.001",
        ]
    )

    assert args.experiment_name == "fixture"
    assert args.data_path == r"D:\scores"
    assert args.dataset_bundle_path == r"D:\bundle"
    assert args.method == "linear_prob"
    assert args.shape_patches == [8, 64]
    assert args.learning_rate == 0.001
    assert not hasattr(args, "ds_name")
    assert not hasattr(args, "lr")


def test_namespace_adapter_builds_same_normalized_config(tmp_path):
    data, bundle = _paths(tmp_path)
    namespace = argparse.Namespace(
        experiment_name="fixture",
        data_path=str(data),
        dataset_bundle_path=str(bundle),
        model_name="musvit",
        model_revision=None,
        resolve_model_revision=False,
        method="lora",
        patch_rows=8,
        patch_cols=64,
        shape_patches=None,
        augmentation_profile="none",
        train_infeasible_policy="fail",
        train_exclusions_path=None,
        batch_size=2,
        num_workers=0,
        learning_rate=0.001,
        max_epochs=1,
        start_eval=1,
        patience=2,
        seed=7,
        output_root=str(tmp_path / "runs"),
        device="cpu",
        verify_image_hashes="always",
    )

    config = arguments.config_from_namespace(namespace)

    assert config.to_launch_config() == entrypoint.build_train_config(
        **vars(namespace)
    ).to_launch_config()


def test_resume_parser_exposes_only_allowed_overrides():
    args = arguments.parser_resume.parse_args(
        [
            r"D:\run",
            "--max_epochs",
            "20",
            "--num_workers",
            "2",
            "--allow_env_drift",
        ]
    )

    assert args.run_dir == r"D:\run"
    assert args.max_epochs == 20
    assert args.num_workers == 2
    assert args.allow_env_drift is True
    assert not hasattr(args, "method")


def test_executable_v1_runtime_modules_are_removed():
    package = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "staff_level_omr"
    )

    obsolete = (
        "augments.py",
        "config.py",
        "datasets.py",
        "executions.sh",
        "model.py",
        "utils/data_utils.py",
        "utils/utils.py",
    )
    assert [name for name in obsolete if (package / name).exists()] == []


def test_fire_help_uses_v2_names_and_lists_allowed_values():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "musvit.cli",
            "staff-level-omr",
            "train",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
        env={
            **os.environ,
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )

    output = (result.stdout + result.stderr).replace("`", "")
    assert result.returncode == 0, output
    assert "EXPERIMENT_NAME" in output
    assert "DATASET_BUNDLE_PATH" in output
    assert "linear_probe, compatibility alias linear_prob, or lora" in output
    assert "cpu, cuda, or auto" in output
    assert "always or cached" in output
    assert "--ds_name" not in output
    assert "\n    --lr=" not in output


def test_root_readme_staff_section_describes_only_v2_workflow():
    readme = (
        Path(__file__).resolve().parents[1] / "README.md"
    ).read_text(encoding="utf-8")
    section = readme.split(
        "### Staff-level OMR (fine-tuning)",
        maxsplit=1,
    )[1].split("### Object Detection", maxsplit=1)[0]

    assert "staff-level-omr prepare-data" in section
    assert "staff-level-omr train" in section
    assert "--dataset_bundle_path" in section
    assert "-DatasetBundlePath" in section
    assert "--ds_name" not in section
    assert "staff_level_omr/config.py" not in section
    assert "_ctc.pt" not in section
