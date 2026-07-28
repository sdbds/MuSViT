from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from musvit import cli


GROUP_REGEX = r"(?P<group_id>score[0-9]+)/staff_region[.]png"


def _make_cli_fixture(root: Path) -> Path:
    data_path = root / "data"
    for index in range(3):
        group = data_path / f"score{index:02d}"
        group.mkdir(parents=True)
        Image.new("RGB", (24, 16), (30 + index, 50, 90)).save(
            group / "staff_region.png"
        )
        (group / "staff_gt.txt").write_text(
            f"clef note-{index}", encoding="utf-8"
        )
    return data_path


def test_staff_loader_exposes_canonical_subcommands():
    target = cli._load_staff_level_omr()

    assert set(target) == {"train", "prepare-data", "resume"}
    assert callable(target["train"])
    assert callable(target["prepare-data"])
    assert callable(target["resume"])


def test_default_subcommand_routes_only_legacy_option_form():
    train = object()
    prepare = object()
    target = {"train": train, "prepare-data": prepare, "resume": object()}
    experiment = cli.Experiment(
        command="staff-level-omr",
        location="experiments/staff_level_omr",
        loader=lambda: target,
        help_rows=(),
        default_subcommand="train",
    )

    selected, arguments = cli._select_fire_target(
        experiment, target, ["--method=lora"]
    )
    assert selected is train
    assert arguments == ["--method=lora"]

    selected, arguments = cli._select_fire_target(
        experiment, target, ["train", "--method=lora"]
    )
    assert selected is target
    assert arguments == ["train", "--method=lora"]

    selected, arguments = cli._select_fire_target(
        experiment, target, ["prepare-data", "--data_path=x"]
    )
    assert selected is target
    assert arguments == ["prepare-data", "--data_path=x"]


def test_prepare_data_public_cli_generates_complete_bundle(tmp_path):
    data_path = _make_cli_fixture(tmp_path)
    bundle_path = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "musvit.cli",
            "staff-level-omr",
            "prepare-data",
            "--data_path",
            str(data_path),
            "--dataset_id",
            "fixture",
            "--group_regex",
            GROUP_REGEX,
            "--split_ratios",
            "0.8",
            "0.1",
            "0.1",
            "--seed",
            "7",
            "--out",
            str(bundle_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    assert result.returncode == 0, result.stderr
    assert {
        path.name for path in bundle_path.iterdir() if path.is_file()
    } == {
        "bundle.json",
        "split_manifest.json",
        "vocabulary.json",
        "image_verification_index.json",
    }
    assert "bundle_sha256" in result.stdout


def test_cli_discovery_lists_prepare_and_explicit_train():
    result = subprocess.run(
        [sys.executable, "-m", "musvit.cli", "list"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    assert result.returncode == 0, result.stderr
    assert "staff-level-omr prepare-data" in result.stdout
    assert "staff-level-omr train" in result.stdout
    assert "staff-level-omr resume" in result.stdout
