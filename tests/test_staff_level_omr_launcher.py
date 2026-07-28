import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "4.staff_level_omr.ps1"


def _dataset(root: Path) -> Path:
    nested = root / "scores" / "work"
    nested.mkdir(parents=True)
    (nested / "page_staff_region.png").touch()
    (nested / "page_staff_gt.txt").write_text("note", encoding="utf-8")
    return root / "scores"


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    for name in (
        "bundle.json",
        "split_manifest.json",
        "vocabulary.json",
        "image_verification_index.json",
    ):
        (bundle / name).write_text("{}", encoding="utf-8")
    return bundle


def _run_launcher(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(LAUNCHER),
            *map(str, arguments),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )


def test_launcher_leaves_model_dependent_geometry_to_v2_preflight():
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "$PatchColumns -ne 64" not in script
    assert "native patch_cols value 64" not in script


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="PowerShell launcher is Windows-specific",
)
def test_powershell_dry_run_builds_only_v2_staff_command(tmp_path):
    data = _dataset(tmp_path)
    bundle = _bundle(tmp_path)

    result = _run_launcher(
        "-DryRun",
        "-DataPath",
        data,
        "-DatasetBundlePath",
        bundle,
    )

    assert result.returncode == 0, result.stderr
    assert "uv run --frozen musvit staff-level-omr train" in result.stdout
    assert "--experiment_name=catedrales" in result.stdout
    assert f"--data_path={data}" in result.stdout
    assert f"--dataset_bundle_path={bundle}" in result.stdout
    assert "--method=lora" in result.stdout
    assert "--patch_rows=8" in result.stdout
    assert "--patch_cols=128" in result.stdout
    assert "--learning_rate=0.0003" in result.stdout
    assert "--verify_image_hashes=always" in result.stdout
    assert "--ds_name" not in result.stdout
    assert "--shape_patches" not in result.stdout
    assert "--lr=" not in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="PowerShell launcher is Windows-specific",
)
def test_powershell_rejects_dataset_without_pairs(tmp_path):
    data = tmp_path / "scores"
    data.mkdir()
    bundle = _bundle(tmp_path)

    result = _run_launcher(
        "-DryRun",
        "-DataPath",
        data,
        "-DatasetBundlePath",
        bundle,
    )

    assert result.returncode != 0
    assert "paired *_region.png and *_gt.txt" in result.stderr


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="PowerShell launcher is Windows-specific",
)
def test_powershell_rejects_incomplete_dataset_bundle(tmp_path):
    data = _dataset(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    result = _run_launcher(
        "-DryRun",
        "-DataPath",
        data,
        "-DatasetBundlePath",
        bundle,
    )

    assert result.returncode != 0
    assert "dataset_bundle_path is missing required file" in result.stderr
