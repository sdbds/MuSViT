import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def test_export_script_uses_tracked_foundation_config():
    root = Path(__file__).resolve().parents[1]
    relative_config = (
        "experiments/full_page_omr/config/LSMT-MAE-Base-1024-16.json"
    )

    assert (root / relative_config).is_file()
    assert relative_config in (root / "3.export_full_page_omr_onnx.ps1").read_text(
        encoding="utf-8"
    )


def test_checkpoint_converter_writes_inference_only_assets(tmp_path):
    checkpoint_path = tmp_path / "model.ckpt"
    weights_path = tmp_path / "model.safetensors"
    config_path = tmp_path / "model.config.json"
    torch.save(
        {
            "epoch": 12,
            "global_step": 34,
            "state_dict": {
                "model.encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "model.decoder.bias": torch.tensor([1.0, 2.0]),
            },
            "hyper_parameters": {
                "smt_config": {
                    "maxlen": 8,
                    "w2i": {"<bos>": 1, "<eos>": 2},
                    "i2w": {1: "<bos>", 2: "<eos>"},
                }
            },
            "optimizer_states": [{"state": {"large": "training-only"}}],
        },
        checkpoint_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.full_page_omr.convert_checkpoint",
            "--checkpoint-path",
            str(checkpoint_path),
            "--weights-path",
            str(weights_path),
            "--config-path",
            str(config_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert set(load_file(weights_path)) == {"encoder.weight", "decoder.bias"}
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "maxlen": 8,
        "w2i": {"<bos>": 1, "<eos>": 2},
        "i2w": {"1": "<bos>", "2": "<eos>"},
    }
    assert "optimizer" not in config_path.read_text(encoding="utf-8")
