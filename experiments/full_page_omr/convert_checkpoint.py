from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file


def _extract_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise RuntimeError("checkpoint hyper_parameters must be a mapping")
    config = hyper_parameters.get("smt_config")
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        raise RuntimeError("checkpoint hyper_parameters.smt_config must be a mapping")
    return dict(config)


def _extract_inference_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("checkpoint state_dict must be a non-empty mapping")
    names = [str(name) for name in state]
    prefixed = [name.startswith("model.") for name in names]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("checkpoint state_dict mixes model-prefixed and unprefixed keys")

    inference_state = {}
    for raw_name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"checkpoint state value is not a tensor: {raw_name}")
        name = str(raw_name)
        if all(prefixed):
            name = name.removeprefix("model.")
        inference_state[name] = value.detach().cpu().contiguous()
    return inference_state


def convert_checkpoint(
    checkpoint_path: str | Path,
    weights_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_path = Path(weights_path).resolve()
    config_path = Path(config_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint root must be a mapping")
    state = _extract_inference_state(checkpoint)
    config = _extract_model_config(checkpoint)
    config_text = json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_weights = weights_path.with_name(weights_path.name + ".tmp")
    temporary_config = config_path.with_name(config_path.name + ".tmp")
    temporary_weights.unlink(missing_ok=True)
    temporary_config.unlink(missing_ok=True)
    try:
        save_file(state, str(temporary_weights))
        temporary_config.write_text(config_text, encoding="utf-8")
        os.replace(temporary_weights, weights_path)
        os.replace(temporary_config, config_path)
    finally:
        temporary_weights.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)

    return {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "tensor_count": len(state),
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract inference-only safetensors and config from an OMR checkpoint."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = convert_checkpoint(
        args.checkpoint_path,
        args.weights_path,
        args.config_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
