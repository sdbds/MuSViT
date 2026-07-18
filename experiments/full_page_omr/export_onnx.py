from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .smt_foundation.configuration_smt import SMTFoundationConfig
from .smt_foundation.modeling_smt import MHA, SMTFoundationModelForCausalLM


def _read_json_object(path: str | Path, *, label: str) -> dict:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {resolved}")
    return value


def load_standalone_model(
    weights_path: str | Path,
    model_config_path: str | Path,
    encoder_config_path: str | Path,
) -> tuple[SMTFoundationModelForCausalLM, dict]:
    model_config = _read_json_object(model_config_path, label="model config")
    encoder_config = _read_json_object(encoder_config_path, label="encoder config")
    merged_config = dict(model_config)
    merged_config["foundation_config"] = encoder_config
    merged_config["attention_backend"] = "eager"

    config = SMTFoundationConfig(**merged_config)
    model = SMTFoundationModelForCausalLM(config)
    state = load_file(str(Path(weights_path).resolve()), device="cpu")
    model.load_state_dict(state, strict=True)
    for module in model.modules():
        if isinstance(module, MHA):
            module.attention_backend = "eager"
    model.eval()
    return model, config.to_dict()


class FullPageOMREncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.model.forward_encoder(pixel_values)
        prepared = self.model._prepare_decoder_features(
            encoder_output.permute(0, 2, 1).contiguous()
        )
        return (
            prepared.raw_features.permute(1, 0, 2).contiguous(),
            prepared.enhanced_features.permute(1, 0, 2).contiguous(),
        )


class FullPageOMRDecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        raw_features: torch.Tensor,
        enhanced_features: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        raw_sequence = raw_features.permute(1, 0, 2).contiguous()
        enhanced_sequence = enhanced_features.permute(1, 0, 2).contiguous()
        positioned = self.decoder.embedding(token_ids).permute(0, 2, 1)
        positioned = self.decoder.positional_1D(positioned, start=0)
        positioned = positioned.permute(2, 0, 1).contiguous()
        output, _, _ = self.decoder.decoder(
            positioned,
            memory_key=enhanced_sequence,
            memory_value=raw_sequence,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            use_cache=False,
            cache=None,
            predict_last_n_only=False,
            keep_all_weights=False,
            self_attention_is_causal=True,
            self_attention_window=(-1, -1),
        )
        projected = self.decoder.dropout(self.decoder.end_relu(output[-1:]))
        return self.decoder.out_layer(
            projected.permute(1, 2, 0).contiguous()
        ).squeeze(-1)
