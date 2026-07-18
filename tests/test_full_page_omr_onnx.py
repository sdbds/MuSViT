import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import ViTModel

from experiments.full_page_omr.export_onnx import (
    FullPageOMRDecoderWrapper,
    FullPageOMREncoderWrapper,
    load_standalone_model,
)
from experiments.full_page_omr.smt_foundation.configuration_smt import (
    SMTFoundationConfig,
)
from experiments.full_page_omr.smt_foundation.modeling_smt import (
    Decoder,
    PreparedDecoderFeatures,
    SMTFoundationModelForCausalLM,
)


def _small_encoder_config():
    return {
        "attention_probs_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 768,
        "image_size": 32,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "num_attention_heads": 12,
        "num_channels": 3,
        "num_hidden_layers": 1,
        "patch_size": 16,
        "qkv_bias": True,
    }


def _small_smt_config(*, foundation_config=None):
    return SMTFoundationConfig(
        foundation_architecture="ViTMAEBase",
        foundation_weights="must-not-be-loaded",
        foundation_config=foundation_config,
        maxh=32,
        maxw=32,
        maxlen=8,
        out_categories=5,
        padding_token=0,
        in_channels=3,
        w2i={"<bos>": 1, "<eos>": 2},
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>"},
        out_dir="test",
        d_model=256,
        dim_ff=256,
        num_dec_layers=1,
        attention_backend="eager",
    )


def test_config_retains_embedded_foundation_config():
    embedded = _small_encoder_config()

    config = _small_smt_config(foundation_config=embedded)

    assert config.foundation_config == embedded


def test_model_uses_embedded_config_without_loading_foundation_weights(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("from_pretrained must not be called")

    monkeypatch.setattr(ViTModel, "from_pretrained", fail_if_called)

    model = SMTFoundationModelForCausalLM(
        _small_smt_config(foundation_config=_small_encoder_config())
    )

    assert model.encoder.config.image_size == 32
    assert model.encoder.config.hidden_size == 768
    assert model.encoder.config.num_hidden_layers == 1


def test_model_rejects_non_mapping_embedded_foundation_config():
    with pytest.raises(TypeError, match="foundation_config must be a dictionary"):
        SMTFoundationModelForCausalLM(
            _small_smt_config(foundation_config=["not", "a", "mapping"])
        )


class _FakeEncoderModel(torch.nn.Module):
    def forward_encoder(self, pixel_values):
        batch = pixel_values.shape[0]
        return torch.arange(
            batch * 5 * 16,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        ).reshape(batch, 5, 16)

    def _prepare_decoder_features(self, encoder_output):
        raw = encoder_output[:, :, 1:].permute(2, 0, 1).contiguous()
        return PreparedDecoderFeatures(
            raw_features=raw,
            enhanced_features=raw + 10.0,
            reduced_size=((2, 2),),
            feature_size=torch.Size((encoder_output.shape[0], 16, 2, 2)),
        )


class _DecoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(
            d_model=16,
            dim_ff=32,
            n_layers=2,
            maxlen=8,
            out_categories=5,
            attention_window=9,
            attention_backend="eager",
        )


def _reference_full_prefix_logits(decoder, raw_features, enhanced_features, token_ids):
    raw_sequence = raw_features.permute(1, 0, 2).contiguous()
    enhanced_sequence = enhanced_features.permute(1, 0, 2).contiguous()
    positioned = decoder.embedding(token_ids).permute(0, 2, 1)
    positioned = decoder.positional_1D(positioned, start=0)
    positioned = positioned.permute(2, 0, 1).contiguous()
    output, _, _ = decoder.decoder(
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
    projected = decoder.dropout(decoder.end_relu(output[-1:]))
    return decoder.out_layer(projected.permute(1, 2, 0).contiguous()).squeeze(-1)


def test_encoder_wrapper_returns_batch_first_prepared_features():
    pixels = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    raw, enhanced = FullPageOMREncoderWrapper(_FakeEncoderModel())(pixels)

    assert raw.shape == (1, 4, 16)
    assert enhanced.shape == (1, 4, 16)
    torch.testing.assert_close(enhanced, raw + 10.0)


def test_decoder_wrapper_matches_uncached_full_prefix_reference():
    torch.manual_seed(7)
    model = _DecoderModel().eval()
    raw = torch.randn((1, 4, 16), dtype=torch.float32)
    enhanced = torch.randn((1, 4, 16), dtype=torch.float32)
    token_ids = torch.tensor([[1, 3, 4]], dtype=torch.long)

    actual = FullPageOMRDecoderWrapper(model)(raw, enhanced, token_ids)
    expected = _reference_full_prefix_logits(
        model.decoder,
        raw,
        enhanced,
        token_ids,
    )

    assert actual.shape == (1, 5)
    torch.testing.assert_close(actual, expected)


def _write_small_model_assets(tmp_path, *, remove_state_key=False):
    encoder_config = _small_encoder_config()
    source_model = SMTFoundationModelForCausalLM(
        _small_smt_config(foundation_config=encoder_config)
    )
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in source_model.state_dict().items()
    }
    if remove_state_key:
        state.pop(next(iter(state)))

    weights_path = tmp_path / "model.safetensors"
    model_config_path = tmp_path / "model.config.json"
    encoder_config_path = tmp_path / "encoder.config.json"
    save_file(state, weights_path)
    model_config_path.write_text(
        json.dumps(_small_smt_config().to_dict()),
        encoding="utf-8",
    )
    encoder_config_path.write_text(json.dumps(encoder_config), encoding="utf-8")
    return weights_path, model_config_path, encoder_config_path


def test_standalone_loader_strictly_loads_complete_state(tmp_path, monkeypatch):
    assets = _write_small_model_assets(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("from_pretrained must not be called")

    monkeypatch.setattr(ViTModel, "from_pretrained", fail_if_called)
    model, merged_config = load_standalone_model(*assets)

    assert model.training is False
    assert merged_config["foundation_config"] == _small_encoder_config()
    assert len(model.state_dict()) > 0


def test_standalone_loader_rejects_missing_state_key(tmp_path):
    assets = _write_small_model_assets(tmp_path, remove_state_key=True)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_standalone_model(*assets)
