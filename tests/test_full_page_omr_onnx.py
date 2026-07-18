import pytest
from transformers import ViTModel

from experiments.full_page_omr.smt_foundation.configuration_smt import (
    SMTFoundationConfig,
)
from experiments.full_page_omr.smt_foundation.modeling_smt import (
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
