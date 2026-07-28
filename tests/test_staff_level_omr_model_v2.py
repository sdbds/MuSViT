import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import ViTConfig, ViTModel

from experiments.staff_level_omr.protocol.backbone import BackboneMetadata
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol.modeling import (
    LORA_CONTRACT,
    TASK_HEAD_SCHEMA,
    build_model,
    greedy_ctc_decode,
    load_trainable_state_dict,
    task_head_contract,
    task_head_state_keys,
    trainable_state_dict,
)
from experiments.staff_level_omr.protocol.optimization import (
    build_optimizer,
    optimizer_contract,
    optimizer_parameter_names,
)


META = BackboneMetadata(
    model_id="fixture/vit",
    revision="a" * 40,
    image_height=32,
    image_width=32,
    patch_height=8,
    patch_width=8,
    hidden_size=16,
    num_channels=3,
    prefix_tokens=1,
    model_type="vit_mae",
    architectures=("ViTMAEForPreTraining",),
)

EXPECTED_HEAD_KEYS = {
    "projection.weight",
    "rnn.weight_ih_l0",
    "rnn.weight_hh_l0",
    "rnn.bias_ih_l0",
    "rnn.bias_hh_l0",
    "rnn.weight_ih_l0_reverse",
    "rnn.weight_hh_l0_reverse",
    "rnn.bias_ih_l0_reverse",
    "rnn.bias_hh_l0_reverse",
    "rnn.weight_ih_l1",
    "rnn.weight_hh_l1",
    "rnn.bias_ih_l1",
    "rnn.bias_hh_l1",
    "rnn.weight_ih_l1_reverse",
    "rnn.weight_hh_l1_reverse",
    "rnn.bias_ih_l1_reverse",
    "rnn.bias_hh_l1_reverse",
    "classifier_ctc.weight",
    "classifier_ctc.bias",
}


class RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, *, pixel_values, interpolate_pos_encoding):
        self.calls.append(interpolate_pos_encoding)
        rows = pixel_values.shape[2] // META.patch_height
        cols = pixel_values.shape[3] // META.patch_width
        values = torch.arange(
            pixel_values.shape[0] * (1 + rows * cols) * META.hidden_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        hidden = values.reshape(
            pixel_values.shape[0],
            1 + rows * cols,
            META.hidden_size,
        )
        return SimpleNamespace(last_hidden_state=hidden)


def _config(method: str, *, rows: int = 2, cols: int = 4, seed: int = 7):
    return SimpleNamespace(
        method=method,
        input_geometry=(
            "native_pad" if method == "linear_probe" else "exact_grid"
        ),
        patch_rows=rows,
        patch_cols=cols,
        seed=seed,
    )


def _tiny_vit():
    return ViTModel(
        ViTConfig(
            image_size=32,
            patch_size=8,
            num_channels=3,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            add_pooling_layer=False,
        ),
        add_pooling_layer=False,
    )


def test_task_head_contract_is_complete_and_exact():
    assert task_head_contract(META, num_classes=9) == {
        "schema_version": TASK_HEAD_SCHEMA,
        "spatial_order": "rows_then_columns",
        "input_dropout": {
            "p": 0.25,
            "position": "before_projection",
        },
        "projection": {
            "in_features": 16,
            "out_features": 256,
            "bias": False,
        },
        "row_pool": {
            "operation": "mean",
            "axis": "rows",
            "position": "after_projection",
        },
        "rnn": {
            "type": "LSTM",
            "input_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "bias": True,
            "batch_first": True,
            "dropout": 0.5,
            "bidirectional": True,
            "proj_size": 0,
            "initial_state": "zeros_same_dtype_and_device_as_input",
        },
        "classifier": {
            "in_features": 512,
            "out_features": 9,
            "bias": True,
        },
        "output": {"log_softmax_dim": -1},
        "decoder": {
            "type": "greedy_ctc",
            "collapse_repeats": True,
            "remove_blank_id": 0,
        },
    }


def test_linear_probe_builds_exact_head_and_freezes_only_backbone():
    backbone = RecordingBackbone()
    model = build_model(backbone, META, _config("linear_probe"), num_classes=9)

    assert model.projection.weight.shape == (256, 16)
    assert model.projection.bias is None
    assert model.rnn.input_size == 256
    assert model.rnn.hidden_size == 256
    assert model.rnn.num_layers == 2
    assert model.rnn.dropout == 0.5
    assert model.rnn.bidirectional is True
    assert model.rnn.proj_size == 0
    assert model.classifier_ctc.weight.shape == (9, 512)
    assert task_head_state_keys(model) == EXPECTED_HEAD_KEYS
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )


def test_linear_and_lora_paths_have_same_time_axis_for_same_grid():
    linear = build_model(
        RecordingBackbone(),
        META,
        _config("linear_probe"),
        num_classes=9,
    )
    lora = build_model(_tiny_vit(), META, _config("lora"), num_classes=9)

    linear_output = linear(torch.zeros((2, 3, 32, 32)))
    lora_output = lora(torch.zeros((2, 3, 16, 32)))

    assert linear_output.shape == lora_output.shape == (2, 4, 9)
    assert linear.backbone.calls == [False]


def test_lora_trains_only_adapter_and_complete_task_head():
    model = build_model(_tiny_vit(), META, _config("lora"), num_classes=9)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert LORA_CONTRACT == {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "bias": "none",
        "target_modules": ["query", "key", "value"],
        "use_rslora": True,
    }
    assert EXPECTED_HEAD_KEYS <= trainable
    assert any("lora_" in name for name in trainable)
    assert not any(
        name.startswith("backbone.")
        and "lora_" not in name
        and parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_trainable_initialization_is_independent_of_prior_rng_consumption():
    first_backbone = _tiny_vit()
    second_backbone = copy.deepcopy(first_backbone)
    torch.manual_seed(111)
    torch.rand(37)
    first = build_model(
        first_backbone,
        META,
        _config("lora", seed=23),
        num_classes=9,
    )
    torch.manual_seed(999)
    torch.rand(91)
    second = build_model(
        second_backbone,
        META,
        _config("lora", seed=23),
        num_classes=9,
    )

    first_state = trainable_state_dict(first)
    second_state = trainable_state_dict(second)

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(
            first_state[name],
            second_state[name],
            rtol=0,
            atol=0,
        )


def test_trainable_state_loader_restores_exact_named_tensors():
    model = build_model(
        RecordingBackbone(),
        META,
        _config("linear_probe"),
        num_classes=9,
    )
    expected = trainable_state_dict(model)
    with torch.no_grad():
        model.projection.weight.add_(1)

    load_trainable_state_dict(model, expected)

    actual = trainable_state_dict(model)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_model_rejects_config_geometry_that_disagrees_with_method():
    config = _config("lora")
    config.input_geometry = "native_pad"

    with pytest.raises(ProtocolError, match="input_geometry"):
        build_model(_tiny_vit(), META, config, num_classes=9)


def test_greedy_decoder_collapses_repeats_and_blank_resets_repeat():
    ids = torch.tensor(
        [
            [0, 1, 1, 0, 1, 2, 2],
            [3, 3, 0, 0, 3, 0, 4],
        ]
    )
    log_probs = torch.full((2, 7, 5), -100.0)
    log_probs.scatter_(2, ids.unsqueeze(-1), 0.0)

    assert greedy_ctc_decode(log_probs) == [[1, 1, 2], [3, 3, 4]]


def test_optimizer_contract_and_parameter_mapping_are_exact():
    model = build_model(
        RecordingBackbone(),
        META,
        _config("linear_probe"),
        num_classes=9,
    )
    optimizer, names = build_optimizer(model, learning_rate=3e-4)

    assert optimizer_contract(3e-4) == {
        "type": "torch.optim.Adam",
        "learning_rate": 0.0003,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "scheduler": "none",
        "parameter_groups": 1,
        "parameter_order": "full_name_utf8_ascending",
        "requires_grad_only": True,
    }
    assert names == optimizer_parameter_names(model)
    assert names == sorted(names, key=lambda value: value.encode("utf-8"))
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert all(
        parameter.requires_grad
        for parameter in optimizer.param_groups[0]["params"]
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_optimizer_rejects_nonpositive_or_nonfinite_learning_rate(value):
    model = build_model(
        RecordingBackbone(),
        META,
        _config("linear_probe"),
        num_classes=9,
    )

    with pytest.raises(ProtocolError, match="learning_rate"):
        build_optimizer(model, value)
