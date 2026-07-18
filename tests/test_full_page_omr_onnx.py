import json
from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import ViTModel

from experiments.full_page_omr import export_onnx
from experiments.full_page_omr.export_onnx import (
    ExportPaths,
    FullPageOMRDecoderWrapper,
    FullPageOMREncoderWrapper,
    build_bundle_metadata,
    build_preprocessor_config,
    export_bundle,
    export_decoder_graph,
    export_encoder_graph,
    load_standalone_model,
    parse_args,
    validate_graph_files,
    write_json_atomic,
)
from experiments.full_page_omr.onnx_runtime import (
    FullPageOMROnnxRuntime,
    preprocess_page,
    resolve_providers,
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


def test_export_encoder_graph_uses_locked_contract(tmp_path, monkeypatch):
    calls = []

    def fake_export(model, args, output_path, **kwargs):
        calls.append((model, args, Path(output_path), kwargs))
        Path(output_path).write_bytes(b"encoder")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(export_onnx, "_validate_onnx_graph", lambda _path: None)
    target = tmp_path / "encoder.onnx"

    export_encoder_graph(torch.nn.Identity(), target, torch.device("cpu"))

    assert target.read_bytes() == b"encoder"
    assert not (tmp_path / "encoder.onnx.tmp").exists()
    _, args, temporary_path, kwargs = calls[0]
    assert args[0].shape == (1, 3, 1024, 1024)
    assert temporary_path.name == "encoder.onnx.tmp"
    assert kwargs["opset_version"] == 20
    assert kwargs["dynamo"] is False
    assert kwargs["external_data"] is False
    assert kwargs["input_names"] == ["pixel_values"]
    assert kwargs["output_names"] == ["raw_features", "enhanced_features"]
    assert kwargs["dynamic_axes"] is None


def test_export_decoder_graph_has_only_dynamic_prefix_axis(tmp_path, monkeypatch):
    calls = []

    def fake_export(model, args, output_path, **kwargs):
        calls.append((model, args, Path(output_path), kwargs))
        Path(output_path).write_bytes(b"decoder")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(export_onnx, "_validate_onnx_graph", lambda _path: None)
    target = tmp_path / "decoder.onnx"

    export_decoder_graph(torch.nn.Identity(), target, torch.device("cpu"))

    _, args, _, kwargs = calls[0]
    assert args[0].shape == (1, 4096, 256)
    assert args[1].shape == (1, 4096, 256)
    assert args[2].shape == (1, 1)
    assert kwargs["input_names"] == [
        "raw_features",
        "enhanced_features",
        "token_ids",
    ]
    assert kwargs["output_names"] == ["next_token_logits"]
    assert kwargs["dynamic_axes"] == {
        "token_ids": {1: "sequence_length"},
    }


def test_validate_graph_files_rejects_empty_oversized_and_sidecar_files(tmp_path):
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    encoder.write_bytes(b"")
    decoder.write_bytes(b"ok")
    paths = ExportPaths.from_output_dir(tmp_path)

    with pytest.raises(RuntimeError, match="empty"):
        validate_graph_files(paths, max_onnx_bytes=10)

    encoder.write_bytes(b"123456789")
    with pytest.raises(RuntimeError, match="size limit"):
        validate_graph_files(paths, max_onnx_bytes=10)

    encoder.write_bytes(b"abc")
    (tmp_path / "encoder.onnx.data").write_bytes(b"sidecar")
    with pytest.raises(RuntimeError, match="external tensor data"):
        validate_graph_files(paths, max_onnx_bytes=10)


def test_atomic_json_and_bundle_metadata_record_runtime_contract(tmp_path):
    paths = ExportPaths.from_output_dir(tmp_path)
    preprocessor = build_preprocessor_config()
    metadata = build_bundle_metadata(
        paths=paths,
        source={"weights_sha256": "abc"},
        versions={"torch": "2.13.0", "onnxruntime": "1.27.0"},
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        validation={"status": "pending"},
    )

    write_json_atomic(paths.preprocessor_config, preprocessor)
    write_json_atomic(paths.metadata, metadata)

    assert not (tmp_path / "metadata.json.tmp").exists()
    assert json.loads(paths.preprocessor_config.read_text(encoding="utf-8")) == {
        "color": "RGB",
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_size": [1024, 1024],
        "input_layout": "NCHW",
        "interpolation": "bilinear",
        "rescale_factor": 1 / 255,
    }
    loaded = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert loaded["graphs"]["encoder"]["inputs"]["pixel_values"] == [
        1,
        3,
        1024,
        1024,
    ]
    assert loaded["graphs"]["decoder"]["inputs"]["token_ids"] == [
        1,
        "sequence_length",
    ]
    assert loaded["tokens"] == {"bos": 100, "eos": 183, "max_length": 7512}


def test_parse_args_accepts_explicit_standalone_assets(tmp_path):
    weights = tmp_path / "weights.safetensors"
    model_config = tmp_path / "model.json"
    encoder_config = tmp_path / "encoder.json"
    output_dir = tmp_path / "onnx"

    args = parse_args(
        [
            "--weights-path",
            str(weights),
            "--model-config-path",
            str(model_config),
            "--encoder-config-path",
            str(encoder_config),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
    )

    assert args.weights_path == weights
    assert args.model_config_path == model_config
    assert args.encoder_config_path == encoder_config
    assert args.output_dir == output_dir
    assert args.device == "cpu"
    assert args.opset_version == 20


class _FakeBundleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Identity()


def test_export_bundle_writes_self_contained_configs_and_hashed_metadata(
    tmp_path,
    monkeypatch,
):
    weights = tmp_path / "weights.safetensors"
    model_config = tmp_path / "model.json"
    encoder_config = tmp_path / "encoder.json"
    output_dir = tmp_path / "onnx"
    weights.write_bytes(b"weights")
    model_config.write_text("{}", encoding="utf-8")
    encoder_config.write_text("{}", encoding="utf-8")
    merged_config = {
        "foundation_config": {"hidden_size": 768},
        "w2i": {"<bos>": 100, "<eos>": 183},
    }
    monkeypatch.setattr(
        export_onnx,
        "load_standalone_model",
        lambda *_args: (_FakeBundleModel(), merged_config),
    )

    def fake_encoder(_wrapper, path, _device, *, opset_version):
        assert opset_version == 20
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"encoder graph")
        return Path(path)

    def fake_decoder(_wrapper, path, _device, *, opset_version):
        assert opset_version == 20
        Path(path).write_bytes(b"decoder graph")
        return Path(path)

    monkeypatch.setattr(export_onnx, "export_encoder_graph", fake_encoder)
    monkeypatch.setattr(export_onnx, "export_decoder_graph", fake_decoder)

    paths = export_bundle(
        weights_path=weights,
        model_config_path=model_config,
        encoder_config_path=encoder_config,
        output_dir=output_dir,
        device=torch.device("cpu"),
    )

    assert json.loads(paths.config.read_text(encoding="utf-8")) == merged_config
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["artifacts"]["encoder"]["bytes"] == len(b"encoder graph")
    assert metadata["artifacts"]["decoder"]["bytes"] == len(b"decoder graph")
    assert metadata["source"]["weights_sha256"] == export_onnx.sha256_file(weights)
    assert metadata["validation"] == {"status": "not_run"}


def test_preprocess_page_matches_rgb_nchw_rescale_contract():
    image = Image.new("L", (1, 1), color=128)
    config = {
        "color": "RGB",
        "image_size": [2, 2],
        "interpolation": "bilinear",
        "rescale_factor": 1 / 255,
    }

    pixels = preprocess_page(image, config)

    assert pixels.shape == (1, 3, 2, 2)
    assert pixels.dtype == np.float32
    np.testing.assert_allclose(pixels, np.float32(128 / 255), rtol=0, atol=0)
    assert pixels.flags.c_contiguous


def test_resolve_providers_prefers_cuda_with_cpu_fallback():
    available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    assert resolve_providers(None, available=available) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert resolve_providers(["CPUExecutionProvider"], available=available) == [
        "CPUExecutionProvider"
    ]


class _FakeIO:
    def __init__(self, name):
        self.name = name


class _FakeEncoderSession:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [_FakeIO("pixel_values")]

    def get_outputs(self):
        return [_FakeIO("raw_features"), _FakeIO("enhanced_features")]

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feeds):
        self.calls.append((output_names, feeds))
        raw = np.zeros((1, 4, 16), dtype=np.float32)
        return [raw, raw + 1]


class _FakeDecoderSession:
    def __init__(self, generated_ids):
        self.generated_ids = iter(generated_ids)
        self.prefixes = []

    def get_inputs(self):
        return [
            _FakeIO("raw_features"),
            _FakeIO("enhanced_features"),
            _FakeIO("token_ids"),
        ]

    def get_outputs(self):
        return [_FakeIO("next_token_logits")]

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feeds):
        del output_names
        self.prefixes.append(feeds["token_ids"].copy())
        logits = np.full((1, 5), -10.0, dtype=np.float32)
        logits[0, next(self.generated_ids)] = 10.0
        return [logits]


def _write_runtime_bundle(tmp_path, *, maxlen=5):
    config = {
        "i2w": {"0": "<pad>", "1": "<bos>", "2": "<eos>", "3": "note", "4": "rest"},
        "w2i": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "note": 3, "rest": 4},
        "maxlen": maxlen,
    }
    preprocessor = build_preprocessor_config()
    (tmp_path / "encoder.onnx").write_bytes(b"encoder")
    (tmp_path / "decoder.onnx").write_bytes(b"decoder")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(preprocessor),
        encoding="utf-8",
    )


def test_runtime_encodes_once_and_grows_complete_prefix_until_eos(tmp_path):
    _write_runtime_bundle(tmp_path)
    encoder = _FakeEncoderSession()
    decoder = _FakeDecoderSession([3, 4, 2])

    runtime = FullPageOMROnnxRuntime(
        tmp_path,
        encoder_session=encoder,
        decoder_session=decoder,
    )
    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    )

    assert len(encoder.calls) == 1
    assert [prefix.tolist() for prefix in decoder.prefixes] == [
        [[1]],
        [[1, 3]],
        [[1, 3, 4]],
    ]
    assert result.token_ids == (1, 3, 4, 2)
    assert result.tokens == ("note", "rest")
    assert result.terminated_by_eos is True
    assert result.truncated is False


def test_runtime_reports_truncation_at_max_length(tmp_path):
    _write_runtime_bundle(tmp_path, maxlen=3)
    runtime = FullPageOMROnnxRuntime(
        tmp_path,
        encoder_session=_FakeEncoderSession(),
        decoder_session=_FakeDecoderSession([3, 4]),
    )

    result = runtime.generate_pixel_values(
        np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    )

    assert result.token_ids == (1, 3, 4)
    assert result.terminated_by_eos is False
    assert result.truncated is True


def test_runtime_rejects_malformed_session_contract(tmp_path):
    _write_runtime_bundle(tmp_path)
    encoder = _FakeEncoderSession()
    encoder.get_outputs = lambda: [_FakeIO("wrong")]

    with pytest.raises(ValueError, match="encoder output contract"):
        FullPageOMROnnxRuntime(
            tmp_path,
            encoder_session=encoder,
            decoder_session=_FakeDecoderSession([2]),
        )
