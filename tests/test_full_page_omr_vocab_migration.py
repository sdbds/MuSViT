from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.full_page_omr.migrate_vocabulary_checkpoint import (
    TOKEN_AXIS_KEYS,
    load_source_vocabulary_tokens,
    load_vocabulary_aware_weights,
    remap_vocabulary_state_dict,
    write_legacy_source_vocabulary_manifest,
)
from experiments.full_page_omr.utils.vocab_manifest import (
    VocabularyManifest,
    ordered_token_sha256,
    write_vocabulary_manifest,
)


EMBEDDING_KEY, OUTPUT_WEIGHT_KEY, OUTPUT_BIAS_KEY = TOKEN_AXIS_KEYS


def _state(vocab_size: int, *, offset: float) -> dict[str, torch.Tensor]:
    return {
        EMBEDDING_KEY: (
            torch.arange(vocab_size * 2, dtype=torch.float32)
            .reshape(vocab_size, 2)
            .add(offset)
        ),
        OUTPUT_WEIGHT_KEY: (
            torch.arange(vocab_size * 2, dtype=torch.float32)
            .reshape(vocab_size, 2, 1)
            .add(offset + 100)
        ),
        OUTPUT_BIAS_KEY: (
            torch.arange(vocab_size, dtype=torch.float32)
            .add(offset + 200)
        ),
        "model.shared.weight": torch.tensor([[offset + 300.0]]),
    }


def _manifest(tokens: tuple[str, ...], name: str) -> VocabularyManifest:
    digest = ordered_token_sha256(tokens)
    return VocabularyManifest(
        schema_version=1,
        name=name,
        tokenization_mode="bekern",
        base_name=name,
        base_size=len(tokens),
        base_digest=digest,
        ordered_tokens=tokens,
        token_provenance={},
        source_dataset_manifests=(),
        vocab_sha256=digest,
    )


def test_token_axes_are_copied_by_name_not_source_id():
    source_tokens = ("<pad>", "b", "a")
    target_tokens = ("<pad>", "a", "b", "new")
    source = _state(3, offset=0)
    target = _state(4, offset=1_000)

    migrated, report = remap_vocabulary_state_dict(
        source,
        target,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )

    assert torch.equal(migrated[EMBEDDING_KEY][1], source[EMBEDDING_KEY][2])
    assert torch.equal(
        migrated[OUTPUT_WEIGHT_KEY][2],
        source[OUTPUT_WEIGHT_KEY][1],
    )
    assert torch.equal(
        migrated[OUTPUT_BIAS_KEY][1],
        source[OUTPUT_BIAS_KEY][2],
    )
    assert torch.equal(migrated[EMBEDDING_KEY][3], target[EMBEDDING_KEY][3])
    assert torch.equal(
        migrated["model.shared.weight"],
        source["model.shared.weight"],
    )
    assert report["new_tokens"] == ["new"]
    assert report["dropped_tokens"] == []
    assert report["common_token_count"] == 3


def test_source_only_token_is_fatal():
    with pytest.raises(ValueError, match="missing from target"):
        remap_vocabulary_state_dict(
            _state(2, offset=0),
            _state(1, offset=100),
            source_tokens=("<pad>", "lost"),
            target_tokens=("<pad>",),
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda state: state.pop("model.shared.weight"),
            "key sets differ",
        ),
        (
            lambda state: state.__setitem__(
                "model.shared.weight",
                state["model.shared.weight"].to(torch.float64),
            ),
            "dtype",
        ),
        (
            lambda state: state.__setitem__(
                OUTPUT_WEIGHT_KEY,
                torch.zeros(3, 3, 1),
            ),
            "non-token dimensions",
        ),
    ],
)
def test_migration_rejects_incompatible_state_contract(mutation, match):
    source = _state(3, offset=0)
    target = _state(3, offset=100)
    mutation(source)

    with pytest.raises(ValueError, match=match):
        remap_vocabulary_state_dict(
            source,
            target,
            source_tokens=("<pad>", "a", "b"),
            target_tokens=("<pad>", "a", "b"),
        )


class _ToyDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 2)
        self.out_layer = torch.nn.Conv1d(2, vocab_size, kernel_size=1)


class _ToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.decoder = _ToyDecoder(vocab_size)
        self.shared = torch.nn.Linear(1, 1, bias=False)


class _ToyWrapper(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.model = _ToyModel(vocab_size)


def test_checkpoint_loader_uses_only_state_dict_and_writes_report(tmp_path):
    source_tokens = ("<pad>", "b", "a")
    target_tokens = ("<pad>", "a", "b", "new")
    source_wrapper = _ToyWrapper(len(source_tokens))
    target_wrapper = _ToyWrapper(len(target_tokens))
    target_wrapper.model.i2w = dict(enumerate(target_tokens))
    with torch.no_grad():
        for index, tensor in enumerate(source_wrapper.state_dict().values()):
            tensor.copy_(
                torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(
                    tensor.shape
                )
                + index * 100
            )
    source_state = {
        key: value.clone()
        for key, value in source_wrapper.state_dict().items()
    }
    target_new_row = (
        target_wrapper.state_dict()[EMBEDDING_KEY][3].clone()
    )
    checkpoint_path = tmp_path / "source.ckpt"
    torch.save(
        {
            "state_dict": source_state,
            "global_step": 999,
            "optimizer_states": [{"must": "not load"}],
        },
        checkpoint_path,
    )
    source_manifest_path = write_vocabulary_manifest(
        _manifest(source_tokens, "source"),
        tmp_path / "source.json",
    )
    target_manifest_path = write_vocabulary_manifest(
        _manifest(target_tokens, "target"),
        tmp_path / "target.json",
    )
    report_path = tmp_path / "migration.json"

    report = load_vocabulary_aware_weights(
        target_wrapper,
        checkpoint_path,
        source_vocab_manifest=source_manifest_path,
        target_vocab_manifest=target_manifest_path,
        report_path=report_path,
    )

    migrated = target_wrapper.state_dict()
    assert torch.equal(
        migrated[EMBEDDING_KEY][1],
        source_state[EMBEDDING_KEY][2],
    )
    assert torch.equal(migrated[EMBEDDING_KEY][3], target_new_row)
    assert report["new_tokens"] == ["new"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_loader_validates_target_token_order_before_checkpoint_load(
    tmp_path,
    monkeypatch,
):
    tokens = ("<pad>", "a")
    source_manifest_path = write_vocabulary_manifest(
        _manifest(tokens, "source"),
        tmp_path / "source.json",
    )
    target_manifest_path = write_vocabulary_manifest(
        _manifest(tokens, "target"),
        tmp_path / "target.json",
    )
    wrapper = _ToyWrapper(len(tokens))
    wrapper.model.i2w = {0: "<pad>", 1: "different"}
    load_mock = lambda *args, **kwargs: pytest.fail(
        "checkpoint tensors loaded before target vocabulary validation"
    )
    monkeypatch.setattr(torch, "load", load_mock)

    with pytest.raises(ValueError, match="target model vocabulary"):
        load_vocabulary_aware_weights(
            wrapper,
            tmp_path / "unread.ckpt",
            source_vocab_manifest=source_manifest_path,
            target_vocab_manifest=target_manifest_path,
            report_path=tmp_path / "report.json",
        )


def test_legacy_source_manifest_pins_both_numpy_files(tmp_path):
    tokens = ("<pad>", "b", "a")
    w2i_path = tmp_path / "Legacyw2i.npy"
    i2w_path = tmp_path / "Legacyi2w.npy"
    np.save(w2i_path, {token: index for index, token in enumerate(tokens)})
    np.save(i2w_path, {index: token for index, token in enumerate(tokens)})
    manifest_path = write_legacy_source_vocabulary_manifest(
        w2i_path,
        i2w_path,
        output_path=tmp_path / "Legacy.source-vocab.json",
        name="Legacy",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["manifest_type"] == "legacy_vocabulary_source"
    assert len(payload["source_files"]["w2i"]["sha256"]) == 64
    assert len(payload["source_files"]["i2w"]["sha256"]) == 64
    assert load_source_vocabulary_tokens(manifest_path) == tokens

    np.save(i2w_path, {0: "<pad>", 1: "changed", 2: "a"})
    with pytest.raises(ValueError, match="SHA-256"):
        load_source_vocabulary_tokens(manifest_path)
