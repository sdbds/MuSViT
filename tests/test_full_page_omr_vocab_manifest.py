import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.full_page_omr.utils.vocab_manifest import (
    POLISH_PREFIX_SHA256,
    VocabularyManifest,
    build_project_seed_tokens,
    extend_ordered_tokens,
    load_legacy_ordered_tokens,
    load_vocabulary_manifest,
    ordered_token_sha256,
    write_legacy_numpy_pair,
    write_vocabulary_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_DIR = REPO_ROOT / "experiments" / "full_page_omr" / "vocab"


def _write_npy_pair(
    root: Path,
    w2i: dict[str, int],
    i2w: dict[int, str],
) -> tuple[Path, Path]:
    w2i_path = root / "Legacyw2i.npy"
    i2w_path = root / "Legacyi2w.npy"
    np.save(w2i_path, w2i)
    np.save(i2w_path, i2w)
    return w2i_path, i2w_path


def test_project_seed_preserves_polish_ids_and_appends_known_extras():
    seed = build_project_seed_tokens(VOCAB_DIR)

    assert len(seed) == 221
    assert ordered_token_sha256(seed[:215]) == POLISH_PREFIX_SHA256
    assert seed[215:] == (
        "*M6/16",
        "*staff1",
        "*staff2",
        "88",
        "=:|!;",
        "==;",
    )
    assert seed[0] == "<pad>"
    assert seed[29] == "<t>"
    assert seed[44] == "<s>"
    assert seed[100] == "<bos>"
    assert seed[132] == "<b>"
    assert seed[183] == "<eos>"


def test_legacy_loader_rejects_non_inverse_maps(tmp_path):
    w2i_path, i2w_path = _write_npy_pair(
        tmp_path,
        {"<pad>": 0, "x": 1},
        {0: "<pad>", 1: "y"},
    )

    with pytest.raises(ValueError, match="strict inverses"):
        load_legacy_ordered_tokens(w2i_path, i2w_path)


@pytest.mark.parametrize(
    ("w2i", "i2w", "message"),
    [
        ({"<pad>": 0, "x": 2}, {0: "<pad>", 2: "x"}, "contiguous"),
        ({"<pad>": 0, "x": True}, {0: "<pad>", 1: "x"}, "boolean"),
        ({"<pad>": 0, 3: 1}, {0: "<pad>", 1: "3"}, "string"),
    ],
)
def test_legacy_loader_rejects_invalid_id_contracts(
    tmp_path,
    w2i,
    i2w,
    message,
):
    w2i_path, i2w_path = _write_npy_pair(tmp_path, w2i, i2w)

    with pytest.raises((TypeError, ValueError), match=message):
        load_legacy_ordered_tokens(w2i_path, i2w_path)


def test_extension_is_order_independent_and_append_only():
    base = ("<pad>", "z")

    first = extend_ordered_tokens(base, [["é", "a"], ["ß", "a"]])
    second = extend_ordered_tokens(base, [["ß"], ["a", "é"]])

    assert first == second
    assert first[:2] == base
    assert first[2:] == tuple(
        sorted({"é", "a", "ß"}, key=lambda token: token.encode("utf-8"))
    )


def test_manifest_round_trip_verifies_digest_and_numpy_compatibility(tmp_path):
    tokens = ("<pad>", "<bos>", "a", "é")
    digest = ordered_token_sha256(tokens)
    manifest = VocabularyManifest(
        schema_version=1,
        name="TestVocabulary",
        tokenization_mode="bekern",
        base_name="TestBase",
        base_size=2,
        base_digest=ordered_token_sha256(tokens[:2]),
        ordered_tokens=tokens,
        token_provenance={"a": "train", "é": "train"},
        source_dataset_manifests=("dataset-sha",),
        vocab_sha256=digest,
    )

    json_path = write_vocabulary_manifest(manifest, tmp_path / "vocab.json")
    w2i_path, i2w_path = write_legacy_numpy_pair(manifest, tmp_path)

    assert load_vocabulary_manifest(json_path) == manifest
    assert load_legacy_ordered_tokens(w2i_path, i2w_path) == tokens
    assert json_path.read_bytes().endswith(b"\n")


def test_manifest_loader_rejects_tampered_ordered_tokens(tmp_path):
    tokens = ("<pad>", "a")
    manifest = VocabularyManifest(
        schema_version=1,
        name="TamperTest",
        tokenization_mode="bekern",
        base_name="base",
        base_size=1,
        base_digest=ordered_token_sha256(tokens[:1]),
        ordered_tokens=tokens,
        token_provenance={"a": "train"},
        source_dataset_manifests=(),
        vocab_sha256=ordered_token_sha256(tokens),
    )
    path = write_vocabulary_manifest(manifest, tmp_path / "vocab.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ordered_tokens"].append("injected")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="vocabulary SHA-256 mismatch"):
        load_vocabulary_manifest(path)


def test_ordered_token_digest_uses_canonical_compact_json():
    tokens = ("<pad>", "é")
    expected = hashlib.sha256(
        json.dumps(
            tokens,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert ordered_token_sha256(tokens) == expected
