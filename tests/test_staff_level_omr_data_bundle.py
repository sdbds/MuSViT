from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import pytest
from PIL import Image

from experiments.staff_level_omr.protocol import (
    ProtocolError,
    canonical_sha256,
    read_json,
    write_canonical_json,
)
from experiments.staff_level_omr.protocol.data_bundle import (
    load_dataset_bundle,
    prepare_dataset_bundle,
)


GROUP_REGEX = (
    r"(?P<group_id>score[0-9]+)/staff[0-9]+_region[.]png"
)


def _make_dataset(root: Path, group_count: int = 4) -> Path:
    data_path = root / "data"
    for group_index in range(group_count):
        group = data_path / f"score{group_index:02d}"
        group.mkdir(parents=True, exist_ok=True)
        for staff_index in range(2):
            stem = f"staff{staff_index:02d}"
            color = (
                20 + group_index * 30,
                40 + staff_index * 50,
                120,
            )
            Image.new("RGB", (24, 16), color).save(
                group / f"{stem}_region.png"
            )
            tokens = (
                "note note barline"
                if staff_index == 0
                else "clef accidental-\u266f note"
            )
            (group / f"{stem}_gt.txt").write_text(tokens, encoding="utf-8")
    return data_path


def _prepare(tmp_path: Path):
    data_path = _make_dataset(tmp_path)
    bundle_path = tmp_path / "bundle"
    report = prepare_dataset_bundle(
        data_path=data_path,
        dataset_id="fixture",
        group_regex=GROUP_REGEX,
        split_ratios=("0.8", "0.1", "0.1"),
        seed=7,
        out=bundle_path,
    )
    return data_path, bundle_path, report


def _rebind_manifest(bundle_path: Path, manifest: dict) -> None:
    manifest_path = bundle_path / "split_manifest.json"
    vocabulary_path = bundle_path / "vocabulary.json"
    index_path = bundle_path / "image_verification_index.json"
    bundle_file = bundle_path / "bundle.json"

    write_canonical_json(manifest_path, manifest)
    manifest_hash = canonical_sha256(manifest)

    vocabulary = read_json(vocabulary_path)
    vocabulary["source_manifest_sha256"] = manifest_hash
    write_canonical_json(vocabulary_path, vocabulary)

    index = read_json(index_path)
    index["source_manifest_sha256"] = manifest_hash
    write_canonical_json(index_path, index)

    bundle = read_json(bundle_file)
    bundle["manifest_sha256"] = manifest_hash
    bundle["vocabulary_sha256"] = canonical_sha256(vocabulary)
    write_canonical_json(bundle_file, bundle)


def _rebind_vocabulary(bundle_path: Path, vocabulary: dict) -> None:
    write_canonical_json(bundle_path / "vocabulary.json", vocabulary)
    bundle = read_json(bundle_path / "bundle.json")
    bundle["vocabulary_sha256"] = canonical_sha256(vocabulary)
    write_canonical_json(bundle_path / "bundle.json", bundle)


def test_prepare_data_is_deterministic_for_identity_documents(tmp_path):
    data_path = _make_dataset(tmp_path)
    first = tmp_path / "bundle-a"
    second = tmp_path / "bundle-b"

    first_report = prepare_dataset_bundle(
        data_path=data_path,
        dataset_id="fixture",
        group_regex=GROUP_REGEX,
        split_ratios=("0.80", "0.10", "0.10"),
        seed=7,
        out=first,
    )
    second_report = prepare_dataset_bundle(
        data_path=data_path,
        dataset_id="fixture",
        group_regex=GROUP_REGEX,
        split_ratios=("8", "1", "1"),
        seed=7,
        out=second,
    )

    for filename in ("bundle.json", "split_manifest.json", "vocabulary.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    manifest = read_json(first / "split_manifest.json")
    vocabulary = read_json(first / "vocabulary.json")
    bundle = read_json(first / "bundle.json")
    samples = manifest["samples"]

    assert bundle["generation_contract"]["split_weights"] == [8, 1, 1]
    assert {sample["split"] for sample in samples} == {"train", "val", "test"}
    assert [sample["sample_id"] for sample in samples] == sorted(
        (sample["sample_id"] for sample in samples),
        key=lambda value: value.encode("utf-8"),
    )
    group_splits = {}
    for sample in samples:
        group_splits.setdefault(sample["group_id"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert vocabulary["tokens"] == sorted(
        vocabulary["tokens"], key=lambda value: value.encode("utf-8")
    )
    assert not any("patch" in key for key in manifest)
    assert first_report.bundle_sha256 == second_report.bundle_sha256


def test_load_bundle_exposes_stable_vocabulary_mapping(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)

    loaded = load_dataset_bundle(bundle_path, data_path)
    vocabulary = loaded.vocabulary

    assert vocabulary.blank_id == 0
    assert vocabulary.num_classes == len(vocabulary.tokens) + 1
    assert vocabulary.decode(vocabulary.encode(["note", "barline"])) == (
        "note",
        "barline",
    )
    with pytest.raises(ProtocolError, match="OOV"):
        vocabulary.encode(["not-in-vocabulary"])
    with pytest.raises(ProtocolError, match="blank"):
        vocabulary.decode([0])


def test_prepare_data_rejects_invalid_regex_without_publishing(tmp_path):
    data_path = _make_dataset(tmp_path)
    missing_group_out = tmp_path / "missing-group"
    unmatched_out = tmp_path / "unmatched"

    with pytest.raises(ProtocolError, match="group_id"):
        prepare_dataset_bundle(
            data_path=data_path,
            dataset_id="fixture",
            group_regex=r".*_region[.]png",
            split_ratios=("0.8", "0.1", "0.1"),
            seed=7,
            out=missing_group_out,
        )
    assert not missing_group_out.exists()

    with pytest.raises(ProtocolError, match=r"unmatched.*score01"):
        prepare_dataset_bundle(
            data_path=data_path,
            dataset_id="fixture",
            group_regex=(
                r"(?P<group_id>score00)/staff[0-9]+_region[.]png"
            ),
            split_ratios=("0.8", "0.1", "0.1"),
            seed=7,
            out=unmatched_out,
        )
    assert not unmatched_out.exists()


def test_prepare_data_requires_three_groups_and_complete_pairs(tmp_path):
    two_group_data = _make_dataset(tmp_path / "few", group_count=2)
    with pytest.raises(ProtocolError, match="at least three groups"):
        prepare_dataset_bundle(
            data_path=two_group_data,
            dataset_id="fixture",
            group_regex=GROUP_REGEX,
            split_ratios=("0.8", "0.1", "0.1"),
            seed=7,
            out=tmp_path / "few-bundle",
        )

    data_path = _make_dataset(tmp_path / "orphan")
    (data_path / "score99").mkdir()
    (data_path / "score99" / "staff00_region.png").touch()
    with pytest.raises(ProtocolError, match="orphan image"):
        prepare_dataset_bundle(
            data_path=data_path,
            dataset_id="fixture",
            group_regex=GROUP_REGEX,
            split_ratios=("0.8", "0.1", "0.1"),
            seed=7,
            out=tmp_path / "orphan-bundle",
        )


def test_prepare_data_never_overwrites_existing_output(tmp_path):
    data_path = _make_dataset(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()

    with pytest.raises(ProtocolError, match="already exists"):
        prepare_dataset_bundle(
            data_path=data_path,
            dataset_id="fixture",
            group_regex=GROUP_REGEX,
            split_ratios=("0.8", "0.1", "0.1"),
            seed=7,
            out=out,
        )


def test_prepare_data_rejects_scalar_split_ratios_without_type_leak(tmp_path):
    data_path = _make_dataset(tmp_path)

    with pytest.raises(ProtocolError, match="split_ratios"):
        prepare_dataset_bundle(
            data_path=data_path,
            dataset_id="fixture",
            group_regex=GROUP_REGEX,
            split_ratios=0.8,
            seed=7,
            out=tmp_path / "bundle",
        )


def test_bundle_rejects_duplicate_sample_and_cross_split_group(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    manifest = read_json(bundle_path / "split_manifest.json")
    manifest["samples"].append(copy.deepcopy(manifest["samples"][0]))
    _rebind_manifest(bundle_path, manifest)
    with pytest.raises(ProtocolError, match="duplicate sample_id"):
        load_dataset_bundle(bundle_path, data_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    data_path, bundle_path, _ = _prepare(other_root)
    manifest = read_json(bundle_path / "split_manifest.json")
    first_group = manifest["samples"][0]["group_id"]
    same_group = [
        sample for sample in manifest["samples"] if sample["group_id"] == first_group
    ]
    same_group[0]["split"] = (
        "val" if same_group[0]["split"] != "val" else "test"
    )
    _rebind_manifest(bundle_path, manifest)
    with pytest.raises(ProtocolError, match="multiple splits"):
        load_dataset_bundle(bundle_path, data_path)


def test_bundle_rejects_path_escape_and_empty_split(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    manifest = read_json(bundle_path / "split_manifest.json")
    manifest["samples"][0]["image_path"] = "../escape_region.png"
    _rebind_manifest(bundle_path, manifest)
    with pytest.raises(ProtocolError, match="escape"):
        load_dataset_bundle(bundle_path, data_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    data_path, bundle_path, _ = _prepare(other_root)
    manifest = read_json(bundle_path / "split_manifest.json")
    for sample in manifest["samples"]:
        if sample["split"] == "test":
            sample["split"] = "val"
    _rebind_manifest(bundle_path, manifest)
    with pytest.raises(ProtocolError, match="test.*empty"):
        load_dataset_bundle(bundle_path, data_path)


def test_bundle_rejects_hash_and_dataset_identity_mismatches(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    sample = read_json(bundle_path / "split_manifest.json")["samples"][0]
    (data_path / Path(sample["target_path"])).write_text(
        "changed tokens", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="target SHA-256"):
        load_dataset_bundle(bundle_path, data_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    data_path, bundle_path, _ = _prepare(other_root)
    bundle = read_json(bundle_path / "bundle.json")
    bundle["dataset_id"] = "wrong"
    write_canonical_json(bundle_path / "bundle.json", bundle)
    with pytest.raises(ProtocolError, match="dataset_id"):
        load_dataset_bundle(bundle_path, data_path)


def test_bundle_rejects_verification_index_omissions_and_extras(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    index = read_json(bundle_path / "image_verification_index.json")
    index["entries"].pop()
    write_canonical_json(bundle_path / "image_verification_index.json", index)
    with pytest.raises(ProtocolError, match="verification index"):
        load_dataset_bundle(bundle_path, data_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    data_path, bundle_path, _ = _prepare(other_root)
    index = read_json(bundle_path / "image_verification_index.json")
    extra = copy.deepcopy(index["entries"][0])
    extra["image_path"] = "extra_region.png"
    index["entries"].append(extra)
    write_canonical_json(bundle_path / "image_verification_index.json", index)
    with pytest.raises(ProtocolError, match="verification index"):
        load_dataset_bundle(bundle_path, data_path)


def test_bundle_rejects_invalid_vocabulary_and_oov(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    vocabulary = read_json(bundle_path / "vocabulary.json")
    vocabulary["tokens"].append(vocabulary["tokens"][0])
    _rebind_vocabulary(bundle_path, vocabulary)
    with pytest.raises(ProtocolError, match="duplicate token"):
        load_dataset_bundle(bundle_path, data_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    data_path, bundle_path, _ = _prepare(other_root)
    vocabulary = read_json(bundle_path / "vocabulary.json")
    vocabulary["tokens"].remove("note")
    _rebind_vocabulary(bundle_path, vocabulary)
    with pytest.raises(ProtocolError, match="OOV.*note"):
        load_dataset_bundle(bundle_path, data_path)


def test_bundle_rejects_empty_target_even_when_hash_is_rebound(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    manifest = read_json(bundle_path / "split_manifest.json")
    sample = manifest["samples"][0]
    target_path = data_path / Path(sample["target_path"])
    target_path.write_bytes(b"")
    sample["target_sha256"] = hashlib.sha256(b"").hexdigest()
    _rebind_manifest(bundle_path, manifest)

    with pytest.raises(ProtocolError, match="empty target"):
        load_dataset_bundle(bundle_path, data_path)


def test_image_verification_always_and_cached_modes_are_explicit(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    sample_count = len(read_json(bundle_path / "split_manifest.json")["samples"])

    always = load_dataset_bundle(
        bundle_path, data_path, verify_image_hashes="always"
    )
    cached = load_dataset_bundle(
        bundle_path, data_path, verify_image_hashes="cached"
    )

    assert always.image_verification.hits == 0
    assert always.image_verification.recomputed == sample_count
    assert always.image_verification.status == "content_verified"
    assert always.image_verification.trusted_baseline
    assert cached.image_verification.hits == sample_count
    assert cached.image_verification.recomputed == 0
    assert cached.image_verification.status == "cached_metadata"
    assert not cached.image_verification.trusted_baseline


def test_cached_mode_does_not_claim_same_size_mtime_content_identity(tmp_path):
    data_path, bundle_path, _ = _prepare(tmp_path)
    sample = read_json(bundle_path / "split_manifest.json")["samples"][0]
    image_path = data_path / Path(sample["image_path"])
    original_stat = image_path.stat()
    payload = bytearray(image_path.read_bytes())
    payload[-1] ^= 1
    image_path.write_bytes(payload)
    os.utime(
        image_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    cached = load_dataset_bundle(
        bundle_path, data_path, verify_image_hashes="cached"
    )
    assert cached.image_verification.status == "cached_metadata"

    with pytest.raises(ProtocolError, match="image SHA-256"):
        load_dataset_bundle(
            bundle_path, data_path, verify_image_hashes="always"
        )
