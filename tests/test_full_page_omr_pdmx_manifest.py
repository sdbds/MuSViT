from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile

from PIL import Image
import pytest

from experiments.full_page_omr.pdmx_manifest import (
    PDMX_DATASET_ID,
    PDMX_DATASET_REVISION,
    build_pdmx_dataset_manifest,
    discover_official_pdmx_shards,
    load_pdmx_dataset_manifest,
    normalize_source_id,
    scan_pdmx_tar,
    verify_pdmx_dataset_manifest,
    write_pdmx_dataset_manifest,
)


VALID_KERN = "**kern\n*staff1\n*clefG2\n4c\n*-"


def _png_bytes(size: tuple[int, int] = (48, 32)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _write_tar(
    path: Path,
    samples: dict[str, dict[str, bytes]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for key, fields in samples.items():
            for suffix, payload in fields.items():
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))
    return path


def _sample(
    source: str,
    *,
    size: tuple[int, int] = (48, 32),
    fill: str | None = None,
) -> dict[str, bytes]:
    fields = {
        "image.png": _png_bytes(size),
        "kern.txt": VALID_KERN.encode("utf-8"),
        "source.txt": f"{source}\n".encode("utf-8"),
    }
    if fill is not None:
        fields["fill.txt"] = f"{fill}\n".encode("utf-8")
    return fields


def _make_snapshot(
    root: Path,
    *,
    train_source: str = "scores/train.mxl",
    validation_source: str = "scores/validation.mxl",
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    verovio = (
        "shards_pdmx_a1_broadened_518x728/"
        "voices_2_full/pdmx-v-000000.tar"
    )
    mscore = (
        "shards_pdmx_a1_mscore_518x728/"
        "voices_3_4_medium/pdmx-m-000000.tar"
    )
    validation = "curated_validation_pdmx.tar"
    _write_tar(root / verovio, {"verovio.a": _sample(train_source, fill="full")})
    _write_tar(root / mscore, {"mscore.a": _sample("scores/mscore.mxl")})
    _write_tar(
        root / validation,
        {"validation.a": _sample(validation_source, size=(31, 67))},
    )
    return (
        (("verovio", verovio), ("mscore", mscore)),
        (validation,),
    )


def test_scan_requires_image_kern_and_source(tmp_path):
    tar_path = _write_tar(
        tmp_path / "broken.tar",
        {
            "a": {
                "image.png": _png_bytes(),
                "kern.txt": VALID_KERN.encode("utf-8"),
            }
        },
    )

    with pytest.raises(ValueError, match=r"a.*source\.txt"):
        scan_pdmx_tar(
            tar_path,
            renderer="verovio",
            logical_path="broken.tar",
        )


def test_scan_accepts_optional_fill_and_records_token_lengths(tmp_path):
    tar_path = _write_tar(
        tmp_path / "valid.tar",
        {"a": _sample("scores/a.mxl", fill="medium")},
    )

    result = scan_pdmx_tar(
        tar_path,
        renderer="verovio",
        logical_path="train/valid.tar",
    )

    assert result.sample_count == 1
    assert result.records[0].source_id == "scores/a.mxl"
    assert result.records[0].fill == "medium"
    assert result.records[0].sequence_length == 4
    assert result.records[0].tokens == ("<bos>", "4c", "<b>", "<eos>")
    assert result.records[0].image_size == (48, 32)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        (" ./scores\\bach\\piece.mxl \n", "scores/bach/piece.mxl"),
        ("scores//mozart/./piece.mxl", "scores/mozart/piece.mxl"),
    ],
)
def test_source_ids_are_platform_independent(raw, normalized):
    assert normalize_source_id(raw) == normalized


def test_source_id_rejects_parent_traversal():
    with pytest.raises(ValueError, match="parent traversal"):
        normalize_source_id("scores/../other.mxl")


def test_discovery_accepts_only_official_kern_roots(tmp_path):
    expected_train, expected_validation = _make_snapshot(tmp_path)
    _write_tar(
        tmp_path / "curated_validation_pdmx_abc.tar",
        {"abc": _sample("abc.mxl")},
    )
    _write_tar(
        tmp_path / "unexpected" / "other.tar",
        {"other": _sample("other.mxl")},
    )

    train, validation = discover_official_pdmx_shards(tmp_path)

    assert train == tuple(sorted(expected_train, key=lambda item: item[1]))
    assert validation == expected_validation


def test_manifest_rejects_train_validation_source_overlap(tmp_path):
    train, validation = _make_snapshot(
        tmp_path,
        train_source="same.mxl",
        validation_source="./same.mxl",
    )

    with pytest.raises(ValueError, match="train/validation source overlap"):
        build_pdmx_dataset_manifest(
            snapshot_root=tmp_path,
            dataset_id=PDMX_DATASET_ID,
            dataset_revision=PDMX_DATASET_REVISION,
            train_shards=train,
            validation_shards=validation,
            renderer_weights={"verovio": 0.5, "mscore": 0.5},
        )


def test_manifest_rejects_duplicate_logical_shards(tmp_path):
    train, validation = _make_snapshot(tmp_path)
    duplicated = (*train, train[0])

    with pytest.raises(ValueError, match="duplicate train shard"):
        build_pdmx_dataset_manifest(
            snapshot_root=tmp_path,
            dataset_id=PDMX_DATASET_ID,
            dataset_revision=PDMX_DATASET_REVISION,
            train_shards=duplicated,
            validation_shards=validation,
            renderer_weights={"verovio": 0.5, "mscore": 0.5},
        )


def test_manifest_rejects_duplicate_keys_within_a_renderer(tmp_path):
    train, validation = _make_snapshot(tmp_path)
    extra_path = (
        "shards_pdmx_a1_broadened_518x728/"
        "voices_5_8_short/pdmx-v-000001.tar"
    )
    _write_tar(
        tmp_path / extra_path,
        {"verovio.a": _sample("scores/another.mxl")},
    )

    with pytest.raises(ValueError, match=r"duplicate sample key.*verovio\.a"):
        build_pdmx_dataset_manifest(
            snapshot_root=tmp_path,
            dataset_id=PDMX_DATASET_ID,
            dataset_revision=PDMX_DATASET_REVISION,
            train_shards=(*train, ("verovio", extra_path)),
            validation_shards=validation,
            renderer_weights={"verovio": 0.5, "mscore": 0.5},
        )


def test_manifest_digest_is_independent_of_input_shard_order(tmp_path):
    train, validation = _make_snapshot(tmp_path)

    first = build_pdmx_dataset_manifest(
        snapshot_root=tmp_path,
        dataset_id=PDMX_DATASET_ID,
        dataset_revision=PDMX_DATASET_REVISION,
        train_shards=train,
        validation_shards=validation,
        renderer_weights={"verovio": 0.5, "mscore": 0.5},
    )
    second = build_pdmx_dataset_manifest(
        snapshot_root=tmp_path,
        dataset_id=PDMX_DATASET_ID,
        dataset_revision=PDMX_DATASET_REVISION,
        train_shards=tuple(reversed(train)),
        validation_shards=validation,
        renderer_weights={"mscore": 0.5, "verovio": 0.5},
    )

    assert first == second
    assert first["manifest_sha256"]
    assert first["train_sample_count"] == 2
    assert first["validation_sample_count"] == 1
    assert first["train_token_frequencies"]["4c"] == 2


def test_written_manifest_round_trips_and_verifies_local_files(tmp_path):
    train, validation = _make_snapshot(tmp_path)
    manifest = build_pdmx_dataset_manifest(
        snapshot_root=tmp_path,
        dataset_id=PDMX_DATASET_ID,
        dataset_revision=PDMX_DATASET_REVISION,
        train_shards=train,
        validation_shards=validation,
        renderer_weights={"verovio": 0.5, "mscore": 0.5},
    )

    path = write_pdmx_dataset_manifest(manifest, tmp_path / "manifest.json")

    assert load_pdmx_dataset_manifest(path) == manifest
    verify_pdmx_dataset_manifest(manifest, tmp_path)

    (tmp_path / train[0][1]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify_pdmx_dataset_manifest(manifest, tmp_path)
