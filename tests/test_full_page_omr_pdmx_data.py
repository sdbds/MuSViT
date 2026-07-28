from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest
import torch

from experiments.full_page_omr import _globals
from experiments.full_page_omr.config.ExperimentConfigWrapper import PDMXData
from experiments.full_page_omr.pdmx_data import (
    PDMXConsumptionAuditCallback,
    PDMXPretrainingDataModule,
    _PDMXSampleDecoder,
    _webdataset_samples,
    deterministic_teacher_forcing,
)
from experiments.full_page_omr.pdmx_manifest import (
    PDMX_DATASET_ID,
    PDMX_DATASET_REVISION,
    build_pdmx_dataset_manifest,
    load_pdmx_dataset_manifest,
    resolve_local_snapshot,
    write_pdmx_dataset_manifest,
)
from experiments.full_page_omr.utils.vocab_manifest import (
    VocabularyManifest,
    build_project_seed_tokens,
    extend_ordered_tokens,
    ordered_token_sha256,
    load_vocabulary_manifest,
    write_vocabulary_manifest,
)


VALID_KERN = "**kern\n*staff1\n*clefG2\n4c\n*-"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VOCAB_DIR = REPO_ROOT / "experiments" / "full_page_omr" / "vocab"


def _png_bytes(size: tuple[int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _write_tar(
    path: Path,
    samples: dict[str, tuple[str, tuple[int, int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for key, (source, size) in samples.items():
            fields = {
                "image.png": _png_bytes(size),
                "kern.txt": VALID_KERN.encode("utf-8"),
                "source.txt": source.encode("utf-8"),
                "fill.txt": b"medium",
            }
            for suffix, payload in fields.items():
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))


@pytest.fixture
def pdmx_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(_globals, "resolution", 64)
    package_root = tmp_path / "full_page_omr"
    snapshot_root = tmp_path / "snapshot"
    train_shards = []
    for renderer, root_name in (
        ("verovio", "shards_pdmx_a1_broadened_518x728"),
        ("mscore", "shards_pdmx_a1_mscore_518x728"),
    ):
        for index in range(2):
            logical_path = (
                f"{root_name}/voices_2_medium/"
                f"pdmx-{renderer}-{index:06d}.tar"
            )
            _write_tar(
                snapshot_root / logical_path,
                {
                    f"{renderer}-{index}": (
                        f"scores/{renderer}-{index}.mxl",
                        (48 + index, 32 + index),
                    )
                },
            )
            train_shards.append((renderer, logical_path))

    validation_path = "curated_validation_pdmx.tar"
    _write_tar(
        snapshot_root / validation_path,
        {
            "val-a": ("scores/val-a.mxl", (31, 67)),
            "val-b": ("scores/val-b.mxl", (79, 29)),
        },
    )
    dataset_manifest = build_pdmx_dataset_manifest(
        snapshot_root=snapshot_root,
        dataset_id=PDMX_DATASET_ID,
        dataset_revision=PDMX_DATASET_REVISION,
        train_shards=train_shards,
        validation_shards=(validation_path,),
        renderer_weights={"verovio": 0.5, "mscore": 0.5},
    )
    dataset_manifest_path = (
        package_root / "config" / "Page_OMR_PDMX" / "dataset.json"
    )
    write_pdmx_dataset_manifest(dataset_manifest, dataset_manifest_path)

    seed = build_project_seed_tokens(PROJECT_VOCAB_DIR)
    ordered_tokens = extend_ordered_tokens(
        seed,
        [dataset_manifest["train_token_frequencies"]],
    )
    vocabulary = VocabularyManifest(
        schema_version=1,
        name="FixtureFullPageOMR",
        tokenization_mode="bekern",
        base_name="FullPageOMRProjectSeed",
        base_size=len(seed),
        base_digest=ordered_token_sha256(seed),
        ordered_tokens=ordered_tokens,
        token_provenance={
            token: "fixture-train"
            for token in ordered_tokens[len(seed) :]
        },
        source_dataset_manifests=(dataset_manifest["manifest_sha256"],),
        vocab_sha256=ordered_token_sha256(ordered_tokens),
    )
    vocab_path = package_root / "vocab" / "fixture.json"
    write_vocabulary_manifest(vocabulary, vocab_path)

    config = PDMXData(
        dataset_id=PDMX_DATASET_ID,
        dataset_revision=PDMX_DATASET_REVISION,
        dataset_manifest="config/Page_OMR_PDMX/dataset.json",
        vocab_manifest="vocab/fixture.json",
        renderer_weights={"mscore": 0.5, "verovio": 0.5},
        batch_size=1,
        num_workers=0,
        tokenization_mode="bekern",
        steps_per_epoch=12,
        shuffle_buffer=8,
        seed=3407,
        runtime_augmentation=False,
    )
    return {
        "package_root": package_root,
        "snapshot_root": snapshot_root,
        "config": config,
    }


def _build_data(pdmx_fixture, **config_changes):
    config = replace(pdmx_fixture["config"], **config_changes)
    return PDMXPretrainingDataModule(
        config,
        package_root=pdmx_fixture["package_root"],
        snapshot_root=pdmx_fixture["snapshot_root"],
    )


def _collect_metadata(data):
    return [batch[3] for batch in data.train_dataloader()]


def test_train_dataset_yields_model_contract_and_metadata(pdmx_fixture):
    data = _build_data(pdmx_fixture)

    image, decoder_input, target, metadata = next(
        iter(data.train_dataloader())
    )

    assert image.shape == (1, 3, 64, 64)
    assert decoder_input.shape == target.shape
    assert metadata["renderer"] in {"verovio", "mscore"}
    assert metadata["sample_key"]
    assert metadata["original_size_wh"] in ([48, 32], [49, 33])
    assert metadata["final_shape_nchw"] == [1, 3, 64, 64]


def test_validation_dataset_is_ordered_and_variable_size_safe(pdmx_fixture):
    data = _build_data(pdmx_fixture)

    rows = list(data.val_dataloader())

    assert [row[3]["sample_key"] for row in rows] == ["val-a", "val-b"]
    assert all(row[0].shape == (1, 3, 64, 64) for row in rows)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_virtual_epoch_has_exact_global_length(pdmx_fixture, num_workers):
    data = _build_data(
        pdmx_fixture,
        num_workers=num_workers,
        steps_per_epoch=17,
    )

    rows = list(data.train_dataloader())

    assert len(rows) == 17


def test_renderer_sequence_is_reproducible_for_fixed_topology(pdmx_fixture):
    first = _build_data(pdmx_fixture, steps_per_epoch=32)
    second = _build_data(pdmx_fixture, steps_per_epoch=32)
    first.set_train_epoch(3)
    second.set_train_epoch(3)

    first_rows = _collect_metadata(first)
    second_rows = _collect_metadata(second)

    assert first_rows == second_rows
    assert {row["renderer"] for row in first_rows} == {
        "verovio",
        "mscore",
    }


def test_artifact_paths_are_resolved_against_package_root(
    pdmx_fixture,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    data = _build_data(pdmx_fixture)

    assert data.dataset_manifest_path == (
        pdmx_fixture["package_root"]
        / "config"
        / "Page_OMR_PDMX"
        / "dataset.json"
    ).resolve()
    assert data.vocab_manifest_path == (
        pdmx_fixture["package_root"] / "vocab" / "fixture.json"
    ).resolve()


def test_data_module_explicitly_has_no_test_split(pdmx_fixture):
    data = _build_data(pdmx_fixture)

    assert data.has_validation_split is True
    assert data.has_test_split is False
    assert data.stream_resume_mode == "virtual_epoch_boundary"


def test_protocol_metadata_captures_dataset_vocab_and_runtime_identity(
    pdmx_fixture,
):
    data = _build_data(pdmx_fixture)

    metadata = data.protocol_metadata()

    assert metadata["dataset_revision"] == PDMX_DATASET_REVISION
    assert metadata["dataset_manifest_sha256"]
    assert metadata["vocab_sha256"]
    assert metadata["vocab_base_digest"]
    assert metadata["renderer_shard_counts"] == {
        "mscore": 2,
        "verovio": 2,
    }
    assert metadata["renderer_sample_counts"] == {
        "mscore": 2,
        "verovio": 2,
    }
    assert metadata["validation_sample_count"] == 2
    assert metadata["train_validation_source_overlap"] == 0
    assert metadata["software_versions"]["webdataset"]
    assert metadata["software_versions"]["torch"]


def test_teacher_forcing_is_sample_scoped_and_does_not_touch_global_rng():
    target = torch.tensor([100, 1, 2, 3, 4, 5, 183])
    before = torch.random.get_rng_state().clone()

    first = deterministic_teacher_forcing(
        target,
        vocab_size=221,
        padding_token=0,
        probability=1.0,
        seed_material="sample-a",
    )
    second = deterministic_teacher_forcing(
        target,
        vocab_size=221,
        padding_token=0,
        probability=1.0,
        seed_material="sample-a",
    )

    assert torch.equal(first, second)
    assert first[0] == target[0]
    assert not torch.equal(first[1:], target[1:])
    assert torch.equal(torch.random.get_rng_state(), before)


def test_tiny_snapshot_scans_builds_vocab_and_yields_both_splits(
    pdmx_fixture,
):
    data = _build_data(pdmx_fixture, steps_per_epoch=32)

    train_rows = list(data.train_dataloader())
    validation_rows = list(data.val_dataloader())

    assert len(train_rows) == 32
    assert {row[3]["renderer"] for row in train_rows} == {
        "verovio",
        "mscore",
    }
    assert len(validation_rows) == 2
    assert data.has_test_split is False


@pytest.mark.pdmx_official
def test_fixed_revision_official_snapshot_smoke(monkeypatch):
    package_root = REPO_ROOT / "experiments" / "full_page_omr"
    dataset_manifest_path = (
        package_root
        / "config"
        / "Page_OMR_PDMX"
        / "dataset-manifest.v1.json"
    )
    vocab_manifest_path = (
        package_root / "vocab" / "FullPageOMR_BeKern_v1.json"
    )
    try:
        snapshot_root = resolve_local_snapshot()
    except FileNotFoundError:
        pytest.skip(
            "fixed PDMX revision is absent; run `python -m "
            "experiments.full_page_omr.pdmx_manifest prepare`"
        )
    if not dataset_manifest_path.is_file() or not vocab_manifest_path.is_file():
        pytest.skip(
            "PDMX artifacts are absent; run the scan and build-vocabulary "
            "commands documented in README.md"
        )

    monkeypatch.setattr(_globals, "resolution", 1024)
    manifest = load_pdmx_dataset_manifest(dataset_manifest_path)
    vocabulary = load_vocabulary_manifest(vocab_manifest_path)
    decoder = _PDMXSampleDecoder(
        vocabulary,
        teacher_forcing_probability=0.0,
    )
    selected = [
        next(
            shard
            for shard in manifest["train_shards"]
            if shard["renderer"] == renderer
        )
        for renderer in ("verovio", "mscore")
    ]
    selected.append(manifest["validation_shards"][0])

    for ordinal, shard in enumerate(selected):
        path = snapshot_root / Path(
            *Path(shard["logical_path"]).parts
        )
        assert path.is_file()
        assert path.stat().st_size == shard["bytes"]
        sample = next(_webdataset_samples(path))
        decoded = decoder.decode(
            sample,
            shard=shard,
            virtual_epoch=0,
            global_ordinal=ordinal,
            source_cycle=0,
            occurrence_index=0,
        )
        assert decoded[0].shape == (1, 3, 1024, 1024)
        assert decoded[1].shape == decoded[2].shape
        assert decoded[3]["sample_key"]


def test_consumption_audit_records_both_renderers_and_first_images(tmp_path):
    callback = PDMXConsumptionAuditCallback(tmp_path)
    trainer = SimpleNamespace(
        current_epoch=3,
        datamodule=SimpleNamespace(
            config=SimpleNamespace(
                renderer_weights={"verovio": 0.5, "mscore": 0.5}
            )
        ),
    )
    model = Mock()
    for renderer, source_cycle in (("verovio", 0), ("mscore", 2)):
        batch = (
            torch.zeros(1, 3, 64, 64),
            torch.zeros(1, 4, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            {
                "renderer": renderer,
                "source_cycle": source_cycle,
                "voice_bucket": "voices_2",
                "density_bucket": "medium",
                "original_size_wh": [48, 32],
                "final_shape_nchw": [1, 3, 64, 64],
            },
        )
        callback.on_train_batch_end(trainer, model, None, batch, 0)

    callback.on_train_epoch_end(trainer, model)

    directory = tmp_path / "pdmx_consumption" / "epoch_000003"
    audit = json.loads(
        (directory / "audit.json").read_text(encoding="utf-8")
    )
    assert audit["renderer_counts"] == {"mscore": 1, "verovio": 1}
    assert audit["renderer_ratios"] == {"mscore": 0.5, "verovio": 0.5}
    assert audit["source_cycle_counts"] == {
        "mscore:2": 1,
        "verovio:0": 1,
    }
    assert (directory / "mscore_first.png").is_file()
    assert (directory / "verovio_first.png").is_file()
    assert model.log.call_count == 2


def test_consumption_audit_rejects_epoch_missing_enabled_renderer(tmp_path):
    callback = PDMXConsumptionAuditCallback(tmp_path)
    trainer = SimpleNamespace(
        current_epoch=0,
        datamodule=SimpleNamespace(
            config=SimpleNamespace(
                renderer_weights={"verovio": 0.5, "mscore": 0.5}
            )
        ),
    )
    batch = (
        torch.zeros(1, 3, 64, 64),
        torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.long),
        {
            "renderer": "verovio",
            "source_cycle": 0,
            "voice_bucket": "voices_2",
            "density_bucket": "medium",
            "original_size_wh": [48, 32],
            "final_shape_nchw": [1, 3, 64, 64],
        },
    )
    callback.on_train_batch_end(trainer, Mock(), None, batch, 0)

    with pytest.raises(RuntimeError, match="mscore"):
        callback.on_train_epoch_end(trainer, Mock())
