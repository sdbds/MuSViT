from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.staff_level_omr.protocol.backbone import BackboneMetadata
from experiments.staff_level_omr.protocol.ctc import (
    CTCFeasibilityRecord,
    CTCPreflight,
)
from experiments.staff_level_omr.protocol.data_bundle import BundleSample
from experiments.staff_level_omr.protocol.data_pipeline import (
    EpochSampleSampler,
    build_data_loaders,
    build_datasets,
    build_train_loader,
    loader_contract,
    require_in_order_capability,
)
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol.geometry import build_geometry_plan
from experiments.staff_level_omr.protocol.seeding import sample_order_key
from experiments.staff_level_omr.protocol.vocabulary import (
    TARGET_PARSER,
    TOKEN_SORT,
    VOCAB_SCHEMA,
    Vocabulary,
)


META = BackboneMetadata(
    model_id="fixture/vit",
    revision="a" * 40,
    image_height=24,
    image_width=40,
    patch_height=8,
    patch_width=8,
    hidden_size=16,
    num_channels=3,
    prefix_tokens=1,
    model_type="vit_mae",
    architectures=("ViTMAEForPreTraining",),
)


def _vocabulary() -> Vocabulary:
    return Vocabulary.from_document(
        {
            "schema_version": VOCAB_SCHEMA,
            "dataset_id": "fixture",
            "vocabulary_scope": "closed_corpus",
            "source_manifest_sha256": "a" * 64,
            "target_parser": TARGET_PARSER,
            "token_sort": TOKEN_SORT,
            "unicode_normalization": "none",
            "blank_id": 0,
            "tokens": ["bar", "note"],
        },
        manifest_sha256="a" * 64,
        dataset_id="fixture",
    )


def _samples(data_path: Path) -> tuple[BundleSample, ...]:
    result = []
    split_names = ("train",) * 8 + ("val",) * 2 + ("test",) * 2
    for index, split in enumerate(split_names):
        sample_id = f"sample-{index:02d}"
        image_name = f"{sample_id}_region.png"
        array = np.zeros((24, 40, 3), dtype=np.uint8)
        array[:, :, 0] = index * 17
        array[:, :, 1] = np.arange(40, dtype=np.uint8)
        array[:, :, 2] = np.arange(24, dtype=np.uint8)[:, None]
        Image.fromarray(array, "RGB").save(data_path / image_name)
        result.append(
            BundleSample(
                sample_id=sample_id,
                group_id=f"group-{index:02d}",
                image_path=image_name,
                image_size_bytes=(data_path / image_name).stat().st_size,
                image_sha256="b" * 64,
                target_path=f"{sample_id}_gt.txt",
                target_sha256="c" * 64,
                split=split,
                target_tokens=("note", "bar") if index % 2 else ("note",),
            )
        )
    return tuple(reversed(result))


def _preflight(samples: tuple[BundleSample, ...]) -> CTCPreflight:
    records = tuple(
        CTCFeasibilityRecord(
            sample_id=sample.sample_id,
            split=sample.split,
            target_length=len(sample.target_tokens),
            adjacent_repeats=0,
            required_frames=len(sample.target_tokens),
            available_frames=5,
            feasible=True,
        )
        for sample in sorted(
            samples,
            key=lambda item: item.sample_id.encode("utf-8"),
        )
    )
    return CTCPreflight(
        patch_cols=5,
        records=records,
        split_summaries={
            split: {
                "samples": sum(record.split == split for record in records),
            }
            for split in ("train", "val", "test")
        },
    )


def _make_datasets(tmp_path: Path, *, augmentation_profile: str):
    samples = _samples(tmp_path)
    preflight = _preflight(samples)
    plan = build_geometry_plan(META, "lora", patch_rows=3, patch_cols=5)
    retained = tuple(
        sample.sample_id for sample in samples if sample.split == "train"
    )
    return build_datasets(
        samples=samples,
        data_path=tmp_path,
        vocabulary=_vocabulary(),
        preflight=preflight,
        plan=plan,
        augmentation_profile=augmentation_profile,
        base_seed=7,
        retained_train_sample_ids=retained,
    )


def test_sampler_is_a_stable_permutation_independent_of_manifest_order(tmp_path):
    datasets = _make_datasets(tmp_path, augmentation_profile="none")
    sampler = EpochSampleSampler(datasets.train, base_seed=7, epoch=3)

    observed = list(sampler)
    observed_ids = [
        datasets.train.samples[index].sample_id for _, index in observed
    ]
    expected_ids = sorted(
        (sample.sample_id for sample in datasets.train.samples),
        key=lambda sample_id: sample_order_key(7, 3, sample_id),
    )

    assert observed_ids == expected_ids
    assert all(epoch == 3 for epoch, _ in observed)
    assert len(set(index for _, index in observed)) == len(datasets.train)


def test_dataset_requires_explicit_epoch_index_for_training(tmp_path):
    datasets = _make_datasets(tmp_path, augmentation_profile="none")

    with pytest.raises(ProtocolError, match="epoch"):
        datasets.train[0]

    image, target, length, sample_id, feasible = datasets.train[(2, 0)]
    assert image.shape == (3, 24, 40)
    assert target.ndim == 1
    assert target.numel() == length
    assert sample_id == datasets.train.samples[0].sample_id
    assert feasible is True


def test_validation_and_test_are_in_manifest_sample_id_order(tmp_path):
    datasets = _make_datasets(tmp_path, augmentation_profile="none")

    assert [item.sample_id for item in datasets.validation.samples] == [
        "sample-08",
        "sample-09",
    ]
    assert [item.sample_id for item in datasets.test.samples] == [
        "sample-10",
        "sample-11",
    ]
    assert datasets.validation[0][3] == "sample-08"
    assert datasets.test[1][3] == "sample-11"


def test_build_datasets_requires_exact_retained_train_set(tmp_path):
    samples = _samples(tmp_path)
    plan = build_geometry_plan(META, "lora", patch_rows=3, patch_cols=5)

    with pytest.raises(ProtocolError, match="retained"):
        build_datasets(
            samples=samples,
            data_path=tmp_path,
            vocabulary=_vocabulary(),
            preflight=_preflight(samples),
            plan=plan,
            augmentation_profile="none",
            base_seed=7,
            retained_train_sample_ids=("sample-00",),
        )


def test_loader_contract_fixes_semantics_but_excludes_worker_count():
    contract = loader_contract(batch_size=3)

    assert contract == {
        "batch_size": 3,
        "collate": "concatenated_ctc_targets_v1",
        "drop_last": False,
        "in_order": True,
        "persistent_workers": False,
        "pin_memory": False,
        "shuffle": False,
        "train_sampler": "sha256_epoch_order_v1",
        "val_test_order": "manifest_sample_id_utf8_ascending",
        "worker_count_semantic": False,
    }
    assert "num_workers" not in contract


def test_loader_capability_uses_signature_not_version_string():
    class BackportedLoader:
        def __init__(self, dataset, *, in_order=True):
            pass

    class MissingCapabilityLoader:
        def __init__(self, dataset):
            pass

    require_in_order_capability(BackportedLoader)
    with pytest.raises(ProtocolError, match="in_order"):
        require_in_order_capability(MissingCapabilityLoader)


def test_last_small_batch_keeps_true_size_and_lengths(tmp_path):
    datasets = _make_datasets(tmp_path, augmentation_profile="none")
    loader = build_train_loader(
        datasets.train,
        epoch=1,
        batch_size=3,
        num_workers=0,
    )

    batches = list(loader)

    assert [batch.images.shape[0] for batch in batches] == [3, 3, 2]
    assert [batch.target_lengths.numel() for batch in batches] == [3, 3, 2]
    assert all(batch.targets.ndim == 1 for batch in batches)


def test_build_data_loaders_uses_same_dataset_contract_for_all_splits(tmp_path):
    datasets = _make_datasets(tmp_path, augmentation_profile="none")
    loaders = build_data_loaders(
        datasets,
        epoch=1,
        batch_size=4,
        num_workers=0,
    )

    assert len(loaders.train) == 2
    assert next(iter(loaders.validation)).sample_ids == (
        "sample-08",
        "sample-09",
    )
    assert next(iter(loaders.test)).sample_ids == (
        "sample-10",
        "sample-11",
    )


@pytest.mark.slow
def test_augmented_epoch_is_byte_identical_with_workers_zero_two_and_four(
    tmp_path,
):
    datasets = _make_datasets(
        tmp_path,
        augmentation_profile="staff_omr_train_v1",
    )

    def collect(workers: int):
        loader = build_train_loader(
            datasets.train,
            epoch=2,
            batch_size=3,
            num_workers=workers,
        )
        return [
            (
                batch.sample_ids,
                tuple(batch.target_lengths.tolist()),
                batch.images.numpy().tobytes(),
            )
            for batch in loader
        ]

    baseline = collect(0)
    assert collect(2) == baseline
    assert collect(4) == baseline
