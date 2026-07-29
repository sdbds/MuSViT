"""Worker-invariant dataset and DataLoader construction for staff OMR v2."""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .augmentation import build_augmentation
from .batching import ctc_collate
from .ctc import CTCPreflight
from .data_bundle import BundleSample
from .errors import ProtocolError
from .geometry import GeometryPlan, StaffImageProcessor
from .seeding import (
    sample_augment_seed,
    sample_order_key,
    worker_base_seed,
)
from .vocabulary import Vocabulary


SAMPLER_VERSION = "sha256_epoch_order_v1"
COLLATE_VERSION = "concatenated_ctc_targets_v1"


class StaffOMRDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int, str, bool]]
):
    """Immutable samples with explicit per-item epoch augmentation seeds."""

    def __init__(
        self,
        *,
        samples: tuple[BundleSample, ...],
        data_path: Path,
        vocabulary: Vocabulary,
        feasibility: dict[str, bool],
        plan: GeometryPlan,
        augmentation_profile: str,
        base_seed: int,
        training: bool,
    ):
        if not samples:
            raise ProtocolError("staff OMR dataset split cannot be empty")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
            raise ProtocolError("base_seed must be a non-negative integer")
        ordered = tuple(
            sorted(samples, key=lambda item: item.sample_id.encode("utf-8"))
        )
        if len({sample.sample_id for sample in ordered}) != len(ordered):
            raise ProtocolError("staff OMR dataset contains duplicate sample ids")
        missing = [
            sample.sample_id
            for sample in ordered
            if sample.sample_id not in feasibility
        ]
        if missing:
            raise ProtocolError(
                f"CTC feasibility is missing sample ids: {missing!r}"
            )
        self.samples = ordered
        self.data_path = Path(data_path).resolve(strict=True)
        self.target_ids = {
            sample.sample_id: vocabulary.encode(sample.target_tokens)
            for sample in ordered
        }
        self.feasibility = dict(feasibility)
        self.processor = StaffImageProcessor(plan)
        self.augmentation_profile = augmentation_profile
        self.base_seed = base_seed
        self.training = training
        self.augmentation = (
            build_augmentation(augmentation_profile) if training else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image(self, sample: BundleSample) -> Path:
        try:
            path = (self.data_path / sample.image_path).resolve(strict=True)
            path.relative_to(self.data_path)
        except (OSError, ValueError) as exc:
            raise ProtocolError(
                f"sample {sample.sample_id!r} image path is unavailable or "
                "escapes data_path"
            ) from exc
        if not path.is_file():
            raise ProtocolError(
                f"sample {sample.sample_id!r} image is not a file"
            )
        return path

    def __getitem__(
        self,
        item: int | tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, int, str, bool]:
        if self.training:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in item
                )
            ):
                raise ProtocolError(
                    "training dataset index must contain explicit "
                    "(epoch, manifest_index)"
                )
            epoch, index = item
            if epoch <= 0:
                raise ProtocolError("training epoch index must be positive")
        else:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ProtocolError("evaluation dataset index must be an integer")
            epoch = None
            index = item
        if index < 0 or index >= len(self.samples):
            raise IndexError(index)
        sample = self.samples[index]
        path = self._resolve_image(sample)
        try:
            with Image.open(path) as opened:
                image = np.asarray(opened.convert("RGB"))
        except OSError as exc:
            raise ProtocolError(
                f"cannot decode image for sample {sample.sample_id!r}: {path}"
            ) from exc
        if self.training and self.augmentation is not None:
            self.augmentation.set_random_seed(
                sample_augment_seed(
                    self.base_seed,
                    int(epoch),
                    sample.sample_id,
                )
            )
            image = self.augmentation(image=image)["image"]
        image_tensor = self.processor(image)
        target_ids = self.target_ids[sample.sample_id]
        target = torch.tensor(target_ids, dtype=torch.long)
        return (
            image_tensor,
            target,
            len(target_ids),
            sample.sample_id,
            self.feasibility[sample.sample_id],
        )


class EpochSampleSampler(Sampler[tuple[int, int]]):
    """Yield one full epoch ordered by a complete SHA-256 digest."""

    def __init__(
        self,
        dataset: StaffOMRDataset,
        *,
        base_seed: int,
        epoch: int,
    ):
        if not dataset.training:
            raise ProtocolError("EpochSampleSampler requires a training dataset")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ProtocolError("epoch must be a positive integer")
        self.dataset = dataset
        self.base_seed = base_seed
        self.epoch = epoch

    def __iter__(self):
        indices = sorted(
            range(len(self.dataset)),
            key=lambda index: sample_order_key(
                self.base_seed,
                self.epoch,
                self.dataset.samples[index].sample_id,
            ),
        )
        return iter((self.epoch, index) for index in indices)

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: StaffOMRDataset
    validation: StaffOMRDataset
    test: StaffOMRDataset


@dataclass(frozen=True, slots=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_datasets(
    *,
    samples: tuple[BundleSample, ...],
    data_path: str | Path,
    vocabulary: Vocabulary,
    preflight: CTCPreflight,
    plan: GeometryPlan,
    augmentation_profile: str,
    base_seed: int,
    retained_train_sample_ids: tuple[str, ...],
) -> DatasetSplits:
    if preflight.patch_cols != plan.output_cols:
        raise ProtocolError(
            "CTC preflight patch_cols differs from model output columns"
        )
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ProtocolError("dataset contains duplicate sample ids")
    record_by_id = {record.sample_id: record for record in preflight.records}
    if set(record_by_id) != set(sample_by_id):
        raise ProtocolError(
            "CTC preflight sample ids differ from dataset sample ids"
        )
    train_ids = {
        sample.sample_id for sample in samples if sample.split == "train"
    }
    retained = set(retained_train_sample_ids)
    if len(retained) != len(retained_train_sample_ids):
        raise ProtocolError("retained train sample ids contain duplicates")
    infeasible_train = {
        record.sample_id
        for record in preflight.records
        if record.split == "train" and not record.feasible
    }
    expected_retained = train_ids - infeasible_train
    if retained != expected_retained:
        raise ProtocolError(
            "retained train sample ids must equal all feasible train samples; "
            f"missing={sorted(expected_retained - retained)!r}, "
            f"unexpected={sorted(retained - expected_retained)!r}"
        )
    feasibility = {
        sample_id: record.feasible
        for sample_id, record in record_by_id.items()
    }
    source = Path(data_path)
    train_samples = tuple(sample_by_id[sample_id] for sample_id in retained)
    validation_samples = tuple(
        sample for sample in samples if sample.split == "val"
    )
    test_samples = tuple(sample for sample in samples if sample.split == "test")
    return DatasetSplits(
        train=StaffOMRDataset(
            samples=train_samples,
            data_path=source,
            vocabulary=vocabulary,
            feasibility=feasibility,
            plan=plan,
            augmentation_profile=augmentation_profile,
            base_seed=base_seed,
            training=True,
        ),
        validation=StaffOMRDataset(
            samples=validation_samples,
            data_path=source,
            vocabulary=vocabulary,
            feasibility=feasibility,
            plan=plan,
            augmentation_profile="none",
            base_seed=base_seed,
            training=False,
        ),
        test=StaffOMRDataset(
            samples=test_samples,
            data_path=source,
            vocabulary=vocabulary,
            feasibility=feasibility,
            plan=plan,
            augmentation_profile="none",
            base_seed=base_seed,
            training=False,
        ),
    )


def loader_contract(batch_size: int) -> dict[str, object]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ProtocolError("batch_size must be a positive integer")
    return {
        "batch_size": batch_size,
        "collate": COLLATE_VERSION,
        "drop_last": False,
        "in_order": True,
        "persistent_workers": False,
        "pin_memory": False,
        "shuffle": False,
        "train_sampler": SAMPLER_VERSION,
        "val_test_order": "manifest_sample_id_utf8_ascending",
        "worker_count_semantic": False,
    }


def require_in_order_capability(loader_class: type = DataLoader) -> None:
    try:
        parameters = inspect.signature(loader_class).parameters
    except (TypeError, ValueError) as exc:
        raise ProtocolError("cannot inspect DataLoader signature") from exc
    if "in_order" not in parameters:
        raise ProtocolError(
            "DataLoader signature lacks required in_order capability"
        )


def _validate_loader_inputs(batch_size: int, num_workers: int) -> None:
    loader_contract(batch_size)
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ProtocolError("num_workers must be a non-negative integer")


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))


def _loader_options(
    *,
    base_seed: int,
    epoch: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    _validate_loader_inputs(batch_size, num_workers)
    require_in_order_capability(DataLoader)
    generator = torch.Generator()
    generator.manual_seed(worker_base_seed(base_seed, epoch))
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": ctc_collate,
        "drop_last": False,
        "persistent_workers": False,
        "pin_memory": False,
        "in_order": True,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }


def build_train_loader(
    dataset: StaffOMRDataset,
    *,
    epoch: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    sampler = EpochSampleSampler(
        dataset,
        base_seed=dataset.base_seed,
        epoch=epoch,
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        shuffle=False,
        **_loader_options(
            base_seed=dataset.base_seed,
            epoch=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    )


def build_evaluation_loader(
    dataset: StaffOMRDataset,
    *,
    epoch: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if dataset.training:
        raise ProtocolError("evaluation loader requires a non-training dataset")
    return DataLoader(
        dataset,
        shuffle=False,
        **_loader_options(
            base_seed=dataset.base_seed,
            epoch=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    )


def build_data_loaders(
    datasets: DatasetSplits,
    *,
    epoch: int,
    batch_size: int,
    num_workers: int,
) -> DataLoaders:
    return DataLoaders(
        train=build_train_loader(
            datasets.train,
            epoch=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        validation=build_evaluation_loader(
            datasets.validation,
            epoch=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        test=build_evaluation_loader(
            datasets.test,
            epoch=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    )
