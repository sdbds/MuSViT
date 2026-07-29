from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

from lightning.pytorch import Callback, LightningDataModule
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import webdataset as wds

from .batching import batch_preparation_img2seq
from .config.ExperimentConfigWrapper import ExperimentConfig, PDMXData
from .data_augmentation.data_augmentation import convert_img_to_tensor
from .pdmx_manifest import (
    PDMX_DATASET_ID,
    PDMX_DATASET_REVISION,
    load_pdmx_dataset_manifest,
    normalize_source_id,
    resolve_local_snapshot,
    verify_pdmx_dataset_manifest,
)
from .tokenization import parse_kern_file
from .utils.vocab_manifest import (
    VocabularyManifest,
    load_vocabulary_manifest,
)


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def deterministic_teacher_forcing(
    target: torch.Tensor,
    *,
    vocab_size: int,
    padding_token: int,
    probability: float,
    seed_material: str,
) -> torch.Tensor:
    if target.ndim != 1:
        raise ValueError("teacher-forcing target must be one-dimensional")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("teacher-forcing probability must be between zero and one")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("seed_material must be a non-empty string")

    result = target.clone()
    generator = np.random.default_rng(_stable_seed("teacher-forcing", seed_material))
    for index in range(1, len(result)):
        if int(target[index]) == padding_token:
            continue
        if generator.random() < probability:
            result[index] = int(generator.integers(0, vocab_size))
    return result


def _buffered_shuffle(
    source: Iterable[tuple[dict[str, Any], Mapping[str, Any]]],
    *,
    buffer_size: int,
    rng: random.Random,
) -> Iterator[tuple[dict[str, Any], Mapping[str, Any]]]:
    buffer = []
    for item in source:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def _required_bytes(
    sample: Mapping[str, Any],
    field_name: str,
    *,
    renderer: str,
    shard: str,
    key: str,
) -> bytes:
    value = sample.get(field_name)
    if not isinstance(value, bytes):
        raise ValueError(
            f"{renderer}:{shard}:{key} is missing byte field {field_name}"
        )
    return value


def _decode_text(
    payload: bytes,
    *,
    renderer: str,
    shard: str,
    key: str,
    field_name: str,
) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{renderer}:{shard}:{key} has invalid UTF-8 in {field_name}"
        ) from error


def _logical_path(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"invalid logical shard path: {value!r}")
    candidate = (root / Path(*pure.parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"logical shard path escapes snapshot: {value!r}")
    return candidate


class _PDMXSampleDecoder:
    def __init__(
        self,
        vocabulary: VocabularyManifest,
        *,
        teacher_forcing_probability: float,
    ) -> None:
        self.vocabulary = vocabulary
        self.w2i = vocabulary.w2i
        self.i2w = vocabulary.i2w
        self.padding_token = self.w2i["<pad>"]
        self.teacher_forcing_probability = teacher_forcing_probability

    def decode(
        self,
        sample: Mapping[str, Any],
        *,
        shard: Mapping[str, Any],
        virtual_epoch: int,
        global_ordinal: int,
        source_cycle: int,
        occurrence_index: int,
    ):
        renderer = str(shard["renderer"])
        logical_path = str(shard["logical_path"])
        key_value = sample.get("__key__")
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(f"{renderer}:{logical_path} sample has no __key__")
        key = key_value

        image_bytes = _required_bytes(
            sample,
            "image.png",
            renderer=renderer,
            shard=logical_path,
            key=key,
        )
        kern = _decode_text(
            _required_bytes(
                sample,
                "kern.txt",
                renderer=renderer,
                shard=logical_path,
                key=key,
            ),
            renderer=renderer,
            shard=logical_path,
            key=key,
            field_name="kern.txt",
        )
        source_id = normalize_source_id(
            _decode_text(
                _required_bytes(
                    sample,
                    "source.txt",
                    renderer=renderer,
                    shard=logical_path,
                    key=key,
                ),
                renderer=renderer,
                shard=logical_path,
                key=key,
                field_name="source.txt",
            )
        )
        content_tokens = tuple(
            token
            for token in parse_kern_file(kern, tokenization_mode="bekern")
            if token != ""
        )
        tokens = ("<bos>", *content_tokens, "<eos>")
        missing_tokens = sorted(
            set(tokens) - set(self.w2i),
            key=lambda token: token.encode("utf-8"),
        )
        if missing_tokens:
            raise ValueError(
                f"{renderer}:{logical_path}:{key} has OOV tokens: {missing_tokens}"
            )
        target = torch.tensor(
            [self.w2i[token] for token in tokens],
            dtype=torch.long,
        )
        seed_material = (
            f"{virtual_epoch}:{renderer}:{logical_path}:{key}:"
            f"{occurrence_index}"
        )
        decoder_input = deterministic_teacher_forcing(
            target,
            vocab_size=len(self.w2i),
            padding_token=self.padding_token,
            probability=self.teacher_forcing_probability,
            seed_material=seed_material,
        )

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            rgb = image.convert("RGB")
            original_size = tuple(rgb.size)
            image_array = np.array(rgb, copy=True)
        image_tensor = convert_img_to_tensor(image_array)

        fill = None
        fill_bytes = sample.get("fill.txt")
        if fill_bytes is not None:
            if not isinstance(fill_bytes, bytes):
                raise ValueError(
                    f"{renderer}:{logical_path}:{key} fill.txt must be bytes"
                )
            fill = _decode_text(
                fill_bytes,
                renderer=renderer,
                shard=logical_path,
                key=key,
                field_name="fill.txt",
            ).strip() or None

        metadata = {
            "renderer": renderer,
            "shard": logical_path,
            "sample_key": key,
            "source_id": source_id,
            "source_cycle": source_cycle,
            "virtual_epoch": virtual_epoch,
            "global_ordinal": global_ordinal,
            "occurrence_index": occurrence_index,
            "voice_bucket": shard.get("voice_bucket"),
            "density_bucket": shard.get("density_bucket"),
            "fill": fill,
            "original_size_wh": list(original_size),
            "final_shape_nchw": list(image_tensor.shape),
        }
        return image_tensor, decoder_input, target, metadata


def _webdataset_samples(path: Path) -> Iterator[dict[str, Any]]:
    # A bare Windows path is parsed as the unsupported "c:" URL scheme.
    # WebDataset 1.0 opens ``urlparse(url).path`` directly. ``Path.as_uri()``
    # produces ``/C:/...`` on Windows, so use the equivalent single-slash form.
    url = f"file:{path.resolve().as_posix()}"
    dataset = wds.WebDataset(
        [url],
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=True,
    )
    yield from dataset


class PDMXTrainDataset(IterableDataset):
    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        vocabulary: VocabularyManifest,
        snapshot_root: str | Path,
        steps_per_epoch: int,
        shuffle_buffer: int,
        seed: int,
        num_workers_contract: int,
        teacher_forcing_probability: float = 0.2,
    ) -> None:
        super().__init__()
        self.manifest = dict(manifest)
        self.snapshot_root = Path(snapshot_root)
        self.steps_per_epoch = int(steps_per_epoch)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.num_workers_contract = int(num_workers_contract)
        self.virtual_epoch = 0
        self.decoder = _PDMXSampleDecoder(
            vocabulary,
            teacher_forcing_probability=teacher_forcing_probability,
        )
        self.w2i = self.decoder.w2i
        self.i2w = self.decoder.i2w
        self._shards_by_renderer = {
            renderer: tuple(
                shard
                for shard in self.manifest["train_shards"]
                if shard["renderer"] == renderer
            )
            for renderer in sorted(self.manifest["renderer_weights"])
        }
        if any(not shards for shards in self._shards_by_renderer.values()):
            raise ValueError("each renderer must have at least one train shard")

    def set_virtual_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("virtual epoch must be a non-negative integer")
        self.virtual_epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _renderer_for_ordinal(self, ordinal: int) -> str:
        renderers = tuple(sorted(self.manifest["renderer_weights"]))
        unit = _stable_seed(
            "renderer",
            self.seed,
            self.virtual_epoch,
            ordinal,
        ) / float(2**64)
        cumulative = 0.0
        for renderer in renderers:
            cumulative += self.manifest["renderer_weights"][renderer]
            if unit < cumulative:
                return renderer
        return renderers[-1]

    def _renderer_stream(
        self,
        renderer: str,
        *,
        worker_id: int,
        worker_count: int,
    ) -> Iterator[tuple[dict[str, Any], Mapping[str, Any], int]]:
        cycle = 0
        all_shards = list(self._shards_by_renderer[renderer])
        per_worker_buffer = max(
            1,
            math.ceil(self.shuffle_buffer / worker_count),
        )
        while True:
            shard_rng = random.Random(
                _stable_seed(
                    "shards",
                    self.seed,
                    self.virtual_epoch,
                    renderer,
                    cycle,
                )
            )
            ordered_shards = list(all_shards)
            shard_rng.shuffle(ordered_shards)
            worker_shards = ordered_shards[worker_id::worker_count]
            if not worker_shards:
                raise RuntimeError(
                    f"renderer {renderer} has {len(all_shards)} shards but "
                    f"worker topology requires {worker_count}"
                )

            def raw_samples():
                for shard in worker_shards:
                    path = _logical_path(
                        self.snapshot_root,
                        shard["logical_path"],
                    )
                    for sample in _webdataset_samples(path):
                        yield sample, shard

            sample_rng = random.Random(
                _stable_seed(
                    "samples",
                    self.seed,
                    self.virtual_epoch,
                    renderer,
                    cycle,
                    worker_id,
                )
            )
            yielded = False
            for sample, shard in _buffered_shuffle(
                raw_samples(),
                buffer_size=per_worker_buffer,
                rng=sample_rng,
            ):
                yielded = True
                yield sample, shard, cycle
            if not yielded:
                raise RuntimeError(
                    f"renderer {renderer} worker {worker_id} produced no samples"
                )
            cycle += 1

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        configured_count = self.num_workers_contract or 1
        if worker_count != configured_count:
            raise RuntimeError(
                "DataLoader worker topology does not match PDMX config: "
                f"configured={configured_count}, actual={worker_count}"
            )

        streams = {
            renderer: self._renderer_stream(
                renderer,
                worker_id=worker_id,
                worker_count=worker_count,
            )
            for renderer in self._shards_by_renderer
        }
        occurrences: Counter[tuple[str, str, str]] = Counter()
        for ordinal in range(
            worker_id,
            self.steps_per_epoch,
            worker_count,
        ):
            renderer = self._renderer_for_ordinal(ordinal)
            sample, shard, cycle = next(streams[renderer])
            key = str(sample.get("__key__", ""))
            occurrence_key = (renderer, str(shard["logical_path"]), key)
            occurrence_index = occurrences[occurrence_key]
            occurrences[occurrence_key] += 1
            yield self.decoder.decode(
                sample,
                shard=shard,
                virtual_epoch=self.virtual_epoch,
                global_ordinal=ordinal,
                source_cycle=cycle,
                occurrence_index=occurrence_index,
            )


class PDMXValidationDataset(Dataset):
    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        vocabulary: VocabularyManifest,
        snapshot_root: str | Path,
    ) -> None:
        super().__init__()
        self.decoder = _PDMXSampleDecoder(
            vocabulary,
            teacher_forcing_probability=0.0,
        )
        self.w2i = self.decoder.w2i
        self.i2w = self.decoder.i2w
        rows = []
        root = Path(snapshot_root)
        for shard in sorted(
            manifest["validation_shards"],
            key=lambda item: item["logical_path"],
        ):
            path = _logical_path(root, shard["logical_path"])
            for sample in _webdataset_samples(path):
                key = str(sample.get("__key__", ""))
                rows.append(
                    (
                        str(shard["logical_path"]),
                        key,
                        sample,
                        dict(shard),
                    )
                )
        self._encoded_rows = tuple(
            (sample, shard)
            for _, _, sample, shard in sorted(
                rows,
                key=lambda item: (item[0], item[1]),
            )
        )
        if len(self._encoded_rows) != manifest["validation_sample_count"]:
            raise ValueError(
                "validation sample count differs from dataset manifest"
            )

    def __len__(self) -> int:
        return len(self._encoded_rows)

    def __getitem__(self, index: int):
        sample, shard = self._encoded_rows[index]
        return self.decoder.decode(
            sample,
            shard=shard,
            virtual_epoch=0,
            global_ordinal=index,
            source_cycle=0,
            occurrence_index=0,
        )


class PDMXPretrainingDataModule(LightningDataModule):
    has_validation_split = True
    has_test_split = False
    encoder_unfreeze_step = 0
    curriculum_step_offset = 0
    stream_resume_mode = "virtual_epoch_boundary"

    def __init__(
        self,
        config: ExperimentConfig | PDMXData,
        *,
        package_root: str | Path | None = None,
        snapshot_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        data_config = config.data if isinstance(config, ExperimentConfig) else config
        if not isinstance(data_config, PDMXData):
            raise TypeError("PDMXPretrainingDataModule requires PDMXData")
        self.config = data_config
        self.package_root = (
            Path(package_root).resolve()
            if package_root is not None
            else Path(__file__).resolve().parent
        )
        self.dataset_manifest_path = self.resolve_artifact_path(
            data_config.dataset_manifest
        )
        self.vocab_manifest_path = self.resolve_artifact_path(
            data_config.vocab_manifest
        )
        self.dataset_manifest = load_pdmx_dataset_manifest(
            self.dataset_manifest_path
        )
        self.vocabulary = load_vocabulary_manifest(self.vocab_manifest_path)
        self.snapshot_root = (
            Path(snapshot_root).resolve()
            if snapshot_root is not None
            else resolve_local_snapshot(
                data_config.dataset_id,
                data_config.dataset_revision,
            ).resolve()
        )
        verify_pdmx_dataset_manifest(
            self.dataset_manifest,
            self.snapshot_root,
        )
        self._validate_identity()

        self.data_path = data_config.dataset_id
        self.vocab_name = self.vocabulary.name
        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.tokenization_mode = data_config.tokenization_mode
        self.steps_per_epoch = data_config.steps_per_epoch
        self.shuffle_buffer = data_config.shuffle_buffer
        self.seed = data_config.seed

        self.train_dataset = PDMXTrainDataset(
            manifest=self.dataset_manifest,
            vocabulary=self.vocabulary,
            snapshot_root=self.snapshot_root,
            steps_per_epoch=self.steps_per_epoch,
            shuffle_buffer=self.shuffle_buffer,
            seed=self.seed,
            num_workers_contract=self.num_workers,
        )
        self.val_dataset = PDMXValidationDataset(
            manifest=self.dataset_manifest,
            vocabulary=self.vocabulary,
            snapshot_root=self.snapshot_root,
        )

    def resolve_artifact_path(self, value: str | Path) -> Path:
        path = Path(value)
        candidate = (
            path.resolve()
            if path.is_absolute()
            else (self.package_root / path).resolve()
        )
        if candidate != self.package_root and self.package_root not in candidate.parents:
            raise ValueError(f"artifact path escapes package root: {value}")
        return candidate

    def _validate_identity(self) -> None:
        if self.dataset_manifest["dataset_id"] != self.config.dataset_id:
            raise ValueError("PDMX config and dataset manifest id differ")
        if self.dataset_manifest["dataset_revision"] != self.config.dataset_revision:
            raise ValueError("PDMX config and dataset manifest revision differ")
        if self.dataset_manifest["renderer_weights"] != self.config.renderer_weights:
            raise ValueError("PDMX config and dataset renderer weights differ")
        if self.vocabulary.tokenization_mode != self.config.tokenization_mode:
            raise ValueError("PDMX config and vocabulary tokenization differ")
        dataset_digest = self.dataset_manifest["manifest_sha256"]
        if dataset_digest not in self.vocabulary.source_dataset_manifests:
            raise ValueError(
                "vocabulary does not reference the selected dataset manifest"
            )
        dataset_tokens = set(self.dataset_manifest["train_token_frequencies"])
        dataset_tokens.update(
            self.dataset_manifest["validation_token_frequencies"]
        )
        missing_tokens = sorted(
            dataset_tokens - set(self.vocabulary.ordered_tokens),
            key=lambda token: token.encode("utf-8"),
        )
        if missing_tokens:
            raise ValueError(
                f"dataset manifest contains vocabulary OOVs: {missing_tokens}"
            )

    def set_train_epoch(self, epoch: int) -> None:
        self.train_dataset.set_virtual_epoch(epoch)

    def protocol_metadata(self) -> dict[str, Any]:
        renderer_shard_counts = Counter(
            shard["renderer"]
            for shard in self.dataset_manifest["train_shards"]
        )
        renderer_sample_counts = Counter()
        for shard in self.dataset_manifest["train_shards"]:
            renderer_sample_counts[shard["renderer"]] += shard["sample_count"]
        return {
            "dataset_id": PDMX_DATASET_ID,
            "dataset_revision": PDMX_DATASET_REVISION,
            "dataset_license": self.dataset_manifest["license"],
            "dataset_manifest_sha256": self.dataset_manifest["manifest_sha256"],
            "selected_renderers": list(
                self.dataset_manifest["selected_renderers"]
            ),
            "renderer_shard_counts": dict(
                sorted(renderer_shard_counts.items())
            ),
            "renderer_sample_counts": dict(
                sorted(renderer_sample_counts.items())
            ),
            "train_sample_count": self.dataset_manifest["train_sample_count"],
            "validation_sample_count": self.dataset_manifest[
                "validation_sample_count"
            ],
            "train_sequence_lengths": dict(
                self.dataset_manifest["train_sequence_lengths"]
            ),
            "validation_sequence_lengths": dict(
                self.dataset_manifest["validation_sequence_lengths"]
            ),
            "excluded_samples": list(
                self.dataset_manifest["excluded_samples"]
            ),
            "train_validation_source_overlap": self.dataset_manifest[
                "train_validation_source_overlap"
            ],
            "max_sequence_length": self.dataset_manifest[
                "max_sequence_length"
            ],
            "vocab_name": self.vocabulary.name,
            "vocab_size": len(self.vocabulary.ordered_tokens),
            "vocab_sha256": self.vocabulary.vocab_sha256,
            "vocab_schema_version": self.vocabulary.schema_version,
            "vocab_base_name": self.vocabulary.base_name,
            "vocab_base_size": self.vocabulary.base_size,
            "vocab_base_digest": self.vocabulary.base_digest,
            "renderer_weights": dict(self.config.renderer_weights),
            "steps_per_epoch": self.steps_per_epoch,
            "shuffle_buffer": self.shuffle_buffer,
            "stream_seed": self.seed,
            "stream_resume_mode": self.stream_resume_mode,
            "stream_topology": {
                "world_size": 1,
                "num_workers": self.num_workers,
                "batch_size": self.batch_size,
            },
            "runtime_augmentation": False,
            "software_versions": {
                distribution: _package_version(distribution)
                for distribution in (
                    "webdataset",
                    "torch",
                    "lightning",
                    "datasets",
                    "huggingface-hub",
                )
            },
        }

    def _loader(self, dataset, *, num_workers: int) -> DataLoader:
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "shuffle": False,
            "collate_fn": batch_preparation_img2seq,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": False,
        }
        if num_workers > 0:
            kwargs.update(
                prefetch_factor=1,
                in_order=True,
            )
        return DataLoader(**kwargs)

    def train_dataloader(self):
        return self._loader(
            self.train_dataset,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return self._loader(self.val_dataset, num_workers=0)


class PDMXVirtualEpochCallback(Callback):
    def on_fit_start(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.world_size != 1:
            raise RuntimeError(
                "PDMX v1 supports only single-process training; "
                f"received world_size={trainer.world_size}"
            )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        data = trainer.datamodule
        if data is None or not hasattr(data, "set_train_epoch"):
            raise RuntimeError("PDMX callback requires a compatible data module")
        data.set_train_epoch(int(trainer.current_epoch))


class PDMXConsumptionAuditCallback(Callback):
    def __init__(self, run_instance_directory: str | Path) -> None:
        super().__init__()
        self.run_instance_directory = Path(run_instance_directory)
        self._reset()

    def _reset(self) -> None:
        self.renderer_counts: Counter[str] = Counter()
        self.source_cycle_counts: Counter[str] = Counter()
        self.voice_bucket_counts: Counter[str] = Counter()
        self.density_bucket_counts: Counter[str] = Counter()
        self.first_samples: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        del trainer, pl_module, outputs, batch_idx
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise ValueError("PDMX audit requires batch metadata")
        metadata = batch[3]
        if not isinstance(metadata, dict):
            raise ValueError("PDMX batch metadata must be a dictionary")
        renderer = metadata.get("renderer")
        if not isinstance(renderer, str) or not renderer:
            raise ValueError("PDMX batch metadata has no renderer")

        self.renderer_counts[renderer] += 1
        self.source_cycle_counts[
            f"{renderer}:{metadata.get('source_cycle')}"
        ] += 1
        for field_name, counter in (
            ("voice_bucket", self.voice_bucket_counts),
            ("density_bucket", self.density_bucket_counts),
        ):
            value = metadata.get(field_name)
            counter[f"{renderer}:{value}"] += 1

        if renderer not in self.first_samples:
            image = batch[0]
            if (
                not isinstance(image, torch.Tensor)
                or image.ndim != 4
                or image.shape[0] != 1
                or image.shape[1] not in (1, 3)
            ):
                raise ValueError(
                    "PDMX audit image must be NCHW with batch size one"
                )
            self.first_samples[renderer] = (
                image.detach().cpu().clone(),
                dict(metadata),
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        config = getattr(getattr(trainer, "datamodule", None), "config", None)
        renderer_weights = getattr(config, "renderer_weights", None)
        if not isinstance(renderer_weights, dict) or not renderer_weights:
            raise RuntimeError("PDMX audit cannot determine enabled renderers")
        enabled = tuple(sorted(renderer_weights))
        missing = [
            renderer
            for renderer in enabled
            if self.renderer_counts[renderer] == 0
        ]
        if missing:
            raise RuntimeError(
                "completed PDMX virtual epoch contains no samples from: "
                + ", ".join(missing)
            )

        total = sum(self.renderer_counts.values())
        ratios = {
            renderer: self.renderer_counts[renderer] / total
            for renderer in enabled
        }
        for renderer, ratio in ratios.items():
            pl_module.log(
                f"data/renderer_{renderer}_ratio",
                ratio,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        epoch = int(trainer.current_epoch)
        directory = (
            self.run_instance_directory
            / "pdmx_consumption"
            / f"epoch_{epoch:06d}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        first_sample_metadata = {}
        for renderer in enabled:
            image, metadata = self.first_samples[renderer]
            _save_tensor_image(
                directory / f"{renderer}_first.png",
                image,
            )
            first_sample_metadata[renderer] = {
                "image": f"{renderer}_first.png",
                "original_size_wh": metadata.get("original_size_wh"),
                "final_shape_nchw": metadata.get("final_shape_nchw"),
                "sample_key": metadata.get("sample_key"),
                "shard": metadata.get("shard"),
            }

        payload = {
            "virtual_epoch": epoch,
            "renderer_counts": dict(sorted(self.renderer_counts.items())),
            "renderer_ratios": ratios,
            "source_cycle_counts": dict(
                sorted(self.source_cycle_counts.items())
            ),
            "voice_bucket_counts": dict(
                sorted(self.voice_bucket_counts.items())
            ),
            "density_bucket_counts": dict(
                sorted(self.density_bucket_counts.items())
            ),
            "first_samples": first_sample_metadata,
        }
        (directory / "audit.json").write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self._reset()


def _save_tensor_image(path: Path, image: torch.Tensor) -> None:
    values = (
        image.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    if values.shape[2] == 1:
        values = values[:, :, 0]
    Image.fromarray(values).save(path)
