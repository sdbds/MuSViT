from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, TypeVar, Type, cast

from ..pdmx_manifest import (
    PDMX_DATASET_ID,
    PDMX_DATASET_REVISION,
    PDMX_RENDERER_WEIGHTS,
)
from ..tokenization import validate_tokenization_mode


T = TypeVar("T")


def from_str(x: Any, field_name: str = "value") -> str:
    if not isinstance(x, str):
        raise TypeError(f"{field_name} must be a string")
    return x


def from_int(x: Any, field_name: str = "value") -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{field_name} must be an integer")
    return x


def from_float(x: Any, field_name: str = "value") -> float:
    if not isinstance(x, (float, int)) or isinstance(x, bool):
        raise TypeError(f"{field_name} must be a number")
    return float(x)


def to_float(x: Any, field_name: str = "value") -> float:
    return from_float(x, field_name)


def to_class(c: Type[T], x: Any) -> dict:
    if not isinstance(x, c):
        raise TypeError(f"value must be an instance of {c.__name__}")
    return cast(Any, x).to_dict()


@dataclass
class Data:
    data_path: str
    batch_size: int
    vocab_name: str
    num_workers: int
    tokenization_mode: str
    reduce_ratio: float
    skip_steps: int = 0

    @staticmethod
    def from_dict(obj: Any) -> 'Data':
        if not isinstance(obj, dict):
            raise TypeError("data must be a dictionary")

        data_path = from_str(obj.get("data_path"), "data_path")
        batch_size = from_int(obj.get("batch_size"), "batch_size")
        vocab_name = from_str(obj.get("vocab_name"), "vocab_name")
        num_workers = from_int(obj.get("num_workers"), "num_workers")
        tokenization_mode = validate_tokenization_mode(
            from_str(obj.get("tokenization_mode"), "tokenization_mode")
        )
        reduce_ratio = from_float(obj.get("reduce_ratio"), "reduce_ratio")
        skip_steps = from_int(obj.get("skip_steps", 0), "skip_steps")

        if batch_size != 1:
            raise ValueError(f"batch_size must be exactly 1; got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative; got {num_workers}")
        if reduce_ratio <= 0:
            raise ValueError(f"reduce_ratio must be greater than zero; got {reduce_ratio}")
        if skip_steps < 0:
            raise ValueError(f"skip_steps must be non-negative; got {skip_steps}")

        return Data(data_path, batch_size, vocab_name, num_workers, tokenization_mode, reduce_ratio, skip_steps)

    def to_dict(self) -> dict:
        validated = Data.from_dict(
            {
                "data_path": self.data_path,
                "batch_size": self.batch_size,
                "vocab_name": self.vocab_name,
                "num_workers": self.num_workers,
                "tokenization_mode": self.tokenization_mode,
                "reduce_ratio": self.reduce_ratio,
                "skip_steps": self.skip_steps,
            }
        )
        return {
            "data_path": validated.data_path,
            "batch_size": validated.batch_size,
            "vocab_name": validated.vocab_name,
            "num_workers": validated.num_workers,
            "tokenization_mode": validated.tokenization_mode,
            "reduce_ratio": validated.reduce_ratio,
            "skip_steps": validated.skip_steps,
        }


_PDMX_DATA_FIELDS = {
    "type",
    "dataset_id",
    "dataset_revision",
    "dataset_manifest",
    "vocab_manifest",
    "renderer_weights",
    "batch_size",
    "num_workers",
    "tokenization_mode",
    "steps_per_epoch",
    "shuffle_buffer",
    "seed",
    "runtime_augmentation",
}


def _artifact_path(value: Any, field_name: str) -> str:
    path = from_str(value, field_name).strip().replace("\\", "/")
    if not path:
        raise ValueError(f"{field_name} must be non-empty")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"{field_name} must be package-relative")
    return pure_path.as_posix()


@dataclass
class PDMXData:
    dataset_id: str
    dataset_revision: str
    dataset_manifest: str
    vocab_manifest: str
    renderer_weights: dict[str, float]
    batch_size: int
    num_workers: int
    tokenization_mode: str
    steps_per_epoch: int
    shuffle_buffer: int
    seed: int
    runtime_augmentation: bool

    @property
    def skip_steps(self) -> int:
        return 0

    @property
    def reduce_ratio(self) -> float:
        return 1.0

    @staticmethod
    def from_dict(obj: Any) -> 'PDMXData':
        if not isinstance(obj, dict):
            raise TypeError("data must be a dictionary")
        unexpected = sorted(set(obj) - _PDMX_DATA_FIELDS)
        if unexpected:
            raise ValueError(f"unexpected PDMX data fields: {unexpected}")
        missing = sorted(_PDMX_DATA_FIELDS - set(obj))
        if missing:
            raise ValueError(f"missing PDMX data fields: {missing}")
        if obj.get("type") != "pdmx_webdataset":
            raise ValueError("PDMX data type must be 'pdmx_webdataset'")

        dataset_id = from_str(obj.get("dataset_id"), "dataset_id")
        if dataset_id != PDMX_DATASET_ID:
            raise ValueError(f"dataset_id must be {PDMX_DATASET_ID!r}")
        dataset_revision = from_str(
            obj.get("dataset_revision"),
            "dataset_revision",
        )
        if dataset_revision != PDMX_DATASET_REVISION:
            raise ValueError(
                f"dataset_revision must be {PDMX_DATASET_REVISION}"
            )

        dataset_manifest = _artifact_path(
            obj.get("dataset_manifest"),
            "dataset_manifest",
        )
        vocab_manifest = _artifact_path(
            obj.get("vocab_manifest"),
            "vocab_manifest",
        )

        raw_weights = obj.get("renderer_weights")
        if not isinstance(raw_weights, dict):
            raise TypeError("renderer_weights must be a dictionary")
        if set(raw_weights) != set(PDMX_RENDERER_WEIGHTS):
            raise ValueError(
                "renderer_weights must contain exactly "
                f"{sorted(PDMX_RENDERER_WEIGHTS)}"
            )
        renderer_weights = {}
        for renderer in sorted(raw_weights):
            value = from_float(
                raw_weights[renderer],
                f"renderer_weights.{renderer}",
            )
            renderer_weights[renderer] = value
        if renderer_weights != PDMX_RENDERER_WEIGHTS:
            raise ValueError(
                f"renderer_weights must be {PDMX_RENDERER_WEIGHTS}"
            )

        batch_size = from_int(obj.get("batch_size"), "batch_size")
        num_workers = from_int(obj.get("num_workers"), "num_workers")
        tokenization_mode = validate_tokenization_mode(
            from_str(obj.get("tokenization_mode"), "tokenization_mode")
        )
        steps_per_epoch = from_int(
            obj.get("steps_per_epoch"),
            "steps_per_epoch",
        )
        shuffle_buffer = from_int(
            obj.get("shuffle_buffer"),
            "shuffle_buffer",
        )
        seed = from_int(obj.get("seed"), "seed")
        runtime_augmentation = obj.get("runtime_augmentation")

        if batch_size != 1:
            raise ValueError(f"batch_size must be exactly 1; got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative; got {num_workers}")
        if tokenization_mode != "bekern":
            raise ValueError("PDMX tokenization_mode must be 'bekern'")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(runtime_augmentation, bool):
            raise TypeError("runtime_augmentation must be a boolean")
        if runtime_augmentation:
            raise ValueError("runtime_augmentation must be false for PDMX v1")

        return PDMXData(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            dataset_manifest=dataset_manifest,
            vocab_manifest=vocab_manifest,
            renderer_weights=renderer_weights,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenization_mode=tokenization_mode,
            steps_per_epoch=steps_per_epoch,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            runtime_augmentation=runtime_augmentation,
        )

    def to_dict(self) -> dict:
        return {
            "type": "pdmx_webdataset",
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "dataset_manifest": self.dataset_manifest,
            "vocab_manifest": self.vocab_manifest,
            "renderer_weights": dict(self.renderer_weights),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "tokenization_mode": self.tokenization_mode,
            "steps_per_epoch": self.steps_per_epoch,
            "shuffle_buffer": self.shuffle_buffer,
            "seed": self.seed,
            "runtime_augmentation": self.runtime_augmentation,
        }


DataConfig = Data | PDMXData


@dataclass
class ExperimentConfig:
    data: DataConfig

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentConfig':
        if not isinstance(obj, dict):
            raise TypeError("experiment config must be a dictionary")
        data_obj = obj.get("data")
        if not isinstance(data_obj, dict):
            raise TypeError("data must be a dictionary")
        data_type = data_obj.get("type")
        if data_type == "pdmx_webdataset":
            data: DataConfig = PDMXData.from_dict(data_obj)
        elif data_type in {None, "arrow"}:
            legacy_obj = dict(data_obj)
            legacy_obj.pop("type", None)
            data = Data.from_dict(legacy_obj)
        else:
            raise ValueError(f"unsupported data type: {data_type!r}")
        return ExperimentConfig(data)

    def to_dict(self) -> dict:
        return {"data": self.data.to_dict()}


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    return ExperimentConfig.from_dict(s)


def experiment_config_to_dict(x: ExperimentConfig) -> Any:
    return to_class(ExperimentConfig, x)
