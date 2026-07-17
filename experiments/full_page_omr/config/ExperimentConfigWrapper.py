from dataclasses import dataclass
from typing import Any, TypeVar, Type, cast

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


@dataclass
class ExperimentConfig:
    data: Data

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentConfig':
        if not isinstance(obj, dict):
            raise TypeError("experiment config must be a dictionary")
        data = Data.from_dict(obj.get("data"))
        return ExperimentConfig(data)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = to_class(Data, self.data)
        return result


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    return ExperimentConfig.from_dict(s)


def experiment_config_to_dict(x: ExperimentConfig) -> Any:
    return to_class(ExperimentConfig, x)
