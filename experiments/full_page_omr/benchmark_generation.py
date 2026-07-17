"""Locked CUDA benchmark for full-prefix and incremental OMR generation."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


BENCHMARK_VERSION = "full_page_omr_generation_rtx4090_v1"
GPU_NAME = "NVIDIA GeForce RTX 4090"
GPU_UUID = "GPU-a70be80e-9cef-95c2-7557-52448110b38e"
CHECKPOINT_FILENAME = "polish_scores_cl_CL-epoch3500.ckpt"
CHECKPOINT_SHA256 = "ebfbcb14b7bb6fad60280939341c7b603d3713b3d7f67cd9cae3c44b8f161d0c"
CHECKPOINT_GLOBAL_STEP = 282200
FOUNDATION_MODEL_ID = "carlospm12/LSMT-MAE-Base-1024-16"
FOUNDATION_REVISION = "eecd5b327521225e65e1c2fe38ab99eb667c1609"
DATASET_ID = "antoniorv6/polish-scores"
DATASET_REVISION = "b3170c8b8f322885b566efe9e264af9328b5603f"
PREFIX_LENGTHS = (1024, 2048, 4096)
MICRO_WARMUPS = 3
MICRO_SAMPLES = 10
FULL_WARMUPS = 1
FULL_SAMPLES = 3


class BenchmarkUnavailable(RuntimeError):
    pass


def locked_inputs():
    return {
        "gpu": {"name": GPU_NAME, "uuid": GPU_UUID},
        "dtype": "torch.float16",
        "checkpoint": {
            "filename": CHECKPOINT_FILENAME,
            "sha256": CHECKPOINT_SHA256,
            "global_step": CHECKPOINT_GLOBAL_STEP,
        },
        "foundation": {
            "model_id": FOUNDATION_MODEL_ID,
            "revision": FOUNDATION_REVISION,
        },
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "validation": {
            "split": "val",
            "rows": list(range(10)),
            "reduce_ratio": 0.5,
            "resolution": 1024,
            "batch_size": 1,
            "maxlen": 7512,
            "decoding": "greedy",
            "attention_backend": "auto",
        },
        "prefix_source_row": 5,
        "prefix_lengths": list(PREFIX_LENGTHS),
        "micro_timing": {"warmups": MICRO_WARMUPS, "samples": MICRO_SAMPLES},
        "full_validation_timing": {
            "warmups": FULL_WARMUPS,
            "samples": FULL_SAMPLES,
        },
    }


def build_cycled_prefix(target_ids, *, length, bos_id, eos_id, pad_id):
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    special = {bos_id, eos_id, pad_id}
    content = [int(token) for token in target_ids if int(token) not in special]
    if not content:
        raise ValueError("target must contain at least one content token")
    prefix = [int(bos_id)]
    prefix.extend(content[index % len(content)] for index in range(length - 1))
    return tuple(prefix)


def _measure(function, *, warmups, samples, synchronize, clock):
    for _ in range(warmups):
        function()
        synchronize()

    seconds = []
    results = []
    for _ in range(samples):
        synchronize()
        started = clock()
        results.append(function())
        synchronize()
        seconds.append(clock() - started)
    return (
        {
            "warmups": warmups,
            "sample_count": samples,
            "seconds": seconds,
            "median_seconds": statistics.median(seconds),
        },
        results,
    )


def measure_microbenchmark(
    function,
    *,
    synchronize=torch.cuda.synchronize,
    clock=time.perf_counter,
):
    return _measure(
        function,
        warmups=MICRO_WARMUPS,
        samples=MICRO_SAMPLES,
        synchronize=synchronize,
        clock=clock,
    )


def measure_full_validation(
    function,
    *,
    synchronize=torch.cuda.synchronize,
    clock=time.perf_counter,
):
    return _measure(
        function,
        warmups=FULL_WARMUPS,
        samples=FULL_SAMPLES,
        synchronize=synchronize,
        clock=clock,
    )


def evaluate_gates(
    microbenchmark,
    full_validation,
    *,
    memory_contract_passed,
    numerical_passed,
):
    micro_1024 = microbenchmark["1024"]
    micro_2048 = microbenchmark["2048"]
    gates = {
        "micro_1024_not_slower": (
            micro_1024["incremental"]["median_seconds"]
            <= micro_1024["uncached"]["median_seconds"]
        ),
        "micro_2048_at_least_2x": (
            micro_2048["incremental"]["median_seconds"]
            <= micro_2048["uncached"]["median_seconds"] / 2.0
        ),
        "full_validation_at_most_90_percent": (
            full_validation["incremental"]["median_seconds"]
            <= full_validation["uncached"]["median_seconds"] * 0.9
        ),
        "token_sequences_equal": bool(full_validation["token_sequences_equal"]),
        "memory_contract": bool(memory_contract_passed),
        "micro_prefix_argmax_equal": bool(numerical_passed),
    }
    gates["passed"] = all(gates.values())
    return gates


def _write_report(output_path, report):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(checkpoint_path, output_path):
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "not-run",
        "locked_inputs": locked_inputs(),
    }
    runtime = _LockedBenchmarkRuntime()
    try:
        gpu_identity = runtime.ensure_reference_gpu()
        measured = runtime.run(checkpoint_path, gpu_identity)
    except BenchmarkUnavailable as exc:
        report["reason"] = str(exc)
        _write_report(output_path, report)
        return report

    report.update(measured)
    report["gates"] = evaluate_gates(
        report["decoder_microbenchmark"],
        report["full_validation"],
        memory_contract_passed=report["memory_contract"]["passed"],
        numerical_passed=report["numerical"]["all_prefix_argmax_equal"],
    )
    report["status"] = "passed" if report["gates"]["passed"] else "failed"
    _write_report(output_path, report)
    return report


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_nvidia_smi(arguments):
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkUnavailable("nvidia-smi could not identify the reference GPU") from exc


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_identity():
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": None}
    return {"revision": revision, "dirty": bool(status.strip())}


class _LockedBenchmarkRuntime:
    def ensure_reference_gpu(self):
        output = _run_nvidia_smi(
            [
                "--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        reference = None
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                continue
            index, name, uuid, driver, memory_used, utilization = fields
            if uuid == GPU_UUID:
                reference = {
                    "index": index,
                    "name": name,
                    "uuid": uuid,
                    "driver": driver,
                    "memory_used_mib_before": int(memory_used),
                    "utilization_percent_before": int(utilization),
                }
                break
        if reference is None:
            raise BenchmarkUnavailable(f"reference GPU {GPU_UUID} is unavailable")
        if reference["name"] != GPU_NAME:
            raise BenchmarkUnavailable(
                f"reference GPU name mismatch: {reference['name']!r}"
            )
        if (
            reference["memory_used_mib_before"] > 1024
            or reference["utilization_percent_before"] > 10
        ):
            raise BenchmarkUnavailable(
                "reference GPU is busy; locked benchmark requires the released RTX 4090"
            )
        if torch.cuda.is_initialized():
            raise BenchmarkUnavailable(
                "CUDA was initialized before the benchmark could lock the reference GPU"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = reference["index"]
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise BenchmarkUnavailable("reference GPU could not be isolated as CUDA device 0")
        if torch.cuda.get_device_name(0) != GPU_NAME:
            raise BenchmarkUnavailable("CUDA device 0 is not the locked RTX 4090")
        return reference

    @staticmethod
    def _verify_checkpoint(checkpoint_path):
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise BenchmarkUnavailable(f"reference checkpoint is unavailable: {path}")
        if path.name != CHECKPOINT_FILENAME:
            raise BenchmarkUnavailable(
                f"reference checkpoint filename mismatch: {path.name!r}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != CHECKPOINT_SHA256:
            raise BenchmarkUnavailable(
                "reference checkpoint SHA-256 mismatch; no substitute is permitted"
            )
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        actual_step = payload.get("global_step")
        if actual_step != CHECKPOINT_GLOBAL_STEP:
            raise BenchmarkUnavailable(
                f"reference checkpoint global_step mismatch: {actual_step!r}"
            )
        epoch = payload.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise BenchmarkUnavailable("reference checkpoint has an invalid epoch")
        return path, payload, {
            "path": str(path),
            "filename": path.name,
            "sha256": actual_sha256,
            "epoch": epoch,
            "global_step": actual_step,
        }

    @staticmethod
    def _external_inputs():
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        try:
            snapshot = snapshot_download(
                repo_id=FOUNDATION_MODEL_ID,
                revision=FOUNDATION_REVISION,
            )
        except Exception as exc:
            raise BenchmarkUnavailable(
                "locked foundation revision is unavailable"
            ) from exc
        try:
            rows = load_dataset(
                DATASET_ID,
                revision=DATASET_REVISION,
                split="val",
                keep_in_memory=False,
            )
        except Exception as exc:
            raise BenchmarkUnavailable("locked dataset revision is unavailable") from exc
        if len(rows) < 10:
            raise BenchmarkUnavailable("locked dataset revision has fewer than 10 val rows")
        return snapshot, rows

    @staticmethod
    def _load_model(payload, snapshot):
        from .smt_foundation.configuration_smt import SMTFoundationConfig
        from .smt_foundation.modeling_smt import SMTFoundationModelForCausalLM

        hyper_parameters = payload.get("hyper_parameters")
        if not isinstance(hyper_parameters, dict) or "smt_config" not in hyper_parameters:
            raise RuntimeError("reference checkpoint is missing smt_config")
        raw_config = hyper_parameters["smt_config"]
        if isinstance(raw_config, SMTFoundationConfig):
            config = copy.deepcopy(raw_config)
        elif isinstance(raw_config, dict):
            config = SMTFoundationConfig(**raw_config)
        else:
            raise RuntimeError("reference checkpoint has an unsupported smt_config")
        if config.foundation_weights != FOUNDATION_MODEL_ID:
            raise RuntimeError("reference checkpoint foundation model does not match")
        if config.maxlen != 7512 or config.padding_token != 0:
            raise RuntimeError("reference checkpoint sequence contract does not match")
        config.foundation_weights = snapshot
        config.attention_backend = "auto"
        model = SMTFoundationModelForCausalLM(config)

        state = payload.get("state_dict")
        if not isinstance(state, dict) or not state:
            raise RuntimeError("reference checkpoint is missing state_dict")
        if all(name.startswith("model.") for name in state):
            state = {name.removeprefix("model."): value for name, value in state.items()}
        model.load_state_dict(state, strict=True)
        return model.eval().to(torch.device("cuda", 0))

    @staticmethod
    def _prepare_samples(model, rows):
        import cv2
        import numpy as np

        from . import _globals
        from .data import parse_kern_file
        from .data_augmentation.data_augmentation import convert_img_to_tensor

        _globals.resolution = 1024
        samples = []
        for row_index in range(10):
            row = rows[row_index]
            image = np.asarray(row["image"])
            width = int(np.ceil(image.shape[1] * 0.5))
            height = int(np.ceil(image.shape[0] * 0.5))
            image = cv2.resize(image, (width, height))
            image_tensor = convert_img_to_tensor(image).to(torch.device("cuda", 0))
            tokens = [
                "<bos>",
                *parse_kern_file(row["transcription"], tokenization_mode="bekern"),
                "<eos>",
            ]
            try:
                target_ids = tuple(model.w2i[token] for token in tokens)
            except KeyError as exc:
                raise RuntimeError(
                    f"locked validation row {row_index} has unknown token {exc.args[0]!r}"
                ) from None
            samples.append({"row": row_index, "image": image_tensor, "target_ids": target_ids})
        return samples

    @staticmethod
    def _inference(function):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return function()

    @staticmethod
    def _resolved_backend(model):
        backends = {
            attention.last_backend
            for layer in model.decoder.decoder.layers
            for attention in (layer.input_attention, layer.cross_attention)
            if attention.last_backend is not None
        }
        if len(backends) != 1:
            raise RuntimeError(f"benchmark resolved mixed attention backends: {sorted(backends)}")
        return next(iter(backends))

    def _microbenchmark(self, model, sample):
        bos_id = model.w2i["<bos>"]
        eos_id = model.w2i["<eos>"]
        prefix = build_cycled_prefix(
            sample["target_ids"],
            length=max(PREFIX_LENGTHS),
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=model.padding_token,
        )
        prefix_tensor = torch.tensor(
            [prefix],
            device=torch.device("cuda", 0),
            dtype=torch.long,
        )
        encoder_output = self._inference(
            lambda: model.forward_encoder(sample["image"]).permute(0, 2, 1).contiguous()
        )
        memory = self._inference(lambda: model.prepare_generation_memory(encoder_output))
        state = model.decoder.init_generation_state(memory)
        states = {}
        needed_positions = {length - 1 for length in PREFIX_LENGTHS}
        for index in range(max(PREFIX_LENGTHS) - 1):
            state = self._inference(
                lambda index=index, state=state: model.decoder.decode_step(
                    memory,
                    prefix_tensor[:, index:index + 1],
                    state,
                )[2]
            )
            if state.position in needed_positions:
                states[state.position] = state

        results = {}
        numerical = {}
        post_states = {}
        for length in PREFIX_LENGTHS:
            full_prefix = prefix_tensor[:, :length]
            previous_state = states[length - 1]

            def uncached():
                return self._inference(
                    lambda: model.forward_decoder(
                        encoder_output,
                        full_prefix,
                        output_attentions=False,
                        use_cache=False,
                    )
                )

            def incremental():
                return self._inference(
                    lambda: model.decoder.decode_step(
                        memory,
                        full_prefix[:, -1:],
                        previous_state,
                    )
                )

            uncached_timing, _ = measure_microbenchmark(uncached)
            incremental_timing, _ = measure_microbenchmark(incremental)
            uncached_output = uncached()
            _, incremental_logits, post_state = incremental()
            post_states[length] = post_state
            uncached_logits = uncached_output.logits[:, :, -1:]
            max_difference = float(
                torch.max(torch.abs(uncached_logits.float() - incremental_logits.float()))
            )
            argmax_equal = bool(
                torch.equal(
                    torch.argmax(uncached_logits, dim=1),
                    torch.argmax(incremental_logits, dim=1),
                )
            )
            results[str(length)] = {
                "uncached": uncached_timing,
                "incremental": incremental_timing,
                "speedup": (
                    uncached_timing["median_seconds"]
                    / incremental_timing["median_seconds"]
                ),
            }
            numerical[str(length)] = {
                "max_logits_abs_diff": max_difference,
                "argmax_equal": argmax_equal,
            }

        longest = max(PREFIX_LENGTHS)
        before_lengths = [
            layer.self_kv.key.size(2) for layer in states[longest - 1].layers
        ]
        after_lengths = [
            layer.self_kv.key.size(2) for layer in post_states[longest].layers
        ]
        layer_count = len(model.decoder.decoder.layers)
        memory_contract = {
            "production_attention_window": model.decoder.dec_attn_win,
            "expected_attention_window": model.maxlen + 1,
            "self_kv_lengths_before_4096_step": before_lengths,
            "self_kv_lengths_after_4096_step": after_lengths,
            "static_cross_kv_layer_count": len(memory.cross_kv),
            "decoder_layer_count": layer_count,
            "state_contains_cross_kv": any(
                hasattr(layer, "cross_kv") for layer in post_states[longest].layers
            ),
        }
        memory_contract["passed"] = (
            memory_contract["production_attention_window"]
            == memory_contract["expected_attention_window"]
            and before_lengths == [longest - 1] * layer_count
            and after_lengths == [longest] * layer_count
            and len(memory.cross_kv) == layer_count
            and not memory_contract["state_contains_cross_kv"]
        )
        return results, numerical, memory_contract

    def _full_validation(self, model, samples):
        def run_path(use_incremental):
            def generate_all_pages():
                sequences = []
                for sample in samples:
                    generation = self._inference(
                        lambda sample=sample: model.generate_token_ids(
                            sample["image"],
                            use_incremental=use_incremental,
                        )
                    )
                    sequences.append(generation.token_ids)
                return tuple(sequences)

            return measure_full_validation(generate_all_pages)

        uncached_timing, uncached_results = run_path(False)
        incremental_timing, incremental_results = run_path(True)
        baseline = uncached_results[0]
        uncached_deterministic = all(result == baseline for result in uncached_results)
        incremental_deterministic = all(
            result == incremental_results[0] for result in incremental_results
        )
        token_sequences_equal = (
            uncached_deterministic
            and incremental_deterministic
            and all(result == baseline for result in incremental_results)
        )
        per_page = []
        for row, (uncached_tokens, incremental_tokens) in enumerate(
            zip(baseline, incremental_results[0], strict=True)
        ):
            per_page.append(
                {
                    "row": row,
                    "equal": uncached_tokens == incremental_tokens,
                    "uncached_token_count": len(uncached_tokens),
                    "incremental_token_count": len(incremental_tokens),
                    "token_sha256": hashlib.sha256(
                        json.dumps(list(uncached_tokens), separators=(",", ":")).encode("ascii")
                    ).hexdigest(),
                }
            )
        return {
            "uncached": uncached_timing,
            "incremental": incremental_timing,
            "incremental_to_uncached_ratio": (
                incremental_timing["median_seconds"]
                / uncached_timing["median_seconds"]
            ),
            "token_sequences_equal": token_sequences_equal,
            "uncached_repetitions_deterministic": uncached_deterministic,
            "incremental_repetitions_deterministic": incremental_deterministic,
            "pages": per_page,
        }

    @staticmethod
    def _environment(gpu_identity, resolved_backend):
        return {
            "gpu": gpu_identity,
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "flash_attention": _package_version("flash-attn"),
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "source": _git_identity(),
            "dtype": "torch.float16",
            "requested_attention_backend": "auto",
            "resolved_attention_backend": resolved_backend,
        }

    def run(self, checkpoint_path, gpu_identity):
        _, payload, checkpoint_identity = self._verify_checkpoint(checkpoint_path)
        snapshot, rows = self._external_inputs()
        model = self._load_model(payload, snapshot)
        del payload
        try:
            samples = self._prepare_samples(model, rows)
            micro, numerical_by_prefix, memory_contract = self._microbenchmark(
                model,
                samples[5],
            )
            full_validation = self._full_validation(model, samples)
            resolved_backend = self._resolved_backend(model)
            return {
                "identity": {
                    "checkpoint": checkpoint_identity,
                    "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
                    "gpu": gpu_identity,
                },
                "environment": self._environment(gpu_identity, resolved_backend),
                "decoder_microbenchmark": micro,
                "full_validation": full_validation,
                "memory_contract": memory_contract,
                "numerical": {
                    "by_prefix": numerical_by_prefix,
                    "micro_max_logits_abs_diff": max(
                        result["max_logits_abs_diff"]
                        for result in numerical_by_prefix.values()
                    ),
                    "all_prefix_argmax_equal": all(
                        result["argmax_equal"]
                        for result in numerical_by_prefix.values()
                    ),
                    "full_validation_token_sequences_equal": full_validation[
                        "token_sequences_equal"
                    ],
                },
            }
        finally:
            model.cpu()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
