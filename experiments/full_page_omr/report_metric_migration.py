"""One-time canonical-v2 versus legacy metric migration report."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_generation import (
    DATASET_ID,
    DATASET_REVISION,
    BenchmarkUnavailable,
    _LockedBenchmarkRuntime,
    locked_inputs,
)
from .eval.eval_functions import (
    canonical_text,
    canonicalize_prediction_ids,
    canonicalize_target_ids,
    compute_canonical_metrics,
    compute_poliphony_metrics_legacy,
)


MIGRATION_VERSION = "canonical_v2_vs_legacy_polish_val_v1"
METRIC_NAMES = ("CER", "SER", "LER")
LOCKED_ROW_COUNT = 10


def _metric_dict(values):
    return {
        metric_name: float(value)
        for metric_name, value in zip(METRIC_NAMES, values, strict=True)
    }


def _metric_delta(left, right):
    return {
        metric_name: left[metric_name] - right[metric_name]
        for metric_name in METRIC_NAMES
    }


def _legacy_text(stream):
    return canonical_text(stream)


def _legacy_target_text(stream):
    # The historical validation code sliced the trailing PAD but retained EOS.
    return canonical_text(stream) + "<eos>"


def _prediction_digest(token_ids):
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_metric_migration_report(*, prediction_ids, target_ids, i2w, maxlen,
                                  checkpoint_identity, dataset_identity,
                                  generation_seconds, environment):
    predictions = tuple(prediction_ids)
    targets = tuple(target_ids)
    if len(predictions) != LOCKED_ROW_COUNT or len(targets) != LOCKED_ROW_COUNT:
        raise ValueError("metric migration requires exactly 10 prediction/target pages")
    if generation_seconds < 0:
        raise ValueError("generation_seconds must be non-negative")

    canonical_predictions = tuple(
        canonicalize_prediction_ids(token_ids, i2w, maxlen=maxlen)
        for token_ids in predictions
    )
    canonical_targets = tuple(
        canonicalize_target_ids(token_ids, i2w)
        for token_ids in targets
    )
    legacy_predictions = tuple(_legacy_text(stream) for stream in canonical_predictions)
    legacy_targets = tuple(_legacy_target_text(stream) for stream in canonical_targets)

    v2 = _metric_dict(compute_canonical_metrics(canonical_predictions, canonical_targets))
    legacy = _metric_dict(
        compute_poliphony_metrics_legacy(legacy_predictions, legacy_targets)
    )

    pages = []
    for row, (raw_prediction, prediction, target, legacy_prediction, legacy_target) in enumerate(
        zip(
            predictions,
            canonical_predictions,
            canonical_targets,
            legacy_predictions,
            legacy_targets,
            strict=True,
        )
    ):
        page_v2 = _metric_dict(compute_canonical_metrics((prediction,), (target,)))
        page_legacy = _metric_dict(
            compute_poliphony_metrics_legacy((legacy_prediction,), (legacy_target,))
        )
        pages.append(
            {
                "row": row,
                "v2": page_v2,
                "legacy": page_legacy,
                "delta_v2_minus_legacy": _metric_delta(page_v2, page_legacy),
                "terminated_by_eos": prediction.terminated_by_eos,
                "truncated": prediction.truncated,
                "prediction_token_count": len(raw_prediction),
                "prediction_token_sha256": _prediction_digest(raw_prediction),
                "legacy_target_suffix": "<eos>",
            }
        )

    return {
        "metric_migration_version": MIGRATION_VERSION,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "identity": {
            "checkpoint": checkpoint_identity,
            "dataset": dataset_identity,
        },
        "rows": list(range(LOCKED_ROW_COUNT)),
        "generation": {
            "path": "uncached",
            "decoding": "greedy",
            "seconds": float(generation_seconds),
        },
        "environment": environment,
        "aggregate": {
            "v2": v2,
            "legacy": legacy,
            "delta_v2_minus_legacy": _metric_delta(v2, legacy),
        },
        "pages": pages,
    }


def write_report(output_path, report) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output.resolve()


def main(checkpoint_path, output_path):
    runtime = _LockedBenchmarkRuntime()
    try:
        gpu_identity = runtime.ensure_reference_gpu()
        _, payload, checkpoint_identity = runtime._verify_checkpoint(checkpoint_path)
        snapshot, rows = runtime._external_inputs()
        model = runtime._load_model(payload, snapshot)
        del payload
    except BenchmarkUnavailable as exc:
        report = {
            "metric_migration_version": MIGRATION_VERSION,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "not-run",
            "reason": str(exc),
            "locked_inputs": locked_inputs(),
        }
        write_report(output_path, report)
        return report

    try:
        import torch

        samples = runtime._prepare_samples(model, rows)
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = []
        for sample in samples:
            generation = runtime._inference(
                lambda sample=sample: model.generate_token_ids(
                    sample["image"],
                    use_incremental=False,
                )
            )
            predictions.append(generation.token_ids)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started
        resolved_backend = runtime._resolved_backend(model)
        environment = runtime._environment(gpu_identity, resolved_backend)
        report = build_metric_migration_report(
            prediction_ids=predictions,
            target_ids=[sample["target_ids"] for sample in samples],
            i2w=model.i2w,
            maxlen=model.maxlen,
            checkpoint_identity=checkpoint_identity,
            dataset_identity={"id": DATASET_ID, "revision": DATASET_REVISION},
            generation_seconds=generation_seconds,
            environment=environment,
        )
        report["locked_inputs"] = locked_inputs()
        write_report(output_path, report)
        return report
    finally:
        model.cpu()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
