"""Public adapter for deterministic staff-level OMR dataset preparation."""

from __future__ import annotations

from pathlib import Path

from .protocol.data_bundle import prepare_dataset_bundle


def prepare_data(
    data_path: str,
    dataset_id: str,
    group_regex: str,
    out: str,
    split_ratios: object = None,
    *split_ratio_tail: object,
    seed: int = 7,
) -> dict[str, object]:
    """Generate one atomic manifest/vocabulary/image-index bundle.

    Args:
        data_path: Directory recursively containing paired staff files.
        dataset_id: Stable identifier written into every identity document.
        group_regex: Full-match regex with a named ``group_id`` capture.
        out: New output directory; existing paths are never overwritten.
        split_ratios: First ratio, or a three-value sequence for Python calls.
        split_ratio_tail: Remaining two ratios in the public CLI form.
        seed: Non-negative deterministic group-split seed.
    """
    if split_ratios is None:
        ratios = ("0.8", "0.1", "0.1")
    elif split_ratio_tail:
        ratios = (split_ratios, *split_ratio_tail)
    elif isinstance(split_ratios, (tuple, list)):
        ratios = tuple(split_ratios)
    else:
        ratios = (split_ratios,)

    report = prepare_dataset_bundle(
        data_path=Path(data_path),
        dataset_id=dataset_id,
        group_regex=group_regex,
        split_ratios=ratios,
        seed=seed,
        out=Path(out),
    )
    return report.to_dict()
