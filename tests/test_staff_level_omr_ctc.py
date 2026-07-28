from pathlib import Path

import pytest

from experiments.staff_level_omr.protocol import (
    BundleSample,
    ProtocolError,
    Vocabulary,
    canonical_sha256,
    read_json,
    write_canonical_json,
)
from experiments.staff_level_omr.protocol.ctc import (
    CTCInfeasibleError,
    analyze_ctc_feasibility,
    apply_train_policy,
    minimum_ctc_frames,
)


MANIFEST_SHA = "1" * 64


def _vocabulary() -> Vocabulary:
    document = {
        "schema_version": "staff_omr_vocab_v1",
        "dataset_id": "fixture",
        "vocabulary_scope": "closed_corpus",
        "source_manifest_sha256": MANIFEST_SHA,
        "target_parser": "utf8_unicode_whitespace_v1",
        "token_sort": "utf8_bytes_ascending_v1",
        "unicode_normalization": "none",
        "blank_id": 0,
        "tokens": ["a", "b"],
    }
    return Vocabulary.from_document(
        document,
        manifest_sha256=MANIFEST_SHA,
        dataset_id="fixture",
    )


def _sample(sample_id: str, split: str, tokens: tuple[str, ...]) -> BundleSample:
    return BundleSample(
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        image_path=f"{sample_id}_region.png",
        image_size_bytes=1,
        image_sha256="2" * 64,
        target_path=f"{sample_id}_gt.txt",
        target_sha256="3" * 64,
        split=split,
        target_tokens=tokens,
    )


def _samples() -> tuple[BundleSample, ...]:
    return (
        _sample("train-infeasible", "train", ("a", "a")),
        _sample("train-boundary", "train", ("a", "b")),
        _sample("val-infeasible", "val", ("b", "b")),
        _sample("val-feasible", "val", ("a",)),
        _sample("test-feasible", "test", ("b",)),
    )


def _exclusions(sample_ids, patch_cols=2, manifest_sha=MANIFEST_SHA):
    return {
        "schema_version": "staff_omr_train_exclusions_v1",
        "source_manifest_sha256": manifest_sha,
        "patch_cols": patch_cols,
        "required_frames_algorithm": "ctc_minimum_frames_v1",
        "sample_ids": sorted(sample_ids, key=lambda value: value.encode("utf-8")),
    }


def test_minimum_ctc_frames_counts_adjacent_repeats():
    assert minimum_ctc_frames([1, 2]) == 2
    assert minimum_ctc_frames([1, 1]) == 3
    assert minimum_ctc_frames([1, 1, 2, 2]) == 6


def test_required_frames_equal_to_patch_cols_is_feasible():
    preflight = analyze_ctc_feasibility(
        _samples(), _vocabulary(), patch_cols=2
    )
    records = {record.sample_id: record for record in preflight.records}

    assert records["train-boundary"].required_frames == 2
    assert records["train-boundary"].feasible
    assert not records["train-infeasible"].feasible
    assert not records["val-infeasible"].feasible
    assert preflight.split_summaries["val"]["infeasible_samples"] == 1


def test_fail_policy_writes_reusable_candidate(tmp_path):
    preflight = analyze_ctc_feasibility(
        _samples(), _vocabulary(), patch_cols=2
    )
    candidate_path = tmp_path / "candidate.json"

    with pytest.raises(CTCInfeasibleError) as captured:
        apply_train_policy(
            preflight,
            manifest_sha256=MANIFEST_SHA,
            policy="fail",
            candidate_path=candidate_path,
        )

    candidate = read_json(candidate_path)
    assert candidate == captured.value.candidate_document
    assert candidate == _exclusions(["train-infeasible"])


def test_fail_policy_retains_all_train_when_every_target_is_feasible():
    samples = (
        _sample("train", "train", ("a", "b")),
        _sample("val", "val", ("a",)),
        _sample("test", "test", ("b",)),
    )
    preflight = analyze_ctc_feasibility(
        samples, _vocabulary(), patch_cols=2
    )

    result = apply_train_policy(
        preflight,
        manifest_sha256=MANIFEST_SHA,
        policy="fail",
    )

    assert result.retained_train_sample_ids == ("train",)
    assert result.excluded_train_sample_ids == ()
    assert result.exclusions_sha256 is None
    assert result.exclusion_count == 0


def test_exclude_listed_requires_exact_bound_set(tmp_path):
    preflight = analyze_ctc_feasibility(
        _samples(), _vocabulary(), patch_cols=2
    )
    path = tmp_path / "exclusions.json"
    document = _exclusions(["train-infeasible"])
    write_canonical_json(path, document)

    result = apply_train_policy(
        preflight,
        manifest_sha256=MANIFEST_SHA,
        policy="exclude_listed",
        exclusions_path=path,
    )

    assert result.retained_train_sample_ids == ("train-boundary",)
    assert result.excluded_train_sample_ids == ("train-infeasible",)
    assert result.exclusions_sha256 == canonical_sha256(document)
    assert result.exclusion_count == 1


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (_exclusions([]), "missing"),
        (_exclusions(["train-infeasible", "train-boundary"]), "feasible"),
        (_exclusions(["train-infeasible", "val-infeasible"]), "val/test"),
        (_exclusions(["train-infeasible"], patch_cols=3), "patch_cols"),
        (
            _exclusions(["train-infeasible"], manifest_sha="4" * 64),
            "source_manifest_sha256",
        ),
        (
            _exclusions(["train-infeasible", "train-infeasible"]),
            "duplicate",
        ),
    ],
)
def test_exclude_listed_rejects_mismatched_documents(
    tmp_path, document, message
):
    preflight = analyze_ctc_feasibility(
        _samples(), _vocabulary(), patch_cols=2
    )
    path = tmp_path / "exclusions.json"
    write_canonical_json(path, document)

    with pytest.raises(ProtocolError, match=message):
        apply_train_policy(
            preflight,
            manifest_sha256=MANIFEST_SHA,
            policy="exclude_listed",
            exclusions_path=path,
        )


def test_excluding_every_train_sample_is_rejected(tmp_path):
    samples = (
        _sample("only-train", "train", ("a", "a")),
        _sample("val", "val", ("a",)),
        _sample("test", "test", ("b",)),
    )
    preflight = analyze_ctc_feasibility(
        samples, _vocabulary(), patch_cols=2
    )
    path = tmp_path / "exclusions.json"
    write_canonical_json(path, _exclusions(["only-train"]))

    with pytest.raises(ProtocolError, match="retained_train_samples"):
        apply_train_policy(
            preflight,
            manifest_sha256=MANIFEST_SHA,
            policy="exclude_listed",
            exclusions_path=path,
        )


def test_exclusion_hash_depends_on_content_not_source_path(tmp_path):
    preflight = analyze_ctc_feasibility(
        _samples(), _vocabulary(), patch_cols=2
    )
    document = _exclusions(["train-infeasible"])
    first = tmp_path / "first.json"
    second = tmp_path / "nested" / "second.json"
    write_canonical_json(first, document)
    write_canonical_json(second, document)

    first_result = apply_train_policy(
        preflight,
        manifest_sha256=MANIFEST_SHA,
        policy="exclude_listed",
        exclusions_path=first,
    )
    second_result = apply_train_policy(
        preflight,
        manifest_sha256=MANIFEST_SHA,
        policy="exclude_listed",
        exclusions_path=second,
    )

    assert first_result.exclusions_sha256 == second_result.exclusions_sha256


def test_val_and_test_infeasible_samples_are_never_excluded(tmp_path):
    samples = (
        _sample("train", "train", ("a",)),
        _sample("val", "val", ("a", "a")),
        _sample("test", "test", ("b", "b")),
    )
    preflight = analyze_ctc_feasibility(
        samples, _vocabulary(), patch_cols=2
    )
    result = apply_train_policy(
        preflight,
        manifest_sha256=MANIFEST_SHA,
        policy="fail",
    )

    assert result.retained_train_sample_ids == ("train",)
    assert {
        record.sample_id
        for record in preflight.records
        if not record.feasible
    } == {"val", "test"}
