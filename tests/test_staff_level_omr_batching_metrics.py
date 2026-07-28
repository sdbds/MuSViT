import math

import pytest
import torch

from experiments.staff_level_omr.protocol import ProtocolError
from experiments.staff_level_omr.protocol.batching import (
    ctc_collate,
    input_lengths_for,
    split_concatenated_targets,
)
from experiments.staff_level_omr.protocol.metrics import (
    layered_metrics,
    micro_cer,
)


def _item(sample_id, target, feasible=True):
    image = torch.full((3, 4, 5), float(len(target)))
    target_tensor = torch.tensor(target, dtype=torch.long)
    return image, target_tensor, len(target), sample_id, feasible


def test_ctc_collate_concatenates_variable_targets_without_global_padding():
    batch = ctc_collate(
        [
            _item("a", [1]),
            _item("b", [2, 2, 3], feasible=False),
            _item("c", [4, 5]),
        ]
    )

    assert batch.images.shape == (3, 3, 4, 5)
    assert batch.targets.tolist() == [1, 2, 2, 3, 4, 5]
    assert batch.target_lengths.tolist() == [1, 3, 2]
    assert batch.sample_ids == ("a", "b", "c")
    assert batch.ctc_feasible.tolist() == [True, False, True]
    assert split_concatenated_targets(
        batch.targets, batch.target_lengths
    ) == [[1], [2, 2, 3], [4, 5]]


def test_target_reconstruction_uses_lengths_not_blank_sentinel():
    targets = torch.tensor([1, 0, 2], dtype=torch.long)

    reconstructed = split_concatenated_targets(targets, [2, 1])

    assert reconstructed == [[1, 0], [2]]


def test_last_small_batch_gets_runtime_input_lengths():
    final_batch = ctc_collate([_item("last", [1, 2])])
    log_probs = torch.randn(7, final_batch.images.shape[0], 4)

    input_lengths = input_lengths_for(log_probs)

    assert input_lengths.dtype == torch.long
    assert input_lengths.device.type == "cpu"
    assert input_lengths.tolist() == [7]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "empty"),
        (
            [
                (
                    torch.zeros(3, 4, 5),
                    torch.tensor([[1, 2]]),
                    2,
                    "bad",
                    True,
                )
            ],
            "1-D",
        ),
        (
            [
                (
                    torch.zeros(3, 4, 5),
                    torch.tensor([1, 2]),
                    1,
                    "bad",
                    True,
                )
            ],
            "target_length",
        ),
    ],
)
def test_ctc_collate_rejects_invalid_batches(items, message):
    with pytest.raises(ProtocolError, match=message):
        ctc_collate(items)


def test_split_targets_rejects_inconsistent_total_length():
    with pytest.raises(ProtocolError, match="sum"):
        split_concatenated_targets(torch.tensor([1, 2]), [1, 2])


def test_micro_cer_uses_total_edits_over_total_target_length():
    targets = [[1, 2], [1, 1], [2]]
    predictions = [[1, 2], [1], [1]]

    assert micro_cer(predictions, targets) == pytest.approx(2 / 5)


def test_layered_metrics_keep_infeasible_samples_in_all_cer():
    targets = [[1, 2], [1, 1], [2]]
    predictions = [[1, 2], [1], [1]]
    feasible = [True, False, True]
    losses = [4.0, None, 2.0]

    metrics = layered_metrics(
        predictions,
        targets,
        feasible,
        feasible_ctc_losses=losses,
    )

    assert metrics.cer_all == pytest.approx(2 / 5)
    assert metrics.cer_feasible == pytest.approx(1 / 3)
    assert metrics.ctc_loss_feasible == pytest.approx(2.0)
    assert metrics.feasible_samples == 2
    assert metrics.infeasible_samples == 1
    assert metrics.infeasible_ratio == pytest.approx(1 / 3)
    assert metrics.to_dict("val") == {
        "val_CER_all": pytest.approx(2 / 5),
        "val_CER_feasible": pytest.approx(1 / 3),
        "val_CTC_loss_feasible": pytest.approx(2.0),
        "val_feasible_samples": 2,
        "val_infeasible_samples": 1,
        "val_infeasible_ratio": pytest.approx(1 / 3),
    }


def test_layered_metrics_return_null_for_empty_feasible_subset():
    metrics = layered_metrics(
        predictions=[[1], []],
        targets=[[1, 1], [2]],
        feasible=[False, False],
        feasible_ctc_losses=[None, None],
    )

    assert metrics.cer_all == pytest.approx(2 / 3)
    assert metrics.cer_feasible is None
    assert metrics.ctc_loss_feasible is None
    assert metrics.feasible_samples == 0


@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_layered_metrics_reject_non_finite_feasible_loss(loss):
    with pytest.raises(ProtocolError, match="finite"):
        layered_metrics(
            predictions=[[1]],
            targets=[[1]],
            feasible=[True],
            feasible_ctc_losses=[loss],
        )


def test_micro_cer_rejects_empty_targets_and_alignment_errors():
    with pytest.raises(ProtocolError, match="non-empty"):
        micro_cer([[]], [[]])
    with pytest.raises(ProtocolError, match="same number"):
        micro_cer([[1]], [[1], [2]])
    with pytest.raises(ProtocolError, match="same number"):
        layered_metrics([[1]], [[1]], [True, False])


def test_layered_metric_result_contains_only_finite_numbers_or_null():
    metrics = layered_metrics([[1]], [[1]], [True])
    for value in metrics.to_dict("test").values():
        assert value is None or not isinstance(value, float) or math.isfinite(value)
