import pytest
import torch
from torch import nn

from experiments.staff_level_omr.protocol.batching import CTCBatch, ctc_collate
from experiments.staff_level_omr.protocol.ctc import minimum_ctc_frames
from experiments.staff_level_omr.protocol.errors import ProtocolError
from experiments.staff_level_omr.protocol.evaluation import (
    evaluate_split,
    train_epoch,
)


class TrainableLogits(nn.Module):
    def __init__(self, time_steps: int, classes: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(time_steps, classes))

    def forward(self, images):
        batch = images.shape[0]
        return self.logits.unsqueeze(0).expand(batch, -1, -1).log_softmax(-1)


class FixedPredictions(nn.Module):
    def forward(self, images):
        ids = torch.tensor(
            [
                [1, 0, 2],
                [3, 3, 0],
            ],
            device=images.device,
        )
        logits = torch.full((2, 3, 4), -20.0, device=images.device)
        logits.scatter_(2, ids.unsqueeze(-1), 0.0)
        return logits.log_softmax(-1)


def _batch(targets, feasibility) -> CTCBatch:
    return ctc_collate(
        [
            (
                torch.zeros((3, 8, 8)),
                torch.tensor(target, dtype=torch.long),
                len(target),
                f"sample-{index}",
                feasible,
            )
            for index, (target, feasible) in enumerate(
                zip(targets, feasibility, strict=True)
            )
        ]
    )


def test_ctc_boundary_sample_has_finite_loss_and_nonzero_head_gradient():
    target = [1, 1, 2]
    assert minimum_ctc_frames(target) == 4
    model = TrainableLogits(time_steps=4, classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = train_epoch(
        model,
        [_batch([target], [True])],
        optimizer,
        device=torch.device("cpu"),
    )

    assert result.samples == 1
    assert result.batches == 1
    assert result.global_steps == 1
    assert result.loss > 0
    assert model.logits.grad is not None
    assert torch.count_nonzero(model.logits.grad)


def test_train_epoch_exposes_infinite_ctc_instead_of_zeroing_it():
    target = [1, 1, 1]
    assert minimum_ctc_frames(target) == 5
    model = TrainableLogits(time_steps=3, classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ProtocolError, match="non-finite"):
        train_epoch(
            model,
            [_batch([target], [False])],
            optimizer,
            device=torch.device("cpu"),
        )


def test_train_epoch_checks_gradients_with_one_aggregated_norm(monkeypatch):
    calls = []

    def record_norm(tensors, *, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        calls.append(
            {
                "count": len(list(tensors)),
                "error_if_nonfinite": error_if_nonfinite,
                "foreach": foreach,
            }
        )
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", record_norm)
    model = TrainableLogits(time_steps=3, classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_epoch(
        model,
        [_batch([[1, 2]], [True])],
        optimizer,
        device=torch.device("cpu"),
    )

    assert calls == [
        {
            "count": 1,
            "error_if_nonfinite": True,
            "foreach": None,
        }
    ]


def test_aggregated_gradient_guard_rejects_non_finite_values():
    model = TrainableLogits(time_steps=3, classes=3)
    model.logits.register_hook(lambda gradient: gradient.fill_(float("nan")))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ProtocolError, match="non-finite gradients"):
        train_epoch(
            model,
            [_batch([[1, 2]], [True])],
            optimizer,
            device=torch.device("cpu"),
        )


def test_evaluation_reports_all_and_feasible_metrics_plus_capacity():
    result = evaluate_split(
        FixedPredictions(),
        [_batch([[1, 2], [3, 3, 3, 3]], [True, False])],
        split="val",
        device=torch.device("cpu"),
    )
    report = result.to_dict()

    assert report["val_CER_all"] == pytest.approx(0.5)
    assert report["val_CER_feasible"] == 0.0
    assert report["val_CTC_loss_feasible"] is not None
    assert report["val_feasible_samples"] == 1
    assert report["val_infeasible_samples"] == 1
    assert report["val_infeasible_ratio"] == 0.5
    assert report["val_capacity"]["all"]["target_length"] == {
        "count": 2,
        "min": 2,
        "max": 4,
        "mean": 3.0,
    }
    assert report["val_capacity"]["infeasible"]["required_frames"]["min"] == 7
    assert result.sample_ids == ("sample-0", "sample-1")


def test_test_split_does_not_compute_ctc_loss():
    result = evaluate_split(
        FixedPredictions(),
        [_batch([[1, 2], [3, 3, 3, 3]], [True, False])],
        split="test",
        device=torch.device("cpu"),
    )

    assert "test_CTC_loss_feasible" not in result.to_dict()
    assert result.ctc_losses == (None, None)


def test_validation_with_no_feasible_samples_uses_json_null_metrics():
    model = TrainableLogits(time_steps=2, classes=3)
    result = evaluate_split(
        model,
        [_batch([[1, 1, 1]], [False])],
        split="val",
        device=torch.device("cpu"),
    )
    report = result.to_dict()

    assert report["val_CER_feasible"] is None
    assert report["val_CTC_loss_feasible"] is None
    assert report["val_feasible_samples"] == 0
