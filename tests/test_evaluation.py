"""Tests for AUROC aggregation without model training."""

from __future__ import annotations

import numpy as np
import pytest

from smell_space.evaluation import compute_auroc_metrics


def test_auroc_skips_constant_tasks_and_aggregates() -> None:
    labels = np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=float,
    )
    scores = np.array(
        [
            [0.9, 0.2, 0.5],
            [0.8, 0.7, 0.5],
            [0.2, 0.8, 0.5],
            [0.1, 0.3, 0.5],
        ]
    )
    summary, per_task = compute_auroc_metrics(
        labels,
        scores,
        ["task_a", "task_b", "constant"],
    )

    assert summary["n_tasks_evaluated"] == 2
    assert summary["n_tasks_skipped"] == 1
    assert summary["skipped_tasks"] == ["constant"]
    assert summary["macro_auroc"] == pytest.approx(1.0)
    assert summary["micro_auroc"] == pytest.approx(1.0)
    assert per_task == {"task_a": 1.0, "task_b": 1.0, "constant": None}


def test_auroc_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_auroc_metrics(
            np.zeros((4, 2)),
            np.zeros((3, 2)),
            ["a", "b"],
        )
