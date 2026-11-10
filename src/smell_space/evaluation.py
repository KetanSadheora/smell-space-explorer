"""AUROC evaluation helpers for multi-label odor predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, float | None]]:
    """Compute per-task, macro, and micro AUROC.

    Tasks with fewer than two classes in ``y_true`` are skipped. Macro AUROC is
    the unweighted mean over evaluable tasks, which is also the mean per-task
    AUROC. Micro AUROC pools the predictions of evaluable tasks.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    if labels.shape != scores.shape:
        raise ValueError(
            f"Labels and predictions must match: {labels.shape} vs {scores.shape}."
        )
    if labels.ndim != 2 or labels.shape[1] != len(task_names):
        raise ValueError("Labels must be (n_molecules, n_tasks).")

    per_task: dict[str, float | None] = {}
    skipped: list[str] = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []

    for index, task in enumerate(task_names):
        task_labels = labels[:, index]
        if np.unique(task_labels).size < 2:
            per_task[task] = None
            skipped.append(task)
            continue
        score = float(roc_auc_score(task_labels, scores[:, index]))
        per_task[task] = score
        pooled_labels.append(task_labels)
        pooled_scores.append(scores[:, index])

    evaluable = [score for score in per_task.values() if score is not None]
    if not evaluable:
        raise ValueError("No task has both classes in this split.")

    summary = {
        "n_tasks": len(task_names),
        "n_tasks_evaluated": len(evaluable),
        "n_tasks_skipped": len(skipped),
        "skipped_tasks": skipped,
        "macro_auroc": float(np.mean(evaluable)),
        "mean_task_auroc": float(np.mean(evaluable)),
        "micro_auroc": float(
            roc_auc_score(
                np.concatenate(pooled_labels),
                np.concatenate(pooled_scores),
            )
        ),
    }
    return summary, per_task
