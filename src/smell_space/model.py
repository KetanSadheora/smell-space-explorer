"""OpenPOM MPNN construction and training helpers."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible training run."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    n_tasks: int,
    class_imbalance_ratio: Sequence[float],
    model_dir: Path,
) -> Any:
    """Construct the OpenPOM MPNN with the project's fixed architecture.

    The architecture and optimizer settings intentionally match the original
    notebook. Changing them changes the scientific behavior of the project.
    """
    from openpom.models.mpnn_pom import MPNNPOMModel

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return MPNNPOMModel(
        n_tasks=n_tasks,
        batch_size=128,
        learning_rate=0.001,
        class_imbalance_ratio=class_imbalance_ratio,
        loss_aggr_type="sum",
        node_out_feats=64,
        edge_hidden_feats=128,
        num_step_message_passing=5,
        mpnn_residual=True,
        message_aggregator_type="sum",
        mode="classification",
        number_of_molecules=128,
        model_dir=str(model_dir),
    )


def split_train_valid_dataset(
    dataset: Any,
    valid_fraction: float,
    *,
    seed: int | None = None,
) -> tuple[Any, Any]:
    """Split a multi-label DeepChem dataset with iterative stratification.

    Args:
        dataset: A DeepChem dataset with graph ``X`` and multi-label ``y``.
        valid_fraction: Fraction of molecules reserved for validation.
        seed: Optional seed passed to the iterative stratifier.

    Returns:
        A ``(train_dataset, valid_dataset)`` tuple of NumpyDatasets.
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be strictly between 0 and 1.")

    import deepchem as dc
    from skmultilearn.model_selection import IterativeStratification

    labels = np.asarray(dataset.y)
    n_samples = len(labels)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)
    shuffled_labels = labels[permutation]

    # skmultilearn 0.2.0 ignores random_state because its KFold is
    # shuffle=False, and one fallback path draws from the global NumPy RNG.
    # Pre-shuffling plus a scoped global seed makes the split reproducible
    # without leaking a changed RNG state into the training loop.
    saved_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        stratifier = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[
                valid_fraction,
                1.0 - valid_fraction,
            ],
        )
        train_local, valid_local = next(
            stratifier.split(
                np.zeros((n_samples, 1)),
                shuffled_labels,
            )
        )
    finally:
        if seed is not None:
            np.random.set_state(saved_state)

    train_indices = permutation[train_local]
    valid_indices = permutation[valid_local]

    def subset(indices: np.ndarray) -> Any:
        return dc.data.NumpyDataset(
            X=dataset.X[indices],
            y=dataset.y[indices],
            w=dataset.w[indices],
            ids=[dataset.ids[index] for index in indices],
        )

    return subset(train_indices), subset(valid_indices)


def _evaluation_subset(
    dataset: Any,
    max_samples: int,
    *,
    seed: int | None,
) -> Any:
    """Return the full dataset or a deterministic random evaluation subset."""
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset

    import deepchem as dc

    rng = np.random.default_rng(0 if seed is None else seed)
    indices = np.sort(
        rng.choice(len(dataset), size=max_samples, replace=False)
    )
    logger.info(
        "Using %d of %d training molecules for train AUROC evaluation",
        max_samples,
        len(dataset),
    )
    return dc.data.NumpyDataset(
        X=dataset.X[indices],
        y=dataset.y[indices],
        w=dataset.w[indices],
        ids=[dataset.ids[index] for index in indices],
    )


def train_model(
    model: Any,
    train_dataset: Any,
    valid_dataset: Any,
    epochs: int = 30,
    *,
    eval_every: int = 5,
    train_eval_max_samples: int = 0,
    early_stopping_patience: int = 0,
    restore_best_checkpoint: bool = False,
    seed: int | None = None,
) -> Any:
    """Train an OpenPOM model and optionally apply early stopping.

    Defaults preserve the original schedule: one epoch per ``fit`` call,
    evaluation on epochs 1, every 5 epochs, and the final epoch, using the full
    training set for train AUROC. New behavior is opt-in.

    Args:
        model: An OpenPOM ``MPNNPOMModel``.
        train_dataset: Dataset used for gradient updates.
        valid_dataset: Dataset used for validation AUROC and early stopping.
        epochs: Maximum number of training epochs.
        eval_every: Evaluate every N epochs (plus epoch 1 and the last).
        train_eval_max_samples: If positive, evaluate train AUROC on a random
            subset of this many molecules.
        early_stopping_patience: Number of evaluations without validation AUROC
            improvement before stopping. Zero disables early stopping.
        restore_best_checkpoint: Restore the best validation weights in memory
            when training ends. The caller remains responsible for saving.
        seed: Optional seed for the train-evaluation subsample.

    Returns:
        The trained (or best-restored) model.
    """
    import deepchem as dc

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1.")
    if train_eval_max_samples < 0:
        raise ValueError("train_eval_max_samples cannot be negative.")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")

    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score,
        np.nanmean,
        mode="classification",
    )
    train_eval_dataset = _evaluation_subset(
        train_dataset,
        train_eval_max_samples,
        seed=seed,
    )

    best_valid_auroc = -np.inf
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    evaluations_without_improvement = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        loss = model.fit(train_dataset, nb_epoch=1)
        epoch_seconds = time.perf_counter() - epoch_start
        logger.info(
            "Epoch %02d/%02d complete in %.1f s",
            epoch,
            epochs,
            epoch_seconds,
        )
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            train_metrics = model.evaluate(
                train_eval_dataset,
                [metric],
                n_classes=2,
            )
            valid_metrics = model.evaluate(valid_dataset, [metric], n_classes=2)
            train_auroc = next(iter(train_metrics.values()), 0.0)
            valid_auroc = next(iter(valid_metrics.values()), 0.0)
            logger.info(
                "Epoch %02d/%02d | loss %.4f | train AUROC %.4f | "
                "valid AUROC %.4f",
                epoch,
                epochs,
                loss,
                train_auroc,
                valid_auroc,
            )

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                best_epoch = epoch
                evaluations_without_improvement = 0
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.model.state_dict().items()
                }
                logger.info(
                    "New best validation AUROC %.4f at epoch %d",
                    best_valid_auroc,
                    epoch,
                )
            else:
                evaluations_without_improvement += 1
                if (
                    early_stopping_patience > 0
                    and evaluations_without_improvement
                    >= early_stopping_patience
                ):
                    logger.info(
                        "Early stopping after %d evaluations without "
                        "improvement (best epoch %d, AUROC %.4f)",
                        evaluations_without_improvement,
                        best_epoch,
                        best_valid_auroc,
                    )
                    break

    if restore_best_checkpoint and best_state is not None:
        model.model.load_state_dict(best_state)
        logger.info(
            "Restored best validation weights from epoch %d (AUROC %.4f)",
            best_epoch,
            best_valid_auroc,
        )
    return model
