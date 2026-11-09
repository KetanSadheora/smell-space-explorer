"""OpenPOM MPNN construction and training helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


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


def train_model(
    model: Any,
    train_dataset: Any,
    valid_dataset: Any,
    epochs: int = 30,
) -> Any:
    """Train an OpenPOM model and periodically log AUROC.

    This preserves the original training schedule: one epoch per ``fit`` call
    and evaluation on the first, fifth, and final epochs.
    """
    import deepchem as dc

    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score,
        np.nanmean,
        mode="classification",
    )
    for epoch in range(1, epochs + 1):
        loss = model.fit(train_dataset, nb_epoch=1)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            train_metrics = model.evaluate(train_dataset, [metric], n_classes=2)
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
    return model
