"""Evaluate the split-trained model once on train, validation, and test."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from smell_space.config import (
    DATASET_PATH,
    METRICS_PATH,
    METRICS_PER_TASK_PATH,
    SPLIT_PATH,
    ensure_directories,
)
from smell_space.data import (
    get_dataset,
    load_split,
    sha256_file,
    subset_dataset,
)
from smell_space.evaluation import compute_auroc_metrics
from smell_space.model import build_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=SPLIT_PATH,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train", "valid", "test"),
        help="Splits to evaluate. The production run uses all three.",
    )
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--per-task-path",
        type=Path,
        default=METRICS_PER_TASK_PATH,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing metrics.json.",
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    """Return the current Git commit hash if available."""
    git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    try:
        return subprocess.check_output(
            [git, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _latest_checkpoint(model: object) -> Path:
    """Return the most recently written DeepChem checkpoint."""
    checkpoints = [Path(path) for path in model.get_checkpoints()]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in the evaluation model dir.")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    """Run one evaluation pass and write metrics JSON plus per-task CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ensure_directories()
    if args.metrics_path.exists() and not args.force:
        raise FileExistsError(
            f"{args.metrics_path} already exists. Refusing to evaluate twice "
            "without --force."
        )

    split_record = load_split(args.split_file)
    dataset_hash = sha256_file(DATASET_PATH)
    if split_record["dataset_sha256"] != dataset_hash:
        raise ValueError("Split dataset hash does not match the current CSV.")

    dataset, task_columns, _ = get_dataset()
    train_dataset = subset_dataset(dataset, split_record["indices"]["train"])
    split_datasets = {
        name: subset_dataset(dataset, split_record["indices"][name])
        for name in ("train", "valid", "test")
    }

    # The loss weighting is irrelevant at inference, but construct the model
    # exactly as the training script did before restoring weights.
    from openpom.utils.data_utils import get_class_imbalance_ratio

    model = build_model(
        len(task_columns),
        get_class_imbalance_ratio(train_dataset),
        args.model_dir,
    )
    checkpoint = _latest_checkpoint(model)
    logger.info("Restoring %s", checkpoint)
    model.restore(checkpoint=str(checkpoint))
    model.model.eval()

    summary_path = args.model_dir / "training_summary.json"
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if training_summary.get("split_hash") != split_record["split_hash"]:
        raise ValueError(
            "Training summary split hash does not match the split file."
        )

    metrics: dict[str, object] = {
        "dataset_sha256": dataset_hash,
        "split_file": str(args.split_file),
        "split_hash": split_record["split_hash"],
        "seed": split_record["seed"],
        "model_dir": str(args.model_dir),
        "checkpoint": str(checkpoint),
        "epochs_trained": training_summary["epochs_completed"],
        "global_step": training_summary["global_step"],
        "git_commit": _git_commit(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    per_task_rows: list[dict[str, object]] = []

    for split_name in args.splits:
        split_dataset = split_datasets[split_name]
        logger.info(
            "Predicting %s split (%d molecules) once",
            split_name,
            len(split_dataset),
        )
        predictions = np.asarray(model.predict(split_dataset))
        if predictions.shape != split_dataset.y.shape:
            raise RuntimeError(
                f"Prediction shape {predictions.shape} does not match labels "
                f"{split_dataset.y.shape} for {split_name}."
            )

        summary, per_task = compute_auroc_metrics(
            split_dataset.y,
            predictions,
            task_columns,
        )
        summary["n_molecules"] = len(split_dataset)
        metrics["splits"][split_name] = summary
        logger.info(
            "%s macro AUROC %.6f | micro AUROC %.6f | skipped %d tasks",
            split_name,
            summary["macro_auroc"],
            summary["micro_auroc"],
            summary["n_tasks_skipped"],
        )

        labels = np.asarray(split_dataset.y)
        for task_index, task in enumerate(task_columns):
            task_labels = labels[:, task_index]
            per_task_rows.append(
                {
                    "split": split_name,
                    "descriptor": task,
                    "auroc": per_task[task],
                    "positive_count": int(task_labels.sum()),
                    "n_molecules": len(split_dataset),
                }
            )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(per_task_rows).to_csv(args.per_task_path, index=False)
    logger.info("Wrote %s", args.metrics_path)
    logger.info("Wrote %s", args.per_task_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
