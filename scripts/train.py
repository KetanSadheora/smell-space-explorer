"""Train the OpenPOM MPNN on the curated odor dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpom.utils.data_utils import get_class_imbalance_ratio

from smell_space.config import MODEL_DIR, ensure_directories
from smell_space.data import get_dataset
from smell_space.model import build_model, train_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25, matching the notebook).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the model directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"Directory for checkpoints (default: {MODEL_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    """Run training and leave the latest checkpoint on disk."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ensure_directories()

    dataset, task_columns, _ = get_dataset()
    logger.info("Dataset: %d molecules, %d tasks", len(dataset), len(task_columns))

    imbalance_ratios = get_class_imbalance_ratio(dataset)
    model = build_model(len(task_columns), imbalance_ratios, args.model_dir)

    if args.resume:
        checkpoints = model.get_checkpoints()
        if not checkpoints:
            raise FileNotFoundError(
                f"--resume was requested but no checkpoint exists in {args.model_dir}."
            )
        logger.info("Resuming from %s", checkpoints[0])
        model.restore(checkpoints[0])

    logger.info("Training for %d epochs on CPU-compatible device", args.epochs)
    train_model(model, dataset, dataset, epochs=args.epochs)
    model.save_checkpoint()
    logger.info("Saved checkpoint to %s", args.model_dir)


if __name__ == "__main__":
    main()
