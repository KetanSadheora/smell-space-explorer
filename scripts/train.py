"""Train the OpenPOM MPNN on the curated odor dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpom.utils.data_utils import get_class_imbalance_ratio

from smell_space.config import MODEL_DIR, ensure_directories
from smell_space.data import get_dataset
from smell_space.model import (
    build_model,
    set_global_seed,
    split_train_valid_dataset,
    train_model,
)

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
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Evaluate every N epochs (default: 5, matching the notebook).",
    )
    parser.add_argument(
        "--train-eval-max-samples",
        type=int,
        default=0,
        help=(
            "Evaluate train AUROC on this many random molecules; 0 keeps the "
            "full training set (default: 0)."
        ),
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction held out for validation; 0 trains and validates on the "
            "full dataset as before (default: 0)."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many unevaluated improvements; 0 disables early "
            "stopping (default: 0)."
        ),
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        help="Restore and save the best validation-AUROC weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed Python/NumPy/PyTorch and the validation split. Omit to keep "
            "the original non-seeded behavior."
        ),
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
    if args.early_stopping_patience > 0 and args.valid_fraction <= 0:
        raise ValueError(
            "--early-stopping-patience requires a non-zero --valid-fraction."
        )
    if args.seed is not None:
        set_global_seed(args.seed)
        logger.info("Using global seed %d", args.seed)

    dataset, task_columns, _ = get_dataset()
    logger.info("Dataset: %d molecules, %d tasks", len(dataset), len(task_columns))

    if args.valid_fraction > 0:
        train_dataset, valid_dataset = split_train_valid_dataset(
            dataset,
            args.valid_fraction,
            seed=args.seed,
        )
        logger.info(
            "Train/validation split: %d / %d molecules",
            len(train_dataset),
            len(valid_dataset),
        )
    else:
        train_dataset = dataset
        valid_dataset = dataset

    imbalance_ratios = get_class_imbalance_ratio(train_dataset)
    model = build_model(len(task_columns), imbalance_ratios, args.model_dir)

    if args.resume:
        checkpoints = model.get_checkpoints()
        if not checkpoints:
            raise FileNotFoundError(
                f"--resume was requested but no checkpoint exists in {args.model_dir}."
            )
        checkpoint = max(checkpoints, key=lambda path: Path(path).stat().st_mtime)
        logger.info("Resuming from %s", checkpoint)
        model.restore(checkpoint)

    logger.info(
        "Training for %d epochs (eval every %d, train-eval cap %d, "
        "early-stop patience %d)",
        args.epochs,
        args.eval_every,
        args.train_eval_max_samples,
        args.early_stopping_patience,
    )
    train_model(
        model,
        train_dataset,
        valid_dataset,
        epochs=args.epochs,
        eval_every=args.eval_every,
        train_eval_max_samples=args.train_eval_max_samples,
        early_stopping_patience=args.early_stopping_patience,
        restore_best_checkpoint=(
            args.save_best_checkpoint or args.early_stopping_patience > 0
        ),
        seed=args.seed,
    )
    model.save_checkpoint()
    logger.info("Saved checkpoint to %s", args.model_dir)


if __name__ == "__main__":
    main()
