"""Train the OpenPOM MPNN on the curated odor dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from openpom.utils.data_utils import get_class_imbalance_ratio

from smell_space.config import DATASET_PATH, MODEL_DIR, ensure_directories
from smell_space.data import (
    get_dataset,
    load_split,
    sha256_file,
    subset_dataset,
)
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
        "--split-file",
        type=Path,
        default=None,
        help=(
            "Saved split JSON containing train/valid/test indices. Training "
            "uses only train and valid; test is never evaluated here."
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
    if args.split_file is not None and args.valid_fraction > 0:
        raise ValueError("--split-file and --valid-fraction are mutually exclusive.")
    if (
        args.early_stopping_patience > 0
        and args.valid_fraction <= 0
        and args.split_file is None
    ):
        raise ValueError(
            "--early-stopping-patience requires --valid-fraction or --split-file."
        )
    if args.seed is not None:
        set_global_seed(args.seed)
        logger.info("Using global seed %d", args.seed)

    dataset, task_columns, _ = get_dataset()
    logger.info("Dataset: %d molecules, %d tasks", len(dataset), len(task_columns))

    split_record = None
    if args.split_file is not None:
        split_record = load_split(args.split_file)
        dataset_hash = sha256_file(DATASET_PATH)
        if split_record["dataset_sha256"] != dataset_hash:
            raise ValueError(
                "Split file dataset_sha256 does not match the current CSV."
            )
        train_dataset = subset_dataset(
            dataset,
            split_record["indices"]["train"],
        )
        valid_dataset = subset_dataset(
            dataset,
            split_record["indices"]["valid"],
        )
        logger.info(
            "Using split %s (hash %s): train=%d valid=%d test=%d (test unused)",
            args.split_file,
            split_record["split_hash"],
            len(train_dataset),
            len(valid_dataset),
            len(split_record["indices"]["test"]),
        )
    elif args.valid_fraction > 0:
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
    steps_per_epoch = math.ceil(len(train_dataset) / model.batch_size)
    global_step = int(model.get_global_step())
    summary = {
        "model_dir": str(args.model_dir),
        "epochs_requested": args.epochs,
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "epochs_completed": global_step / steps_per_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "split_hash": (
            split_record["split_hash"] if split_record is not None else None
        ),
    }
    (args.model_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Saved checkpoint and training summary to %s", args.model_dir)


if __name__ == "__main__":
    main()
