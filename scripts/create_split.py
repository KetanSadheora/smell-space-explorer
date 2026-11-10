"""Create and persist the deterministic 80/10/10 evaluation split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smell_space.config import (
    DATASET_PATH,
    LABEL_PREVALENCE_PATH,
    SPLIT_PATH,
    SPLITS_DIR,
    ensure_directories,
)
from smell_space.data import (
    get_task_columns,
    load_curated_dataframe,
    make_stratified_splits,
    save_split,
    sha256_file,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse split creation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=SPLIT_PATH)
    return parser.parse_args()


def main() -> None:
    """Create the split, print prevalence, and write split/prevalence files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ensure_directories()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_curated_dataframe(args.csv)
    task_columns = get_task_columns(dataframe)
    labels = dataframe[task_columns].fillna(0).to_numpy(dtype=np.float32)
    dataset_hash = sha256_file(args.csv)
    splits = make_stratified_splits(labels, seed=args.seed)
    record = save_split(
        args.output,
        splits,
        dataset_hash,
        seed=args.seed,
    )

    print(
        "split_method=iterative-stratification "
        "(MultilabelStratifiedKFold, n_splits=10, 8/1/1 folds)"
    )
    print(f"seed={record['seed']}")
    print(f"dataset_sha256={record['dataset_sha256']}")
    print(f"split_hash={record['split_hash']}")
    print(f"n_molecules={record['n_molecules']}")
    print(
        "sizes="
        + ", ".join(
            f"{name}={record['sizes'][name]}"
            for name in ("train", "valid", "test")
        )
    )
    print("label_prevalence")

    prevalence_rows = []
    for split_name in ("train", "valid", "test"):
        split_indices = splits[split_name]
        split_labels = labels[split_indices]
        positive_rate = float(split_labels.mean()) if split_labels.size else 0.0
        mean_active = float(split_labels.sum(axis=1).mean())
        print(
            f"  {split_name}: positive_label_rate={positive_rate:.6f} "
            f"mean_active_descriptors_per_molecule={mean_active:.6f}"
        )
        counts = split_labels.sum(axis=0)
        for task_index, task in enumerate(task_columns):
            prevalence_rows.append(
                {
                    "split": split_name,
                    "descriptor": task,
                    "positive_count": int(counts[task_index]),
                    "n_molecules": int(len(split_indices)),
                    "prevalence": float(
                        counts[task_index] / len(split_indices)
                    ),
                }
            )

    prevalence = pd.DataFrame(prevalence_rows)
    LABEL_PREVALENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(LABEL_PREVALENCE_PATH, index=False)
    print(f"label_prevalence_csv={LABEL_PREVALENCE_PATH}")

    pivot = prevalence.pivot(
        index="descriptor",
        columns="split",
        values="prevalence",
    )[["train", "valid", "test"]]
    print("\nper-descriptor prevalence (first 10):")
    print(pivot.head(10).to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nper-descriptor prevalence rows written: {len(prevalence)}")


if __name__ == "__main__":
    main()
