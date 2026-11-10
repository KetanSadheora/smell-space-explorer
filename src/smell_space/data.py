"""Dataset loading, column normalization, and graph featurization."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATASET_PATH, DATASET_URL, DISK_DATASET_DIR

logger = logging.getLogger(__name__)

CACHE_MANIFEST_FILENAME = "cache_manifest.json"
FEATURIZATION_CACHE_VERSION = 1

# Columns that describe a molecule or its provenance rather than an odor task.
NON_TASK_COLUMNS = frozenset(
    {
        "nonStereoSMILES",
        "SMILES",
        "smiles",
        "canonical_smiles",
        "cas",
        "CASNumber",
        "CID",
        "cid",
        "name",
        "Name",
        "odor",
        "descriptors",
        "split",
    }
)

SMILES_COLUMN_CANDIDATES = (
    "nonStereoSMILES",
    "SMILES",
    "smiles",
    "canonical_smiles",
)


def find_smiles_column(df: pd.DataFrame) -> str:
    """Return the canonical SMILES column present in ``df``.

    Raises:
        ValueError: If no supported SMILES column is present.
    """
    for column in SMILES_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "No SMILES column found. Expected one of "
        f"{SMILES_COLUMN_CANDIDATES}; got {list(df.columns)!r}."
    )


def get_task_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric odor-descriptor task columns in dataset order."""
    return [
        column
        for column in df.columns
        if column not in NON_TASK_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def download_dataset(destination: Path = DATASET_PATH) -> Path:
    """Download the curated OpenPOM dataset if it is not present locally."""
    destination = Path(destination)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    logger.info("Downloading curated dataset to %s", destination)
    try:
        urllib.request.urlretrieve(DATASET_URL, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_curated_dataframe(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the curated odor dataset, downloading it when necessary."""
    path = download_dataset(Path(csv_path))
    logger.info("Loading curated dataset from %s", path)
    return pd.read_csv(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(cache_dir: Path) -> Path:
    """Return the manifest path for a featurization cache directory."""
    return Path(cache_dir) / CACHE_MANIFEST_FILENAME


def _read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    """Read a cache manifest, returning ``None`` when it is absent or invalid."""
    path = _manifest_path(cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable cache manifest %s", path)
        return None


def _write_manifest(
    cache_dir: Path,
    csv_path: Path,
    csv_hash: str,
    task_columns: list[str],
    smiles_column: str,
    n_molecules: int,
) -> None:
    """Write metadata that ties a cache to a specific dataset contents."""
    manifest = {
        "cache_version": FEATURIZATION_CACHE_VERSION,
        "source_csv": Path(csv_path).name,
        "csv_sha256": csv_hash,
        "n_molecules": n_molecules,
        "n_tasks": len(task_columns),
        "smiles_column": smiles_column,
        "task_columns": task_columns,
    }
    path = _manifest_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _cache_matches_dataframe(
    dataset: Any,
    dataframe: pd.DataFrame,
    task_columns: list[str],
    smiles_column: str,
) -> bool:
    """Validate an unmanifested legacy cache against the current CSV."""
    if len(dataset) != len(dataframe):
        return False
    expected_ids = dataframe[smiles_column].astype(str).to_numpy()
    if not np.array_equal(np.asarray(dataset.ids, dtype=str), expected_ids):
        return False
    expected_labels = dataframe[task_columns].fillna(0).to_numpy()
    return dataset.y.shape == expected_labels.shape and bool(
        np.allclose(dataset.y, expected_labels, atol=1e-6)
    )


def _remove_cache(cache_dir: Path) -> None:
    """Remove a stale cache and report a useful error if it is locked."""
    try:
        shutil.rmtree(cache_dir)
    except OSError as error:
        raise RuntimeError(
            f"Could not remove stale featurization cache {cache_dir}. "
            "Close processes using it and delete the directory manually."
        ) from error


def make_stratified_splits(
    labels: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create a deterministic multi-label stratified 80/10/10 split.

    The labels are partitioned into ten multi-label stratified folds with
    ``iterstrat``. Two folds whose sizes are closest to one tenth of the data
    become validation and test; the remaining eight folds form training. This
    keeps the requested proportions while preserving label stratification.
    """
    from itertools import combinations

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    labels = np.asarray(labels)
    if labels.ndim != 2 or len(labels) < 10:
        raise ValueError("labels must be a 2-D matrix with at least 10 rows.")

    splitter = MultilabelStratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed,
    )
    folds = [
        np.sort(test_indices).astype(np.int64)
        for _, test_indices in splitter.split(np.zeros((len(labels), 1)), labels)
    ]

    target = len(labels) / 10.0
    selected_valid, selected_test = min(
        combinations(range(10), 2),
        key=lambda pair: (
            abs(len(folds[pair[0]]) - target)
            + abs(len(folds[pair[1]]) - target),
            pair,
        ),
    )
    test_indices = folds[selected_test]
    valid_indices = folds[selected_valid]
    train_indices = np.sort(
        np.concatenate(
            [
                fold
                for index, fold in enumerate(folds)
                if index not in (selected_valid, selected_test)
            ]
        )
    )

    splits = {
        "train": train_indices.astype(np.int64),
        "valid": valid_indices.astype(np.int64),
        "test": test_indices.astype(np.int64),
    }

    all_indices = np.concatenate(list(splits.values()))
    if len(all_indices) != len(np.unique(all_indices)):
        raise RuntimeError("The generated splits contain overlapping indices.")
    if sorted(all_indices.tolist()) != list(range(len(labels))):
        raise RuntimeError("The generated splits do not cover every molecule.")
    return splits


def split_digest(
    indices: dict[str, np.ndarray],
    dataset_sha256: str,
    seed: int,
) -> str:
    """Return a stable hash for a split definition and dataset version."""
    payload = {
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "indices": {
            name: [int(value) for value in values]
            for name, values in sorted(indices.items())
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_split(
    path: Path,
    indices: dict[str, np.ndarray],
    dataset_sha256: str,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Persist split indices, sizes, dataset hash, and split hash as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "seed": seed,
        "method": (
            "MultilabelStratifiedKFold(n_splits=10, shuffle=True, "
            f"random_state={seed})"
        ),
        "fractions": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "dataset_sha256": dataset_sha256,
        "n_molecules": int(sum(len(values) for values in indices.values())),
        "sizes": {
            name: int(len(values))
            for name, values in sorted(indices.items())
        },
        "split_hash": split_digest(indices, dataset_sha256, seed),
        "indices": {
            name: [int(value) for value in values]
            for name, values in sorted(indices.items())
        },
    }
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return record


def load_split(path: Path) -> dict[str, Any]:
    """Load a saved split JSON and return its record with NumPy indices."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"seed", "dataset_sha256", "split_hash", "indices"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Split file is missing fields: {sorted(missing)}.")
    record["indices"] = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in record["indices"].items()
    }
    for name in ("train", "valid", "test"):
        if name not in record["indices"]:
            raise ValueError(f"Split file does not define {name!r}.")
    return record


def subset_dataset(dataset: Any, indices: np.ndarray) -> Any:
    """Return a NumpyDataset subset for the given integer indices."""
    import deepchem as dc

    indices = np.asarray(indices, dtype=np.int64)
    return dc.data.NumpyDataset(
        X=dataset.X[indices],
        y=dataset.y[indices],
        w=dataset.w[indices],
        ids=[dataset.ids[index] for index in indices],
    )


def get_dataset(
    csv_path: Path = DATASET_PATH,
    cache_dir: Path = DISK_DATASET_DIR,
    *,
    force_rebuild: bool = False,
) -> tuple[Any, list[str], pd.DataFrame]:
    """Build or load the DeepChem featurized dataset.

    The cache is tied to a SHA-256 digest of the source CSV and to the task/SMILES
    columns used to build it. A cache with a stale or missing manifest is
    validated against the CSV when possible, then rebuilt.

    Args:
        csv_path: Curated CSV containing SMILES and one column per odor task.
        cache_dir: Directory used for the DeepChem ``DiskDataset`` cache.
        force_rebuild: Remove and recreate the cache even if it is current.

    Returns:
        A ``(dataset, task_columns, raw_dataframe)`` tuple. ``dataset.ids``
        contains the SMILES strings in the same order as ``dataset.y``.
    """
    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphFeaturizer

    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir)
    dataframe = load_curated_dataframe(csv_path)
    task_columns = get_task_columns(dataframe)
    smiles_column = find_smiles_column(dataframe)

    if not task_columns:
        raise ValueError("The curated dataset contains no numeric odor tasks.")
    if not dataframe[smiles_column].astype(str).is_unique:
        raise ValueError("The curated dataset contains duplicate SMILES.")

    logger.info(
        "Loaded %d molecules with %d odor descriptors",
        len(dataframe),
        len(task_columns),
    )
    csv_hash = sha256_file(csv_path)
    manifest = _read_manifest(cache_dir)

    cache_is_current = bool(
        manifest
        and manifest.get("cache_version") == FEATURIZATION_CACHE_VERSION
        and manifest.get("csv_sha256") == csv_hash
        and manifest.get("task_columns") == task_columns
        and manifest.get("smiles_column") == smiles_column
    )

    if cache_dir.exists() and not force_rebuild and cache_is_current:
        logger.info("Loading validated featurization cache from %s", cache_dir)
        return dc.data.DiskDataset(str(cache_dir)), task_columns, dataframe

    if cache_dir.exists() and not force_rebuild and manifest is None:
        logger.info("Migrating legacy featurization cache at %s", cache_dir)
        dataset = dc.data.DiskDataset(str(cache_dir))
        if _cache_matches_dataframe(
            dataset,
            dataframe,
            task_columns,
            smiles_column,
        ):
            _write_manifest(
                cache_dir,
                csv_path,
                csv_hash,
                task_columns,
                smiles_column,
                len(dataframe),
            )
            return dataset, task_columns, dataframe
        logger.warning("Legacy featurization cache does not match the CSV")

    if cache_dir.exists():
        if manifest and not cache_is_current:
            logger.info(
                "Featurization cache hash/version changed; rebuilding %s",
                cache_dir,
            )
        elif force_rebuild:
            logger.info("Forced featurization cache rebuild for %s", cache_dir)
        _remove_cache(cache_dir)

    logger.info("Featurizing %d molecules (one-time cache build)", len(dataframe))
    featurizer = GraphFeaturizer()
    loader = dc.data.CSVLoader(
        tasks=task_columns,
        feature_field=smiles_column,
        featurizer=featurizer,
    )
    dataset = loader.create_dataset(str(csv_path), data_dir=str(cache_dir))
    _write_manifest(
        cache_dir,
        csv_path,
        csv_hash,
        task_columns,
        smiles_column,
        len(dataframe),
    )
    logger.info("Featurization cache written to %s", cache_dir)
    return dataset, task_columns, dataframe
