"""Dataset loading, column normalization, and graph featurization."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATASET_PATH, DATASET_URL, DISK_DATASET_DIR

logger = logging.getLogger(__name__)

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


def get_dataset(
    csv_path: Path = DATASET_PATH,
    cache_dir: Path = DISK_DATASET_DIR,
) -> tuple[Any, list[str], pd.DataFrame]:
    """Build or load the DeepChem featurized dataset.

    Args:
        csv_path: Curated CSV containing SMILES and one column per odor task.
        cache_dir: Directory used for the DeepChem ``DiskDataset`` cache.

    Returns:
        A ``(dataset, task_columns, raw_dataframe)`` tuple. ``dataset.ids``
        contains the SMILES strings in the same order as ``dataset.y``.
    """
    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphFeaturizer

    dataframe = load_curated_dataframe(Path(csv_path))
    task_columns = get_task_columns(dataframe)
    smiles_column = find_smiles_column(dataframe)
    cache_dir = Path(cache_dir)

    if not task_columns:
        raise ValueError("The curated dataset contains no numeric odor tasks.")
    logger.info(
        "Loaded %d molecules with %d odor descriptors",
        len(dataframe),
        len(task_columns),
    )

    if cache_dir.exists():
        logger.info("Loading featurization cache from %s", cache_dir)
        return dc.data.DiskDataset(str(cache_dir)), task_columns, dataframe

    logger.info("Featurizing %d molecules (one-time cache build)", len(dataframe))
    featurizer = GraphFeaturizer()
    loader = dc.data.CSVLoader(
        tasks=task_columns,
        feature_field=smiles_column,
        featurizer=featurizer,
    )
    dataset = loader.create_dataset(str(csv_path), data_dir=str(cache_dir))
    return dataset, task_columns, dataframe
