"""Single-molecule inference helpers.

The original notebook used an illustrative fallback for unseen molecules. That
fallback is retained here only so the Phase 1 repository extraction preserves
current behavior; Phase 2 replaces it with checkpoint-backed inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import DATASET_PATH, EMBEDDINGS_PATH, METADATA_PATH
from .data import get_task_columns, load_curated_dataframe

logger = logging.getLogger(__name__)

_FALLBACK_PROFILE: tuple[tuple[str, float], ...] = (
    ("sweet", 0.74),
    ("fruity", 0.62),
    ("floral", 0.45),
    ("herbal", 0.31),
    ("woody", 0.15),
)


def _metadata_smiles_column(metadata: pd.DataFrame) -> str:
    """Return the SMILES column used by the embedding metadata table."""
    return "smiles" if "smiles" in metadata.columns else "nonStereoSMILES"


def predict_odor_profile_fallback(
    smiles: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Return dataset labels when available and illustrative values otherwise.

    This helper intentionally mirrors the original notebook and is not a
    validated model prediction for unseen molecules.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES representation: {smiles}")

    embeddings_path = Path(EMBEDDINGS_PATH)
    metadata_path = Path(METADATA_PATH)
    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Embedding artifacts are missing. Run scripts/extract_embeddings.py."
        )

    np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    dataset_df = load_curated_dataframe(Path(DATASET_PATH))
    task_columns = get_task_columns(dataset_df)

    smiles_column = _metadata_smiles_column(metadata)
    matched_rows = metadata[metadata[smiles_column] == smiles]
    if len(matched_rows) > 0:
        index = int(matched_rows.index[0])
        labels = dataset_df[task_columns].fillna(0).to_numpy()
        active_indices = np.where(labels[index] > 0.5)[0]
        predictions = [(task_columns[i], 1.0) for i in active_indices]
    else:
        predictions = list(_FALLBACK_PROFILE)

    return sorted(predictions, key=lambda item: item[1], reverse=True)[:top_n]
