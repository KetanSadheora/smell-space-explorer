"""Perceptual cosine similarity and cross-scaffold pair discovery."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .config import DEFAULT_POM_THRESHOLD, DEFAULT_TANIMOTO_THRESHOLD
from .data import find_smiles_column

logger = logging.getLogger(__name__)


def morgan_fingerprints(
    smiles: Sequence[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[Any]:
    """Compute ECFP4 Morgan fingerprints for a sequence of SMILES.

    Raises:
        ValueError: If any SMILES cannot be parsed, so callers never receive a
            fingerprint list that is silently shorter than the input.
    """
    fingerprints: list[Any] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(
                f"Invalid SMILES at index {index}: {value!r}. "
                "Fingerprints must align one-to-one with metadata rows."
            )
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(molecule, radius=radius, nBits=n_bits)
        )
    return fingerprints


def compute_cosine_similarity(
    embeddings: np.ndarray,
    *,
    fill_diagonal: float = -1.0,
) -> np.ndarray:
    """Compute pairwise cosine similarity, excluding self-matches by default."""
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(embeddings)
    if fill_diagonal is not None:
        np.fill_diagonal(similarity, fill_diagonal)
    return similarity


def top_k_neighbor_indices(
    similarity_row: np.ndarray,
    k: int,
    *,
    exclude_index: int | None = None,
) -> np.ndarray:
    """Return indices of the ``k`` highest-scoring neighbors."""
    scores = np.asarray(similarity_row, dtype=float).copy()
    if exclude_index is not None:
        scores[exclude_index] = -np.inf
    return np.argsort(scores)[::-1][:k]


def evaluate_cross_scaffold_pairs(
    metadata_df: pd.DataFrame,
    pom_sim_matrix: np.ndarray,
    fingerprints: Sequence[Any],
    num_pairs: int = 2,
    pom_threshold: float = DEFAULT_POM_THRESHOLD,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
) -> pd.DataFrame:
    """Find structurally dissimilar molecules with highly aligned scent maps.

    A pair qualifies when its OpenPOM cosine similarity is above
    ``pom_threshold`` and its Morgan Tanimoto similarity is below
    ``tanimoto_threshold``. Results are sorted by POM similarity.
    """
    columns = [
        "mol_a_idx",
        "mol_b_idx",
        "mol_a_name",
        "mol_b_name",
        "mol_a_smiles",
        "mol_b_smiles",
        "pom_similarity",
        "tanimoto_similarity",
    ]
    if len(metadata_df) != len(fingerprints):
        raise ValueError(
            "metadata_df and fingerprints must describe the same molecules: "
            f"{len(metadata_df)} != {len(fingerprints)}."
        )
    if pom_sim_matrix.shape != (len(fingerprints), len(fingerprints)):
        raise ValueError("pom_sim_matrix shape does not match fingerprints.")

    smiles_column = find_smiles_column(metadata_df)
    rows: list[dict[str, Any]] = []
    logger.info("Scanning %d molecules for cross-scaffold pairs", len(fingerprints))

    for i in range(len(fingerprints)):
        tanimoto_scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints)
        )
        candidates = np.where(
            (pom_sim_matrix[i] > pom_threshold)
            & (tanimoto_scores < tanimoto_threshold)
        )[0]
        for j in candidates:
            if j <= i:
                continue
            row_a = metadata_df.iloc[i]
            row_b = metadata_df.iloc[j]
            smiles_a = str(row_a[smiles_column])
            smiles_b = str(row_b[smiles_column])
            rows.append(
                {
                    "mol_a_idx": i,
                    "mol_b_idx": int(j),
                    "mol_a_name": row_a.get("name", smiles_a),
                    "mol_b_name": row_b.get("name", smiles_b),
                    "mol_a_smiles": smiles_a,
                    "mol_b_smiles": smiles_b,
                    "pom_similarity": float(pom_sim_matrix[i, j]),
                    "tanimoto_similarity": float(tanimoto_scores[j]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("pom_similarity", ascending=False)
        .head(num_pairs)
        .reset_index(drop=True)
    )
