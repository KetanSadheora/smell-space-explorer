"""Original loop-based implementations kept for numerical equivalence tests.

These functions intentionally mirror the pre-optimization code. They are not
imported by the application.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from rdkit.Chem import DataStructs
from sklearn.metrics.pairwise import cosine_similarity


def top_k_neighbor_indices_reference(
    similarity_row: np.ndarray,
    k: int,
    *,
    exclude_index: int | None = None,
) -> np.ndarray:
    """Original full-argsort top-k implementation."""
    scores = np.asarray(similarity_row, dtype=float).copy()
    if exclude_index is not None:
        scores[exclude_index] = -np.inf
    return np.argsort(scores)[::-1][:k]


def compute_cosine_similarity_reference(
    embeddings: np.ndarray,
    *,
    fill_diagonal: float = -1.0,
) -> np.ndarray:
    """Original sklearn pairwise cosine implementation."""
    similarity = cosine_similarity(embeddings)
    if fill_diagonal is not None:
        np.fill_diagonal(similarity, fill_diagonal)
    return similarity


def compute_tanimoto_matrix_reference(
    fingerprints: Sequence[Any],
) -> np.ndarray:
    """Original row-by-row ``BulkTanimotoSimilarity`` implementation."""
    n_molecules = len(fingerprints)
    matrix = np.zeros((n_molecules, n_molecules), dtype=float)
    for row in range(n_molecules):
        matrix[row] = DataStructs.BulkTanimotoSimilarity(
            fingerprints[row],
            fingerprints,
        )
    return matrix


def evaluate_cross_scaffold_pairs_reference(
    metadata_df: pd.DataFrame,
    pom_sim_matrix: np.ndarray,
    fingerprints: Sequence[Any],
    num_pairs: int = 2,
    pom_threshold: float = 0.88,
    tanimoto_threshold: float = 0.32,
) -> pd.DataFrame:
    """Original nested-loop cross-scaffold search."""
    n_mols = len(fingerprints)
    scaffold_leaps = []

    for i in range(n_mols):
        tanimoto_scores = np.array(
            DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints)
        )
        valid_candidates = np.where(
            (pom_sim_matrix[i] > pom_threshold)
            & (tanimoto_scores < tanimoto_threshold)
        )[0]
        for j in valid_candidates:
            if j > i:
                scaffold_leaps.append(
                    {
                        "mol_a_idx": i,
                        "mol_b_idx": int(j),
                        "mol_a_name": metadata_df.iloc[i].get("name", "Unknown"),
                        "mol_b_name": metadata_df.iloc[j].get("name", "Unknown"),
                        "mol_a_smiles": metadata_df.iloc[i]["smiles"],
                        "mol_b_smiles": metadata_df.iloc[j]["smiles"],
                        "pom_similarity": pom_sim_matrix[i, j],
                        "tanimoto_similarity": tanimoto_scores[j],
                    }
                )

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
    leap_df = pd.DataFrame(scaffold_leaps, columns=columns)
    if len(leap_df) == 0:
        return leap_df

    return (
        leap_df.sort_values(by="pom_similarity", ascending=False)
        .head(num_pairs)
        .reset_index(drop=True)
    )
