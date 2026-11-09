"""Perceptual cosine similarity and cross-scaffold pair discovery."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import (
    DEFAULT_POM_THRESHOLD,
    DEFAULT_TANIMOTO_THRESHOLD,
    SIMILARITY_PATH,
)
from .data import find_smiles_column

logger = logging.getLogger(__name__)

PAIR_COLUMNS = [
    "mol_a_idx",
    "mol_b_idx",
    "mol_a_name",
    "mol_b_name",
    "mol_a_smiles",
    "mol_b_smiles",
    "pom_similarity",
    "tanimoto_similarity",
]


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
            AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                radius=radius,
                nBits=n_bits,
            )
        )
    return fingerprints


def fingerprint_matrix(
    fingerprints: Sequence[Any],
    *,
    n_bits: int | None = None,
) -> np.ndarray:
    """Pack bit-vector fingerprints into a dense uint8 matrix once.

    A dense ``(n_molecules, n_bits)`` matrix makes the Tanimoto calculation a
    few matrix products instead of one RDKit call per molecule pair.
    """
    if not fingerprints:
        width = n_bits or 2048
        return np.zeros((0, width), dtype=np.uint8)

    width = n_bits or int(fingerprints[0].GetNumBits())
    matrix = np.zeros((len(fingerprints), width), dtype=np.uint8)
    for index, fingerprint in enumerate(fingerprints):
        if int(fingerprint.GetNumBits()) != width:
            raise ValueError("All fingerprints must use the same bit width.")
        bits = np.fromiter(fingerprint.GetOnBits(), dtype=np.int64)
        matrix[index, bits] = 1
    return matrix


def tanimoto_similarity_matrix(
    fingerprints: Sequence[Any],
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Compute the full Morgan Tanimoto matrix with one packed bit matrix.

    The matrix is evaluated in row chunks with float32 arithmetic so the
    intermediate intersection products stay bounded. Tanimoto is
    ``intersection / (count_a + count_b - intersection)``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    packed = fingerprint_matrix(fingerprints)
    n_molecules = packed.shape[0]
    if n_molecules == 0:
        return np.zeros((0, 0), dtype=np.float32)

    bit_matrix = packed.astype(np.float32, copy=False)
    counts = bit_matrix.sum(axis=1, dtype=np.float32)
    result = np.empty((n_molecules, n_molecules), dtype=np.float32)

    for start in range(0, n_molecules, chunk_size):
        stop = min(start + chunk_size, n_molecules)
        intersections = bit_matrix[start:stop] @ bit_matrix.T
        unions = counts[start:stop, None] + counts[None, :] - intersections
        result[start:stop] = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float32),
            where=unions > 0,
        )
    return result


def compute_cosine_similarity(
    embeddings: np.ndarray,
    *,
    fill_diagonal: float = -1.0,
) -> np.ndarray:
    """Compute a float32 pairwise cosine-similarity matrix.

    Embeddings are L2-normalized once, then multiplied by their transpose
    instead of invoking a full pairwise estimator.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    normalized = matrix / norms[:, None]
    similarity = (normalized @ normalized.T).astype(np.float32, copy=False)
    if fill_diagonal is not None:
        np.fill_diagonal(similarity, fill_diagonal)
    return similarity


def save_cosine_similarity(
    similarity: np.ndarray,
    path: Path = SIMILARITY_PATH,
) -> Path:
    """Persist a cosine matrix as a float32 ``.npy`` file."""
    array = np.asarray(similarity, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Cosine similarity must be a square 2-D matrix.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return path


def load_cosine_similarity(
    path: Path = SIMILARITY_PATH,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Load a cached cosine matrix and validate its dtype and dimensions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cosine similarity cache not found: {path}. "
            "Run `python scripts/extract_embeddings.py` to regenerate it."
        )
    similarity = np.load(path, allow_pickle=False)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"Cosine cache is not square: {similarity.shape}.")
    if expected_size is not None and similarity.shape != (
        expected_size,
        expected_size,
    ):
        raise ValueError(
            "Cosine cache shape does not match the embedding matrix: "
            f"{similarity.shape} != ({expected_size}, {expected_size})."
        )
    return similarity.astype(np.float32, copy=False)


def load_or_compute_cosine_similarity(
    embeddings: np.ndarray,
    path: Path = SIMILARITY_PATH,
    *,
    force: bool = False,
) -> np.ndarray:
    """Load the cached cosine matrix, computing and saving it on a miss."""
    if not force:
        try:
            return load_cosine_similarity(path, expected_size=len(embeddings))
        except (FileNotFoundError, ValueError) as error:
            logger.info("Rebuilding cosine cache: %s", error)

    similarity = compute_cosine_similarity(embeddings)
    save_cosine_similarity(similarity, path)
    logger.info("Saved cosine similarity cache to %s", path)
    return similarity


def cosine_similarity_to_query(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity from one query embedding to a matrix.

    Raises:
        ValueError: If the query or matrix dimensions are inconsistent or a
            zero-length vector is supplied.
    """
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(
            "Query and embedding matrix dimensions are inconsistent: "
            f"{query.shape} vs {matrix.shape}."
        )

    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0.0 or np.any(row_norms == 0.0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    normalized_matrix = matrix / row_norms[:, None]
    return normalized_matrix @ (query / query_norm)


def top_k_neighbor_indices(
    similarity_row: np.ndarray,
    k: int,
    *,
    exclude_index: int | None = None,
) -> np.ndarray:
    """Return indices of the ``k`` highest-scoring neighbors.

    ``argpartition`` selects the top-k candidates in linear time; only those
    candidates are sorted. Ties are ordered by descending index, matching the
    reverse of a stable ascending sort.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    scores = np.asarray(similarity_row, dtype=float).copy()
    if scores.ndim != 1:
        raise ValueError("similarity_row must be one-dimensional.")
    if exclude_index is not None:
        scores[exclude_index] = -np.inf

    n_scores = scores.size
    if k >= n_scores:
        return np.argsort(scores, kind="stable")[::-1]

    partition = np.argpartition(scores, n_scores - k)[n_scores - k :]
    order = np.lexsort((-partition, -scores[partition]))
    return partition[order]


def evaluate_cross_scaffold_pairs(
    metadata_df: pd.DataFrame,
    pom_sim_matrix: np.ndarray,
    fingerprints: Sequence[Any],
    num_pairs: int = 2,
    pom_threshold: float = DEFAULT_POM_THRESHOLD,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    *,
    tanimoto_matrix: np.ndarray | None = None,
    chunk_size: int = 1024,
) -> pd.DataFrame:
    """Find structurally dissimilar molecules with aligned scent maps.

    A pair qualifies when its OpenPOM cosine similarity is above
    ``pom_threshold`` and its Morgan Tanimoto similarity is below
    ``tanimoto_threshold``. The full boolean mask is reduced to upper-triangle
    indices, so no Python loop over molecule pairs is required.

    Args:
        metadata_df: Metadata rows in the same order as ``pom_sim_matrix``.
        pom_sim_matrix: Pairwise OpenPOM cosine similarities.
        fingerprints: Morgan fingerprints aligned with ``metadata_df``.
        num_pairs: Maximum number of pairs to return.
        pom_threshold: Minimum OpenPOM cosine similarity.
        tanimoto_threshold: Maximum Morgan Tanimoto similarity.
        tanimoto_matrix: Optional precomputed matrix. Computed when omitted.
        chunk_size: Row chunk size used by the Tanimoto calculation.
    """
    if len(metadata_df) != len(fingerprints):
        raise ValueError(
            "metadata_df and fingerprints must describe the same molecules: "
            f"{len(metadata_df)} != {len(fingerprints)}."
        )
    if pom_sim_matrix.shape != (len(fingerprints), len(fingerprints)):
        raise ValueError("pom_sim_matrix shape does not match fingerprints.")

    n_molecules = len(fingerprints)
    if n_molecules == 0:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    if tanimoto_matrix is None:
        tanimoto_matrix = tanimoto_similarity_matrix(
            fingerprints,
            chunk_size=chunk_size,
        )
    if tanimoto_matrix.shape != (n_molecules, n_molecules):
        raise ValueError("tanimoto_matrix shape does not match fingerprints.")

    logger.info("Scanning %d molecules for cross-scaffold pairs", n_molecules)
    mask = (pom_sim_matrix > pom_threshold) & (
        tanimoto_matrix < tanimoto_threshold
    )
    np.fill_diagonal(mask, False)
    rows, columns = np.where(np.triu(mask, k=1))
    if rows.size == 0:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    pom_values = pom_sim_matrix[rows, columns]
    order = np.argsort(-pom_values, kind="stable")[: max(num_pairs, 0)]
    if order.size == 0:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    smiles_column = find_smiles_column(metadata_df)
    smiles_values = metadata_df[smiles_column].astype(str).to_numpy()
    if "name" in metadata_df.columns:
        name_values = metadata_df["name"].astype(str).to_numpy()
    else:
        name_values = smiles_values

    records: list[dict[str, Any]] = []
    for pair_rank, flat_index in enumerate(order):
        row = int(rows[flat_index])
        column = int(columns[flat_index])
        records.append(
            {
                "mol_a_idx": row,
                "mol_b_idx": column,
                "mol_a_name": name_values[row],
                "mol_b_name": name_values[column],
                "mol_a_smiles": smiles_values[row],
                "mol_b_smiles": smiles_values[column],
                "pom_similarity": float(pom_sim_matrix[row, column]),
                "tanimoto_similarity": float(
                    tanimoto_matrix[row, column]
                ),
            }
        )
    return pd.DataFrame(records, columns=PAIR_COLUMNS).reset_index(drop=True)
