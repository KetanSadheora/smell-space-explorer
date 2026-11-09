"""Numerical equivalence tests for the optimized similarity routines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smell_space.similarity import (
    compute_cosine_similarity,
    evaluate_cross_scaffold_pairs,
    load_cosine_similarity,
    morgan_fingerprints,
    save_cosine_similarity,
    tanimoto_similarity_matrix,
    top_k_neighbor_indices,
)
from tests.reference import (
    compute_cosine_similarity_reference,
    compute_tanimoto_matrix_reference,
    evaluate_cross_scaffold_pairs_reference,
    top_k_neighbor_indices_reference,
)

SAMPLE_SMILES = [
    "CC(O)CN",
    "CCC(=O)C(=O)O",
    "O=C(O)CCc1ccccc1",
    "OCc1ccc(O)cc1",
    "O=Cc1ccc(O)cc1",
    "O=C(O)c1ccc(O)cc1",
    "CC(=O)O",
    "CC=O",
    "CC(=O)C(C)O",
    "CC(C)=O",
    "O=C(O)CCCCC(=O)O",
    "O=Cc1ccccc1",
    "O=C(O)c1ccccc1",
    "OCc1ccccc1",
    "CCCC=O",
    "CCCCO",
    "CCCC(=O)O",
    "O=CO",
    "O=C(O)CC(O)(CC(=O)O)C(=O)O",
    "CC(C)c1ccc(CO)cc1",
    "CC(C)c1ccc(C=O)cc1",
    "C=Cc1ccc(O)c(OC)c1",
    "Cc1ccccc1O",
    "Cc1cccc(O)c1",
    "CCCCCCCC(=O)O",
    "CCCCCCCC=O",
    "COc1ccccc1O",
    "O=C(O)CC(O)C(=O)O",
    "CCC=O",
    "CSCCC(=O)O",
    "CC(O)C(=O)O",
    "CC(=O)C(C)=O",
    "O=C1CCc2ccccc2O1",
    "O=C(CO)CO",
    "CCO",
    "O=C(O)CO",
    "c1ccc2[nH]ccc2c1",
    "CC(C)CC(N)C(=O)O",
    "O=C(O)C(O)C(O)C(=O)O",
    "CSCCC(N)C(=O)O",
    "CS",
    "CC(=O)C=O",
    "c1ccc2ccccc2c1",
    "CCCCCCCCO",
    "O=C(O)C(=O)CO",
    "CCCCCCCCCCCCCCCC(=O)O",
    "NC(Cc1ccccc1)C(=O)O",
    "Oc1ccccc1",
    "O=C(O)C(=O)Cc1ccccc1",
    "O=CCc1ccccc1",
]


def test_vectorized_tanimoto_matches_rdkit_bulk() -> None:
    fingerprints = morgan_fingerprints(SAMPLE_SMILES)
    expected = compute_tanimoto_matrix_reference(fingerprints)
    actual = tanimoto_similarity_matrix(fingerprints, chunk_size=13)

    assert actual.shape == (50, 50)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.diag(actual), np.ones(50), atol=1e-6)


def test_vectorized_cross_scaffold_matches_reference() -> None:
    fingerprints = morgan_fingerprints(SAMPLE_SMILES)
    tanimoto = compute_tanimoto_matrix_reference(fingerprints)
    n_molecules = len(SAMPLE_SMILES)

    rng = np.random.default_rng(7)
    raw = rng.random((n_molecules, n_molecules), dtype=np.float32)
    pom = np.triu(raw, k=1)
    pom = pom + pom.T
    pom = 0.80 + 0.18 * pom
    np.fill_diagonal(pom, -1.0)

    metadata = pd.DataFrame(
        {
            "smiles": SAMPLE_SMILES,
            "name": [f"molecule-{index}" for index in range(n_molecules)],
        }
    )
    expected = evaluate_cross_scaffold_pairs_reference(
        metadata,
        pom,
        fingerprints,
        num_pairs=12,
        pom_threshold=0.88,
        tanimoto_threshold=0.5,
    )
    actual = evaluate_cross_scaffold_pairs(
        metadata,
        pom,
        fingerprints,
        num_pairs=12,
        pom_threshold=0.88,
        tanimoto_threshold=0.5,
        tanimoto_matrix=tanimoto,
        chunk_size=13,
    )

    assert len(actual) > 0
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-6,
    )


def test_top_k_argpartition_matches_argsort() -> None:
    rng = np.random.default_rng(11)
    scores = rng.random(200).astype(np.float32)
    expected = top_k_neighbor_indices_reference(scores, 10, exclude_index=42)
    actual = top_k_neighbor_indices(scores, 10, exclude_index=42)
    np.testing.assert_array_equal(actual, expected)


def test_cosine_similarity_matches_sklearn() -> None:
    rng = np.random.default_rng(23)
    embeddings = rng.normal(size=(50, 256)).astype(np.float32)
    expected = compute_cosine_similarity_reference(embeddings)
    actual = compute_cosine_similarity(embeddings)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
    np.testing.assert_array_equal(np.diag(actual), np.full(50, -1.0))


def test_cosine_cache_round_trip() -> None:
    rng = np.random.default_rng(31)
    embeddings = rng.normal(size=(12, 8)).astype(np.float32)
    similarity = compute_cosine_similarity(embeddings)
    path = Path(__file__).resolve().parent / "_tmp_similarity_cache.npy"

    try:
        save_cosine_similarity(similarity, path)
        loaded = load_cosine_similarity(path, expected_size=12)

        np.testing.assert_array_equal(loaded, similarity)
        with pytest.raises(ValueError):
            load_cosine_similarity(path, expected_size=13)
    finally:
        path.unlink(missing_ok=True)
