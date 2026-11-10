"""Tests for deterministic multi-label 80/10/10 splitting."""

from __future__ import annotations

import numpy as np

from smell_space.data import (
    load_split,
    make_stratified_splits,
    save_split,
    split_digest,
)


def test_same_seed_gives_identical_indices() -> None:
    labels = (np.arange(120)[:, None] + np.arange(6)[None, :]) % 3 == 0
    first = make_stratified_splits(labels, seed=42)
    second = make_stratified_splits(labels, seed=42)

    for name in ("train", "valid", "test"):
        np.testing.assert_array_equal(first[name], second[name])


def test_splits_are_disjoint_and_cover_dataset() -> None:
    rng = np.random.default_rng(9)
    labels = (rng.random((200, 8)) > 0.65).astype(np.float32)
    labels[:, 0] = 1
    splits = make_stratified_splits(labels, seed=42)

    assert set(splits) == {"train", "valid", "test"}
    assert len(splits["train"]) == 160
    # Iterative stratification can shift one sample between the equal-sized
    # validation/test folds because labels, not only counts, drive assignment.
    assert abs(len(splits["valid"]) - 20) <= 1
    assert abs(len(splits["test"]) - 20) <= 1

    combined = np.concatenate(
        [splits["train"], splits["valid"], splits["test"]]
    )
    assert len(combined) == len(labels)
    assert len(np.unique(combined)) == len(labels)
    assert set(combined.tolist()) == set(range(len(labels)))
    assert set(splits["train"]).isdisjoint(splits["valid"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["valid"]).isdisjoint(splits["test"])


def test_save_and_load_split_preserves_hash() -> None:
    from pathlib import Path

    labels = (np.arange(100)[:, None] + np.arange(4)[None, :]) % 2 == 0
    splits = make_stratified_splits(labels, seed=42)
    dataset_hash = "a" * 64
    expected_hash = split_digest(splits, dataset_hash, 42)
    path = Path(__file__).resolve().parent / "_tmp_split.json"
    try:
        record = save_split(path, splits, dataset_hash, seed=42)
        loaded = load_split(path)
        assert record["split_hash"] == expected_hash
        assert loaded["split_hash"] == expected_hash
        assert loaded["sizes"] == {"train": 80, "valid": 10, "test": 10}
        for name in ("train", "valid", "test"):
            np.testing.assert_array_equal(
                loaded["indices"][name],
                splits[name],
            )
    finally:
        path.unlink(missing_ok=True)
