"""Tests for opt-in training utilities that do not require model training."""

from __future__ import annotations

import random

import numpy as np

from smell_space.model import set_global_seed, split_train_valid_dataset


def test_set_global_seed_repeats_python_and_numpy_streams() -> None:
    set_global_seed(123)
    first = (random.random(), np.random.rand())
    set_global_seed(123)
    second = (random.random(), np.random.rand())
    assert first == second


def test_stratified_split_is_disjoint_and_complete() -> None:
    import deepchem as dc

    rng = np.random.default_rng(5)
    labels = (rng.random((30, 4)) > 0.6).astype(np.float32)
    labels[:, 0] = 1
    dataset = dc.data.NumpyDataset(
        X=np.arange(60, dtype=np.float32).reshape(30, 2),
        y=labels,
        ids=[f"mol-{index}" for index in range(30)],
    )

    train, valid = split_train_valid_dataset(dataset, 0.25, seed=7)

    assert len(train) == 22
    assert len(valid) == 8
    assert set(train.ids).isdisjoint(valid.ids)
    assert set(train.ids) | set(valid.ids) == set(dataset.ids)
    assert train.X.shape[1] == 2 and valid.X.shape[1] == 2
