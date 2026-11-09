"""Tests for CSV-hash-keyed DeepChem featurization caching."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from smell_space.data import (
    CACHE_MANIFEST_FILENAME,
    FEATURIZATION_CACHE_VERSION,
    get_dataset,
    sha256_file,
)

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "_tmp_cache_dataset.csv"
CACHE_DIR = BASE / "_tmp_cache_dataset"


def _write_dataset(path: Path, first_task_value: int) -> None:
    dataframe = pd.DataFrame(
        {
            "nonStereoSMILES": ["CCO", "CC(=O)O", "c1ccccc1"],
            "task_a": [first_task_value, 0, 0],
            "task_b": [0, 1, 0],
        }
    )
    dataframe.to_csv(path, index=False)


def test_featurization_cache_is_keyed_to_csv_hash() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CSV_PATH.unlink(missing_ok=True)

    try:
        _write_dataset(CSV_PATH, first_task_value=1)
        dataset, tasks, dataframe = get_dataset(CSV_PATH, CACHE_DIR)
        assert len(dataset) == 3
        assert tasks == ["task_a", "task_b"]
        assert len(dataframe) == 3

        manifest_path = CACHE_DIR / CACHE_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["cache_version"] == FEATURIZATION_CACHE_VERSION
        assert manifest["csv_sha256"] == sha256_file(CSV_PATH)
        assert manifest["n_molecules"] == 3
        assert manifest["n_tasks"] == 2

        _write_dataset(CSV_PATH, first_task_value=0)
        rebuilt, _, _ = get_dataset(CSV_PATH, CACHE_DIR)
        new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert new_manifest["csv_sha256"] == sha256_file(CSV_PATH)
        assert float(rebuilt.y[0, 0]) == 0.0
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CSV_PATH.unlink(missing_ok=True)
