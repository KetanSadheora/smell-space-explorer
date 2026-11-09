"""Inference validation, dataset contract, and embedding-shape tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smell_space.config import MODEL_DIR
from smell_space.data import get_task_columns, load_curated_dataframe
from smell_space.inference import predict_odor_profile


@pytest.mark.parametrize("invalid_smiles", ["", "   ", "not-a-smiles", "C1CC"])
def test_invalid_smiles_raises_value_error(invalid_smiles: str) -> None:
    with pytest.raises(ValueError):
        predict_odor_profile(invalid_smiles, top_n=5)


def test_invalid_top_n_raises_before_model_load() -> None:
    with pytest.raises(ValueError, match="top_n"):
        predict_odor_profile("CCO", top_n=0)


def test_curated_dataset_has_exactly_138_numeric_tasks() -> None:
    dataframe = load_curated_dataframe()
    task_columns = get_task_columns(dataframe)

    assert len(task_columns) == 138
    assert len(set(task_columns)) == 138
    assert dataframe.shape == (4983, 140)
    assert "descriptors" not in task_columns
    assert "nonStereoSMILES" not in task_columns


@pytest.mark.slow
def test_embedding_extraction_shape() -> None:
    checkpoints = list(Path(MODEL_DIR).glob("checkpoint*.pt"))
    if not checkpoints:
        pytest.skip("No trained checkpoint found.")

    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphFeaturizer
    from rdkit import Chem

    from smell_space.embeddings import extract_embeddings
    from smell_space.inference import get_model

    dataframe = load_curated_dataframe()
    task_columns = get_task_columns(dataframe)
    smiles = ["COc1cc(C=O)ccc1O", "CC1CCCC2C1(CCCC2(C)O)C"]
    featurizer = GraphFeaturizer()
    graphs = featurizer.featurize([Chem.MolFromSmiles(value) for value in smiles])
    dataset = dc.data.NumpyDataset(
        X=graphs,
        y=np.zeros((len(graphs), len(task_columns))),
        ids=smiles,
    )

    embeddings = extract_embeddings(get_model(), dataset)

    assert embeddings.shape == (len(smiles), 256)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
