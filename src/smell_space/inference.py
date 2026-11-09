"""Checkpoint-backed single-molecule odor prediction.

The predictor uses the same OpenPOM graph featurizer and MPNN forward pass as
training. It never fabricates probabilities: if no trained checkpoint exists it
raises an error telling the caller to run ``scripts/train.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from .config import MODEL_DIR
from .data import get_task_columns, load_curated_dataframe
from .embeddings import _find_penultimate_linear
from .model import build_model

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 256


@dataclass(frozen=True)
class OdorPrediction:
    """A real model prediction plus the query's principal odor embedding.

    Attributes:
        smiles: SMILES supplied by the caller.
        canonical_smiles: RDKit canonical form of the query.
        probabilities: Every descriptor and probability, sorted descending.
        embedding: The 256-D penultimate OpenPOM embedding.
        top_n: Number of entries exposed by :attr:`top_predictions`.
    """

    smiles: str
    canonical_smiles: str
    probabilities: tuple[tuple[str, float], ...]
    embedding: np.ndarray
    top_n: int = 5

    @property
    def top_predictions(self) -> tuple[tuple[str, float], ...]:
        """Return the highest-probability descriptors, sorted descending."""
        return self.probabilities[: self.top_n]

    def probability_dict(self) -> dict[str, float]:
        """Return all descriptor probabilities as a name-to-score mapping."""
        return dict(self.probabilities)


def _project_checkpoints(model_dir: Path) -> list[Path]:
    """Return DeepChem checkpoint files ordered oldest to newest."""
    if not model_dir.is_dir():
        return []
    return sorted(
        model_dir.glob("checkpoint*.pt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


@lru_cache(maxsize=1)
def _load_model_bundle(
    model_dir_string: str,
) -> tuple[Any, Any, tuple[str, ...]]:
    """Load the model, featurizer, and task names once per process.

    The class-imbalance ratios are needed only to construct the training loss,
    but are part of the model constructor contract. They are recomputed from
    the curated labels exactly as the training script does, without rebuilding
    the expensive DeepChem graph dataset.
    """
    from openpom.feat.graph_featurizer import GraphFeaturizer

    model_dir = Path(model_dir_string)
    checkpoints = _project_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained OpenPOM checkpoint found in {model_dir}. "
            "Run `python scripts/train.py` before requesting predictions."
        )

    dataframe = load_curated_dataframe()
    task_columns = get_task_columns(dataframe)
    if not task_columns:
        raise ValueError("The curated dataset contains no odor descriptor columns.")

    counts = dataframe[task_columns].fillna(0).sum(axis=0).to_numpy(dtype=float)
    maximum_count = float(counts.max()) if counts.size else 0.0
    imbalance_ratios = (
        (counts / maximum_count).tolist()
        if maximum_count > 0
        else [1.0] * len(task_columns)
    )

    model = build_model(len(task_columns), imbalance_ratios, model_dir)
    checkpoint = checkpoints[-1]
    logger.info("Restoring OpenPOM checkpoint %s", checkpoint)
    try:
        model.restore(checkpoint=str(checkpoint))
    except Exception as error:
        raise RuntimeError(
            f"Failed to restore checkpoint {checkpoint}. Re-run "
            "`python scripts/train.py` if the checkpoint is corrupt."
        ) from error

    model.model.eval()
    featurizer = GraphFeaturizer()
    return model, featurizer, tuple(task_columns)


def get_model(model_dir: Path | str = MODEL_DIR) -> Any:
    """Return the process-wide cached OpenPOM model.

    Raises:
        FileNotFoundError: If no checkpoint exists in ``model_dir``.
    """
    resolved = str(Path(model_dir).expanduser().resolve())
    return _load_model_bundle(resolved)[0]


def clear_model_cache() -> None:
    """Drop the cached model, featurizer, and task names.

    Useful after training a new checkpoint in the same Python process and in
    tests that point ``SMELL_MODEL_DIR`` at a fixture directory.
    """
    _load_model_bundle.cache_clear()


def _validate_smiles(smiles: str) -> tuple[Any, str]:
    """Validate a SMILES string and return its RDKit molecule and canonical form."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError(
            f"Invalid SMILES {smiles!r}. Provide a valid RDKit-parsable structure."
        )
    return molecule, Chem.MolToSmiles(molecule)


def predict_odor_profile(smiles: str, top_n: int = 5) -> OdorPrediction:
    """Predict all 138 odor-descriptor probabilities for one molecule.

    Args:
        smiles: A valid SMILES string.
        top_n: Number of top descriptors exposed via
            :attr:`OdorPrediction.top_predictions`. All probabilities are
            always retained in :attr:`OdorPrediction.probabilities`.

    Returns:
        An :class:`OdorPrediction` containing every descriptor probability
        sorted descending and the query's 256-D embedding.

    Raises:
        ValueError: If the SMILES is invalid or ``top_n`` is less than one.
        FileNotFoundError: If no trained checkpoint is available.
        RuntimeError: If featurization or the model forward pass fails, or if
            the model returns an unexpected embedding shape.
    """
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    molecule, canonical_smiles = _validate_smiles(smiles)
    model, featurizer, task_columns = _load_model_bundle(
        str(Path(MODEL_DIR).expanduser().resolve())
    )

    from deepchem.feat.graph_data import GraphData

    features = featurizer.featurize([molecule])
    if len(features) != 1 or not isinstance(features[0], GraphData):
        raise RuntimeError(
            "OpenPOM graph featurization failed for this molecule. "
            "The structure may contain unsupported chemistry."
        )

    import dgl
    import torch

    graph = features[0].to_dgl_graph(
        self_loop=bool(getattr(model, "_self_loop", False))
    )
    batched_graph = dgl.batch([graph]).to(model.device)

    model.model.eval()
    captured_embeddings: list[np.ndarray] = []
    target_layer = _find_penultimate_linear(model)

    def capture_embedding(
        _module: Any,
        layer_inputs: tuple[Any, ...],
    ) -> None:
        captured_embeddings.append(
            layer_inputs[0].detach().cpu().numpy()
        )

    hook = target_layer.register_forward_pre_hook(capture_embedding)
    try:
        with torch.no_grad():
            outputs = model.model(batched_graph)
    except Exception as error:
        raise RuntimeError(
            f"The OpenPOM forward pass failed for {canonical_smiles}."
        ) from error
    finally:
        hook.remove()

    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise RuntimeError(
            "The OpenPOM model returned an unexpected output. Expected "
            "(probabilities, logits, embedding)."
        )
    if not captured_embeddings:
        raise RuntimeError("The OpenPOM embedding layer was never called.")
    probability_tensor, _logits, _model_embedding = outputs
    probabilities = (
        probability_tensor.detach().cpu().numpy().astype(float).reshape(-1)
    )
    embedding = np.asarray(captured_embeddings[0], dtype=np.float32).reshape(-1)

    if probabilities.shape[0] != len(task_columns):
        raise RuntimeError(
            "Model output size does not match the curated task columns: "
            f"{probabilities.shape[0]} != {len(task_columns)}."
        )
    if embedding.shape != (EMBEDDING_DIMENSION,):
        raise RuntimeError(
            "Expected a 256-D OpenPOM embedding, got "
            f"{embedding.shape}."
        )
    if not np.isfinite(probabilities).all() or not np.isfinite(embedding).all():
        raise RuntimeError("The model returned non-finite values.")

    order = np.argsort(-probabilities, kind="stable")
    ordered = tuple(
        (task_columns[index], float(probabilities[index])) for index in order
    )
    logger.info(
        "Predicted %d descriptors for %s; top=%s",
        len(ordered),
        canonical_smiles,
        ordered[: min(top_n, len(ordered))],
    )
    return OdorPrediction(
        smiles=smiles,
        canonical_smiles=canonical_smiles,
        probabilities=ordered,
        embedding=embedding,
        top_n=min(top_n, len(ordered)),
    )
