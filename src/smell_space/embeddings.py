"""Penultimate OpenPOM embedding extraction and UMAP projection."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch.nn as nn

from .config import DEFAULT_SEED

logger = logging.getLogger(__name__)

EMBEDDING_LAYER_NAME = "ffn.linears.2"
EMBEDDING_DIMENSION = 256

ODOR_FAMILIES: dict[str, list[str]] = {
    "Fruity/Citrus": [
        "fruity",
        "citrus",
        "apple",
        "banana",
        "berry",
        "grape",
        "orange",
        "lemon",
        "peach",
        "pear",
        "pineapple",
        "strawberry",
        "melon",
        "cherry",
        "estery",
    ],
    "Floral": [
        "floral",
        "rose",
        "jasmine",
        "lavender",
        "violet",
        "muguet",
        "lily",
        "hyacinth",
        "carnation",
        "gardenia",
        "orris",
        "orange flower",
        "geranium",
    ],
    "Woody/Earthy": [
        "woody",
        "cedar",
        "sandalwood",
        "pine",
        "earthy",
        "mossy",
        "patchouli",
        "vetiver",
        "bark",
        "soil",
        "musty",
        "oakmoss",
        "tobacco",
    ],
    "Sweet/Vanilla": [
        "sweet",
        "vanilla",
        "caramel",
        "honey",
        "chocolate",
        "cocoa",
        "almond",
        "nutty",
        "coumarin",
        "creamy",
        "buttery",
        "molasses",
        "toffee",
        "balsamic",
    ],
    "Green/Herbal": [
        "green",
        "herbal",
        "grassy",
        "minty",
        "cooling",
        "leafy",
        "tea",
        "thyme",
        "rosemary",
        "sage",
        "eucalyptus",
        "camphoreous",
        "vegetable",
        "weedy",
    ],
    "Spicy": [
        "spicy",
        "cinnamon",
        "clove",
        "pepper",
        "ginger",
        "anise",
        "nutmeg",
        "cardamom",
        "fennel",
        "warm",
    ],
    "Sulfurous/Chemical": [
        "sulfurous",
        "garlic",
        "onion",
        "pungent",
        "gasoline",
        "ethereal",
        "chemical",
        "solvent",
        "metallic",
        "fermented",
        "sour",
        "acidic",
        "burnt",
    ],
    "Musk/Animalic": [
        "musk",
        "leathery",
        "animalic",
        "ambergris",
        "civet",
        "castoreum",
        "fatty",
        "waxy",
    ],
}


def get_dominant_family(active_tasks: Iterable[str]) -> str:
    """Map active descriptor names to the strongest broad odor family."""
    counts = {family: 0 for family in ODOR_FAMILIES}
    for task in active_tasks:
        for family, members in ODOR_FAMILIES.items():
            if any(member in task.lower() for member in members):
                counts[family] += 1
                break
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return ranked[0][0] if ranked[0][1] > 0 else "Other"


def _find_penultimate_linear(model: Any) -> nn.Linear:
    """Return the named linear layer whose input is the 256-D POM embedding.

    The fixed OpenPOM FFN is ``256 -> 300 -> 256 -> 138``. Its output layer is
    ``ffn.linears.2``, and the input to that layer is the 256-D embedding used
    by the original notebook hook.
    """
    try:
        target_layer = model.model.get_submodule(EMBEDDING_LAYER_NAME)
    except AttributeError as error:
        raise RuntimeError(
            "The OpenPOM model does not expose the expected embedding layer "
            f"{EMBEDDING_LAYER_NAME!r}."
        ) from error
    if not isinstance(target_layer, nn.Linear):
        raise RuntimeError(
            f"{EMBEDDING_LAYER_NAME!r} is not an nn.Linear layer."
        )
    if target_layer.in_features != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Expected {EMBEDDING_LAYER_NAME!r} to consume a "
            f"{EMBEDDING_DIMENSION}-D embedding, got "
            f"{target_layer.in_features}."
        )
    return target_layer


def extract_embeddings(model: Any, dataset: Any) -> np.ndarray:
    """Extract the 256-D penultimate embedding for every dataset molecule.

    Prediction runs in batches via ``model.predict`` under ``torch.no_grad``.
    The hook captures the input to the named 256-D output layer.
    """
    import torch

    py_model = model.model
    py_model.eval()
    captured: list[np.ndarray] = []
    target_layer = _find_penultimate_linear(model)
    logger.info(
        "Hooked %s (in_features=%d); batch_size=%s",
        EMBEDDING_LAYER_NAME,
        target_layer.in_features,
        getattr(model, "batch_size", "unknown"),
    )

    def capture_input(
        _module: nn.Module,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        captured.append(inputs[0].detach().cpu().numpy())

    hook = target_layer.register_forward_hook(capture_input)
    try:
        with torch.no_grad():
            model.predict(dataset)
    finally:
        hook.remove()

    if not captured:
        raise RuntimeError("Embedding extraction produced no activations.")
    embeddings = np.concatenate(captured, axis=0)[: len(dataset)]
    if embeddings.shape != (len(dataset), EMBEDDING_DIMENSION):
        raise RuntimeError(
            "Expected embedding shape "
            f"{(len(dataset), EMBEDDING_DIMENSION)}, got {embeddings.shape}."
        )
    logger.info("Extracted embeddings with shape %s", embeddings.shape)
    return embeddings


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings into two dimensions with cosine UMAP."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=DEFAULT_SEED,
    )
    return reducer.fit_transform(embeddings)
