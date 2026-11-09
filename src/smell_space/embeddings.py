"""Penultimate OpenPOM embedding extraction and UMAP projection."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch.nn as nn

from .config import DEFAULT_SEED

logger = logging.getLogger(__name__)

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
    """Return the linear layer whose input is the 256-D POM embedding."""
    target_layer = None
    for module in reversed(list(model.model.modules())):
        if isinstance(module, nn.Linear):
            target_layer = module
            break
    if target_layer is None:
        raise RuntimeError("No nn.Linear layer found in the OpenPOM model.")
    return target_layer


def extract_embeddings(model: Any, dataset: Any) -> np.ndarray:
    """Extract the 256-D penultimate embedding for every dataset molecule."""
    py_model = model.model
    py_model.eval()
    captured: list[np.ndarray] = []
    target_layer = _find_penultimate_linear(model)

    def capture_input(
        _module: nn.Module,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        captured.append(inputs[0].detach().cpu().numpy())

    hook = target_layer.register_forward_hook(capture_input)
    try:
        model.predict(dataset)
    finally:
        hook.remove()

    if not captured:
        raise RuntimeError("Embedding extraction produced no activations.")
    embeddings = np.concatenate(captured, axis=0)[: len(dataset)]
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
