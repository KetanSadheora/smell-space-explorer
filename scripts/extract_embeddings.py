"""Extract 256-D OpenPOM embeddings, UMAP coordinates, and figure artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smell_space.config import (
    EMBEDDINGS_PATH,
    METADATA_PATH,
    MODEL_DIR,
    UMAP_FIGURE_PATH,
    ensure_directories,
)
from smell_space.data import find_smiles_column, get_dataset
from smell_space.embeddings import (
    extract_embeddings,
    get_dominant_family,
    run_umap,
)
from smell_space.model import build_model
from smell_space.similarity import (
    compute_cosine_similarity,
    save_cosine_similarity,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line extraction options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"Directory containing the trained checkpoint (default: {MODEL_DIR}).",
    )
    return parser.parse_args()


def save_umap_figure(metadata: pd.DataFrame, output_path: Path) -> None:
    """Save a static UMAP scatter plot colored by dominant odor family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 7))
    for family, group in metadata.groupby("dominant_family"):
        axis.scatter(
            group["umap_x"],
            group["umap_y"],
            s=6,
            alpha=0.75,
            label=family,
        )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("OpenPOM latent odor space (256-D embeddings via UMAP)")
    axis.legend(markerscale=3, fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Extract embeddings from an existing checkpoint and persist metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ensure_directories()

    dataset, task_columns, raw_dataframe = get_dataset()
    from openpom.utils.data_utils import get_class_imbalance_ratio

    model = build_model(
        len(task_columns),
        get_class_imbalance_ratio(dataset),
        args.model_dir,
    )
    checkpoints = model.get_checkpoints()
    if not checkpoints:
        raise FileNotFoundError(
            "No trained checkpoint found. Run scripts/train.py first."
        )
    model.restore(checkpoints[0])

    embeddings = extract_embeddings(model, dataset)
    if embeddings.ndim != 2 or embeddings.shape[1] != 256:
        raise RuntimeError(
            "Expected an (n_molecules, 256) embedding matrix, got "
            f"{embeddings.shape}."
        )

    np.save(EMBEDDINGS_PATH, embeddings)
    logger.info("Saved embeddings %s to %s", embeddings.shape, EMBEDDINGS_PATH)

    similarity = compute_cosine_similarity(embeddings)
    save_cosine_similarity(similarity)
    logger.info("Saved cosine similarity cache %s", similarity.shape)

    umap_projection = run_umap(embeddings)
    smiles_column = find_smiles_column(raw_dataframe)
    smiles = [str(value) for value in dataset.ids]

    families: list[str] = []
    active_descriptors: list[str] = []
    for row in np.asarray(dataset.y):
        active_tasks = [
            task_columns[index] for index in np.where(row == 1.0)[0]
        ]
        families.append(get_dominant_family(active_tasks))
        active_descriptors.append(
            ", ".join(active_tasks[:5]) if active_tasks else "None"
        )

    names = (
        raw_dataframe["name"].astype(str).tolist()
        if "name" in raw_dataframe.columns
        else smiles
    )
    metadata = pd.DataFrame(
        {
            "smiles": smiles,
            "name": names,
            "dominant_family": families,
            "active_descriptors": active_descriptors,
            "umap_x": umap_projection[:, 0],
            "umap_y": umap_projection[:, 1],
        }
    )
    metadata.to_csv(METADATA_PATH, index=False)
    logger.info("Saved metadata to %s", METADATA_PATH)

    save_umap_figure(metadata, UMAP_FIGURE_PATH)
    logger.info("Saved UMAP figure to %s", UMAP_FIGURE_PATH)

    try:
        import plotly.express as px

        figure = px.scatter(
            metadata,
            x="umap_x",
            y="umap_y",
            color="dominant_family",
            hover_name="name",
            hover_data=["smiles", "active_descriptors"],
            title="OpenPOM Primary Olfactory Space",
            template="plotly_white",
            width=950,
            height=650,
        )
        html_path = UMAP_FIGURE_PATH.with_suffix(".html")
        figure.write_html(html_path)
        logger.info("Saved interactive UMAP to %s", html_path)
    except ImportError:
        logger.warning("Plotly is unavailable; skipped the interactive map.")

    logger.info(
        "Molecule distribution by family:\n%s",
        metadata["dominant_family"].value_counts().to_string(),
    )
    logger.debug("SMILES column used for source data: %s", smiles_column)


if __name__ == "__main__":
    main()
