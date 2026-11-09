"""Streamlit interface for OpenPOM odor-space exploration.

Run from the repository root with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from smell_space.config import (
    DATASET_PATH,
    DEFAULT_TOP_K,
    EMBEDDINGS_PATH,
    METADATA_PATH,
)
from smell_space.data import get_task_columns, load_curated_dataframe
from smell_space.inference import (
    OdorPrediction,
    get_model,
    predict_odor_profile,
)
from smell_space.similarity import (
    cosine_similarity_to_query,
    load_or_compute_cosine_similarity,
    top_k_neighbor_indices,
)

st.set_page_config(
    page_title="Smell Space Explorer",
    layout="wide",
    page_icon="👃",
)

PRESETS: dict[str, str] = {
    "Vanillin (Sweet/Creamy)": "COc1cc(C=O)ccc1O",
    "Eugenol (Clove/Spicy)": "COc1cc(CC=C)ccc1O",
    "Citral (Lemon/Citrus)": "CC(C)=CCCC(C)=CC=O",
    "Geosmin (Earthy/Beet)": "CC1CCCC2C1(CCCC2(C)O)C",
    "Muscone (Animalic/Musk)": "CC1CCCCCCCCCCCCC(=O)C1",
    "Custom SMILES": "",
}


@st.cache_resource(show_spinner="Loading OpenPOM embeddings and dataset...")
def load_artifacts() -> tuple[
    np.ndarray,
    pd.DataFrame,
    list[str],
    np.ndarray,
]:
    """Load embeddings, metadata, labels, and task names once per process."""
    missing = [
        str(path)
        for path in (EMBEDDINGS_PATH, METADATA_PATH, DATASET_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts:\n- " + "\n- ".join(missing)
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    dataset_df = load_curated_dataframe(DATASET_PATH)
    task_columns = get_task_columns(dataset_df)
    labels = dataset_df[task_columns].fillna(0).to_numpy(dtype=float)

    if not (len(embeddings) == len(metadata) == len(labels)):
        raise ValueError(
            "Embedding, metadata, and dataset row counts do not match: "
            f"{len(embeddings)}, {len(metadata)}, {len(labels)}."
        )

    return embeddings, metadata, task_columns, labels


@st.cache_data(show_spinner="Loading cached cosine similarity matrix...")
def load_similarity_matrix() -> np.ndarray:
    """Load the float32 cosine cache, computing it once only if absent."""
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    return load_or_compute_cosine_similarity(embeddings)


@st.cache_resource(show_spinner="Loading OpenPOM model...")
def load_model_resource() -> Any:
    """Load and cache the trained OpenPOM model for this Streamlit process."""
    return get_model()


def active_descriptors(row: np.ndarray, task_columns: list[str]) -> list[str]:
    """Return descriptor names active in one curated label row."""
    return [task_columns[index] for index in np.flatnonzero(row > 0.5)]


def radar_figure(
    labels: Sequence[str],
    values: Sequence[float],
) -> go.Figure:
    """Build a radar chart from descriptor labels and real probabilities."""
    if len(labels) != len(values):
        raise ValueError("Radar labels and values must have equal length.")
    figure = go.Figure(
        data=go.Scatterpolar(
            r=[float(value) for value in values],
            theta=list(labels),
            fill="toself",
            line={"color": "#ab63fa"},
        )
    )
    figure.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 1.0],
                "tickformat": ".2f",
            }
        },
        showlegend=False,
        height=350,
        margin={"l": 40, "r": 40, "t": 20, "b": 20},
    )
    return figure


def render_molecule(smiles: str, caption: str) -> None:
    """Render a 2D structure when RDKit can parse the SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.warning("This SMILES could not be rendered.")
        return
    st.image(Draw.MolToImage(molecule, size=(320, 240)), caption=caption)


def neighbor_table(
    metadata: pd.DataFrame,
    smiles_column: str,
    scores: np.ndarray,
    neighbor_indices: np.ndarray,
) -> pd.DataFrame:
    """Build the nearest-neighbor display table."""
    rows: list[dict[str, Any]] = []
    for rank, neighbor_index in enumerate(neighbor_indices, start=1):
        neighbor = metadata.iloc[neighbor_index]
        rows.append(
            {
                "Rank": rank,
                "Neighbor name": neighbor.get("name", "Unknown"),
                "Neighbor SMILES": neighbor[smiles_column],
                "Dominant family": neighbor.get("dominant_family", "Unknown"),
                "OpenPOM cosine": f"{scores[neighbor_index]:.4f}",
                "Known notes": neighbor.get("active_descriptors", "N/A"),
            }
        )
    return pd.DataFrame(rows)


def render_neighbor_structures(
    metadata: pd.DataFrame,
    smiles_column: str,
    neighbor_indices: np.ndarray,
) -> None:
    """Render the top three neighbor structures."""
    st.write("**Top 3 neighbor structures:**")
    columns = st.columns(3)
    for rank, neighbor_index in enumerate(neighbor_indices[:3]):
        with columns[rank]:
            render_molecule(
                str(metadata.iloc[neighbor_index][smiles_column]),
                caption=f"Rank {rank + 1}",
            )


def main() -> None:
    """Render the Streamlit application."""
    st.title("👃 Smell Space Explorer (OpenPOM)")
    st.markdown(
        "### Predicting human olfactory perception from molecular graph structure"
    )

    try:
        embeddings, metadata, task_columns, labels = load_artifacts()
        similarity = load_similarity_matrix()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info(
            "Train the model with `python scripts/train.py`, then build the "
            "artifacts with `python scripts/extract_embeddings.py`."
        )
        st.stop()

    with st.sidebar:
        st.header("Molecule input")
        selected_preset = st.selectbox(
            "Choose an odorant or custom input:",
            list(PRESETS),
        )
        if selected_preset == "Custom SMILES":
            query_smiles = st.text_input(
                "Enter custom SMILES:",
                value="COc1cc(C=O)ccc1O",
            )
        else:
            query_smiles = PRESETS[selected_preset]
            st.text_input("SMILES string:", value=query_smiles, disabled=True)

    query_smiles = query_smiles.strip()
    molecule = Chem.MolFromSmiles(query_smiles) if query_smiles else None
    if molecule is None:
        st.error("Invalid SMILES structure. Please enter a valid chemical.")
        st.stop()

    smiles_column = "smiles" if "smiles" in metadata.columns else "nonStereoSMILES"
    matched_rows = metadata[metadata[smiles_column] == query_smiles]
    match_index = int(matched_rows.index[0]) if len(matched_rows) else None

    prediction: OdorPrediction | None = None
    query_embedding: np.ndarray | None = None
    query_scores: np.ndarray | None = None
    excluded_index: int | None = None

    column_structure, column_prediction = st.columns([1, 1.5])
    with column_structure:
        st.subheader("2D Molecular Structure")
        render_molecule(query_smiles, "RDKit-rendered molecular graph")

    with column_prediction:
        st.subheader("Olfactory Profile")
        if match_index is not None:
            row = metadata.iloc[match_index]
            st.success(f"Verified in curated dataset (index {match_index})")
            st.write(f"**Dominant family:** `{row.get('dominant_family', 'Unknown')}`")
            st.write(
                "**Known ground-truth notes:** "
                f"`{row.get('active_descriptors', 'N/A')}`"
            )
            descriptors = active_descriptors(
                labels[match_index],
                task_columns,
            )[:8]
            if descriptors:
                st.plotly_chart(
                    radar_figure(descriptors, [1.0] * len(descriptors)),
                    use_container_width=True,
                )
            query_embedding = embeddings[match_index]
            query_scores = similarity[match_index]
            excluded_index = match_index
        else:
            st.info(
                "Unseen molecule: generating a checkpoint-backed OpenPOM "
                "prediction. These are model estimates, not experimental labels."
            )
            try:
                load_model_resource()
                with st.spinner("Running OpenPOM inference..."):
                    prediction = predict_odor_profile(query_smiles, top_n=8)
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                st.error(str(error))
                st.info(
                    "Train a checkpoint with `python scripts/train.py`; the app "
                    "never substitutes invented predictions."
                )
            else:
                st.success("Real OpenPOM prediction (not experimental ground truth)")
                st.caption(f"Canonical SMILES: `{prediction.canonical_smiles}`")
                top_pairs = prediction.top_predictions
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Descriptor": [name for name, _ in top_pairs],
                            "Probability": [score for _, score in top_pairs],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.plotly_chart(
                    radar_figure(
                        [name for name, _ in top_pairs],
                        [score for _, score in top_pairs],
                    ),
                    use_container_width=True,
                )
                with st.expander("All 138 descriptor probabilities"):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Descriptor": [
                                    name for name, _ in prediction.probabilities
                                ],
                                "Probability": [
                                    score for _, score in prediction.probabilities
                                ],
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                query_embedding = prediction.embedding
                query_scores = cosine_similarity_to_query(
                    prediction.embedding,
                    embeddings,
                )

    neighbor_indices: np.ndarray | None = None
    if query_scores is not None:
        neighbor_indices = top_k_neighbor_indices(
            query_scores,
            DEFAULT_TOP_K,
            exclude_index=excluded_index,
        )

    st.markdown("---")
    tab_neighbors, tab_map = st.tabs(
        ["🔍 Perceptual Nearest Neighbors", "🗺️ Latent Odor Space Map"]
    )

    with tab_neighbors:
        st.subheader("Top 'smells-like' neighbors in the 256-D OpenPOM space")
        if neighbor_indices is None or query_scores is None:
            st.info(
                "Nearest neighbors become available when a real query embedding "
                "can be computed. No placeholder neighbors are shown."
            )
        else:
            st.dataframe(
                neighbor_table(
                    metadata,
                    smiles_column,
                    query_scores,
                    neighbor_indices,
                ),
                use_container_width=True,
                hide_index=True,
            )
            render_neighbor_structures(metadata, smiles_column, neighbor_indices)

    with tab_map:
        st.subheader("Position in the 2D latent olfactory space")
        figure = px.scatter(
            metadata,
            x="umap_x",
            y="umap_y",
            color="dominant_family",
            hover_name="name",
            hover_data=[smiles_column, "active_descriptors"],
            height=520,
            opacity=0.75,
        )
        if match_index is not None:
            figure.add_scatter(
                x=[metadata.iloc[match_index]["umap_x"]],
                y=[metadata.iloc[match_index]["umap_y"]],
                mode="markers+text",
                marker={"color": "black", "size": 14, "symbol": "star"},
                text=["QUERY"],
                textposition="top center",
                name="Query molecule",
            )
        elif neighbor_indices is not None:
            neighbors = metadata.iloc[neighbor_indices]
            figure.add_scatter(
                x=neighbors["umap_x"],
                y=neighbors["umap_y"],
                mode="markers",
                marker={
                    "color": "black",
                    "size": 10,
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 1},
                },
                text=[f"Neighbor {rank}" for rank in range(1, len(neighbors) + 1)],
                name="Nearest neighbors",
            )
            st.caption(
                "An unseen query has no UMAP coordinate, so its nearest "
                "neighbors are highlighted instead."
            )
        st.plotly_chart(figure, use_container_width=True)


if __name__ == "__main__":
    main()
