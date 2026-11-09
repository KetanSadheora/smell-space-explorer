"""Streamlit interface for OpenPOM odor-space exploration.

Run from the repository root with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
from typing import Any

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
from smell_space.similarity import (
    compute_cosine_similarity,
    top_k_neighbor_indices,
)

logger = logging.getLogger(__name__)

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
def load_artifacts() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load embeddings, metadata, labels, and the cached cosine matrix once."""
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

    return embeddings, metadata, dataset_df, task_columns


@st.cache_data(show_spinner=False)
def embedding_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute and cache the pairwise cosine matrix for this session."""
    return compute_cosine_similarity(embeddings).astype(np.float32)


def active_descriptors(row: np.ndarray, task_columns: list[str]) -> list[str]:
    """Return descriptor names active for one label row."""
    return [
        task_columns[index]
        for index in np.flatnonzero(row > 0.5)
    ]


def radar_figure(descriptors: list[str]) -> go.Figure:
    """Build a radar chart of positive descriptors."""
    figure = go.Figure(
        data=go.Scatterpolar(
            r=[1.0] * len(descriptors),
            theta=descriptors,
            fill="toself",
            line={"color": "#ab63fa"},
        )
    )
    figure.update_layout(
        polar={"radialaxis": {"visible": False, "range": [0, 1.05]}},
        showlegend=False,
        height=320,
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


def main() -> None:
    """Render the Streamlit application."""
    st.title("👃 Smell Space Explorer (OpenPOM)")
    st.markdown(
        "### Predicting human olfactory perception from molecular graph structure"
    )

    try:
        embeddings, metadata, dataset_df, task_columns = load_artifacts()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info(
            "Train the model with `python scripts/train.py`, then build the "
            "artifacts with `python scripts/extract_embeddings.py`."
        )
        st.stop()

    similarity = embedding_similarity(embeddings)

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
                dataset_df[task_columns].fillna(0).to_numpy()[match_index],
                task_columns,
            )[:8]
            if descriptors:
                st.plotly_chart(
                    radar_figure(descriptors),
                    use_container_width=True,
                )
        else:
            match_index = None
            st.info(
                "Unseen molecule. This molecule is not in the curated dataset; "
                "the checkpoint-backed predictor is not yet connected in this "
                "repository build."
            )

    st.markdown("---")
    tab_neighbors, tab_map = st.tabs(
        ["🔍 Perceptual Nearest Neighbors", "🗺️ Latent Odor Space Map"]
    )

    with tab_neighbors:
        st.subheader("Top 'smells-like' neighbors in the 256-D OpenPOM space")
        if match_index is None:
            st.info(
                "Nearest-neighbor retrieval for unseen molecules requires their "
                "model embedding and will be available with model-backed inference."
            )
        else:
            neighbor_indices = top_k_neighbor_indices(
                similarity[match_index],
                DEFAULT_TOP_K,
                exclude_index=match_index,
            )
            cards: list[dict[str, Any]] = []
            for rank, neighbor_index in enumerate(neighbor_indices, start=1):
                neighbor = metadata.iloc[neighbor_index]
                cards.append(
                    {
                        "Rank": rank,
                        "Neighbor name": neighbor.get("name", "Unknown"),
                        "Neighbor SMILES": neighbor[smiles_column],
                        "Dominant family": neighbor.get(
                            "dominant_family", "Unknown"
                        ),
                        "OpenPOM cosine": f"{similarity[match_index, neighbor_index]:.4f}",
                        "Known notes": neighbor.get("active_descriptors", "N/A"),
                    }
                )
            st.dataframe(pd.DataFrame(cards), use_container_width=True)

            st.write("**Top 3 neighbor structures:**")
            columns = st.columns(3)
            for rank, neighbor_index in enumerate(neighbor_indices[:3]):
                with columns[rank]:
                    render_molecule(
                        str(metadata.iloc[neighbor_index][smiles_column]),
                        caption=f"Rank {rank + 1}",
                    )

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
        st.plotly_chart(figure, use_container_width=True)


if __name__ == "__main__":
    main()
