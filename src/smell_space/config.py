"""Project paths, filenames, and environment-driven configuration.

All filesystem locations are resolved from the repository root by default and
can be overridden with environment variables. Relative environment values are
interpreted relative to the repository root, which makes the project portable
across local machines and hosted notebook environments.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILENAME = "curated_GS_LF_merged_4983.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/ARY2260/openpom/main/"
    "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)

EMBEDDINGS_FILENAME = "openpom_embeddings_256d.npy"
METADATA_FILENAME = "molecules_metadata_umap.csv"
SIMILARITY_FILENAME = "openpom_cosine_similarity_float32.npy"
UMAP_FIGURE_FILENAME = "umap_odor_space.png"

DEFAULT_SEED = 42
DEFAULT_TOP_K = 5
DEFAULT_POM_THRESHOLD = 0.88
DEFAULT_TANIMOTO_THRESHOLD = 0.32


def _env_path(variable: str, default: Path) -> Path:
    """Return a resolved path from an environment variable or ``default``."""
    value = os.environ.get(variable)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_DIR = _env_path("SMELL_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = _env_path("SMELL_MODELS_DIR", PROJECT_ROOT / "models")
RESULTS_DIR = _env_path("SMELL_RESULTS_DIR", PROJECT_ROOT / "results")

DATASET_PATH = _env_path("SMELL_DATASET_PATH", DATA_DIR / DATASET_FILENAME)
DISK_DATASET_DIR = _env_path("SMELL_CACHE_DIR", DATA_DIR / "disk_dataset")
MODEL_DIR = _env_path("SMELL_MODEL_DIR", MODELS_DIR / "openpom_scaffold")

EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / EMBEDDINGS_FILENAME
METADATA_PATH = EMBEDDINGS_DIR / METADATA_FILENAME
SIMILARITY_PATH = EMBEDDINGS_DIR / SIMILARITY_FILENAME

FIGURES_DIR = RESULTS_DIR / "figures"
UMAP_FIGURE_PATH = FIGURES_DIR / UMAP_FIGURE_FILENAME
RETRIEVAL_DIR = RESULTS_DIR / "retrieval"
CROSS_SCAFFOLD_CSV_PATH = RETRIEVAL_DIR / "cross_scaffold_pairs.csv"
CROSS_SCAFFOLD_FIGURE_PATH = FIGURES_DIR / "cross_scaffold_pairs.png"


def ensure_directories() -> None:
    """Create the writable directories used by the project."""
    for directory in (
        DATA_DIR,
        MODELS_DIR,
        MODEL_DIR,
        EMBEDDINGS_DIR,
        FIGURES_DIR,
        RETRIEVAL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
