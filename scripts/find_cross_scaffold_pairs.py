"""Find and render cross-scaffold odor-similar molecule pairs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from smell_space.config import (
    CROSS_SCAFFOLD_CSV_PATH,
    CROSS_SCAFFOLD_FIGURE_PATH,
    DEFAULT_POM_THRESHOLD,
    DEFAULT_TANIMOTO_THRESHOLD,
    EMBEDDINGS_PATH,
    METADATA_PATH,
)
from smell_space.data import find_smiles_column
from smell_space.similarity import (
    compute_cosine_similarity,
    evaluate_cross_scaffold_pairs,
    morgan_fingerprints,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line discovery options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--pom-threshold", type=float, default=DEFAULT_POM_THRESHOLD)
    parser.add_argument(
        "--tanimoto-threshold",
        type=float,
        default=DEFAULT_TANIMOTO_THRESHOLD,
    )
    return parser.parse_args()


def save_pair_figure(pairs: pd.DataFrame, output_path: Path) -> None:
    """Render the discovered pairs side by side as a PNG grid."""
    molecules = []
    legends = []
    for _, row in pairs.iterrows():
        molecule_a = Chem.MolFromSmiles(row["mol_a_smiles"])
        molecule_b = Chem.MolFromSmiles(row["mol_b_smiles"])
        if molecule_a is None or molecule_b is None:
            continue
        molecules.extend([molecule_a, molecule_b])
        legends.extend(
            [
                f"A: {row['mol_a_name']}",
                (
                    f"B: {row['mol_b_name']}\n"
                    f"POM: {row['pom_similarity']:.3f} | "
                    f"Tanimoto: {row['tanimoto_similarity']:.3f}"
                ),
            ]
        )

    if not molecules:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(420, 300),
        legends=legends,
    )
    image.save(output_path)


def main() -> None:
    """Load saved artifacts, rank cross-scaffold pairs, and write results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Embedding artifacts are missing. Run scripts/extract_embeddings.py."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    if len(embeddings) != len(metadata):
        raise ValueError("Embeddings and metadata have different row counts.")

    smiles_column = find_smiles_column(metadata)
    fingerprints = morgan_fingerprints(metadata[smiles_column].astype(str).tolist())
    similarity = compute_cosine_similarity(embeddings)
    pairs = evaluate_cross_scaffold_pairs(
        metadata,
        similarity,
        fingerprints,
        num_pairs=args.num_pairs,
        pom_threshold=args.pom_threshold,
        tanimoto_threshold=args.tanimoto_threshold,
    )

    if pairs.empty:
        logger.info("No pairs passed the current thresholds; nothing to save.")
        return

    CROSS_SCAFFOLD_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(CROSS_SCAFFOLD_CSV_PATH, index=False)
    logger.info("Saved %d pairs to %s", len(pairs), CROSS_SCAFFOLD_CSV_PATH)

    logger.info(
        "Top pairs:\n%s",
        pairs[
            ["mol_a_name", "mol_b_name", "pom_similarity", "tanimoto_similarity"]
        ].to_string(index=False),
    )
    save_pair_figure(pairs, CROSS_SCAFFOLD_FIGURE_PATH)
    logger.info("Saved pair figure to %s", CROSS_SCAFFOLD_FIGURE_PATH)


if __name__ == "__main__":
    main()
