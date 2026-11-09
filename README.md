# Smell Space Explorer

Smell Space Explorer predicts human olfactory descriptors from molecular
structure with **OpenPOM**, a multi-task message-passing neural network built on
DeepChem, DGL-LifeSci, and RDKit. It maps each molecule to a 256-dimensional
principal odor embedding, projects the curated ~4,983-molecule odor dataset with
UMAP, and provides nearest-neighbor retrieval plus a cross-scaffold discovery
analysis in a Streamlit app. The goal is exploratory: find molecules that are
structurally dissimilar (low Morgan/Tanimoto similarity) but perceptually close
(high cosine similarity in OpenPOM space).

> **Refactor status:** Phase 2 (real model inference) is complete. Unseen
> molecules now run through the trained MPNN and receive probabilities for all
> 138 descriptors plus a 256-D embedding; no fallback predictions are used.
> Optimization and the test suite are still pending. See
> [Limitations](#limitations).

## Screenshots

> **UMAP screenshot TODO:** generate `results/figures/umap_odor_space.png` with
> `python scripts/extract_embeddings.py`, then replace this note with the image.
>
> **App screenshot TODO:** add `results/figures/streamlit_app.png` once the app
> is running against a real trained checkpoint. Do not commit a fabricated
> screenshot.

## Project structure

```text
smell-space-explorer/
  app/streamlit_app.py
  data/                         # downloaded CSV and DeepChem cache (ignored)
  models/                       # checkpoints (ignored)
  notebooks/demo.ipynb          # thin, output-free demo
  results/
    embeddings/                 # .npy embeddings and metadata (ignored)
    figures/                    # small UMAP/discovery PNGs used by this README
    retrieval/                  # generated CSVs
  scripts/
    train.py
    extract_embeddings.py
    find_cross_scaffold_pairs.py
  src/smell_space/
    config.py                   # paths and environment overrides
    data.py                     # download, column handling, featurization
    model.py                    # OpenPOM construction and training
    embeddings.py               # 256-D extraction and UMAP
    inference.py                # single-molecule inference
    similarity.py               # cosine retrieval and scaffold discovery
  tests/
```

## Installation

The scientific stack is pinned for **Python 3.10** (DeepChem 2.8 does not
support Python 3.12+). A virtual environment is strongly recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch first, so pip does not pull a full CUDA build.
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e . --no-deps
```

The install was not executed inside the refactoring environment because this
machine only has Python 3.12/3.15; the repository targets Python 3.10. Treat the
pins as a reproducible starting point and verify them on your platform.

### DGL GraphBolt note

The original Colab notebook overwrote
`dgl/graphbolt/__init__.py` with a stub that disables GraphBolt. That edit is
**not** reproduced here. OpenPOM uses DGL's message-passing APIs, not GraphBolt,
and the pinned DGL 2.1.0 wheel should provide a matching library. If
`import dgl` fails with "Cannot load Graphbolt C++ library", the DGL wheel and
PyTorch build do not match: reinstall a DGL 2.1.0 wheel built for your
torch/CUDA combination. Do not patch installed package files by hand. If you
must reuse the old stub as a last resort, know that it disables GraphBolt
globally and is unsupported; it is not needed for the code in this repository.

## Data

The curated CSV is downloaded automatically from the OpenPOM repository on
first use and stored at `data/curated_GS_LF_merged_4983.csv`. It contains 4,983
molecules and 138 binary odor descriptors.

Paths can be overridden without editing code:

| Variable | Default |
| --- | --- |
| `SMELL_DATA_DIR` | `<repo>/data` |
| `SMELL_DATASET_PATH` | `<data>/curated_GS_LF_merged_4983.csv` |
| `SMELL_CACHE_DIR` | `<data>/disk_dataset` |
| `SMELL_MODELS_DIR` | `<repo>/models` |
| `SMELL_MODEL_DIR` | `<models>/openpom_scaffold` |
| `SMELL_RESULTS_DIR` | `<repo>/results` |

Relative values are resolved against the repository root.

## Train

```bash
python scripts/train.py --epochs 25
```

Training writes DeepChem checkpoints under `models/openpom_scaffold/`. Use
`--resume` to continue from the most recent checkpoint. Training on CPU is
expensive; GPU is not required by the code but is supported by the underlying
stack. The training schedule and model hyperparameters match the original
notebook.

## Build embeddings and the UMAP map

Generate artifacts only after a checkpoint exists:

```bash
python scripts/extract_embeddings.py
```

This writes:

- `results/embeddings/openpom_embeddings_256d.npy` — shape `(n_molecules, 256)`
- `results/embeddings/molecules_metadata_umap.csv`
- `results/figures/umap_odor_space.png`
- `results/figures/umap_odor_space.html`

## Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Known dataset entries show their curated labels, UMAP position, and nearest
neighbors. For an unseen SMILES, the app runs the trained MPNN, displays the top
predicted descriptors and radar chart, and retrieves nearest neighbors using
that query's own 256-D embedding. If no checkpoint exists, the app shows an
instruction to run `scripts/train.py`; it never displays fabricated
predictions.

### Single-molecule inference API

```python
from smell_space.inference import predict_odor_profile

result = predict_odor_profile("COc1cc(C=O)ccc1O", top_n=5)
print(result.top_predictions)          # five highest probabilities
print(len(result.probabilities))      # all 138, sorted descending
print(result.embedding.shape)         # (256,)
```

The model, featurizer, and task names are loaded lazily once per process and
reused. Invalid SMILES raise `ValueError`; a missing checkpoint raises
`FileNotFoundError` with the training command. Re-run
`scripts/extract_embeddings.py` whenever you train a new checkpoint so the
dataset embeddings and the query model come from the same weights.

## Cross-scaffold discovery

```bash
python scripts/find_cross_scaffold_pairs.py --num-pairs 5
```

The default criteria are OpenPOM cosine **> 0.88** and Morgan (ECFP4) Tanimoto
**< 0.32**; both are command-line parameters. Results are written to
`results/retrieval/cross_scaffold_pairs.csv` and rendered to
`results/figures/cross_scaffold_pairs.png`.

## Evaluation

> **TODO:** Fill in every value below from the real training/evaluation run.
> These are placeholders, not reported results. Do not merge invented numbers.

| Split | Macro AUROC | Micro AUROC | Notes |
| --- | --- | --- | --- |
| Test | TODO | TODO | TODO: populated by `scripts/train.py` |
| Validation | TODO | TODO | TODO: early-stopping split |
| Train (subsample) | TODO | TODO | TODO: for monitoring only |

Per-descriptor AUROC will be written to
`results/metrics/per_descriptor_auroc.csv` by the training script once the
evaluation pipeline is added. The original notebook's error-analysis cell used
hand-entered AUROC values; those are intentionally not carried into this
repository.

## Tests

```bash
pytest
```

Tests cover invalid SMILES handling, vectorized Tanimoto equivalence, top-k
retrieval, the 138-task column contract, and embedding shape. Model-dependent
tests are marked `slow` and skip when no checkpoint or dataset is available.

## Model weights and embeddings

`models/*.pt`, `results/embeddings/*.npy`, and generated CSVs are git-ignored
because they are large build artifacts. Publish them through a GitHub Release
and document the download command, or track them with Git LFS if the host
supports it:

```bash
git lfs install
git lfs track "results/embeddings/*.npy" "models/**/*.pt"
git add .gitattributes
```

The trained checkpoint and the 256-D embedding matrix are not included in this
repository.

## Reproducibility

- The UMAP projection uses `n_neighbors=15`, `min_dist=0.1`, cosine metric, and
  `random_state=42`.
- Exact numerical reproduction also depends on the pinned PyTorch/DGL/DeepChem
  versions and CPU/GPU kernels.
- Deterministic training, seeded data splits, and CSV-hash-keyed featurization
  caches are tracked as follow-up work in the optimization phase.

## Credits and citations

This project builds directly on:

- **OpenPOM** — [ARY2260/openpom](https://github.com/ARY2260/openpom).
  Lee, B. K. et al. "A principal odor map unifies diverse tasks in olfactory
  perception." *Science* 381, 999–1006 (2023).
  [doi:10.1126/science.ade4401](https://doi.org/10.1126/science.ade4401)
- **DeepChem** — [deepchem.io](https://deepchem.io/).
- **DGL-LifeSci** — [awslabs/dgl-lifesci](https://github.com/awslabs/dgl-lifesci).
- **RDKit** — [rdkit.org](https://www.rdkit.org/).
- **UMAP** — McInnes, L., Healy, J., & Melville, J. "UMAP: Uniform Manifold
  Approximation and Projection for Dimension Reduction." arXiv:1802.03426
  (2018).
- **GoodScents / Leffingwell odor data** — the curated OpenPOM dataset merges
  records derived from the Good Scents Company and Leffingwell & Associates
  databases. Refer to the OpenPOM data documentation for provenance and terms.

## License

This repository's original code is released under the MIT License; see
[LICENSE](LICENSE). **TODO:** verify that the OpenPOM code and model weights and
the GoodScents/Leffingwell-derived dataset permit redistribution and commercial
use under compatible terms before publishing artifacts.

## Limitations

Predictions are produced by a model trained on roughly 5,000 molecules and are
**not experimental ground truth**. Odor perception is context-dependent,
subject-dependent, and can be influenced by concentration and mixtures. The
training labels are aggregated from heterogeneous sources and are sparse;
many descriptors have few positives. A high cosine similarity in OpenPOM space
is a model-based hypothesis, not proof that two molecules smell alike. Do not
use this project for safety, toxicity, allergenicity, regulatory, or
formulation decisions without independent experimental validation.
