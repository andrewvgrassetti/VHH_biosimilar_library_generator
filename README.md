# VHH Biosimilar Library Generator

A computational pipeline for designing producible VHH (nanobody) variant libraries with multi-objective scoring, codon optimisation, and LC-MS/MS barcoding support.

## Features

- **Stability Prediction** — Multi-factor scoring (disulfide bonds, aggregation propensity, charge balance, VHH hallmarks) with ESM-2 protein language model integration
- **Nativeness Scoring** — AbNatiV VQ-VAE assessment of VHH nativeness (how closely a sequence resembles natural camelid nanobody repertoires) — integrated as a mandatory scoring axis
- **Orthogonal Validation** — Independent scoring via germline consensus methods
- **Mutation Engine** — Ranks single-point mutations and generates combinatorial variant libraries with three strategies (exhaustive, random sampling, iterative anchor-and-explore refinement)
- **Developability Assessment** — PTM liability detection, surface hydrophobicity patches, clearance risk scoring
- **Codon Optimisation** — Organism-specific codon usage tables (E. coli, H. sapiens, P. pastoris, S. cerevisiae) with CAI calculation and restriction site flagging
- **Construct Builder** — Automated tag/linker assembly with DNA encoding
- **Peptide Barcoding** — Unique trypsin-cleavable barcodes for multiplexed LC-MS/MS screening
- **ESM-2 PLL Rescoring** — Protein language model validation (PyTorch & fair-esm included by default)

## Scoring & Ranking

The pipeline uses a produceability-oriented workflow built around two primary scoring axes:

1. **ESM-2 Stability** — Protein language model pseudo-log-likelihood scoring, calibrated to predicted melting temperature (Tm)
2. **AbNatiV Nativeness** — VQ-VAE model assessing how closely a VHH sequence resembles natural camelid nanobody repertoires

Variants are ranked by a composite score:

```
composite_score = w_stability × normalized_stability + w_nativeness × normalized_nativeness
```

Default weights: `w_stability = 0.70`, `w_nativeness = 0.30`. Surface hydrophobicity can optionally be enabled as an additional scoring axis.

## Installation

### Option A — Conda (recommended)

Conda is the recommended installation method. It installs `numba` and `llvmlite`
via `conda-forge`, which avoids macOS build failures related to `iomp5` and `cmake`.

**Quick start** — run the helper script:
```bash
bash setup_conda.sh
conda activate vhh_biosimilar
```

**Step-by-step:**
```bash
# 1. Create the conda environment
conda env create -f environment.yml

# 2. Activate it
conda activate vhh_biosimilar

# 3. Install the package in editable mode
pip install -e .

# 4. Download AbNatiV model weights
vhh-init
```

### Option B — pip / venv

> **macOS users:** this method requires Homebrew dependencies first:
> ```bash
> brew install cmake libomp
> ```

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -e .
vhh-init   # download AbNatiV model weights (cross-platform wrapper)
```

> **Note:** `vhh-init` is a cross-platform wrapper around `abnativ init`.
> On Linux/macOS, `abnativ init` also works.  On Windows, always use `vhh-init`.

### Optional: NanoMelt

[NanoMelt](https://github.com/NanoMelt/nanomelt) is a thermal-stability (Tm)
predictor that can be used as an alternative or complementary stability backend.
It is **not** required for the default workflow.

```bash
pip install ".[nanomelt]"
```

Once installed, select **nanomelt** or **both** as the stability backend in the
sidebar (or set the `VHH_STABILITY_BACKEND` environment variable).

## Usage

```bash
streamlit run app.py
```

### Backend & Device Selection

The sidebar exposes backend and device controls:

| Control | Options | Default |
|---------|---------|---------|
| **Stability backend** | `esm2`, `nanomelt`, `both` | `esm2` |
| **Nativeness backend** | `abnativ` | `abnativ` |
| **Device** | `auto`, `cpu`, `cuda` | `auto` |

The resolved device and predictor availability are shown below the controls.
If a selected backend is not installed, the app falls back gracefully and
displays a warning.

These can also be set via environment variables for headless / CI use:

```bash
export VHH_DEVICE=cuda
export VHH_STABILITY_BACKEND=both
streamlit run app.py
```

### GPU-Aware Workflow (AWS / Cloud Workstations)

On an AWS instance with NVIDIA GPU:

1. Install the CUDA-enabled PyTorch build (see [PyTorch docs](https://pytorch.org/get-started/locally/)).
2. Set `Device = cuda` in the sidebar, or export `VHH_DEVICE=cuda`.
3. For large libraries, use the `t33_650M` or `t36_3B` ESM-2 model tier in the sidebar.

The app auto-detects CUDA/MPS/CPU when device is set to `auto`.

### Position Policy

The **Mutation Selection** tab now includes a **Position Policy** section where
you can review the three-class classification (frozen / conservative / mutable)
for every IMGT position, and import/export policies as JSON or YAML files.

### Default Workflow

1. **Input & Analysis** — Paste a VHH sequence; the tool scores it for stability, nativeness, and surface hydrophobicity
2. **Mutation Selection** — Configure off-limit regions, review position policy, and rank single-point mutations by combined stability + nativeness score
3. **Library Generation** — Generate a combinatorial variant library from top-ranked mutations
4. **Library Results** — View, filter, and download the scored library with stability/nativeness/composite scores
5. **Barcoding** — Optionally assign LC-MS/MS barcodes
6. **Construct Builder** — Build codon-optimised DNA constructs with tags/linkers

## Running Tests

```bash
# If using conda:
conda activate vhh_biosimilar
pip install -e ".[dev]"
pytest

# If using pip/venv:
pip install -e ".[dev]"
pytest
```

## Project Structure

```
├── app.py                          # Streamlit application
├── pyproject.toml                  # Project configuration
├── data/
│   ├── human_vh_germlines.json     # Human VH germline sequences & position frequencies
│   ├── vhh_germlines.json          # Camelid VHH germline sequences
│   ├── tag_sequences.json          # Purification/display tag sequences
│   ├── barcode_pool.json           # Pre-validated barcode peptide pool
│   └── codon_tables/               # Organism-specific codon usage tables
├── vhh_library/
│   ├── sequence.py                 # VHHSequence with cached properties & fast mutation
│   ├── humanness.py                # Humanness scoring (HumAnnotator) — legacy, not used in main workflow
│   ├── stability.py                # Stability scoring with ESM-2 integration
│   ├── nativeness.py               # AbNatiV nativeness scoring (required)
│   ├── developability.py           # PTM, clearance risk, surface hydrophobicity
│   ├── orthogonal_scoring.py       # Consensus stability scorer
│   ├── mutation_engine.py          # Mutation ranking & library generation
│   ├── runtime_config.py           # Backend/device configuration
│   ├── position_policy.py          # Three-class position mutation policy
│   ├── position_classifier.py      # Rule-based IMGT position classifier
│   ├── codon_optimizer.py          # Codon optimisation (4 organisms)
│   ├── tags.py                     # Tag/linker construct assembly
│   ├── barcodes.py                 # LC-MS/MS peptide barcoding
│   ├── library_manager.py          # Session persistence
│   ├── visualization.py            # HTML sequence visualisation
│   ├── utils.py                    # Shared utilities (translation, pI, tryptic digest)
│   └── components/                 # Custom Streamlit components
└── tests/                          # Test suite (pytest)
```

## Key Optimisations

Compared to the original implementation, this rewrite addresses several inefficiencies:

1. **Cached properties** — `VHHSequence.regions`, `cdr_positions`, and `framework_positions` are computed once and cached via `functools.cached_property`
2. **Fast mutation path** — `VHHSequence.mutate()` creates mutant sequences without redundant validation, eliminating the overhead of re-validating known-good variants during library generation
3. **Consolidated utilities** — Shared `net_charge` and `isoelectric_point` functions in `utils.py` with tolerance-based bisection convergence (vs. fixed 1000 iterations)
4. **frozenset for amino acids** — O(1) membership checks instead of O(n) list lookups
5. **stdlib random** — `CodonOptimizer` uses `random.choices` instead of NumPy for the harmonised strategy
6. **Deduplicated hydrophobicity scale** — Single source of truth in `AA_PROPERTIES` used by both `barcodes.py` and `developability.py`
7. **Lazy scorer initialisation** — Optional scorers (ESM-2) are only loaded on demand
8. **`__slots__`** on `VHHSequence` for reduced memory footprint during large library generation

## Backward Compatibility & Migration Notes

The pipeline is being migrated to a new design system with explicit backend
selection, position-policy controls, and device-aware scoring. The migration is
**staged** — the legacy workflow continues to work unchanged.

### What changed

- **Sidebar** now includes a **Backend & Device** section. The default values
  (`esm2` / `abnativ` / `auto`) reproduce the old behaviour exactly.
- **Position Policy** expander in Tab 2 provides a read-only view of the
  three-class position classification and supports import/export. The legacy
  off-limit checkboxes and forbidden-substitution CSV upload still work and
  feed into the policy.
- **`RuntimeConfig`** (`vhh_library/runtime_config.py`) centralises device and
  backend settings.  `VHH_*` environment variables allow headless configuration
  without code changes.
- **`combined_score`** is a temporary compatibility output. New downstream code
  should consume axis scores (`stability_score`, `nativeness_score`) directly.

### What did not change

- All existing session-state keys, CSV/FASTA export columns, and library
  DataFrame columns are preserved.
- `pip install -e .` and `vhh-init` remain the only required install steps
  for the default workflow.
- `streamlit run app.py` remains the single entry point.

## Citations

This project relies on the following open-source libraries. If you use this tool in published work, please consider citing the relevant references:

| Library | Reference |
|---------|-----------|
| **ANARCI** | Dunbar, J. & Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics*, 32(2), 298–300. doi:[10.1093/bioinformatics/btv552](https://doi.org/10.1093/bioinformatics/btv552) |
| **HMMER** | Eddy, S.R. (2011). Accelerated profile HMM searches. *PLoS Comput Biol*, 7(10), e1002195. doi:[10.1371/journal.pcbi.1002195](https://doi.org/10.1371/journal.pcbi.1002195) |
| **ESM-2** (fair-esm) | Lin, Z. *et al.* (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130. doi:[10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) |
| **AbNatiV** | Kenlay, H. *et al.* (2024). AbNatiV: VQ-VAE-based assessment of antibody and nanobody nativeness for hit selection, humanisation, and engineering. *Nature Machine Intelligence*, 6, 1–11. doi:[10.1038/s42256-023-00778-3](https://doi.org/10.1038/s42256-023-00778-3) |
| **DNA Chisel** | Zulkower, V. & Rosser, S. (2020). DNA Chisel, a versatile sequence optimizer. *Bioinformatics*, 36(16), 4508–4509. doi:[10.1093/bioinformatics/btaa558](https://doi.org/10.1093/bioinformatics/btaa558) |
| **python-codon-tables** | Edinburgh Genome Foundry. Codon usage tables for Python, sourced from the Kazusa database. [GitHub](https://github.com/Edinburgh-Genome-Foundry/python-codon-tables) |
| **BioPython** | Cock, P.J.A. *et al.* (2009). Biopython: freely available Python tools for computational molecular biology. *Bioinformatics*, 25(11), 1422–1423. doi:[10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163) |
| **PyTorch** | Paszke, A. *et al.* (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*, 32. [Paper](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) |
| **NumPy** | Harris, C.R. *et al.* (2020). Array programming with NumPy. *Nature*, 585, 357–362. doi:[10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) |
| **SciPy** | Virtanen, P. *et al.* (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. doi:[10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) |
| **pandas** | McKinney, W. (2010). Data structures for statistical computing in Python. *Proc. of the 9th Python in Science Conf.*, 56–61. doi:[10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a) |
| **Matplotlib** | Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. doi:[10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55) |
| **Streamlit** | Streamlit Inc. [https://streamlit.io](https://streamlit.io) |