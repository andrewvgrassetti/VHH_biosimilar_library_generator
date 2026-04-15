# VHH Biosimilar Library Generator

A computational pipeline for designing humanised VHH (nanobody) variant libraries with multi-objective scoring, codon optimisation, and LC-MS/MS barcoding support.

## Features

- **Humanness Scoring** — Compares framework regions to human VH germlines using identity and position-frequency matrices
- **Stability Prediction** — Multi-factor scoring (disulfide bonds, aggregation propensity, charge balance, VHH hallmarks) with ESM-2 protein language model integration
- **Orthogonal Validation** — Independent scoring via Human String Content (HSC) and germline consensus methods
- **Mutation Engine** — Ranks single-point mutations and generates combinatorial variant libraries with three strategies (exhaustive, random sampling, iterative anchor-and-explore refinement)
- **Developability Assessment** — PTM liability detection, surface hydrophobicity patches, clearance risk scoring
- **Codon Optimisation** — Organism-specific codon usage tables (E. coli, H. sapiens, P. pastoris, S. cerevisiae) with CAI calculation and restriction site flagging
- **Construct Builder** — Automated tag/linker assembly with DNA encoding
- **Peptide Barcoding** — Unique trypsin-cleavable barcodes for multiplexed LC-MS/MS screening
- **ESM-2 PLL Rescoring** — Protein language model validation (PyTorch & fair-esm included by default)

## Installation

Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

Install the package in editable mode (includes ESM-2 / PyTorch by default):
```bash
pip install -e .
```

## Usage

```bash
streamlit run app.py
```

## Running Tests

```bash
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
│   ├── humanness.py                # Humanness scoring (HumAnnotator)
│   ├── stability.py                # Stability scoring with ESM-2 integration
│   ├── developability.py           # PTM, clearance risk, surface hydrophobicity
│   ├── orthogonal_scoring.py       # HSC, consensus scorers
│   ├── mutation_engine.py          # Mutation ranking & library generation
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
3. **Precomputed germline frameworks** — `HumAnnotator` concatenates germline framework strings once at init instead of on every `score()` call
4. **Consolidated utilities** — Shared `net_charge` and `isoelectric_point` functions in `utils.py` with tolerance-based bisection convergence (vs. fixed 1000 iterations)
5. **frozenset for amino acids** — O(1) membership checks instead of O(n) list lookups
6. **stdlib random** — `CodonOptimizer` uses `random.choices` instead of NumPy for the harmonised strategy
7. **Deduplicated hydrophobicity scale** — Single source of truth in `AA_PROPERTIES` used by both `barcodes.py` and `developability.py`
8. **Lazy scorer initialisation** — Optional scorers (ESM-2) are only loaded on demand
9. **`__slots__`** on `VHHSequence` for reduced memory footprint during large library generation

## Citations

This project relies on the following open-source libraries. If you use this tool in published work, please consider citing the relevant references:

| Library | Reference |
|---------|-----------|
| **ANARCI** | Dunbar, J. & Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics*, 32(2), 298–300. doi:[10.1093/bioinformatics/btv552](https://doi.org/10.1093/bioinformatics/btv552) |
| **HMMER** | Eddy, S.R. (2011). Accelerated profile HMM searches. *PLoS Comput Biol*, 7(10), e1002195. doi:[10.1371/journal.pcbi.1002195](https://doi.org/10.1371/journal.pcbi.1002195) |
| **ESM-2** (fair-esm) | Lin, Z. *et al.* (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130. doi:[10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) |
| **DNA Chisel** | Zulkower, V. & Rosser, S. (2020). DNA Chisel, a versatile sequence optimizer. *Bioinformatics*, 36(16), 4508–4509. doi:[10.1093/bioinformatics/btaa558](https://doi.org/10.1093/bioinformatics/btaa558) |
| **python-codon-tables** | Edinburgh Genome Foundry. Codon usage tables for Python, sourced from the Kazusa database. [GitHub](https://github.com/Edinburgh-Genome-Foundry/python-codon-tables) |
| **BioPython** | Cock, P.J.A. *et al.* (2009). Biopython: freely available Python tools for computational molecular biology. *Bioinformatics*, 25(11), 1422–1423. doi:[10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163) |
| **PyTorch** | Paszke, A. *et al.* (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*, 32. [Paper](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) |
| **NumPy** | Harris, C.R. *et al.* (2020). Array programming with NumPy. *Nature*, 585, 357–362. doi:[10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) |
| **SciPy** | Virtanen, P. *et al.* (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. doi:[10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) |
| **pandas** | McKinney, W. (2010). Data structures for statistical computing in Python. *Proc. of the 9th Python in Science Conf.*, 56–61. doi:[10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a) |
| **Matplotlib** | Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. doi:[10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55) |
| **Streamlit** | Streamlit Inc. [https://streamlit.io](https://streamlit.io) |