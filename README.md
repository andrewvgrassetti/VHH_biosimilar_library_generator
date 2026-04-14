# VHH Biosimilar Library Generator

A computational pipeline for designing humanised VHH (nanobody) variant libraries with multi-objective scoring, codon optimisation, and LC-MS/MS barcoding support.

## Features

- **Humanness Scoring** — Compares framework regions to human VH germlines using identity and position-frequency matrices
- **Stability Prediction** — Multi-factor scoring (disulfide bonds, aggregation propensity, charge balance, VHH hallmarks) with optional NanoMelt Tm prediction
- **Orthogonal Validation** — Independent scoring via Human String Content (HSC) and germline consensus methods
- **Mutation Engine** — Ranks single-point mutations and generates combinatorial variant libraries with three strategies (exhaustive, random sampling, iterative anchor-and-explore refinement)
- **Developability Assessment** — PTM liability detection, surface hydrophobicity patches, clearance risk scoring
- **Codon Optimisation** — Organism-specific codon usage tables (E. coli, H. sapiens, P. pastoris, S. cerevisiae) with CAI calculation and restriction site flagging
- **Construct Builder** — Automated tag/linker assembly with DNA encoding
- **Peptide Barcoding** — Unique trypsin-cleavable barcodes for multiplexed LC-MS/MS screening
- **ESM-2 PLL Rescoring** — Optional protein language model validation (requires PyTorch)

## Installation

```bash
pip install -e .
```

With optional ML dependencies:
```bash
pip install -e ".[ml,nanomelt]"
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
│   ├── stability.py                # Stability scoring with NanoMelt integration
│   ├── developability.py           # PTM, clearance risk, surface hydrophobicity
│   ├── orthogonal_scoring.py       # HSC, consensus, NanoMelt scorers
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
8. **Lazy scorer initialisation** — Optional scorers (NanoMelt, ESM-2) are only loaded on demand
9. **`__slots__`** on `VHHSequence` for reduced memory footprint during large library generation