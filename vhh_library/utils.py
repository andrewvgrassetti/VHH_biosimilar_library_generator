"""VHH library utility functions for sequence analysis and manipulation."""

from __future__ import annotations

AMINO_ACIDS: frozenset[str] = frozenset(
    {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}
)

AA_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {"charge": 0.0, "hydrophobicity": 1.8, "mw": 89.09},
    "R": {"charge": 1.0, "hydrophobicity": -4.5, "mw": 174.20},
    "N": {"charge": 0.0, "hydrophobicity": -3.5, "mw": 132.12},
    "D": {"charge": -1.0, "hydrophobicity": -3.5, "mw": 133.10},
    "C": {"charge": 0.0, "hydrophobicity": 2.5, "mw": 121.16},
    "Q": {"charge": 0.0, "hydrophobicity": -3.5, "mw": 146.15},
    "E": {"charge": -1.0, "hydrophobicity": -3.5, "mw": 147.13},
    "G": {"charge": 0.0, "hydrophobicity": -0.4, "mw": 75.03},
    "H": {"charge": 0.0, "hydrophobicity": -3.2, "mw": 155.16},
    "I": {"charge": 0.0, "hydrophobicity": 4.5, "mw": 131.17},
    "L": {"charge": 0.0, "hydrophobicity": 3.8, "mw": 131.17},
    "K": {"charge": 1.0, "hydrophobicity": -3.9, "mw": 146.19},
    "M": {"charge": 0.0, "hydrophobicity": 1.9, "mw": 149.21},
    "F": {"charge": 0.0, "hydrophobicity": 2.8, "mw": 165.19},
    "P": {"charge": 0.0, "hydrophobicity": -1.6, "mw": 115.13},
    "S": {"charge": 0.0, "hydrophobicity": -0.8, "mw": 105.09},
    "T": {"charge": 0.0, "hydrophobicity": -0.7, "mw": 119.12},
    "W": {"charge": 0.0, "hydrophobicity": -0.9, "mw": 204.23},
    "Y": {"charge": 0.0, "hydrophobicity": -1.3, "mw": 181.19},
    "V": {"charge": 0.0, "hydrophobicity": 4.2, "mw": 117.15},
}

CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


# Chemically similar amino acid groups for conservative substitution.
# When a position is set to CONSERVATIVE without a pre-defined allowed set,
# these groups provide a sensible default based on the wild-type residue.
SIMILAR_AA_GROUPS: dict[str, frozenset[str]] = {
    "A": frozenset({"A", "G", "S", "T", "V"}),
    "G": frozenset({"G", "A", "S"}),
    "S": frozenset({"S", "T", "A", "G", "N"}),
    "T": frozenset({"T", "S", "A", "V"}),
    "V": frozenset({"V", "I", "L", "A", "M"}),
    "I": frozenset({"I", "V", "L", "M"}),
    "L": frozenset({"L", "I", "V", "M", "F"}),
    "M": frozenset({"M", "L", "I", "V"}),
    "F": frozenset({"F", "Y", "W", "L"}),
    "Y": frozenset({"Y", "F", "W", "H"}),
    "W": frozenset({"W", "F", "Y"}),
    "P": frozenset({"P", "A"}),
    "D": frozenset({"D", "E", "N", "Q"}),
    "E": frozenset({"E", "D", "Q", "N"}),
    "N": frozenset({"N", "D", "Q", "S", "T"}),
    "Q": frozenset({"Q", "E", "N", "K"}),
    "K": frozenset({"K", "R", "Q", "H"}),
    "R": frozenset({"R", "K", "Q", "H"}),
    "H": frozenset({"H", "K", "R", "N", "Q"}),
    "C": frozenset({"C", "S", "A"}),
}


DEFAULT_CONSERVATIVE_FALLBACK: frozenset[str] = frozenset({"A", "G", "S", "T", "V"})


def translate(dna_sequence: str) -> str:
    """Translate a DNA sequence to an amino acid sequence."""
    seq = dna_sequence.upper()
    amino_acids: list[str] = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        amino_acids.append(CODON_TABLE.get(seq[i : i + 3], "X"))
    return "".join(amino_acids)


def net_charge(sequence: str, pH: float = 7.4) -> float:
    """Calculate the net charge of a peptide at a given pH using Henderson-Hasselbalch."""
    if not sequence:
        return 0.0

    # pKa values (Bjellqvist scale) for ionisable groups
    pka_positive = {"K": 10.5, "R": 12.4, "H": 6.0}
    pka_negative = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
    pka_nterm = 8.0
    pka_cterm = 3.1

    charge = 0.0

    # N-terminus (positive when protonated)
    charge += 10**pka_nterm / (10**pka_nterm + 10**pH)
    # C-terminus (negative when deprotonated)
    charge -= 10**pH / (10**pka_cterm + 10**pH)

    for aa in sequence:
        if aa in pka_positive:
            charge += 10 ** pka_positive[aa] / (10 ** pka_positive[aa] + 10**pH)
        elif aa in pka_negative:
            charge -= 10**pH / (10 ** pka_negative[aa] + 10**pH)

    return charge


def isoelectric_point(sequence: str) -> float:
    """Calculate the isoelectric point using bisection with tolerance-based convergence."""
    if not sequence:
        return 0.0

    low = 0.0
    high = 14.0
    tol = 0.01

    while (high - low) > tol:
        mid = (low + high) / 2.0
        if net_charge(sequence, mid) > 0.0:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def tryptic_digest(sequence: str, missed_cleavages: int = 0) -> list[str]:
    """Simulate trypsin digestion (cleave after K/R, not before P)."""
    if not sequence:
        return []

    # Find cleavage sites
    sites: list[int] = []
    for i in range(len(sequence) - 1):
        if sequence[i] in ("K", "R") and sequence[i + 1] != "P":
            sites.append(i + 1)

    # Split into base peptides
    peptides: list[str] = []
    prev = 0
    for site in sites:
        peptides.append(sequence[prev:site])
        prev = site
    peptides.append(sequence[prev:])

    # Include missed cleavages by concatenating adjacent peptides
    results: list[str] = list(peptides)
    for mc in range(1, missed_cleavages + 1):
        for i in range(len(peptides) - mc):
            results.append("".join(peptides[i : i + mc + 1]))

    return results
