"""Codon optimization backed by python-codon-tables (Kazusa) and DNA Chisel."""

from __future__ import annotations

import math
import random
from typing import Sequence

import python_codon_tables as pct
from dnachisel import (
    AvoidHairpins,
    AvoidPattern,
    CodonOptimize,
    DnaOptimizationProblem,
    EnforceGCContent,
    EnforceTranslation,
    UniquifyAllKmers,
)

from vhh_library.utils import AMINO_ACIDS

_STRATEGIES = ("most_frequent", "harmonized", "gc_balanced", "dnachisel_optimized")

# Fallback codon-usage tables for organisms not bundled in python-codon-tables.
# Pichia pastoris (Komagataella phaffii) — widely used for recombinant protein
# expression but absent from the small bundled set.
_FALLBACK_TABLES: dict[str, dict[str, dict[str, float]]] = {
    "p_pastoris": {
        "A": {"GCT": 0.35, "GCC": 0.25, "GCA": 0.25, "GCG": 0.15},
        "R": {"CGT": 0.15, "CGC": 0.10, "CGA": 0.08, "CGG": 0.07, "AGA": 0.40, "AGG": 0.20},
        "N": {"AAT": 0.55, "AAC": 0.45},
        "D": {"GAT": 0.60, "GAC": 0.40},
        "C": {"TGT": 0.55, "TGC": 0.45},
        "Q": {"CAA": 0.60, "CAG": 0.40},
        "E": {"GAA": 0.65, "GAG": 0.35},
        "G": {"GGT": 0.40, "GGC": 0.22, "GGA": 0.23, "GGG": 0.15},
        "H": {"CAT": 0.60, "CAC": 0.40},
        "I": {"ATT": 0.44, "ATC": 0.30, "ATA": 0.26},
        "L": {"TTA": 0.20, "TTG": 0.28, "CTT": 0.16, "CTC": 0.09, "CTA": 0.12, "CTG": 0.15},
        "K": {"AAA": 0.55, "AAG": 0.45},
        "M": {"ATG": 1.00},
        "F": {"TTT": 0.55, "TTC": 0.45},
        "P": {"CCT": 0.30, "CCC": 0.18, "CCA": 0.37, "CCG": 0.15},
        "S": {"TCT": 0.23, "TCC": 0.18, "TCA": 0.19, "TCG": 0.12, "AGT": 0.16, "AGC": 0.12},
        "T": {"ACT": 0.32, "ACC": 0.25, "ACA": 0.28, "ACG": 0.15},
        "W": {"TGG": 1.00},
        "Y": {"TAT": 0.55, "TAC": 0.45},
        "V": {"GTT": 0.35, "GTC": 0.22, "GTA": 0.20, "GTG": 0.23},
        "*": {"TAA": 0.50, "TAG": 0.20, "TGA": 0.30},
    },
}

_DEFAULT_RESTRICTION_ENZYMES: list[str] = [
    "BamHI",
    "EcoRI",
    "HindIII",
    "NdeI",
    "BsaI",
    "BpiI",
]

_RESTRICTION_SITE_SEQS: dict[str, str] = {
    "BamHI": "GGATCC",
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "NdeI": "CATATG",
    "BsaI": "GGTCTC",
    "BpiI": "GAAGAC",
}


# ------------------------------------------------------------------
# Organism resolution
# ------------------------------------------------------------------

def _resolve_organism(organism: str) -> tuple[str, dict[str, dict[str, float]]]:
    """Resolve *organism* (name, prefix, or taxonomy ID) to a codon table.

    Returns ``(resolved_name, table)`` where *table* maps amino-acid
    one-letter codes to ``{codon: frequency}`` dicts.

    Raises :class:`ValueError` when the organism cannot be found.
    """
    key = str(organism).strip()

    # 1. Check fallback tables first (exact match)
    if key in _FALLBACK_TABLES:
        return key, _FALLBACK_TABLES[key]

    # 2. Try python-codon-tables (supports prefix matching)
    try:
        table = pct.get_codons_table(key)
    except Exception as exc:
        available = sorted(pct.available_codon_tables_names) + sorted(_FALLBACK_TABLES)
        raise ValueError(
            f"Unknown organism '{organism}'. Could not load codon table: {exc}. "
            f"Available tables: {', '.join(available)}"
        ) from exc

    # Derive a resolved name
    resolved = key
    for name in pct.available_codon_tables_names:
        if name.startswith(key) or name == key:
            resolved = name
            break

    return resolved, table


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

def _most_frequent_codons(aa_sequence: str, table: dict[str, dict[str, float]]) -> str:
    """Return a DNA sequence using the highest-frequency codon per amino acid."""
    codons: list[str] = []
    for aa in aa_sequence:
        if aa == "*":
            continue
        if aa not in table:
            raise ValueError(f"Invalid amino acid '{aa}'")
        codon_freqs = table[aa]
        best_codon = max(codon_freqs, key=codon_freqs.get)  # type: ignore[arg-type]
        codons.append(best_codon)
    return "".join(codons)


def _harmonized_codons(aa_sequence: str, table: dict[str, dict[str, float]]) -> str:
    """Return a DNA sequence using weighted random sampling from the frequency table."""
    codons: list[str] = []
    for aa in aa_sequence:
        if aa == "*":
            continue
        if aa not in table:
            raise ValueError(f"Invalid amino acid '{aa}'")
        codon_freqs = table[aa]
        codon_list = list(codon_freqs.keys())
        weights = list(codon_freqs.values())
        codons.append(random.choices(codon_list, weights=weights, k=1)[0])
    return "".join(codons)


def _gc_balanced_codons(aa_sequence: str, table: dict[str, dict[str, float]]) -> str:
    """Return a DNA sequence that targets 50% GC content using available codons."""
    codons_out: list[str] = []
    for aa in aa_sequence:
        if aa == "*":
            continue
        if aa not in table:
            raise ValueError(f"Invalid amino acid '{aa}'")
        candidates = list(table[aa].keys())

        current_dna = "".join(codons_out)
        current_len = len(current_dna)
        current_gc = sum(1 for nt in current_dna if nt in "GC")

        best_codon = candidates[0]
        best_diff = float("inf")
        for codon in candidates:
            new_gc = current_gc + sum(1 for nt in codon if nt in "GC")
            new_len = current_len + len(codon)
            diff = abs(new_gc / new_len - 0.5) if new_len else 0.0
            if diff < best_diff:
                best_diff = diff
                best_codon = codon
        codons_out.append(best_codon)
    return "".join(codons_out)


def _gc_content(dna: str) -> float:
    """Fraction of G + C nucleotides in *dna*."""
    if not dna:
        return 0.0
    return sum(1 for nt in dna if nt in "GC") / len(dna)


def _compute_cai(
    aa_sequence: str,
    dna_sequence: str,
    table: dict[str, dict[str, float]],
) -> float:
    """Compute the Codon Adaptation Index from *table* frequency data."""
    log_sum = 0.0
    count = 0
    idx = 0
    for aa in aa_sequence:
        if aa == "*":
            idx += 3
            continue
        codon = dna_sequence[idx : idx + 3]
        idx += 3
        if aa not in table:
            continue
        codon_freqs = table[aa]
        max_freq = max(codon_freqs.values())
        if max_freq == 0:
            continue
        freq = codon_freqs.get(codon, 0.0)
        w = freq / max_freq
        if w > 0:
            log_sum += math.log(w)
            count += 1
    if count == 0:
        return 0.0
    return math.exp(log_sum / count)


def _flag_sites(dna_sequence: str) -> list[str]:
    """Flag internal stop codons and common restriction sites."""
    flagged: list[str] = []

    stop_codons = {"TAA", "TAG", "TGA"}
    for i in range(3, len(dna_sequence) - 2, 3):
        codon = dna_sequence[i : i + 3]
        if codon in stop_codons:
            flagged.append(f"Internal stop codon {codon} at position {i}")

    for name, site in _RESTRICTION_SITE_SEQS.items():
        idx = dna_sequence.find(site)
        while idx != -1:
            flagged.append(f"{name} site ({site}) at position {idx}")
            idx = dna_sequence.find(site, idx + 1)

    return flagged


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class CodonOptimizer:
    """Codon-optimize amino-acid sequences for a target organism.

    Backed by ``python-codon-tables`` (Kazusa data) and ``dnachisel``
    for constraint-based optimization.
    """

    def optimize(
        self,
        aa_sequence: str,
        host: str = "e_coli",
        strategy: str = "most_frequent",
        *,
        restriction_enzymes: Sequence[str] | None = None,
        gc_window: int = 50,
        gc_mini: float = 0.30,
        gc_maxi: float = 0.65,
        avoid_hairpins: bool = True,
        uniquify_kmers: int | None = 9,
    ) -> dict:
        """Optimize *aa_sequence* for expression in *host*.

        Parameters
        ----------
        aa_sequence:
            Amino-acid one-letter sequence (no stop ``*``).
        host:
            Organism name, prefix, or NCBI taxonomy ID.
        strategy:
            ``"most_frequent"``, ``"harmonized"``, ``"gc_balanced"``,
            or ``"dnachisel_optimized"``.
        restriction_enzymes:
            Enzyme names whose recognition sites should be avoided
            (only used by ``dnachisel_optimized``).  Defaults to the
            standard list when *None*.
        gc_window:
            Sliding-window size for GC-content constraint (bp).
        gc_mini / gc_maxi:
            Allowed GC-content range.
        avoid_hairpins:
            Add ``AvoidHairpins`` constraint (dnachisel_optimized only).
        uniquify_kmers:
            *k* for ``UniquifyAllKmers``; ``None`` to skip.
        """
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {', '.join(_STRATEGIES)}"
            )

        clean_aa = aa_sequence.replace("*", "")
        for aa in clean_aa:
            if aa not in AMINO_ACIDS:
                raise ValueError(f"Invalid amino acid '{aa}'")

        resolved_name, table = _resolve_organism(host)

        # -- Generate DNA --
        constraints_passed: bool | None = None
        constraint_violations: list[str] = []

        if strategy == "most_frequent":
            dna = _most_frequent_codons(clean_aa, table)
        elif strategy == "harmonized":
            dna = _harmonized_codons(clean_aa, table)
        elif strategy == "gc_balanced":
            dna = _gc_balanced_codons(clean_aa, table)
        elif strategy == "dnachisel_optimized":
            dna, constraints_passed, constraint_violations = self._dnachisel_optimize(
                clean_aa,
                host=host,
                table=table,
                restriction_enzymes=restriction_enzymes,
                gc_window=gc_window,
                gc_mini=gc_mini,
                gc_maxi=gc_maxi,
                avoid_hairpins=avoid_hairpins,
                uniquify_kmers=uniquify_kmers,
            )
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        gc = _gc_content(dna)
        cai = _compute_cai(clean_aa, dna, table)

        warnings: list[str] = []
        if gc < 0.30:
            warnings.append(f"Low GC content: {gc:.2%}")
        if gc > 0.70:
            warnings.append(f"High GC content: {gc:.2%}")

        flagged_sites = _flag_sites(dna)

        result: dict = {
            "dna_sequence": dna,
            "gc_content": gc,
            "cai": cai,
            "warnings": warnings,
            "flagged_sites": flagged_sites,
            "organism": resolved_name,
        }
        if strategy == "dnachisel_optimized":
            result["constraints_passed"] = constraints_passed
            result["constraint_violations"] = constraint_violations
        return result

    # ------------------------------------------------------------------
    # DnaChisel strategy
    # ------------------------------------------------------------------
    @staticmethod
    def _dnachisel_optimize(
        aa_sequence: str,
        *,
        host: str,
        table: dict[str, dict[str, float]],
        restriction_enzymes: Sequence[str] | None,
        gc_window: int,
        gc_mini: float,
        gc_maxi: float,
        avoid_hairpins: bool,
        uniquify_kmers: int | None,
    ) -> tuple[str, bool, list[str]]:
        """Run DnaChisel constraint-based optimization.

        Returns ``(dna_sequence, constraints_passed, violations)``.
        """
        if restriction_enzymes is None:
            restriction_enzymes = _DEFAULT_RESTRICTION_ENZYMES

        # Initial DNA: most-frequent codons
        initial_dna = _most_frequent_codons(aa_sequence, table)

        # Build constraints
        constraints: list = [EnforceTranslation()]

        for enzyme in restriction_enzymes:
            pattern = f"{enzyme}_site"
            try:
                constraints.append(AvoidPattern(pattern))
            except (ValueError, KeyError):
                # DnaChisel doesn't recognise this enzyme name; fall back to
                # the raw recognition sequence if we know it.
                if enzyme in _RESTRICTION_SITE_SEQS:
                    constraints.append(AvoidPattern(_RESTRICTION_SITE_SEQS[enzyme]))

        seq_len = len(initial_dna)
        if seq_len >= gc_window:
            constraints.append(
                EnforceGCContent(mini=gc_mini, maxi=gc_maxi, window=gc_window)
            )

        if avoid_hairpins:
            try:
                constraints.append(AvoidHairpins())
            except (TypeError, ValueError, ImportError):
                pass  # AvoidHairpins unavailable or unsupported in this version

        if uniquify_kmers is not None and seq_len >= uniquify_kmers:
            constraints.append(UniquifyAllKmers(k=uniquify_kmers))

        objectives = [CodonOptimize(species=host)]

        problem = DnaOptimizationProblem(
            sequence=initial_dna,
            constraints=constraints,
            objectives=objectives,
        )

        problem.resolve_constraints()
        problem.optimize()

        # Evaluate constraint satisfaction
        violations: list[str] = []
        all_passed = True
        for ev in problem.constraints_evaluations():
            if not ev.passes:
                all_passed = False
                violations.append(str(ev))

        return problem.sequence, all_passed, violations
