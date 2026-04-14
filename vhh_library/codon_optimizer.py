from __future__ import annotations

import json
import math
import random
from pathlib import Path

from vhh_library.utils import AMINO_ACIDS

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "codon_tables"

_RESTRICTION_SITES: dict[str, str] = {
    "BamHI": "GGATCC",
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "NdeI": "CATATG",
}


class CodonOptimizer:
    def __init__(self) -> None:
        self._tables: dict[str, dict[str, dict[str, float]]] = {}
        for path in sorted(_DATA_DIR.glob("*.json")):
            host = path.stem
            with open(path) as fh:
                self._tables[host] = json.load(fh)

    def optimize(
        self,
        aa_sequence: str,
        host: str = "e_coli",
        strategy: str = "most_frequent",
    ) -> dict:
        if host not in self._tables:
            raise ValueError(f"Unknown host '{host}'. Available: {sorted(self._tables)}")
        if strategy not in ("most_frequent", "harmonized", "gc_balanced"):
            raise ValueError(f"Unknown strategy '{strategy}'")

        table = self._tables[host]
        codons_out: list[str] = []

        for aa in aa_sequence:
            if aa not in AMINO_ACIDS:
                raise ValueError(f"Invalid amino acid '{aa}'")
            codon_freqs = table[aa]
            codons = list(codon_freqs.keys())
            freqs = list(codon_freqs.values())

            if strategy == "most_frequent":
                codons_out.append(codons[freqs.index(max(freqs))])
            elif strategy == "harmonized":
                codons_out.append(random.choices(codons, weights=freqs, k=1)[0])
            elif strategy == "gc_balanced":
                codons_out.append(self._pick_gc_balanced(codons_out, codons))

        dna_sequence = "".join(codons_out)
        gc_content = self._gc_content(dna_sequence)
        cai = self._compute_cai(aa_sequence, codons_out, table)

        warnings: list[str] = []
        if gc_content < 0.30:
            warnings.append(f"Low GC content: {gc_content:.2%}")
        if gc_content > 0.70:
            warnings.append(f"High GC content: {gc_content:.2%}")

        flagged_sites = self._flag_sites(dna_sequence, table)

        return {
            "dna_sequence": dna_sequence,
            "gc_content": gc_content,
            "cai": cai,
            "warnings": warnings,
            "flagged_sites": flagged_sites,
        }

    @staticmethod
    def _gc_content(dna: str) -> float:
        if not dna:
            return 0.0
        return sum(1 for nt in dna if nt in "GC") / len(dna)

    @staticmethod
    def _pick_gc_balanced(codons_so_far: list[str], candidates: list[str]) -> str:
        current_dna = "".join(codons_so_far)
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
        return best_codon

    @staticmethod
    def _compute_cai(
        aa_sequence: str,
        codons_out: list[str],
        table: dict[str, dict[str, float]],
    ) -> float:
        log_sum = 0.0
        count = 0
        for aa, codon in zip(aa_sequence, codons_out):
            codon_freqs = table[aa]
            max_freq = max(codon_freqs.values())
            if max_freq == 0:
                continue
            w = codon_freqs[codon] / max_freq
            if w > 0:
                log_sum += math.log(w)
                count += 1
        if count == 0:
            return 0.0
        return math.exp(log_sum / count)

    @staticmethod
    def _flag_sites(
        dna_sequence: str,
        table: dict[str, dict[str, float]],
    ) -> list[str]:
        flagged: list[str] = []

        stop_codons = set(table.get("*", {}).keys())
        for i in range(3, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i : i + 3]
            if codon in stop_codons:
                flagged.append(f"Internal stop codon {codon} at position {i}")

        for name, site in _RESTRICTION_SITES.items():
            idx = dna_sequence.find(site)
            while idx != -1:
                flagged.append(f"{name} site ({site}) at position {idx}")
                idx = dna_sequence.find(site, idx + 1)

        return flagged
