"""Peptide barcode assignment for multiplexed LC-MS/MS screening."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vhh_library.utils import AA_PROPERTIES, tryptic_digest

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_DEAMIDATION_RE = re.compile(r"N[GSH]")
_GLYCOSYLATION_RE = re.compile(r"N[^P][ST]")


def _hydrophobicity(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(AA_PROPERTIES[aa]["hydrophobicity"] for aa in sequence) / len(sequence)


def _peptide_neutral_mass(sequence: str) -> float:
    return sum(AA_PROPERTIES[aa]["mw"] for aa in sequence) + 18.015


def _mz(mass: float, charge: int) -> float:
    return (mass + charge * 1.00728) / charge


def _barcode_passes_rules(sequence: str) -> bool:
    length = len(sequence)
    if length < 6 or length > 12:
        return False
    if sequence[0] not in ("K", "R"):
        return False
    if sequence[-1] not in ("K", "R"):
        return False
    if any(aa in ("K", "R") for aa in sequence[1:-1]):
        return False
    if "M" in sequence:
        return False
    if "C" in sequence:
        return False
    if _DEAMIDATION_RE.search(sequence):
        return False
    if _GLYCOSYLATION_RE.search(sequence):
        return False
    return True


class BarcodeGenerator:
    def __init__(self) -> None:
        pool_path = _DATA_DIR / "barcode_pool.json"
        with open(pool_path) as fh:
            raw = json.load(fh)
        self.pool: list[str] = [entry["sequence"] for entry in raw]

    def assign_barcodes(
        self,
        library: pd.DataFrame,
        top_n: int = 100,
        linker: str = "GGS",
        c_terminal_tail: str = "",
    ) -> pd.DataFrame:
        if "aa_sequence" not in library.columns:
            raise ValueError("library must contain an 'aa_sequence' column")
        if c_terminal_tail and re.search(r"[KR]", c_terminal_tail):
            raise ValueError("c_terminal_tail must not contain K or R")

        df = library.nlargest(top_n, "combined_score").copy()

        barcode_ids: list[str] = []
        barcode_peptides: list[str] = []
        barcoded_sequences: list[str] = []
        barcode_tryptic_peptides: list[str] = []

        pool_idx = 0
        for _, row in df.iterrows():
            variant_peptides = set(tryptic_digest(row["aa_sequence"], missed_cleavages=1))
            while pool_idx < len(self.pool) and self.pool[pool_idx] in variant_peptides:
                pool_idx += 1
            if pool_idx >= len(self.pool):
                raise RuntimeError("Barcode pool exhausted before all variants were assigned")
            bc = self.pool[pool_idx]
            pool_idx += 1

            bc_id = f"BC-{pool_idx:06d}"
            barcode_ids.append(bc_id)
            barcode_peptides.append(bc)
            barcoded_sequences.append(row["aa_sequence"] + linker + bc + c_terminal_tail)
            barcode_tryptic_peptides.append(bc)

        df["barcode_id"] = barcode_ids
        df["barcode_peptide"] = barcode_peptides
        df["barcoded_sequence"] = barcoded_sequences
        df["barcode_tryptic_peptide"] = barcode_tryptic_peptides

        return df

    def generate_barcode_reference(self, barcoded: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for _, row in barcoded.iterrows():
            pep = row["barcode_tryptic_peptide"]
            mass = _peptide_neutral_mass(pep)
            rows.append(
                {
                    "variant_id": row.get("variant_id", ""),
                    "barcode_id": row["barcode_id"],
                    "barcode_peptide": row["barcode_peptide"],
                    "barcode_tryptic_peptide": pep,
                    "neutral_mass_da": mass,
                    "mz_1plus": _mz(mass, 1),
                    "mz_2plus": _mz(mass, 2),
                    "mz_3plus": _mz(mass, 3),
                    "hydrophobicity": _hydrophobicity(pep),
                    "source": "precomputed_pool",
                }
            )
        return pd.DataFrame(rows)

    def generate_barcoded_fasta(self, barcoded: pd.DataFrame) -> str:
        lines: list[str] = []
        for _, row in barcoded.iterrows():
            vid = row.get("variant_id", "")
            lines.append(f">{vid}|{row['barcode_id']}")
            lines.append(row["barcoded_sequence"])
        return "\n".join(lines) + "\n"

    def plot_barcode_distributions(self, ref_table: pd.DataFrame | None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        if ref_table is None or ref_table.empty:
            for ax, title in zip(axes, ["Mass Distribution", "Hydrophobicity Distribution", "m/z 2+ Distribution"]):
                ax.set_title(title)
            fig.tight_layout()
            return fig

        axes[0].hist(ref_table["neutral_mass_da"], bins=20, edgecolor="black")
        axes[0].set_title("Mass Distribution")
        axes[0].set_xlabel("Neutral Mass (Da)")
        axes[0].set_ylabel("Count")

        axes[1].hist(ref_table["hydrophobicity"], bins=20, edgecolor="black")
        axes[1].set_title("Hydrophobicity Distribution")
        axes[1].set_xlabel("Hydrophobicity")
        axes[1].set_ylabel("Count")

        axes[2].hist(ref_table["mz_2plus"], bins=20, edgecolor="black")
        axes[2].set_title("m/z 2+ Distribution")
        axes[2].set_xlabel("m/z (2+)")
        axes[2].set_ylabel("Count")

        fig.tight_layout()
        return fig
