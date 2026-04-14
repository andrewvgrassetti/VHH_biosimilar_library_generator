from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_LINKER_CODON_MAP: dict[str, str] = {
    "G": "GGT",
    "S": "AGC",
    "A": "GCT",
    "L": "CTG",
    "P": "CCG",
    "E": "GAA",
}


def _encode_linker_dna(linker_aa: str) -> str | None:
    parts: list[str] = []
    for aa in linker_aa:
        codon = _LINKER_CODON_MAP.get(aa)
        if codon is None:
            return None
        parts.append(codon)
    return "".join(parts)


class TagManager:
    def __init__(self) -> None:
        tag_path = _DATA_DIR / "tag_sequences.json"
        with open(tag_path) as fh:
            self._tags: dict[str, dict] = json.load(fh)

    def get_available_tags(self) -> dict:
        return self._tags

    def build_construct(
        self,
        aa_sequence: str,
        dna_sequence: str,
        n_tag: str | None = None,
        c_tag: str | None = None,
        linker: str = "GSGSGS",
    ) -> dict:
        n_tag_info = self._tags[n_tag] if n_tag else None
        c_tag_info = self._tags[c_tag] if c_tag else None

        # --- amino-acid construct ---
        aa_construct = aa_sequence
        if n_tag_info:
            aa_construct = n_tag_info["aa_sequence"] + linker + aa_construct
        if c_tag_info:
            aa_construct = aa_construct + linker + c_tag_info["aa_sequence"]

        # --- DNA construct ---
        linker_dna = _encode_linker_dna(linker)
        dna_parts_valid = True

        if linker_dna is None:
            dna_parts_valid = False

        if n_tag_info and not n_tag_info.get("dna_sequence"):
            dna_parts_valid = False
        if c_tag_info and not c_tag_info.get("dna_sequence"):
            dna_parts_valid = False
        if not dna_sequence:
            dna_parts_valid = False

        if dna_parts_valid:
            dna_construct = dna_sequence
            if n_tag_info:
                dna_construct = n_tag_info["dna_sequence"] + linker_dna + dna_construct
            if c_tag_info:
                dna_construct = dna_construct + linker_dna + c_tag_info["dna_sequence"]
        else:
            dna_construct = ""

        # --- schematic ---
        parts: list[str] = []
        if n_tag_info:
            parts.append(f"[{n_tag}]")
            parts.append("[linker]")
        parts.append("[VHH]")
        if c_tag_info:
            parts.append("[linker]")
            parts.append(f"[{c_tag}]")
        schematic = "-".join(parts)

        return {
            "aa_construct": aa_construct,
            "dna_construct": dna_construct,
            "schematic": schematic,
        }
