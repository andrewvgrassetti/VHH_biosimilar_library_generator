"""VHH sequence representation with IMGT numbering, validation, and region annotation."""

from __future__ import annotations

import logging
from functools import cached_property

from vhh_library.numbering import NumberingError, number_sequence
from vhh_library.utils import AMINO_ACIDS

logger = logging.getLogger(__name__)

# IMGT region boundaries (inclusive start and end positions).
IMGT_REGIONS: dict[str, tuple[int, int]] = {
    "FR1": (1, 26),
    "CDR1": (27, 38),
    "FR2": (39, 55),
    "CDR2": (56, 65),
    "FR3": (66, 104),
    "CDR3": (105, 117),
    "FR4": (118, 128),
}

_MIN_LENGTH = 80
_MAX_LENGTH = 180
_CDR_LENGTH_RANGES: dict[str, tuple[int, int]] = {
    "CDR1": (1, 15),
    "CDR2": (1, 20),
    "CDR3": (1, 30),
}


class VHHSequence:
    """Memory-efficient VHH sequence with IMGT numbering and cached region accessors.

    ``imgt_numbered`` is a ``dict[str, str]`` mapping IMGT position keys to
    amino-acid characters.  Standard positions use plain integer-like string
    keys (``"1"``, ``"2"``, …).  Insertion positions carry the insertion code
    appended (``"111A"``, ``"111B"``).  Positions absent from the sequence
    (IMGT gaps) are simply missing from the dict.
    """

    __slots__ = (
        "sequence",
        "length",
        "strict",
        "imgt_numbered",
        "validation_result",
        "chain_type",
        "species",
        "_pos_to_seq_idx",
        "__dict__",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, sequence: str, strict: bool = True) -> None:
        self.sequence: str = sequence.upper().strip()
        self.length: int = len(self.sequence)
        self.strict: bool = strict

        # ANARCI-based IMGT numbering.
        self.chain_type: str = ""
        self.species: str = ""
        self.imgt_numbered: dict[str, str] = {}
        self._pos_to_seq_idx: dict[str, int] = {}
        self.validation_result: dict = self._number_and_validate()

    @classmethod
    def mutate(cls, source: VHHSequence, position: int | str, new_aa: str) -> VHHSequence:
        """Create a new *VHHSequence* with a single-point mutation, skipping ANARCI re-annotation.

        This is a **fast path** for batch mutagenesis of known-good variants.
        The caller is responsible for ensuring:
        * *position* is a 1-based IMGT position (int or string key).
        * *new_aa* is a valid standard amino acid character.
        * The resulting sequence still satisfies any domain constraints (e.g.
          conserved residues at positions 23, 41, 104).

        The parent's ANARCI numbering is **copied** and only the changed
        position is updated.  ``validation_result`` is likewise copied from
        *source* without re-running checks.
        """
        pos_key = str(position)
        seq_idx = source._pos_to_seq_idx.get(pos_key)
        if seq_idx is None:
            raise ValueError(f"IMGT position {position!r} not found in source numbering")

        logger.debug(
            "Fast-path mutate: position=%s, %s→%s (ANARCI bypassed)",
            pos_key,
            source.imgt_numbered.get(pos_key, "?"),
            new_aa,
        )

        seq_list = list(source.sequence)
        seq_list[seq_idx] = new_aa.upper()
        mutated = object.__new__(cls)
        mutated.sequence = "".join(seq_list)
        mutated.length = len(mutated.sequence)

        # Copy parent numbering and update only the mutated position.
        mutated.imgt_numbered = dict(source.imgt_numbered)
        mutated.imgt_numbered[pos_key] = new_aa.upper()

        mutated._pos_to_seq_idx = source._pos_to_seq_idx  # shared (read-only)
        mutated.chain_type = source.chain_type
        mutated.species = source.species
        mutated.strict = source.strict
        mutated.validation_result = source.validation_result  # skip re-validation
        return mutated

    # ------------------------------------------------------------------
    # ANARCI numbering + validation (combined to avoid double work)
    # ------------------------------------------------------------------

    def _number_and_validate(self) -> dict:
        """Run ANARCI numbering and basic validation in one pass."""
        errors: list[str] = []
        warnings: list[str] = []

        # Check for invalid characters.
        invalid = {aa for aa in self.sequence if aa not in AMINO_ACIDS}
        if invalid:
            errors.append(f"Invalid amino acid(s): {', '.join(sorted(invalid))}")

        # Length check.
        if not (_MIN_LENGTH <= self.length <= _MAX_LENGTH):
            errors.append(f"Length {self.length} outside valid range ({_MIN_LENGTH}-{_MAX_LENGTH})")

        # Attempt ANARCI numbering.
        try:
            result = number_sequence(self.sequence)
            self.imgt_numbered = result.numbered
            self.chain_type = result.chain_type
            self.species = result.species
            # Build mapping from IMGT position key → 0-based sequence index.
            # The numbered dict is ordered by ANARCI output; each entry maps
            # to the next residue in the raw sequence string.
            self._pos_to_seq_idx = {pos_key: idx for idx, pos_key in enumerate(self.imgt_numbered)}
        except NumberingError as exc:
            errors.append(str(exc))
            return {"valid": False, "errors": errors, "warnings": warnings}

        reconstructed = "".join(self.imgt_numbered.values())
        if reconstructed != self.sequence:
            errors.append("ANARCI numbering does not reconstruct the input sequence exactly")

        # Conserved residue checks (using real IMGT positions).
        if self.imgt_numbered.get("23") != "C":
            msg = "Missing conserved Cys at IMGT position 23"
            (errors if self.strict else warnings).append(msg)
        if self.imgt_numbered.get("104") != "C":
            msg = "Missing conserved Cys at IMGT position 104"
            (errors if self.strict else warnings).append(msg)
        if self.imgt_numbered.get("41") != "W":
            warnings.append("Missing conserved Trp at IMGT position 41")

        for cdr_name, (min_len, max_len) in _CDR_LENGTH_RANGES.items():
            start, end = IMGT_REGIONS[cdr_name]
            cdr_len = sum(1 for pos in range(start, end + 1) if str(pos) in self.imgt_numbered)
            if cdr_len == 0:
                errors.append(f"{cdr_name} is missing from ANARCI numbering output")
            elif not (min_len <= cdr_len <= max_len):
                warnings.append(f"{cdr_name} length {cdr_len} outside plausible range ({min_len}-{max_len})")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Cached region / position properties
    # ------------------------------------------------------------------

    @cached_property
    def regions(self) -> dict[str, tuple[int, int, str]]:
        """Map each IMGT region to ``(start, end, subsequence)``.

        Positions that are absent from :attr:`imgt_numbered` (IMGT gaps)
        are silently skipped when building the subsequence.
        """
        result: dict[str, tuple[int, int, str]] = {}
        for name, (start, end) in IMGT_REGIONS.items():
            subseq = "".join(self.imgt_numbered.get(str(pos), "") for pos in range(start, end + 1))
            result[name] = (start, end, subseq)
        return result

    @cached_property
    def cdr_positions(self) -> frozenset[str]:
        """All IMGT positions (as string keys) belonging to CDR regions."""
        positions: set[str] = set()
        for name, (start, end) in IMGT_REGIONS.items():
            if name.startswith("CDR"):
                positions.update(str(p) for p in range(start, end + 1))
        return frozenset(positions)

    @cached_property
    def framework_positions(self) -> frozenset[str]:
        """All IMGT positions (as string keys) belonging to framework regions."""
        positions: set[str] = set()
        for name, (start, end) in IMGT_REGIONS.items():
            if name.startswith("FR"):
                positions.update(str(p) for p in range(start, end + 1))
        return frozenset(positions)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"VHHSequence(length={self.length}, valid={self.validation_result['valid']})"

    def __len__(self) -> int:
        return self.length
