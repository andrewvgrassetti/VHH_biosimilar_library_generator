"""VHH sequence representation with IMGT numbering, validation, and region annotation."""

from __future__ import annotations

from functools import cached_property

from vhh_library.utils import AMINO_ACIDS

# IMGT region boundaries (inclusive start and end positions).
IMGT_REGIONS: dict[str, tuple[int, int]] = {
    "FR1": (1, 25),
    "CDR1": (26, 35),
    "FR2": (36, 49),
    "CDR2": (50, 58),
    "FR3": (59, 90),
    "CDR3": (91, 110),
    "FR4": (111, 128),
}

_MIN_LENGTH = 80
_MAX_LENGTH = 180
_MAX_IMGT_POS = 128


class VHHSequence:
    """Memory-efficient VHH sequence with IMGT numbering and cached region accessors."""

    __slots__ = ("sequence", "length", "imgt_numbered", "validation_result", "_skip_validation", "__dict__")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, sequence: str) -> None:
        self.sequence: str = sequence.upper().strip()
        self.length: int = len(self.sequence)
        self.imgt_numbered: dict[int, str] = self._imgt_number()
        self.validation_result: dict = self._validate()

    @classmethod
    def mutate(cls, source: VHHSequence, position: int, new_aa: str) -> VHHSequence:
        """Create a new *VHHSequence* with a single-point mutation, skipping validation.

        This is a **fast path** for batch mutagenesis of known-good variants.
        The caller is responsible for ensuring:
        * *position* is a 1-based IMGT position in ``[1, len(source.sequence)]``.
        * *new_aa* is a valid standard amino acid character.
        * The resulting sequence still satisfies any domain constraints (e.g.
          conserved residues at positions 23, 36, 104).

        The ``validation_result`` is copied from *source* without re-running
        checks, so it may be stale if the mutation breaks a conserved site.
        """
        seq_list = list(source.sequence)
        # ``position`` is 1-based IMGT position; convert to 0-based index.
        seq_list[position - 1] = new_aa.upper()
        mutated = object.__new__(cls)
        mutated.sequence = "".join(seq_list)
        mutated.length = len(mutated.sequence)
        mutated.imgt_numbered = {i + 1: aa for i, aa in enumerate(mutated.sequence[:_MAX_IMGT_POS])}
        mutated.validation_result = source.validation_result  # skip re-validation
        return mutated

    # ------------------------------------------------------------------
    # IMGT numbering (identity mapping, capped at 128)
    # ------------------------------------------------------------------

    def _imgt_number(self) -> dict[int, str]:
        return {i + 1: aa for i, aa in enumerate(self.sequence[:_MAX_IMGT_POS])}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> dict:
        errors: list[str] = []
        warnings: list[str] = []

        # Check for invalid characters.
        invalid = {aa for aa in self.sequence if aa not in AMINO_ACIDS}
        if invalid:
            errors.append(f"Invalid amino acid(s): {', '.join(sorted(invalid))}")

        # Length check.
        if not (_MIN_LENGTH <= self.length <= _MAX_LENGTH):
            errors.append(f"Length {self.length} outside valid range ({_MIN_LENGTH}-{_MAX_LENGTH})")

        # Conserved residue warnings (only when residues are available).
        if self.imgt_numbered.get(23) != "C":
            warnings.append("Missing conserved Cys at IMGT position 23")
        if self.imgt_numbered.get(104) != "C":
            warnings.append("Missing conserved Cys at IMGT position 104")
        if self.imgt_numbered.get(36) != "W":
            warnings.append("Missing conserved Trp at IMGT position 36")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Cached region / position properties
    # ------------------------------------------------------------------

    @cached_property
    def regions(self) -> dict[str, tuple[int, int, str]]:
        """Map each IMGT region to ``(start, end, subsequence)``."""
        result: dict[str, tuple[int, int, str]] = {}
        for name, (start, end) in IMGT_REGIONS.items():
            subseq = "".join(self.imgt_numbered.get(pos, "") for pos in range(start, end + 1))
            result[name] = (start, end, subseq)
        return result

    @cached_property
    def cdr_positions(self) -> frozenset[int]:
        """All IMGT positions belonging to CDR regions."""
        positions: set[int] = set()
        for name, (start, end) in IMGT_REGIONS.items():
            if name.startswith("CDR"):
                positions.update(range(start, end + 1))
        return frozenset(positions)

    @cached_property
    def framework_positions(self) -> frozenset[int]:
        """All IMGT positions belonging to framework regions."""
        positions: set[int] = set()
        for name, (start, end) in IMGT_REGIONS.items():
            if name.startswith("FR"):
                positions.update(range(start, end + 1))
        return frozenset(positions)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"VHHSequence(length={self.length}, valid={self.validation_result['valid']})"

    def __len__(self) -> int:
        return self.length
