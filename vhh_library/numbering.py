"""ANARCI-backed IMGT numbering helpers for VHH sequences."""

from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_SPECIES = ["human", "mouse", "alpaca", "llama"]


class NumberingError(ValueError):
    """Raised when ANARCI numbering cannot be assigned to a VH/VHH sequence."""


class NumberingBackendUnavailable(NumberingError):
    """Raised when ANARCI backend tooling is unavailable in the runtime."""


@dataclass(frozen=True)
class NumberingResult:
    """Result of antibody numbering.

    ``numbered`` uses IMGT positions as keys. Integer keys represent canonical
    IMGT positions (e.g. ``111``), while insertion codes are represented with
    string keys (e.g. ``"111A"``, ``"111B"``).
    """

    numbered: dict[int | str, str]
    chain_type: str
    species: str
    v_gene: str | None
    scheme: str


def _format_imgt_key(position: tuple[int, str] | int) -> int | str:
    if isinstance(position, tuple):
        pos, insertion = position
    else:
        pos, insertion = position, ""
    insertion = insertion.strip()
    return pos if insertion == "" else f"{pos}{insertion}"


def number_sequence(sequence: str) -> NumberingResult:
    """Assign IMGT numbering with ANARCI for a single sequence."""
    try:
        from anarci import anarci as anarci_run
    except ImportError as exc:  # pragma: no cover - dependency-level failure
        raise NumberingError(
            "ANARCI is required for IMGT numbering. Install project dependencies including `anarci`."
        ) from exc

    try:
        numbered, alignment_details, _ = anarci_run(
            [("input", sequence)],
            scheme="imgt",
            allowed_species=_ALLOWED_SPECIES,
        )
    except (FileNotFoundError, OSError) as exc:
        raise NumberingBackendUnavailable(
            "ANARCI backend is unavailable (missing external HMMER tooling such as `hmmscan`)."
        ) from exc

    if not numbered or numbered[0] is None:
        raise NumberingError(
            "ANARCI could not assign IMGT numbering to this sequence — it may not be a valid VH/VHH domain."
        )

    detail = alignment_details[0][0]
    raw_chain_type = detail.get("chain_type", "unknown")
    if raw_chain_type != "H":
        raise NumberingError(
            f"ANARCI assigned chain type '{raw_chain_type}', but VH/VHH (heavy-chain) input is required."
        )

    numbering_tuples = numbered[0][0][0]
    parsed: dict[int | str, str] = {}
    for position, aa in numbering_tuples:
        if aa == "-":
            continue
        parsed[_format_imgt_key(position)] = aa

    return NumberingResult(
        numbered=parsed,
        chain_type="VH",
        species=detail.get("species", "unknown"),
        v_gene=None,
        scheme="imgt",
    )
