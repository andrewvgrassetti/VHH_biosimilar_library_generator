"""Thin wrapper around ANARCI for IMGT antibody numbering.

This module provides :func:`number_sequence`, the single entry-point used by
:class:`~vhh_library.sequence.VHHSequence` to obtain IMGT-numbered residue
maps from raw amino-acid strings.

**Insertion-code convention** – ANARCI returns IMGT positions as
``(int, str)`` tuples where the second element is an insertion code
(e.g. ``(111, 'A')``).  We convert these to **string keys**: plain
positions become ``"1"``, ``"2"``, … and insertions become ``"111A"``,
``"111B"``, etc.  All downstream code that accesses the numbered dict
should use ``str`` keys.
"""

from __future__ import annotations

import importlib
from typing import NamedTuple

# ---------------------------------------------------------------------------
# ANARCI / BioPython compatibility patch
# ---------------------------------------------------------------------------
# BioPython >= 1.83 sometimes returns ``None`` for ``query_start`` /
# ``query_end`` on HMMER 3.4 HSP objects, which crashes ANARCI's
# ``_domains_are_same`` and ``_parse_hmmer_query``.
#
# BioPython 1.87 additionally returns ``None`` for ``hit_start`` /
# ``hit_end`` on HSP fragments.  Python slicing tolerates ``None``
# (``list[None:None]`` gives the full slice), so most of ANARCI is
# unaffected.  However, the forward-extension branch in
# ``_hmm_alignment_to_states`` compares ``_hmm_end`` (from
# ``hsp.hit_end``) against integer constants, which raises ``TypeError``
# when the value is ``None``.  The same comparison also uses
# ``_hmm_length`` from ``get_hmm_length()``, which may return ``None``.
#
# We apply a targeted monkey-patch **before** the first ``anarci()`` call.

_PATCHED = False


def _apply_anarci_compat_patch() -> None:
    """Patch ANARCI internals to tolerate ``None`` coordinates from BioPython.

    Fixes applied:

    1. ``_domains_are_same`` -- guards ``query_start`` / ``query_end``
       being ``None`` (falls back to ``env_start`` / ``env_end``).
    2. ``_parse_hmmer_query`` -- back-fills ``None`` ``query_start`` /
       ``query_end`` on HSP fragments using the envelope coordinates.
    3. ``_hmm_alignment_to_states`` -- guards the forward-extension branch
       (line ~420) against ``_hmm_end`` or ``_hmm_length`` being ``None``
       (avoids ``TypeError`` from ``123 < _hmm_end < _hmm_length``).
       ``hit_start`` / ``hit_end`` are intentionally left as ``None``
       because Python slicing handles them correctly and they carry
       different semantics to the envelope coordinates.
    """
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return

    anarci_mod = importlib.import_module("anarci.anarci")

    # --- patch _domains_are_same -------------------------------------------
    def _patched_domains_are_same(dom1, dom2):  # type: ignore[no-untyped-def]
        def _start(d):  # type: ignore[no-untyped-def]
            return d.query_start if d.query_start is not None else getattr(d, "env_start", 0)

        def _end(d):  # type: ignore[no-untyped-def]
            return d.query_end if d.query_end is not None else getattr(d, "env_end", 0)

        s1, s2 = _start(dom1), _start(dom2)
        e1, e2 = _end(dom1), _end(dom2)
        if s1 <= s2:
            return s2 < e1
        return s1 < e2

    anarci_mod._domains_are_same = _patched_domains_are_same  # type: ignore[attr-defined]

    # --- patch _parse_hmmer_query ------------------------------------------
    _original_parse = anarci_mod._parse_hmmer_query  # type: ignore[attr-defined]

    def _patched_parse(query, bit_score_threshold=80, hmmer_species=None):  # type: ignore[no-untyped-def]
        for hsp in query.hsps:
            if hsp.query_start is None and hasattr(hsp, "env_start"):
                for frag in hsp.fragments:
                    if frag.query_start is None:
                        frag.query_start = hsp.env_start
                    if frag.query_end is None:
                        frag.query_end = hsp.env_end
        return _original_parse(
            query,
            bit_score_threshold=bit_score_threshold,
            hmmer_species=hmmer_species,
        )

    anarci_mod._parse_hmmer_query = _patched_parse  # type: ignore[attr-defined]

    # --- patch _hmm_alignment_to_states ------------------------------------
    # Guard the forward-extension branch (line ~420) against ``_hmm_end``
    # or ``_hmm_length`` being ``None``.  The comparison
    #     ``123 < _hmm_end < _hmm_length``
    # raises ``TypeError`` when either value is ``None``.  When that
    # happens the extension is simply not applicable, so we catch the
    # ``TypeError`` and re-invoke the original with a ``seq_length`` of 0
    # to disable both extension branches (they require
    # ``_seq_end < seq_length``).
    _original_hmm_align = anarci_mod._hmm_alignment_to_states  # type: ignore[attr-defined]

    def _patched_hmm_alignment_to_states(hsp, n, seq_length):  # type: ignore[no-untyped-def]
        try:
            return _original_hmm_align(hsp, n, seq_length)
        except TypeError as exc:
            if "NoneType" not in str(exc):
                raise
            # Disable both n-terminal and c-terminal extension branches by
            # making ``_seq_end < seq_length`` always ``False``.
            return _original_hmm_align(hsp, n, 0)

    anarci_mod._hmm_alignment_to_states = _patched_hmm_alignment_to_states  # type: ignore[attr-defined]

    _PATCHED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VH_CHAIN_TYPES = frozenset({"H"})


class NumberingResult(NamedTuple):
    """Result of ANARCI numbering for a single sequence.

    Attributes
    ----------
    numbered : dict[str, str]
        IMGT position (string key) → amino-acid character.
        Plain positions use integer-like keys (``"1"``, ``"2"``, …).
        Insertion codes are appended directly (``"111A"``, ``"111B"``).
    chain_type : str
        Single-letter chain type returned by ANARCI (e.g. ``"H"``).
    species : str
        Species of the best-matching germline (e.g. ``"alpaca"``).
    v_gene : str | None
        V-gene identifier when available, otherwise ``None``.
    scheme : str
        Numbering scheme used (always ``"imgt"`` in this wrapper).
    """

    numbered: dict[str, str]
    chain_type: str
    species: str
    v_gene: str | None
    scheme: str


class NumberingError(Exception):
    """Raised when ANARCI cannot number a sequence."""


def number_sequence(sequence: str) -> NumberingResult:
    """Number an antibody *sequence* using ANARCI with the IMGT scheme.

    Parameters
    ----------
    sequence:
        Raw amino-acid string (case-insensitive, will be uppercased).

    Returns
    -------
    NumberingResult
        A named tuple with ``numbered``, ``chain_type``, ``species``,
        ``v_gene``, and ``scheme`` fields.

    Raises
    ------
    NumberingError
        If ANARCI cannot recognise or number the input sequence, or if
        the detected chain type is not VH/VHH.
    """
    _apply_anarci_compat_patch()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from anarci import anarci as run_anarci

    sequence = sequence.upper().strip()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run_anarci(
                [("input", sequence)],
                scheme="imgt",
                allowed_species=["human", "mouse", "alpaca", "llama"],
            )
    except Exception as exc:
        raise NumberingError(f"ANARCI failed: {exc}") from exc

    # ANARCI returns a 3-tuple: (numbering, alignment_details, hit_tables).
    numbering_list = results[0]
    alignment_details = results[1]

    # ANARCI returns None for the first element when it cannot number the seq.
    if numbering_list[0] is None or alignment_details[0] is None:
        raise NumberingError("ANARCI could not number the input sequence (no significant HMM hit).")

    # Parse alignment metadata from the best-scoring domain hit.
    detail = alignment_details[0][0]  # dict for best domain
    chain_type: str = detail.get("chain_type", "")
    species: str = detail.get("species", "")
    hit_id: str = detail.get("id", "")

    if chain_type not in _VH_CHAIN_TYPES:
        raise NumberingError(
            f"Input sequence is chain type '{chain_type}' (hit: {hit_id}), "
            f"but only VH/VHH chains (type 'H') are accepted."
        )

    # Parse the numbered residues into a dict[str, str].
    domain = numbering_list[0][0][0]  # first sequence, first domain
    numbered: dict[str, str] = {}
    for (pos_int, insertion_code), aa in domain:
        if aa == "-":
            continue
        insertion_code_stripped = insertion_code.strip()
        key = f"{pos_int}{insertion_code_stripped}" if insertion_code_stripped else str(pos_int)
        numbered[key] = aa

    return NumberingResult(
        numbered=numbered,
        chain_type=chain_type,
        species=species,
        v_gene=None,  # ANARCI does not return V-gene directly
        scheme="imgt",
    )
