"""Shared fixtures for the VHH biosimilar library test suite."""

from __future__ import annotations

import pytest

from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture(scope="session")
def sample_vhh_seq() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


@pytest.fixture(scope="session")
def stability_scorer() -> StabilityScorer:
    return StabilityScorer()


def make_mock_vhh(sequence: str | None = None) -> VHHSequence:
    """Create a VHHSequence with pre-populated IMGT numbering (no ANARCI).

    This helper bypasses ANARCI entirely by using ``object.__new__`` and
    manually setting all attributes.  Use for any test that does not need
    real ANARCI/HMMER numbering.
    """
    if sequence is None:
        sequence = SAMPLE_VHH

    vhh = object.__new__(VHHSequence)
    vhh.sequence = sequence
    vhh.length = len(sequence)
    vhh.strict = True
    vhh.chain_type = "H"
    vhh.species = "alpaca"

    # Build synthetic IMGT numbering — one key per residue, sequential
    numbered: dict[str, str] = {}
    pos_to_idx: dict[str, int] = {}
    for idx, aa in enumerate(sequence):
        key = str(idx + 1)
        numbered[key] = aa
        pos_to_idx[key] = idx
    vhh.imgt_numbered = numbered
    vhh._pos_to_seq_idx = pos_to_idx
    vhh.validation_result = {"valid": True, "errors": [], "warnings": []}
    return vhh


@pytest.fixture()
def mock_vhh() -> VHHSequence:
    """Return a VHHSequence-like object with pre-populated numbering, no ANARCI needed."""
    return make_mock_vhh()
