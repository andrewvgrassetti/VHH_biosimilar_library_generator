"""Shared fixtures for the VHH biosimilar library test suite."""

from __future__ import annotations

import pytest

from vhh_library.humanness import HumAnnotator
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
def hum_annotator() -> HumAnnotator:
    return HumAnnotator()


@pytest.fixture(scope="session")
def stability_scorer() -> StabilityScorer:
    return StabilityScorer(use_nanomelt=False)
