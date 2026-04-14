"""Tests for vhh_library.sequence – VHHSequence class."""

from __future__ import annotations

import pytest

from vhh_library.sequence import VHHSequence

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


class TestVHHSequenceValidation:
    def test_valid_vhh(self, vhh: VHHSequence) -> None:
        assert vhh.validation_result["valid"] is True

    def test_invalid_too_short(self) -> None:
        short = VHHSequence("QVQLVES")
        assert short.validation_result["valid"] is False
        assert len(short.validation_result["errors"]) > 0

    def test_invalid_too_long(self) -> None:
        long = VHHSequence("Q" * 200)
        assert long.validation_result["valid"] is False


class TestIMGTNumbering:
    def test_imgt_numbering(self, vhh: VHHSequence) -> None:
        numbered = vhh.imgt_numbered
        assert isinstance(numbered, dict)
        assert numbered[1] == "Q"


class TestRegions:
    def test_get_regions(self, vhh: VHHSequence) -> None:
        regions = vhh.regions
        assert isinstance(regions, dict)
        expected_names = {"FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"}
        assert set(regions.keys()) == expected_names


class TestCDRPositions:
    def test_cdr_positions(self, vhh: VHHSequence) -> None:
        cdr = vhh.cdr_positions
        assert isinstance(cdr, (frozenset, set))
        assert len(cdr) > 0
        assert 26 in cdr


class TestLength:
    def test_length(self, vhh: VHHSequence) -> None:
        assert vhh.length == len(SAMPLE_VHH)
