"""Tests for vhh_library.sequence – VHHSequence class."""

from __future__ import annotations

import pytest

from vhh_library.sequence import VHHSequence

# A realistic camelid VHH (nanobody) sequence that ANARCI can number.
# Based on a cAbBCII10-like nanobody.
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

    def test_anarci_failure_sets_invalid(self) -> None:
        """An unrecognisable sequence should be marked invalid, not crash."""
        bogus = VHHSequence("A" * 100)
        assert bogus.validation_result["valid"] is False
        # Should have an error message mentioning ANARCI or numbering.
        assert any(
            "ANARCI" in e or "number" in e.lower()
            for e in bogus.validation_result["errors"]
        )


class TestIMGTNumbering:
    def test_imgt_numbering(self, vhh: VHHSequence) -> None:
        numbered = vhh.imgt_numbered
        assert isinstance(numbered, dict)
        # String keys after ANARCI integration.
        assert numbered["1"] == "Q"

    def test_conserved_positions(self, vhh: VHHSequence) -> None:
        """Known conserved residues at IMGT positions 23 (Cys), 41 (Trp), 104 (Cys)."""
        assert vhh.imgt_numbered["23"] == "C"
        assert vhh.imgt_numbered["41"] == "W"
        assert vhh.imgt_numbered["104"] == "C"

    def test_gaps_present(self, vhh: VHHSequence) -> None:
        """IMGT numbering should contain gaps for positions absent in the sequence."""
        # Position 10 is typically a gap in VHH IMGT numbering.
        assert "10" not in vhh.imgt_numbered

    def test_cdr_boundaries(self, vhh: VHHSequence) -> None:
        """CDR1 spans IMGT positions 26–35; verify some residues fall in that region."""
        cdr1_positions = {str(p) for p in range(26, 36)}
        cdr1_residues = {
            pos: vhh.imgt_numbered[pos]
            for pos in cdr1_positions
            if pos in vhh.imgt_numbered
        }
        assert len(cdr1_residues) > 0


class TestChainTypeAndSpecies:
    def test_chain_type_is_h(self, vhh: VHHSequence) -> None:
        assert vhh.chain_type == "H"

    def test_species_detected(self, vhh: VHHSequence) -> None:
        # The sample is a camelid (alpaca/llama) nanobody.
        assert vhh.species in {"alpaca", "llama", "human", "mouse"}


class TestRegions:
    def test_get_regions(self, vhh: VHHSequence) -> None:
        regions = vhh.regions
        assert isinstance(regions, dict)
        expected_names = {"FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"}
        assert set(regions.keys()) == expected_names

    def test_region_subsequences_are_nonempty(self, vhh: VHHSequence) -> None:
        for name, (_start, _end, subseq) in vhh.regions.items():
            assert len(subseq) > 0, f"Region {name} has empty subsequence"


class TestCDRPositions:
    def test_cdr_positions(self, vhh: VHHSequence) -> None:
        cdr = vhh.cdr_positions
        assert isinstance(cdr, (frozenset, set))
        assert len(cdr) > 0
        assert "26" in cdr


class TestMutate:
    def test_mutate_preserves_numbering(self, vhh: VHHSequence) -> None:
        """mutate() should copy the parent numbering and skip ANARCI."""
        mutant = VHHSequence.mutate(vhh, 1, "E")
        assert mutant.sequence[0] == "E"
        assert mutant.imgt_numbered["1"] == "E"
        # Other positions remain unchanged.
        assert mutant.imgt_numbered["2"] == vhh.imgt_numbered["2"]
        # Chain type / species carried over.
        assert mutant.chain_type == vhh.chain_type
        assert mutant.species == vhh.species


class TestLength:
    def test_length(self, vhh: VHHSequence) -> None:
        assert vhh.length == len(SAMPLE_VHH)
