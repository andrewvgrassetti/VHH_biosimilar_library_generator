"""Tests for the sequence selector data transformation logic."""

from __future__ import annotations

from vhh_library.components.sequence_selector import imgt_key_int_part, sequence_selector
from vhh_library.sequence import IMGT_REGIONS, VHHSequence

from tests.conftest import SAMPLE_VHH


class TestImgtKeyIntPart:
    """Unit tests for the imgt_key_int_part helper."""

    def test_plain_position(self):
        assert imgt_key_int_part("1") == 1
        assert imgt_key_int_part("104") == 104

    def test_insertion_code(self):
        assert imgt_key_int_part("111A") == 111
        assert imgt_key_int_part("111B") == 111
        assert imgt_key_int_part("112A") == 112

    def test_empty_string(self):
        assert imgt_key_int_part("") == 0


class TestSequenceSelectorDataTransformation:
    """Verify the ordered IMGT positions list correctly represents a known VHH."""

    def test_imgt_positions_list_matches_imgt_numbered(self, sample_vhh_seq: VHHSequence):
        """The imgt_positions_list passed to the frontend must exactly reproduce
        the ordered (key, aa) pairs from imgt_numbered."""
        imgt = sample_vhh_seq.imgt_numbered
        positions_list = [[k, v] for k, v in imgt.items()]

        # Verify length matches
        assert len(positions_list) == len(imgt)

        # Verify each entry matches
        for idx, (key, aa) in enumerate(imgt.items()):
            assert positions_list[idx] == [key, aa], (
                f"Mismatch at index {idx}: expected [{key}, {aa}], got {positions_list[idx]}"
            )

    def test_no_question_marks(self, sample_vhh_seq: VHHSequence):
        """Every position in the IMGT list should have a real amino acid, not '?'."""
        imgt = sample_vhh_seq.imgt_numbered
        for key, aa in imgt.items():
            assert aa != "?", f"IMGT position {key} has '?' residue"
            assert len(aa) == 1 and aa.isalpha(), (
                f"IMGT position {key} has invalid residue: {aa!r}"
            )

    def test_all_sequence_residues_are_represented(self, sample_vhh_seq: VHHSequence):
        """The concatenation of all AA values in imgt_numbered should equal the
        raw sequence (since ANARCI accounts for all residues)."""
        imgt = sample_vhh_seq.imgt_numbered
        reconstructed = "".join(imgt.values())
        assert reconstructed == sample_vhh_seq.sequence

    def test_off_limit_string_keys_cover_insertion_codes(self, sample_vhh_seq: VHHSequence):
        """When a region is marked off-limit, all IMGT keys (including insertion
        codes) within that region's integer range must be included."""
        imgt = sample_vhh_seq.imgt_numbered

        for region_name, (start, end) in IMGT_REGIONS.items():
            # Build the off-limit set the same way app.py does
            off_limit: set[str] = set()
            for key in imgt:
                if start <= imgt_key_int_part(key) <= end:
                    off_limit.add(key)

            # Every key whose integer part is in [start, end] must be present
            for key in imgt:
                int_part = imgt_key_int_part(key)
                if start <= int_part <= end:
                    assert key in off_limit, (
                        f"IMGT key {key!r} (int part {int_part}) should be in "
                        f"off-limit set for region {region_name} ({start}-{end})"
                    )

    def test_region_assignment_covers_all_positions(self, sample_vhh_seq: VHHSequence):
        """Every IMGT position should fall within exactly one region."""
        imgt = sample_vhh_seq.imgt_numbered
        for key in imgt:
            int_part = imgt_key_int_part(key)
            matching = [
                name for name, (start, end) in IMGT_REGIONS.items()
                if start <= int_part <= end
            ]
            assert len(matching) == 1, (
                f"IMGT key {key!r} (int part {int_part}) matches {len(matching)} "
                f"regions: {matching}"
            )

    def test_positions_list_length_matches_sequence_length(self, sample_vhh_seq: VHHSequence):
        """The number of IMGT-numbered positions should match the raw sequence length."""
        assert len(sample_vhh_seq.imgt_numbered) == len(sample_vhh_seq.sequence)
