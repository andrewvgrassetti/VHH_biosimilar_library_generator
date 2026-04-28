"""Tests for the two-part assembly (yeast surface display) module."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import make_mock_vhh
from vhh_library.two_part_assembly import combine_parts, lock_overlap_positions, split_mutations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mutations(positions: list[str], imgt_pos_col: str = "imgt_pos") -> pd.DataFrame:
    """Build a minimal ranked-mutations DataFrame for testing."""
    rows = []
    for pos in positions:
        rows.append(
            {
                "position": int(pos),
                "imgt_pos": pos,
                "original_aa": "A",
                "suggested_aa": "G",
                "combined_score": 0.5,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# lock_overlap_positions
# ---------------------------------------------------------------------------


class TestLockOverlapPositions:
    def test_basic_lock(self):
        positions = [str(i) for i in range(1, 21)]
        locked = lock_overlap_positions("8", "13", positions)
        assert isinstance(locked, set)
        assert locked == {"8", "9", "10", "11", "12", "13"}

    def test_overlap_at_start(self):
        positions = [str(i) for i in range(1, 21)]
        locked = lock_overlap_positions("1", "4", positions)
        assert "1" in locked
        assert locked == {"1", "2", "3", "4"}

    def test_overlap_at_end(self):
        positions = [str(i) for i in range(1, 21)]
        locked = lock_overlap_positions("17", "20", positions)
        assert "20" in locked
        assert locked == {"17", "18", "19", "20"}

    def test_invalid_n_boundary(self):
        positions = [str(i) for i in range(1, 11)]
        with pytest.raises(ValueError, match="N-terminal boundary.*not found"):
            lock_overlap_positions("99", "5", positions)

    def test_invalid_c_boundary(self):
        positions = [str(i) for i in range(1, 11)]
        with pytest.raises(ValueError, match="C-terminal boundary.*not found"):
            lock_overlap_positions("5", "99", positions)

    def test_n_after_c_raises(self):
        positions = [str(i) for i in range(1, 21)]
        with pytest.raises(ValueError, match="comes after"):
            lock_overlap_positions("13", "8", positions)

    def test_single_position(self):
        positions = [str(i) for i in range(1, 21)]
        locked = lock_overlap_positions("10", "10", positions)
        assert locked == {"10"}

    def test_odd_width_overlap(self):
        """Boundaries '56' to '62' = 7 residues (odd width, no restriction)."""
        positions = [str(i) for i in range(50, 70)]
        locked = lock_overlap_positions("56", "62", positions)
        assert locked == {"56", "57", "58", "59", "60", "61", "62"}
        assert len(locked) == 7


# ---------------------------------------------------------------------------
# split_mutations
# ---------------------------------------------------------------------------


class TestSplitMutations:
    def test_basic_split(self):
        positions = [str(i) for i in range(1, 21)]
        muts = _make_mock_mutations(["3", "5", "10", "15", "18"])
        p1, p2 = split_mutations(muts, "10", positions)
        assert set(p1["imgt_pos"].astype(str)) == {"3", "5", "10"}
        assert set(p2["imgt_pos"].astype(str)) == {"15", "18"}

    def test_split_at_first_position(self):
        positions = [str(i) for i in range(1, 11)]
        muts = _make_mock_mutations(["1", "5", "9"])
        p1, p2 = split_mutations(muts, "1", positions)
        assert len(p1) == 1  # Only position "1"
        assert len(p2) == 2

    def test_split_at_last_position(self):
        positions = [str(i) for i in range(1, 11)]
        muts = _make_mock_mutations(["1", "5", "10"])
        p1, p2 = split_mutations(muts, "10", positions)
        assert len(p1) == 3  # All positions ≤ "10"
        assert len(p2) == 0

    def test_no_mutations_in_one_half(self):
        positions = [str(i) for i in range(1, 11)]
        muts = _make_mock_mutations(["1", "2", "3"])
        p1, p2 = split_mutations(muts, "5", positions)
        assert len(p1) == 3
        assert len(p2) == 0

    def test_invalid_split_position(self):
        positions = [str(i) for i in range(1, 11)]
        muts = _make_mock_mutations(["1", "5"])
        with pytest.raises(ValueError, match="not found"):
            split_mutations(muts, "99", positions)

    def test_uses_position_column_fallback(self):
        """When imgt_pos is absent, fall back to position column."""
        positions = [str(i) for i in range(1, 11)]
        muts = pd.DataFrame(
            [
                {"position": 3, "original_aa": "A", "suggested_aa": "G", "combined_score": 0.5},
                {"position": 7, "original_aa": "A", "suggested_aa": "G", "combined_score": 0.5},
            ]
        )
        p1, p2 = split_mutations(muts, "5", positions)
        assert len(p1) == 1
        assert len(p2) == 1


# ---------------------------------------------------------------------------
# combine_parts
# ---------------------------------------------------------------------------


class TestCombineParts:
    def _make_vhh_and_parts(self):
        """Create a mock VHH and simple Part 1 / Part 2 variant DataFrames."""
        vhh = make_mock_vhh()
        seq = vhh.sequence

        # Part 1 variant: mutate position 5 (index 4) from original to 'W'
        p1_seq = list(seq)
        p1_seq[4] = "W"
        p1_seq = "".join(p1_seq)

        # Part 2 variant: mutate position 80 (index 79) from original to 'Y'
        p2_seq = list(seq)
        p2_seq[79] = "Y"
        p2_seq = "".join(p2_seq)

        part1 = pd.DataFrame(
            [
                {"variant_id": "P1_V1", "mutations": f"{seq[4]}5W", "n_mutations": 1, "aa_sequence": p1_seq},
                {"variant_id": "P1_V2", "mutations": "", "n_mutations": 0, "aa_sequence": seq},
            ]
        )
        part2 = pd.DataFrame(
            [
                {"variant_id": "P2_V1", "mutations": f"{seq[79]}80Y", "n_mutations": 1, "aa_sequence": p2_seq},
                {"variant_id": "P2_V2", "mutations": "", "n_mutations": 0, "aa_sequence": seq},
            ]
        )
        return vhh, part1, part2

    def test_nxm_combinations(self):
        vhh, part1, part2 = self._make_vhh_and_parts()
        result = combine_parts(part1, part2, vhh, "50", n_boundary="48", c_boundary="53")
        assert len(result) == 4  # 2 × 2

    def test_columns_present(self):
        vhh, part1, part2 = self._make_vhh_and_parts()
        result = combine_parts(part1, part2, vhh, "50", n_boundary="48", c_boundary="53")
        for col in [
            "variant_id",
            "part1_id",
            "part2_id",
            "part1_mutations",
            "part2_mutations",
            "mutations",
            "n_mutations",
            "aa_sequence",
            "stability_score",
            "nativeness_score",
            "combined_score",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sequence_length_preserved(self):
        vhh, part1, part2 = self._make_vhh_and_parts()
        result = combine_parts(part1, part2, vhh, "50", n_boundary="48", c_boundary="53")
        for _, row in result.iterrows():
            assert len(row["aa_sequence"]) == len(vhh.sequence)

    def test_mutations_combined(self):
        vhh, part1, part2 = self._make_vhh_and_parts()
        result = combine_parts(part1, part2, vhh, "50", n_boundary="48", c_boundary="53")
        # Row where Part 1 mutant + Part 2 mutant
        combined_row = result[(result["part1_id"] == "P1_V1") & (result["part2_id"] == "P2_V1")]
        assert len(combined_row) == 1
        assert combined_row.iloc[0]["n_mutations"] == 2

    def test_wild_type_combination(self):
        vhh, part1, part2 = self._make_vhh_and_parts()
        result = combine_parts(part1, part2, vhh, "50", n_boundary="48", c_boundary="53")
        wt_row = result[(result["part1_id"] == "P1_V2") & (result["part2_id"] == "P2_V2")]
        assert len(wt_row) == 1
        assert wt_row.iloc[0]["aa_sequence"] == vhh.sequence

    def test_invalid_split_position(self):
        vhh, part1, part2 = self._make_vhh_and_parts()
        with pytest.raises(ValueError, match="not found"):
            combine_parts(part1, part2, vhh, "999", n_boundary="48", c_boundary="53")

    def test_single_part1_single_part2(self):
        vhh = make_mock_vhh()
        part1 = pd.DataFrame([{"variant_id": "P1_V1", "mutations": "", "n_mutations": 0, "aa_sequence": vhh.sequence}])
        part2 = pd.DataFrame([{"variant_id": "P2_V1", "mutations": "", "n_mutations": 0, "aa_sequence": vhh.sequence}])
        result = combine_parts(part1, part2, vhh, "50", n_boundary="48", c_boundary="53")
        assert len(result) == 1
        assert result.iloc[0]["aa_sequence"] == vhh.sequence
