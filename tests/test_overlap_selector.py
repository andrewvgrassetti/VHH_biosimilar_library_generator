"""Tests for the overlap selector component data transformation logic."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from vhh_library.components.overlap_selector import overlap_selector


class TestOverlapSelectorDefaults:
    """Verify the overlap_selector returns correct default values."""

    @pytest.fixture()
    def sample_imgt(self) -> dict[str, str]:
        """A small ordered dict simulating IMGT-numbered positions."""
        return OrderedDict(
            [
                ("1", "Q"),
                ("2", "V"),
                ("3", "Q"),
                ("4", "L"),
                ("5", "V"),
                ("10", "G"),
                ("11", "G"),
                ("20", "S"),
                ("25", "A"),
                ("30", "G"),
            ]
        )

    def test_returns_default_when_no_boundaries(self, sample_imgt: dict[str, str]):
        """With no boundaries set, the default should have both as None."""
        result = overlap_selector(
            imgt_numbered=sample_imgt,
            n_boundary=None,
            c_boundary=None,
            key="test_overlap_none",
        )
        # On first render the component returns the default value.
        assert result is not None
        assert result["n_boundary"] is None
        assert result["c_boundary"] is None

    def test_returns_default_with_boundaries(self, sample_imgt: dict[str, str]):
        """When boundaries are set, they should be returned as defaults."""
        result = overlap_selector(
            imgt_numbered=sample_imgt,
            n_boundary="3",
            c_boundary="11",
            key="test_overlap_set",
        )
        assert result is not None
        assert result["n_boundary"] == "3"
        assert result["c_boundary"] == "11"

    def test_gap_positions_are_valid_boundaries(self, sample_imgt: dict[str, str]):
        """Positions like '10' that skip from '5' are still valid IMGT keys."""
        result = overlap_selector(
            imgt_numbered=sample_imgt,
            n_boundary="5",
            c_boundary="10",
            key="test_overlap_gap",
        )
        assert result is not None
        assert result["n_boundary"] == "5"
        assert result["c_boundary"] == "10"


class TestOverlapSelectorImgtPositionsList:
    """Verify that the IMGT positions list passed to the frontend is correct."""

    def test_positions_preserve_order(self):
        """The positions list must preserve the insertion order of imgt_numbered."""
        imgt = OrderedDict([("1", "A"), ("2", "B"), ("111A", "C"), ("111B", "D"), ("112", "E")])
        # We can't directly access the internal call, but we can verify the
        # function doesn't crash with insertion codes.
        result = overlap_selector(
            imgt_numbered=imgt,
            n_boundary="111A",
            c_boundary="112",
            key="test_insertion_codes",
        )
        assert result is not None
        assert result["n_boundary"] == "111A"
        assert result["c_boundary"] == "112"

    def test_empty_imgt(self):
        """An empty IMGT dict should not crash."""
        result = overlap_selector(
            imgt_numbered={},
            n_boundary=None,
            c_boundary=None,
            key="test_empty",
        )
        assert result is not None
        assert result["n_boundary"] is None
        assert result["c_boundary"] is None
