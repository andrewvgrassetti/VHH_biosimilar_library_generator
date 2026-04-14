"""Tests for vhh_library.barcodes – BarcodeGenerator class."""

from __future__ import annotations

import pandas as pd
import pytest

from vhh_library.barcodes import BarcodeGenerator, _barcode_passes_rules
from vhh_library.utils import tryptic_digest

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def gen() -> BarcodeGenerator:
    return BarcodeGenerator()


def _make_library(n: int = 5) -> pd.DataFrame:
    """Create a small mock library DataFrame."""
    return pd.DataFrame(
        {
            "variant_id": [f"V{i:04d}" for i in range(n)],
            "aa_sequence": [SAMPLE_VHH] * n,
            "combined_score": [0.9 - i * 0.01 for i in range(n)],
        }
    )


class TestBarcodeRules:
    def test_valid_barcode(self) -> None:
        assert _barcode_passes_rules("KAFEDLVK") is True

    def test_invalid_too_short(self) -> None:
        assert _barcode_passes_rules("KAFK") is False

    def test_invalid_internal_k(self) -> None:
        assert _barcode_passes_rules("KAKELYR") is False

    def test_invalid_no_leading_kr(self) -> None:
        assert _barcode_passes_rules("AAFEDLVK") is False

    def test_invalid_contains_met(self) -> None:
        assert _barcode_passes_rules("KAFMDLVK") is False

    def test_invalid_contains_cys(self) -> None:
        assert _barcode_passes_rules("KAFCDLVK") is False


class TestTrypticDigest:
    def test_basic_cleavage(self) -> None:
        peptides = tryptic_digest("ABCKDEFR", missed_cleavages=0)
        assert isinstance(peptides, list)
        assert len(peptides) > 0
        # After K and R: "ABCK", "DEFR"
        assert "ABCK" in peptides
        assert "DEFR" in peptides


class TestPool:
    def test_pool_size(self, gen: BarcodeGenerator) -> None:
        assert len(gen.pool) > 250

    def test_majority_pass_rules(self, gen: BarcodeGenerator) -> None:
        passing = [bc for bc in gen.pool if _barcode_passes_rules(bc)]
        # The pre-computed pool may include edge-case barcodes; at least 90 %
        # should satisfy the design rules.
        ratio = len(passing) / len(gen.pool)
        assert ratio >= 0.90, f"Only {ratio:.1%} of pool barcodes pass rules"

    def test_all_unique(self, gen: BarcodeGenerator) -> None:
        assert len(gen.pool) == len(set(gen.pool))


class TestAssignBarcodes:
    def test_returns_dataframe(self, gen: BarcodeGenerator) -> None:
        lib = _make_library(5)
        result = gen.assign_barcodes(lib, top_n=5)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"barcode_id", "barcode_peptide", "barcoded_sequence", "barcode_tryptic_peptide"}
        assert expected_cols.issubset(set(result.columns))

    def test_correct_row_count(self, gen: BarcodeGenerator) -> None:
        lib = _make_library(3)
        result = gen.assign_barcodes(lib, top_n=3)
        assert len(result) == 3

    def test_unique_ids(self, gen: BarcodeGenerator) -> None:
        lib = _make_library(5)
        result = gen.assign_barcodes(lib, top_n=5)
        assert result["barcode_id"].nunique() == len(result)


class TestFASTA:
    def test_correct_format(self, gen: BarcodeGenerator) -> None:
        lib = _make_library(2)
        barcoded = gen.assign_barcodes(lib, top_n=2)
        fasta = gen.generate_barcoded_fasta(barcoded)
        lines = fasta.strip().split("\n")
        # Should have pairs of header + sequence lines
        assert len(lines) == 4
        assert lines[0].startswith(">")
        assert not lines[1].startswith(">")


class TestReference:
    def test_expected_columns(self, gen: BarcodeGenerator) -> None:
        lib = _make_library(3)
        barcoded = gen.assign_barcodes(lib, top_n=3)
        ref = gen.generate_barcode_reference(barcoded)
        expected = {
            "variant_id",
            "barcode_id",
            "barcode_peptide",
            "barcode_tryptic_peptide",
            "neutral_mass_da",
            "mz_1plus",
            "mz_2plus",
            "mz_3plus",
            "hydrophobicity",
            "source",
        }
        assert expected.issubset(set(ref.columns))

    def test_mz_ordering(self, gen: BarcodeGenerator) -> None:
        lib = _make_library(3)
        barcoded = gen.assign_barcodes(lib, top_n=3)
        ref = gen.generate_barcode_reference(barcoded)
        for _, row in ref.iterrows():
            assert row["mz_1plus"] >= row["mz_2plus"] >= row["mz_3plus"]
