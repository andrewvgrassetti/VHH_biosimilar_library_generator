"""Tests for vhh_library.codon_optimizer – CodonOptimizer class."""

from __future__ import annotations

import pytest

from vhh_library.codon_optimizer import CodonOptimizer
from vhh_library.utils import translate

SHORT_AA = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGK"


@pytest.fixture
def optimizer() -> CodonOptimizer:
    return CodonOptimizer()


class TestOptimize:
    def test_optimize_returns_dict(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA)
        assert "dna_sequence" in result

    def test_dna_translates_correctly(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA)
        translated = translate(result["dna_sequence"])
        assert translated == SHORT_AA

    def test_gc_content_range(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA)
        assert 0.0 <= result["gc_content"] <= 1.0

    def test_no_stop_codons_in_middle(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA)
        dna = result["dna_sequence"]
        codons = [dna[i : i + 3] for i in range(0, len(dna) - 3, 3)]
        stop_codons = {"TAA", "TAG", "TGA"}
        for codon in codons:
            assert codon not in stop_codons

    def test_result_contains_organism(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA)
        assert "organism" in result
        assert isinstance(result["organism"], str)

    def test_most_frequent_short_backtranslates(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize("MKTL", "e_coli", "most_frequent")
        assert translate(result["dna_sequence"]) == "MKTL"

    def test_most_frequent_cai_is_one(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize("MKTL", "e_coli", "most_frequent")
        assert result["cai"] == pytest.approx(1.0)


class TestHosts:
    @pytest.mark.parametrize("host", ["s_cerevisiae", "p_pastoris", "h_sapiens"])
    def test_host(self, optimizer: CodonOptimizer, host: str) -> None:
        result = optimizer.optimize(SHORT_AA, host=host)
        assert "dna_sequence" in result
        assert len(result["dna_sequence"]) == len(SHORT_AA) * 3

    def test_invalid_organism_raises(self, optimizer: CodonOptimizer) -> None:
        with pytest.raises(ValueError, match="Unknown organism"):
            optimizer.optimize("MKTL", host="totally_invalid_organism_xyz")


class TestDnaChiselOptimized:
    def test_backtranslates(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA, host="e_coli", strategy="dnachisel_optimized")
        assert translate(result["dna_sequence"]) == SHORT_AA

    def test_no_bamhi_ecori_sites(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA, host="e_coli", strategy="dnachisel_optimized")
        dna = result["dna_sequence"]
        assert "GGATCC" not in dna, "BamHI site found in optimized sequence"
        assert "GAATTC" not in dna, "EcoRI site found in optimized sequence"

    def test_gc_content_within_range(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA, host="e_coli", strategy="dnachisel_optimized")
        assert 0.30 <= result["gc_content"] <= 0.65

    def test_constraints_passed_field(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(SHORT_AA, host="e_coli", strategy="dnachisel_optimized")
        assert "constraints_passed" in result
        assert isinstance(result["constraints_passed"], bool)
        assert "constraint_violations" in result

    def test_h_sapiens_dnachisel(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize("MKTL", host="h_sapiens", strategy="dnachisel_optimized")
        assert translate(result["dna_sequence"]) == "MKTL"

    def test_custom_restriction_enzymes(self, optimizer: CodonOptimizer) -> None:
        result = optimizer.optimize(
            SHORT_AA, host="e_coli", strategy="dnachisel_optimized",
            restriction_enzymes=["BamHI", "EcoRI"],
        )
        dna = result["dna_sequence"]
        assert "GGATCC" not in dna
        assert "GAATTC" not in dna


class TestFallbackTable:
    def test_p_pastoris_uses_fallback(self, optimizer: CodonOptimizer) -> None:
        """p_pastoris is not bundled in python-codon-tables; verify the embedded fallback works."""
        result = optimizer.optimize("MKTL", host="p_pastoris", strategy="most_frequent")
        assert translate(result["dna_sequence"]) == "MKTL"
        assert result["organism"] == "p_pastoris"
