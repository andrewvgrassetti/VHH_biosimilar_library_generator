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


class TestHosts:
    @pytest.mark.parametrize("host", ["s_cerevisiae", "p_pastoris", "h_sapiens"])
    def test_host(self, optimizer: CodonOptimizer, host: str) -> None:
        result = optimizer.optimize(SHORT_AA, host=host)
        assert "dna_sequence" in result
        assert len(result["dna_sequence"]) == len(SHORT_AA) * 3
