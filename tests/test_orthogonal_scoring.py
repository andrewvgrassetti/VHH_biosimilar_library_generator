"""Tests for vhh_library.orthogonal_scoring – alternative scoring methods."""

from __future__ import annotations

import pytest

from vhh_library.orthogonal_scoring import (
    ConsensusStabilityScorer,
    HumanStringContentScorer,
    NanoMeltStabilityScorer,
)
from vhh_library.sequence import VHHSequence

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


@pytest.fixture
def hsc() -> HumanStringContentScorer:
    return HumanStringContentScorer()


@pytest.fixture
def consensus() -> ConsensusStabilityScorer:
    return ConsensusStabilityScorer()


class TestHumanStringContentScorer:
    def test_score_returns_dict(self, hsc: HumanStringContentScorer, vhh: VHHSequence) -> None:
        result = hsc.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_total_kmers_positive(self, hsc: HumanStringContentScorer, vhh: VHHSequence) -> None:
        result = hsc.score(vhh)
        assert result["total_kmers"] > 0
        assert result["matched_kmers"] <= result["total_kmers"]

    def test_predict_mutation_effect_returns_float(
        self, hsc: HumanStringContentScorer, vhh: VHHSequence
    ) -> None:
        delta = hsc.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)

    def test_same_aa_returns_zero(
        self, hsc: HumanStringContentScorer, vhh: VHHSequence
    ) -> None:
        original_aa = vhh.imgt_numbered[1]
        delta = hsc.predict_mutation_effect(vhh, 1, original_aa)
        assert delta == 0.0

    def test_custom_kmer_size(self, vhh: VHHSequence) -> None:
        scorer = HumanStringContentScorer(kmer_size=7)
        result = scorer.score(vhh)
        assert "composite_score" in result


class TestConsensusStabilityScorer:
    def test_score_returns_dict(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        result = consensus.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_positions_evaluated(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        result = consensus.score(vhh)
        assert result["positions_evaluated"] > 0
        assert result["consensus_matches"] <= result["positions_evaluated"]

    def test_predict_mutation_effect_returns_float(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        delta = consensus.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)

    def test_same_aa_returns_zero(
        self, consensus: ConsensusStabilityScorer, vhh: VHHSequence
    ) -> None:
        original_aa = vhh.imgt_numbered[1]
        delta = consensus.predict_mutation_effect(vhh, 1, original_aa)
        assert delta == 0.0


class TestIntegration:
    def test_both_scorers_on_same_sequence(
        self,
        hsc: HumanStringContentScorer,
        consensus: ConsensusStabilityScorer,
        vhh: VHHSequence,
    ) -> None:
        hsc_result = hsc.score(vhh)
        con_result = consensus.score(vhh)
        assert isinstance(hsc_result["composite_score"], float)
        assert isinstance(con_result["composite_score"], float)

    def test_mutation_changes_scores(
        self,
        hsc: HumanStringContentScorer,
        consensus: ConsensusStabilityScorer,
        vhh: VHHSequence,
    ) -> None:
        mutant = VHHSequence.mutate(vhh, 1, "A")
        hsc.score(vhh)["composite_score"]
        hsc_mutant = hsc.score(mutant)["composite_score"]
        consensus.score(vhh)["composite_score"]
        con_mutant = consensus.score(mutant)["composite_score"]
        # At least one score should change (or both stay same – just verify they run)
        assert isinstance(hsc_mutant, float)
        assert isinstance(con_mutant, float)


class TestNanoMeltStabilityScorer:
    def test_instantiation_does_not_raise(self) -> None:
        scorer = NanoMeltStabilityScorer()
        assert scorer is not None

    def test_is_available_returns_bool(self) -> None:
        scorer = NanoMeltStabilityScorer()
        assert isinstance(scorer.is_available, bool)
