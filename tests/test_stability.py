"""Tests for vhh_library.stability – StabilityScorer class."""

from __future__ import annotations

import pytest

from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer, _esm2_pll_available

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def scorer() -> StabilityScorer:
    return StabilityScorer()


@pytest.fixture
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


class TestScoring:
    def test_score_returns_dict(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert "composite_score" in result

    def test_composite_score_range(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_pI_range(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert 3.0 <= result["pI"] <= 12.0

    def test_disulfide_score(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert 0.0 <= result["disulfide_score"] <= 1.0


class TestMutationEffect:
    def test_predict_mutation_effect(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        delta = scorer.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)


class TestScoringMethod:
    def test_scoring_method_present(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert result["scoring_method"] in ("legacy", "esm2")

    def test_legacy_fallback(self, vhh: VHHSequence) -> None:
        scorer = StabilityScorer()
        result = scorer.score(vhh)
        assert result["scoring_method"] == "legacy"


class TestAvailability:
    def test_esm2_pll_available_returns_bool(self) -> None:
        assert isinstance(_esm2_pll_available(), bool)
