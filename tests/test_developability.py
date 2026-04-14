"""Tests for vhh_library.developability – PTM, clearance, and surface scorers."""

from __future__ import annotations

import pytest

from vhh_library.developability import (
    ClearanceRiskScorer,
    PTMLiabilityScorer,
    SurfaceHydrophobicityScorer,
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
def ptm() -> PTMLiabilityScorer:
    return PTMLiabilityScorer()


@pytest.fixture
def clearance() -> ClearanceRiskScorer:
    return ClearanceRiskScorer()


@pytest.fixture
def surface() -> SurfaceHydrophobicityScorer:
    return SurfaceHydrophobicityScorer()


class TestPTMLiabilityScorer:
    def test_score_returns_dict(self, ptm: PTMLiabilityScorer, vhh: VHHSequence) -> None:
        result = ptm.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_hits_is_list(self, ptm: PTMLiabilityScorer, vhh: VHHSequence) -> None:
        result = ptm.score(vhh)
        assert isinstance(result["hits"], list)

    def test_predict_mutation_effect(self, ptm: PTMLiabilityScorer, vhh: VHHSequence) -> None:
        delta = ptm.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)


class TestClearanceRiskScorer:
    def test_score_returns_dict(
        self, clearance: ClearanceRiskScorer, vhh: VHHSequence
    ) -> None:
        result = clearance.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_pI_present(self, clearance: ClearanceRiskScorer, vhh: VHHSequence) -> None:
        result = clearance.score(vhh)
        assert "pI" in result

    def test_pI_deviation_non_negative(
        self, clearance: ClearanceRiskScorer, vhh: VHHSequence
    ) -> None:
        result = clearance.score(vhh)
        assert result["pI_deviation"] >= 0


class TestSurfaceHydrophobicityScorer:
    def test_score_returns_dict(
        self, surface: SurfaceHydrophobicityScorer, vhh: VHHSequence
    ) -> None:
        result = surface.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_n_patches_non_negative(
        self, surface: SurfaceHydrophobicityScorer, vhh: VHHSequence
    ) -> None:
        result = surface.score(vhh)
        assert result["n_patches"] >= 0

    def test_predict_mutation_effect(
        self, surface: SurfaceHydrophobicityScorer, vhh: VHHSequence
    ) -> None:
        delta = surface.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)
