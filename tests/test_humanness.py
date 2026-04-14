"""Tests for vhh_library.humanness – HumAnnotator class."""

from __future__ import annotations

import pytest

from vhh_library.humanness import HumAnnotator
from vhh_library.sequence import VHHSequence

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def annotator() -> HumAnnotator:
    return HumAnnotator()


@pytest.fixture
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


class TestGermlineData:
    def test_loads_germlines(self, annotator: HumAnnotator) -> None:
        assert len(annotator.germlines) > 0


class TestScoring:
    def test_score_returns_dict(self, annotator: HumAnnotator, vhh: VHHSequence) -> None:
        result = annotator.score(vhh)
        assert "composite_score" in result
        assert "germline_identity" in result

    def test_composite_score_range(self, annotator: HumAnnotator, vhh: VHHSequence) -> None:
        result = annotator.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_position_scores_exist(self, annotator: HumAnnotator, vhh: VHHSequence) -> None:
        result = annotator.score(vhh)
        assert "position_scores" in result
        assert len(result["position_scores"]) > 0


class TestMutationSuggestions:
    def test_mutation_suggestions(self, annotator: HumAnnotator, vhh: VHHSequence) -> None:
        suggestions = annotator.get_mutation_suggestions(vhh, off_limits=set())
        assert isinstance(suggestions, list)

    def test_excluded_target_aas_filters_cysteine(
        self, annotator: HumAnnotator, vhh: VHHSequence
    ) -> None:
        suggestions = annotator.get_mutation_suggestions(
            vhh, off_limits=set(), excluded_target_aas={"C"}
        )
        for s in suggestions:
            assert s["suggested_aa"] != "C"

    def test_excluded_target_aas_multiple(
        self, annotator: HumAnnotator, vhh: VHHSequence
    ) -> None:
        excluded = {"C", "M", "W"}
        suggestions = annotator.get_mutation_suggestions(
            vhh, off_limits=set(), excluded_target_aas=excluded
        )
        for s in suggestions:
            assert s["suggested_aa"] not in excluded

    def test_excluded_target_aas_empty_set_no_effect(
        self, annotator: HumAnnotator, vhh: VHHSequence
    ) -> None:
        baseline = annotator.get_mutation_suggestions(vhh, off_limits=set())
        filtered = annotator.get_mutation_suggestions(
            vhh, off_limits=set(), excluded_target_aas=set()
        )
        assert len(filtered) == len(baseline)
