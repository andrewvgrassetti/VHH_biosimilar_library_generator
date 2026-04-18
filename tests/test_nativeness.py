"""Tests for vhh_library.nativeness – NativenessScorer class."""

from __future__ import annotations

from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if abnativ is not installed.
# ---------------------------------------------------------------------------

abnativ = pytest.importorskip("abnativ", reason="abnativ not installed")

from vhh_library.nativeness import NativenessScorer  # noqa: E402
from vhh_library.sequence import VHHSequence  # noqa: E402

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture(scope="module")
def scorer() -> NativenessScorer:
    return NativenessScorer(model_type="VHH")


@pytest.fixture(scope="module")
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


class TestNativenessScorer:
    def test_instantiation(self, scorer: NativenessScorer) -> None:
        assert scorer is not None

    def test_score_returns_dict_with_composite(self, scorer: NativenessScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_predict_mutation_effect_returns_float(self, scorer: NativenessScorer, vhh: VHHSequence) -> None:
        # Pick a framework position to mutate
        pos = list(vhh.imgt_numbered.keys())[0]
        original = vhh.imgt_numbered[pos]
        new_aa = "A" if original != "A" else "G"
        delta = scorer.predict_mutation_effect(vhh, pos, new_aa)
        assert isinstance(delta, float)

    def test_score_batch(self, scorer: NativenessScorer) -> None:
        sequences = [SAMPLE_VHH, SAMPLE_VHH[:60] + "A" + SAMPLE_VHH[61:]]
        scores = scorer.score_batch(sequences)
        assert len(scores) == 2
        for s in scores:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_score_batch_empty(self, scorer: NativenessScorer) -> None:
        assert scorer.score_batch([]) == []

    def test_missing_model_weights_raises_helpful_error(self) -> None:
        """When model weights are not downloaded, a clear error is raised."""
        scorer = NativenessScorer(model_type="VHH")
        scorer._scoring_fn = None  # ensure not cached

        with mock.patch("abnativ.init.PRETRAINED_MODELS_DIR", "/nonexistent/path"):
            with pytest.raises(FileNotFoundError, match="vhh-init"):
                scorer._load_scoring_fn()
