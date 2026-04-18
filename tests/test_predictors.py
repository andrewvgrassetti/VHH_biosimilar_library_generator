"""Tests for vhh_library.predictors — Predictor protocol and adapter wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vhh_library.predictors.abnativ import AbNatiVPredictor
from vhh_library.predictors.base import Predictor
from vhh_library.predictors.esm2_prior import ESM2PriorPredictor, _sigmoid_normalize
from vhh_library.sequence import VHHSequence

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture(scope="module")
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


# ---------------------------------------------------------------------------
# Predictor ABC
# ---------------------------------------------------------------------------


class TestPredictorProtocol:
    """Verify the abstract base class enforces the contract."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Predictor()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_name_and_score(self) -> None:
        class Incomplete(Predictor):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_minimal_subclass(self, vhh: VHHSequence) -> None:
        class Dummy(Predictor):
            @property
            def name(self) -> str:
                return "dummy"

            def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
                return {"composite_score": 0.42}

        d = Dummy()
        assert d.name == "dummy"
        result = d.score_sequence(vhh)
        assert result["composite_score"] == pytest.approx(0.42)

    def test_default_score_batch(self, vhh: VHHSequence) -> None:
        """Default score_batch should loop over score_sequence."""

        class Counter(Predictor):
            def __init__(self) -> None:
                self.call_count = 0

            @property
            def name(self) -> str:
                return "counter"

            def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
                self.call_count += 1
                return {"composite_score": 0.5}

        c = Counter()
        results = c.score_batch([vhh, vhh, vhh])
        assert len(results) == 3
        assert c.call_count == 3
        for r in results:
            assert "composite_score" in r


# ---------------------------------------------------------------------------
# AbNatiVPredictor adapter
# ---------------------------------------------------------------------------


class TestAbNatiVPredictor:
    """Tests for the AbNatiV adapter using a mocked NativenessScorer."""

    def test_name(self) -> None:
        mock_scorer = MagicMock()
        pred = AbNatiVPredictor(scorer=mock_scorer)
        assert pred.name == "abnativ"

    def test_is_predictor_subclass(self) -> None:
        mock_scorer = MagicMock()
        pred = AbNatiVPredictor(scorer=mock_scorer)
        assert isinstance(pred, Predictor)

    def test_score_sequence_delegates(self, vhh: VHHSequence) -> None:
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {"composite_score": 0.85}
        pred = AbNatiVPredictor(scorer=mock_scorer)

        result = pred.score_sequence(vhh)
        assert result["composite_score"] == pytest.approx(0.85)
        mock_scorer.score.assert_called_once_with(vhh)

    def test_score_batch_delegates(self, vhh: VHHSequence) -> None:
        mock_scorer = MagicMock()
        mock_scorer.score_batch.return_value = [0.85, 0.90]
        pred = AbNatiVPredictor(scorer=mock_scorer)

        results = pred.score_batch([vhh, vhh])
        assert len(results) == 2
        assert results[0]["composite_score"] == pytest.approx(0.85)
        assert results[1]["composite_score"] == pytest.approx(0.90)

    def test_lazy_loading_not_called_on_init(self) -> None:
        """Scorer should NOT be created during __init__ when no scorer is injected."""
        with patch("vhh_library.nativeness.NativenessScorer") as mock_cls:
            pred = AbNatiVPredictor.__new__(AbNatiVPredictor)
            pred._model_type = "VHH"
            pred._batch_size = 128
            pred._scorer = None
            mock_cls.assert_not_called()

    def test_lazy_loading_triggered_on_score(self, vhh: VHHSequence) -> None:
        """Scorer should be created on first score_sequence call."""
        mock_inner = MagicMock()
        mock_inner.score.return_value = {"composite_score": 0.75}

        with patch(
            "vhh_library.nativeness.NativenessScorer",
            return_value=mock_inner,
        ) as mock_cls:
            pred = AbNatiVPredictor.__new__(AbNatiVPredictor)
            pred._model_type = "VHH"
            pred._batch_size = 128
            pred._scorer = None
            pred.score_sequence(vhh)
            mock_cls.assert_called_once_with(model_type="VHH", batch_size=128)

    def test_composite_score_in_range(self, vhh: VHHSequence) -> None:
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {"composite_score": 0.5}
        pred = AbNatiVPredictor(scorer=mock_scorer)
        result = pred.score_sequence(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0


# ---------------------------------------------------------------------------
# ESM2PriorPredictor adapter
# ---------------------------------------------------------------------------


class TestESM2PriorPredictor:
    """Tests for the ESM-2 prior adapter using a mocked ESMStabilityScorer."""

    def test_name(self) -> None:
        mock_scorer = MagicMock()
        pred = ESM2PriorPredictor(scorer=mock_scorer)
        assert pred.name == "esm2_prior"

    def test_is_predictor_subclass(self) -> None:
        mock_scorer = MagicMock()
        pred = ESM2PriorPredictor(scorer=mock_scorer)
        assert isinstance(pred, Predictor)

    def test_score_sequence_delegates(self, vhh: VHHSequence) -> None:
        mock_scorer = MagicMock()
        mock_scorer.score_single.return_value = -100.0
        pred = ESM2PriorPredictor(scorer=mock_scorer)

        result = pred.score_sequence(vhh)
        assert "composite_score" in result
        assert "esm2_pll" in result
        assert result["esm2_pll"] == pytest.approx(-100.0)
        assert 0.0 <= result["composite_score"] <= 1.0
        mock_scorer.score_single.assert_called_once_with(vhh.sequence)

    def test_score_batch_delegates(self, vhh: VHHSequence) -> None:
        mock_scorer = MagicMock()
        mock_scorer.score_batch.return_value = [-100.0, -120.0]
        pred = ESM2PriorPredictor(scorer=mock_scorer)

        results = pred.score_batch([vhh, vhh])
        assert len(results) == 2
        for r in results:
            assert "composite_score" in r
            assert "esm2_pll" in r
            assert 0.0 <= r["composite_score"] <= 1.0

    def test_pll_to_score_high_pll(self) -> None:
        """Very high (near 0) PLL should map to near 1.0."""
        score = ESM2PriorPredictor._pll_to_score(-10.0, 120)
        assert score > 0.9

    def test_pll_to_score_low_pll(self) -> None:
        """Very low (very negative) PLL should map to near 0.0."""
        score = ESM2PriorPredictor._pll_to_score(-10000.0, 120)
        assert score < 0.1

    def test_pll_to_score_moderate(self) -> None:
        """Moderate PLL should be somewhere in (0, 1)."""
        score = ESM2PriorPredictor._pll_to_score(-100.0, 120)
        assert 0.0 < score < 1.0

    def test_lazy_loading_not_called_on_init(self) -> None:
        """ESMStabilityScorer should NOT be created during __init__."""
        pred = ESM2PriorPredictor()
        assert pred._scorer is None

    def test_lazy_loading_triggered_on_score(self, vhh: VHHSequence) -> None:
        """Scorer should be created on first score_sequence call."""
        mock_inner = MagicMock()
        mock_inner.score_single.return_value = -100.0

        with patch(
            "vhh_library.esm_scorer.ESMStabilityScorer",
            return_value=mock_inner,
        ) as mock_cls:
            pred = ESM2PriorPredictor(device="cpu")
            pred.score_sequence(vhh)
            mock_cls.assert_called_once()
            # Verify device was resolved
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("device") == "cpu" or call_kwargs[1].get("device") == "cpu"

    def test_composite_score_in_range(self, vhh: VHHSequence) -> None:
        mock_scorer = MagicMock()
        mock_scorer.score_single.return_value = -100.0
        pred = ESM2PriorPredictor(scorer=mock_scorer)
        result = pred.score_sequence(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0


# ---------------------------------------------------------------------------
# Sigmoid normalise helper
# ---------------------------------------------------------------------------


class TestSigmoidNormalize:
    def test_midpoint(self) -> None:
        val = _sigmoid_normalize(67.5, 55.0, 80.0)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_high(self) -> None:
        val = _sigmoid_normalize(120.0, 55.0, 80.0)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_low(self) -> None:
        val = _sigmoid_normalize(0.0, 55.0, 80.0)
        assert val == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Wrapper compatibility — existing scorer interfaces still work
# ---------------------------------------------------------------------------


class TestWrapperCompatibility:
    """Verify that wrapping existing scorers doesn't change their behavior."""

    def test_abnativ_wrapper_preserves_scorer_output(self, vhh: VHHSequence) -> None:
        """AbNatiVPredictor.score_sequence returns same dict as NativenessScorer.score."""
        mock_scorer = MagicMock()
        expected = {"composite_score": 0.78}
        mock_scorer.score.return_value = expected

        pred = AbNatiVPredictor(scorer=mock_scorer)
        result = pred.score_sequence(vhh)
        assert result == expected

    def test_esm2_wrapper_adds_pll_key(self, vhh: VHHSequence) -> None:
        """ESM2PriorPredictor.score_sequence adds esm2_pll alongside composite_score."""
        mock_scorer = MagicMock()
        mock_scorer.score_single.return_value = -150.0

        pred = ESM2PriorPredictor(scorer=mock_scorer)
        result = pred.score_sequence(vhh)
        assert "composite_score" in result
        assert "esm2_pll" in result
        assert result["esm2_pll"] == -150.0

    def test_batch_empty_lists(self, vhh: VHHSequence) -> None:
        """Batch scoring with empty list should return empty list."""
        mock_scorer = MagicMock()
        mock_scorer.score_batch.return_value = []

        pred = AbNatiVPredictor(scorer=mock_scorer)
        assert pred.score_batch([]) == []

    def test_esm2_batch_empty(self) -> None:
        """ESM-2 batch with empty list."""
        mock_scorer = MagicMock()
        mock_scorer.score_batch.return_value = []

        pred = ESM2PriorPredictor(scorer=mock_scorer)
        assert pred.score_batch([]) == []
