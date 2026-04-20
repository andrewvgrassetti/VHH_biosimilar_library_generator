"""Tests for vhh_library.stability – StabilityScorer class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vhh_library.sequence import VHHSequence
from vhh_library.stability import (
    StabilityScorer,
    _esm2_pll_available,
    _pll_to_predicted_tm,
    _sigmoid_normalize,
)

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


class TestCalibrationHelpers:
    def test_pll_to_predicted_tm(self) -> None:
        # PLL = -120.0, seq_len = 120 → per_residue = -1.0
        # Tm = 12.5 * (-1.0) + 95.0 = 82.5
        tm = _pll_to_predicted_tm(-120.0, 120)
        assert tm == pytest.approx(82.5)

    def test_pll_to_predicted_tm_zero_length(self) -> None:
        # seq_len=0 should not crash (uses max(seq_len, 1))
        tm = _pll_to_predicted_tm(-10.0, 0)
        assert isinstance(tm, float)

    def test_sigmoid_normalize_midpoint(self) -> None:
        # Midpoint of 55..80 is 67.5; sigmoid at midpoint → ≈0.5
        val = _sigmoid_normalize(67.5, 55.0, 80.0)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_sigmoid_normalize_high(self) -> None:
        # Well above tm_max → ≈1.0
        val = _sigmoid_normalize(120.0, 55.0, 80.0)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_sigmoid_normalize_low(self) -> None:
        # Well below tm_min → ≈0.0
        val = _sigmoid_normalize(0.0, 55.0, 80.0)
        assert val == pytest.approx(0.0, abs=0.01)


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


class TestESM2Scoring:
    def test_predicted_tm_present_with_esm2(self, vhh: VHHSequence) -> None:
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        scorer = StabilityScorer(esm_scorer=mock_esm)
        result = scorer.score(vhh)
        assert isinstance(result["predicted_tm"], float)
        assert "tm_score" in result
        assert result["scoring_method"] == "esm2"

    def test_composite_score_range_with_esm2(self, vhh: VHHSequence) -> None:
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        scorer = StabilityScorer(esm_scorer=mock_esm)
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_disulfide_penalty_lowers_score(self) -> None:
        """A construct missing the canonical disulfide should score at least 0.15 lower."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0

        # Normal VHH with both Cys at positions 23 and 104
        vhh_with_cys = VHHSequence(SAMPLE_VHH)
        scorer = StabilityScorer(esm_scorer=mock_esm)
        score_with = scorer.score(vhh_with_cys)

        # Create a mutant that removes both Cys at canonical positions
        # Replace C→A at IMGT position 23 and 104
        mutant = VHHSequence.mutate(vhh_with_cys, 23, "A")
        mutant = VHHSequence.mutate(mutant, 104, "A")
        score_without = scorer.score(mutant)

        assert score_with["composite_score"] - score_without["composite_score"] >= 0.15


class TestMutationEffect:
    def test_predict_mutation_effect(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        delta = scorer.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)


class TestScoringMethod:
    def test_scoring_method_present(self, scorer: StabilityScorer, vhh: VHHSequence) -> None:
        result = scorer.score(vhh)
        assert result["scoring_method"] in ("legacy", "esm2", "nanomelt")

    def test_legacy_fallback(self, vhh: VHHSequence) -> None:
        scorer = StabilityScorer()
        result = scorer.score(vhh)
        assert result["scoring_method"] == "legacy"


class TestAvailability:
    def test_esm2_pll_available_returns_bool(self) -> None:
        assert isinstance(_esm2_pll_available(), bool)


class TestCalibrationIntegration:
    """Test StabilityScorer loading calibration from file."""

    def test_scorer_uses_calibration_params(self, tmp_path: Path) -> None:
        cal_path = tmp_path / "stability_calibration.json"
        cal_data = {
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "calibration_vhhs": [{"name": "VHH_1", "sequence": "ACDE", "experimental_tm": 65.0}],
            "parameters": {
                "pll_to_tm_slope": 20.0,
                "pll_to_tm_intercept": 80.0,
                "tm_ideal_min": 50.0,
                "tm_ideal_max": 90.0,
                "penalty_disulfide": 0.30,
                "penalty_aggregation": 0.15,
                "penalty_charge": 0.08,
                "hallmark_bonus_weight": 0.12,
                "legacy_weights": {
                    "disulfide": 0.30,
                    "hallmark": 0.25,
                    "aggregation": 0.20,
                    "charge": 0.10,
                    "hydrophobic": 0.15,
                },
            },
        }
        cal_path.write_text(json.dumps(cal_data))
        scorer = StabilityScorer(calibration_path=str(cal_path))
        assert scorer._calibrated is True
        assert scorer._pll_slope == 20.0
        assert scorer._pll_intercept == 80.0
        assert scorer._tm_min == 50.0
        assert scorer._tm_max == 90.0
        assert scorer._penalty_disulfide == 0.30
        assert scorer._w_disulfide == 0.30

    def test_scorer_falls_back_to_defaults(self, tmp_path: Path) -> None:
        cal_path = tmp_path / "nonexistent_calibration.json"
        scorer = StabilityScorer(calibration_path=str(cal_path))
        assert scorer._calibrated is False
        assert scorer._pll_slope == 12.5
        assert scorer._pll_intercept == 95.0
        assert scorer._tm_min == 55.0
        assert scorer._tm_max == 80.0

    def test_scorer_falls_back_on_empty_calibration(self, tmp_path: Path) -> None:
        cal_path = tmp_path / "stability_calibration.json"
        cal_data = {
            "version": 1,
            "created_at": None,
            "calibration_vhhs": [],
            "parameters": {"pll_to_tm_slope": 12.5},
        }
        cal_path.write_text(json.dumps(cal_data))
        scorer = StabilityScorer(calibration_path=str(cal_path))
        assert scorer._calibrated is False

    def test_calibrated_scorer_uses_instance_tm_method(self, tmp_path: Path) -> None:
        """Verify the instance _pll_to_predicted_tm uses calibrated slope/intercept."""
        cal_path = tmp_path / "stability_calibration.json"
        cal_data = {
            "version": 1,
            "created_at": "2025-01-01",
            "calibration_vhhs": [{"name": "x", "sequence": "A", "experimental_tm": 60.0}],
            "parameters": {
                "pll_to_tm_slope": 10.0,
                "pll_to_tm_intercept": 100.0,
                "tm_ideal_min": 55.0,
                "tm_ideal_max": 80.0,
            },
        }
        cal_path.write_text(json.dumps(cal_data))
        scorer = StabilityScorer(calibration_path=str(cal_path))
        # PLL=-100, seq_len=100 → per_residue=-1.0 → 10.0*(-1.0)+100.0=90.0
        tm = scorer._pll_to_predicted_tm(-100.0, 100)
        assert tm == pytest.approx(90.0)

    def test_calibrated_scorer_scoring_still_works(self) -> None:
        """Default scorer (no calibration file populated) should produce valid scores."""
        scorer = StabilityScorer()
        vhh = VHHSequence(SAMPLE_VHH)
        result = scorer.score(vhh)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0


# ---------------------------------------------------------------------------
# NanoMelt integration in StabilityScorer
# ---------------------------------------------------------------------------


class TestNanoMeltScoring:
    """Tests for the NanoMelt branch in StabilityScorer.score()."""

    @staticmethod
    def _make_mock_nanomelt(tm: float = 70.0) -> MagicMock:
        """Return a mock NanoMeltPredictor that returns a fixed Tm."""
        mock = MagicMock()
        mock.score_sequence.return_value = {
            "composite_score": 0.6,
            "nanomelt_tm": tm,
        }
        return mock

    def test_nanomelt_only(self, vhh: VHHSequence) -> None:
        """StabilityScorer with only NanoMelt should use 'nanomelt' method."""
        mock_nm = self._make_mock_nanomelt(70.0)
        scorer = StabilityScorer(nanomelt_predictor=mock_nm)
        result = scorer.score(vhh)
        assert result["scoring_method"] == "nanomelt"
        assert "nanomelt_tm" in result
        assert result["nanomelt_tm"] == 70.0
        assert 0.0 <= result["composite_score"] <= 1.0
        mock_nm.score_sequence.assert_called_once()

    def test_nanomelt_composite_in_range(self, vhh: VHHSequence) -> None:
        """Composite score from NanoMelt must be in [0, 1]."""
        for tm in [20.0, 55.0, 67.5, 80.0, 95.0]:
            mock_nm = self._make_mock_nanomelt(tm)
            scorer = StabilityScorer(nanomelt_predictor=mock_nm)
            result = scorer.score(vhh)
            assert 0.0 <= result["composite_score"] <= 1.0, f"Failed for tm={tm}"

    def test_nanomelt_tm_score_present(self, vhh: VHHSequence) -> None:
        mock_nm = self._make_mock_nanomelt(72.0)
        scorer = StabilityScorer(nanomelt_predictor=mock_nm)
        result = scorer.score(vhh)
        assert "nanomelt_tm_score" in result
        assert isinstance(result["nanomelt_tm_score"], float)

    def test_both_backends(self, vhh: VHHSequence) -> None:
        """StabilityScorer with ESM-2 + NanoMelt should use 'nanomelt' (NanoMelt is primary)."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        mock_nm = self._make_mock_nanomelt(70.0)
        scorer = StabilityScorer(esm_scorer=mock_esm, nanomelt_predictor=mock_nm)
        result = scorer.score(vhh)
        assert result["scoring_method"] == "nanomelt"
        assert "predicted_tm" in result
        assert "nanomelt_tm" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_both_backends_composite_equals_nanomelt_only(self, vhh: VHHSequence) -> None:
        """When both backends are present, composite_score equals the NanoMelt-only composite."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        mock_nm = self._make_mock_nanomelt(70.0)

        # Get NanoMelt-only score
        nm_only = StabilityScorer(nanomelt_predictor=self._make_mock_nanomelt(70.0))
        nm_result = nm_only.score(vhh)
        nm_composite = nm_result["composite_score"]

        # Get both score — should match NanoMelt-only
        both = StabilityScorer(esm_scorer=mock_esm, nanomelt_predictor=mock_nm)
        both_result = both.score(vhh)

        assert both_result["composite_score"] == pytest.approx(nm_composite, abs=1e-9)

    def test_esm2_diagnostic_keys_present_with_both(self, vhh: VHHSequence) -> None:
        """When both backends are present, ESM-2 diagnostic keys should still be populated."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        mock_nm = self._make_mock_nanomelt(70.0)
        scorer = StabilityScorer(esm_scorer=mock_esm, nanomelt_predictor=mock_nm)
        result = scorer.score(vhh)
        # ESM-2 diagnostic keys must still be present
        assert "esm2_pll" in result
        assert "predicted_tm" in result
        assert "tm_score" in result
        # But the scoring method is nanomelt
        assert result["scoring_method"] == "nanomelt"

    def test_nanomelt_failure_falls_back(self, vhh: VHHSequence) -> None:
        """If NanoMelt raises, scoring should fall back to legacy."""
        mock_nm = MagicMock()
        mock_nm.score_sequence.side_effect = RuntimeError("backend failed")
        scorer = StabilityScorer(nanomelt_predictor=mock_nm)
        result = scorer.score(vhh)
        assert result["scoring_method"] == "legacy"
        assert "nanomelt_tm" not in result

    def test_both_nanomelt_failure_falls_back_to_esm2(self, vhh: VHHSequence) -> None:
        """If NanoMelt fails in 'both' mode, score should use ESM-2 only."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        mock_nm = MagicMock()
        mock_nm.score_sequence.side_effect = RuntimeError("backend failed")
        scorer = StabilityScorer(esm_scorer=mock_esm, nanomelt_predictor=mock_nm)
        result = scorer.score(vhh)
        assert result["scoring_method"] == "esm2"

    def test_predict_mutation_effect_with_nanomelt(self, vhh: VHHSequence) -> None:
        """predict_mutation_effect should work when NanoMelt is configured."""
        mock_nm = self._make_mock_nanomelt(70.0)
        scorer = StabilityScorer(nanomelt_predictor=mock_nm)
        delta = scorer.predict_mutation_effect(vhh, 1, "A")
        assert isinstance(delta, float)

    def test_esm2_path_unchanged_without_nanomelt(self, vhh: VHHSequence) -> None:
        """ESM-2-only path should remain identical when no nanomelt_predictor is given."""
        mock_esm = MagicMock()
        mock_esm.score_single.return_value = -100.0
        scorer = StabilityScorer(esm_scorer=mock_esm)
        result = scorer.score(vhh)
        assert result["scoring_method"] == "esm2"
        assert "predicted_tm" in result
        assert "nanomelt_tm" not in result

    def test_use_nanomelt_deprecation_warning(self) -> None:
        """use_nanomelt=True should emit a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="use_nanomelt is deprecated"):
            StabilityScorer(use_nanomelt=True)
