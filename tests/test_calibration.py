"""Tests for vhh_library.calibration – persistent calibration system."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vhh_library.calibration import (
    CalibrationResult,
    _compute_r_squared,
    _least_squares_fit,
    load_calibration,
    reset_calibration,
    run_calibration,
)


@pytest.fixture
def tmp_cal_path(tmp_path: Path) -> Path:
    """Return a temporary calibration file path."""
    return tmp_path / "stability_calibration.json"


class TestLoadCalibration:
    def test_returns_none_when_file_missing(self, tmp_cal_path: Path) -> None:
        assert load_calibration(tmp_cal_path) is None

    def test_returns_none_when_calibration_vhhs_empty(self, tmp_cal_path: Path) -> None:
        data = {
            "version": 1,
            "created_at": None,
            "calibration_vhhs": [],
            "parameters": {"pll_to_tm_slope": 12.5},
        }
        tmp_cal_path.write_text(json.dumps(data))
        assert load_calibration(tmp_cal_path) is None

    def test_returns_data_when_calibrated(self, tmp_cal_path: Path) -> None:
        data = {
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "calibration_vhhs": [{"name": "VHH_1", "sequence": "ACDE", "experimental_tm": 65.0}],
            "parameters": {
                "pll_to_tm_slope": 15.0,
                "pll_to_tm_intercept": 90.0,
                "tm_ideal_min": 50.0,
                "tm_ideal_max": 75.0,
            },
        }
        tmp_cal_path.write_text(json.dumps(data))
        result = load_calibration(tmp_cal_path)
        assert result is not None
        assert result["parameters"]["pll_to_tm_slope"] == 15.0
        assert len(result["calibration_vhhs"]) == 1

    def test_returns_none_on_corrupt_json(self, tmp_cal_path: Path) -> None:
        tmp_cal_path.write_text("NOT VALID JSON {{{")
        assert load_calibration(tmp_cal_path) is None

    def test_returns_none_with_default_path_no_real_calibration(self) -> None:
        """The shipped default file has empty calibration_vhhs, so load should return None."""
        result = load_calibration()
        assert result is None


class TestResetCalibration:
    def test_reset_creates_default_file(self, tmp_cal_path: Path) -> None:
        reset_calibration(tmp_cal_path)
        assert tmp_cal_path.exists()
        data = json.loads(tmp_cal_path.read_text())
        assert data["calibration_vhhs"] == []
        assert data["parameters"]["pll_to_tm_slope"] == 12.5

    def test_reset_overwrites_existing(self, tmp_cal_path: Path) -> None:
        # Write something custom first
        tmp_cal_path.write_text(json.dumps({"custom": True}))
        reset_calibration(tmp_cal_path)
        data = json.loads(tmp_cal_path.read_text())
        assert "custom" not in data
        assert data["version"] == 1


class TestRunCalibration:
    def test_run_calibration_with_mocked_esm(self, tmp_cal_path: Path) -> None:
        """Test run_calibration with a mocked ESMStabilityScorer."""
        mock_scorer_instance = MagicMock()
        # Return PLLs proportional to sequence length (simulate realistic behavior)
        mock_scorer_instance.score_batch.return_value = [-120.0, -130.0, -110.0, -140.0, -125.0]

        sequences = ["A" * 120, "A" * 130, "A" * 110, "A" * 140, "A" * 125]
        known_tms = [65.0, 60.0, 70.0, 55.0, 67.0]
        names = ["VHH_A", "VHH_B", "VHH_C", "VHH_D", "VHH_E"]

        with patch("vhh_library.esm_scorer.ESMStabilityScorer", return_value=mock_scorer_instance):
            result = run_calibration(
                sequences, known_tms, names=names, calibration_path=tmp_cal_path
            )

        assert isinstance(result, CalibrationResult)
        assert result.n_samples == 5
        assert isinstance(result.pll_to_tm_slope, float)
        assert isinstance(result.pll_to_tm_intercept, float)
        assert isinstance(result.r_squared, float)
        assert 0.0 <= result.r_squared <= 1.0 or result.r_squared < 0  # Can be negative for bad fits
        assert result.tm_ideal_min <= result.tm_ideal_max

        # Verify file was saved
        assert tmp_cal_path.exists()
        data = json.loads(tmp_cal_path.read_text())
        assert len(data["calibration_vhhs"]) == 5
        assert data["calibration_vhhs"][0]["name"] == "VHH_A"
        assert data["created_at"] is not None

    def test_run_calibration_requires_min_2_sequences(self, tmp_cal_path: Path) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            run_calibration(["AAA"], [65.0], calibration_path=tmp_cal_path)

    def test_run_calibration_mismatched_lengths(self, tmp_cal_path: Path) -> None:
        with pytest.raises(ValueError, match="same length"):
            run_calibration(["AAA", "BBB"], [65.0], calibration_path=tmp_cal_path)

    def test_run_calibration_without_names(self, tmp_cal_path: Path) -> None:
        """Names should default to VHH_1, VHH_2, etc."""
        mock_scorer_instance = MagicMock()
        mock_scorer_instance.score_batch.return_value = [-100.0, -110.0]

        with patch("vhh_library.esm_scorer.ESMStabilityScorer", return_value=mock_scorer_instance):
            result = run_calibration(
                ["A" * 100, "A" * 110], [65.0, 70.0], calibration_path=tmp_cal_path
            )

        assert result.n_samples == 2
        data = json.loads(tmp_cal_path.read_text())
        assert data["calibration_vhhs"][0]["name"] == "VHH_1"
        assert data["calibration_vhhs"][1]["name"] == "VHH_2"


class TestLeastSquaresFit:
    def test_perfect_fit(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
        slope, intercept = _least_squares_fit(xs, ys)
        assert slope == pytest.approx(2.0, abs=1e-6)
        assert intercept == pytest.approx(1.0, abs=1e-6)

    def test_constant_x(self) -> None:
        xs = [1.0, 1.0, 1.0]
        ys = [2.0, 3.0, 4.0]
        slope, intercept = _least_squares_fit(xs, ys)
        # Degenerate case — slope should be 0
        assert slope == pytest.approx(0.0)


class TestComputeRSquared:
    def test_perfect_fit(self) -> None:
        xs = [1.0, 2.0, 3.0]
        ys = [3.0, 5.0, 7.0]
        r2 = _compute_r_squared(xs, ys, slope=2.0, intercept=1.0)
        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_poor_fit(self) -> None:
        xs = [1.0, 2.0, 3.0]
        ys = [10.0, 1.0, 10.0]
        r2 = _compute_r_squared(xs, ys, slope=0.0, intercept=7.0)
        assert r2 < 1.0
