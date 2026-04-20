"""Tests for the original-sequence score overlay on library distribution plots.

Validates that ``_build_original_scores`` correctly maps session-state scorer
outputs to the library DataFrame column names and derives ``combined_score``.
"""

from __future__ import annotations

import pytest

from app import _build_original_scores

# ---------------------------------------------------------------------------
# _build_original_scores — mapping logic
# ---------------------------------------------------------------------------


class TestBuildOriginalScores:
    """Unit tests for the pure helper that maps scorer outputs to columns."""

    def test_all_scores_present(self) -> None:
        stab = {"composite_score": 0.8, "predicted_tm": 65.0, "nanomelt_tm": 70.0}
        nat = {"composite_score": 0.7}
        hydro = {"composite_score": 0.6}

        result = _build_original_scores(stab, nat, hydro)

        assert result["stability_score"] == 0.8
        assert result["nativeness_score"] == 0.7
        assert result["surface_hydrophobicity_score"] == 0.6
        assert result["predicted_tm"] == 65.0
        assert result["nanomelt_tm"] == 70.0
        # Default combined = (stab + nat) / 2
        assert result["combined_score"] == pytest.approx((0.8 + 0.7) / 2.0)

    def test_nativeness_missing(self) -> None:
        stab = {"composite_score": 0.8}
        result = _build_original_scores(stab, None, None)

        assert result["stability_score"] == 0.8
        assert result.get("nativeness_score") is None
        # Falls back to stability-only
        assert result["combined_score"] == pytest.approx(0.8)

    def test_stability_missing(self) -> None:
        result = _build_original_scores(None, {"composite_score": 0.7}, None)
        assert result.get("stability_score") is None
        assert result["nativeness_score"] == 0.7
        assert "combined_score" not in result

    def test_all_none(self) -> None:
        result = _build_original_scores(None, None, None)
        assert result == {}

    def test_optional_tm_keys_absent(self) -> None:
        stab = {"composite_score": 0.5}
        result = _build_original_scores(stab, None, None)
        assert result.get("predicted_tm") is None
        assert result.get("nanomelt_tm") is None

    def test_engine_combined_score_used_when_available(self) -> None:
        """When a mock engine is passed, its _combined_score should be used."""

        class _FakeEngine:
            def _combined_score(self, raw: dict[str, float]) -> float:
                return raw["stability"] * 0.6 + raw["nativeness"] * 0.4

        stab = {"composite_score": 0.8}
        nat = {"composite_score": 0.7}
        engine = _FakeEngine()

        result = _build_original_scores(stab, nat, None, engine=engine)
        expected = 0.8 * 0.6 + 0.7 * 0.4
        assert result["combined_score"] == pytest.approx(expected)

    def test_engine_none_falls_back_to_average(self) -> None:
        stab = {"composite_score": 0.8}
        nat = {"composite_score": 0.6}

        result = _build_original_scores(stab, nat, None, engine=None)
        assert result["combined_score"] == pytest.approx(0.7)
