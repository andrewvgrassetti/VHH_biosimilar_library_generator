"""Edge-case tests for NativenessScorer – empty / short AbNatiV results."""

from __future__ import annotations

from unittest import mock

import pandas as pd
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


def _make_scorer_with_mock_fn(return_df: pd.DataFrame) -> NativenessScorer:
    """Create a NativenessScorer whose scoring function returns *return_df*."""
    scorer = NativenessScorer(model_type="VHH")
    scorer._scoring_fn = mock.Mock(return_value=(return_df, None))
    return scorer


class TestScoreSequencesEmptyDataFrame:
    """_score_sequences must handle AbNatiV returning zero rows."""

    def test_returns_correct_count_for_empty_df(self, caplog) -> None:
        empty_df = pd.DataFrame({"score": pd.Series([], dtype=float)})
        scorer = _make_scorer_with_mock_fn(empty_df)

        with caplog.at_level("WARNING", logger="vhh_library.nativeness"):
            result = scorer._score_sequences(["AAAA", "BBBB"])
        assert len(result) == 2
        assert all(s == 0.5 for s in result)
        assert "returned 0 scores for 2 input sequences" in caplog.text

    def test_returns_correct_count_for_single_input_empty_df(self) -> None:
        empty_df = pd.DataFrame({"score": pd.Series([], dtype=float)})
        scorer = _make_scorer_with_mock_fn(empty_df)

        result = scorer._score_sequences(["AAAA"])
        assert len(result) == 1
        assert result[0] == 0.5


class TestScoreSequencesFewerRows:
    """_score_sequences must pad when AbNatiV returns fewer rows than inputs."""

    def test_pads_missing_scores(self, caplog) -> None:
        partial_df = pd.DataFrame({"score": [0.9]})
        scorer = _make_scorer_with_mock_fn(partial_df)

        with caplog.at_level("WARNING", logger="vhh_library.nativeness"):
            result = scorer._score_sequences(["SEQ1", "SEQ2", "SEQ3"])
        assert len(result) == 3
        assert result[0] == pytest.approx(0.9)
        assert result[1] == 0.5
        assert result[2] == 0.5
        assert "returned 1 scores for 3 input sequences" in caplog.text


class TestScoreWithEmptyAbNatiVResult:
    """score() must return a valid dict even when AbNatiV returns empty."""

    def test_score_returns_dict_on_empty_result(self) -> None:
        empty_df = pd.DataFrame({"score": pd.Series([], dtype=float)})
        scorer = _make_scorer_with_mock_fn(empty_df)

        vhh = VHHSequence(SAMPLE_VHH)
        result = scorer.score(vhh)

        assert isinstance(result, dict)
        assert "composite_score" in result
        assert result["composite_score"] == 0.5


class TestScoreBatchWithEmptyAbNatiVResult:
    """score_batch() must return correct-length list when AbNatiV returns fewer rows."""

    def test_score_batch_pads_on_empty(self) -> None:
        empty_df = pd.DataFrame({"score": pd.Series([], dtype=float)})
        scorer = _make_scorer_with_mock_fn(empty_df)

        result = scorer.score_batch(["SEQ1", "SEQ2"])
        assert len(result) == 2
        assert all(s == 0.5 for s in result)

    def test_score_batch_pads_on_partial(self) -> None:
        partial_df = pd.DataFrame({"score": [0.8]})
        scorer = _make_scorer_with_mock_fn(partial_df)

        result = scorer.score_batch(["SEQ1", "SEQ2", "SEQ3"])
        assert len(result) == 3
        assert result[0] == pytest.approx(0.8)
