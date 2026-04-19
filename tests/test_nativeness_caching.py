"""Tests for NativenessScorer caching optimisations.

These tests use a fully mocked scoring function and do NOT require
AbNatiV model weights to be present.
"""

from __future__ import annotations

from unittest import mock

import pytest

abnativ = pytest.importorskip("abnativ", reason="abnativ not installed")

from vhh_library.nativeness import NativenessScorer  # noqa: E402
from vhh_library.sequence import VHHSequence  # noqa: E402

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


def _make_scorer_with_mock_fn(return_score: float = 0.85) -> NativenessScorer:
    """Create a NativenessScorer whose _score_sequences is a mock."""
    scorer = NativenessScorer(model_type="VHH")
    # Replace _score_sequences with a mock that returns a deterministic score.
    scorer._score_sequences = mock.Mock(side_effect=lambda seqs: [return_score] * len(seqs))
    return scorer


class TestScoreCaching:
    """score() called twice with the same sequence should invoke _score_sequences only once."""

    def test_score_same_sequence_calls_score_sequences_once(self) -> None:
        scorer = _make_scorer_with_mock_fn(0.9)
        vhh = VHHSequence(SAMPLE_VHH)

        result1 = scorer.score(vhh)
        result2 = scorer.score(vhh)

        assert result1 == result2
        assert result1["composite_score"] == pytest.approx(0.9)
        # _score_sequences should have been called exactly once
        scorer._score_sequences.assert_called_once()

    def test_score_different_sequences_calls_score_sequences_twice(self) -> None:
        scorer = _make_scorer_with_mock_fn(0.8)
        vhh1 = VHHSequence(SAMPLE_VHH)
        # Create a mutant with a different sequence
        mutant_seq = SAMPLE_VHH[:10] + ("A" if SAMPLE_VHH[10] != "A" else "G") + SAMPLE_VHH[11:]
        vhh2 = VHHSequence(mutant_seq)

        scorer.score(vhh1)
        scorer.score(vhh2)

        assert scorer._score_sequences.call_count == 2


class TestPredictMutationEffectCaching:
    """predict_mutation_effect called multiple times should score the parent only once."""

    def test_multiple_mutations_same_parent_scores_parent_once(self) -> None:
        scorer = NativenessScorer(model_type="VHH")

        # Track calls via a mock on _score_sequences
        call_log: list[str] = []
        original_parent_seq = SAMPLE_VHH

        def fake_score_sequences(seqs: list[str]) -> list[float]:
            for s in seqs:
                call_log.append(s)
            # Parent gets 0.8, mutants get 0.85
            return [0.8 if s == original_parent_seq else 0.85 for s in seqs]

        scorer._score_sequences = mock.Mock(side_effect=fake_score_sequences)

        vhh = VHHSequence(SAMPLE_VHH)

        # Call predict_mutation_effect several times with different positions
        positions = list(vhh.imgt_numbered.keys())[:3]
        for pos in positions:
            original_aa = vhh.imgt_numbered[pos]
            new_aa = "A" if original_aa != "A" else "G"
            delta = scorer.predict_mutation_effect(vhh, pos, new_aa)
            assert isinstance(delta, float)

        # The parent sequence should appear in call_log exactly once
        parent_calls = [s for s in call_log if s == original_parent_seq]
        assert len(parent_calls) == 1, (
            f"Parent sequence was scored {len(parent_calls)} times, expected 1"
        )


class TestScoreBatchBypassesCache:
    """score_batch should call _score_sequences directly, not the cache."""

    def test_score_batch_calls_score_sequences_directly(self) -> None:
        scorer = _make_scorer_with_mock_fn(0.75)
        sequences = [SAMPLE_VHH, SAMPLE_VHH]

        result = scorer.score_batch(sequences)

        assert len(result) == 2
        assert all(s == pytest.approx(0.75) for s in result)
        # score_batch should call _score_sequences once with all sequences
        scorer._score_sequences.assert_called_once_with(sequences)

    def test_score_batch_does_not_populate_single_cache(self) -> None:
        scorer = _make_scorer_with_mock_fn(0.75)

        # First, score via batch
        scorer.score_batch([SAMPLE_VHH])

        # Now score via score() — should trigger _score_sequences again
        # because score_batch bypasses the single-sequence cache
        vhh = VHHSequence(SAMPLE_VHH)
        scorer.score(vhh)

        # _score_sequences should have been called twice:
        # once from score_batch, once from score()
        assert scorer._score_sequences.call_count == 2


class TestCacheMaxsize:
    """Verify cache_maxsize parameter works."""

    def test_cache_maxsize_zero_still_caches(self) -> None:
        """maxsize=0 means unlimited cache (lru_cache(maxsize=None))."""
        scorer = NativenessScorer(model_type="VHH", cache_maxsize=0)
        scorer._score_sequences = mock.Mock(side_effect=lambda seqs: [0.9] * len(seqs))

        vhh = VHHSequence(SAMPLE_VHH)
        scorer.score(vhh)
        scorer.score(vhh)

        scorer._score_sequences.assert_called_once()
