"""Tests for rank_single_mutations diagnostic instrumentation.

ALL tests here use mock scorers and pre-built VHHSequence objects so that
ANARCI / HMMER is never required.  They validate:

1. Diagnostic preamble prints to stdout
2. Progress callbacks fire for enumeration, stability scoring, nativeness
   scoring phases, and ranking complete
3. Backend health check disables broken NanoMelt gracefully
4. Nativeness failure degrades gracefully with delta_nativeness=0.0
5. Timeout — NanoMelt batch scoring exceeds limit falls back to heuristics
6. Summary timing prints at end
7. Background thread submission (via mock)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from vhh_library.mutation_engine import (
    IterativeProgress,
    MutationEngine,
)
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer

# ---------------------------------------------------------------------------
# Shared test sequence (same as conftest.py)
# ---------------------------------------------------------------------------

_TEST_SEQ = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


# ---------------------------------------------------------------------------
# Mock scorers
# ---------------------------------------------------------------------------


class _MockNativenessScorer:
    """Deterministic mock nativeness scorer — no ANARCI needed."""

    def score(self, vhh: VHHSequence) -> dict:
        return {"composite_score": 0.75}

    def predict_mutation_effect(self, vhh: VHHSequence, position: int | str, new_aa: str) -> float:
        return 0.02 if new_aa in "AGILV" else -0.01

    def score_batch(self, sequences: list[str]) -> list[float]:
        return [0.75] * len(sequences)


class _FailingNativenessScorer:
    """Nativeness scorer that always raises."""

    def score(self, vhh: VHHSequence) -> dict:
        raise RuntimeError("AbNatiV model unavailable")

    def predict_mutation_effect(self, vhh: VHHSequence, position: int | str, new_aa: str) -> float:
        raise RuntimeError("AbNatiV model unavailable")

    def score_batch(self, sequences: list[str]) -> list[float]:
        raise RuntimeError("AbNatiV model unavailable")


class _MockNanoMeltPredictor:
    """Mock NanoMelt predictor that returns deterministic Tm values."""

    def score_sequence(self, vhh: VHHSequence) -> dict[str, float]:
        return {"composite_score": 0.7, "nanomelt_tm": 65.0}

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        return [{"composite_score": 0.7, "nanomelt_tm": 65.0} for _ in sequences]

    def score_batch_prealigned(self, parent_seq: str, variant_seqs: list[str]) -> list[dict[str, float]]:
        return [{"composite_score": 0.7, "nanomelt_tm": 65.0} for _ in variant_seqs]


class _FailingNanoMeltPredictor:
    """NanoMelt predictor that raises on every call."""

    def score_sequence(self, vhh: VHHSequence) -> dict[str, float]:
        raise RuntimeError("NanoMelt model failed to load")

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        raise RuntimeError("NanoMelt model failed to load")

    def score_batch_prealigned(self, parent_seq: str, variant_seqs: list[str]) -> list[dict[str, float]]:
        raise RuntimeError("NanoMelt model failed to load")


class _SlowNanoMeltPredictor:
    """NanoMelt predictor whose batch methods sleep, for timeout testing."""

    def __init__(self, sleep_seconds: float = 10.0):
        self._sleep = sleep_seconds

    def score_sequence(self, vhh: VHHSequence) -> dict[str, float]:
        # Health check — fast
        return {"composite_score": 0.7, "nanomelt_tm": 65.0}

    def score_batch_prealigned(self, parent_seq: str, variant_seqs: list[str]) -> list[dict[str, float]]:
        time.sleep(self._sleep)
        return [{"composite_score": 0.7, "nanomelt_tm": 65.0} for _ in variant_seqs]

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        time.sleep(self._sleep)
        return [{"composite_score": 0.7, "nanomelt_tm": 65.0} for _ in sequences]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    nativeness_scorer=None,
    nanomelt_predictor=None,
    esm_scorer=None,
) -> MutationEngine:
    """Build a MutationEngine with mock scorers."""
    stability_scorer = StabilityScorer()
    if nanomelt_predictor is not None:
        stability_scorer.nanomelt_predictor = nanomelt_predictor
    if esm_scorer is not None:
        stability_scorer.esm_scorer = esm_scorer
    return MutationEngine(
        stability_scorer=stability_scorer,
        nativeness_scorer=nativeness_scorer or _MockNativenessScorer(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_vhh_rank() -> VHHSequence:
    """VHHSequence with pre-populated IMGT numbering — no ANARCI needed.

    Only a short sequence is used so tests run quickly (few positions × 19 AAs).
    """
    seq = _TEST_SEQ[:20]  # 20 residues → ~380 candidates max before PTM/exclusion filters
    vhh = object.__new__(VHHSequence)
    vhh.sequence = seq
    vhh.length = len(seq)
    vhh.strict = True
    vhh.chain_type = "H"
    vhh.species = "alpaca"
    numbered: dict[str, str] = {}
    pos_to_idx: dict[str, int] = {}
    for idx, aa in enumerate(seq):
        key = str(idx + 1)
        numbered[key] = aa
        pos_to_idx[key] = idx
    vhh.imgt_numbered = numbered
    vhh._pos_to_seq_idx = pos_to_idx
    vhh.validation_result = {"valid": True, "errors": [], "warnings": []}
    return vhh


# ---------------------------------------------------------------------------
# Test: diagnostic preamble
# ---------------------------------------------------------------------------


class TestRankDiagnosticPreamble:
    def test_preamble_prints_to_stdout(self, mock_vhh_rank, capsys):
        engine = _make_engine()
        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        assert "RANK SINGLE MUTATIONS STARTED" in captured.out

    def test_preamble_includes_all_diagnostic_fields(self, mock_vhh_rank, capsys):
        engine = _make_engine()
        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        assert f"vhh_sequence length: {len(mock_vhh_rank.sequence)}" in captured.out
        assert "Stability scorer:" in captured.out
        assert "Nativeness scorer:" in captured.out
        assert "Active weights:" in captured.out
        assert "progress_callback is None:" in captured.out
        assert "Has NanoMelt:" in captured.out
        assert "Has ESM on stability:" in captured.out

    def test_summary_timing_prints_at_end(self, mock_vhh_rank, capsys):
        engine = _make_engine()
        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        assert "RANK SINGLE MUTATIONS COMPLETE" in captured.out
        assert "Total time:" in captured.out
        assert "Candidates generated:" in captured.out
        assert "Final ranked mutations:" in captured.out


# ---------------------------------------------------------------------------
# Test: progress callbacks
# ---------------------------------------------------------------------------


class TestRankProgressCallbacks:
    def test_progress_callback_fires_for_enumeration_phase(self, mock_vhh_rank):
        engine = _make_engine()
        progress_phases: list[str] = []

        def _cb(prog: IterativeProgress) -> None:
            progress_phases.append(prog.phase)

        # Use only a few mutable positions so enumeration is fast but fires
        # at least one 100-candidate checkpoint (need 100+ candidates).
        # With 6 mutable positions × 19 AAs = ~114 candidates (before PTM filter).
        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(7, len(mock_vhh_rank.sequence) + 1)),
            progress_callback=_cb,
        )
        # Should fire at least one enumerating_candidates phase
        assert "enumerating_candidates" in progress_phases, (
            f"Expected 'enumerating_candidates' in progress phases, got: {progress_phases}"
        )

    def test_progress_callback_fires_for_batch_stability_scoring(self, mock_vhh_rank):
        engine = _make_engine()
        phases: list[str] = []

        def _cb(prog: IterativeProgress) -> None:
            phases.append(prog.phase)

        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
            progress_callback=_cb,
        )
        assert "batch_stability_scoring" in phases, (
            f"Expected 'batch_stability_scoring' in progress phases, got: {phases}"
        )

    def test_progress_callback_fires_for_batch_nativeness_scoring(self, mock_vhh_rank):
        engine = _make_engine()
        phases: list[str] = []

        def _cb(prog: IterativeProgress) -> None:
            phases.append(prog.phase)

        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
            progress_callback=_cb,
        )
        assert "batch_nativeness_scoring" in phases, (
            f"Expected 'batch_nativeness_scoring' in progress phases, got: {phases}"
        )

    def test_progress_callback_fires_ranking_complete(self, mock_vhh_rank):
        engine = _make_engine()
        phases: list[str] = []

        def _cb(prog: IterativeProgress) -> None:
            phases.append(prog.phase)

        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
            progress_callback=_cb,
        )
        assert "ranking_complete" in phases, (
            f"Expected 'ranking_complete' in progress phases, got: {phases}"
        )


# ---------------------------------------------------------------------------
# Test: backend health check — NanoMelt fails gracefully
# ---------------------------------------------------------------------------


class TestRankBackendHealthCheck:
    def test_failing_nanomelt_is_disabled_and_continues(self, mock_vhh_rank, capsys):
        """If NanoMelt health check fails, it must be disabled and ranking continues."""
        engine = _make_engine(nanomelt_predictor=_FailingNanoMeltPredictor())
        result = engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        # Health check failure should be printed
        assert "[RANKING] NanoMelt health check FAILED" in captured.out
        assert "[RANKING] Disabling NanoMelt for this ranking run" in captured.out
        # Should still return a DataFrame (possibly empty if very few candidates)
        assert result is not None

    def test_healthy_nanomelt_reports_ok(self, mock_vhh_rank, capsys):
        """A healthy NanoMelt backend should print the OK message."""
        engine = _make_engine(nanomelt_predictor=_MockNanoMeltPredictor())
        engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        assert "[RANKING] NanoMelt health check OK" in captured.out

    def test_nativeness_health_check_failure_is_logged(self, mock_vhh_rank, capsys):
        """If nativeness health check fails, it is logged but ranking continues."""
        engine = _make_engine(nativeness_scorer=_FailingNativenessScorer())
        # Nativeness fails — ranking should still return without raising
        result = engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        assert "[RANKING] Nativeness health check FAILED" in captured.out
        # Result is a DataFrame (may be empty or have delta_nativeness=0.0)
        assert result is not None


# ---------------------------------------------------------------------------
# Test: nativeness failure degrades gracefully
# ---------------------------------------------------------------------------


class TestRankNativenessFallback:
    def test_nativeness_failure_sets_zero_delta(self, mock_vhh_rank, capsys):
        """When nativeness scoring fails in the main body, delta_nativeness=0.0."""
        engine = _make_engine(nativeness_scorer=_FailingNativenessScorer())
        result = engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        # With failing nativeness, delta_nativeness should be 0.0 for all rows
        if not result.empty:
            assert (result["delta_nativeness"] == 0.0).all(), (
                f"Expected delta_nativeness=0.0 for all rows, got: {result['delta_nativeness'].unique()}"
            )


# ---------------------------------------------------------------------------
# Test: timeout — NanoMelt batch scoring falls back to heuristics
# ---------------------------------------------------------------------------


class TestRankTimeout:
    def test_slow_nanomelt_batch_uses_heuristic_deltas(self, mock_vhh_rank, capsys, monkeypatch):
        """When NanoMelt batch scoring sleeps beyond timeout, fall back to heuristics."""
        import vhh_library.mutation_engine as me

        # Use a slow predictor that is healthy for the single-sequence health
        # check but sleeps during batch scoring.
        slow_predictor = _SlowNanoMeltPredictor(sleep_seconds=10.0)
        engine = _make_engine(nanomelt_predictor=slow_predictor)

        # Set a very short timeout so the batch scoring triggers it
        monkeypatch.setattr(me, "_OPERATION_TIMEOUT_SECONDS", 2)

        result = engine.rank_single_mutations(
            mock_vhh_rank,
            off_limits=set(str(i) for i in range(4, len(mock_vhh_rank.sequence) + 1)),
        )
        captured = capsys.readouterr()
        # Should log a timeout warning
        assert "[TIMEOUT]" in captured.out
        # Should still return a valid (non-crashing) result
        assert result is not None


# ---------------------------------------------------------------------------
# Test: enumeration progress printing
# ---------------------------------------------------------------------------


class TestRankEnumerationProgress:
    def test_enumeration_progress_prints_with_enough_candidates(self, capsys):
        """When enough candidates are generated, enumeration progress is printed."""
        # Use a long sequence with many mutable positions so we cross the
        # 100-candidate threshold that triggers progress printing.
        long_seq = _TEST_SEQ  # ~117 residues → many mutable positions × 19 AAs
        vhh = object.__new__(VHHSequence)
        vhh.sequence = long_seq
        vhh.length = len(long_seq)
        vhh.strict = True
        vhh.chain_type = "H"
        vhh.species = "alpaca"
        numbered: dict[str, str] = {}
        pos_to_idx: dict[str, int] = {}
        for idx, aa in enumerate(long_seq):
            key = str(idx + 1)
            numbered[key] = aa
            pos_to_idx[key] = idx
        vhh.imgt_numbered = numbered
        vhh._pos_to_seq_idx = pos_to_idx
        vhh.validation_result = {"valid": True, "errors": [], "warnings": []}

        engine = _make_engine()
        # Allow only a small window of mutable positions (6 positions = ~114 candidates)
        off_limits = set(str(i) for i in range(7, len(long_seq) + 1))
        engine.rank_single_mutations(vhh, off_limits=off_limits)

        captured = capsys.readouterr()
        assert "[RANKING] Enumerated" in captured.out


# ---------------------------------------------------------------------------
# Test: background thread submission (app-level mock)
# ---------------------------------------------------------------------------


class TestRankBackgroundSubmission:
    def test_submit_task_is_called_with_rank_mutations(self):
        """Verify that submit_task('rank_mutations', …) would be called correctly."""
        # We test the callable structure rather than importing app.py (which
        # requires Streamlit context).  This test ensures the rank_work
        # closure captures the engine and forwards progress_callback.
        engine = _make_engine()
        progress_calls: list[str] = []

        def _fake_progress_cb(prog: IterativeProgress) -> None:
            progress_calls.append(prog.phase)

        seq = _TEST_SEQ[:10]
        vhh = object.__new__(VHHSequence)
        vhh.sequence = seq
        vhh.length = len(seq)
        vhh.strict = True
        vhh.chain_type = "H"
        vhh.species = "alpaca"
        numbered = {str(i + 1): aa for i, aa in enumerate(seq)}
        vhh.imgt_numbered = numbered
        vhh._pos_to_seq_idx = {str(i + 1): i for i in range(len(seq))}
        vhh.validation_result = {"valid": True, "errors": [], "warnings": []}

        # Simulate the closure created in app.py's button handler
        _vhh = vhh
        _off_limits = set(str(i) for i in range(4, len(seq) + 1))
        _rank_progress_cb = _fake_progress_cb

        def _rank_work():
            return engine.rank_single_mutations(
                _vhh,
                off_limits=_off_limits,
                excluded_target_aas=None,
                max_per_position=3,
                progress_callback=_rank_progress_cb,
            )

        # Call the work function directly (simulating what submit_task does)
        result = _rank_work()
        assert isinstance(result, __import__("pandas").DataFrame)
        # Progress callback should have been invoked
        assert len(progress_calls) > 0


# ---------------------------------------------------------------------------
# Test: new phases registered in background.py _simple_phases
# ---------------------------------------------------------------------------


class TestRankProgressPhases:
    def test_new_phases_in_simple_phases(self):
        """Verify new ranking phases are registered in make_progress_callback."""

        import vhh_library.background as bg

        state: dict = {}
        mock_st = MagicMock()
        mock_st.session_state = state

        original_st = bg.st
        bg.st = mock_st
        try:
            cb = bg.make_progress_callback("rank_mutations")
            for phase in [
                "enumerating_candidates",
                "batch_stability_scoring",
                "batch_nativeness_scoring",
                "ranking_complete",
            ]:
                prog = IterativeProgress(
                    phase=phase,
                    round_number=1,
                    total_rounds=10,
                    best_score=0.0,
                    mean_score=0.0,
                    population_size=100,
                    n_anchors=0,
                    diversity_entropy=0.0,
                    message=f"Testing phase {phase}",
                )
                cb(prog)
                text = state.get("bg_rank_mutations_progress_text", "")
                # For simple phases, message (not iterative format) should be used
                assert f"Testing phase {phase}" in text, (
                    f"Phase {phase!r} not handled as simple phase; text={text!r}"
                )
        finally:
            bg.st = original_st
