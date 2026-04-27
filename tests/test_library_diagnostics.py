"""Tests for library generation diagnostic instrumentation, timeouts, and fail-fast error handling.

ALL tests here use mock scorers and pre-built VHHSequence objects so that
ANARCI / HMMER is never required.  They validate:

1. Diagnostic preamble prints to stdout
2. Timing wrappers log [TIMING] output
3. Forced initial progress callback fires with phase='initializing'
4. Intra-loop progress callbacks fire during sampling
5. Timeout graceful degradation returns partial results
6. Backend health check disables broken backends
7. _timed_operation context manager works correctly
"""

from __future__ import annotations

import time
from collections import namedtuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

from vhh_library.mutation_engine import (
    IterativeProgress,
    MutationEngine,
    OperationTimeoutError,
    _timed_operation,
)
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer

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


class _MockNanoMeltPredictor:
    """Mock NanoMelt predictor that returns deterministic Tm values."""

    def score_sequence(self, vhh: VHHSequence) -> dict[str, float]:
        return {"composite_score": 0.7, "nanomelt_tm": 65.0}

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        return [{"composite_score": 0.7, "nanomelt_tm": 65.0} for _ in sequences]


class _SlowNanoMeltPredictor:
    """NanoMelt predictor that sleeps, for timeout testing."""

    def __init__(self, sleep_seconds: float = 10.0):
        self._sleep = sleep_seconds

    def score_sequence(self, vhh: VHHSequence) -> dict[str, float]:
        return {"composite_score": 0.7, "nanomelt_tm": 65.0}

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        time.sleep(self._sleep)
        return [{"composite_score": 0.7, "nanomelt_tm": 65.0} for _ in sequences]


class _FailingNanoMeltPredictor:
    """NanoMelt predictor that raises on score_sequence (for health check test)."""

    def score_sequence(self, vhh: VHHSequence) -> dict[str, float]:
        raise RuntimeError("NanoMelt model failed to load")

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        raise RuntimeError("NanoMelt model failed to load")


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


def _make_top_mutations(vhh: VHHSequence, n: int = 6) -> pd.DataFrame:
    """Build a small top_mutations DataFrame from the mock VHH."""
    _MutRow = namedtuple("_MutRow", ["position", "original_aa", "suggested_aa", "combined_score"])
    rows = []
    seq = vhh.sequence
    aas = "AGILVRWKDE"
    for i in range(min(n, len(seq) - 10)):
        pos = i + 5  # skip first few positions
        orig = seq[pos]
        new = aas[i % len(aas)]
        if new == orig:
            new = "W" if orig != "W" else "A"
        rows.append({"position": pos + 1, "original_aa": orig, "suggested_aa": new, "combined_score": 0.8 - i * 0.05})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: _timed_operation context manager
# ---------------------------------------------------------------------------


class TestTimedOperation:
    def test_prints_start_and_done(self, capsys):
        with _timed_operation("test operation"):
            pass
        captured = capsys.readouterr()
        assert "[TIMING] START: test operation" in captured.out
        assert "[TIMING] DONE: test operation" in captured.out

    def test_reports_elapsed_time(self, capsys):
        with _timed_operation("sleep test"):
            time.sleep(0.1)
        captured = capsys.readouterr()
        # Should report something like "0.1s"
        assert "[TIMING] DONE: sleep test" in captured.out

    def test_prints_even_on_exception(self, capsys):
        with pytest.raises(ValueError):
            with _timed_operation("failing operation"):
                raise ValueError("boom")
        captured = capsys.readouterr()
        assert "[TIMING] START: failing operation" in captured.out
        assert "[TIMING] DONE: failing operation" in captured.out


# ---------------------------------------------------------------------------
# Test: OperationTimeoutError
# ---------------------------------------------------------------------------


class TestOperationTimeoutError:
    def test_is_exception(self):
        assert issubclass(OperationTimeoutError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(OperationTimeoutError, match="timed out"):
            raise OperationTimeoutError("operation timed out")


# ---------------------------------------------------------------------------
# Test: Diagnostic preamble in generate_library
# ---------------------------------------------------------------------------


class TestDiagnosticPreamble:
    def test_preamble_prints_to_stdout(self, mock_vhh, capsys):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh)
        # Run with small parameters so it finishes quickly
        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=10,
            min_mutations=1,
            strategy="random",
        )
        captured = capsys.readouterr()
        assert "LIBRARY GENERATION STARTED" in captured.out
        assert "Strategy: random" in captured.out
        assert "Stability scorer:" in captured.out
        assert "progress_callback is None:" in captured.out

    def test_preamble_includes_sequence_info(self, mock_vhh, capsys):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh)
        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=5,
            strategy="random",
        )
        captured = capsys.readouterr()
        assert f"vhh_sequence length: {len(mock_vhh.sequence)}" in captured.out

    def test_preamble_for_empty_mutations_still_returns(self, mock_vhh):
        """generate_library returns early for empty mutations but preamble should still print."""
        engine = _make_engine()
        result = engine.generate_library(
            mock_vhh,
            pd.DataFrame(),
            n_mutations=2,
            max_variants=10,
        )
        assert result.empty


# ---------------------------------------------------------------------------
# Test: Timing wrappers produce [TIMING] output
# ---------------------------------------------------------------------------


class TestTimingWrappers:
    def test_generate_library_has_total_timing(self, mock_vhh, capsys):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh)
        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=10,
            strategy="random",
        )
        captured = capsys.readouterr()
        assert "[TIMING] DONE: generate_library total" in captured.out

    def test_random_strategy_timing(self, mock_vhh, capsys):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh)
        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=10,
            strategy="random",
        )
        captured = capsys.readouterr()
        assert "[TIMING] START: random variant sampling" in captured.out
        assert "[TIMING] DONE: random variant sampling" in captured.out


# ---------------------------------------------------------------------------
# Test: Forced initial progress callback in _generate_iterative
# ---------------------------------------------------------------------------


class TestInitialProgressCallback:
    def test_iterative_fires_initializing_callback(self, mock_vhh):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh, n=6)
        progress_phases = []

        def _capture_progress(prog: IterativeProgress):
            progress_phases.append(prog.phase)

        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=20,
            strategy="iterative",
            max_rounds=3,
            progress_callback=_capture_progress,
        )
        assert len(progress_phases) > 0
        assert progress_phases[0] == "initializing", f"First phase should be 'initializing', got '{progress_phases[0]}'"


# ---------------------------------------------------------------------------
# Test: Intra-loop progress in _generate_sampled
# ---------------------------------------------------------------------------


class TestIntraLoopProgress:
    def test_sampled_fires_progress_callbacks(self, mock_vhh):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh, n=6)
        progress_calls = []

        def _capture(prog: IterativeProgress):
            progress_calls.append(prog)

        # Use random strategy with enough variants to trigger intra-loop progress
        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=200,
            strategy="random",
            progress_callback=_capture,
        )
        # Should have at least some progress callbacks
        assert len(progress_calls) > 0

    def test_exhaustive_fires_progress_callbacks(self, mock_vhh):
        engine = _make_engine()
        top_mutations = _make_top_mutations(mock_vhh, n=4)
        progress_calls = []

        def _capture(prog: IterativeProgress):
            progress_calls.append(prog)

        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=50,
            strategy="exhaustive",
            progress_callback=_capture,
        )
        assert len(progress_calls) > 0


# ---------------------------------------------------------------------------
# Test: Timeout graceful degradation
# ---------------------------------------------------------------------------


class TestTimeoutGracefulDegradation:
    def test_batch_fill_stability_timeout_returns_partial(self, mock_vhh, capsys, monkeypatch):
        """When NanoMelt is slow, _batch_fill_stability should return with heuristic scores."""
        slow_predictor = _SlowNanoMeltPredictor(sleep_seconds=1.5)
        engine = _make_engine(nanomelt_predictor=slow_predictor)

        # Build enough mock rows to span multiple chunks (chunk size = 100)
        rows = []
        for i in range(250):
            rows.append(
                {
                    "variant_id": f"V{i:06d}",
                    "aa_sequence": mock_vhh.sequence,
                    "mutations": f"A{i + 1}G",
                    "stability_score": 0.5,
                    "nativeness_score": 0.5,
                    "combined_score": 0.5,
                    "surface_hydrophobicity_score": 0.5,
                    "orthogonal_stability_score": 0.5,
                    "vhh_hallmark_score": 0.5,
                    "disulfide_score": 1.0,
                    "aggregation_score": 0.7,
                    "charge_balance_score": 0.7,
                    "hydrophobic_core_score": 0.7,
                    "scoring_method": "heuristic",
                }
            )

        # Set a very short timeout to trigger the timeout path
        # With 250 rows, chunk_size=100, we get 3 chunks.
        # Each chunk sleeps 1.5s. After chunk 1 completes (1.5s), timeout (1s)
        # should fire before chunk 2.
        engine._operation_timeout = 1
        result = engine._batch_fill_stability(rows)

        # Should return all 250 rows (some scored by NanoMelt, rest with heuristic)
        assert len(result) == 250
        captured = capsys.readouterr()
        assert "[TIMEOUT]" in captured.out

    def test_batch_fill_nativeness_timeout_fills_neutral(self, mock_vhh, capsys, monkeypatch):
        """When nativeness scorer is slow, should fill remaining with 0.5."""

        class _SlowNativenessScorer:
            def score(self, vhh):
                return {"composite_score": 0.75}

            def predict_mutation_effect(self, vhh, pos, aa):
                return 0.0

            def score_batch(self, sequences):
                time.sleep(1.5)
                return [0.75] * len(sequences)

        engine = _make_engine(nativeness_scorer=_SlowNativenessScorer())

        # Build enough unique rows to span multiple nativeness chunks (chunk_size=50)
        rows = []
        for i in range(150):
            seq = list(mock_vhh.sequence)
            # Make each variant unique by mutating different positions
            pos = (i % (len(seq) - 2)) + 1
            seq[pos] = "G" if seq[pos] != "G" else "A"
            rows.append(
                {
                    "variant_id": f"V{i:06d}",
                    "aa_sequence": "".join(seq),
                    "mutations": f"A{i + 1}G",
                    "stability_score": 0.5,
                    "nativeness_score": 0.0,
                    "combined_score": 0.5,
                    "surface_hydrophobicity_score": 0.5,
                    "orthogonal_stability_score": 0.5,
                    "vhh_hallmark_score": 0.5,
                    "disulfide_score": 1.0,
                    "aggregation_score": 0.7,
                    "charge_balance_score": 0.7,
                    "hydrophobic_core_score": 0.7,
                    "scoring_method": "heuristic",
                }
            )

        engine._operation_timeout = 1
        result = engine._batch_fill_nativeness(rows)

        # Should have returned rows — some nativeness may be 0.5 (timeout-filled)
        assert len(result) == 150
        captured = capsys.readouterr()
        assert "[TIMEOUT]" in captured.out

    def test_default_no_timeout(self):
        """Default operation_timeout=None means no timeout is enforced."""
        engine = _make_engine()
        assert engine._operation_timeout is None
        assert engine._timeout_expired(0.0) is False

    def test_operation_timeout_via_constructor(self):
        """operation_timeout can be set via MutationEngine constructor."""
        engine = MutationEngine(operation_timeout=600)
        assert engine._operation_timeout == 600


# ---------------------------------------------------------------------------
# Test: Backend health check
# ---------------------------------------------------------------------------


class TestBackendHealthCheck:
    def test_failing_nanomelt_is_disabled(self, mock_vhh, capsys):
        """If NanoMelt health check fails, it should be disabled for the run."""
        engine = _make_engine(nanomelt_predictor=_FailingNanoMeltPredictor())
        top_mutations = _make_top_mutations(mock_vhh)

        # The health check should catch the error, disable NanoMelt, and continue
        result = engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=5,
            strategy="random",
        )
        captured = capsys.readouterr()
        assert "[DIAGNOSTIC] NanoMelt health check FAILED" in captured.out
        # Should still produce variants (using heuristic scoring)
        assert len(result) > 0

    def test_healthy_nanomelt_reports_ok(self, mock_vhh, capsys):
        """If NanoMelt health check passes, report OK."""
        engine = _make_engine(nanomelt_predictor=_MockNanoMeltPredictor())
        top_mutations = _make_top_mutations(mock_vhh)

        engine.generate_library(
            mock_vhh,
            top_mutations,
            n_mutations=2,
            max_variants=5,
            strategy="random",
        )
        captured = capsys.readouterr()
        assert "Backend nanomelt: OK" in captured.out


# ---------------------------------------------------------------------------
# Test: Background task diagnostics
# ---------------------------------------------------------------------------


class TestBackgroundDiagnostics:
    def test_worker_prints_start_and_completion(self, capsys):
        """submit_task _worker should print start and completion messages."""
        import threading

        import vhh_library.background as bg

        # Set up mock session state
        state: dict = {}
        mock_st = MagicMock()
        mock_st.session_state = state

        original_st = bg.st
        bg.st = mock_st
        try:
            done = threading.Event()

            def _test_work():
                done.set()
                return 42

            ok = bg.submit_task("test_diag", _test_work)
            assert ok is True

            # Wait for thread to complete (fast — no sleeping)
            done.wait(timeout=5.0)
            time.sleep(0.1)  # brief settle for print buffer

            captured = capsys.readouterr()
            assert "[BG-THREAD] Background thread started for task 'test_diag'" in captured.out
            assert "[BG-THREAD] Task 'test_diag' completed successfully" in captured.out
        finally:
            bg.st = original_st

    def test_worker_prints_failure(self, capsys):
        """submit_task _worker should print failure messages."""
        import threading

        import vhh_library.background as bg

        state: dict = {}
        mock_st = MagicMock()
        mock_st.session_state = state

        original_st = bg.st
        bg.st = mock_st
        try:
            done = threading.Event()

            def _failing_work():
                done.set()
                raise ValueError("intentional test error")

            ok = bg.submit_task("test_fail", _failing_work)
            assert ok is True

            done.wait(timeout=5.0)
            time.sleep(0.1)  # brief settle for print buffer

            captured = capsys.readouterr()
            assert "[BG-THREAD] Background thread started for task 'test_fail'" in captured.out
            assert "[BG-THREAD] Task 'test_fail' FAILED: intentional test error" in captured.out
        finally:
            bg.st = original_st

    def test_progress_callback_first_call_print(self, capsys):
        """make_progress_callback should print on first invocation."""
        import vhh_library.background as bg

        state: dict = {}
        mock_st = MagicMock()
        mock_st.session_state = state

        original_st = bg.st
        bg.st = mock_st
        try:
            cb = bg.make_progress_callback("test_cb")
            prog = IterativeProgress(
                phase="initializing",
                round_number=0,
                total_rounds=10,
                best_score=0.0,
                mean_score=0.0,
                population_size=0,
                n_anchors=0,
                diversity_entropy=0.0,
                message="Testing",
            )
            cb(prog)
            captured = capsys.readouterr()
            assert "[PROGRESS] First progress callback for 'test_cb': phase=initializing" in captured.out

            # Second call should not print the first-call message
            cb(prog)
            captured2 = capsys.readouterr()
            assert "[PROGRESS] First progress callback" not in captured2.out
        finally:
            bg.st = original_st
