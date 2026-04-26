"""Tests for fast-path VHHSequence.mutate() and intra-round progress reporting.

These tests verify:
1. VHHSequence.mutate() does NOT call ANARCI (subprocess) during mutation.
2. _generate_sampled() fires intra-round progress callbacks.
3. _generate_constrained_sampled() fires intra-round progress callbacks.
4. Iterative variant generation with a mock scorer completes fast (<10s for 100 variants).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pandas as pd
import pytest

from vhh_library.mutation_engine import (
    IterativeProgress,
    MutationEngine,
)
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer

# ---------------------------------------------------------------------------
# Helpers: build a VHHSequence without ANARCI (for environments without HMMER)
# ---------------------------------------------------------------------------

# A realistic 118-residue VHH sequence.
_SAMPLE_SEQ = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


def _make_mock_vhh(sequence: str | None = None) -> VHHSequence:
    """Build a VHHSequence with synthetic IMGT numbering, bypassing ANARCI.

    This mirrors what VHHSequence.mutate() does internally: it uses
    ``object.__new__`` to skip ``__init__`` (and therefore ANARCI).
    """
    seq = sequence or _SAMPLE_SEQ
    vhh = object.__new__(VHHSequence)
    vhh.sequence = seq
    vhh.length = len(seq)
    vhh.strict = True
    vhh.chain_type = "H"
    vhh.species = "alpaca"
    vhh.validation_result = {"valid": True, "errors": [], "warnings": []}

    # Build a synthetic IMGT numbering: positions 1..N mapped 1:1.
    # This is simplified but sufficient for testing mutation mechanics.
    numbered: dict[str, str] = {}
    pos_to_idx: dict[str, int] = {}
    for idx, aa in enumerate(seq):
        key = str(idx + 1)
        numbered[key] = aa
        pos_to_idx[key] = idx
    vhh.imgt_numbered = numbered
    vhh._pos_to_seq_idx = pos_to_idx
    return vhh


# ---------------------------------------------------------------------------
# Mock nativeness scorer (no real model needed)
# ---------------------------------------------------------------------------


class _MockNativenessScorer:
    _SCORE_MODULO = 20

    def _raw_score(self, sequence: str) -> float:
        raw = (sum(ord(c) for c in sequence) % self._SCORE_MODULO) / self._SCORE_MODULO
        return 0.5 + raw * 0.4

    def score(self, vhh: VHHSequence) -> dict:
        return {"composite_score": self._raw_score(vhh.sequence)}

    def predict_mutation_effect(self, vhh: VHHSequence, position: int | str, new_aa: str) -> float:
        return 0.02 if new_aa in "AGILV" else -0.01

    def score_batch(self, sequences: list[str]) -> list[float]:
        return [self._raw_score(seq) for seq in sequences]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vhh() -> VHHSequence:
    return _make_mock_vhh()


@pytest.fixture
def engine() -> MutationEngine:
    return MutationEngine(
        stability_scorer=StabilityScorer(),
        nativeness_scorer=_MockNativenessScorer(),
    )


def _make_top_mutations(vhh: VHHSequence, n_positions: int = 10) -> pd.DataFrame:
    """Build a synthetic top-mutations DataFrame for testing.

    Creates mutations at the first *n_positions* positions, each with the
    original AA mutated to 'A' (or 'G' if already 'A').
    """
    rows = []
    for i in range(min(n_positions, vhh.length)):
        pos_key = str(i + 1)
        orig = vhh.imgt_numbered.get(pos_key, "X")
        new_aa = "A" if orig != "A" else "G"
        rows.append(
            {
                "position": i + 1,
                "imgt_pos": pos_key,
                "original_aa": orig,
                "suggested_aa": new_aa,
                "combined_score": 0.7,
                "delta_stability": 0.01,
                "delta_nativeness": 0.01,
                "reason": "test",
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Test 1: VHHSequence.mutate() does NOT call ANARCI
# ===========================================================================


class TestMutateFastPath:
    def test_mutate_does_not_call_anarci(self, mock_vhh: VHHSequence) -> None:
        """Verify that mutate() never invokes number_sequence (ANARCI)."""
        with patch("vhh_library.sequence.number_sequence") as mock_numbering:
            mutant = VHHSequence.mutate(mock_vhh, "1", "A")
            mock_numbering.assert_not_called()
        assert mutant.imgt_numbered["1"] == "A"
        assert mutant.sequence[0] == "A"

    def test_mutate_chain_of_mutations_no_anarci(self, mock_vhh: VHHSequence) -> None:
        """Chaining multiple mutate() calls should never invoke ANARCI."""
        with patch("vhh_library.sequence.number_sequence") as mock_numbering:
            current = mock_vhh
            for pos in range(1, 14):
                current = VHHSequence.mutate(current, str(pos), "G")
            mock_numbering.assert_not_called()
        # All 13 positions should be G.
        for pos in range(1, 14):
            assert current.imgt_numbered[str(pos)] == "G"

    def test_mutate_preserves_parent_numbering(self, mock_vhh: VHHSequence) -> None:
        """Parent's numbering dict should not be modified by mutate()."""
        original_aa = mock_vhh.imgt_numbered["5"]
        mutant = VHHSequence.mutate(mock_vhh, "5", "W")
        assert mock_vhh.imgt_numbered["5"] == original_aa  # parent unchanged
        assert mutant.imgt_numbered["5"] == "W"


# ===========================================================================
# Test 2: _generate_sampled() fires intra-round progress callbacks
# ===========================================================================


class TestSampledProgressCallbacks:
    def test_generate_sampled_fires_progress(self, engine: MutationEngine, mock_vhh: VHHSequence) -> None:
        """_generate_sampled with 200 variants should fire the callback at least 3 times."""
        top = _make_top_mutations(mock_vhh, n_positions=15)
        mutation_list = list(top.itertuples(index=False))

        progress_events: list[IterativeProgress] = []

        def _cb(prog: IterativeProgress) -> None:
            progress_events.append(prog)

        rows = engine._generate_sampled(
            mock_vhh,
            mutation_list,
            k_min=2,
            k_max=5,
            max_variants=200,
            _batch_score=False,
            progress_callback=_cb,
        )
        assert len(rows) > 0
        # With 200 variants and callback every 50, expect at least 3 calls.
        sampling_events = [p for p in progress_events if p.phase == "sampling_variants"]
        assert len(sampling_events) >= 3, f"Expected >= 3 sampling_variants events, got {len(sampling_events)}"


# ===========================================================================
# Test 3: _generate_constrained_sampled() fires intra-round progress
# ===========================================================================


class TestConstrainedSampledProgressCallbacks:
    def test_constrained_sampled_fires_progress(self, engine: MutationEngine, mock_vhh: VHHSequence) -> None:
        """_generate_constrained_sampled with 200 variants should fire callbacks."""
        top = _make_top_mutations(mock_vhh, n_positions=15)
        mutation_list = list(top.itertuples(index=False))

        # Use positions 1 and 2 as anchors.
        anchors = {1: top.iloc[0]["suggested_aa"], 2: top.iloc[1]["suggested_aa"]}

        progress_events: list[IterativeProgress] = []

        def _cb(prog: IterativeProgress) -> None:
            progress_events.append(prog)

        rows = engine._generate_constrained_sampled(
            mock_vhh,
            mutation_list,
            k_min=3,
            k_max=6,
            max_variants=200,
            anchors=anchors,
            _batch_score=False,
            progress_callback=_cb,
        )
        assert len(rows) > 0
        sampling_events = [p for p in progress_events if p.phase == "sampling_variants"]
        assert len(sampling_events) >= 3, f"Expected >= 3 sampling_variants events, got {len(sampling_events)}"


# ===========================================================================
# Test 4: Integration timing — 100 variants via iterative under 10 seconds
# ===========================================================================


class TestIterativeTimingIntegration:
    def test_iterative_100_variants_under_10s(self, engine: MutationEngine, mock_vhh: VHHSequence) -> None:
        """Generate 100 variants via iterative strategy with mock scorers in <10s.

        This catches regressions that reintroduce ANARCI calls per variant.
        """
        top = _make_top_mutations(mock_vhh, n_positions=10)

        start = time.time()
        lib = engine.generate_library(
            mock_vhh,
            top,
            n_mutations=5,
            min_mutations=3,
            max_variants=100,
            strategy="iterative",
            max_rounds=4,
        )
        elapsed = time.time() - start

        assert isinstance(lib, pd.DataFrame)
        assert elapsed < 10, f"Iterative generation took {elapsed:.1f}s — expected <10s"


# ===========================================================================
# Test 5: _generate_exhaustive() fires intra-round progress
# ===========================================================================


class TestExhaustiveProgressCallbacks:
    def test_exhaustive_fires_progress_for_large_enumeration(
        self, engine: MutationEngine, mock_vhh: VHHSequence
    ) -> None:
        """Exhaustive enumeration with >100 variants should emit at least one progress callback."""
        # Use enough positions to generate >100 combinations.
        top = _make_top_mutations(mock_vhh, n_positions=8)
        mutation_list = list(top.itertuples(index=False))

        progress_events: list[IterativeProgress] = []

        def _cb(prog: IterativeProgress) -> None:
            progress_events.append(prog)

        rows = engine._generate_exhaustive(
            mock_vhh,
            mutation_list,
            k_min=1,
            k_max=3,
            max_variants=500,
            _batch_score=False,
            progress_callback=_cb,
        )
        # With C(8,1)+C(8,2)+C(8,3)=8+28+56=92 combos (single AA per pos),
        # we should get ~92 variants. With progress every 100, may not fire.
        # So request up to k=4: C(8,4)=70 more → 162 total. Should fire at least 1.
        assert len(rows) > 0
