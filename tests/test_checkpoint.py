"""Tests for vhh_library.checkpoint — disk-based checkpointing."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from vhh_library.checkpoint import (
    checkpoint_path,
    cleanup_stale_checkpoints,
    compute_run_id,
    load_checkpoint,
    load_result,
    remove_checkpoint,
    result_path,
    save_checkpoint,
    save_result,
)


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for checkpoint tests."""
    return tmp_path


# ---------------------------------------------------------------------------
# compute_run_id
# ---------------------------------------------------------------------------


class TestComputeRunId:
    def test_deterministic(self) -> None:
        id1 = compute_run_id("ACDEFG", n_mutations=3, max_variants=100, strategy="iterative")
        id2 = compute_run_id("ACDEFG", n_mutations=3, max_variants=100, strategy="iterative")
        assert id1 == id2

    def test_different_sequence_different_id(self) -> None:
        id1 = compute_run_id("ACDEFG", n_mutations=3, max_variants=100)
        id2 = compute_run_id("GHIKLM", n_mutations=3, max_variants=100)
        assert id1 != id2

    def test_different_params_different_id(self) -> None:
        id1 = compute_run_id("ACDEFG", n_mutations=3, max_variants=100)
        id2 = compute_run_id("ACDEFG", n_mutations=5, max_variants=100)
        assert id1 != id2

    def test_returns_short_hex(self) -> None:
        rid = compute_run_id("ABC")
        assert len(rid) == 16
        # All hex characters
        int(rid, 16)


# ---------------------------------------------------------------------------
# save / load checkpoint
# ---------------------------------------------------------------------------


class TestSaveLoadCheckpoint:
    def test_save_creates_file(self, tmp_root: Path) -> None:
        df = pd.DataFrame({"variant_id": ["V1"], "aa_sequence": ["ACDEF"]})
        path = save_checkpoint(tmp_root, "abc123", df, completed_rounds=3)
        assert path.is_file()
        assert path == checkpoint_path(tmp_root, "abc123")

    def test_round_trip(self, tmp_root: Path) -> None:
        df = pd.DataFrame(
            {
                "variant_id": ["V1", "V2"],
                "aa_sequence": ["ACDEF", "GHIKL"],
                "combined_score": [0.8, 0.6],
            }
        )
        save_checkpoint(tmp_root, "rt01", df, completed_rounds=5)
        loaded = load_checkpoint(tmp_root, "rt01")
        assert loaded is not None
        loaded_df, rounds = loaded
        assert rounds == 5
        assert len(loaded_df) == 2
        assert list(loaded_df["variant_id"]) == ["V1", "V2"]
        assert loaded_df["combined_score"].iloc[0] == pytest.approx(0.8)

    def test_load_missing_returns_none(self, tmp_root: Path) -> None:
        assert load_checkpoint(tmp_root, "nonexistent") is None

    def test_overwrite_updates_file(self, tmp_root: Path) -> None:
        df1 = pd.DataFrame({"variant_id": ["V1"]})
        save_checkpoint(tmp_root, "ow01", df1, completed_rounds=1)

        df2 = pd.DataFrame({"variant_id": ["V1", "V2", "V3"]})
        save_checkpoint(tmp_root, "ow01", df2, completed_rounds=3)

        loaded = load_checkpoint(tmp_root, "ow01")
        assert loaded is not None
        loaded_df, rounds = loaded
        assert rounds == 3
        assert len(loaded_df) == 3


# ---------------------------------------------------------------------------
# save / load result
# ---------------------------------------------------------------------------


class TestSaveLoadResult:
    def test_round_trip(self, tmp_root: Path) -> None:
        df = pd.DataFrame(
            {
                "variant_id": ["V1", "V2"],
                "stability_score": [0.9, 0.7],
            }
        )
        path = save_result(tmp_root, "res01", df)
        assert path.is_file()
        assert path == result_path(tmp_root, "res01")

        loaded = load_result(tmp_root, "res01")
        assert loaded is not None
        assert len(loaded) == 2

    def test_load_missing_returns_none(self, tmp_root: Path) -> None:
        assert load_result(tmp_root, "nope") is None


# ---------------------------------------------------------------------------
# remove_checkpoint
# ---------------------------------------------------------------------------


class TestRemoveCheckpoint:
    def test_removes_existing(self, tmp_root: Path) -> None:
        df = pd.DataFrame({"x": [1]})
        save_checkpoint(tmp_root, "rm01", df, completed_rounds=1)
        assert checkpoint_path(tmp_root, "rm01").is_file()
        remove_checkpoint(tmp_root, "rm01")
        assert not checkpoint_path(tmp_root, "rm01").is_file()

    def test_idempotent_on_missing(self, tmp_root: Path) -> None:
        # Should not raise
        remove_checkpoint(tmp_root, "missing")


# ---------------------------------------------------------------------------
# cleanup_stale_checkpoints
# ---------------------------------------------------------------------------


class TestCleanupStaleCheckpoints:
    def test_removes_old_files(self, tmp_root: Path) -> None:
        df = pd.DataFrame({"x": [1]})
        save_checkpoint(tmp_root, "stale01", df, completed_rounds=1)
        path = checkpoint_path(tmp_root, "stale01")

        # Backdate by 25 hours
        old_time = time.time() - 90_000
        os.utime(path, (old_time, old_time))

        removed = cleanup_stale_checkpoints(tmp_root, max_age_seconds=86_400)
        assert removed == 1
        assert not path.is_file()

    def test_keeps_recent_files(self, tmp_root: Path) -> None:
        df = pd.DataFrame({"x": [1]})
        save_checkpoint(tmp_root, "recent01", df, completed_rounds=1)

        removed = cleanup_stale_checkpoints(tmp_root, max_age_seconds=86_400)
        assert removed == 0
        assert checkpoint_path(tmp_root, "recent01").is_file()

    def test_no_dir_returns_zero(self, tmp_root: Path) -> None:
        empty = tmp_root / "empty_subdir"
        assert cleanup_stale_checkpoints(empty) == 0


# ---------------------------------------------------------------------------
# Integration: checkpoint-in-iterative-engine
# ---------------------------------------------------------------------------


class TestCheckpointInEngine:
    """Verify that generate_library writes checkpoints during iterative runs."""

    @pytest.fixture()
    def engine(self):
        """Lightweight engine with mock scorers."""
        from vhh_library.mutation_engine import MutationEngine
        from vhh_library.stability import StabilityScorer

        class _MockNat:
            def score(self, vhh):
                return {"composite_score": 0.7}

            def predict_mutation_effect(self, vhh, pos, aa):
                return 0.01

            def score_batch(self, seqs):
                return [0.7] * len(seqs)

        return MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_MockNat(),
        )

    @pytest.fixture()
    def vhh(self):
        from tests.conftest import SAMPLE_VHH
        from vhh_library.sequence import VHHSequence

        return VHHSequence(SAMPLE_VHH)

    @pytest.fixture()
    def ranked(self, engine, vhh):
        return engine.rank_single_mutations(vhh)

    def test_checkpoint_written_during_iterative(self, engine, vhh, ranked, tmp_root: Path) -> None:
        """Checkpoint file should exist after a short iterative run completes."""
        if ranked.empty:
            pytest.skip("No mutations ranked (ANARCI unavailable)")

        rid = compute_run_id(
            vhh.sequence,
            n_mutations=2,
            max_variants=50,
            min_mutations=1,
            strategy="iterative",
        )

        df = engine.generate_library(
            vhh,
            ranked.head(10),
            n_mutations=2,
            max_variants=50,
            min_mutations=1,
            strategy="iterative",
            max_rounds=3,
            checkpoint_dir=tmp_root,
        )

        # The final result should be saved.
        assert result_path(tmp_root, rid).is_file()
        # The intermediate checkpoint should have been cleaned up.
        assert not checkpoint_path(tmp_root, rid).is_file()
        # The returned DataFrame should not be empty.
        assert not df.empty

    def test_cleanup_after_completion(self, engine, vhh, ranked, tmp_root: Path) -> None:
        """After generate_library finishes, checkpoint file is removed."""
        if ranked.empty:
            pytest.skip("No mutations ranked (ANARCI unavailable)")

        rid = compute_run_id(
            vhh.sequence,
            n_mutations=2,
            max_variants=50,
            min_mutations=1,
            strategy="iterative",
        )

        engine.generate_library(
            vhh,
            ranked.head(10),
            n_mutations=2,
            max_variants=50,
            min_mutations=1,
            strategy="iterative",
            max_rounds=3,
            checkpoint_dir=tmp_root,
        )

        assert not checkpoint_path(tmp_root, rid).is_file()

    def test_no_checkpoint_without_opt_in(self, engine, vhh, ranked, tmp_root: Path) -> None:
        """Without checkpoint_dir, no files are written."""
        if ranked.empty:
            pytest.skip("No mutations ranked (ANARCI unavailable)")

        engine.generate_library(
            vhh,
            ranked.head(10),
            n_mutations=2,
            max_variants=50,
            min_mutations=1,
            strategy="iterative",
            max_rounds=3,
        )

        # No checkpoint subdir should have been created
        from vhh_library.checkpoint import _CHECKPOINT_SUBDIR

        assert not (tmp_root / _CHECKPOINT_SUBDIR).exists()

    def test_empty_mutations_no_crash(self, engine, vhh, tmp_root: Path) -> None:
        """Empty mutation list should not crash with checkpoint_dir set."""
        empty_df = pd.DataFrame()
        df = engine.generate_library(
            vhh,
            empty_df,
            n_mutations=2,
            max_variants=50,
            checkpoint_dir=tmp_root,
        )
        assert df.empty
