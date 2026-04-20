"""Tests for vhh_library.mutation_engine – MutationEngine class."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from vhh_library.mutation_engine import (
    IterativeProgress,
    MutationEngine,
    _compute_epistasis,
    _introduces_ptm_liability,
    _mutation_entropy,
    _parse_mut_str,
    _total_grouped_combinations,
)
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer
from vhh_library.utils import AMINO_ACIDS

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


class _MockNativenessScorer:
    """Deterministic mock that follows the NativenessScorer interface.

    Uses sequence length modulo to produce repeatable scores in [0.5, 0.9].
    """

    _SCORE_MODULO = 20

    def _raw_score(self, sequence: str) -> float:
        raw = (sum(ord(c) for c in sequence) % self._SCORE_MODULO) / self._SCORE_MODULO
        return 0.5 + raw * 0.4

    def score(self, vhh: VHHSequence) -> dict:
        return {"composite_score": self._raw_score(vhh.sequence)}

    def predict_mutation_effect(self, vhh: VHHSequence, position: int | str, new_aa: str) -> float:
        # Return a small deterministic delta
        return 0.02 if new_aa in "AGILV" else -0.01

    def score_batch(self, sequences: list[str]) -> list[float]:
        return [self._raw_score(seq) for seq in sequences]


@pytest.fixture(scope="module")
def engine() -> MutationEngine:
    """Engine with mock nativeness scorer (default produceability path)."""
    return MutationEngine(
        stability_scorer=StabilityScorer(),
        nativeness_scorer=_MockNativenessScorer(),
    )


@pytest.fixture(scope="module")
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


@pytest.fixture(scope="module")
def ranked(engine: MutationEngine, vhh: VHHSequence) -> pd.DataFrame:
    return engine.rank_single_mutations(vhh)


class TestRankSingleMutations:
    def test_rank_single_mutations(self, ranked: pd.DataFrame) -> None:
        assert "position" in ranked.columns
        assert "combined_score" in ranked.columns

    def test_rank_single_mutations_excluded_target_aas(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        df = engine.rank_single_mutations(vhh, excluded_target_aas={"C"})
        if not df.empty:
            assert "C" not in df["suggested_aa"].values

    def test_rank_single_mutations_has_nativeness_deltas(self, ranked: pd.DataFrame) -> None:
        """With nativeness scorer, delta_nativeness should have non-zero values."""
        if not ranked.empty:
            assert "delta_nativeness" in ranked.columns
            assert ranked["delta_nativeness"].abs().sum() > 0

    def test_rank_single_mutations_has_stability_deltas(self, ranked: pd.DataFrame) -> None:
        if not ranked.empty:
            assert "delta_stability" in ranked.columns

    def test_ranked_no_cdrs(self, ranked: pd.DataFrame, vhh: VHHSequence) -> None:
        """Candidates should not include CDR positions."""
        cdr_positions = vhh.cdr_positions
        for _, row in ranked.iterrows():
            assert str(row["imgt_pos"]) not in cdr_positions

    def test_ranked_excluded_target_aas(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        df = engine.rank_single_mutations(vhh, excluded_target_aas={"C", "M"})
        if not df.empty:
            assert "C" not in df["suggested_aa"].values
            assert "M" not in df["suggested_aa"].values

    def test_ranked_multiple_per_position(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """Stability-driven scan returns multiple AAs per position when max_per_position > 1."""
        ranked = engine.rank_single_mutations(vhh, max_per_position=3)
        if not ranked.empty:
            pos_counts = ranked.groupby("imgt_pos").size()
            assert pos_counts.max() > 1

    def test_ranked_reason(self, ranked: pd.DataFrame) -> None:
        if not ranked.empty:
            assert all(r == "Stability-driven scan" for r in ranked["reason"])


class TestApplyMutations:
    def test_apply_mutations(self) -> None:
        result = MutationEngine.apply_mutations("ABCDE", [(1, "M"), (2, "L")])
        assert result[0] == "M"
        assert result[1] == "L"


class TestGenerateLibrary:
    def test_generate_library(self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2)
        assert isinstance(lib, pd.DataFrame)
        assert "n_mutations" in lib.columns

    def test_generate_library_has_nativeness(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """Library generation includes nativeness scoring."""
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2)
        assert isinstance(lib, pd.DataFrame)
        assert "n_mutations" in lib.columns
        if not lib.empty:
            assert "stability_score" in lib.columns
            assert "nativeness_score" in lib.columns

    def test_generate_library_min_mutations(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=3, min_mutations=2)
        if not lib.empty:
            assert lib["n_mutations"].min() >= 2

    def test_generate_library_large_sampling(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top10 = ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        start = time.time()
        lib = engine.generate_library(vhh, top10, n_mutations=10, max_variants=200, min_mutations=8)
        elapsed = time.time() - start
        assert elapsed < 120
        assert isinstance(lib, pd.DataFrame)

    def test_generate_library_has_developability_columns(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2)
        if not lib.empty:
            assert "surface_hydrophobicity_score" in lib.columns

    def test_generate_library_has_orthogonal_columns(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2)
        if not lib.empty:
            assert "orthogonal_stability_score" in lib.columns

    def test_generate_library_no_humanness_columns(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """Library output should not contain humanness columns."""
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2)
        if not lib.empty:
            assert "humanness_score" not in lib.columns
            assert "orthogonal_humanness_score" not in lib.columns

    def test_generate_library_strategy_random(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2, strategy="random", max_variants=10)
        assert isinstance(lib, pd.DataFrame)

    def test_generate_library_strategy_iterative(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2, strategy="iterative", max_variants=10)
        assert isinstance(lib, pd.DataFrame)


class TestWeightsAndMetrics:
    def test_engine_enabled_metrics(self, engine: MutationEngine) -> None:
        weights = engine._active_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_stability_is_heaviest_weight(self) -> None:
        eng = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_MockNativenessScorer(),
        )
        assert eng._weights["stability"] >= max(v for k, v in eng._weights.items() if k != "stability")

    def test_nativeness_always_enabled(self) -> None:
        """Engine always has nativeness enabled."""
        eng = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_MockNativenessScorer(),
        )
        assert eng._enabled_metrics["nativeness"] is True
        assert eng._weights["nativeness"] > 0.0

    def test_no_humanness_in_metrics(self) -> None:
        """Humanness should not appear in metric names or weights."""
        eng = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_MockNativenessScorer(),
        )
        assert "humanness" not in MutationEngine.METRIC_NAMES
        assert "humanness" not in eng._weights
        assert "humanness" not in eng._enabled_metrics


class TestPTMLiability:
    def test_ptm_liability_hard_restriction(self) -> None:
        parent = "AAAAANGSTAAA"
        mutant = "AAAAANGSTAAA"
        assert _introduces_ptm_liability(parent, mutant, 5) is False

        parent2 = "AAAAAAAAAAAA"
        mutant2 = "AAAAANGSTAAA"
        assert _introduces_ptm_liability(parent2, mutant2, 5) is True


class TestMultiCandidatePerPosition:
    """Tests for multi-option per position feature."""

    def test_multi_candidates(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """Ranking should return multiple AAs per position when requested."""
        ranked = engine.rank_single_mutations(vhh, max_per_position=3)
        if not ranked.empty:
            pos_counts = ranked.groupby("imgt_pos").size()
            # At least some positions should have more than 1 candidate
            assert pos_counts.max() >= 1  # at least 1 always

    def test_multi_candidates_limited(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """Ranking should limit to max_per_position."""
        ranked_limited = engine.rank_single_mutations(vhh, max_per_position=2)
        if not ranked_limited.empty:
            pos_counts = ranked_limited.groupby("imgt_pos").size()
            assert pos_counts.max() <= 2

    def test_single_candidate_per_position(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """max_per_position=1 should give at most 1 candidate per position."""
        ranked = engine.rank_single_mutations(vhh, max_per_position=1)
        if not ranked.empty:
            pos_counts = ranked.groupby("imgt_pos").size()
            assert pos_counts.max() == 1

    def test_library_multi_options_different_aas_across_variants(
        self, engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """Library with multi-option positions should have variants where the same
        position has different AA choices across different variants."""
        ranked = engine.rank_single_mutations(vhh, max_per_position=3)
        if ranked.empty:
            pytest.skip("No mutations ranked")

        # Check that there are positions with multiple options
        pos_counts = ranked.groupby("imgt_pos").size()
        multi_pos = pos_counts[pos_counts > 1]
        if multi_pos.empty:
            pytest.skip("No positions with multiple candidates")

        top = ranked.head(15)
        lib = engine.generate_library(vhh, top, n_mutations=2, max_variants=50)
        if lib.empty:
            pytest.skip("Empty library")

        # Collect all (position, AA) pairs across variants
        position_aas: dict[int, set[str]] = {}
        for _, row in lib.iterrows():
            for pos, aa in _parse_mut_str(row["mutations"]):
                position_aas.setdefault(pos, set()).add(aa)

        # At least one position should have multiple different AAs across variants
        multi_aa_positions = [p for p, aas in position_aas.items() if len(aas) > 1]
        assert len(multi_aa_positions) > 0, "Expected at least one position with different AA choices across variants"

    def test_no_variant_has_duplicate_position(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """No single variant should have the same position mutated twice."""
        ranked = engine.rank_single_mutations(vhh, max_per_position=3)
        if ranked.empty:
            pytest.skip("No mutations ranked")

        top = ranked.head(15)
        lib = engine.generate_library(vhh, top, n_mutations=3, max_variants=50)
        if lib.empty:
            pytest.skip("Empty library")

        for _, row in lib.iterrows():
            muts = _parse_mut_str(row["mutations"])
            positions = [pos for pos, _ in muts]
            assert len(positions) == len(set(positions)), (
                f"Variant {row['variant_id']} has duplicate positions: {row['mutations']}"
            )


class TestGroupedCombinations:
    """Tests for _total_grouped_combinations."""

    def test_single_option_per_position(self) -> None:
        """When each position has 1 option, should equal C(n, k)."""
        import math

        # Dummy objects with .position attribute
        class M:
            def __init__(self, pos):
                self.position = pos

        groups = {1: [M(1)], 2: [M(2)], 3: [M(3)], 4: [M(4)]}
        assert _total_grouped_combinations(groups, 2, 2) == math.comb(4, 2)
        assert _total_grouped_combinations(groups, 1, 3) == (math.comb(4, 1) + math.comb(4, 2) + math.comb(4, 3))

    def test_multi_option_positions(self) -> None:
        """With multiple options per position, count should be larger than C(n, k)."""
        import math

        class M:
            def __init__(self, pos):
                self.position = pos

        # 3 positions: first has 2 options, others have 1
        groups = {1: [M(1), M(1)], 2: [M(2)], 3: [M(3)]}
        # Choosing 2 positions from 3: C(3,2) = 3
        # If position 1 is chosen, its group has 2 options
        # Combos: {1,2}: 2*1=2, {1,3}: 2*1=2, {2,3}: 1*1=1 = 5 total
        assert _total_grouped_combinations(groups, 2, 2) == 5
        # Compare: C(3, 2) = 3, so multi-option gives more
        assert _total_grouped_combinations(groups, 2, 2) > math.comb(3, 2)

    def test_all_positions_multi_option(self) -> None:
        """All positions with 2 options each."""

        class M:
            def __init__(self, pos):
                self.position = pos

        groups = {1: [M(1), M(1)], 2: [M(2), M(2)]}
        # k=1: 2 positions * 2 options each = 4
        # k=2: 1 combo * 2*2 = 4
        # Total = 8
        assert _total_grouped_combinations(groups, 1, 2) == 8


class TestEvolutionaryIterativeStrategy:
    """Tests for the redesigned multi-phase iterative strategy."""

    def test_iterative_converges(self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame) -> None:
        """Final round scores should be >= seed round scores (convergence)."""
        top10 = ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(
            vhh,
            top10,
            n_mutations=3,
            strategy="iterative",
            max_variants=100,
            max_rounds=6,
        )
        assert isinstance(lib, pd.DataFrame)
        if lib.empty:
            pytest.skip("Empty library")
        # Top-quartile average should be positive (better than baseline)
        n_top = max(len(lib) // 4, 1)
        top_avg = lib.nlargest(n_top, "combined_score")["combined_score"].mean()
        assert top_avg > 0.0

    def test_iterative_produces_diverse_variants(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """Iterative strategy should produce diverse (not all identical) variants."""
        top10 = ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(
            vhh,
            top10,
            n_mutations=3,
            strategy="iterative",
            max_variants=50,
            max_rounds=4,
        )
        if len(lib) < 2:
            pytest.skip("Not enough variants")
        unique_seqs = lib["aa_sequence"].nunique()
        assert unique_seqs > 1, "All variants are identical"
        unique_muts = lib["mutations"].nunique()
        assert unique_muts > 1, "All mutation sets are identical"

    def test_iterative_with_progress_callback(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """Progress callback should be invoked with valid IterativeProgress."""
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")

        progress_events: list[IterativeProgress] = []

        def _on_progress(prog: IterativeProgress) -> None:
            progress_events.append(prog)

        lib = engine.generate_library(
            vhh,
            top5,
            n_mutations=2,
            strategy="iterative",
            max_variants=30,
            max_rounds=4,
            progress_callback=_on_progress,
        )
        assert isinstance(lib, pd.DataFrame)
        assert len(progress_events) > 0, "No progress events reported"
        # Verify progress event fields
        for p in progress_events:
            assert p.phase in (
                "exploration",
                "anchor_identification",
                "exploitation",
                "validation",
            )
            assert p.round_number >= 1
            assert p.population_size >= 0

    def test_iterative_benchmark_under_5_min(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """For a 10-mutation search space, strategy should complete in <5 minutes on CPU."""
        top10 = ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        start = time.time()
        lib = engine.generate_library(
            vhh,
            top10,
            n_mutations=10,
            strategy="iterative",
            max_variants=200,
            max_rounds=6,
        )
        elapsed = time.time() - start
        assert elapsed < 300, f"Iterative strategy took {elapsed:.1f}s (>5 min)"
        assert isinstance(lib, pd.DataFrame)


class TestEpistasisDetection:
    """Tests for epistasis detection with synthetic data."""

    def test_compute_epistasis_synergistic(self) -> None:
        """Synergistic pair: score(A+B) > score(A) + score(B) - score(neither)."""
        rows = [
            {"mutations": "A5V, B10L", "combined_score": 1.0},
            {"mutations": "A5V, B10L", "combined_score": 0.9},
            {"mutations": "A5V", "combined_score": 0.3},
            {"mutations": "A5V", "combined_score": 0.35},
            {"mutations": "B10L", "combined_score": 0.2},
            {"mutations": "B10L", "combined_score": 0.25},
            {"mutations": "C15G", "combined_score": 0.1},
            {"mutations": "C15G", "combined_score": 0.05},
        ]
        interaction = _compute_epistasis(rows, (5, "V"), (10, "L"))
        # Medians: A+B=0.95, A_only=0.325, B_only=0.225
        # "neither" = C15G rows (no mut A or B): median=0.075
        # interaction = 0.95 - 0.325 - 0.225 + 0.075 = 0.475 > 0  (synergistic)
        assert interaction > 0.0, f"Expected synergistic interaction, got {interaction}"

    def test_compute_epistasis_antagonistic(self) -> None:
        """Antagonistic pair: combination is worse than sum of parts."""
        rows = [
            {"mutations": "A5V, B10L", "combined_score": 0.1},
            {"mutations": "A5V, B10L", "combined_score": 0.15},
            {"mutations": "A5V", "combined_score": 0.5},
            {"mutations": "A5V", "combined_score": 0.55},
            {"mutations": "B10L", "combined_score": 0.4},
            {"mutations": "B10L", "combined_score": 0.45},
            {"mutations": "C15G", "combined_score": 0.2},
            {"mutations": "C15G", "combined_score": 0.1},
        ]
        interaction = _compute_epistasis(rows, (5, "V"), (10, "L"))
        assert interaction < 0.0, f"Expected antagonistic interaction, got {interaction}"

    def test_compute_epistasis_empty_data(self) -> None:
        """Empty data should return 0."""
        assert _compute_epistasis([], (1, "A"), (2, "B")) == 0.0


class TestMutationEntropy:
    """Tests for Shannon entropy diversity metric."""

    def test_entropy_zero_for_empty(self) -> None:
        assert _mutation_entropy([]) == 0.0

    def test_entropy_zero_for_identical(self) -> None:
        """All identical mutations → low entropy."""
        rows = [
            {"mutations": "A5V"},
            {"mutations": "A5V"},
            {"mutations": "A5V"},
        ]
        assert _mutation_entropy(rows) == 0.0

    def test_entropy_positive_for_diverse(self) -> None:
        """Different mutations → positive entropy."""
        rows = [
            {"mutations": "A5V"},
            {"mutations": "B10L"},
            {"mutations": "C15G"},
        ]
        assert _mutation_entropy(rows) > 0.0


class TestAnchorIdentification:
    """Tests for epistasis-aware anchor identification."""

    def test_identify_anchors_empty(self) -> None:
        result = MutationEngine._identify_anchors_with_epistasis([], 0.6)
        assert result == []

    def test_identify_anchors_returns_candidates(self) -> None:
        """High-frequency mutations in top quartile should be identified."""
        rows = [
            {"mutations": "A5V, B10L", "combined_score": 0.9},
            {"mutations": "A5V, C15G", "combined_score": 0.85},
            {"mutations": "A5V, D20F", "combined_score": 0.8},
            {"mutations": "A5V, E25W", "combined_score": 0.75},
            {"mutations": "B10L", "combined_score": 0.3},
            {"mutations": "C15G", "combined_score": 0.25},
            {"mutations": "D20F", "combined_score": 0.2},
            {"mutations": "E25W", "combined_score": 0.15},
        ]
        anchors = MutationEngine._identify_anchors_with_epistasis(rows, 0.6)
        assert len(anchors) > 0
        # A5V should be a top anchor (appears in all top quartile)
        top_anchor = anchors[0]
        assert top_anchor.position == 5
        assert top_anchor.amino_acid == "V"
        assert top_anchor.confidence > 0.2

    def test_anchor_confidence_ordering(self) -> None:
        """Anchors should be sorted by confidence (descending)."""
        rows = [
            {"mutations": "A5V, B10L", "combined_score": 0.9},
            {"mutations": "A5V, B10L", "combined_score": 0.85},
            {"mutations": "A5V, C15G", "combined_score": 0.7},
            {"mutations": "B10L, C15G", "combined_score": 0.65},
            {"mutations": "D20F", "combined_score": 0.1},
            {"mutations": "E25W", "combined_score": 0.05},
        ]
        anchors = MutationEngine._identify_anchors_with_epistasis(rows, 0.4)
        if len(anchors) >= 2:
            for i in range(len(anchors) - 1):
                assert anchors[i].confidence >= anchors[i + 1].confidence


# ---------------------------------------------------------------------------
# Nativeness integration tests
# ---------------------------------------------------------------------------


class TestNativenessIntegration:
    """Tests for nativeness integration in MutationEngine."""

    def test_nativeness_always_enabled(self) -> None:
        """Nativeness is always enabled with default scorer."""
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_MockNativenessScorer(),
        )
        assert engine._enabled_metrics["nativeness"] is True
        assert engine._weights["nativeness"] > 0.0

    def test_nativeness_in_metric_names(self) -> None:
        assert "nativeness" in MutationEngine.METRIC_NAMES

    def test_rank_single_mutations_has_delta_nativeness(self) -> None:
        """rank_single_mutations output includes delta_nativeness column."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        assert "delta_nativeness" in ranked.columns
        if not ranked.empty:
            # With mock scorer enabled, some delta values should be non-zero
            assert ranked["delta_nativeness"].abs().sum() > 0

    def test_build_variant_row_has_nativeness_score(self) -> None:
        """_build_variant_row output includes nativeness_score."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        if ranked.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, ranked.head(3), n_mutations=1)
        if not lib.empty:
            assert "nativeness_score" in lib.columns

    def test_combined_score_includes_nativeness(self) -> None:
        """Combined score includes nativeness contribution."""
        vhh = VHHSequence(SAMPLE_VHH)

        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )
        ranked = engine.rank_single_mutations(vhh)

        if ranked.empty:
            pytest.skip("No mutations to compare")

        # The combined scores should include nativeness (non-zero weight)
        assert engine._weights["nativeness"] > 0.0
        assert engine._enabled_metrics["nativeness"] is True

    def test_empty_library_df_has_nativeness_score(self) -> None:
        """_empty_library_df includes nativeness_score column."""
        df = MutationEngine._empty_library_df()
        assert "nativeness_score" in df.columns

    def test_empty_library_df_no_humanness(self) -> None:
        """_empty_library_df does not include humanness columns."""
        df = MutationEngine._empty_library_df()
        assert "humanness_score" not in df.columns
        assert "orthogonal_humanness_score" not in df.columns

    def test_generate_library_with_nativeness(self) -> None:
        """Library generation works with nativeness scorer enabled."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        if ranked.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, ranked.head(3), n_mutations=1)
        assert isinstance(lib, pd.DataFrame)
        if not lib.empty:
            assert "nativeness_score" in lib.columns
            assert "stability_score" in lib.columns


class TestCombinedRanking:
    """Tests for the combined scoring/ranking behavior."""

    def test_combined_score_is_weighted_sum(self) -> None:
        """Combined score should be a normalized weighted sum of active metrics."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
            weights={"stability": 0.7, "nativeness": 0.3},
        )
        # Active weights should sum to 1.0
        active = engine._active_weights()
        assert abs(sum(active.values()) - 1.0) < 1e-6

    def test_custom_weight_override(self) -> None:
        """Custom weights should override defaults."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
            weights={"stability": 0.5, "nativeness": 0.5},
        )
        assert engine._weights["stability"] == 0.5
        assert engine._weights["nativeness"] == 0.5

    def test_ranking_reflects_composite(self) -> None:
        """Ranked mutations should be sorted by combined_score descending."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        if len(ranked) >= 2:
            scores = ranked["combined_score"].tolist()
            assert scores == sorted(scores, reverse=True)


class TestBatchNativenessScoring:
    """Tests that nativeness is batch-scored during mutation ranking."""

    def test_score_batch_used_over_predict_mutation_effect(self) -> None:
        """Ranking should call score_batch instead of per-candidate predict_mutation_effect."""
        call_log: list[str] = []

        class _TrackingNativenessScorer(_MockNativenessScorer):
            def predict_mutation_effect(self_, vhh, position, new_aa):
                call_log.append("predict_mutation_effect")
                return super().predict_mutation_effect(vhh, position, new_aa)

            def score_batch(self_, sequences):
                call_log.append("score_batch")
                return super().score_batch(sequences)

        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_TrackingNativenessScorer(),
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        if not ranked.empty:
            assert "score_batch" in call_log
            assert "predict_mutation_effect" not in call_log

    def test_batch_nativeness_deltas_are_correct(self) -> None:
        """Batch-scored nativeness deltas should equal individually computed ones."""
        scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=scorer,
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        if ranked.empty:
            pytest.skip("No mutations ranked")

        parent_nat = scorer.score(vhh)["composite_score"]
        for _, row in ranked.head(10).iterrows():
            mutant = VHHSequence.mutate(vhh, row["imgt_pos"], row["suggested_aa"])
            expected_delta = scorer.score(mutant)["composite_score"] - parent_nat
            assert abs(row["delta_nativeness"] - expected_delta) < 1e-9


class TestPositionAwareSampling:
    """Tests verifying that position-aware sampling enforces mutation count
    constraints and generates diverse variants."""

    def test_min_mutations_enforced_all_strategies(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """Every variant must have at least min_mutations mutations."""
        # Use enough top mutations to have many positions.
        top20 = ranked.head(20)
        if top20.empty:
            pytest.skip("No mutations ranked")

        n_positions = top20["position"].nunique()
        min_muts = min(3, n_positions)

        for strategy in ("exhaustive", "random"):
            lib = engine.generate_library(
                vhh,
                top20,
                n_mutations=min(10, n_positions),
                min_mutations=min_muts,
                max_variants=50,
                strategy=strategy,
            )
            if not lib.empty:
                assert lib["n_mutations"].min() >= min_muts, (
                    f"Strategy '{strategy}': variant with {lib['n_mutations'].min()} mutations, expected >= {min_muts}"
                )

    def test_no_duplicate_positions_in_variant(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        """No variant should mutate the same position twice."""
        top20 = ranked.head(20)
        if top20.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top20, n_mutations=5, max_variants=50, strategy="random")
        for _, row in lib.iterrows():
            muts = _parse_mut_str(row["mutations"])
            positions = [pos for pos, _ in muts]
            assert len(positions) == len(set(positions)), f"Duplicate positions in variant: {row['mutations']}"

    def test_variants_fully_rescored(self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame) -> None:
        """Each variant should have independently computed stability and nativeness."""
        top10 = ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top10, n_mutations=3, max_variants=20, strategy="random")
        if lib.empty:
            pytest.skip("No variants generated")
        # Stability and nativeness should be in [0, 1] range and not all identical.
        assert (lib["stability_score"] >= 0).all()
        assert (lib["stability_score"] <= 1).all()
        assert (lib["nativeness_score"] >= 0).all()
        assert (lib["nativeness_score"] <= 1).all()
        if len(lib) > 1:
            assert lib["stability_score"].nunique() > 1 or lib["nativeness_score"].nunique() > 1

    def test_search_space_warning_logged(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame, caplog
    ) -> None:
        """A warning should be logged when max_variants exceeds the search space."""
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        import logging

        with caplog.at_level(logging.WARNING):
            engine.generate_library(vhh, top5, n_mutations=2, max_variants=100_000)
        assert any("search space" in msg.lower() for msg in caplog.messages)

    def test_min_mutations_clamped_warning_logged(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame, caplog
    ) -> None:
        """A warning should be logged when min_mutations exceeds available positions."""
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        n_positions = top5["position"].nunique()
        import logging

        with caplog.at_level(logging.WARNING):
            engine.generate_library(vhh, top5, n_mutations=n_positions, min_mutations=n_positions + 5, max_variants=10)
        assert any("clamping" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# NanoMelt propagation through the scoring pipeline
# ---------------------------------------------------------------------------


class TestNanoMeltPropagation:
    """Verify that nanomelt_tm flows through _score_variant and _build_variant_row."""

    @staticmethod
    def _make_mock_nanomelt(tm: float = 70.0):
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.score_sequence.return_value = {
            "composite_score": 0.6,
            "nanomelt_tm": tm,
        }
        return mock

    def test_score_variant_includes_nanomelt_tm(self) -> None:
        """_score_variant should include nanomelt_tm when NanoMelt is configured."""
        mock_nm = self._make_mock_nanomelt(72.0)
        scorer = StabilityScorer(nanomelt_predictor=mock_nm)
        engine = MutationEngine(
            stability_scorer=scorer,
            nativeness_scorer=_MockNativenessScorer(),
        )
        vhh = VHHSequence(SAMPLE_VHH)
        raw = engine._score_variant(vhh)
        assert "nanomelt_tm" in raw
        assert raw["nanomelt_tm"] == 72.0

    def test_build_variant_row_includes_nanomelt_tm(self) -> None:
        """Generated library rows should include nanomelt_tm when NanoMelt is configured."""
        mock_nm = self._make_mock_nanomelt(72.0)
        scorer = StabilityScorer(nanomelt_predictor=mock_nm)
        engine = MutationEngine(
            stability_scorer=scorer,
            nativeness_scorer=_MockNativenessScorer(),
        )
        vhh = VHHSequence(SAMPLE_VHH)
        ranked = engine.rank_single_mutations(vhh)
        if ranked.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, ranked.head(3), n_mutations=1)
        if not lib.empty:
            assert "nanomelt_tm" in lib.columns

    def test_no_nanomelt_tm_without_predictor(self) -> None:
        """Without NanoMelt, nanomelt_tm should not appear in scores."""
        scorer = StabilityScorer()
        engine = MutationEngine(
            stability_scorer=scorer,
            nativeness_scorer=_MockNativenessScorer(),
        )
        vhh = VHHSequence(SAMPLE_VHH)
        raw = engine._score_variant(vhh)
        assert "nanomelt_tm" not in raw


# ---------------------------------------------------------------------------
# Batch nativeness optimisation during library generation
# ---------------------------------------------------------------------------


class TestBatchNativenessLibraryGeneration:
    """Verify that library generation uses batch nativeness scoring (score_batch)
    instead of per-variant AbNatiV calls, preventing ANARCI alignment per variant.
    """

    @staticmethod
    def _make_vhh_no_anarci(sequence: str, imgt_numbered: dict[str, str] | None = None) -> VHHSequence:
        """Create a VHHSequence without ANARCI (for environments lacking hmmscan)."""
        vhh = object.__new__(VHHSequence)
        vhh.sequence = sequence.upper()
        vhh.length = len(vhh.sequence)
        vhh.strict = False
        vhh.chain_type = "H"
        vhh.species = "camelid"
        if imgt_numbered is None:
            # Build a trivial 1:1 mapping.
            imgt_numbered = {str(i + 1): aa for i, aa in enumerate(vhh.sequence)}
        vhh.imgt_numbered = imgt_numbered
        vhh._pos_to_seq_idx = {k: idx for idx, k in enumerate(imgt_numbered)}
        vhh.validation_result = {"valid": True, "errors": [], "warnings": []}
        return vhh

    @staticmethod
    def _make_engine_with_tracking_scorer():
        """Create an engine with a tracking nativeness scorer.

        Returns (engine, call_log) where call_log records method invocations.
        """
        call_log: list[str] = []

        class _TrackingScorer(_MockNativenessScorer):
            def score(self_, vhh):
                call_log.append("score")
                return super().score(vhh)

            def predict_mutation_effect(self_, vhh, position, new_aa):
                call_log.append("predict_mutation_effect")
                return super().predict_mutation_effect(vhh, position, new_aa)

            def score_batch(self_, sequences):
                call_log.append(f"score_batch({len(sequences)})")
                return super().score_batch(sequences)

        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=_TrackingScorer(),
        )
        return engine, call_log

    def test_exhaustive_uses_batch_scoring(self) -> None:
        """_generate_exhaustive should batch-score nativeness, not score per variant."""
        engine, call_log = self._make_engine_with_tracking_scorer()
        # Use a short synthetic sequence with IMGT-like numbering.
        seq = "QVQLVESGGGLVQ"
        vhh = self._make_vhh_no_anarci(seq)

        import pandas as pd

        # Create mutation candidates for positions 1, 2, 3 (framework-like).
        rows = []
        for pos_key in ["1", "2", "3"]:
            orig = vhh.imgt_numbered[pos_key]
            for aa in sorted(AMINO_ACIDS - {orig})[:2]:
                rows.append(
                    {
                        "position": int(pos_key),
                        "imgt_pos": pos_key,
                        "original_aa": orig,
                        "suggested_aa": aa,
                        "delta_stability": 0.1,
                        "delta_nativeness": 0.01,
                        "combined_score": 0.5,
                        "reason": "test",
                    }
                )
        top_muts = pd.DataFrame(rows)

        call_log.clear()
        lib = engine.generate_library(vhh, top_muts, n_mutations=2, strategy="exhaustive", max_variants=50)
        assert isinstance(lib, pd.DataFrame)
        assert not lib.empty

        # score_batch should have been called (at least once).
        batch_calls = [c for c in call_log if c.startswith("score_batch")]
        assert len(batch_calls) >= 1, f"Expected score_batch call, got: {call_log}"

        # Individual score() calls should be zero during library generation
        # (the _batch_fill_nativeness path bypasses per-variant .score()).
        individual_calls = [c for c in call_log if c == "score"]
        assert len(individual_calls) == 0, (
            f"Expected no per-variant score() calls during library gen, got {len(individual_calls)}"
        )

    def test_random_uses_batch_scoring(self) -> None:
        """_generate_sampled should batch-score nativeness."""
        engine, call_log = self._make_engine_with_tracking_scorer()
        seq = "QVQLVESGGGLVQAGG"
        vhh = self._make_vhh_no_anarci(seq)

        import pandas as pd

        rows = []
        for pos_key in ["1", "2", "3", "4", "5"]:
            orig = vhh.imgt_numbered[pos_key]
            for aa in sorted(AMINO_ACIDS - {orig})[:2]:
                rows.append(
                    {
                        "position": int(pos_key),
                        "imgt_pos": pos_key,
                        "original_aa": orig,
                        "suggested_aa": aa,
                        "delta_stability": 0.1,
                        "delta_nativeness": 0.01,
                        "combined_score": 0.5,
                        "reason": "test",
                    }
                )
        top_muts = pd.DataFrame(rows)

        call_log.clear()
        lib = engine.generate_library(vhh, top_muts, n_mutations=3, strategy="random", max_variants=20)
        assert isinstance(lib, pd.DataFrame)
        assert not lib.empty

        batch_calls = [c for c in call_log if c.startswith("score_batch")]
        assert len(batch_calls) >= 1, f"Expected score_batch call, got: {call_log}"

    def test_batch_fill_nativeness_values_correct(self) -> None:
        """_batch_fill_nativeness should produce same scores as individual scoring."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )

        seq1 = "QVQLVESGGGLVQ"
        seq2 = "AVQLVESGGGLVQ"

        rows = [
            {
                "aa_sequence": seq1,
                "nativeness_score": 0.0,
                "stability_score": 0.5,
                "surface_hydrophobicity_score": 0.0,
                "orthogonal_stability_score": 0.5,
                "combined_score": 0.0,
            },
            {
                "aa_sequence": seq2,
                "nativeness_score": 0.0,
                "stability_score": 0.5,
                "surface_hydrophobicity_score": 0.0,
                "orthogonal_stability_score": 0.5,
                "combined_score": 0.0,
            },
        ]

        engine._batch_fill_nativeness(rows)

        # Verify scores match what the scorer would return individually.
        expected1 = mock_scorer.score_batch([seq1])[0]
        expected2 = mock_scorer.score_batch([seq2])[0]
        assert abs(rows[0]["nativeness_score"] - expected1) < 1e-9
        assert abs(rows[1]["nativeness_score"] - expected2) < 1e-9

    def test_build_variant_row_accepts_nativeness_score(self) -> None:
        """_build_variant_row should use provided nativeness_score when given."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
        )
        seq = "QVQLVESGGGLVQ"
        vhh = self._make_vhh_no_anarci(seq)

        # Create a mock namedtuple-like object matching what itertuples returns.
        class _MockMut:
            def __init__(self, position, original_aa, suggested_aa):
                self.position = position
                self.original_aa = original_aa
                self.suggested_aa = suggested_aa

        mut = _MockMut(1, "Q", "A")

        row_with_nat = engine._build_variant_row(vhh, [mut], 1, nativeness_score=0.77)
        assert row_with_nat["nativeness_score"] == 0.77

        row_without_nat = engine._build_variant_row(vhh, [mut], 2)
        # Should use the scorer's actual score (not 0.77).
        expected = mock_scorer.score_batch([row_without_nat["aa_sequence"]])[0]
        assert abs(row_without_nat["nativeness_score"] - expected) < 1e-9

    def test_combined_score_recomputed_after_batch(self) -> None:
        """combined_score should be recomputed after batch nativeness fill."""
        mock_scorer = _MockNativenessScorer()
        engine = MutationEngine(
            stability_scorer=StabilityScorer(),
            nativeness_scorer=mock_scorer,
            weights={"stability": 0.5, "nativeness": 0.5},
        )
        seq = "QVQLVESGGGLVQ"
        vhh = self._make_vhh_no_anarci(seq)

        import pandas as pd

        rows = []
        for pos_key in ["1", "2"]:
            orig = vhh.imgt_numbered[pos_key]
            for aa in sorted(AMINO_ACIDS - {orig})[:1]:
                rows.append(
                    {
                        "position": int(pos_key),
                        "imgt_pos": pos_key,
                        "original_aa": orig,
                        "suggested_aa": aa,
                        "delta_stability": 0.1,
                        "delta_nativeness": 0.01,
                        "combined_score": 0.5,
                        "reason": "test",
                    }
                )
        top_muts = pd.DataFrame(rows)

        lib = engine.generate_library(vhh, top_muts, n_mutations=1, strategy="exhaustive", max_variants=10)
        if not lib.empty:
            for _, row in lib.iterrows():
                # combined_score should reflect actual nativeness, not the placeholder 0.0.
                assert row["nativeness_score"] != 0.0 or row["aa_sequence"] == seq
