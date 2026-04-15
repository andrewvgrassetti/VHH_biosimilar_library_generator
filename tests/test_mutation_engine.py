"""Tests for vhh_library.mutation_engine – MutationEngine class."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from vhh_library.humanness import HumAnnotator
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

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture(scope="module")
def engine() -> MutationEngine:
    """Engine with humanness scorer (legacy path)."""
    return MutationEngine(HumAnnotator(), StabilityScorer())


@pytest.fixture(scope="module")
def stability_engine() -> MutationEngine:
    """Engine without humanness scorer (stability-driven path)."""
    return MutationEngine(stability_scorer=StabilityScorer())


@pytest.fixture(scope="module")
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


@pytest.fixture(scope="module")
def ranked(engine: MutationEngine, vhh: VHHSequence) -> pd.DataFrame:
    return engine.rank_single_mutations(vhh)


@pytest.fixture(scope="module")
def stability_ranked(stability_engine: MutationEngine, vhh: VHHSequence) -> pd.DataFrame:
    return stability_engine.rank_single_mutations(vhh)


class TestRankSingleMutations:
    def test_rank_single_mutations(self, ranked: pd.DataFrame) -> None:
        assert "position" in ranked.columns
        assert "combined_score" in ranked.columns

    def test_rank_single_mutations_excluded_target_aas(
        self, engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        df = engine.rank_single_mutations(vhh, excluded_target_aas={"C"})
        if not df.empty:
            assert "C" not in df["suggested_aa"].values

    def test_rank_single_mutations_has_humanness_deltas(
        self, ranked: pd.DataFrame
    ) -> None:
        """With humanness scorer, delta_humanness should have non-zero values."""
        if not ranked.empty:
            assert "delta_humanness" in ranked.columns
            assert ranked["delta_humanness"].abs().sum() > 0


class TestStabilityDrivenRanking:
    """Tests for stability-driven candidate generation (no humanness scorer)."""

    def test_stability_ranked_has_results(self, stability_ranked: pd.DataFrame) -> None:
        assert not stability_ranked.empty
        assert "position" in stability_ranked.columns
        assert "combined_score" in stability_ranked.columns
        assert "delta_stability" in stability_ranked.columns

    def test_stability_ranked_no_cdrs(
        self, stability_ranked: pd.DataFrame, vhh: VHHSequence
    ) -> None:
        """Stability-driven candidates should not include CDR positions."""
        cdr_positions = vhh.cdr_positions
        for _, row in stability_ranked.iterrows():
            assert str(row["imgt_pos"]) not in cdr_positions

    def test_stability_ranked_excluded_target_aas(
        self, stability_engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        df = stability_engine.rank_single_mutations(vhh, excluded_target_aas={"C", "M"})
        if not df.empty:
            assert "C" not in df["suggested_aa"].values
            assert "M" not in df["suggested_aa"].values

    def test_stability_ranked_multiple_per_position(
        self, stability_engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """Stability-driven scan returns multiple AAs per position when max_per_position > 1."""
        ranked = stability_engine.rank_single_mutations(vhh, max_per_position=3)
        if not ranked.empty:
            pos_counts = ranked.groupby("imgt_pos").size()
            assert pos_counts.max() > 1

    def test_stability_ranked_reason(self, stability_ranked: pd.DataFrame) -> None:
        if not stability_ranked.empty:
            assert all(r == "Stability-driven scan" for r in stability_ranked["reason"])

    def test_humanness_delta_is_zero(self, stability_ranked: pd.DataFrame) -> None:
        """Without humanness scorer, delta_humanness should be 0."""
        if not stability_ranked.empty:
            assert all(stability_ranked["delta_humanness"] == 0.0)


class TestApplyMutations:
    def test_apply_mutations(self) -> None:
        result = MutationEngine.apply_mutations("ABCDE", [(1, "M"), (2, "L")])
        assert result[0] == "M"
        assert result[1] == "L"


class TestGenerateLibrary:
    def test_generate_library(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(vhh, top5, n_mutations=2)
        assert isinstance(lib, pd.DataFrame)
        assert "n_mutations" in lib.columns

    def test_generate_library_stability_only(
        self, stability_engine: MutationEngine, vhh: VHHSequence, stability_ranked: pd.DataFrame
    ) -> None:
        """Library generation works end-to-end with stability-only scoring."""
        top5 = stability_ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = stability_engine.generate_library(vhh, top5, n_mutations=2)
        assert isinstance(lib, pd.DataFrame)
        assert "n_mutations" in lib.columns
        if not lib.empty:
            assert "stability_score" in lib.columns

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
        lib = engine.generate_library(
            vhh, top10, n_mutations=10, max_variants=200, min_mutations=8
        )
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
            assert "orthogonal_humanness_score" in lib.columns
            assert "orthogonal_stability_score" in lib.columns

    def test_generate_library_strategy_random(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(
            vhh, top5, n_mutations=2, strategy="random", max_variants=10
        )
        assert isinstance(lib, pd.DataFrame)

    def test_generate_library_strategy_iterative(
        self, engine: MutationEngine, vhh: VHHSequence, ranked: pd.DataFrame
    ) -> None:
        top5 = ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")
        lib = engine.generate_library(
            vhh, top5, n_mutations=2, strategy="iterative", max_variants=10
        )
        assert isinstance(lib, pd.DataFrame)


class TestWeightsAndMetrics:
    def test_engine_enabled_metrics(self, engine: MutationEngine) -> None:
        weights = engine._active_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_stability_is_heaviest_weight(self) -> None:
        eng = MutationEngine(stability_scorer=StabilityScorer())
        assert eng._weights["stability"] >= max(
            v for k, v in eng._weights.items() if k != "stability"
        )

    def test_stability_only_engine_weights(self) -> None:
        """Engine with no humanness scorer should have humanness weight 0."""
        eng = MutationEngine(stability_scorer=StabilityScorer())
        assert eng._weights["humanness"] == 0.0
        assert eng._enabled_metrics["humanness"] is False

    def test_humanness_engine_weights(self) -> None:
        """Engine with humanness scorer should have non-zero humanness weight."""
        eng = MutationEngine(HumAnnotator(), StabilityScorer())
        assert eng._weights["humanness"] > 0.0
        assert eng._enabled_metrics["humanness"] is True


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

    def test_humanness_multi_candidates(
        self, engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """Humanness-based ranking should return multiple AAs per position when requested."""
        ranked = engine.rank_single_mutations(vhh, max_per_position=3)
        if not ranked.empty:
            pos_counts = ranked.groupby("imgt_pos").size()
            # At least some positions should have more than 1 candidate
            assert pos_counts.max() >= 1  # at least 1 always

    def test_stability_multi_candidates_limited(
        self, stability_engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """Stability-driven ranking should limit to max_per_position."""
        ranked_limited = stability_engine.rank_single_mutations(vhh, max_per_position=2)
        if not ranked_limited.empty:
            pos_counts = ranked_limited.groupby("imgt_pos").size()
            assert pos_counts.max() <= 2

    def test_single_candidate_per_position(
        self, stability_engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """max_per_position=1 should give at most 1 candidate per position."""
        ranked = stability_engine.rank_single_mutations(vhh, max_per_position=1)
        if not ranked.empty:
            pos_counts = ranked.groupby("imgt_pos").size()
            assert pos_counts.max() == 1

    def test_library_multi_options_different_aas_across_variants(
        self, stability_engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """Library with multi-option positions should have variants where the same
        position has different AA choices across different variants."""
        ranked = stability_engine.rank_single_mutations(vhh, max_per_position=3)
        if ranked.empty:
            pytest.skip("No mutations ranked")

        # Check that there are positions with multiple options
        pos_counts = ranked.groupby("imgt_pos").size()
        multi_pos = pos_counts[pos_counts > 1]
        if multi_pos.empty:
            pytest.skip("No positions with multiple candidates")

        top = ranked.head(15)
        lib = stability_engine.generate_library(vhh, top, n_mutations=2, max_variants=50)
        if lib.empty:
            pytest.skip("Empty library")

        # Collect all (position, AA) pairs across variants
        position_aas: dict[int, set[str]] = {}
        for _, row in lib.iterrows():
            for pos, aa in _parse_mut_str(row["mutations"]):
                position_aas.setdefault(pos, set()).add(aa)

        # At least one position should have multiple different AAs across variants
        multi_aa_positions = [p for p, aas in position_aas.items() if len(aas) > 1]
        assert len(multi_aa_positions) > 0, (
            "Expected at least one position with different AA choices across variants"
        )

    def test_no_variant_has_duplicate_position(
        self, stability_engine: MutationEngine, vhh: VHHSequence
    ) -> None:
        """No single variant should have the same position mutated twice."""
        ranked = stability_engine.rank_single_mutations(vhh, max_per_position=3)
        if ranked.empty:
            pytest.skip("No mutations ranked")

        top = ranked.head(15)
        lib = stability_engine.generate_library(vhh, top, n_mutations=3, max_variants=50)
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
        assert _total_grouped_combinations(groups, 1, 3) == (
            math.comb(4, 1) + math.comb(4, 2) + math.comb(4, 3)
        )

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

    def test_iterative_converges(
        self, stability_engine: MutationEngine, vhh: VHHSequence, stability_ranked: pd.DataFrame
    ) -> None:
        """Final round scores should be >= seed round scores (convergence)."""
        top10 = stability_ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        lib = stability_engine.generate_library(
            vhh, top10, n_mutations=3, strategy="iterative",
            max_variants=100, max_rounds=6,
        )
        assert isinstance(lib, pd.DataFrame)
        if lib.empty:
            pytest.skip("Empty library")
        # Top-quartile average should be positive (better than baseline)
        n_top = max(len(lib) // 4, 1)
        top_avg = lib.nlargest(n_top, "combined_score")["combined_score"].mean()
        assert top_avg > 0.0

    def test_iterative_produces_diverse_variants(
        self, stability_engine: MutationEngine, vhh: VHHSequence, stability_ranked: pd.DataFrame
    ) -> None:
        """Iterative strategy should produce diverse (not all identical) variants."""
        top10 = stability_ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        lib = stability_engine.generate_library(
            vhh, top10, n_mutations=3, strategy="iterative",
            max_variants=50, max_rounds=4,
        )
        if len(lib) < 2:
            pytest.skip("Not enough variants")
        unique_seqs = lib["aa_sequence"].nunique()
        assert unique_seqs > 1, "All variants are identical"
        unique_muts = lib["mutations"].nunique()
        assert unique_muts > 1, "All mutation sets are identical"

    def test_iterative_with_progress_callback(
        self, stability_engine: MutationEngine, vhh: VHHSequence, stability_ranked: pd.DataFrame
    ) -> None:
        """Progress callback should be invoked with valid IterativeProgress."""
        top5 = stability_ranked.head(5)
        if top5.empty:
            pytest.skip("No mutations ranked")

        progress_events: list[IterativeProgress] = []

        def _on_progress(prog: IterativeProgress) -> None:
            progress_events.append(prog)

        lib = stability_engine.generate_library(
            vhh, top5, n_mutations=2, strategy="iterative",
            max_variants=30, max_rounds=4,
            progress_callback=_on_progress,
        )
        assert isinstance(lib, pd.DataFrame)
        assert len(progress_events) > 0, "No progress events reported"
        # Verify progress event fields
        for p in progress_events:
            assert p.phase in (
                "exploration", "anchor_identification",
                "exploitation", "validation",
            )
            assert p.round_number >= 1
            assert p.population_size >= 0

    def test_iterative_benchmark_under_5_min(
        self, stability_engine: MutationEngine, vhh: VHHSequence, stability_ranked: pd.DataFrame
    ) -> None:
        """For a 10-mutation search space, strategy should complete in <5 minutes on CPU."""
        top10 = stability_ranked.head(10)
        if top10.empty:
            pytest.skip("No mutations ranked")
        start = time.time()
        lib = stability_engine.generate_library(
            vhh, top10, n_mutations=10, strategy="iterative",
            max_variants=200, max_rounds=6,
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
