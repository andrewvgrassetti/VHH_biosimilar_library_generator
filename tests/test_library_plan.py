"""Tests for vhh_library.library_plan — multi-objective ranking and quota-based planning."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from vhh_library.library_plan import (
    ALL_BUCKETS,
    BUCKET_EXPLOIT,
    BUCKET_EXPLORATION,
    BUCKET_INTERACTION,
    LibraryPlan,
    annotate_pareto_metadata,
    assign_selection_buckets,
    compute_dominates_count,
    compute_pareto_ranks,
    plan_library,
)

# ---------------------------------------------------------------------------
# Helpers: build synthetic library DataFrames
# ---------------------------------------------------------------------------


def _make_library_df(n: int = 20, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic library DataFrame with deterministic scores."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        stab = rng.uniform(0.3, 0.95)
        nat = rng.uniform(0.3, 0.95)
        combined = 0.7 * stab + 0.3 * nat
        rows.append(
            {
                "variant_id": f"V{i + 1:06d}",
                "mutations": f"A{5 + i}V" if i % 3 != 0 else f"A{5 + i}V, B{10 + i}L",
                "n_mutations": 1 if i % 3 != 0 else 2,
                "stability_score": round(stab, 4),
                "nativeness_score": round(nat, 4),
                "combined_score": round(combined, 4),
                "aa_sequence": "A" * 120,
            }
        )
    return pd.DataFrame(rows)


def _make_dominated_df() -> pd.DataFrame:
    """Build a DataFrame where Pareto structure is manually verifiable.

    Variant 0: (0.9, 0.9)  — dominates most
    Variant 1: (0.8, 0.95) — Pareto rank 1 (not dominated by 0)
    Variant 2: (0.5, 0.5)  — dominated by both 0 and 1
    Variant 3: (0.3, 0.3)  — dominated by all above
    """
    rows = [
        {
            "variant_id": "V000001",
            "mutations": "A5V",
            "n_mutations": 1,
            "stability_score": 0.9,
            "nativeness_score": 0.9,
            "combined_score": 0.9,
            "aa_sequence": "A" * 120,
        },
        {
            "variant_id": "V000002",
            "mutations": "B10L",
            "n_mutations": 1,
            "stability_score": 0.8,
            "nativeness_score": 0.95,
            "combined_score": 0.845,
            "aa_sequence": "A" * 120,
        },
        {
            "variant_id": "V000003",
            "mutations": "A5V, B10L",
            "n_mutations": 2,
            "stability_score": 0.5,
            "nativeness_score": 0.5,
            "combined_score": 0.5,
            "aa_sequence": "A" * 120,
        },
        {
            "variant_id": "V000004",
            "mutations": "C15G",
            "n_mutations": 1,
            "stability_score": 0.3,
            "nativeness_score": 0.3,
            "combined_score": 0.3,
            "aa_sequence": "A" * 120,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LibraryPlan dataclass tests
# ---------------------------------------------------------------------------


class TestLibraryPlan:
    def test_default_fractions_sum_to_one(self) -> None:
        plan = LibraryPlan()
        total = plan.exploit_fraction + plan.interaction_fraction + plan.exploration_fraction
        assert abs(total - 1.0) < 1e-6

    def test_quotas_sum_to_budget(self) -> None:
        plan = LibraryPlan(budget=100)
        assert plan.exploit_quota + plan.interaction_quota + plan.exploration_quota == plan.budget

    def test_quotas_sum_to_budget_odd(self) -> None:
        plan = LibraryPlan(budget=97, exploit_fraction=0.5, interaction_fraction=0.25, exploration_fraction=0.25)
        assert plan.exploit_quota + plan.interaction_quota + plan.exploration_quota == plan.budget

    def test_quotas_sum_to_budget_prime(self) -> None:
        plan = LibraryPlan(budget=13)
        assert plan.exploit_quota + plan.interaction_quota + plan.exploration_quota == plan.budget

    def test_custom_fractions(self) -> None:
        plan = LibraryPlan(budget=100, exploit_fraction=0.6, interaction_fraction=0.2, exploration_fraction=0.2)
        assert plan.exploit_quota == 60
        assert plan.interaction_quota == 20
        assert plan.exploration_quota == 20

    def test_invalid_fractions_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            LibraryPlan(exploit_fraction=0.5, interaction_fraction=0.5, exploration_fraction=0.5)

    def test_negative_fraction(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            LibraryPlan(exploit_fraction=-0.1, interaction_fraction=0.6, exploration_fraction=0.5)

    def test_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="budget must be >= 0"):
            LibraryPlan(budget=-1)

    def test_zero_budget(self) -> None:
        plan = LibraryPlan(budget=0)
        assert plan.exploit_quota == 0
        assert plan.interaction_quota == 0
        assert plan.exploration_quota == 0

    def test_invalid_n_diversity_buckets(self) -> None:
        with pytest.raises(ValueError, match="n_diversity_buckets"):
            LibraryPlan(n_diversity_buckets=0)


# ---------------------------------------------------------------------------
# Pareto ranking tests
# ---------------------------------------------------------------------------


class TestParetoRanking:
    def test_empty_scores(self) -> None:
        assert compute_pareto_ranks([]) == []
        assert compute_dominates_count([]) == []

    def test_single_variant(self) -> None:
        assert compute_pareto_ranks([[0.5, 0.5]]) == [1]
        assert compute_dominates_count([[0.5, 0.5]]) == [0]

    def test_two_non_dominated(self) -> None:
        """Two variants that do not dominate each other."""
        ranks = compute_pareto_ranks([[0.9, 0.3], [0.3, 0.9]])
        assert ranks == [1, 1]

    def test_clear_dominance(self) -> None:
        """One variant strictly dominates the other."""
        ranks = compute_pareto_ranks([[0.9, 0.9], [0.5, 0.5]])
        assert ranks[0] == 1
        assert ranks[1] == 2

    def test_three_layer_fronts(self) -> None:
        """Three Pareto fronts: each layer is dominated by the one above."""
        scores = [
            [0.9, 0.9],  # front 1
            [0.8, 0.95],  # front 1 (not dominated by [0.9,0.9])
            [0.5, 0.5],  # front 2
            [0.3, 0.3],  # front 3
        ]
        ranks = compute_pareto_ranks(scores)
        assert ranks[0] == 1
        assert ranks[1] == 1
        assert ranks[2] == 2
        assert ranks[3] == 3

    def test_dominates_count_manual(self) -> None:
        scores = [
            [0.9, 0.9],  # dominates 2 (variants 2,3)
            [0.8, 0.95],  # dominates 2 (variants 2,3)
            [0.5, 0.5],  # dominates 1 (variant 3)
            [0.3, 0.3],  # dominates 0
        ]
        counts = compute_dominates_count(scores)
        assert counts[0] == 2
        assert counts[1] == 2
        assert counts[2] == 1
        assert counts[3] == 0

    def test_all_identical_scores(self) -> None:
        """All identical scores: all rank 1, dominates_count = 0."""
        scores = [[0.5, 0.5]] * 5
        ranks = compute_pareto_ranks(scores)
        assert all(r == 1 for r in ranks)
        counts = compute_dominates_count(scores)
        assert all(c == 0 for c in counts)


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------


class TestAnnotatePareto:
    def test_metadata_columns_present(self) -> None:
        df = _make_library_df(10)
        result = annotate_pareto_metadata(df)
        assert "pareto_rank" in result.columns
        assert "dominates_count" in result.columns
        assert "diversity_bucket" in result.columns

    def test_combined_score_preserved(self) -> None:
        df = _make_library_df(10)
        result = annotate_pareto_metadata(df)
        assert "combined_score" in result.columns
        pd.testing.assert_series_equal(result["combined_score"], df["combined_score"], check_names=True)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["stability_score", "nativeness_score", "mutations", "combined_score"])
        result = annotate_pareto_metadata(df)
        assert "pareto_rank" in result.columns
        assert len(result) == 0

    def test_does_not_modify_input(self) -> None:
        df = _make_library_df(5)
        original_cols = set(df.columns)
        _ = annotate_pareto_metadata(df)
        assert set(df.columns) == original_cols

    def test_pareto_rank_values_known_structure(self) -> None:
        df = _make_dominated_df()
        result = annotate_pareto_metadata(df)
        ranks = result["pareto_rank"].tolist()
        assert ranks[0] == 1  # (0.9, 0.9)
        assert ranks[1] == 1  # (0.8, 0.95)
        assert ranks[2] == 2  # (0.5, 0.5)
        assert ranks[3] == 3  # (0.3, 0.3)

    def test_diversity_bucket_deterministic(self) -> None:
        """Same input produces same diversity_bucket with repeated calls."""
        df = _make_library_df(15, seed=99)
        r1 = annotate_pareto_metadata(df, n_diversity_buckets=4)
        r2 = annotate_pareto_metadata(df, n_diversity_buckets=4)
        pd.testing.assert_series_equal(r1["diversity_bucket"], r2["diversity_bucket"])

    def test_missing_axes_raises(self) -> None:
        df = pd.DataFrame({"mutations": ["A5V"], "combined_score": [0.5]})
        with pytest.raises(ValueError, match="None of the requested scoring axes"):
            annotate_pareto_metadata(df, axes=("stability_score", "nativeness_score"))

    def test_partial_axes_graceful(self) -> None:
        """When only some axes are present, use available ones."""
        df = _make_library_df(5)
        df = df.drop(columns=["nativeness_score"])
        result = annotate_pareto_metadata(df, axes=("stability_score", "nativeness_score"))
        assert "pareto_rank" in result.columns


# ---------------------------------------------------------------------------
# Bucket assignment tests
# ---------------------------------------------------------------------------


class TestBucketAssignment:
    def test_buckets_are_valid(self) -> None:
        df = _make_library_df(20)
        annotated = annotate_pareto_metadata(df)
        plan = LibraryPlan(budget=15)
        result = assign_selection_buckets(annotated, plan)
        assigned = result[result["selection_bucket"] != ""]
        assert all(b in ALL_BUCKETS for b in assigned["selection_bucket"].values)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=[
                "stability_score",
                "nativeness_score",
                "mutations",
                "combined_score",
                "pareto_rank",
                "dominates_count",
                "diversity_bucket",
            ]
        )
        plan = LibraryPlan(budget=10)
        result = assign_selection_buckets(df, plan)
        assert "selection_bucket" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# plan_library integration tests
# ---------------------------------------------------------------------------


class TestPlanLibrary:
    def test_selected_count_matches_budget(self) -> None:
        df = _make_library_df(50, seed=7)
        plan = LibraryPlan(budget=20)
        result = plan_library(df, plan)
        assert len(result) == plan.budget

    def test_quota_allocation_sums_to_budget(self) -> None:
        df = _make_library_df(50, seed=7)
        plan = LibraryPlan(budget=20)
        result = plan_library(df, plan)
        exploit_n = (result["selection_bucket"] == BUCKET_EXPLOIT).sum()
        interaction_n = (result["selection_bucket"] == BUCKET_INTERACTION).sum()
        exploration_n = (result["selection_bucket"] == BUCKET_EXPLORATION).sum()
        assert exploit_n + interaction_n + exploration_n == plan.budget

    def test_exploit_quota_correct(self) -> None:
        df = _make_library_df(50, seed=7)
        plan = LibraryPlan(budget=20)
        result = plan_library(df, plan)
        exploit_n = (result["selection_bucket"] == BUCKET_EXPLOIT).sum()
        assert exploit_n == plan.exploit_quota

    def test_combined_score_still_present(self) -> None:
        df = _make_library_df(30)
        result = plan_library(df)
        assert "combined_score" in result.columns

    def test_pareto_metadata_present(self) -> None:
        df = _make_library_df(30)
        result = plan_library(df)
        assert "pareto_rank" in result.columns
        assert "dominates_count" in result.columns
        assert "diversity_bucket" in result.columns
        assert "selection_bucket" in result.columns

    def test_result_sorted_by_combined_score(self) -> None:
        df = _make_library_df(50, seed=7)
        result = plan_library(df, LibraryPlan(budget=20))
        scores = result["combined_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=[
                "variant_id",
                "mutations",
                "n_mutations",
                "stability_score",
                "nativeness_score",
                "combined_score",
                "aa_sequence",
            ]
        )
        result = plan_library(df)
        assert len(result) == 0
        assert "pareto_rank" in result.columns
        assert "selection_bucket" in result.columns

    def test_single_variant(self) -> None:
        df = _make_library_df(1, seed=0)
        plan = LibraryPlan(budget=1)
        result = plan_library(df, plan)
        assert len(result) == 1
        assert result.iloc[0]["selection_bucket"] in ALL_BUCKETS

    def test_deterministic_with_fixed_seed(self) -> None:
        """Calling plan_library twice with same input gives identical results."""
        df = _make_library_df(30, seed=123)
        r1 = plan_library(df, LibraryPlan(budget=15))
        r2 = plan_library(df, LibraryPlan(budget=15))
        pd.testing.assert_frame_equal(r1, r2)

    def test_budget_larger_than_input(self) -> None:
        """When budget exceeds input, select all available (no crash)."""
        df = _make_library_df(5, seed=10)
        plan = LibraryPlan(budget=100)
        result = plan_library(df, plan)
        # Can only select up to the number of available variants.
        assert len(result) <= len(df)

    def test_none_plan_annotates_all(self) -> None:
        """With plan=None, all variants should be annotated and returned."""
        df = _make_library_df(10, seed=55)
        result = plan_library(df)
        # Default plan sets budget=len(df), so all should be included.
        assert len(result) == len(df)

    def test_custom_axes(self) -> None:
        """Custom axes are used for Pareto analysis."""
        df = _make_library_df(20, seed=77)
        plan = LibraryPlan(budget=10, axes=("stability_score",))
        result = plan_library(df, plan)
        assert "pareto_rank" in result.columns
        assert len(result) == 10

    def test_all_identical_scores(self) -> None:
        """When all variants have the same scores, all are Pareto rank 1."""
        df = _make_library_df(10, seed=0)
        # Set identical scores.
        df["stability_score"] = 0.5
        df["nativeness_score"] = 0.5
        df["combined_score"] = 0.5
        result = plan_library(df, LibraryPlan(budget=10))
        assert all(result["pareto_rank"] == 1)
