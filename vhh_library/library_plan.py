"""Multi-objective ranking and quota-based library planning.

This module adds Pareto-dominance analysis, diversity bucketing, and
quota-based selection on top of the existing library generation pipeline.

The planner does **not** replace the existing generation strategies
(exhaustive, random, iterative).  Instead, it sits as a post-processing
layer: given a scored library DataFrame, it annotates each variant with
multi-objective metadata and selects a final subset that balances
exploitation, interaction mapping, and exploration.

``combined_score`` is preserved as a **temporary compatibility output**
(see library_design.instructions.md).  The planner does not build new
ranking logic on top of it; it reads the individual axis columns
(``stability_score``, ``nativeness_score``) and any optional columns
(``nanomelt_tm``, ``esm2_pll``) for Pareto analysis.

Buckets
-------
* **exploit** — high-scoring variants (Pareto-optimal or near-optimal).
* **interaction** — variants with moderate scores that cover diverse
  mutation positions, useful for mapping epistatic interactions.
* **exploration** — variants that are dissimilar to the exploit set,
  providing coverage of under-explored regions of sequence space.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default scoring axes used for Pareto analysis
# ---------------------------------------------------------------------------

_DEFAULT_AXES: tuple[str, ...] = ("stability_score", "nativeness_score")

# ---------------------------------------------------------------------------
# SelectionBucket enum-like constants
# ---------------------------------------------------------------------------

BUCKET_EXPLOIT: str = "exploit"
BUCKET_INTERACTION: str = "interaction"
BUCKET_EXPLORATION: str = "exploration"

ALL_BUCKETS: tuple[str, ...] = (BUCKET_EXPLOIT, BUCKET_INTERACTION, BUCKET_EXPLORATION)


# ---------------------------------------------------------------------------
# LibraryPlan dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LibraryPlan:
    """Configurable quota plan for library selection.

    The three fractions should sum to 1.0 (a ``ValueError`` is raised
    otherwise).  The ``budget`` is the total number of variants to select.

    Parameters
    ----------
    budget : int
        Total number of variants to include in the final library.
    exploit_fraction : float
        Fraction of the budget allocated to exploit (high-score) variants.
    interaction_fraction : float
        Fraction allocated to interaction-mapping variants.
    exploration_fraction : float
        Fraction allocated to exploration (diversity) variants.
    axes : tuple[str, ...]
        DataFrame column names used as scoring axes for Pareto analysis.
    n_diversity_buckets : int
        Number of diversity hash buckets for exploration diversity.
    """

    budget: int = 96
    exploit_fraction: float = 0.5
    interaction_fraction: float = 0.25
    exploration_fraction: float = 0.25
    axes: tuple[str, ...] = _DEFAULT_AXES
    n_diversity_buckets: int = 8

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError(f"budget must be >= 0, got {self.budget}")
        total = self.exploit_fraction + self.interaction_fraction + self.exploration_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Bucket fractions must sum to 1.0, got {total:.6f} "
                f"(exploit={self.exploit_fraction}, interaction={self.interaction_fraction}, "
                f"exploration={self.exploration_fraction})"
            )
        if any(f < 0.0 for f in (self.exploit_fraction, self.interaction_fraction, self.exploration_fraction)):
            raise ValueError("Bucket fractions must be non-negative")
        if self.n_diversity_buckets < 1:
            raise ValueError(f"n_diversity_buckets must be >= 1, got {self.n_diversity_buckets}")

    # -- Quota computation ------------------------------------------------

    @property
    def exploit_quota(self) -> int:
        """Number of variants allocated to exploit bucket."""
        return math.floor(self.budget * self.exploit_fraction)

    @property
    def interaction_quota(self) -> int:
        """Number of variants allocated to interaction bucket."""
        return math.floor(self.budget * self.interaction_fraction)

    @property
    def exploration_quota(self) -> int:
        """Number of variants allocated to exploration bucket.

        Absorbs any rounding remainder so quotas sum exactly to budget.
        """
        return self.budget - self.exploit_quota - self.interaction_quota


# ---------------------------------------------------------------------------
# Pareto-dominance utilities
# ---------------------------------------------------------------------------


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return ``True`` if *a* Pareto-dominates *b* (all axes >= and at least one >)."""
    dominated = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            dominated = True
    return dominated


def compute_pareto_ranks(scores: list[list[float]]) -> list[int]:
    """Assign Pareto ranks to each row (rank 1 = non-dominated front).

    Uses successive-front peeling: all non-dominated points get rank 1,
    then they are removed and the process repeats for rank 2, etc.

    Parameters
    ----------
    scores : list[list[float]]
        One row per variant, one column per objective.

    Returns
    -------
    list[int]
        Pareto rank for each variant (1-indexed).
    """
    n = len(scores)
    if n == 0:
        return []

    ranks = [0] * n
    remaining = set(range(n))
    rank = 1

    while remaining:
        front: list[int] = []
        remaining_list = sorted(remaining)
        for i in remaining_list:
            dominated_by_any = False
            for j in remaining_list:
                if i != j and _dominates(scores[j], scores[i]):
                    dominated_by_any = True
                    break
            if not dominated_by_any:
                front.append(i)
        for idx in front:
            ranks[idx] = rank
            remaining.discard(idx)
        rank += 1

    return ranks


def compute_dominates_count(scores: list[list[float]]) -> list[int]:
    """Count how many other variants each variant dominates.

    Parameters
    ----------
    scores : list[list[float]]
        One row per variant, one column per objective.

    Returns
    -------
    list[int]
        Dominates count for each variant.
    """
    n = len(scores)
    counts = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and _dominates(scores[i], scores[j]):
                counts[i] += 1
    return counts


# ---------------------------------------------------------------------------
# Diversity bucketing
# ---------------------------------------------------------------------------


def _mutation_hash(mutations_str: str, n_buckets: int) -> int:
    """Assign a diversity bucket based on the mutation string hash.

    Uses a deterministic hash so bucket assignment is reproducible.

    Parameters
    ----------
    mutations_str : str
        Comma-separated mutation string (e.g. ``"A5V, B10L"``).
    n_buckets : int
        Number of diversity buckets.

    Returns
    -------
    int
        Bucket index in ``[0, n_buckets)``.
    """
    digest = hashlib.md5(mutations_str.encode("utf-8")).hexdigest()  # noqa: S324
    return int(digest, 16) % n_buckets


# ---------------------------------------------------------------------------
# Annotation: add Pareto metadata to a library DataFrame
# ---------------------------------------------------------------------------


def annotate_pareto_metadata(
    df: pd.DataFrame,
    *,
    axes: tuple[str, ...] = _DEFAULT_AXES,
    n_diversity_buckets: int = 8,
) -> pd.DataFrame:
    """Add Pareto-dominance metadata columns to a library DataFrame.

    Columns added:
    * ``pareto_rank`` — Pareto front rank (1 = non-dominated).
    * ``dominates_count`` — number of other variants dominated.
    * ``diversity_bucket`` — deterministic hash bucket for diversity.

    The input DataFrame is **not** modified; a copy is returned.

    Parameters
    ----------
    df : pd.DataFrame
        Library DataFrame with at least the ``axes`` columns and a
        ``mutations`` column.
    axes : tuple[str, ...]
        Column names used as scoring axes for Pareto analysis.
    n_diversity_buckets : int
        Number of diversity hash buckets.

    Returns
    -------
    pd.DataFrame
        A copy with added metadata columns.
    """
    if df.empty:
        result = df.copy()
        result["pareto_rank"] = pd.Series(dtype="int64")
        result["dominates_count"] = pd.Series(dtype="int64")
        result["diversity_bucket"] = pd.Series(dtype="int64")
        return result

    # Validate that axes columns exist.
    available_axes = [ax for ax in axes if ax in df.columns]
    if not available_axes:
        raise ValueError(f"None of the requested scoring axes {axes} found in DataFrame columns {list(df.columns)}")

    scores = df[available_axes].values.tolist()

    ranks = compute_pareto_ranks(scores)
    dom_counts = compute_dominates_count(scores)
    buckets = [_mutation_hash(str(m), n_diversity_buckets) for m in df["mutations"].values]

    result = df.copy()
    result["pareto_rank"] = ranks
    result["dominates_count"] = dom_counts
    result["diversity_bucket"] = buckets
    return result


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------


def assign_selection_buckets(
    df: pd.DataFrame,
    plan: LibraryPlan,
) -> pd.DataFrame:
    """Assign a ``selection_bucket`` to each variant for quota selection.

    Assignment strategy:
    1. **exploit** — variants with ``pareto_rank == 1`` (non-dominated),
       ordered by ``combined_score`` descending.
    2. **interaction** — among remaining variants, pick those with the
       highest ``dominates_count`` (good but not dominant — useful for
       mapping interactions).
    3. **exploration** — remaining variants, spread across diversity
       buckets for maximal coverage.

    The input DataFrame **must** already have ``pareto_rank``,
    ``dominates_count``, ``diversity_bucket``, and ``combined_score``
    columns (call :func:`annotate_pareto_metadata` first).

    Parameters
    ----------
    df : pd.DataFrame
        Annotated library DataFrame.
    plan : LibraryPlan
        Quota configuration.

    Returns
    -------
    pd.DataFrame
        A copy with a ``selection_bucket`` column added.
    """
    if df.empty:
        result = df.copy()
        result["selection_bucket"] = pd.Series(dtype="object")
        return result

    result = df.copy()
    result["selection_bucket"] = ""

    # Sort by combined_score descending for tiebreaking.
    idx_by_score = result.sort_values("combined_score", ascending=False).index.tolist()

    assigned: set[int] = set()

    # 1. Exploit: Pareto rank 1, sorted by combined_score
    exploit_pool = [i for i in idx_by_score if result.at[i, "pareto_rank"] == 1 and i not in assigned]
    exploit_selected = exploit_pool[: plan.exploit_quota]
    for i in exploit_selected:
        result.at[i, "selection_bucket"] = BUCKET_EXPLOIT
        assigned.add(i)

    # 2. Interaction: highest dominates_count among remaining
    remaining_for_interaction = [i for i in idx_by_score if i not in assigned]
    remaining_for_interaction.sort(
        key=lambda i: (result.at[i, "dominates_count"], result.at[i, "combined_score"]),
        reverse=True,
    )
    interaction_selected = remaining_for_interaction[: plan.interaction_quota]
    for i in interaction_selected:
        result.at[i, "selection_bucket"] = BUCKET_INTERACTION
        assigned.add(i)

    # 3. Exploration: spread across diversity buckets
    remaining_for_exploration = [i for i in idx_by_score if i not in assigned]
    # Group by diversity_bucket and round-robin select.
    bucket_groups: dict[int, list[int]] = {}
    for i in remaining_for_exploration:
        b = result.at[i, "diversity_bucket"]
        bucket_groups.setdefault(b, []).append(i)

    exploration_quota = plan.exploration_quota
    exploration_selected: list[int] = []
    if bucket_groups:
        bucket_keys = sorted(bucket_groups.keys())
        round_robin_idx = 0
        while len(exploration_selected) < exploration_quota:
            added_any = False
            for bk in bucket_keys:
                if len(exploration_selected) >= exploration_quota:
                    break
                group = bucket_groups[bk]
                if round_robin_idx < len(group):
                    exploration_selected.append(group[round_robin_idx])
                    added_any = True
            round_robin_idx += 1
            if not added_any:
                break

    for i in exploration_selected:
        result.at[i, "selection_bucket"] = BUCKET_EXPLORATION
        assigned.add(i)

    # 4. Fill-up pass: if any bucket's pool was too small, fill the
    #    remaining budget from unassigned variants in combined_score order.
    #    These overflow slots are assigned to the bucket that has the
    #    largest remaining deficit.
    total_target = plan.budget
    if len(assigned) < total_target:
        unassigned = [i for i in idx_by_score if i not in assigned]
        shortfall = total_target - len(assigned)
        # Determine which bucket is most under-quota.
        bucket_counts = {
            BUCKET_EXPLOIT: sum(1 for i in assigned if result.at[i, "selection_bucket"] == BUCKET_EXPLOIT),
            BUCKET_INTERACTION: sum(1 for i in assigned if result.at[i, "selection_bucket"] == BUCKET_INTERACTION),
            BUCKET_EXPLORATION: sum(1 for i in assigned if result.at[i, "selection_bucket"] == BUCKET_EXPLORATION),
        }
        bucket_targets = {
            BUCKET_EXPLOIT: plan.exploit_quota,
            BUCKET_INTERACTION: plan.interaction_quota,
            BUCKET_EXPLORATION: plan.exploration_quota,
        }
        for i in unassigned[:shortfall]:
            # Assign to the bucket with the largest deficit.
            deficits = {b: bucket_targets[b] - bucket_counts[b] for b in ALL_BUCKETS}
            best_bucket = max(deficits, key=lambda b: deficits[b])
            result.at[i, "selection_bucket"] = best_bucket
            bucket_counts[best_bucket] += 1
            assigned.add(i)

    return result


# ---------------------------------------------------------------------------
# Main planning entry point
# ---------------------------------------------------------------------------


def plan_library(
    df: pd.DataFrame,
    plan: LibraryPlan | None = None,
) -> pd.DataFrame:
    """Annotate and select variants from a scored library DataFrame.

    This is the primary entry point for the library planner.  Given a
    DataFrame from :meth:`MutationEngine.generate_library`, it:

    1. Adds Pareto-dominance metadata (``pareto_rank``, ``dominates_count``,
       ``diversity_bucket``).
    2. Assigns a ``selection_bucket`` to each variant.
    3. Returns only the selected variants (up to ``plan.budget`` rows).

    ``combined_score`` is preserved for backward compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Library DataFrame with at least ``stability_score``,
        ``nativeness_score``, ``combined_score``, and ``mutations``
        columns.
    plan : LibraryPlan | None
        Quota plan.  If ``None``, a default plan is used with the budget
        set to the number of rows in *df* (i.e. annotate only, no
        pruning).

    Returns
    -------
    pd.DataFrame
        Annotated and selected library DataFrame, sorted by
        ``combined_score`` descending.
    """
    if df.empty:
        result = df.copy()
        for col in ("pareto_rank", "dominates_count", "diversity_bucket", "selection_bucket"):
            if col not in result.columns:
                result[col] = pd.Series(dtype="int64" if col != "selection_bucket" else "object")
        return result

    if plan is None:
        plan = LibraryPlan(budget=len(df))

    annotated = annotate_pareto_metadata(
        df,
        axes=plan.axes,
        n_diversity_buckets=plan.n_diversity_buckets,
    )
    bucketed = assign_selection_buckets(annotated, plan)

    # Select only rows that were assigned to a bucket.
    selected = bucketed[bucketed["selection_bucket"] != ""].copy()
    selected = selected.sort_values("combined_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Library plan: budget=%d, selected=%d (exploit=%d, interaction=%d, exploration=%d)",
        plan.budget,
        len(selected),
        (selected["selection_bucket"] == BUCKET_EXPLOIT).sum(),
        (selected["selection_bucket"] == BUCKET_INTERACTION).sum(),
        (selected["selection_bucket"] == BUCKET_EXPLORATION).sum(),
    )

    return selected
