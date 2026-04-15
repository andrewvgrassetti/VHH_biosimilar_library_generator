"""Core module for generating and ranking VHH variant libraries."""

from __future__ import annotations

import itertools
import logging
import math
import random
import re
from typing import TYPE_CHECKING, Optional

import pandas as pd

from vhh_library.developability import SurfaceHydrophobicityScorer
from vhh_library.humanness import HumAnnotator
from vhh_library.orthogonal_scoring import (
    ConsensusStabilityScorer,
    HumanStringContentScorer,
)
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer

if TYPE_CHECKING:
    from vhh_library.esm_scorer import ESMStabilityScorer

logger = logging.getLogger(__name__)

_PTM_LIABILITY_MOTIFS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"D[GSTD]"), "isomerization"),
    (re.compile(r"N[GSH]"), "deamidation"),
    (re.compile(r"N[^P][ST]"), "glycosylation"),
]

_SAMPLING_THRESHOLD = 50_000
_ITERATIVE_THRESHOLD = 1_000_000
_CONVERGENCE_THRESHOLD = 1e-4
_ANCHOR_UNLOCK_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _introduces_ptm_liability(
    parent_seq: str, mutant_seq: str, position_0idx: int
) -> bool:
    """Return True if a new PTM liability appears in a ±3 window around *position_0idx*."""
    start = max(0, position_0idx - 3)
    end = min(len(mutant_seq), position_0idx + 4)

    parent_window = parent_seq[start:end]
    mutant_window = mutant_seq[start:end]

    for pattern, _category in _PTM_LIABILITY_MOTIFS:
        parent_hits = {m.start() for m in pattern.finditer(parent_window)}
        mutant_hits = {m.start() for m in pattern.finditer(mutant_window)}
        if mutant_hits - parent_hits:
            return True
    return False


def _parse_mut_str(mut_str: str) -> list[tuple[int, str]]:
    """Parse ``"X1Y, A2B"`` to ``[(1, 'Y'), (2, 'B')]``."""
    result: list[tuple[int, str]] = []
    for token in mut_str.split(","):
        token = token.strip()
        if not token:
            continue
        pos = int(token[1:-1])
        new_aa = token[-1]
        result.append((pos, new_aa))
    return result


def _total_combinations(n: int, k_min: int, k_max: int) -> int:
    """Sum of C(n, k) for k in [k_min, k_max], capped to avoid overflow."""
    cap = 10**15
    total = 0
    for k in range(k_min, k_max + 1):
        try:
            total += math.comb(n, k)
        except (ValueError, OverflowError):
            return cap
        if total >= cap:
            return cap
    return total


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class MutationEngine:
    """Generate, score and rank VHH variant libraries."""

    METRIC_NAMES = ("humanness", "stability", "surface_hydrophobicity")

    def __init__(
        self,
        humanness_scorer: HumAnnotator,
        stability_scorer: StabilityScorer,
        *,
        hydrophobicity_scorer: Optional[SurfaceHydrophobicityScorer] = None,
        hsc_scorer: Optional[HumanStringContentScorer] = None,
        consensus_scorer: Optional[ConsensusStabilityScorer] = None,
        esm_scorer: Optional[ESMStabilityScorer] = None,
        w_humanness: float = 0.35,
        w_stability: float = 0.50,
        weights: Optional[dict[str, float]] = None,
        enabled_metrics: Optional[dict[str, bool]] = None,
    ) -> None:
        self._humanness_scorer = humanness_scorer
        self._stability_scorer = stability_scorer
        self._hydrophobicity_scorer = hydrophobicity_scorer
        self._hsc_scorer = hsc_scorer
        self._consensus_scorer = consensus_scorer
        self._esm_scorer = esm_scorer

        self.w_humanness = w_humanness
        self.w_stability = w_stability

        self._weights: dict[str, float] = {
            "humanness": 0.35,
            "stability": 0.50,
            "surface_hydrophobicity": 0.15,
        }
        if weights is not None:
            self._weights.update(weights)

        self._enabled_metrics: dict[str, bool] = {
            "humanness": True,
            "stability": True,
            "surface_hydrophobicity": False,
        }
        if enabled_metrics is not None:
            self._enabled_metrics.update(enabled_metrics)

    # ------------------------------------------------------------------
    # Lazy scorer properties
    # ------------------------------------------------------------------

    @property
    def hydrophobicity_scorer(self) -> SurfaceHydrophobicityScorer:
        if self._hydrophobicity_scorer is None:
            self._hydrophobicity_scorer = SurfaceHydrophobicityScorer()
        return self._hydrophobicity_scorer

    @property
    def hsc_scorer(self) -> HumanStringContentScorer:
        if self._hsc_scorer is None:
            self._hsc_scorer = HumanStringContentScorer()
        return self._hsc_scorer

    @property
    def consensus_scorer(self) -> ConsensusStabilityScorer:
        if self._consensus_scorer is None:
            self._consensus_scorer = ConsensusStabilityScorer()
        return self._consensus_scorer

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _active_weights(self) -> dict[str, float]:
        active = {
            k: v for k, v in self._weights.items() if self._enabled_metrics.get(k, False)
        }
        total = sum(active.values())
        if total == 0:
            return active
        return {k: v / total for k, v in active.items()}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_variant(self, vhh: VHHSequence) -> dict[str, float]:
        hum = self._humanness_scorer.score(vhh)
        stab = self._stability_scorer.score(vhh)

        scores: dict[str, float] = {
            "humanness": hum["composite_score"],
            "stability": stab["composite_score"],
            "aggregation_score": stab["aggregation_score"],
            "charge_balance_score": stab["charge_balance_score"],
            "hydrophobic_core_score": stab["hydrophobic_core_score"],
            "disulfide_score": stab["disulfide_score"],
            "vhh_hallmark_score": stab["vhh_hallmark_score"],
            "scoring_method": stab["scoring_method"],
        }

        if "predicted_tm" in stab:
            scores["predicted_tm"] = stab["predicted_tm"]

        if self._enabled_metrics.get("surface_hydrophobicity", False):
            sh = self.hydrophobicity_scorer.score(vhh)
            scores["surface_hydrophobicity"] = sh["composite_score"]
        else:
            scores["surface_hydrophobicity"] = 0.0

        scores["orthogonal_humanness"] = self.hsc_scorer.score(vhh)["composite_score"]
        scores["orthogonal_stability"] = self.consensus_scorer.score(vhh)["composite_score"]

        return scores

    def _combined_score(self, raw_scores: dict[str, float]) -> float:
        weights = self._active_weights()
        return sum(raw_scores.get(metric, 0.0) * w for metric, w in weights.items())

    # ------------------------------------------------------------------
    # Single-mutation ranking
    # ------------------------------------------------------------------

    def rank_single_mutations(
        self,
        vhh_sequence: VHHSequence,
        off_limits: Optional[set[int]] = None,
        forbidden_substitutions: Optional[dict[int, set[str]]] = None,
        excluded_target_aas: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        if off_limits is None:
            off_limits = set()

        suggestions = self._humanness_scorer.get_mutation_suggestions(
            vhh_sequence,
            off_limits=off_limits,
            forbidden_substitutions=forbidden_substitutions,
            excluded_target_aas=excluded_target_aas,
        )

        parent_seq = vhh_sequence.sequence
        rows: list[dict] = []

        for sug in suggestions:
            pos = sug["position"]
            pos_key = str(pos)
            new_aa: str = sug["suggested_aa"]
            original_aa: str = sug["original_aa"]

            mutant = VHHSequence.mutate(vhh_sequence, pos_key, new_aa)
            seq_idx = vhh_sequence._pos_to_seq_idx.get(pos_key, int(pos) - 1)
            if _introduces_ptm_liability(parent_seq, mutant.sequence, seq_idx):
                logger.debug(
                    "Skipping %s%s%s: introduces PTM liability", original_aa, pos_key, new_aa
                )
                continue

            delta_hum = sug["delta_humanness"]
            delta_stab = self._stability_scorer.predict_mutation_effect(
                vhh_sequence, pos_key, new_aa
            )
            delta_sh = (
                self.hydrophobicity_scorer.predict_mutation_effect(
                    vhh_sequence, pos_key, new_aa
                )
                if self._enabled_metrics.get("surface_hydrophobicity", False)
                else 0.0
            )

            raw_deltas: dict[str, float] = {
                "humanness": delta_hum,
                "stability": delta_stab,
                "surface_hydrophobicity": delta_sh,
            }
            combined = self._combined_score(raw_deltas)

            rows.append(
                {
                    "position": int("".join(c for c in pos_key if c.isdigit()) or "0"),
                    "imgt_pos": pos_key,
                    "original_aa": original_aa,
                    "suggested_aa": new_aa,
                    "delta_humanness": delta_hum,
                    "delta_stability": delta_stab,
                    "delta_surface_hydrophobicity": delta_sh,
                    "combined_score": combined,
                    "reason": sug["reason"],
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Mutation application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_mutations(
        sequence: str,
        mutations: list[tuple[int, str]],
        pos_to_seq_idx: dict[str, int] | None = None,
    ) -> str:
        """Apply mutations to a sequence.

        If *pos_to_seq_idx* is provided, use it to translate IMGT positions
        to 0-based sequence indices.  Otherwise fall back to ``pos - 1``.
        """
        seq_list = list(sequence)
        for idx, new_aa in mutations:
            if pos_to_seq_idx is not None:
                seq_idx = pos_to_seq_idx.get(str(idx))
                if seq_idx is None:
                    continue
            else:
                seq_idx = idx - 1
            seq_list[seq_idx] = new_aa
        return "".join(seq_list)

    # ------------------------------------------------------------------
    # Library generation
    # ------------------------------------------------------------------

    def generate_library(
        self,
        vhh_sequence: VHHSequence,
        top_mutations: pd.DataFrame,
        n_mutations: int,
        max_variants: int = 10_000,
        min_mutations: int = 1,
        strategy: str = "auto",
        anchor_threshold: float = 0.6,
        max_rounds: int = 5,
    ) -> pd.DataFrame:
        if top_mutations.empty:
            return self._empty_library_df()

        positions_seen: dict[int, list[int]] = {}
        for idx, row in top_mutations.iterrows():
            pos = int(row["position"])
            positions_seen.setdefault(pos, []).append(idx)

        unique_rows: list[int] = []
        for indices in positions_seen.values():
            unique_rows.append(indices[0])
        unique_mutations = top_mutations.loc[unique_rows].reset_index(drop=True)
        n_available = len(unique_mutations)

        k_max = min(n_mutations, n_available)
        k_min = min(min_mutations, k_max)
        total = _total_combinations(n_available, k_min, k_max)

        if strategy == "auto":
            if total <= _SAMPLING_THRESHOLD:
                strategy = "exhaustive"
            elif total <= _ITERATIVE_THRESHOLD:
                strategy = "random"
            else:
                strategy = "iterative"

        logger.info(
            "Library generation: strategy=%s, n_available=%d, k_min=%d, k_max=%d, "
            "total_combinations=%s",
            strategy,
            n_available,
            k_min,
            k_max,
            total,
        )

        mutation_list = list(unique_mutations.itertuples(index=False))

        if strategy == "exhaustive":
            rows = self._generate_exhaustive(
                vhh_sequence, mutation_list, k_min, k_max, max_variants
            )
        elif strategy == "random":
            rows = self._generate_sampled(
                vhh_sequence, mutation_list, k_min, k_max, max_variants
            )
        elif strategy == "iterative":
            rows = self._generate_iterative(
                vhh_sequence,
                mutation_list,
                k_min,
                k_max,
                max_variants,
                anchor_threshold,
                max_rounds,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        # ESM-2 progressive scoring (when scorer is available)
        if self._esm_scorer is not None and not df.empty:
            df = self._esm_scorer.score_library_progressive(
                vhh_sequence, df
            )

        return df

    # ------------------------------------------------------------------
    # Private: build a single variant row
    # ------------------------------------------------------------------

    def _build_variant_row(
        self,
        vhh_sequence: VHHSequence,
        selected: list,
        variant_counter: int,
    ) -> dict:
        mutations: list[tuple[int, str]] = [
            (int(m.position), m.suggested_aa) for m in selected
        ]
        mut_labels = [
            f"{m.original_aa}{m.position}{m.suggested_aa}" for m in selected
        ]

        mutant_seq = self.apply_mutations(
            vhh_sequence.sequence, mutations, vhh_sequence._pos_to_seq_idx
        )
        mutant_vhh = VHHSequence(mutant_seq)
        raw = self._score_variant(mutant_vhh)
        combined = self._combined_score(raw)

        row: dict = {
            "variant_id": f"V{variant_counter:06d}",
            "mutations": ", ".join(mut_labels),
            "n_mutations": len(mutations),
            "humanness_score": raw["humanness"],
            "stability_score": raw["stability"],
            "aggregation_score": raw["aggregation_score"],
            "charge_balance_score": raw["charge_balance_score"],
            "hydrophobic_core_score": raw["hydrophobic_core_score"],
            "disulfide_score": raw["disulfide_score"],
            "vhh_hallmark_score": raw["vhh_hallmark_score"],
            "surface_hydrophobicity_score": raw["surface_hydrophobicity"],
            "orthogonal_humanness_score": raw["orthogonal_humanness"],
            "orthogonal_stability_score": raw["orthogonal_stability"],
            "combined_score": combined,
            "aa_sequence": mutant_seq,
            "scoring_method": raw.get("scoring_method", "legacy"),
        }

        if "predicted_tm" in raw:
            row["predicted_tm"] = raw["predicted_tm"]

        return row

    # ------------------------------------------------------------------
    # Private: exhaustive enumeration
    # ------------------------------------------------------------------

    def _generate_exhaustive(
        self,
        vhh_sequence: VHHSequence,
        mutation_list: list,
        k_min: int,
        k_max: int,
        max_variants: int,
    ) -> list[dict]:
        rows: list[dict] = []
        counter = 1
        for k in range(k_min, k_max + 1):
            for combo in itertools.combinations(mutation_list, k):
                if self._has_position_conflict(combo):
                    continue
                rows.append(
                    self._build_variant_row(vhh_sequence, list(combo), counter)
                )
                counter += 1
                if len(rows) >= max_variants:
                    return rows
        return rows

    # ------------------------------------------------------------------
    # Private: random sampling (position-deduplicated)
    # ------------------------------------------------------------------

    def _generate_sampled(
        self,
        vhh_sequence: VHHSequence,
        mutation_list: list,
        k_min: int,
        k_max: int,
        max_variants: int,
    ) -> list[dict]:
        rows: list[dict] = []
        seen: set[frozenset[tuple[int, str]]] = set()
        counter = 1
        attempts = 0
        max_attempts = max_variants * 10

        while len(rows) < max_variants and attempts < max_attempts:
            attempts += 1
            k = random.randint(k_min, k_max)
            sample = random.sample(mutation_list, min(k, len(mutation_list)))
            sample = self._deduplicate_positions(sample)
            if not sample:
                continue

            key = frozenset((int(m.position), m.suggested_aa) for m in sample)
            if key in seen:
                continue
            seen.add(key)

            rows.append(self._build_variant_row(vhh_sequence, sample, counter))
            counter += 1

        return rows

    # ------------------------------------------------------------------
    # Private: constrained sampling (anchor-fixed)
    # ------------------------------------------------------------------

    def _generate_constrained_sampled(
        self,
        vhh_sequence: VHHSequence,
        mutation_list: list,
        k_min: int,
        k_max: int,
        max_variants: int,
        anchors: dict[int, str],
    ) -> list[dict]:
        anchor_muts = [m for m in mutation_list if anchors.get(int(m.position)) == m.suggested_aa]
        non_anchor = [m for m in mutation_list if int(m.position) not in anchors]

        rows: list[dict] = []
        seen: set[frozenset[tuple[int, str]]] = set()
        counter = 1
        attempts = 0
        max_attempts = max_variants * 10

        n_anchor = len(anchor_muts)

        while len(rows) < max_variants and attempts < max_attempts:
            attempts += 1
            k = random.randint(k_min, k_max)
            n_extra = max(0, k - n_anchor)
            if n_extra > len(non_anchor):
                n_extra = len(non_anchor)
            extra = random.sample(non_anchor, n_extra) if n_extra else []
            combined = anchor_muts + extra
            combined = self._deduplicate_positions(combined)
            if not combined:
                continue

            key = frozenset((int(m.position), m.suggested_aa) for m in combined)
            if key in seen:
                continue
            seen.add(key)

            rows.append(self._build_variant_row(vhh_sequence, combined, counter))
            counter += 1

        return rows

    # ------------------------------------------------------------------
    # Private: iterative anchor-and-explore
    # ------------------------------------------------------------------

    def _generate_iterative(
        self,
        vhh_sequence: VHHSequence,
        mutation_list: list,
        k_min: int,
        k_max: int,
        max_variants: int,
        anchor_threshold: float,
        max_rounds: int,
    ) -> list[dict]:
        per_round = max(max_variants // max(max_rounds, 1), 100)

        # Seed round
        rows = self._generate_sampled(
            vhh_sequence, mutation_list, k_min, k_max, per_round
        )
        if not rows:
            return rows

        prev_top_avg = -float("inf")
        stagnant_rounds = 0
        anchors: dict[int, str] = {}

        for round_idx in range(1, max_rounds):
            rows.sort(key=lambda r: r["combined_score"], reverse=True)
            top_quartile = rows[: max(len(rows) // 4, 1)]
            top_avg = sum(r["combined_score"] for r in top_quartile) / len(top_quartile)

            # Convergence check
            if abs(top_avg - prev_top_avg) < _CONVERGENCE_THRESHOLD:
                stagnant_rounds += 1
                if stagnant_rounds >= 2:
                    logger.info("Iterative generation converged at round %d", round_idx)
                    break
            else:
                stagnant_rounds = 0
            prev_top_avg = top_avg

            # Identify anchors from top quartile
            position_aa_counts: dict[tuple[int, str], int] = {}
            for r in top_quartile:
                for pos, aa in _parse_mut_str(r["mutations"]):
                    position_aa_counts[(pos, aa)] = position_aa_counts.get((pos, aa), 0) + 1

            n_top = len(top_quartile)
            new_anchors: dict[int, str] = {}
            for (pos, aa), count in position_aa_counts.items():
                if count / n_top >= anchor_threshold:
                    new_anchors[pos] = aa

            # Unlock anchors that don't help
            verified_anchors: dict[int, str] = {}
            for pos, aa in new_anchors.items():
                without = [
                    r["combined_score"]
                    for r in top_quartile
                    if (pos, aa) not in set(_parse_mut_str(r["mutations"]))
                ]
                with_anchor = [
                    r["combined_score"]
                    for r in top_quartile
                    if (pos, aa) in set(_parse_mut_str(r["mutations"]))
                ]
                avg_with = sum(with_anchor) / len(with_anchor) if with_anchor else 0.0
                avg_without = sum(without) / len(without) if without else 0.0
                if avg_with >= avg_without - _ANCHOR_UNLOCK_THRESHOLD:
                    verified_anchors[pos] = aa

            anchors = verified_anchors

            if anchors:
                new_rows = self._generate_constrained_sampled(
                    vhh_sequence, mutation_list, k_min, k_max, per_round, anchors
                )
            else:
                new_rows = self._generate_sampled(
                    vhh_sequence, mutation_list, k_min, k_max, per_round
                )

            rows.extend(new_rows)

            # De-duplicate by mutation set
            seen: set[str] = set()
            deduped: list[dict] = []
            for r in rows:
                if r["mutations"] not in seen:
                    seen.add(r["mutations"])
                    deduped.append(r)
            rows = deduped

            if len(rows) >= max_variants:
                rows.sort(key=lambda r: r["combined_score"], reverse=True)
                rows = rows[:max_variants]
                break

        return rows

    # ------------------------------------------------------------------
    # Private: position-conflict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_position_conflict(combo) -> bool:
        positions: set[int] = set()
        for m in combo:
            p = int(m.position)
            if p in positions:
                return True
            positions.add(p)
        return False

    @staticmethod
    def _deduplicate_positions(sample: list) -> list:
        seen: dict[int, object] = {}
        for m in sample:
            p = int(m.position)
            if p not in seen:
                seen[p] = m
        return list(seen.values())

    # ------------------------------------------------------------------
    # Private: empty DataFrame helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_library_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "variant_id",
                "mutations",
                "n_mutations",
                "humanness_score",
                "stability_score",
                "aggregation_score",
                "charge_balance_score",
                "hydrophobic_core_score",
                "disulfide_score",
                "vhh_hallmark_score",
                "surface_hydrophobicity_score",
                "orthogonal_humanness_score",
                "orthogonal_stability_score",
                "combined_score",
                "aa_sequence",
                "scoring_method",
            ]
        )
