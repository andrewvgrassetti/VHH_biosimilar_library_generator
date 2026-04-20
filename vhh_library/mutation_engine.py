"""Core module for generating and ranking VHH variant libraries."""

from __future__ import annotations

import itertools
import logging
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

from vhh_library.developability import SurfaceHydrophobicityScorer
from vhh_library.nativeness import NativenessScorer
from vhh_library.orthogonal_scoring import (
    ConsensusStabilityScorer,
)
from vhh_library.position_policy import DesignPolicy, PositionClass
from vhh_library.predictors.base import Predictor
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer
from vhh_library.utils import AMINO_ACIDS

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

# Iterative strategy constants
_DEFAULT_MAX_ROUNDS = 15
_MIN_DIVERSITY_ENTROPY = 0.5
_DIVERSITY_INJECTION_FRAC = 0.2
_RESCORE_TOP_N_DEFAULT = 20
_EPISTASIS_PAIR_LIMIT = 15

# Anchor confidence scoring parameters
_ANCHOR_FREQ_FLOOR = 0.3  # Minimum frequency to consider a candidate anchor
_ANCHOR_FREQ_SCALE = 0.8  # freq_threshold = anchor_threshold * this factor
_MARGINAL_SCALE = 10.0  # Scale marginal benefit to [0, 1] range
_MARGINAL_OFFSET = 0.5  # Shift so neutral marginal benefit maps to 0.5
_CONFIDENCE_W_FREQ = 0.4  # Weight of frequency in confidence score
_CONFIDENCE_W_MARGINAL = 0.4  # Weight of marginal benefit in confidence score
_CONFIDENCE_W_BASELINE = 0.2  # Baseline confidence for any passing candidate
_ANTAGONISTIC_SCALE = 2.0  # Penalty multiplier for antagonistic interactions


# ---------------------------------------------------------------------------
# Progress reporting dataclass
# ---------------------------------------------------------------------------


@dataclass
class IterativeProgress:
    """Snapshot of iterative strategy progress for UI callbacks."""

    phase: str
    round_number: int
    total_rounds: int
    best_score: float
    mean_score: float
    population_size: int
    n_anchors: int
    diversity_entropy: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Anchor with confidence
# ---------------------------------------------------------------------------


@dataclass
class AnchorCandidate:
    """An anchor mutation with confidence scoring."""

    position: int
    amino_acid: str
    frequency: float = 0.0
    marginal_benefit: float = 0.0
    confidence: float = 0.0
    interactions: dict[tuple[int, str], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MutationCandidate — rich metadata for a single-point mutation
# ---------------------------------------------------------------------------


@dataclass
class MutationCandidate:
    """Rich metadata container for a single-point mutation candidate.

    Carries all scoring axes and policy information so that downstream
    ranking layers can combine them without re-computing.

    Attributes
    ----------
    imgt_pos : str
        IMGT position string (e.g. ``"27"``, ``"111A"``).  Insertion-coded
        positions are preserved as-is — never collapsed to integers.
    original_aa : str
        Wild-type amino acid at this position.
    suggested_aa : str
        Proposed substitution amino acid.
    position_class : str
        One of ``"frozen"``, ``"conservative"``, ``"mutable"``.
    reasons : list[str]
        Human-readable reasons why this candidate was proposed (or filtered).
    abnativ_score : float | None
        Per-axis AbNatiV nativeness score for the mutant (composite_score).
    abnativ_delta : float | None
        Delta AbNatiV nativeness (mutant − wild-type).
    nanomelt_tm : float | None
        Predicted Tm (°C) from NanoMelt for the single-mutant.
    nanomelt_delta_tm : float | None
        Delta Tm (mutant − wild-type) from NanoMelt.
    esm2_prior_score : float | None
        ESM-2 PLL-derived score for the mutant (optional).
    esm2_pll : float | None
        Raw ESM-2 pseudo-log-likelihood (optional).
    liability_flags : list[str]
        PTM liability motifs introduced by this mutation.
    """

    imgt_pos: str
    original_aa: str
    suggested_aa: str
    position_class: str
    reasons: list[str] = field(default_factory=list)
    abnativ_score: float | None = None
    abnativ_delta: float | None = None
    nanomelt_tm: float | None = None
    nanomelt_delta_tm: float | None = None
    esm2_prior_score: float | None = None
    esm2_pll: float | None = None
    liability_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary for DataFrame construction."""
        return {
            "imgt_pos": self.imgt_pos,
            "original_aa": self.original_aa,
            "suggested_aa": self.suggested_aa,
            "position_class": self.position_class,
            "reasons": ", ".join(self.reasons),
            "abnativ_score": self.abnativ_score,
            "abnativ_delta": self.abnativ_delta,
            "nanomelt_tm": self.nanomelt_tm,
            "nanomelt_delta_tm": self.nanomelt_delta_tm,
            "esm2_prior_score": self.esm2_prior_score,
            "esm2_pll": self.esm2_pll,
            "liability_flags": ", ".join(self.liability_flags) if self.liability_flags else "",
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _detect_new_ptm_liabilities(parent_seq: str, mutant_seq: str, position_0idx: int) -> list[str]:
    """Return a list of PTM liability categories newly introduced by a mutation.

    Checks a ±3 residue window around *position_0idx* for new PTM motif matches.
    """
    start = max(0, position_0idx - 3)
    end = min(len(mutant_seq), position_0idx + 4)

    parent_window = parent_seq[start:end]
    mutant_window = mutant_seq[start:end]

    new_liabilities: list[str] = []
    for pattern, category in _PTM_LIABILITY_MOTIFS:
        parent_hits = {m.start() for m in pattern.finditer(parent_window)}
        mutant_hits = {m.start() for m in pattern.finditer(mutant_window)}
        if mutant_hits - parent_hits:
            new_liabilities.append(category)
    return new_liabilities


def _introduces_ptm_liability(parent_seq: str, mutant_seq: str, position_0idx: int) -> bool:
    """Return True if a new PTM liability appears in a ±3 window around *position_0idx*."""
    return bool(_detect_new_ptm_liabilities(parent_seq, mutant_seq, position_0idx))


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


def _total_grouped_combinations(
    position_groups: dict[int, list],
    k_min: int,
    k_max: int,
) -> int:
    """Count variants for the grouped-position model.

    Each position has one or more candidate AAs.  A variant picks at most one
    AA per position.  For a given *k* (number of positions to mutate), we:
      1. Choose *k* positions from the available ones – C(n_positions, k).
      2. For each chosen position, pick one of its AAs – product of group sizes.

    Because groups may have different sizes we iterate over combinations of
    position-groups (for small *n*) or estimate from the mean.  To keep it
    cheap we use the exact formula:
        total = Σ_{k=k_min}^{k_max} Σ_{S ⊂ positions, |S|=k} Π_{p∈S} |group_p|
    This equals Σ_k  e_k(g_1, g_2, …, g_n)  where e_k is the k-th elementary
    symmetric polynomial over the group sizes.
    """
    cap = 10**15
    sizes = [len(v) for v in position_groups.values()]
    n = len(sizes)
    k_max = min(k_max, n)
    k_min = min(k_min, k_max)

    # Use DP for elementary symmetric polynomials: e[j] after processing i
    # items equals the sum over all j-element subsets of the first i items of
    # the product of their sizes.
    e = [0] * (k_max + 1)
    e[0] = 1
    for sz in sizes:
        # Iterate backwards to avoid using the same element twice.
        for j in range(min(k_max, n), 0, -1):
            e[j] += e[j - 1] * sz
            if e[j] >= cap:
                return cap

    total = 0
    for k in range(k_min, k_max + 1):
        total += e[k]
        if total >= cap:
            return cap
    return total


def _imgt_key_to_int(pos_key: str) -> int:
    """Extract the integer portion from an IMGT position key (e.g. ``"111A"`` → 111)."""
    return int("".join(c for c in pos_key if c.isdigit()) or "0")


def _mutation_entropy(rows: list[dict]) -> float:
    """Compute Shannon entropy of mutation frequencies across all rows.

    Higher entropy indicates more diverse variants (more unique mutation
    combinations).  Returns 0.0 for empty input.
    """
    if not rows:
        return 0.0
    counts: Counter[tuple[int, str]] = Counter()
    total = 0
    for r in rows:
        for pos, aa in _parse_mut_str(r["mutations"]):
            counts[(pos, aa)] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_epistasis(
    rows: list[dict],
    mut_a: tuple[int, str],
    mut_b: tuple[int, str],
) -> float:
    """Compute epistatic interaction score between two mutations.

    interaction = score(A+B) - score(A_only) - score(B_only) + score(neither)

    Positive → synergistic; Negative → antagonistic.
    """
    scores_ab: list[float] = []
    scores_a: list[float] = []
    scores_b: list[float] = []
    scores_none: list[float] = []

    for r in rows:
        muts = set(_parse_mut_str(r["mutations"]))
        has_a = mut_a in muts
        has_b = mut_b in muts
        s = r["combined_score"]
        if has_a and has_b:
            scores_ab.append(s)
        elif has_a and not has_b:
            scores_a.append(s)
        elif has_b and not has_a:
            scores_b.append(s)
        else:
            scores_none.append(s)

    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        sv = sorted(vals)
        return sv[len(sv) // 2]

    return _median(scores_ab) - _median(scores_a) - _median(scores_b) + _median(scores_none)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class MutationEngine:
    """Generate, score and rank VHH variant libraries."""

    METRIC_NAMES = ("stability", "surface_hydrophobicity", "nativeness")

    def __init__(
        self,
        stability_scorer: StabilityScorer | None = None,
        nativeness_scorer: NativenessScorer | None = None,
        *,
        hydrophobicity_scorer: Optional[SurfaceHydrophobicityScorer] = None,
        consensus_scorer: Optional[ConsensusStabilityScorer] = None,
        esm_scorer: Optional["ESMStabilityScorer"] = None,
        w_stability: float = 0.70,
        w_nativeness: float = 0.30,
        weights: Optional[dict[str, float]] = None,
        enabled_metrics: Optional[dict[str, bool]] = None,
    ) -> None:
        self._stability_scorer = stability_scorer if stability_scorer is not None else StabilityScorer()
        self._nativeness_scorer = nativeness_scorer if nativeness_scorer is not None else NativenessScorer()
        self._hydrophobicity_scorer = hydrophobicity_scorer
        self._consensus_scorer = consensus_scorer
        self._esm_scorer = esm_scorer

        self.w_stability = w_stability
        self.w_nativeness = w_nativeness

        self._weights: dict[str, float] = {
            "stability": w_stability,
            "surface_hydrophobicity": 0.0,
            "nativeness": w_nativeness,
        }
        if weights is not None:
            self._weights.update(weights)

        self._enabled_metrics: dict[str, bool] = {
            "stability": True,
            "surface_hydrophobicity": False,
            "nativeness": True,
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
    def consensus_scorer(self) -> ConsensusStabilityScorer:
        if self._consensus_scorer is None:
            self._consensus_scorer = ConsensusStabilityScorer()
        return self._consensus_scorer

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _active_weights(self) -> dict[str, float]:
        active = {k: v for k, v in self._weights.items() if self._enabled_metrics.get(k, False)}
        total = sum(active.values())
        if total == 0:
            return active
        return {k: v / total for k, v in active.items()}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_variant(self, vhh: VHHSequence) -> dict[str, float]:
        stab = self._stability_scorer.score(vhh)

        scores: dict[str, float] = {
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

        nat = self._nativeness_scorer.score(vhh)
        scores["nativeness"] = nat["composite_score"]

        scores["orthogonal_stability"] = self.consensus_scorer.score(vhh)["composite_score"]

        return scores

    def _combined_score(self, raw_scores: dict[str, float]) -> float:
        weights = self._active_weights()
        return sum(raw_scores.get(metric, 0.0) * w for metric, w in weights.items())

    # ------------------------------------------------------------------
    # Stability-driven candidate generation
    # ------------------------------------------------------------------

    def _generate_stability_candidates(
        self,
        vhh_sequence: VHHSequence,
        off_limits: set[str],
        forbidden_substitutions: dict[str, set[str]] | None = None,
        excluded_target_aas: set[str] | None = None,
    ) -> list[dict]:
        """Generate candidate mutations ranked by stability impact.

        For each mutable position (respecting off-limits, CDRs, forbidden
        substitutions, and excluded AAs), all 19 possible substitutions are
        evaluated using :meth:`StabilityScorer.predict_mutation_effect`.
        Mutations introducing PTM liabilities are filtered out.

        Nativeness scoring is performed in a single batch call rather than
        per-candidate to avoid repeated ANARCI re-alignment overhead inside
        AbNatiV.
        """
        cdr_positions = vhh_sequence.cdr_positions
        parent_seq = vhh_sequence.sequence
        forbidden_str: dict[str, set[str]] = {}
        if forbidden_substitutions:
            forbidden_str = {str(k): v for k, v in forbidden_substitutions.items()}
        excluded = excluded_target_aas or set()

        candidates: list[dict] = []
        mutant_sequences: list[str] = []

        for pos_key, original_aa in vhh_sequence.imgt_numbered.items():
            if pos_key in off_limits or pos_key in cdr_positions:
                continue

            seq_idx = vhh_sequence._pos_to_seq_idx.get(pos_key, _imgt_key_to_int(pos_key) - 1)

            for candidate_aa in AMINO_ACIDS:
                if candidate_aa == original_aa:
                    continue
                if candidate_aa in excluded:
                    continue
                if forbidden_str and pos_key in forbidden_str and candidate_aa in forbidden_str[pos_key]:
                    continue

                mutant = VHHSequence.mutate(vhh_sequence, pos_key, candidate_aa)
                if _introduces_ptm_liability(parent_seq, mutant.sequence, seq_idx):
                    continue

                delta_stab = self._stability_scorer.predict_mutation_effect(vhh_sequence, pos_key, candidate_aa)

                candidate_dict: dict = {
                    "position": pos_key,
                    "original_aa": original_aa,
                    "suggested_aa": candidate_aa,
                    "delta_stability": delta_stab,
                    "reason": "Stability-driven scan",
                }

                candidates.append(candidate_dict)
                mutant_sequences.append(mutant.sequence)

        # Batch-score nativeness for all mutants in a single call instead of
        # invoking AbNatiV (and its internal ANARCI alignment) per candidate.
        if candidates:
            parent_nat = self._nativeness_scorer.score(vhh_sequence)["composite_score"]
            if hasattr(self._nativeness_scorer, "score_batch"):
                mutant_nat_scores = self._nativeness_scorer.score_batch(mutant_sequences)
                for candidate, mutant_nat in zip(candidates, mutant_nat_scores):
                    candidate["delta_nativeness"] = mutant_nat - parent_nat
            else:
                # Fallback for scorers that lack batch support — use delta directly.
                for candidate in candidates:
                    candidate["delta_nativeness"] = self._nativeness_scorer.predict_mutation_effect(
                        vhh_sequence, candidate["position"], candidate["suggested_aa"]
                    )

        candidates.sort(key=lambda c: c["delta_stability"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Policy-aware candidate generation (new path)
    # ------------------------------------------------------------------

    def generate_policy_aware_candidates(
        self,
        vhh_sequence: VHHSequence,
        policy: DesignPolicy,
        *,
        abnativ_predictor: Predictor | None = None,
        nanomelt_predictor: Predictor | None = None,
        esm2_predictor: Predictor | None = None,
        excluded_target_aas: set[str] | None = None,
    ) -> list[MutationCandidate]:
        """Generate mutation candidates respecting a :class:`DesignPolicy`.

        For each position in the sequence:

        * **FROZEN** — skipped entirely, no candidates emitted.
        * **CONSERVATIVE** — only substitutions in ``allowed_aas`` are proposed.
        * **MUTABLE** — all 19 standard substitutions are proposed.

        Each candidate is scored with AbNatiV and NanoMelt (when provided),
        and optionally with ESM-2.  PTM liability flags are recorded but do
        **not** filter candidates — the caller decides how to handle them.

        Parameters
        ----------
        vhh_sequence : VHHSequence
            The wild-type VHH sequence.
        policy : DesignPolicy
            Position-level mutation policy.
        abnativ_predictor : Predictor | None
            AbNatiV nativeness predictor (wraps NativenessScorer).
        nanomelt_predictor : Predictor | None
            NanoMelt thermal-stability predictor.
        esm2_predictor : Predictor | None
            ESM-2 prior predictor (optional).
        excluded_target_aas : set[str] | None
            Amino acids to exclude globally (e.g. ``{"C"}``).

        Returns
        -------
        list[MutationCandidate]
            Candidates with fully populated metadata.
        """
        parent_seq = vhh_sequence.sequence
        excluded = excluded_target_aas or set()

        # Pre-compute wild-type scores for delta calculations.
        wt_abnativ: float | None = None
        wt_nanomelt_tm: float | None = None
        if abnativ_predictor is not None:
            wt_result = abnativ_predictor.score_sequence(vhh_sequence)
            wt_abnativ = wt_result["composite_score"]
        if nanomelt_predictor is not None:
            wt_nm_result = nanomelt_predictor.score_sequence(vhh_sequence)
            wt_nanomelt_tm = wt_nm_result.get("nanomelt_tm")

        candidates: list[MutationCandidate] = []

        for pos_key, original_aa in vhh_sequence.imgt_numbered.items():
            pos_class = policy.effective_class(pos_key)

            # FROZEN — skip entirely.
            if pos_class is PositionClass.FROZEN:
                continue

            # Determine allowed amino acids.
            if pos_class is PositionClass.CONSERVATIVE:
                pos_policy = policy.get(pos_key)
                if pos_policy is not None and pos_policy.allowed_aas is not None:
                    allowed_aas = pos_policy.allowed_aas - {original_aa} - excluded
                else:
                    # Conservative position without explicit allowed_aas — this
                    # can happen when the position class is inferred from the
                    # region default without an explicit PositionPolicy entry.
                    logger.warning(
                        "CONSERVATIVE position %s has no allowed_aas defined; skipping",
                        pos_key,
                    )
                    continue
            else:
                # MUTABLE — all standard AAs except self and excluded.
                allowed_aas = AMINO_ACIDS - {original_aa} - excluded

            seq_idx = vhh_sequence._pos_to_seq_idx.get(pos_key, _imgt_key_to_int(pos_key) - 1)

            for candidate_aa in sorted(allowed_aas):
                mutant = VHHSequence.mutate(vhh_sequence, pos_key, candidate_aa)

                # Detect PTM liabilities (recorded, not filtered).
                liability_flags = _detect_new_ptm_liabilities(parent_seq, mutant.sequence, seq_idx)

                reasons: list[str] = []
                if pos_class is PositionClass.CONSERVATIVE:
                    reasons.append(f"Conservative: restricted to allowed_aas at IMGT {pos_key}")
                else:
                    reasons.append(f"Mutable: full exploration at IMGT {pos_key}")

                candidate = MutationCandidate(
                    imgt_pos=pos_key,
                    original_aa=original_aa,
                    suggested_aa=candidate_aa,
                    position_class=pos_class.value,
                    reasons=reasons,
                    liability_flags=liability_flags,
                )

                # Score with AbNatiV.
                if abnativ_predictor is not None:
                    mut_result = abnativ_predictor.score_sequence(mutant)
                    candidate.abnativ_score = mut_result["composite_score"]
                    if wt_abnativ is not None:
                        candidate.abnativ_delta = candidate.abnativ_score - wt_abnativ

                # Score with NanoMelt.
                if nanomelt_predictor is not None:
                    nm_result = nanomelt_predictor.score_sequence(mutant)
                    candidate.nanomelt_tm = nm_result.get("nanomelt_tm")
                    if wt_nanomelt_tm is not None and candidate.nanomelt_tm is not None:
                        candidate.nanomelt_delta_tm = candidate.nanomelt_tm - wt_nanomelt_tm

                # Score with ESM-2 prior (optional).
                if esm2_predictor is not None:
                    esm_result = esm2_predictor.score_sequence(mutant)
                    candidate.esm2_prior_score = esm_result.get("composite_score")
                    candidate.esm2_pll = esm_result.get("esm2_pll")

                candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------
    # Multi-mutant full-sequence rescoring
    # ------------------------------------------------------------------

    @staticmethod
    def rescore_multi_mutant(
        vhh_sequence: VHHSequence,
        mutations: list[tuple[str, str]],
        *,
        abnativ_predictor: Predictor | None = None,
        nanomelt_predictor: Predictor | None = None,
        esm2_predictor: Predictor | None = None,
    ) -> dict[str, float | None]:
        """Re-score a multi-mutant sequence using full-sequence inference.

        This method applies *all* mutations simultaneously and runs each
        predictor on the complete mutant sequence.  It does **not**
        approximate by summing single-mutation deltas.

        Parameters
        ----------
        vhh_sequence : VHHSequence
            The wild-type VHH sequence.
        mutations : list[tuple[str, str]]
            List of ``(imgt_pos, new_aa)`` pairs.
        abnativ_predictor, nanomelt_predictor, esm2_predictor : Predictor | None
            Predictor instances to use for scoring.

        Returns
        -------
        dict[str, float | None]
            Keys include ``"abnativ_score"``, ``"nanomelt_tm"``,
            ``"esm2_prior_score"``, ``"esm2_pll"`` — each ``None`` if
            the corresponding predictor was not provided.
        """
        # Build the multi-mutant VHHSequence by applying mutations sequentially
        # using the fast-path mutate.  Each mutation targets a distinct IMGT
        # position, so application order does not affect the final sequence.
        # VHHSequence.mutate copies the parent's numbering and only updates
        # the changed residue, keeping IMGT alignment intact.
        current = vhh_sequence
        for imgt_pos, new_aa in mutations:
            current = VHHSequence.mutate(current, imgt_pos, new_aa)

        result: dict[str, float | None] = {
            "abnativ_score": None,
            "nanomelt_tm": None,
            "esm2_prior_score": None,
            "esm2_pll": None,
        }

        if abnativ_predictor is not None:
            scores = abnativ_predictor.score_sequence(current)
            result["abnativ_score"] = scores.get("composite_score")

        if nanomelt_predictor is not None:
            scores = nanomelt_predictor.score_sequence(current)
            result["nanomelt_tm"] = scores.get("nanomelt_tm")

        if esm2_predictor is not None:
            scores = esm2_predictor.score_sequence(current)
            result["esm2_prior_score"] = scores.get("composite_score")
            result["esm2_pll"] = scores.get("esm2_pll")

        return result

    # ------------------------------------------------------------------
    # Single-mutation ranking
    # ------------------------------------------------------------------

    def rank_single_mutations(
        self,
        vhh_sequence: VHHSequence,
        off_limits: Optional[set[int] | set[str]] = None,
        forbidden_substitutions: Optional[dict[int, set[str]] | dict[str, set[str]]] = None,
        excluded_target_aas: Optional[set[str]] = None,
        max_per_position: int = 1,
    ) -> pd.DataFrame:
        if off_limits is None:
            off_limits = set()

        # Normalise off_limits to string keys
        off_limits_str = {str(p) for p in off_limits}

        # Normalise forbidden_substitutions keys to strings
        forbidden_str: dict[str, set[str]] | None = None
        if forbidden_substitutions:
            forbidden_str = {str(k): v for k, v in forbidden_substitutions.items()}

        # Candidate generation: stability-driven scan
        suggestions = self._generate_stability_candidates(
            vhh_sequence,
            off_limits=off_limits_str,
            forbidden_substitutions=forbidden_str,
            excluded_target_aas=excluded_target_aas,
        )

        rows: list[dict] = []

        for sug in suggestions:
            pos = sug["position"]
            pos_key = str(pos)
            new_aa: str = sug["suggested_aa"]
            original_aa: str = sug["original_aa"]

            delta_stab = sug.get("delta_stability")
            if delta_stab is None:
                delta_stab = self._stability_scorer.predict_mutation_effect(vhh_sequence, pos_key, new_aa)
            delta_sh = (
                self.hydrophobicity_scorer.predict_mutation_effect(vhh_sequence, pos_key, new_aa)
                if self._enabled_metrics.get("surface_hydrophobicity", False)
                else 0.0
            )
            delta_nat = sug.get("delta_nativeness")
            if delta_nat is None:
                delta_nat = self._nativeness_scorer.predict_mutation_effect(vhh_sequence, pos_key, new_aa)

            raw_deltas: dict[str, float] = {
                "stability": delta_stab,
                "surface_hydrophobicity": delta_sh,
                "nativeness": delta_nat,
            }
            combined = self._combined_score(raw_deltas)

            rows.append(
                {
                    "position": _imgt_key_to_int(pos_key),
                    "imgt_pos": pos_key,
                    "original_aa": original_aa,
                    "suggested_aa": new_aa,
                    "delta_stability": delta_stab,
                    "delta_surface_hydrophobicity": delta_sh,
                    "delta_nativeness": delta_nat,
                    "combined_score": combined,
                    "reason": sug["reason"],
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
            # Limit to max_per_position candidates per IMGT position.
            if max_per_position > 0:
                df = df.groupby("imgt_pos", sort=False).head(max_per_position).reset_index(drop=True)
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
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
        rescore_top_n: int = _RESCORE_TOP_N_DEFAULT,
        progress_callback: Optional[Callable[[IterativeProgress], None]] = None,
    ) -> pd.DataFrame:
        if top_mutations.empty:
            return self._empty_library_df()

        # Keep all candidates (multiple AAs per position allowed).
        mutation_list = list(top_mutations.itertuples(index=False))

        # Build position groups for combination counting.
        position_groups: dict[int, list] = {}
        for m in mutation_list:
            position_groups.setdefault(int(m.position), []).append(m)

        n_positions = len(position_groups)
        k_max = min(n_mutations, n_positions)
        k_min = min(min_mutations, k_max)

        if min_mutations > n_positions:
            logger.warning(
                "Requested min_mutations=%d but only %d unique positions available "
                "in the top mutations list. Clamping min_mutations to %d. "
                "Increase 'Top N mutations for library' to include more positions.",
                min_mutations,
                n_positions,
                k_min,
            )

        total = _total_grouped_combinations(position_groups, k_min, k_max)

        if total < max_variants:
            logger.warning(
                "Requested %d variants but the search space only contains %d "
                "unique combinations (%d positions, k_min=%d, k_max=%d). "
                "Increase 'Top N mutations for library' to expand the search space.",
                max_variants,
                total,
                n_positions,
                k_min,
                k_max,
            )

        if strategy == "auto":
            if total <= _SAMPLING_THRESHOLD:
                strategy = "exhaustive"
            elif total <= _ITERATIVE_THRESHOLD:
                strategy = "random"
            else:
                strategy = "iterative"

        logger.info(
            "Library generation: strategy=%s, n_positions=%d, n_candidates=%d, "
            "k_min=%d, k_max=%d, total_combinations=%s",
            strategy,
            n_positions,
            len(mutation_list),
            k_min,
            k_max,
            total,
        )

        if strategy == "exhaustive":
            rows = self._generate_exhaustive(
                vhh_sequence,
                mutation_list,
                k_min,
                k_max,
                max_variants,
                position_groups=position_groups,
            )
        elif strategy == "random":
            rows = self._generate_sampled(vhh_sequence, mutation_list, k_min, k_max, max_variants)
        elif strategy == "iterative":
            rows = self._generate_iterative(
                vhh_sequence,
                mutation_list,
                k_min,
                k_max,
                max_variants,
                anchor_threshold,
                max_rounds,
                rescore_top_n=rescore_top_n,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        # ESM-2 progressive scoring (when scorer is available)
        if self._esm_scorer is not None and not df.empty:
            df = self._esm_scorer.score_library_progressive(vhh_sequence, df)

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
        mutations: list[tuple[int, str]] = [(int(m.position), m.suggested_aa) for m in selected]
        mut_labels = [f"{m.original_aa}{m.position}{m.suggested_aa}" for m in selected]

        # Use fast-path VHHSequence.mutate() to avoid redundant ANARCI calls.
        current = vhh_sequence
        for imgt_pos, new_aa in mutations:
            current = VHHSequence.mutate(current, str(imgt_pos), new_aa)
        mutant_vhh = current
        mutant_seq = current.sequence
        raw = self._score_variant(mutant_vhh)
        combined = self._combined_score(raw)

        row: dict = {
            "variant_id": f"V{variant_counter:06d}",
            "mutations": ", ".join(mut_labels),
            "n_mutations": len(mutations),
            "stability_score": raw["stability"],
            "nativeness_score": raw["nativeness"],
            "aggregation_score": raw["aggregation_score"],
            "charge_balance_score": raw["charge_balance_score"],
            "hydrophobic_core_score": raw["hydrophobic_core_score"],
            "disulfide_score": raw["disulfide_score"],
            "vhh_hallmark_score": raw["vhh_hallmark_score"],
            "surface_hydrophobicity_score": raw["surface_hydrophobicity"],
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
        position_groups: dict[int, list] | None = None,
    ) -> list[dict]:
        # Build position groups if not provided.
        if position_groups is None:
            position_groups = {}
            for m in mutation_list:
                position_groups.setdefault(int(m.position), []).append(m)

        positions = list(position_groups.keys())
        groups = [position_groups[p] for p in positions]

        rows: list[dict] = []
        counter = 1
        for k in range(k_min, k_max + 1):
            for pos_indices in itertools.combinations(range(len(positions)), k):
                # For each chosen set of positions, take the product of their AA options.
                selected_groups = [groups[i] for i in pos_indices]
                for aa_combo in itertools.product(*selected_groups):
                    rows.append(self._build_variant_row(vhh_sequence, list(aa_combo), counter))
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
        # Build position groups for position-aware sampling.
        position_groups: dict[int, list] = {}
        for m in mutation_list:
            position_groups.setdefault(int(m.position), []).append(m)
        positions = list(position_groups.keys())

        # Clamp k range to available positions.
        effective_k_max = min(k_max, len(positions))
        effective_k_min = min(k_min, effective_k_max)
        if effective_k_min < 1:
            effective_k_min = 1

        rows: list[dict] = []
        seen: set[frozenset[tuple[int, str]]] = set()
        counter = 1
        attempts = 0
        max_attempts = max_variants * 20

        while len(rows) < max_variants and attempts < max_attempts:
            attempts += 1
            k = random.randint(effective_k_min, effective_k_max)
            # Sample positions first, then pick one AA per position.
            sampled_positions = random.sample(positions, k)
            sample = [random.choice(position_groups[p]) for p in sampled_positions]

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

        # Build position groups for non-anchor mutations.
        non_anchor_groups: dict[int, list] = {}
        for m in non_anchor:
            non_anchor_groups.setdefault(int(m.position), []).append(m)
        non_anchor_positions = list(non_anchor_groups.keys())

        rows: list[dict] = []
        seen: set[frozenset[tuple[int, str]]] = set()
        counter = 1
        attempts = 0
        max_attempts = max_variants * 20

        n_anchor = len(anchor_muts)

        while len(rows) < max_variants and attempts < max_attempts:
            attempts += 1
            k = random.randint(k_min, k_max)
            n_extra = max(0, k - n_anchor)
            n_extra = min(n_extra, len(non_anchor_positions))
            if n_extra > 0:
                sampled_positions = random.sample(non_anchor_positions, n_extra)
                extra = [random.choice(non_anchor_groups[p]) for p in sampled_positions]
            else:
                extra = []
            combined = anchor_muts + extra
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
    # Private: iterative anchor-and-explore (Evolutionary Stability
    # Optimization)
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
        *,
        rescore_top_n: int = _RESCORE_TOP_N_DEFAULT,
        progress_callback: Optional[Callable[[IterativeProgress], None]] = None,
    ) -> list[dict]:
        """Multi-phase evolutionary strategy with epistasis-aware anchoring.

        Phase 1 – Broad Exploration: random sampling, no anchors, survey landscape.
        Phase 2 – Anchor Identification: statistical anchor selection + epistasis.
        Phase 3 – Focused Exploitation: anchor-constrained sampling with diversity.
        Phase 4 – Final Validation: ESM-2 full-PLL re-scoring of top variants.
        """
        # Allocate rounds to phases (adaptive to max_rounds)
        n_explore = max(2, max_rounds // 5)
        n_anchor_id = max(1, max_rounds // 7)
        n_exploit = max(2, max_rounds - n_explore - n_anchor_id - 1)
        total_phases = n_explore + n_anchor_id + n_exploit + 1  # +1 validation

        per_round_explore = max(max_variants // max(total_phases, 1), 50)
        per_round_exploit = max(per_round_explore // 2, 30)

        all_rows: list[dict] = []
        seen_keys: set[str] = set()
        counter = 1
        global_round = 0
        anchor_candidates: list[AnchorCandidate] = []
        prev_top_avg = -float("inf")
        prev_diversity = -1.0
        stagnant_rounds = 0

        def _add_rows(new_rows: list[dict]) -> None:
            nonlocal counter
            for r in new_rows:
                if r["mutations"] not in seen_keys:
                    seen_keys.add(r["mutations"])
                    all_rows.append(r)
                    counter += 1

        def _report(phase: str, msg: str = "") -> None:
            if progress_callback is None:
                return
            best = max((r["combined_score"] for r in all_rows), default=0.0)
            mean = sum(r["combined_score"] for r in all_rows) / len(all_rows) if all_rows else 0.0
            div = _mutation_entropy(all_rows) if all_rows else 0.0
            progress_callback(
                IterativeProgress(
                    phase=phase,
                    round_number=global_round,
                    total_rounds=total_phases,
                    best_score=best,
                    mean_score=mean,
                    population_size=len(all_rows),
                    n_anchors=len(anchor_candidates),
                    diversity_entropy=div,
                    message=msg,
                )
            )

        def _esm_score_rows(rows: list[dict]) -> list[dict]:
            """Apply ESM-2 delta PLL scoring to rows when available."""
            if self._esm_scorer is None or not rows:
                return rows
            parent_seq = vhh_sequence.sequence
            variants: list[tuple[list[int], list[str]]] = []
            for r in rows:
                seq = r["aa_sequence"]
                positions: list[int] = []
                new_aas: list[str] = []
                for i, (p_aa, v_aa) in enumerate(zip(parent_seq, seq)):
                    if p_aa != v_aa:
                        positions.append(i)
                        new_aas.append(v_aa)
                variants.append((positions, new_aas))
            delta_scores = self._esm_scorer.score_delta(parent_seq, variants)
            for r, ds in zip(rows, delta_scores):
                r["esm2_delta_pll"] = ds
            return rows

        def _esm_rescore_full(rows: list[dict], top_n: int) -> list[dict]:
            """Re-score top N variants with full PLL for accuracy."""
            if self._esm_scorer is None or not rows or top_n <= 0:
                return rows
            score_key = "esm2_delta_pll" if "esm2_delta_pll" in rows[0] else "combined_score"
            sorted_rows = sorted(rows, key=lambda r: r.get(score_key, 0.0), reverse=True)
            top_rows = sorted_rows[:top_n]
            seqs = [r["aa_sequence"] for r in top_rows]
            plls = self._esm_scorer.score_batch(seqs)
            for r, pll in zip(top_rows, plls):
                r["esm2_full_pll"] = pll
            return rows

        # ==================================================================
        # PHASE 1 — Broad Exploration
        # ==================================================================
        logger.info("Phase 1: Broad exploration (%d rounds)", n_explore)
        for _ in range(n_explore):
            global_round += 1
            new = self._generate_sampled(vhh_sequence, mutation_list, k_min, k_max, per_round_explore)
            new = _esm_score_rows(new)
            _add_rows(new)
            _report("exploration", f"Exploring ({len(all_rows)} variants)")

            if len(all_rows) >= max_variants:
                break

        if not all_rows:
            return all_rows

        # ==================================================================
        # PHASE 2 — Anchor Identification with epistasis detection
        # ==================================================================
        logger.info("Phase 2: Anchor identification (%d rounds)", n_anchor_id)
        for _ in range(n_anchor_id):
            global_round += 1
            new = self._generate_sampled(vhh_sequence, mutation_list, k_min, k_max, per_round_explore)
            new = _esm_score_rows(new)
            _add_rows(new)

        # Identify anchor candidates
        anchor_candidates = self._identify_anchors_with_epistasis(all_rows, anchor_threshold)
        logger.info(
            "Identified %d anchor candidates (top confidence: %.2f)",
            len(anchor_candidates),
            max((a.confidence for a in anchor_candidates), default=0.0),
        )
        _report(
            "anchor_identification",
            f"{len(anchor_candidates)} anchors identified",
        )

        # ==================================================================
        # PHASE 3 — Focused Exploitation
        # ==================================================================
        logger.info("Phase 3: Focused exploitation (%d rounds)", n_exploit)
        for exploit_round in range(n_exploit):
            global_round += 1

            # Build weighted anchor set from confident candidates
            anchors = self._select_anchors_weighted(anchor_candidates)

            if anchors:
                new = self._generate_constrained_sampled(
                    vhh_sequence, mutation_list, k_min, k_max, per_round_exploit, anchors
                )
            else:
                new = self._generate_sampled(vhh_sequence, mutation_list, k_min, k_max, per_round_exploit)

            new = _esm_score_rows(new)
            _add_rows(new)

            # Diversity injection: if entropy drops, add random variants
            diversity = _mutation_entropy(all_rows)
            if diversity < _MIN_DIVERSITY_ENTROPY and mutation_list:
                inject_n = max(int(per_round_exploit * _DIVERSITY_INJECTION_FRAC), 5)
                inject = self._generate_sampled(vhh_sequence, mutation_list, k_min, k_max, inject_n)
                inject = _esm_score_rows(inject)
                _add_rows(inject)
                logger.debug(
                    "Injected %d random variants for diversity (entropy=%.3f)",
                    len(inject),
                    diversity,
                )

            # ESM-2 full PLL rescore top N each round
            _esm_rescore_full(all_rows, rescore_top_n)

            # Convergence check: score stagnation + diversity stabilisation
            sorted_rows = sorted(all_rows, key=lambda r: r["combined_score"], reverse=True)
            top_q = sorted_rows[: max(len(sorted_rows) // 4, 1)]
            top_avg = sum(r["combined_score"] for r in top_q) / len(top_q)
            score_stagnant = abs(top_avg - prev_top_avg) < _CONVERGENCE_THRESHOLD
            diversity_stable = abs(diversity - prev_diversity) < 0.05 if prev_diversity >= 0 else False
            if score_stagnant and diversity_stable:
                stagnant_rounds += 1
                if stagnant_rounds >= 2:
                    logger.info(
                        "Converged at round %d (score=%.4f, entropy=%.3f)",
                        global_round,
                        top_avg,
                        diversity,
                    )
                    break
            else:
                stagnant_rounds = 0
            prev_top_avg = top_avg
            prev_diversity = diversity

            # Adaptive anchor update: re-assess if new data changes landscape
            if exploit_round > 0 and exploit_round % 2 == 0:
                anchor_candidates = self._identify_anchors_with_epistasis(all_rows, anchor_threshold)

            _report(
                "exploitation",
                f"Round {global_round}: {len(all_rows)} variants, best={sorted_rows[0]['combined_score']:.4f}",
            )

            if len(all_rows) >= max_variants:
                break

        # ==================================================================
        # PHASE 4 — Final Validation
        # ==================================================================
        global_round += 1
        logger.info("Phase 4: Final validation")

        # ESM-2 full PLL for top candidates
        _esm_rescore_full(all_rows, rescore_top_n * 2)
        _report("validation", "Final scoring complete")

        # Trim to max_variants, keeping the best
        all_rows.sort(key=lambda r: r["combined_score"], reverse=True)
        if len(all_rows) > max_variants:
            all_rows = all_rows[:max_variants]

        return all_rows

    # ------------------------------------------------------------------
    # Anchor identification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_anchors_with_epistasis(
        rows: list[dict],
        anchor_threshold: float,
    ) -> list[AnchorCandidate]:
        """Identify anchor mutations with marginal contribution + pairwise epistasis.

        Returns a list of :class:`AnchorCandidate` with confidence scores.
        """
        if not rows:
            return []

        sorted_rows = sorted(rows, key=lambda r: r["combined_score"], reverse=True)
        n_top = max(len(sorted_rows) // 4, 1)
        top_quartile = sorted_rows[:n_top]
        all_scores = [r["combined_score"] for r in sorted_rows]
        median_all = sorted(all_scores)[len(all_scores) // 2]

        # 1. Frequency analysis
        pa_counts: Counter[tuple[int, str]] = Counter()
        for r in top_quartile:
            for pos, aa in _parse_mut_str(r["mutations"]):
                pa_counts[(pos, aa)] += 1

        # Candidate anchors: those exceeding a relaxed frequency threshold.
        # _ANCHOR_FREQ_SCALE (0.8×) softens the user-set anchor_threshold so
        # we don't miss borderline candidates; _ANCHOR_FREQ_FLOOR (0.3) is the
        # absolute minimum to avoid pure noise.
        freq_threshold = max(anchor_threshold * _ANCHOR_FREQ_SCALE, _ANCHOR_FREQ_FLOOR)
        candidates: dict[tuple[int, str], AnchorCandidate] = {}
        for (pos, aa), count in pa_counts.items():
            freq = count / n_top
            if freq >= freq_threshold:
                candidates[(pos, aa)] = AnchorCandidate(position=pos, amino_acid=aa, frequency=freq)

        if not candidates:
            return []

        # 2. Marginal contribution analysis (robust: median comparison)
        for key, cand in candidates.items():
            with_scores = []
            without_scores = []
            for r in sorted_rows:
                muts = set(_parse_mut_str(r["mutations"]))
                if key in muts:
                    with_scores.append(r["combined_score"])
                else:
                    without_scores.append(r["combined_score"])
            if with_scores and without_scores:
                median_with = sorted(with_scores)[len(with_scores) // 2]
                median_without = sorted(without_scores)[len(without_scores) // 2]
                cand.marginal_benefit = median_with - median_without
            elif with_scores:
                cand.marginal_benefit = sorted(with_scores)[len(with_scores) // 2] - median_all

        # 3. Pairwise interaction analysis for top candidates
        top_candidates = sorted(
            candidates.values(),
            key=lambda c: c.marginal_benefit,
            reverse=True,
        )[:_EPISTASIS_PAIR_LIMIT]

        for i, ca in enumerate(top_candidates):
            for cb in top_candidates[i + 1 :]:
                interaction = _compute_epistasis(
                    sorted_rows,
                    (ca.position, ca.amino_acid),
                    (cb.position, cb.amino_acid),
                )
                key_a = (ca.position, ca.amino_acid)
                key_b = (cb.position, cb.amino_acid)
                ca.interactions[key_b] = interaction
                cb.interactions[key_a] = interaction

        # 4. Confidence scoring: weighted combination of frequency, marginal
        # benefit, and epistatic interactions.
        # - freq_score: normalised frequency (capped at 1.0).
        # - marginal_score: marginal benefit scaled by _MARGINAL_SCALE and
        #   shifted by _MARGINAL_OFFSET so a neutral mutation maps to 0.5.
        # - _CONFIDENCE_W_BASELINE: a small constant so any candidate that
        #   passed frequency screening starts with non-zero confidence.
        # - antagonistic_penalty: sum of negative interaction terms scaled by
        #   _ANTAGONISTIC_SCALE — reduces confidence for mutations that clash
        #   with other anchors.
        for cand in candidates.values():
            freq_score = min(cand.frequency / anchor_threshold, 1.0)
            marginal_score = max(
                0.0,
                min(cand.marginal_benefit * _MARGINAL_SCALE + _MARGINAL_OFFSET, 1.0),
            )
            antagonistic_penalty = sum(min(v, 0.0) for v in cand.interactions.values()) * _ANTAGONISTIC_SCALE
            cand.confidence = max(
                0.0,
                _CONFIDENCE_W_FREQ * freq_score
                + _CONFIDENCE_W_MARGINAL * marginal_score
                + _CONFIDENCE_W_BASELINE
                + antagonistic_penalty,
            )

        # Filter to reasonable confidence
        result = [c for c in candidates.values() if c.confidence > 0.2]
        result.sort(key=lambda c: c.confidence, reverse=True)
        return result

    @staticmethod
    def _select_anchors_weighted(
        candidates: list[AnchorCandidate],
    ) -> dict[int, str]:
        """Select anchors using confidence-weighted probabilistic sampling."""
        if not candidates:
            return {}
        anchors: dict[int, str] = {}
        for cand in candidates:
            # Higher confidence → higher probability of being selected
            if random.random() < cand.confidence:
                # Don't overwrite if a higher-confidence anchor already occupies
                # this position
                if cand.position not in anchors:
                    anchors[cand.position] = cand.amino_acid
        return anchors

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
        groups: dict[int, list] = {}
        for m in sample:
            p = int(m.position)
            groups.setdefault(p, []).append(m)
        return [random.choice(options) for options in groups.values()]

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
                "stability_score",
                "nativeness_score",
                "aggregation_score",
                "charge_balance_score",
                "hydrophobic_core_score",
                "disulfide_score",
                "vhh_hallmark_score",
                "surface_hydrophobicity_score",
                "orthogonal_stability_score",
                "combined_score",
                "aa_sequence",
                "scoring_method",
            ]
        )
