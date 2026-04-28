"""Core module for generating and ranking VHH variant libraries."""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import random
import re
import time as _time
import traceback as _tb
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
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

# Batch scoring chunk sizes — keeps progress callbacks firing during
# long-running AbNatiV / ML scoring passes so the UI doesn't appear frozen.
_NATIVENESS_BATCH_CHUNK = 50
_STABILITY_BATCH_CHUNK = 100

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

# Per-operation timeout (seconds).  Operations that exceed this budget
# degrade gracefully — they return partial results instead of hanging.
_OPERATION_TIMEOUT_SECONDS: int = 300  # 5 minutes

# Length of the truncated sequence used for backend health check probes.
# Short enough to be fast, long enough to exercise the model pipeline.
_HEALTH_CHECK_SEQ_LENGTH: int = 50


class OperationTimeoutError(Exception):
    """Raised when a library generation sub-operation exceeds its time budget."""


@contextlib.contextmanager
def _timed_operation(description: str, progress_callback: object | None = None):  # noqa: ARG001
    """Log wall-clock time for an operation.  Always prints to stdout.

    Parameters
    ----------
    description : str
        Human-readable label for the operation being timed.
    progress_callback : object | None
        Reserved for future use — not currently consumed but accepted
        so callers can forward their callback without extra logic.
    """
    start = _time.monotonic()
    print(f"[TIMING] START: {description}", flush=True)
    logger.info("[TIMING] START: %s", description)
    try:
        yield
    finally:
        elapsed = _time.monotonic() - start
        msg = f"[TIMING] DONE: {description} — {elapsed:.1f}s"
        print(msg, flush=True)
        logger.info(msg)


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


def _mutation_entropy(rows: list[dict], parsed_mutations: list[set[tuple[int, str]]] | None = None) -> float:
    """Compute Shannon entropy of mutation frequencies across all rows.

    Higher entropy indicates more diverse variants (more unique mutation
    combinations).  Returns 0.0 for empty input.

    Parameters
    ----------
    parsed_mutations : list[set[tuple[int, str]]] | None
        Pre-parsed mutation sets aligned with *rows*.  When provided,
        avoids re-parsing the ``"mutations"`` string for each row.
    """
    if not rows:
        return 0.0
    counts: Counter[tuple[int, str]] = Counter()
    total = 0
    for i, r in enumerate(rows):
        muts = parsed_mutations[i] if parsed_mutations is not None else _parse_mut_str(r["mutations"])
        for pos_aa in muts:
            counts[pos_aa] += 1
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
    parsed_mutations: list[set[tuple[int, str]]] | None = None,
) -> float:
    """Compute epistatic interaction score between two mutations.

    interaction = score(A+B) - score(A_only) - score(B_only) + score(neither)

    Positive → synergistic; Negative → antagonistic.

    Parameters
    ----------
    parsed_mutations : list[set[tuple[int, str]]] | None
        Pre-parsed mutation sets aligned with *rows*.  When provided,
        avoids re-parsing the ``"mutations"`` string for each row.
    """
    scores_ab: list[float] = []
    scores_a: list[float] = []
    scores_b: list[float] = []
    scores_none: list[float] = []

    for i, r in enumerate(rows):
        muts = parsed_mutations[i] if parsed_mutations is not None else set(_parse_mut_str(r["mutations"]))
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
        operation_timeout: int | None = None,
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

        # Per-operation timeout in seconds.  None (the default) means no
        # timeout — operations run to completion regardless of wall-clock time.
        # Pass an explicit integer to re-enable the old 300-second safety net.
        self._operation_timeout: int | None = operation_timeout

    # ------------------------------------------------------------------
    # Timeout helpers
    # ------------------------------------------------------------------

    def _timeout_expired(self, start: float) -> bool:
        """Return ``True`` if the operation timeout has been exceeded.

        When ``self._operation_timeout`` is ``None`` (the default), this
        always returns ``False`` — i.e. no timeout is enforced.
        """
        if self._operation_timeout is None:
            return False
        return _time.monotonic() - start > self._operation_timeout

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
        scores = self._score_variant_without_nativeness(vhh)

        nat = self._nativeness_scorer.score(vhh)
        scores["nativeness"] = nat["composite_score"]

        return scores

    def _score_variant_without_nativeness(self, vhh: VHHSequence, *, _skip_ml: bool = False) -> dict[str, float]:
        """Score a variant on all axes *except* nativeness.

        Nativeness scoring via AbNatiV is expensive (ANARCI alignment per
        call) so library-generation strategies batch it separately using
        :meth:`_batch_fill_nativeness`.  This helper provides the
        remaining cheap/moderate scores.

        Parameters
        ----------
        _skip_ml : bool
            When ``True``, tell the stability scorer to skip expensive ML
            backends (ESM-2, NanoMelt) and use only heuristic sub-scores.
            Library generation passes ``True`` here and batch-scores the
            ML axes afterward via :meth:`_batch_fill_stability`.
        """
        stab = self._stability_scorer.score(vhh, _skip_ml=_skip_ml)

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

        if "nanomelt_tm" in stab:
            scores["nanomelt_tm"] = stab["nanomelt_tm"]

        if self._enabled_metrics.get("surface_hydrophobicity", False):
            sh = self.hydrophobicity_scorer.score(vhh)
            scores["surface_hydrophobicity"] = sh["composite_score"]
        else:
            scores["surface_hydrophobicity"] = 0.0

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
        progress_callback: Optional[Callable[[IterativeProgress], None]] = None,
    ) -> list[dict]:
        """Generate candidate mutations ranked by stability impact.

        For each mutable position (respecting off-limits, forbidden
        substitutions, and excluded AAs), all 19 possible substitutions are
        evaluated for stability.  Mutations introducing PTM liabilities are
        filtered out.

        When NanoMelt or ESM-2 backends are available, stability is
        batch-scored via ML after collecting all candidates.  Otherwise
        the fast heuristic path is used per candidate.

        The ``off_limits`` set is the sole authority on position mutability —
        CDR positions are only skipped when they appear in ``off_limits``
        (as determined by the user's interactive selections).

        Nativeness scoring is performed in a single batch call rather than
        per-candidate to avoid repeated ANARCI re-alignment overhead inside
        AbNatiV.
        """
        parent_seq = vhh_sequence.sequence
        forbidden_str: dict[str, set[str]] = {}
        if forbidden_substitutions:
            forbidden_str = {str(k): v for k, v in forbidden_substitutions.items()}
        excluded = excluded_target_aas or set()

        # Count mutable positions for progress estimation
        n_mutable = sum(1 for pos_key in vhh_sequence.imgt_numbered if pos_key not in off_limits)
        n_aas = len(AMINO_ACIDS) - 1  # 19 substitutions per position
        estimated_total = n_mutable * n_aas
        print(
            f"[RANKING] Scanning {n_mutable} mutable positions × {n_aas} AAs = ~{estimated_total} candidates",
            flush=True,
        )

        candidates: list[dict] = []
        mutant_sequences: list[str] = []

        for pos_key, original_aa in vhh_sequence.imgt_numbered.items():
            if pos_key in off_limits:
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

                # Heuristic delta is computed for all candidates as a
                # fast baseline.  When ML backends are available it will
                # be replaced by batch ML scoring below.
                delta_stab = self._stability_scorer.predict_mutation_effect(
                    vhh_sequence, pos_key, candidate_aa, _skip_ml=True
                )

                candidate_dict: dict = {
                    "position": pos_key,
                    "original_aa": original_aa,
                    "suggested_aa": candidate_aa,
                    "delta_stability": delta_stab,
                    "reason": "Stability-driven scan",
                }

                candidates.append(candidate_dict)
                mutant_sequences.append(mutant.sequence)

                # Progress every 100 enumerated candidates
                if len(candidates) % 100 == 0:
                    print(
                        f"[RANKING] Enumerated {len(candidates)}/{estimated_total} candidates…",
                        flush=True,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            IterativeProgress(
                                phase="enumerating_candidates",
                                round_number=len(candidates),
                                total_rounds=estimated_total,
                                best_score=0.0,
                                mean_score=0.0,
                                population_size=len(candidates),
                                n_anchors=0,
                                diversity_entropy=0.0,
                                message=f"Scanning mutations: {len(candidates)}/{estimated_total} candidates…",
                            )
                        )

        print(f"[RANKING] Enumeration complete: {len(candidates)} candidates generated", flush=True)

        # --- Batch ML stability scoring ---
        # When NanoMelt or ESM-2 is available, batch-score all candidate
        # mutant sequences and replace delta_stability with ML-derived
        # values.  This uses pure ML Tm signals (no heuristic penalty/bonus).
        if candidates:
            if progress_callback is not None:
                progress_callback(
                    IterativeProgress(
                        phase="batch_stability_scoring",
                        round_number=0,
                        total_rounds=1,
                        best_score=0.0,
                        mean_score=0.0,
                        population_size=len(candidates),
                        n_anchors=0,
                        diversity_entropy=0.0,
                        message=f"Batch-scoring stability for {len(candidates)} candidates…",
                    )
                )
            with _timed_operation(f"_batch_rescore_candidates ({len(candidates)} candidates)"):
                self._batch_rescore_candidates(candidates, mutant_sequences, vhh_sequence)

        # Batch-score nativeness for all mutants in a single call instead of
        # invoking AbNatiV (and its internal ANARCI alignment) per candidate.
        if candidates:
            if progress_callback is not None:
                progress_callback(
                    IterativeProgress(
                        phase="batch_nativeness_scoring",
                        round_number=0,
                        total_rounds=1,
                        best_score=0.0,
                        mean_score=0.0,
                        population_size=len(candidates),
                        n_anchors=0,
                        diversity_entropy=0.0,
                        message=f"Batch-scoring nativeness for {len(candidates)} candidates…",
                    )
                )
            _nat_start = _time.monotonic()
            try:
                with _timed_operation("nativeness parent scoring"):
                    parent_nat = self._nativeness_scorer.score(vhh_sequence)["composite_score"]
                if hasattr(self._nativeness_scorer, "score_batch"):
                    print(
                        f"[RANKING] Batch-scoring nativeness for {len(mutant_sequences)} sequences…",
                        flush=True,
                    )
                    with _timed_operation(f"nativeness batch scoring ({len(mutant_sequences)} sequences)"):
                        mutant_nat_scores = self._nativeness_scorer.score_batch(mutant_sequences)
                        if self._timeout_expired(_nat_start):
                            print(
                                f"[TIMEOUT] nativeness batch scoring exceeded "
                                f"{self._operation_timeout}s — setting delta_nativeness=0.0",
                                flush=True,
                            )
                            logger.warning(
                                "Nativeness batch scoring exceeded timeout (%ds); "
                                "setting delta_nativeness=0.0 for all candidates",
                                self._operation_timeout,
                            )
                            for candidate in candidates:
                                candidate["delta_nativeness"] = 0.0
                        else:
                            print("[RANKING] Nativeness batch scoring complete", flush=True)
                            for candidate, mutant_nat in zip(candidates, mutant_nat_scores):
                                candidate["delta_nativeness"] = mutant_nat - parent_nat
                else:
                    # Fallback for scorers that lack batch support — use delta directly.
                    print(
                        "[RANKING] No score_batch — falling back to per-candidate nativeness",
                        flush=True,
                    )
                    for i, candidate in enumerate(candidates):
                        if i % 50 == 0:
                            print(
                                f"[RANKING] Per-candidate nativeness: {i}/{len(candidates)}",
                                flush=True,
                            )
                        if self._timeout_expired(_nat_start):
                            print(
                                f"[TIMEOUT] per-candidate nativeness scoring exceeded "
                                f"{self._operation_timeout}s at candidate {i} — "
                                f"setting delta_nativeness=0.0 for remaining",
                                flush=True,
                            )
                            logger.warning(
                                "Per-candidate nativeness scoring exceeded timeout (%ds) at "
                                "candidate %d/%d; setting delta_nativeness=0.0 for remaining",
                                self._operation_timeout,
                                i,
                                len(candidates),
                            )
                            for remaining in candidates[i:]:
                                remaining.setdefault("delta_nativeness", 0.0)
                            break
                        candidate["delta_nativeness"] = self._nativeness_scorer.predict_mutation_effect(
                            vhh_sequence, candidate["position"], candidate["suggested_aa"]
                        )
            except Exception as exc:
                print(f"[RANKING] Nativeness scoring FAILED: {exc}", flush=True)

                _tb.print_exc()
                logger.warning("Nativeness scoring failed; setting delta_nativeness=0.0: %s", exc)
                for candidate in candidates:
                    candidate.setdefault("delta_nativeness", 0.0)

        candidates.sort(key=lambda c: c["delta_stability"], reverse=True)
        return candidates

    def _batch_rescore_candidates(
        self,
        candidates: list[dict],
        mutant_sequences: list[str],
        parent_vhh: VHHSequence,
    ) -> None:
        """Batch-score candidate mutations with ML backends and update delta_stability.

        When NanoMelt or ESM-2 is available, this method computes the parent
        ML score once and batch-scores all mutant sequences, then replaces
        each candidate's ``delta_stability`` with the ML-derived delta.

        Falls back silently (keeping heuristic deltas) if ML scoring fails.
        """
        from vhh_library.stability import _sigmoid_normalize

        scorer = self._stability_scorer
        has_nanomelt = scorer.nanomelt_predictor is not None
        has_esm = scorer.esm_scorer is not None

        if not has_nanomelt and not has_esm:
            return  # No ML backends — keep heuristic deltas

        _rescore_start = _time.monotonic()

        # --- NanoMelt batch scoring (preferred) ---
        if has_nanomelt:
            try:
                predictor = scorer.nanomelt_predictor
                print("[RANKING] NanoMelt batch rescoring: scoring parent…", flush=True)

                # Score parent
                with _timed_operation("NanoMelt parent scoring"):
                    parent_result = predictor.score_sequence(parent_vhh)
                parent_tm = parent_result["nanomelt_tm"]
                parent_score = _sigmoid_normalize(parent_tm, scorer._tm_min, scorer._tm_max)
                print(f"[RANKING] NanoMelt parent Tm = {parent_tm:.1f}°C", flush=True)

                # Check timeout before batch scoring
                if self._timeout_expired(_rescore_start):
                    print(
                        f"[TIMEOUT] _batch_rescore_candidates exceeded {self._operation_timeout}s "
                        f"after parent scoring — keeping heuristic deltas",
                        flush=True,
                    )
                    return

                print(
                    f"[RANKING] NanoMelt batch rescoring: {len(mutant_sequences)} mutants…",
                    flush=True,
                )

                # Batch-score all mutant sequences using the pre-aligned
                # path (skips redundant ANARCI for point-mutation variants).
                with _timed_operation(f"NanoMelt batch scoring ({len(mutant_sequences)} mutants)"):
                    if hasattr(predictor, "score_batch_prealigned"):
                        mutant_results = predictor.score_batch_prealigned(parent_vhh.sequence, mutant_sequences)
                    else:
                        # Fallback: construct lightweight VHHSequence wrappers —
                        # score_batch only accesses .sequence on each object,
                        # matching the pattern used in _batch_fill_stability.
                        mutant_objects: list[VHHSequence] = []
                        for seq in mutant_sequences:
                            obj = object.__new__(VHHSequence)
                            obj.sequence = seq
                            mutant_objects.append(obj)
                        mutant_results = predictor.score_batch(mutant_objects)

                # Check timeout after batch scoring
                if self._timeout_expired(_rescore_start):
                    print(
                        f"[TIMEOUT] _batch_rescore_candidates exceeded {self._operation_timeout}s "
                        f"after NanoMelt batch scoring — keeping heuristic deltas",
                        flush=True,
                    )
                    return

                for candidate, nm_result in zip(candidates, mutant_results):
                    mutant_tm = nm_result["nanomelt_tm"]
                    mutant_score = _sigmoid_normalize(mutant_tm, scorer._tm_min, scorer._tm_max)
                    candidate["delta_stability"] = mutant_score - parent_score

                print("[RANKING] NanoMelt batch rescoring complete", flush=True)
                logger.info(
                    "Batch-scored %d candidates with NanoMelt (parent Tm=%.1f°C)",
                    len(candidates),
                    parent_tm,
                )
                return
            except Exception as exc:
                print(f"[RANKING] NanoMelt batch rescoring FAILED: {exc}", flush=True)
                _tb.print_exc()
                logger.warning(
                    "NanoMelt batch candidate scoring failed; %s: %s",
                    "falling back to ESM-2" if has_esm else "keeping heuristic deltas",
                    exc,
                )

        # --- ESM-2 batch scoring (fallback) ---
        if has_esm:
            try:
                esm = scorer.esm_scorer
                parent_seq = parent_vhh.sequence
                print(
                    f"[RANKING] ESM-2 batch rescoring: {len(mutant_sequences) + 1} sequences (parent + mutants)…",
                    flush=True,
                )

                # Batch-score parent + all mutants together
                with _timed_operation(f"ESM-2 batch scoring ({len(mutant_sequences)} mutants)"):
                    all_sequences = [parent_seq] + mutant_sequences
                    all_plls = esm.score_batch(all_sequences)

                # Check timeout after ESM-2 batch scoring
                if self._timeout_expired(_rescore_start):
                    print(
                        f"[TIMEOUT] _batch_rescore_candidates exceeded {self._operation_timeout}s "
                        f"after ESM-2 batch scoring — keeping heuristic deltas",
                        flush=True,
                    )
                    return

                parent_pll = all_plls[0]
                parent_tm = scorer._pll_to_predicted_tm(parent_pll, len(parent_seq))
                parent_score = _sigmoid_normalize(parent_tm, scorer._tm_min, scorer._tm_max)

                for candidate, pll in zip(candidates, all_plls[1:]):
                    mutant_tm = scorer._pll_to_predicted_tm(pll, len(parent_seq))
                    mutant_score = _sigmoid_normalize(mutant_tm, scorer._tm_min, scorer._tm_max)
                    candidate["delta_stability"] = mutant_score - parent_score

                print("[RANKING] ESM-2 batch rescoring complete", flush=True)
                logger.info(
                    "Batch-scored %d candidates with ESM-2 (parent Tm=%.1f°C)",
                    len(candidates),
                    parent_tm,
                )
                return
            except Exception as exc:
                print(f"[RANKING] ESM-2 batch rescoring FAILED: {exc}", flush=True)
                _tb.print_exc()
                logger.warning("ESM-2 batch candidate scoring failed; keeping heuristic deltas: %s", exc)

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
        progress_callback: Optional[Callable[[IterativeProgress], None]] = None,
    ) -> pd.DataFrame:
        _rank_start = _time.monotonic()

        # ---- DIAGNOSTIC PREAMBLE ---- prints to both logger AND stdout
        _diag_lines = [
            "=" * 70,
            "RANK SINGLE MUTATIONS STARTED",
            f"  Timestamp: {_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  vhh_sequence length: {len(vhh_sequence.sequence)}",
            f"  vhh_sequence valid: {vhh_sequence.validation_result.get('valid')}",
            f"  vhh_sequence numbered positions: {len(vhh_sequence.imgt_numbered)}",
            f"  off_limits count: {len(off_limits) if off_limits else 0}",
            f"  max_per_position: {max_per_position}",
            f"  Stability scorer: {type(self._stability_scorer).__name__}",
            f"  Has NanoMelt: {self._stability_scorer.nanomelt_predictor is not None}",
            f"  Has ESM on stability: {self._stability_scorer.esm_scorer is not None}",
            f"  Nativeness scorer: {type(self._nativeness_scorer).__name__}",
            f"  Active weights: {self._active_weights()}",
            f"  progress_callback is None: {progress_callback is None}",
            "=" * 70,
        ]
        for line in _diag_lines:
            print(line, flush=True)
            logger.info(line)

        # ---- BACKEND HEALTH CHECK ---- probe each backend before ranking
        if self._stability_scorer.nanomelt_predictor is not None:
            try:
                with _timed_operation("NanoMelt health check (rank)"):
                    _nm_health_result = self._stability_scorer.nanomelt_predictor.score_sequence(vhh_sequence)
                print(
                    f"[RANKING] NanoMelt health check OK (Tm={_nm_health_result.get('nanomelt_tm', '?')})",
                    flush=True,
                )
                logger.info("[RANKING] NanoMelt health check OK (Tm=%s)", _nm_health_result.get("nanomelt_tm", "?"))
            except Exception as exc:
                print(f"[RANKING] NanoMelt health check FAILED: {exc}", flush=True)
                print("[RANKING] Disabling NanoMelt for this ranking run", flush=True)
                _tb.print_exc()
                logger.warning("[RANKING] NanoMelt health check failed; disabling for this run: %s", exc)
                self._stability_scorer.nanomelt_predictor = None

        if self._stability_scorer.esm_scorer is not None:
            try:
                _probe = vhh_sequence.sequence[:_HEALTH_CHECK_SEQ_LENGTH]
                with _timed_operation("ESM-2 health check (rank)"):
                    _esm_health_check_pll = self._stability_scorer.esm_scorer.score_batch([_probe])
                print("[RANKING] ESM-2 health check OK", flush=True)
                logger.info("[RANKING] ESM-2 health check OK")
            except Exception as exc:
                print(f"[RANKING] ESM-2 health check FAILED: {exc}", flush=True)
                _tb.print_exc()
                logger.warning("[RANKING] ESM-2 health check failed; disabling for this run: %s", exc)
                self._stability_scorer.esm_scorer = None

        try:
            with _timed_operation("Nativeness health check (rank)"):
                self._nativeness_scorer.score(vhh_sequence)
            print("[RANKING] Nativeness health check OK", flush=True)
            logger.info("[RANKING] Nativeness health check OK")
        except Exception as exc:
            print(f"[RANKING] Nativeness health check FAILED: {exc}", flush=True)
            _tb.print_exc()
            # Cannot disable nativeness — it's required. Log and continue.
            logger.warning("[RANKING] Nativeness health check failed (continuing): %s", exc)

        if off_limits is None:
            # CDR positions are frozen by default — the mutation engine must
            # not propose mutations inside CDR loops unless explicitly
            # overridden via the off_limits parameter.
            off_limits = vhh_sequence.cdr_positions

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
            progress_callback=progress_callback,
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

        if progress_callback is not None:
            progress_callback(
                IterativeProgress(
                    phase="ranking_complete",
                    round_number=1,
                    total_rounds=1,
                    best_score=float(df["combined_score"].max()) if not df.empty else 0.0,
                    mean_score=float(df["combined_score"].mean()) if not df.empty else 0.0,
                    population_size=len(df),
                    n_anchors=0,
                    diversity_entropy=0.0,
                    message=f"Ranking complete: {len(df)} mutations ranked.",
                )
            )

        _rank_elapsed = _time.monotonic() - _rank_start
        _summary = [
            "=" * 70,
            "RANK SINGLE MUTATIONS COMPLETE",
            f"  Total time: {_rank_elapsed:.1f}s",
            f"  Candidates generated: {len(suggestions)}",
            f"  Final ranked mutations: {len(df)}",
            "  Scoring method breakdown: " + str(df["reason"].value_counts().to_dict() if not df.empty else "N/A"),
            "=" * 70,
        ]
        for line in _summary:
            print(line, flush=True)
            logger.info(line)

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
        checkpoint_dir: Optional[Path] = None,
        assembly_mode: str | None = None,
        split_position: str | None = None,
        overlap_n_boundary: str = "56",
        overlap_c_boundary: str = "66",
    ) -> pd.DataFrame:
        """Generate a scored variant library from *top_mutations*.

        Parameters
        ----------
        checkpoint_dir:
            When set, intermediate checkpoints are written to this directory
            during iterative library generation.  If a prior checkpoint for
            the same run exists, generation resumes from it.  Defaults to
            ``None`` (no checkpointing) for backward compatibility.
        assembly_mode:
            ``None`` for legacy single-construct mode (default).
            ``"two_part"`` for yeast surface display two-part assembly mode.
        split_position:
            IMGT position string key defining the boundary between Part 1
            and Part 2.  Required when ``assembly_mode="two_part"``.
        overlap_n_boundary:
            IMGT position of the first (N-terminal) residue in the overlap
            region for PCR fusion homology. Default is ``"56"``.
        overlap_c_boundary:
            IMGT position of the last (C-terminal) residue in the overlap
            region for PCR fusion homology. Default is ``"66"``.
        """
        _gen_start = _time.monotonic()

        # ---- DIAGNOSTIC PREAMBLE ---- prints to both logger AND stdout
        _diag_lines = [
            "=" * 70,
            "LIBRARY GENERATION STARTED",
            f"  Timestamp: {_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Strategy: {strategy}",
            f"  n_mutations: {n_mutations}, min_mutations: {min_mutations}",
            f"  max_variants: {max_variants}, max_rounds: {max_rounds}",
            f"  top_mutations shape: {top_mutations.shape}",
            f"  vhh_sequence length: {len(vhh_sequence.sequence)}",
            f"  vhh_sequence valid: {vhh_sequence.validation_result.get('valid')}",
            f"  vhh_sequence numbered positions: {len(vhh_sequence.imgt_numbered)}",
            f"  Stability scorer: {type(self._stability_scorer).__name__}",
            f"  Has ESM scorer: {self._esm_scorer is not None}",
            f"  Has NanoMelt: {self._stability_scorer.nanomelt_predictor is not None}",
            f"  Has ESM on stability scorer: {self._stability_scorer.esm_scorer is not None}",
            f"  Nativeness scorer: {type(self._nativeness_scorer).__name__}",
            f"  Active weights: {self._active_weights()}",
            f"  progress_callback is None: {progress_callback is None}",
            "=" * 70,
        ]
        for line in _diag_lines:
            print(line, flush=True)
            logger.info(line)

        # ---- BACKEND HEALTH CHECK ---- probe each backend before generation
        _backend_status: dict[str, str] = {}
        _probe_seq = vhh_sequence.sequence[:_HEALTH_CHECK_SEQ_LENGTH]
        if self._stability_scorer.nanomelt_predictor is not None:
            try:
                _test_result = self._stability_scorer.nanomelt_predictor.score_sequence(vhh_sequence)
                _backend_status["nanomelt"] = f"OK (Tm={_test_result.get('nanomelt_tm', '?')})"
            except Exception as _exc:
                _backend_status["nanomelt"] = f"FAILED: {_exc}"
                print(f"[DIAGNOSTIC] NanoMelt health check FAILED: {_exc}", flush=True)
                logger.warning("NanoMelt health check failed; disabling for this run: %s", _exc)
                self._stability_scorer.nanomelt_predictor = None

        if self._stability_scorer.esm_scorer is not None:
            try:
                _test_pll = self._stability_scorer.esm_scorer.score_batch([_probe_seq])
                _backend_status["esm2"] = f"OK (pll={_test_pll[0]:.4f})" if _test_pll else "OK (empty)"
            except Exception as _exc:
                _backend_status["esm2"] = f"FAILED: {_exc}"
                print(f"[DIAGNOSTIC] ESM-2 health check FAILED: {_exc}", flush=True)
                logger.warning("ESM-2 health check failed; disabling for this run: %s", _exc)
                self._stability_scorer.esm_scorer = None

        if hasattr(self._nativeness_scorer, "score_batch"):
            try:
                _test_nat = self._nativeness_scorer.score_batch([_probe_seq])
                _backend_status["nativeness"] = f"OK (score={_test_nat[0]:.4f})" if _test_nat else "OK (empty)"
            except Exception as _exc:
                _backend_status["nativeness"] = f"FAILED: {_exc}"
                print(f"[DIAGNOSTIC] Nativeness health check FAILED: {_exc}", flush=True)
                logger.warning("Nativeness health check failed: %s", _exc)

        for _bname, _bstatus in _backend_status.items():
            print(f"[DIAGNOSTIC] Backend {_bname}: {_bstatus}", flush=True)
            logger.info("[DIAGNOSTIC] Backend %s: %s", _bname, _bstatus)

        if top_mutations.empty:
            return self._empty_library_df()

        # --- Two-part assembly dispatch ---
        if assembly_mode == "two_part":
            if split_position is None:
                raise ValueError("split_position is required when assembly_mode='two_part'")
            return self._generate_two_part_library(
                vhh_sequence=vhh_sequence,
                top_mutations=top_mutations,
                n_mutations=n_mutations,
                max_variants=max_variants,
                min_mutations=min_mutations,
                strategy=strategy,
                anchor_threshold=anchor_threshold,
                max_rounds=max_rounds,
                rescore_top_n=rescore_top_n,
                progress_callback=progress_callback,
                checkpoint_dir=checkpoint_dir,
                split_position=split_position,
                overlap_n_boundary=overlap_n_boundary,
                overlap_c_boundary=overlap_c_boundary,
            )

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

        # Compute checkpoint run_id when checkpointing is enabled.
        _run_id: str | None = None
        if checkpoint_dir is not None:
            from vhh_library.checkpoint import compute_run_id

            _run_id = compute_run_id(
                vhh_sequence.sequence,
                n_mutations=n_mutations,
                max_variants=max_variants,
                min_mutations=min_mutations,
                strategy=strategy,
            )

        # Progress helper for non-iterative strategies.  The iterative
        # strategy has its own internal ``_report`` function with richer
        # phase semantics; these high-level reports cover the phases that
        # bookend all strategies (variant generation, batch scoring, etc.).
        has_ml_stability = (
            self._stability_scorer.nanomelt_predictor is not None or self._stability_scorer.esm_scorer is not None
        )
        has_esm_progressive = self._esm_scorer is not None

        def _report_progress(
            phase: str,
            step: int,
            total_steps: int,
            n_variants: int = 0,
            message: str = "",
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                IterativeProgress(
                    phase=phase,
                    round_number=step,
                    total_rounds=total_steps,
                    best_score=0.0,
                    mean_score=0.0,
                    population_size=n_variants,
                    n_anchors=0,
                    diversity_entropy=0.0,
                    message=message,
                )
            )

        if strategy == "exhaustive":
            # Steps: generate → (batch stability) → batch nativeness → (ESM-2)
            total_steps = 2 + (1 if has_ml_stability else 0) + (1 if has_esm_progressive else 0)
            step = 1
            _report_progress(
                "generating_variants",
                step,
                total_steps,
                message=f"Building exhaustive combinations ({total:,} max)…",
            )
            with _timed_operation("exhaustive variant generation", progress_callback):
                rows = self._generate_exhaustive(
                    vhh_sequence,
                    mutation_list,
                    k_min,
                    k_max,
                    max_variants,
                    position_groups=position_groups,
                    _batch_score=False,
                    progress_callback=progress_callback,
                )
            if has_ml_stability:
                step += 1
                _report_progress(
                    "scoring_stability",
                    step,
                    total_steps,
                    len(rows),
                    message=f"Scoring stability for {len(rows):,} variants…",
                )
                with _timed_operation(f"batch stability scoring ({len(rows)} variants)", progress_callback):
                    rows = self._batch_fill_stability(rows, progress_callback=progress_callback)
            step += 1
            _report_progress(
                "scoring_nativeness",
                step,
                total_steps,
                len(rows),
                message=f"Scoring nativeness for {len(rows):,} variants…",
            )
            with _timed_operation(f"batch nativeness scoring ({len(rows)} variants)", progress_callback):
                rows = self._batch_fill_nativeness(rows, progress_callback=progress_callback, vhh_sequence=vhh_sequence)
        elif strategy == "random":
            total_steps = 2 + (1 if has_ml_stability else 0) + (1 if has_esm_progressive else 0)
            step = 1
            _report_progress(
                "generating_variants",
                step,
                total_steps,
                message=f"Sampling up to {max_variants:,} random variants…",
            )
            with _timed_operation("random variant sampling", progress_callback):
                rows = self._generate_sampled(
                    vhh_sequence,
                    mutation_list,
                    k_min,
                    k_max,
                    max_variants,
                    _batch_score=False,
                    progress_callback=progress_callback,
                )
            if has_ml_stability:
                step += 1
                _report_progress(
                    "scoring_stability",
                    step,
                    total_steps,
                    len(rows),
                    message=f"Scoring stability for {len(rows):,} variants…",
                )
                with _timed_operation(f"batch stability scoring ({len(rows)} variants)", progress_callback):
                    rows = self._batch_fill_stability(rows, progress_callback=progress_callback)
            step += 1
            _report_progress(
                "scoring_nativeness",
                step,
                total_steps,
                len(rows),
                message=f"Scoring nativeness for {len(rows):,} variants…",
            )
            with _timed_operation(f"batch nativeness scoring ({len(rows)} variants)", progress_callback):
                rows = self._batch_fill_nativeness(rows, progress_callback=progress_callback, vhh_sequence=vhh_sequence)
        elif strategy == "iterative":
            with _timed_operation("iterative strategy (all phases)", progress_callback):
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
                    checkpoint_dir=checkpoint_dir,
                    run_id=_run_id,
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        # ESM-2 progressive scoring (when scorer is available).
        # Reports sub-step phases so the UI can show progress during
        # the multi-stage score_library_progressive operation.
        if self._esm_scorer is not None and not df.empty:
            _report_progress(
                "esm2_scoring_stage1",
                1,
                2,
                len(df),
                message=f"ESM-2 delta-PLL scoring for {len(df):,} variants…",
            )
            with _timed_operation(f"ESM-2 progressive scoring ({len(df)} variants)", progress_callback):
                df = self._esm_scorer.score_library_progressive(vhh_sequence, df)
            _report_progress(
                "esm2_scoring_stage2",
                2,
                2,
                len(df),
                message=f"ESM-2 progressive scoring complete ({len(df):,} variants)",
            )

        # Persist final result and clean up intermediate checkpoint.
        if checkpoint_dir is not None and _run_id is not None:
            from vhh_library.checkpoint import remove_checkpoint, save_result

            save_result(checkpoint_dir, _run_id, df)
            remove_checkpoint(checkpoint_dir, _run_id)

        _gen_elapsed = _time.monotonic() - _gen_start
        print(f"[TIMING] DONE: generate_library total — {_gen_elapsed:.1f}s ({len(df)} variants)", flush=True)
        logger.info("[TIMING] DONE: generate_library total — %.1fs (%d variants)", _gen_elapsed, len(df))

        return df

    # ------------------------------------------------------------------
    # Private: build a single variant row
    # ------------------------------------------------------------------

    def _build_variant_row(
        self,
        vhh_sequence: VHHSequence,
        selected: list,
        variant_counter: int,
        *,
        nativeness_score: float | None = None,
        _skip_ml: bool = False,
    ) -> dict:
        """Build a single library-variant row.

        Parameters
        ----------
        nativeness_score : float | None
            Pre-computed nativeness score for the variant.  When provided
            the expensive per-variant AbNatiV call is skipped — callers
            should supply this from a batch :meth:`score_batch` call.
            When *None*, nativeness is scored individually (legacy path).
        _skip_ml : bool
            When ``True``, skip expensive ML stability backends (ESM-2,
            NanoMelt) during per-variant scoring.  Library generation
            passes ``True`` here and batch-scores the ML axes afterward
            via :meth:`_batch_fill_stability`.
        """
        mutations: list[tuple[int, str]] = [(int(m.position), m.suggested_aa) for m in selected]
        mut_labels = [f"{m.original_aa}{m.position}{m.suggested_aa}" for m in selected]

        logger.debug("Building variant V%06d with %d mutations", variant_counter, len(mutations))

        # Use fast-path VHHSequence.mutate() to avoid redundant ANARCI calls.
        current = vhh_sequence
        for imgt_pos, new_aa in mutations:
            current = VHHSequence.mutate(current, str(imgt_pos), new_aa)
        mutant_seq = current.sequence

        if nativeness_score is not None:
            raw = self._score_variant_without_nativeness(current, _skip_ml=_skip_ml)
            raw["nativeness"] = nativeness_score
        else:
            raw = self._score_variant(current)

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

        if "nanomelt_tm" in raw:
            row["nanomelt_tm"] = raw["nanomelt_tm"]

        return row

    # ------------------------------------------------------------------
    # Private: batch nativeness scoring for library rows
    # ------------------------------------------------------------------

    def _batch_fill_nativeness(
        self,
        rows: list[dict],
        progress_callback: Callable[[IterativeProgress], None] | None = None,
        vhh_sequence: VHHSequence | None = None,
    ) -> list[dict]:
        """Batch-score nativeness for library rows and recompute combined scores.

        This replaces per-variant AbNatiV calls with a single batch call,
        avoiding repeated ANARCI alignment overhead that caused Streamlit
        timeouts during library generation.

        When *vhh_sequence* is provided, the scorer's pre-aligned path is
        used: the parent is aligned once via ANARCI and all variants reuse
        that alignment with ``do_align=False``, reducing alignment cost
        from O(n) to O(1).

        Identical sequences are deduplicated before scoring so that the
        same variant is never sent to AbNatiV twice.

        Sequences are scored in chunks (default 50) so that the progress
        callback fires between chunks, providing visible feedback during
        long-running AbNatiV scoring runs.

        Parameters
        ----------
        progress_callback : Callable[[IterativeProgress], None] | None
            Optional callback fired before, during, and after the batch
            scoring operation.
        vhh_sequence : VHHSequence | None
            Parent/wild-type sequence.  When provided, enables the
            pre-aligned scoring fast path that bypasses per-variant ANARCI.
        """
        if not rows:
            return rows

        n_seqs = len(rows)

        def _report(phase: str, message: str, *, step: int = 0, total: int = 0) -> None:
            if progress_callback is None:
                return
            progress_callback(
                IterativeProgress(
                    phase=phase,
                    round_number=step,
                    total_rounds=total,
                    best_score=0.0,
                    mean_score=0.0,
                    population_size=n_seqs,
                    n_anchors=0,
                    diversity_entropy=0.0,
                    message=message,
                )
            )

        sequences = [r["aa_sequence"] for r in rows]

        # ------------------------------------------------------------------
        # Deduplicate sequences — avoid scoring identical variants twice.
        # ------------------------------------------------------------------
        unique_seqs: list[str] = []
        seq_to_index: dict[str, int] = {}
        row_to_unique: list[int] = []
        for seq in sequences:
            if seq not in seq_to_index:
                seq_to_index[seq] = len(unique_seqs)
                unique_seqs.append(seq)
            row_to_unique.append(seq_to_index[seq])

        n_unique = len(unique_seqs)
        dedup_saved = n_seqs - n_unique
        if dedup_saved > 0:
            logger.info(
                "Deduplication: %d unique sequences from %d total (saved %d AbNatiV calls)",
                n_unique,
                n_seqs,
                dedup_saved,
            )

        # ------------------------------------------------------------------
        # Choose scoring path: pre-aligned (fast) or standard (fallback).
        # ------------------------------------------------------------------
        parent_seq = vhh_sequence.sequence if vhh_sequence is not None else None
        use_prealigned = vhh_sequence is not None and hasattr(self._nativeness_scorer, "score_batch_prealigned")

        if use_prealigned:
            logger.info(
                "Using pre-aligned scoring path (%d unique sequences, parent length %d)",
                n_unique,
                len(parent_seq),
            )
            _report(
                "scoring_nativeness_start",
                f"Scoring nativeness — pre-aligned ({n_unique:,} unique sequences)…",
                step=1,
                total=1,
            )
            try:
                unique_scores = self._nativeness_scorer.score_batch_prealigned(
                    parent_seq,
                    unique_seqs,
                )
            except Exception:
                logger.warning(
                    "Pre-aligned scoring failed; falling back to chunked batch",
                    exc_info=True,
                )
                use_prealigned = False
            else:
                _report(
                    "scoring_nativeness_done",
                    f"Nativeness scoring complete ({n_unique:,} unique sequences, pre-aligned)",
                    step=1,
                    total=1,
                )

        if not use_prealigned:
            # Standard chunked batch scoring path.
            if hasattr(self._nativeness_scorer, "score_batch"):
                chunk_size = _NATIVENESS_BATCH_CHUNK
                n_chunks = (n_unique + chunk_size - 1) // chunk_size
                logger.info("Scoring nativeness for %d unique sequences in %d chunk(s)", n_unique, n_chunks)
                _report(
                    "scoring_nativeness_start",
                    f"Scoring nativeness ({n_unique:,} unique sequences, {n_chunks} chunk(s))…",
                    step=1,
                    total=n_chunks,
                )
                unique_scores: list[float] = []
                _nat_start = _time.monotonic()
                for chunk_idx in range(n_chunks):
                    # Timeout check — return rows with neutral nativeness (0.5)
                    _nat_elapsed = _time.monotonic() - _nat_start
                    if self._timeout_expired(_nat_start):
                        print(
                            f"[TIMEOUT] _batch_fill_nativeness exceeded {self._operation_timeout}s "
                            f"after {chunk_idx}/{n_chunks} chunks — filling remaining with 0.5",
                            flush=True,
                        )
                        logger.warning(
                            "_batch_fill_nativeness timed out after %.1fs (%d/%d chunks scored)",
                            _nat_elapsed,
                            chunk_idx,
                            n_chunks,
                        )
                        # Fill remaining unique sequences with neutral score
                        remaining = n_unique - len(unique_scores)
                        unique_scores.extend([0.5] * remaining)
                        break
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, n_unique)
                    chunk_seqs = unique_seqs[start:end]
                    logger.info(
                        "Nativeness chunk %d/%d: scoring sequences %d–%d",
                        chunk_idx + 1,
                        n_chunks,
                        start + 1,
                        end,
                    )
                    with _timed_operation(f"Nativeness chunk {chunk_idx + 1}/{n_chunks}"):
                        chunk_scores = self._nativeness_scorer.score_batch(chunk_seqs)
                    unique_scores.extend(chunk_scores)
                    _report(
                        "scoring_nativeness_progress",
                        f"Scoring nativeness: {end:,}/{n_unique:,} sequences…",
                        step=chunk_idx + 1,
                        total=n_chunks,
                    )
                _report(
                    "scoring_nativeness_done",
                    f"Nativeness scoring complete ({n_unique:,} unique sequences)",
                    step=n_chunks,
                    total=n_chunks,
                )
            else:
                # Fallback: score individually through the cached scorer interface.
                logger.info("Scoring nativeness for %d unique sequences individually (no score_batch)", n_unique)
                _report(
                    "scoring_nativeness_start",
                    f"Scoring nativeness ({n_unique:,} unique sequences)…",
                    step=1,
                    total=n_unique,
                )
                unique_scores = []
                _progress_interval = max(n_unique // 20, 1)
                for i, seq in enumerate(unique_seqs):
                    dummy = object.__new__(VHHSequence)
                    dummy.sequence = seq
                    unique_scores.append(self._nativeness_scorer.score(dummy)["composite_score"])
                    if (i + 1) % _progress_interval == 0:
                        logger.info("Nativeness scoring: %d/%d sequences", i + 1, n_unique)
                        _report(
                            "scoring_nativeness_progress",
                            f"Scoring nativeness: {i + 1:,}/{n_unique:,} sequences…",
                            step=i + 1,
                            total=n_unique,
                        )
                _report(
                    "scoring_nativeness_done",
                    f"Nativeness scoring complete ({n_unique:,} unique sequences)",
                    step=n_unique,
                    total=n_unique,
                )

        # Map unique scores back to all rows (undoing deduplication).
        nat_scores = [unique_scores[row_to_unique[i]] for i in range(n_seqs)]

        for row, nat in zip(rows, nat_scores):
            row["nativeness_score"] = nat
            # Recompute combined_score with the actual nativeness value.
            raw_scores: dict[str, float] = {
                "stability": row["stability_score"],
                "surface_hydrophobicity": row["surface_hydrophobicity_score"],
                "nativeness": nat,
                "orthogonal_stability": row["orthogonal_stability_score"],
            }
            row["combined_score"] = self._combined_score(raw_scores)

        return rows

    # ------------------------------------------------------------------
    # Private: batch ML stability scoring for library rows
    # ------------------------------------------------------------------

    def _batch_fill_stability(
        self,
        rows: list[dict],
        progress_callback: Callable[[IterativeProgress], None] | None = None,
    ) -> list[dict]:
        """Batch-score stability with ML backends and update library rows.

        When the stability scorer has NanoMelt or ESM-2 available, this
        method batch-scores all variants and updates each row's
        ``stability_score``, ``scoring_method``, and optional ML columns
        (``nanomelt_tm``, ``predicted_tm``).  ``combined_score`` is
        recomputed afterward.

        Sequences are scored in chunks so the progress callback fires
        periodically, keeping the UI responsive during long scoring runs.

        This mirrors :meth:`_batch_fill_nativeness` and avoids the
        per-variant ML inference calls that caused Streamlit timeouts
        during library generation.

        Parameters
        ----------
        progress_callback : Callable[[IterativeProgress], None] | None
            Optional callback fired before, during, and after the batch
            scoring operation to provide UI feedback.
        """
        if not rows:
            return rows

        has_nanomelt = self._stability_scorer.nanomelt_predictor is not None
        has_esm = self._stability_scorer.esm_scorer is not None

        if not has_nanomelt and not has_esm:
            return rows  # No ML backends — heuristic scores are already set

        n_seqs = len(rows)

        def _report(phase: str, message: str, *, step: int = 0, total: int = 0) -> None:
            if progress_callback is None:
                return
            progress_callback(
                IterativeProgress(
                    phase=phase,
                    round_number=step,
                    total_rounds=total,
                    best_score=0.0,
                    mean_score=0.0,
                    population_size=n_seqs,
                    n_anchors=0,
                    diversity_entropy=0.0,
                    message=message,
                )
            )

        from vhh_library.stability import _sigmoid_normalize

        scorer = self._stability_scorer

        # --- NanoMelt batch scoring (preferred stability backend) ---
        if has_nanomelt:
            try:
                predictor = scorer.nanomelt_predictor
                chunk_size = _STABILITY_BATCH_CHUNK
                n_chunks = (n_seqs + chunk_size - 1) // chunk_size
                logger.info("Scoring stability via NanoMelt for %d sequences in %d chunk(s)", n_seqs, n_chunks)
                _report(
                    "scoring_stability_start",
                    f"Scoring stability via NanoMelt ({n_seqs:,} sequences)…",
                    step=1,
                    total=n_chunks,
                )

                all_nm_results: list[dict] = []
                _stab_start = _time.monotonic()
                for chunk_idx in range(n_chunks):
                    # Timeout check — return partial results with heuristic scores
                    _stab_elapsed = _time.monotonic() - _stab_start
                    if self._timeout_expired(_stab_start):
                        print(
                            f"[TIMEOUT] _batch_fill_stability NanoMelt exceeded {self._operation_timeout}s "
                            f"after {chunk_idx}/{n_chunks} chunks — returning partial results",
                            flush=True,
                        )
                        logger.warning(
                            "_batch_fill_stability NanoMelt timed out after %.1fs (%d/%d chunks scored)",
                            _stab_elapsed,
                            chunk_idx,
                            n_chunks,
                        )
                        return rows
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, n_seqs)
                    chunk_objects: list[VHHSequence] = []
                    for row in rows[start:end]:
                        obj = object.__new__(VHHSequence)
                        obj.sequence = row["aa_sequence"]
                        chunk_objects.append(obj)
                    logger.info(
                        "NanoMelt chunk %d/%d: scoring sequences %d–%d",
                        chunk_idx + 1,
                        n_chunks,
                        start + 1,
                        end,
                    )
                    with _timed_operation(f"NanoMelt chunk {chunk_idx + 1}/{n_chunks}"):
                        chunk_results = predictor.score_batch(chunk_objects)
                    all_nm_results.extend(chunk_results)
                    _report(
                        "scoring_stability_progress",
                        f"Scoring stability: {end:,}/{n_seqs:,} sequences…",
                        step=chunk_idx + 1,
                        total=n_chunks,
                    )

                _report(
                    "scoring_stability_done",
                    f"NanoMelt stability scoring complete ({n_seqs:,} sequences)",
                    step=n_chunks,
                    total=n_chunks,
                )

                for row, nm in zip(rows, all_nm_results):
                    nm_tm = nm["nanomelt_tm"]
                    row["nanomelt_tm"] = nm_tm

                    # Use the stability scorer's calibrated sigmoid params
                    # to match StabilityScorer.score() logic exactly.
                    # Pure ML signal — heuristic sub-scores are informational
                    # only and do not modify stability_score.
                    nm_tm_score = _sigmoid_normalize(nm_tm, scorer._tm_min, scorer._tm_max)

                    row["stability_score"] = nm_tm_score
                    row["scoring_method"] = "nanomelt"

                    # Recompute combined_score with updated stability.
                    raw_scores: dict[str, float] = {
                        "stability": row["stability_score"],
                        "surface_hydrophobicity": row["surface_hydrophobicity_score"],
                        "nativeness": row["nativeness_score"],
                        "orthogonal_stability": row["orthogonal_stability_score"],
                    }
                    row["combined_score"] = self._combined_score(raw_scores)

                return rows
            except Exception:
                if has_esm:
                    logger.warning("NanoMelt batch scoring failed; falling back to ESM-2", exc_info=True)
                else:
                    logger.warning("NanoMelt batch scoring failed; keeping heuristic scores", exc_info=True)

        # --- ESM-2 batch scoring (fallback when NanoMelt unavailable) ---
        if has_esm:
            try:
                esm = scorer.esm_scorer
                chunk_size = _STABILITY_BATCH_CHUNK
                n_chunks = (n_seqs + chunk_size - 1) // chunk_size
                logger.info("Scoring stability via ESM-2 for %d sequences in %d chunk(s)", n_seqs, n_chunks)
                _report(
                    "scoring_stability_start",
                    f"Scoring stability via ESM-2 ({n_seqs:,} sequences)…",
                    step=1,
                    total=n_chunks,
                )

                sequences = [r["aa_sequence"] for r in rows]
                all_plls: list[float] = []
                _esm_start = _time.monotonic()
                for chunk_idx in range(n_chunks):
                    # Timeout check
                    _esm_elapsed = _time.monotonic() - _esm_start
                    if self._timeout_expired(_esm_start):
                        print(
                            f"[TIMEOUT] _batch_fill_stability ESM-2 exceeded {self._operation_timeout}s "
                            f"after {chunk_idx}/{n_chunks} chunks — returning partial results",
                            flush=True,
                        )
                        logger.warning(
                            "_batch_fill_stability ESM-2 timed out after %.1fs (%d/%d chunks scored)",
                            _esm_elapsed,
                            chunk_idx,
                            n_chunks,
                        )
                        return rows
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, n_seqs)
                    chunk_seqs = sequences[start:end]
                    logger.info(
                        "ESM-2 chunk %d/%d: scoring sequences %d–%d",
                        chunk_idx + 1,
                        n_chunks,
                        start + 1,
                        end,
                    )
                    with _timed_operation(f"ESM-2 chunk {chunk_idx + 1}/{n_chunks}"):
                        chunk_plls = esm.score_batch(chunk_seqs)
                    all_plls.extend(chunk_plls)
                    _report(
                        "scoring_stability_progress",
                        f"Scoring stability: {end:,}/{n_seqs:,} sequences…",
                        step=chunk_idx + 1,
                        total=n_chunks,
                    )

                _report(
                    "scoring_stability_done",
                    f"ESM-2 stability scoring complete ({n_seqs:,} sequences)",
                    step=n_chunks,
                    total=n_chunks,
                )

                for row, pll in zip(rows, all_plls):
                    seq = row["aa_sequence"]
                    predicted_tm = scorer._pll_to_predicted_tm(pll, len(seq))
                    row["predicted_tm"] = predicted_tm

                    # Pure ML signal — heuristic sub-scores are informational
                    # only and do not modify stability_score.
                    tm_score = _sigmoid_normalize(predicted_tm, scorer._tm_min, scorer._tm_max)

                    row["stability_score"] = tm_score
                    row["scoring_method"] = "esm2"

                    raw_scores: dict[str, float] = {
                        "stability": row["stability_score"],
                        "surface_hydrophobicity": row["surface_hydrophobicity_score"],
                        "nativeness": row["nativeness_score"],
                        "orthogonal_stability": row["orthogonal_stability_score"],
                    }
                    row["combined_score"] = self._combined_score(raw_scores)

                return rows
            except Exception:
                logger.warning("ESM-2 batch scoring failed; keeping heuristic scores", exc_info=True)

        return rows

    # ------------------------------------------------------------------
    # Private: two-part assembly (yeast surface display)
    # ------------------------------------------------------------------

    def _generate_two_part_library(
        self,
        vhh_sequence: VHHSequence,
        top_mutations: pd.DataFrame,
        n_mutations: int,
        max_variants: int,
        min_mutations: int,
        strategy: str,
        anchor_threshold: float,
        max_rounds: int,
        rescore_top_n: int,
        progress_callback: Callable[[IterativeProgress], None] | None,
        checkpoint_dir: Path | None,
        split_position: str,
        overlap_n_boundary: str,
        overlap_c_boundary: str,
    ) -> pd.DataFrame:
        """Orchestrate two-part assembly: split → generate parts → combine → score."""
        from vhh_library.two_part_assembly import combine_parts, split_mutations

        imgt_positions = list(vhh_sequence.imgt_numbered.keys())

        # 1. Split mutations into Part 1 and Part 2.
        part1_mutations, part2_mutations = split_mutations(top_mutations, split_position, imgt_positions)

        if part1_mutations.empty and part2_mutations.empty:
            return self._empty_library_df()

        logger.info(
            "Two-part assembly: Part 1 has %d mutations, Part 2 has %d mutations",
            len(part1_mutations),
            len(part2_mutations),
        )

        # 2. Generate Part 1 variants.
        if not part1_mutations.empty:
            part1_df = self.generate_library(
                vhh_sequence=vhh_sequence,
                top_mutations=part1_mutations,
                n_mutations=n_mutations,
                max_variants=max_variants,
                min_mutations=min_mutations,
                strategy=strategy,
                anchor_threshold=anchor_threshold,
                max_rounds=max_rounds,
                rescore_top_n=rescore_top_n,
                progress_callback=progress_callback,
                checkpoint_dir=checkpoint_dir,
                assembly_mode=None,
            )
        else:
            # No mutations in Part 1 — use the wild-type as the sole Part 1 variant.
            part1_df = self._wildtype_part_df(vhh_sequence)

        # 3. Generate Part 2 variants.
        if not part2_mutations.empty:
            part2_df = self.generate_library(
                vhh_sequence=vhh_sequence,
                top_mutations=part2_mutations,
                n_mutations=n_mutations,
                max_variants=max_variants,
                min_mutations=min_mutations,
                strategy=strategy,
                anchor_threshold=anchor_threshold,
                max_rounds=max_rounds,
                rescore_top_n=rescore_top_n,
                progress_callback=progress_callback,
                checkpoint_dir=checkpoint_dir,
                assembly_mode=None,
            )
        else:
            part2_df = self._wildtype_part_df(vhh_sequence)

        # 4. Combine Part 1 × Part 2.
        combined_df = combine_parts(
            part1_df, part2_df, vhh_sequence, split_position, overlap_n_boundary, overlap_c_boundary
        )

        if combined_df.empty:
            return self._empty_library_df()

        # 5. Batch-score the combined variants.
        rows = combined_df.to_dict("records")

        # Batch stability scoring.
        if self._stability_scorer.nanomelt_predictor is not None or self._stability_scorer.esm_scorer is not None:
            rows = self._batch_fill_stability(rows, progress_callback=progress_callback)

        # Batch nativeness scoring.
        rows = self._batch_fill_nativeness(rows, progress_callback=progress_callback, vhh_sequence=vhh_sequence)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        _n_p1 = len(part1_df)
        _n_p2 = len(part2_df)
        logger.info(
            "Two-part assembly complete: %d Part 1 × %d Part 2 = %d scored combinations",
            _n_p1,
            _n_p2,
            len(df),
        )
        return df

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
        *,
        _batch_score: bool = True,
        progress_callback: Callable[[IterativeProgress], None] | None = None,
    ) -> list[dict]:
        # Build position groups if not provided.
        if position_groups is None:
            position_groups = {}
            for m in mutation_list:
                position_groups.setdefault(int(m.position), []).append(m)

        positions = list(position_groups.keys())
        groups = [position_groups[p] for p in positions]

        # Build rows without nativeness or ML (cheap heuristic scoring only).
        rows: list[dict] = []
        counter = 1
        for k in range(k_min, k_max + 1):
            for pos_indices in itertools.combinations(range(len(positions)), k):
                # For each chosen set of positions, take the product of their AA options.
                selected_groups = [groups[i] for i in pos_indices]
                for aa_combo in itertools.product(*selected_groups):
                    rows.append(
                        self._build_variant_row(
                            vhh_sequence,
                            list(aa_combo),
                            counter,
                            nativeness_score=0.0,
                            _skip_ml=True,
                        )
                    )
                    counter += 1
                    if counter % 100 == 0:
                        logger.info("Exhaustive enumeration: %d variants generated…", len(rows))
                        if progress_callback is not None:
                            progress_callback(
                                IterativeProgress(
                                    phase="sampling_variants",
                                    round_number=0,
                                    total_rounds=0,
                                    best_score=0.0,
                                    mean_score=0.0,
                                    population_size=len(rows),
                                    n_anchors=0,
                                    diversity_entropy=0.0,
                                    message=f"Enumerating variant {len(rows)}/{max_variants}…",
                                )
                            )
                    if len(rows) >= max_variants:
                        if _batch_score:
                            return self._batch_fill_nativeness(
                                self._batch_fill_stability(rows, progress_callback=progress_callback),
                                progress_callback=progress_callback,
                                vhh_sequence=vhh_sequence,
                            )
                        return rows
        if _batch_score:
            return self._batch_fill_nativeness(
                self._batch_fill_stability(rows, progress_callback=progress_callback),
                progress_callback=progress_callback,
                vhh_sequence=vhh_sequence,
            )
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
        *,
        _batch_score: bool = True,
        progress_callback: Callable[[IterativeProgress], None] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> list[dict]:
        """Generate random variant rows with position-aware sampling.

        Parameters
        ----------
        exclude_keys : set[str] | None
            Mutation-key strings (comma-separated mutation labels) to skip.
            Used by the iterative strategy to avoid regenerating variants
            that already exist in the global population.
        """
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

        logger.debug(
            "Sampling up to %d variants (k=%d–%d, %d positions, max_attempts=%d)",
            max_variants,
            effective_k_min,
            effective_k_max,
            len(positions),
            max_attempts,
        )

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

            # Skip variants already present in the global population.
            if exclude_keys is not None:
                mut_labels = sorted(f"{m.original_aa}{m.position}{m.suggested_aa}" for m in sample)
                mut_key = ", ".join(mut_labels)
                if mut_key in exclude_keys:
                    continue

            rows.append(self._build_variant_row(vhh_sequence, sample, counter, nativeness_score=0.0, _skip_ml=True))
            counter += 1

            # Intra-round progress: log every 100, callback every 50.
            if len(rows) % 100 == 0:
                logger.info("Sampled %d/%d variants (%d attempts)…", len(rows), max_variants, attempts)
            if len(rows) % 50 == 0 and progress_callback is not None:
                progress_callback(
                    IterativeProgress(
                        phase="sampling_variants",
                        round_number=0,
                        total_rounds=0,
                        best_score=0.0,
                        mean_score=0.0,
                        population_size=len(rows),
                        n_anchors=0,
                        diversity_entropy=0.0,
                        message=f"Sampling variant {len(rows)}/{max_variants}…",
                    )
                )

        logger.debug("Sampled %d variants in %d attempts", len(rows), attempts)

        if _batch_score:
            return self._batch_fill_nativeness(
                self._batch_fill_stability(rows, progress_callback=progress_callback),
                progress_callback=progress_callback,
                vhh_sequence=vhh_sequence,
            )
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
        *,
        _batch_score: bool = True,
        progress_callback: Callable[[IterativeProgress], None] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> list[dict]:
        """Generate anchor-constrained variant rows.

        Parameters
        ----------
        exclude_keys : set[str] | None
            Mutation-key strings to skip (see :meth:`_generate_sampled`).
        """
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
        logger.debug(
            "Constrained sampling: %d anchors, %d non-anchor positions, target %d variants",
            n_anchor,
            len(non_anchor_positions),
            max_variants,
        )

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

            # Skip variants already present in the global population.
            if exclude_keys is not None:
                mut_labels = sorted(f"{m.original_aa}{m.position}{m.suggested_aa}" for m in combined)
                mut_key = ", ".join(mut_labels)
                if mut_key in exclude_keys:
                    continue

            rows.append(self._build_variant_row(vhh_sequence, combined, counter, nativeness_score=0.0, _skip_ml=True))
            counter += 1

            # Intra-round progress: log every 100, callback every 50.
            if len(rows) % 100 == 0:
                logger.info("Constrained-sampled %d/%d variants (%d attempts)…", len(rows), max_variants, attempts)
            if len(rows) % 50 == 0 and progress_callback is not None:
                progress_callback(
                    IterativeProgress(
                        phase="sampling_variants",
                        round_number=0,
                        total_rounds=0,
                        best_score=0.0,
                        mean_score=0.0,
                        population_size=len(rows),
                        n_anchors=len(anchors),
                        diversity_entropy=0.0,
                        message=f"Sampling variant {len(rows)}/{max_variants}…",
                    )
                )

        logger.debug("Constrained-sampled %d variants in %d attempts", len(rows), attempts)

        if _batch_score:
            return self._batch_fill_nativeness(
                self._batch_fill_stability(rows, progress_callback=progress_callback),
                progress_callback=progress_callback,
                vhh_sequence=vhh_sequence,
            )
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
        checkpoint_dir: Optional[Path] = None,
        run_id: str | None = None,
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

        # Calculate per-round budgets so total across all phases sums to max_variants.
        # Phase 3 (exploitation) uses _EXPLOIT_BUDGET_RATIO of the Phase 1/2 per-round budget, so:
        #   total = (n_explore + n_anchor_id) * per_round_explore + n_exploit * (per_round_explore * ratio)
        # Solving for per_round_explore gives effective_rounds below.
        _EXPLOIT_BUDGET_RATIO = 0.5
        effective_rounds = (n_explore + n_anchor_id) + n_exploit * _EXPLOIT_BUDGET_RATIO
        per_round_explore = max(int(max_variants / max(effective_rounds, 1)), 50)
        per_round_exploit = max(int(per_round_explore * _EXPLOIT_BUDGET_RATIO), 30)

        # Fire an immediate progress callback so the UI shows something
        if progress_callback is not None:
            progress_callback(
                IterativeProgress(
                    phase="initializing",
                    round_number=0,
                    total_rounds=total_phases,
                    best_score=0.0,
                    mean_score=0.0,
                    population_size=0,
                    n_anchors=0,
                    diversity_entropy=0.0,
                    message="Initializing iterative library generation…",
                )
            )

        all_rows: list[dict] = []
        seen_keys: set[str] = set()
        counter = 1
        global_round = 0
        anchor_candidates: list[AnchorCandidate] = []
        prev_top_avg = -float("inf")
        prev_diversity = -1.0
        stagnant_rounds = 0

        # -- Checkpoint resume: reload partial results from a prior run --
        _ckpt_enabled = checkpoint_dir is not None and run_id is not None
        _resume_round = 0
        if _ckpt_enabled:
            from vhh_library.checkpoint import load_checkpoint as _load_ckpt

            loaded = _load_ckpt(checkpoint_dir, run_id)
            if loaded is not None:
                ckpt_df, completed_rounds = loaded
                # Restore rows from the checkpoint DataFrame.
                all_rows = ckpt_df.to_dict(orient="records")
                seen_keys = {r["mutations"] for r in all_rows}
                counter = len(all_rows) + 1
                global_round = completed_rounds
                _resume_round = completed_rounds
                logger.info(
                    "Resuming from checkpoint: %d rows, %d rounds completed",
                    len(all_rows),
                    completed_rounds,
                )

        def _save_ckpt() -> None:
            """Write an intermediate checkpoint if checkpointing is enabled."""
            if not _ckpt_enabled:
                return
            from vhh_library.checkpoint import save_checkpoint as _save_ckpt_fn

            _save_ckpt_fn(
                checkpoint_dir,  # type: ignore[arg-type]
                run_id,  # type: ignore[arg-type]
                pd.DataFrame(all_rows),
                completed_rounds=global_round,
            )

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
        logger.info(
            "Phase 1: Broad exploration (%d rounds, %d variants/round)",
            n_explore,
            per_round_explore,
        )
        for explore_idx in range(n_explore):
            global_round += 1
            if global_round <= _resume_round:
                logger.debug("Skipping round %d (already completed via checkpoint)", global_round)
                continue
            _report(
                "exploration",
                f"Phase 1 — Exploration round {explore_idx + 1}/{n_explore} (sampling {per_round_explore} variants)…",
            )
            with _timed_operation(f"Phase 1 exploration round {explore_idx + 1}/{n_explore}"):
                new = self._generate_sampled(
                    vhh_sequence,
                    mutation_list,
                    k_min,
                    k_max,
                    per_round_explore,
                    _batch_score=False,
                    progress_callback=progress_callback,
                    exclude_keys=seen_keys,
                )
                new = self._batch_fill_stability(new, progress_callback=progress_callback)
                new = _esm_score_rows(new)
            _add_rows(new)
            logger.info(
                "Exploration round %d/%d: +%d new → %d total",
                explore_idx + 1,
                n_explore,
                len(new),
                len(all_rows),
            )
            _report("exploration", f"Exploring ({len(all_rows)} variants)")
            _save_ckpt()

            if len(all_rows) >= max_variants:
                break

        if not all_rows:
            return all_rows

        # ==================================================================
        # PHASE 2 — Anchor Identification with epistasis detection
        # ==================================================================
        logger.info("Phase 2: Anchor identification (%d rounds)", n_anchor_id)
        for anchor_idx in range(n_anchor_id):
            global_round += 1
            if global_round <= _resume_round:
                logger.debug("Skipping round %d (already completed via checkpoint)", global_round)
                continue
            _report(
                "anchor_identification",
                f"Phase 2 — Anchor round {anchor_idx + 1}/{n_anchor_id} (sampling {per_round_explore} variants)…",
            )
            with _timed_operation(f"Phase 2 anchor round {anchor_idx + 1}/{n_anchor_id}"):
                new = self._generate_sampled(
                    vhh_sequence,
                    mutation_list,
                    k_min,
                    k_max,
                    per_round_explore,
                    _batch_score=False,
                    progress_callback=progress_callback,
                    exclude_keys=seen_keys,
                )
                new = self._batch_fill_stability(new, progress_callback=progress_callback)
                new = _esm_score_rows(new)
            _add_rows(new)
            logger.info(
                "Anchor round %d/%d: +%d new → %d total",
                anchor_idx + 1,
                n_anchor_id,
                len(new),
                len(all_rows),
            )
            _report(
                "anchor_identification",
                f"Sampling for anchor analysis ({len(all_rows)} variants)",
            )
            _save_ckpt()

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
        logger.info("Phase 3: Focused exploitation (%d rounds, %d variants/round)", n_exploit, per_round_exploit)
        for exploit_round in range(n_exploit):
            global_round += 1
            if global_round <= _resume_round:
                logger.debug("Skipping round %d (already completed via checkpoint)", global_round)
                continue

            # Build weighted anchor set from confident candidates
            anchors = self._select_anchors_weighted(anchor_candidates)
            _report(
                "exploitation",
                f"Phase 3 — Exploit round {exploit_round + 1}/{n_exploit} "
                f"({len(anchors)} anchors, {len(all_rows)} total variants)…",
            )

            if anchors:
                with _timed_operation(f"Phase 3 constrained sampling round {exploit_round + 1}/{n_exploit}"):
                    new = self._generate_constrained_sampled(
                        vhh_sequence,
                        mutation_list,
                        k_min,
                        k_max,
                        per_round_exploit,
                        anchors,
                        _batch_score=False,
                        progress_callback=progress_callback,
                        exclude_keys=seen_keys,
                    )
            else:
                with _timed_operation(f"Phase 3 sampling round {exploit_round + 1}/{n_exploit}"):
                    new = self._generate_sampled(
                        vhh_sequence,
                        mutation_list,
                        k_min,
                        k_max,
                        per_round_exploit,
                        _batch_score=False,
                        progress_callback=progress_callback,
                        exclude_keys=seen_keys,
                    )

            new = self._batch_fill_stability(new, progress_callback=progress_callback)
            new = _esm_score_rows(new)
            _add_rows(new)
            logger.info(
                "Exploit round %d/%d: +%d new → %d total (anchors=%d)",
                exploit_round + 1,
                n_exploit,
                len(new),
                len(all_rows),
                len(anchors),
            )

            # Diversity injection: if entropy drops, add random variants
            diversity = _mutation_entropy(all_rows)
            if diversity < _MIN_DIVERSITY_ENTROPY and mutation_list:
                inject_n = max(int(per_round_exploit * _DIVERSITY_INJECTION_FRAC), 5)
                inject = self._generate_sampled(
                    vhh_sequence,
                    mutation_list,
                    k_min,
                    k_max,
                    inject_n,
                    _batch_score=False,
                    progress_callback=progress_callback,
                    exclude_keys=seen_keys,
                )
                inject = self._batch_fill_stability(inject, progress_callback=progress_callback)
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
            _save_ckpt()

            if len(all_rows) >= max_variants:
                break

        # ==================================================================
        # PHASE 4 — Final Validation & Batch Scoring
        # ==================================================================
        global_round += 1
        logger.info("Phase 4: Final validation (%d total variants to batch-score)", len(all_rows))

        # Batch-score nativeness once for all accumulated variants.
        # ML stability was already scored per-round in Phases 1–3 so that
        # anchor identification and convergence checks use ML-based scores.
        # We re-run _batch_fill_stability here to ensure any variants that
        # missed ML scoring (e.g. due to timeouts) get a final pass.
        _report("scoring_stability", f"Phase 4 — Batch scoring stability ({len(all_rows):,} variants)…")
        with _timed_operation(f"Phase 4 batch stability scoring ({len(all_rows)} variants)"):
            all_rows = self._batch_fill_stability(all_rows, progress_callback=progress_callback)
        _report("scoring_nativeness", f"Phase 4 — Batch scoring nativeness ({len(all_rows):,} variants)…")
        with _timed_operation(f"Phase 4 batch nativeness scoring ({len(all_rows)} variants)"):
            all_rows = self._batch_fill_nativeness(
                all_rows, progress_callback=progress_callback, vhh_sequence=vhh_sequence
            )

        # ESM-2 full PLL for top candidates
        with _timed_operation(f"Phase 4 ESM-2 rescore (top {rescore_top_n * 2})"):
            _esm_rescore_full(all_rows, rescore_top_n * 2)
        _report("validation", "Final scoring complete")
        logger.info("Phase 4 complete: %d variants scored", len(all_rows))

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

        Mutation strings are pre-parsed once and reused across the
        frequency, marginal-contribution, and epistasis analyses to avoid
        redundant O(candidates × rows) string splitting.
        """
        if not rows:
            return []

        sorted_rows = sorted(rows, key=lambda r: r["combined_score"], reverse=True)
        n_top = max(len(sorted_rows) // 4, 1)
        all_scores = [r["combined_score"] for r in sorted_rows]
        median_all = sorted(all_scores)[len(all_scores) // 2]

        # Pre-parse mutation strings once to avoid repeated string splitting
        # in the O(candidates × rows) loops below.
        parsed_all: list[set[tuple[int, str]]] = [set(_parse_mut_str(r["mutations"])) for r in sorted_rows]
        parsed_top: list[set[tuple[int, str]]] = parsed_all[:n_top]

        # 1. Frequency analysis
        pa_counts: Counter[tuple[int, str]] = Counter()
        for muts in parsed_top:
            for pos_aa in muts:
                pa_counts[pos_aa] += 1

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
            for idx, r in enumerate(sorted_rows):
                if key in parsed_all[idx]:
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
                    parsed_mutations=parsed_all,
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

    @staticmethod
    def _wildtype_part_df(vhh_sequence: VHHSequence) -> pd.DataFrame:
        """Return a single-row DataFrame representing the wild-type sequence as a part variant."""
        return pd.DataFrame(
            [{"variant_id": "V000001", "mutations": "", "n_mutations": 0, "aa_sequence": vhh_sequence.sequence}]
        )
