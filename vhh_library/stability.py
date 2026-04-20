"""Score VHH sequences for biophysical stability properties."""

from __future__ import annotations

import json
import logging
import math
import warnings as _warnings
from pathlib import Path
from typing import TYPE_CHECKING

from vhh_library.sequence import VHHSequence
from vhh_library.utils import AA_PROPERTIES, isoelectric_point, net_charge

if TYPE_CHECKING:
    from vhh_library.esm_scorer import ESMStabilityScorer
    from vhh_library.predictors.nanomelt import NanoMeltPredictor

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_HALLMARK_POSITIONS: list[int] = [37, 44, 45, 47]
_HYDROPHOBIC_AAS: frozenset[str] = frozenset("VILMFYW")
_DISULFIDE_POSITIONS: list[int] = [23, 104]

# Composite score weights (legacy)
_W_DISULFIDE: float = 0.25
_W_HALLMARK: float = 0.20
_W_AGGREGATION: float = 0.25
_W_CHARGE: float = 0.15
_W_HYDROPHOBIC: float = 0.15

# Calibration: per-residue PLL → predicted Tm (°C)
# Starting estimates; replace with empirical fit from NbThermo database.
_PLL_TO_TM_SLOPE: float = 12.5
_PLL_TO_TM_INTERCEPT: float = 95.0

# Ideal VHH Tm range for sigmoid normalisation
_TM_IDEAL_MIN: float = 55.0
_TM_IDEAL_MAX: float = 80.0

# Hard penalty magnitudes (heuristic gates)
_PENALTY_DISULFIDE: float = 0.20
_PENALTY_AGGREGATION: float = 0.10
_PENALTY_CHARGE: float = 0.05

# VHH hallmark bonus weight
_HALLMARK_BONUS_WEIGHT: float = 0.10


def _pll_to_predicted_tm(pll: float, seq_len: int) -> float:
    """Convert total PLL to an estimated Tm via per-residue normalisation."""
    per_residue_pll = pll / max(seq_len, 1)
    return _PLL_TO_TM_SLOPE * per_residue_pll + _PLL_TO_TM_INTERCEPT


def _sigmoid_normalize(tm: float, tm_min: float, tm_max: float) -> float:
    """Map a Tm value to [0, 1] with a sigmoid centred on the ideal range."""
    midpoint = (tm_min + tm_max) / 2.0
    scale = (tm_max - tm_min) / 4.0
    return 1.0 / (1.0 + math.exp(-(tm - midpoint) / max(scale, 1e-6)))


# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------

_esm2_pll_flag: bool | None = None


def _esm2_pll_available() -> bool:
    global _esm2_pll_flag
    if _esm2_pll_flag is None:
        try:
            import esm  # noqa: F401
            import torch  # noqa: F401

            _esm2_pll_flag = True
        except ImportError:
            _esm2_pll_flag = False
    return _esm2_pll_flag


# ---------------------------------------------------------------------------
# ESM-2 pseudo-log-likelihood
# ---------------------------------------------------------------------------


def compute_esm2_pll(sequences: list[str]) -> list[float]:
    """Compute ESM-2 pseudo-log-likelihood for each sequence.

    Delegates to :class:`~vhh_library.esm_scorer.ESMStabilityScorer` with the
    smallest model tier (``t6_8M``) for backward compatibility.

    Raises ``ImportError`` if *torch* or *esm* are not installed.
    """
    from vhh_library.esm_scorer import ESMStabilityScorer

    scorer = ESMStabilityScorer(model_tier="t6_8M", device="auto")
    return scorer.score_batch(sequences)


# ---------------------------------------------------------------------------
# Stability scorer
# ---------------------------------------------------------------------------


class StabilityScorer:
    """Score VHH sequences for biophysical stability using legacy heuristics
    and optional ESM-2 integration."""

    def __init__(
        self,
        esm_scorer: "ESMStabilityScorer | None" = None,
        calibration_path: str | None = None,
        *,
        nanomelt_predictor: "NanoMeltPredictor | None" = None,
        esm2_weight: float | None = None,
        legacy_weight: float | None = None,
        # Kept for backward compatibility; ignored — use nanomelt_predictor instead.
        use_nanomelt: bool = False,
    ) -> None:
        if use_nanomelt:
            _warnings.warn(
                "use_nanomelt is deprecated; pass a NanoMeltPredictor instance via "
                "nanomelt_predictor= instead.  This parameter will be removed in a "
                "future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        if esm2_weight is not None:
            _warnings.warn(
                "esm2_weight is deprecated and ignored; ESM-2 scoring now uses Tm-based gating and penalties.",
                DeprecationWarning,
                stacklevel=2,
            )
        if legacy_weight is not None:
            _warnings.warn(
                "legacy_weight is deprecated and ignored; ESM-2 scoring now uses Tm-based gating and penalties.",
                DeprecationWarning,
                stacklevel=2,
            )
        germline_path = _DATA_DIR / "vhh_germlines.json"
        with open(germline_path) as fh:
            self.germlines: list[dict] = json.load(fh)["germlines"]
        self.esm_scorer: "ESMStabilityScorer | None" = esm_scorer
        self.nanomelt_predictor: "NanoMeltPredictor | None" = nanomelt_predictor

        # Load calibration parameters (or fall back to module-level defaults)
        self._load_calibration_params(calibration_path)

    def _load_calibration_params(self, calibration_path: str | None) -> None:
        """Load calibration from file or fall back to module-level defaults."""
        from vhh_library.calibration import load_calibration

        cal = load_calibration(calibration_path)
        if cal is not None:
            params = cal.get("parameters", {})
            self._pll_slope = params.get("pll_to_tm_slope", _PLL_TO_TM_SLOPE)
            self._pll_intercept = params.get("pll_to_tm_intercept", _PLL_TO_TM_INTERCEPT)
            self._tm_min = params.get("tm_ideal_min", _TM_IDEAL_MIN)
            self._tm_max = params.get("tm_ideal_max", _TM_IDEAL_MAX)
            self._penalty_disulfide = params.get("penalty_disulfide", _PENALTY_DISULFIDE)
            self._penalty_aggregation = params.get("penalty_aggregation", _PENALTY_AGGREGATION)
            self._penalty_charge = params.get("penalty_charge", _PENALTY_CHARGE)
            self._hallmark_bonus_weight = params.get("hallmark_bonus_weight", _HALLMARK_BONUS_WEIGHT)
            lw = params.get("legacy_weights", {})
            self._w_disulfide = lw.get("disulfide", _W_DISULFIDE)
            self._w_hallmark = lw.get("hallmark", _W_HALLMARK)
            self._w_aggregation = lw.get("aggregation", _W_AGGREGATION)
            self._w_charge = lw.get("charge", _W_CHARGE)
            self._w_hydrophobic = lw.get("hydrophobic", _W_HYDROPHOBIC)
            self._calibrated = True
        else:
            self._pll_slope = _PLL_TO_TM_SLOPE
            self._pll_intercept = _PLL_TO_TM_INTERCEPT
            self._tm_min = _TM_IDEAL_MIN
            self._tm_max = _TM_IDEAL_MAX
            self._penalty_disulfide = _PENALTY_DISULFIDE
            self._penalty_aggregation = _PENALTY_AGGREGATION
            self._penalty_charge = _PENALTY_CHARGE
            self._hallmark_bonus_weight = _HALLMARK_BONUS_WEIGHT
            self._w_disulfide = _W_DISULFIDE
            self._w_hallmark = _W_HALLMARK
            self._w_aggregation = _W_AGGREGATION
            self._w_charge = _W_CHARGE
            self._w_hydrophobic = _W_HYDROPHOBIC
            self._calibrated = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, vhh: VHHSequence) -> dict:
        """Return a comprehensive stability score dictionary for *vhh*."""
        seq = vhh.sequence
        numbered = vhh.imgt_numbered
        warnings: list[str] = list(vhh.validation_result.get("warnings", []))

        disulfide = self._disulfide_score(numbered, warnings)
        hallmark = self._hallmark_score(numbered, warnings)
        aggregation = self._aggregation_score(seq)
        charge_balance = self._charge_balance_score(seq)
        hydrophobic_core = self._hydrophobic_core_score(numbered)

        charge = net_charge(seq)
        pi = isoelectric_point(seq)

        legacy = (
            self._w_disulfide * disulfide
            + self._w_hallmark * hallmark
            + self._w_aggregation * aggregation
            + self._w_charge * charge_balance
            + self._w_hydrophobic * hydrophobic_core
        )

        result: dict = {
            "disulfide_score": disulfide,
            "vhh_hallmark_score": hallmark,
            "aggregation_score": aggregation,
            "charge_balance_score": charge_balance,
            "hydrophobic_core_score": hydrophobic_core,
            "net_charge": charge,
            "pI": pi,
            "warnings": warnings,
        }

        # --- Compute shared penalty / bonus terms ---
        penalty = 0.0
        if disulfide < 1.0:
            penalty += self._penalty_disulfide
        if aggregation < 0.5:
            penalty += self._penalty_aggregation
        if charge_balance < 0.5:
            penalty += self._penalty_charge

        vhh_bonus = self._hallmark_bonus_weight * hallmark

        # --- Determine which backends to use ---
        has_esm = self.esm_scorer is not None
        has_nanomelt = self.nanomelt_predictor is not None

        esm2_composite: float | None = None
        nanomelt_composite: float | None = None

        # --- ESM-2 branch ---
        if has_esm:
            try:
                pll = self.esm_scorer.score_single(seq)
                result["esm2_pll"] = pll

                predicted_tm = self._pll_to_predicted_tm(pll, len(seq))
                result["predicted_tm"] = predicted_tm

                tm_score = _sigmoid_normalize(predicted_tm, self._tm_min, self._tm_max)
                result["tm_score"] = tm_score

                esm2_composite = max(0.0, min(1.0, tm_score + vhh_bonus - penalty))
            except Exception:
                warnings.append("ESM-2 scoring failed; fell back to legacy scoring")
                logger.warning("ESM-2 scoring failed", exc_info=True)

        # --- NanoMelt branch ---
        if has_nanomelt:
            try:
                nm_result = self.nanomelt_predictor.score_sequence(vhh)
                nanomelt_tm = nm_result["nanomelt_tm"]
                result["nanomelt_tm"] = nanomelt_tm

                nm_tm_score = _sigmoid_normalize(nanomelt_tm, self._tm_min, self._tm_max)
                result["nanomelt_tm_score"] = nm_tm_score

                nanomelt_composite = max(0.0, min(1.0, nm_tm_score + vhh_bonus - penalty))
            except Exception:
                warnings.append("NanoMelt scoring failed; ignoring NanoMelt contribution")
                logger.warning("NanoMelt scoring failed", exc_info=True)

        # --- Combine into composite_score and scoring_method ---
        # NanoMelt is the primary stability signal.  When it is available,
        # composite_score is derived solely from NanoMelt Tm.  ESM-2 scores
        # are still computed above for diagnostic/informational purposes but
        # no longer influence composite_score.
        if nanomelt_composite is not None:
            result["composite_score"] = nanomelt_composite
            result["scoring_method"] = "nanomelt"
        elif esm2_composite is not None:
            result["composite_score"] = esm2_composite
            result["scoring_method"] = "esm2"
        else:
            result["composite_score"] = legacy
            result["scoring_method"] = "legacy"

        return result

    def _pll_to_predicted_tm(self, pll: float, seq_len: int) -> float:
        """Convert total PLL to an estimated Tm via per-residue normalisation."""
        per_residue_pll = pll / max(seq_len, 1)
        return self._pll_slope * per_residue_pll + self._pll_intercept

    def predict_mutation_effect(self, vhh: VHHSequence, position: int | str, new_aa: str) -> float:
        """Return the change in composite score when mutating *position* to *new_aa*."""
        parent_score = self.score(vhh)["composite_score"]
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        mutant_score = self.score(mutant)["composite_score"]
        return mutant_score - parent_score

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    @staticmethod
    def _disulfide_score(numbered: dict[str, str], warnings: list[str]) -> float:
        cys_count = sum(1 for pos in _DISULFIDE_POSITIONS if numbered.get(str(pos)) == "C")
        if cys_count == 2:
            return 1.0
        if cys_count == 1:
            warnings.append("Only one canonical Cys found for disulfide bond")
            return 0.5
        warnings.append("No canonical Cys residues found for disulfide bond")
        return 0.0

    def _hallmark_score(self, numbered: dict[str, str], warnings: list[str]) -> float:
        allowed: dict[int, set[str]] = {}
        for gl in self.germlines:
            hp = gl.get("hallmark_positions", {})
            for pos_str, aa in hp.items():
                pos = int(pos_str)
                allowed.setdefault(pos, set()).add(aa)

        matches = 0
        for pos in _HALLMARK_POSITIONS:
            aa = numbered.get(str(pos))
            if aa is not None and aa in allowed.get(pos, set()):
                matches += 1
            else:
                warnings.append(f"Non-canonical residue '{aa}' at VHH hallmark position {pos}")

        return matches / len(_HALLMARK_POSITIONS) if _HALLMARK_POSITIONS else 0.0

    @staticmethod
    def _aggregation_score(sequence: str) -> float:
        window = 7
        if len(sequence) < window:
            return 1.0
        n_patches = 0
        for i in range(len(sequence) - window + 1):
            segment = sequence[i : i + window]
            avg_hydro = sum(AA_PROPERTIES.get(aa, {}).get("hydrophobicity", 0.0) for aa in segment) / window
            if avg_hydro > 1.5:
                n_patches += 1
        return max(0.0, 1.0 - n_patches * 0.15)

    @staticmethod
    def _charge_balance_score(sequence: str) -> float:
        charge = net_charge(sequence)
        abs_charge = abs(charge)
        if abs_charge <= 2.0:
            return 1.0
        return max(0.0, 1.0 - (abs_charge - 2.0) * 0.1)

    @staticmethod
    def _hydrophobic_core_score(numbered: dict[str, str]) -> float:
        if not _HALLMARK_POSITIONS:
            return 0.0
        hits = sum(1 for pos in _HALLMARK_POSITIONS if numbered.get(str(pos), "") in _HYDROPHOBIC_AAS)
        return hits / len(_HALLMARK_POSITIONS)
