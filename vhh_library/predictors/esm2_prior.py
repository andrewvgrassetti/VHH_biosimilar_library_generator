"""ESM-2 pseudo-log-likelihood predictor — adapter over :class:`~vhh_library.esm_scorer.ESMStabilityScorer`.

In the target architecture, ESM-2 PLL becomes an **optional prior** — a
language-model plausibility check that can re-weight variants but is not
required for ranking.  This adapter wraps the existing ``ESMStabilityScorer``
behind the unified :class:`~vhh_library.predictors.base.Predictor` protocol.

The raw PLL is normalised to a [0, 1] score via the same sigmoid mapping
used by :class:`~vhh_library.stability.StabilityScorer` so that the
composite_score returned by this predictor is directly comparable to other
axes.

Lazy loading
~~~~~~~~~~~~
The wrapped ``ESMStabilityScorer`` is instantiated on first use, following
the project's model-loading policy.

Device-aware
~~~~~~~~~~~~
Accepts a ``device`` parameter (default ``"auto"``).  Resolution is
delegated to :func:`~vhh_library.runtime_config.resolve_device`.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from vhh_library.predictors.base import Predictor
from vhh_library.runtime_config import resolve_device

if TYPE_CHECKING:
    from vhh_library.esm_scorer import ESMStabilityScorer
    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)

# Default PLL → [0,1] normalisation parameters (mirrors stability.py)
_PLL_TO_TM_SLOPE: float = 12.5
_PLL_TO_TM_INTERCEPT: float = 95.0
_TM_IDEAL_MIN: float = 55.0
_TM_IDEAL_MAX: float = 80.0


def _sigmoid_normalize(tm: float, tm_min: float, tm_max: float) -> float:
    """Map a Tm value to [0, 1] with a sigmoid centred on the ideal range."""
    midpoint = (tm_min + tm_max) / 2.0
    scale = (tm_max - tm_min) / 4.0
    return 1.0 / (1.0 + math.exp(-(tm - midpoint) / max(scale, 1e-6)))


class ESM2PriorPredictor(Predictor):
    """Predictor adapter for ESM-2 pseudo-log-likelihood scoring.

    Parameters
    ----------
    model_tier : str
        ESM-2 model size (``"auto"``, ``"t6_8M"``, ``"t12_35M"``,
        ``"t33_650M"``, ``"t36_3B"``).
    device : str
        PyTorch device (``"auto"``, ``"cpu"``, ``"cuda"``).
    batch_size : int | None
        Batch size override.
    cache_dir : str | None
        SQLite cache directory.
    scorer : ESMStabilityScorer | None
        Pre-built scorer instance.  When *None*, a new ``ESMStabilityScorer``
        is created lazily on first use.
    """

    def __init__(
        self,
        model_tier: str = "auto",
        device: str = "auto",
        batch_size: int | None = None,
        cache_dir: str | None = None,
        *,
        scorer: "ESMStabilityScorer | None" = None,
    ) -> None:
        self._model_tier = model_tier
        self._device = device
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._scorer: "ESMStabilityScorer | None" = scorer

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_scorer(self) -> "ESMStabilityScorer":
        if self._scorer is None:
            from vhh_library.esm_scorer import ESMStabilityScorer

            resolved = resolve_device(self._device)
            self._scorer = ESMStabilityScorer(
                model_tier=self._model_tier,
                device=resolved,
                batch_size=self._batch_size,
                cache_dir=self._cache_dir,
            )
            logger.info("ESM2PriorPredictor: lazily created ESMStabilityScorer on %s", resolved)
        return self._scorer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pll_to_score(pll: float, seq_len: int) -> float:
        """Convert a raw PLL to a normalised [0, 1] score."""
        per_residue = pll / max(seq_len, 1)
        predicted_tm = _PLL_TO_TM_SLOPE * per_residue + _PLL_TO_TM_INTERCEPT
        return _sigmoid_normalize(predicted_tm, _TM_IDEAL_MIN, _TM_IDEAL_MAX)

    # ------------------------------------------------------------------
    # Predictor protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "esm2_prior"

    def score_sequence(self, sequence: "VHHSequence") -> dict[str, float]:
        """Score *sequence* with ESM-2 PLL.

        Returns
        -------
        dict[str, float]
            ``"composite_score"`` — PLL mapped to [0, 1] via sigmoid.
            ``"esm2_pll"``        — raw pseudo-log-likelihood.
        """
        scorer = self._ensure_scorer()
        pll = scorer.score_single(sequence.sequence)
        return {
            "composite_score": self._pll_to_score(pll, sequence.length),
            "esm2_pll": pll,
        }

    def score_batch(self, sequences: list["VHHSequence"]) -> list[dict[str, float]]:
        """Score multiple VHH sequences using ESM-2 batch inference."""
        scorer = self._ensure_scorer()
        raw_seqs = [s.sequence for s in sequences]
        plls = scorer.score_batch(raw_seqs)
        results: list[dict[str, float]] = []
        for seq, pll in zip(sequences, plls):
            results.append({
                "composite_score": self._pll_to_score(pll, seq.length),
                "esm2_pll": pll,
            })
        return results
