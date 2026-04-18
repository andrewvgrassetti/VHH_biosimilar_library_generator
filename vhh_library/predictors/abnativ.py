"""AbNatiV nativeness predictor — adapter over :class:`~vhh_library.nativeness.NativenessScorer`.

This module wraps the existing ``NativenessScorer`` behind the unified
:class:`~vhh_library.predictors.base.Predictor` protocol so that the
mutation engine and future ranking layers can treat all scoring backends
uniformly.

The underlying ``NativenessScorer`` remains the sole source of truth for
AbNatiV inference; this adapter delegates all heavy lifting to it.

Lazy loading
~~~~~~~~~~~~
The wrapped ``NativenessScorer`` is instantiated on first use, not at
import time or ``__init__`` time, following the project's model-loading
policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vhh_library.predictors.base import Predictor

if TYPE_CHECKING:
    from vhh_library.nativeness import NativenessScorer
    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)


class AbNatiVPredictor(Predictor):
    """Predictor adapter for the AbNatiV VQ-VAE nativeness scorer.

    Parameters
    ----------
    model_type : str
        AbNatiV model variant (``"VHH"`` or ``"VHH2"``).
    batch_size : int
        Batch size forwarded to :class:`~vhh_library.nativeness.NativenessScorer`.
    scorer : NativenessScorer | None
        Pre-built scorer instance.  When *None* (the default), a new
        ``NativenessScorer`` is created lazily on first use.
    """

    def __init__(
        self,
        model_type: str = "VHH",
        batch_size: int = 128,
        *,
        scorer: "NativenessScorer | None" = None,
    ) -> None:
        self._model_type = model_type
        self._batch_size = batch_size
        self._scorer: "NativenessScorer | None" = scorer

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_scorer(self) -> "NativenessScorer":
        if self._scorer is None:
            from vhh_library.nativeness import NativenessScorer

            self._scorer = NativenessScorer(
                model_type=self._model_type,
                batch_size=self._batch_size,
            )
            logger.info("AbNatiVPredictor: lazily created NativenessScorer")
        return self._scorer

    # ------------------------------------------------------------------
    # Predictor protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the predictor identifier."""
        return "abnativ"

    def score_sequence(self, sequence: "VHHSequence") -> dict[str, float]:
        """Score *sequence* for nativeness.

        Delegates to ``NativenessScorer.score()`` and returns its result
        dict (contains at least ``"composite_score"``).
        """
        scorer = self._ensure_scorer()
        return scorer.score(sequence)

    def score_batch(self, sequences: list["VHHSequence"]) -> list[dict[str, float]]:
        """Score multiple VHH sequences using AbNatiV's batch path.

        Delegates to ``NativenessScorer.score_batch()`` which passes all
        sequences through the VQ-VAE in a single call for throughput.
        """
        scorer = self._ensure_scorer()
        raw_scores = scorer.score_batch([s.sequence for s in sequences])
        return [{"composite_score": s} for s in raw_scores]
