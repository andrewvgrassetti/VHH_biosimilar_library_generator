"""Abstract predictor interface for VHH scoring backends.

Every scoring backend — AbNatiV nativeness, ESM-2 pseudo-log-likelihood,
the future NanoMelt thermal-stability predictor, etc. — must implement
this protocol so the mutation engine and app layer can consume them
uniformly.

The interface is intentionally minimal:

* **name** — a short, unique identifier (e.g. ``"abnativ"``, ``"esm2_prior"``).
* **score_sequence** — score a single :class:`~vhh_library.sequence.VHHSequence`
  and return a ``dict[str, float]`` with at least a ``"composite_score"`` key.
* **score_batch** (optional) — score multiple sequences efficiently.
  A default implementation that loops over ``score_sequence`` is provided.

New predictors (NanoMelt, etc.) should subclass :class:`Predictor` and
override the abstract methods.  Existing scorers (``StabilityScorer``,
``NativenessScorer``) are wrapped via thin adapter classes that live
alongside this module.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vhh_library.sequence import VHHSequence


class Predictor(abc.ABC):
    """Abstract base for all VHH scoring backends.

    Subclasses must implement :attr:`name` and :meth:`score_sequence`.
    :meth:`score_batch` may be overridden for backends that benefit from
    batched inference (GPU models, network calls, etc.).
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short unique identifier for this predictor (e.g. ``"abnativ"``)."""

    @abc.abstractmethod
    def score_sequence(self, sequence: "VHHSequence") -> dict[str, float]:
        """Score a single VHH sequence.

        Returns
        -------
        dict[str, float]
            Must contain at least ``"composite_score"`` (a per-axis normalised
            value in [0, 1]).  Additional keys are backend-specific and will be
            passed through to the library DataFrame.
        """

    # ------------------------------------------------------------------
    # Optional batch interface (default: serial loop)
    # ------------------------------------------------------------------

    def score_batch(self, sequences: list["VHHSequence"]) -> list[dict[str, float]]:
        """Score multiple VHH sequences.

        The default implementation calls :meth:`score_sequence` in a loop.
        Backends that support batched inference should override this for
        throughput.

        Returns
        -------
        list[dict[str, float]]
            One result dict per input sequence, in the same order.
        """
        return [self.score_sequence(seq) for seq in sequences]
