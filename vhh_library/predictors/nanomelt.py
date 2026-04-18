"""NanoMelt thermal-stability predictor — optional local Tm backend.

This module wraps the `nanomelt <https://pypi.org/project/nanomelt/>`_
package behind the unified :class:`~vhh_library.predictors.base.Predictor`
protocol.  NanoMelt predicts an apparent melting temperature (Tm, °C) for
nanobody sequences using an ensemble ML model backed by ESM embeddings.

Optional dependency
~~~~~~~~~~~~~~~~~~~
``nanomelt`` is **not** a hard requirement.  If the package is absent:

* :data:`NANOMELT_AVAILABLE` is ``False``.
* Constructing a :class:`NanoMeltPredictor` raises :class:`ImportError`
  with a human-readable message.

This lets the rest of the library and its test suite run without NanoMelt.

Lazy loading
~~~~~~~~~~~~
The underlying ``NanoMeltPredictor`` from the ``nanomelt`` package is
instantiated on first inference — not at import time or ``__init__`` time
— following the project's model-loading policy.

Device-aware
~~~~~~~~~~~~
Accepts a ``device`` parameter (default ``"auto"``).  Resolution is
delegated to :func:`~vhh_library.runtime_config.resolve_device`.
NanoMelt uses ESM embeddings internally and may run on CPU, CUDA, or MPS.

Local only
~~~~~~~~~~
All inference happens locally.  No calls are made to external web servers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vhh_library.predictors.base import Predictor
from vhh_library.runtime_config import resolve_device

if TYPE_CHECKING:
    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import guard
# ---------------------------------------------------------------------------

try:
    from nanomelt.nanomelt_predictor import NanoMeltPredictor as _NanoMeltBackend  # type: ignore[import-untyped]

    NANOMELT_AVAILABLE: bool = True
except ImportError:
    _NanoMeltBackend = None  # type: ignore[assignment, misc]
    NANOMELT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tm → [0, 1] normalisation
# ---------------------------------------------------------------------------

# Ideal Tm window for VHH nanobodies (°C).  Values within this range
# receive composite scores near the middle of [0, 1]; values outside
# are pushed toward 0 or 1 via a sigmoid.
_TM_IDEAL_MIN: float = 55.0
_TM_IDEAL_MAX: float = 80.0


def _sigmoid_normalize_tm(tm: float, tm_min: float = _TM_IDEAL_MIN, tm_max: float = _TM_IDEAL_MAX) -> float:
    """Map a Tm value (°C) to [0, 1] with a sigmoid centred on the ideal range.

    Uses the same mapping shape as :func:`vhh_library.predictors.esm2_prior._sigmoid_normalize`
    so that composite_score values from different stability backends are
    directly comparable.
    """
    import math

    midpoint = (tm_min + tm_max) / 2.0
    scale = (tm_max - tm_min) / 4.0
    return 1.0 / (1.0 + math.exp(-(tm - midpoint) / max(scale, 1e-6)))


# ---------------------------------------------------------------------------
# NanoMeltPredictor
# ---------------------------------------------------------------------------


class NanoMeltPredictor(Predictor):
    """Predictor adapter for the NanoMelt thermal-stability backend.

    Parameters
    ----------
    device : str
        PyTorch device for ESM embedding generation (``"auto"``, ``"cpu"``,
        ``"cuda"``, ``"mps"``).  Resolved via
        :func:`~vhh_library.runtime_config.resolve_device`.
    batch_size : int | None
        Optional batch-size override forwarded to the NanoMelt backend.
        ``None`` uses the backend default.

    Raises
    ------
    ImportError
        If the ``nanomelt`` package is not installed.
    """

    def __init__(
        self,
        device: str = "auto",
        batch_size: int | None = None,
    ) -> None:
        if not NANOMELT_AVAILABLE:
            raise ImportError(
                "The 'nanomelt' package is required for NanoMeltPredictor but is not installed. "
                "Install it with:  pip install nanomelt"
            )
        self._device = device
        self._batch_size = batch_size
        self._backend: _NanoMeltBackend | None = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> _NanoMeltBackend:  # type: ignore[return, valid-type]
        """Lazily instantiate the NanoMelt backend on first use."""
        if self._backend is None:
            resolved = resolve_device(self._device)
            self._backend = _NanoMeltBackend()
            logger.info("NanoMeltPredictor: lazily created NanoMelt backend on %s", resolved)
            self._resolved_device = resolved
        return self._backend  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public helpers (exposed per requirements)
    # ------------------------------------------------------------------

    def nanomelt_tm_pred(self, sequence: VHHSequence) -> float:
        """Return the predicted Tm (°C) for a single VHH sequence.

        This is the raw NanoMelt output — **not** normalised to [0, 1].
        """
        backend = self._ensure_backend()
        results = backend.predict_tm([sequence.sequence], device=self._resolved_device)
        # predict_tm returns a list of floats; take the first element.
        return float(results[0])

    def delta_nanomelt_tm(
        self,
        wild_type: VHHSequence,
        mutant: VHHSequence,
    ) -> float:
        """Return the change in predicted Tm (mutant − wild-type) in °C.

        A positive value indicates the mutant is predicted to be *more*
        thermostable than the wild-type.
        """
        backend = self._ensure_backend()
        tms = backend.predict_tm(
            [wild_type.sequence, mutant.sequence],
            device=self._resolved_device,
        )
        wt_tm = float(tms[0])
        mut_tm = float(tms[1])
        return mut_tm - wt_tm

    # ------------------------------------------------------------------
    # Predictor protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the predictor identifier."""
        return "nanomelt"

    def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
        """Score *sequence* for thermal stability using NanoMelt.

        Returns
        -------
        dict[str, float]
            ``"composite_score"`` — Tm mapped to [0, 1] via sigmoid.
            ``"nanomelt_tm"``     — raw predicted Tm in °C.
        """
        tm = self.nanomelt_tm_pred(sequence)
        return {
            "composite_score": _sigmoid_normalize_tm(tm),
            "nanomelt_tm": tm,
        }

    def score_batch(self, sequences: list[VHHSequence]) -> list[dict[str, float]]:
        """Score multiple VHH sequences using NanoMelt batch inference.

        Falls back to the serial :meth:`score_sequence` loop from the
        base class if the batch is empty.

        Returns
        -------
        list[dict[str, float]]
            One result dict per input sequence, in the same order.
        """
        if not sequences:
            return []

        backend = self._ensure_backend()
        raw_seqs = [s.sequence for s in sequences]

        kwargs: dict = {"device": self._resolved_device}
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size

        tms = backend.predict_tm(raw_seqs, **kwargs)

        results: list[dict[str, float]] = []
        for tm_val in tms:
            tm = float(tm_val)
            results.append(
                {
                    "composite_score": _sigmoid_normalize_tm(tm),
                    "nanomelt_tm": tm,
                }
            )
        return results
