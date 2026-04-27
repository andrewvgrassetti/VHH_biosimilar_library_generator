"""NanoMelt thermal-stability predictor — optional local Tm backend.

This module wraps the `nanomelt <https://pypi.org/project/nanomelt/>`_
package (v1.3.0+) behind the unified
:class:`~vhh_library.predictors.base.Predictor` protocol.  NanoMelt
predicts an apparent melting temperature (Tm, °C) for nanobody sequences
using an ensemble ML model backed by ESM embeddings.

Optional dependency
~~~~~~~~~~~~~~~~~~~
``nanomelt`` is **not** a hard requirement.  If the package is absent:

* :data:`NANOMELT_AVAILABLE` is ``False``.
* Constructing a :class:`NanoMeltPredictor` raises :class:`ImportError`
  with a human-readable message.

This lets the rest of the library and its test suite run without NanoMelt.

Lazy loading
~~~~~~~~~~~~
NanoMelt v1.3.0 loads ESM model weights at **import** time (when
``from nanomelt.predict import NanoMeltPredPipe`` is executed).  To
honour the project's lazy-loading policy the import is deferred to
first inference inside :meth:`NanoMeltPredictor._ensure_backend`.
The availability check (:data:`NANOMELT_AVAILABLE`) uses
:func:`importlib.util.find_spec` so it never triggers the heavy import.

Device handling
~~~~~~~~~~~~~~~
NanoMelt v1.3.0 manages GPU transfer internally — there is no
``device`` parameter.  The constructor still accepts ``device`` for
API compatibility but it is unused.  A warning is logged if a value
other than ``"auto"`` or ``"cpu"`` is supplied.

Local only
~~~~~~~~~~
All inference happens locally.  No calls are made to external web servers.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import warnings
from typing import TYPE_CHECKING, Any

from vhh_library.predictors.base import Predictor

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import guard — lightweight, never triggers ESM model loading
# ---------------------------------------------------------------------------

NANOMELT_AVAILABLE: bool = importlib.util.find_spec("nanomelt") is not None

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
# Helpers
# ---------------------------------------------------------------------------

_TM_COLUMN: str = "NanoMelt Tm (C)"


def _vhh_to_seqrecord(vhh: VHHSequence, record_id: str | None = None) -> Any:
    """Convert a :class:`VHHSequence` to a BioPython ``SeqRecord``.

    Parameters
    ----------
    vhh : VHHSequence
        The nanobody sequence to convert.
    record_id : str | None
        Optional identifier for the ``SeqRecord``.  Falls back to
        ``getattr(vhh, "name", "seq")``.

    Returns
    -------
    Bio.SeqRecord.SeqRecord
    """
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rid = record_id or getattr(vhh, "name", "seq")
    return SeqRecord(Seq(vhh.sequence), id=str(rid))


# ---------------------------------------------------------------------------
# NanoMeltPredictor
# ---------------------------------------------------------------------------


class NanoMeltPredictor(Predictor):
    """Predictor adapter for the NanoMelt (v1.3.0+) thermal-stability backend.

    Parameters
    ----------
    device : str
        Retained for API compatibility.  NanoMelt v1.3.0 handles device
        selection internally.  If a value other than ``"auto"`` or
        ``"cpu"`` is passed, a :class:`UserWarning` is emitted.
    batch_size : int | None
        Optional batch-size override forwarded to ``NanoMeltPredPipe``.
        ``None`` uses the backend default (420).
    do_align : bool
        Whether NanoMelt should align sequences via ANARCI before
        prediction.  Defaults to ``True``.
    ncpus : int
        Number of CPUs for alignment and embedding parallelisation.
        Defaults to ``1``.

    Raises
    ------
    ImportError
        If the ``nanomelt`` package is not installed.
    """

    def __init__(
        self,
        device: str = "auto",
        batch_size: int | None = None,
        *,
        do_align: bool = True,
        ncpus: int = 1,
    ) -> None:
        if not NANOMELT_AVAILABLE:
            raise ImportError(
                "The 'nanomelt' package is required for NanoMeltPredictor but is not installed. "
                "Install it with:  pip install nanomelt"
            )

        # Device is no longer forwarded — NanoMelt v1.3.0 manages GPU
        # transfer internally.  Keep the parameter for API compatibility.
        self._device = device
        if device not in ("auto", "cpu"):
            warnings.warn(
                f"NanoMelt v1.3.0 handles device selection internally; the "
                f"requested device={device!r} will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        self._batch_size = batch_size
        self._do_align = do_align
        self._ncpus = ncpus

        # The callable is loaded lazily in _ensure_backend.
        self._backend: Callable[..., pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> Callable[..., pd.DataFrame]:
        """Lazily import ``NanoMeltPredPipe`` on first use.

        NanoMelt v1.3.0 loads ESM weights at import time, so the import
        is deferred to this method.

        The ANARCI compatibility patch is applied before the import so
        that NanoMelt's internal ANARCI calls (with ``do_align=True``)
        are protected against ``None`` coordinates from BioPython >= 1.83.
        """
        if self._backend is None:
            from vhh_library.numbering import _apply_anarci_compat_patch

            _apply_anarci_compat_patch()

            from nanomelt.predict import NanoMeltPredPipe  # type: ignore[import-untyped]

            self._backend = NanoMeltPredPipe
            logger.info("NanoMeltPredictor: lazily imported NanoMeltPredPipe")
        return self._backend

    # ------------------------------------------------------------------
    # Warm-up (pre-load model in main thread)
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Pre-load the NanoMelt backend and run a minimal prediction.

        Forces the underlying ESM model to be loaded and transferred to
        GPU in the **calling** thread.  Call this from the main thread
        before dispatching work to background / daemon threads to avoid
        CUDA context-initialisation hangs.

        The method is non-fatal: if the warm-up prediction fails, a
        warning is logged and the predictor remains usable (the real
        prediction will attempt the load again later).
        """
        self._ensure_backend()

        # Use a short but realistic VHH framework sequence.
        _WARMUP_SEQ = (
            "EVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"
        )

        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        record = SeqRecord(Seq(_WARMUP_SEQ), id="warmup")
        try:
            self._predict_tm_for_records([record])
            logger.info("NanoMeltPredictor warm-up complete (model loaded and GPU transfer done).")
        except Exception:
            logger.warning(
                "NanoMeltPredictor warm-up prediction failed; model will be loaded on first real inference.",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_tm_for_records(self, seq_records: list[Any]) -> pd.DataFrame:
        """Run the NanoMelt pipeline on a list of BioPython ``SeqRecord`` objects.

        NanoMelt v1.3.0 prints ``"Loading ESM data"`` and similar status
        messages to stdout for every batch it processes.  These messages
        are suppressed here by redirecting stdout so they do not flood
        the console during large scoring runs.
        """
        backend = self._ensure_backend()
        kwargs: dict[str, Any] = {
            "seq_records": seq_records,
            "do_align": self._do_align,
            "ncpus": self._ncpus,
        }
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size
        with contextlib.redirect_stdout(io.StringIO()):
            return backend(**kwargs)

    # ------------------------------------------------------------------
    # Public helpers (exposed per requirements)
    # ------------------------------------------------------------------

    def nanomelt_tm_pred(self, sequence: VHHSequence) -> float:
        """Return the predicted Tm (°C) for a single VHH sequence.

        This is the raw NanoMelt output — **not** normalised to [0, 1].
        """
        record = _vhh_to_seqrecord(sequence, record_id="query")
        df = self._predict_tm_for_records([record])
        return float(df[_TM_COLUMN].iloc[0])

    def delta_nanomelt_tm(
        self,
        wild_type: VHHSequence,
        mutant: VHHSequence,
    ) -> float:
        """Return the change in predicted Tm (mutant − wild-type) in °C.

        A positive value indicates the mutant is predicted to be *more*
        thermostable than the wild-type.
        """
        records = [
            _vhh_to_seqrecord(wild_type, record_id="wild_type"),
            _vhh_to_seqrecord(mutant, record_id="mutant"),
        ]
        df = self._predict_tm_for_records(records)
        wt_tm = float(df[_TM_COLUMN].iloc[0])
        mut_tm = float(df[_TM_COLUMN].iloc[1])
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

        Returns
        -------
        list[dict[str, float]]
            One result dict per input sequence, in the same order.
        """
        if not sequences:
            return []

        records = [_vhh_to_seqrecord(seq, record_id=f"seq_{i}") for i, seq in enumerate(sequences)]
        df = self._predict_tm_for_records(records)

        results: list[dict[str, float]] = []
        for tm_val in df[_TM_COLUMN]:
            tm = float(tm_val)
            results.append(
                {
                    "composite_score": _sigmoid_normalize_tm(tm),
                    "nanomelt_tm": tm,
                }
            )
        return results

    def score_batch_prealigned(
        self,
        parent_sequence: str,  # noqa: ARG002
        variant_sequences: list[str],
    ) -> list[dict[str, float]]:
        """Score variants using NanoMelt with ``do_align=False``.

        Variant sequences — which differ from the parent only at a few
        point-mutation sites — are scored *without* re-running ANARCI
        alignment, reducing the cost from O(n) ANARCI calls to O(1).

        The ``parent_sequence`` parameter is accepted for API consistency
        with :meth:`NativenessScorer.score_batch_prealigned` but is not
        used by this implementation.  NanoMelt's ``do_align=False`` mode
        does not require a reference alignment.

        Parameters
        ----------
        parent_sequence : str
            Wild-type amino-acid string.
        variant_sequences : list[str]
            List of variant amino-acid strings to score.

        Returns
        -------
        list[dict[str, float]]
            One result dict per variant (``composite_score``, ``nanomelt_tm``).
        """
        if not variant_sequences:
            return []

        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        records = [SeqRecord(Seq(seq), id=f"var_{i}") for i, seq in enumerate(variant_sequences)]

        # Score with do_align=False to skip per-variant ANARCI
        backend = self._ensure_backend()
        kwargs: dict[str, Any] = {
            "seq_records": records,
            "do_align": False,
            "ncpus": self._ncpus,
        }
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size

        with contextlib.redirect_stdout(io.StringIO()):
            df = backend(**kwargs)

        results: list[dict[str, float]] = []
        for tm_val in df[_TM_COLUMN]:
            tm = float(tm_val)
            results.append(
                {
                    "composite_score": _sigmoid_normalize_tm(tm),
                    "nanomelt_tm": tm,
                }
            )
        return results
