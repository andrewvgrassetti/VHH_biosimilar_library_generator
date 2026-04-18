"""Unified predictor abstractions for VHH scoring backends.

This package exposes a clean :class:`Predictor` protocol and concrete
adapter classes that wrap the existing scoring modules behind it:

* :class:`AbNatiVPredictor` — AbNatiV VQ-VAE nativeness.
* :class:`ESM2PriorPredictor` — ESM-2 pseudo-log-likelihood (optional prior).

Importing from this package is the preferred way for **new** code to
interact with scoring backends.  Existing code that uses
:class:`~vhh_library.nativeness.NativenessScorer` or
:class:`~vhh_library.stability.StabilityScorer` directly continues to
work unchanged.

Example
-------
::

    from vhh_library.predictors import AbNatiVPredictor, Predictor

    predictor: Predictor = AbNatiVPredictor(model_type="VHH")
    result = predictor.score_sequence(vhh_seq)
    print(result["composite_score"])
"""

from __future__ import annotations

from vhh_library.predictors.abnativ import AbNatiVPredictor
from vhh_library.predictors.base import Predictor
from vhh_library.predictors.esm2_prior import ESM2PriorPredictor

__all__ = [
    "AbNatiVPredictor",
    "ESM2PriorPredictor",
    "Predictor",
]
