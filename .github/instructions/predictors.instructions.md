---
applyTo: "vhh_library/stability.py,vhh_library/nativeness.py,vhh_library/esm_scorer.py,vhh_library/calibration.py,vhh_library/developability.py,vhh_library/orthogonal_scoring.py"
---

# Predictor code-path instructions

These rules apply to every file that implements a scoring backend — stability,
nativeness, surface hydrophobicity, ESM-2 PLL, and any future predictor
(NanoMelt, ESM-2 prior, etc.).

---

## Scorer interface contract

Every scorer class must expose:

```text
score(vhh: VHHSequence) -> dict          # at least {"composite_score": float}
predict_mutation_effect(vhh, pos, aa) -> float   # delta score
```

`composite_score` inside each scorer's return dict is a **per-axis normalised
value in [0, 1]**.  It is not the final ranking score — that is computed
externally by the mutation engine or the app layer.

The top-level `combined_score` that appears in library DataFrames is a
**temporary compatibility output**.  New predictor code must not depend on it and
must not write to it.  The long-term plan is to replace it with a
multi-objective ranking layer that consumes individual axis scores.

---

## Lazy model loading

Models must be loaded on first inference, never at import time or in `__init__`.

Pattern:

```text
def __init__(self, ...):
    self._model = None          # placeholder

def _ensure_model(self):
    if self._model is None:
        self._model = <load>    # heavy work here
    return self._model
```

`NativenessScorer` already follows this pattern (`_load_scoring_fn`).
`ESMStabilityScorer` must do the same.  Any new predictor (NanoMelt, ESM-2
prior) must follow it from the start.

---

## Device-aware backends

Every scorer that wraps a PyTorch model must accept `device: str = "auto"` in
its constructor.

Resolution order for `"auto"`:
1. CUDA if `torch.cuda.is_available()`
2. MPS if `torch.backends.mps.is_available()` (Apple Silicon)
3. CPU

If a requested device is unavailable, fall back to CPU and log a warning.

Use the shared helper (planned: `vhh_library.device_utils.resolve_device`) once
it exists.  Until then, each scorer may inline the logic, but keep it consistent
with the resolution order above.

---

## Preparing for NanoMelt

NanoMelt is a thermal-stability (Tm) predictor that will eventually replace the
heuristic sub-scores in `StabilityScorer` (disulfide, aggregation, charge
balance, hallmark, hydrophobic core).

Migration plan:
1. Wrap NanoMelt in a new `NanoMeltScorer` class following the scorer interface
   above.
2. `StabilityScorer` keeps its heuristic path as the fallback.
3. When NanoMelt is available, `StabilityScorer` delegates Tm estimation to it
   instead of the PLL→Tm linear mapping.
4. The heuristic sub-scores become optional penalty/bonus modifiers, not the
   primary signal.

Do not remove or modify the existing heuristic sub-score methods until NanoMelt
is integrated and tested.

---

## Preparing for ESM-2 as optional prior

ESM-2 PLL is currently the primary stability signal.  In the new architecture it
becomes an **optional prior** — a language-model plausibility check that can
re-weight variants but is not required for ranking.

* Keep `esm_scorer.py` self-contained.
* Do not hardcode ESM-2 availability checks in `stability.py` — use the
  existing `esm_scorer` injection pattern (`StabilityScorer(esm_scorer=...)`).
* The app layer decides whether to instantiate and inject the ESM-2 scorer.

---

## Calibration

`calibration.py` stores PLL→Tm mapping parameters.  When NanoMelt lands, the
calibration file format may change.  Keep calibration loading behind a versioned
schema so old calibration files still parse.

---

## AbNatiV nativeness

`NativenessScorer` is already well-structured.  Key rules going forward:

* The VQ-VAE model weights are downloaded via `vhh-init` / `abnativ init`.
  Do not bundle weights in the repository.
* `score()` returns `{"composite_score": float}` — this is the per-axis
  nativeness value, not a combined ranking score.
* Batch scoring (`score_batch`) should remain available for library-generation
  throughput.

---

## Testing predictors

* Mock heavy models in unit tests (ESM-2, AbNatiV).  Use lightweight stubs that
  return deterministic scores.
* Integration tests that load real models go in a separate `tests/integration/`
  directory and are skipped in CI unless the models are present.
* Every scorer must have at least:
  - A test for `score()` on a known-good VHH sequence.
  - A test for `predict_mutation_effect()` confirming the delta sign is correct
    for a known stabilising/destabilising mutation.
  - A test confirming graceful fallback when the model is unavailable.
