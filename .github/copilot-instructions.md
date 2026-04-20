# Copilot Instructions — VHH Biosimilar Library Generator

## Project overview

This is a Streamlit-based computational pipeline for designing producible VHH
(nanobody) variant libraries.  The pipeline scores sequences with multiple
predictors (ESM-2 stability, AbNatiV nativeness), ranks single-point mutations,
and generates combinatorial variant libraries.

The codebase is entering a **staged refactor**.  Every PR must keep the
Streamlit app (`app.py`) fully functional.  Do not break runtime behavior.

---

## Staged-refactor ground rules

1. **Small, testable PRs.**  Each pull request should touch one logical concern.
   Prefer many small PRs over one large one.  Every PR must include or update
   tests for the code it changes.

2. **Backward compatibility.**  Existing public function signatures, session-state
   keys, and CSV/FASTA export formats must continue to work unless a deprecation
   path has been agreed upon.  Use `warnings.warn(..., DeprecationWarning)` for
   any signature that will be removed.

3. **Streamlit must keep working.**  After every change, `streamlit run app.py`
   must launch without errors.  Never remove a session-state key that `app.py`
   reads unless you replace it in the same PR.

4. **Do not change runtime behavior yet.**  These instruction files describe the
   *target architecture*.  Until a PR explicitly implements a change, the old
   behavior is the correct behavior.

---

## Canonical residue identifiers

**IMGT position strings are the canonical residue IDs throughout the codebase.**

`VHHSequence.imgt_numbered` is a `dict[str, str]` (e.g. `{"1": "E", "2": "V",
..., "111A": "G"}`).  All new code must use these string keys — never raw
0-based indices — when referring to residue positions.  Existing integer-based
position arguments (`int | str`) must remain accepted for backward
compatibility, but internally convert to the string key immediately
(`pos_key = str(position)`).

---

## Model loading policy

All ML model objects (ESM-2, AbNatiV, any future predictor) **must be loaded
lazily**.

* The model is instantiated on first use, not at import time or `__init__` time.
* Store the loaded model in an instance attribute guarded by an `if self._model
  is None` check.
* Streamlit caching (`@st.cache_resource`) at the app layer is fine, but the
  library layer itself must not assume Streamlit is present.

---

## Device-aware backends

Every predictor that runs a PyTorch (or other GPU-capable) model must accept a
`device` parameter.

* Default: `device="auto"` — auto-detect CUDA/MPS/CPU.
* Must fall back to CPU gracefully if the requested device is unavailable.
* The `device` parameter must be an `__init__` argument, not a global setting.
* Use a shared helper (e.g. `vhh_library.device_utils.resolve_device`) for the
  detection logic so behavior is consistent across scorers.

---

## Scoring architecture direction

The pipeline is migrating toward three scoring axes:

| Axis | Backend | Status |
|------|---------|--------|
| **Stability** | NanoMelt (thermal Tm predictor) | Active — primary stability backend |
| **Nativeness** | AbNatiV VQ-VAE | Active — `nativeness.py` |
| **Prior / Language-model** | ESM-2 pseudo-log-likelihood | Active — optional prior (demoted); `esm_scorer.py` / `predictors/esm2_prior.py` |

`combined_score` is a **temporary compatibility output**.  It exists so that the
current Streamlit UI and downstream CSV exports continue to work.  It is *not*
the long-term ranking primitive.  Do not build new features on top of
`combined_score`; instead, consume the individual axis scores directly and
combine them at the ranking layer.

---

## CDR handling

CDR regions (CDR1, CDR2, CDR3 as defined by `IMGT_REGIONS`) **remain frozen by
default** in the upcoming redesign.  The mutation engine must not propose
mutations inside CDR loops unless the user explicitly unlocks them.  Any new
code that iterates over mutable positions must respect this freeze.

---

## Code style

* Python ≥ 3.10; use `X | Y` union syntax, not `Union[X, Y]`.
* Line length 120 (`ruff` config in `pyproject.toml`).
* Lint with `ruff check`; format with `ruff format`.
* Type annotations on all public functions; use `from __future__ import
  annotations` at the top of every module.
* `__slots__` on data-heavy classes (see `VHHSequence`).
* Prefer `functools.cached_property` for expensive derived attributes.

---

## Testing

* Framework: `pytest`.
* Test directory: `tests/`.
* Run: `pytest` (after `pip install -e ".[dev]"`).
* Every PR that adds or changes logic must include tests covering the change.
