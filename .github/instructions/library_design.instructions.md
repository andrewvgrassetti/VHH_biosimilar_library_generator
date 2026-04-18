---
applyTo: "vhh_library/mutation_engine.py,vhh_library/sequence.py,vhh_library/library_manager.py,app.py"
---

# Library-design code-path instructions

These rules apply to the mutation engine, sequence representation, library
manager, and the Streamlit app layer — everything involved in proposing
mutations, building variant libraries, and presenting results.

---

## IMGT position strings are canonical

`VHHSequence.imgt_numbered` maps IMGT position string keys (`"1"`, `"27"`,
`"111A"`, …) to single-character amino acids.  All mutation-engine code must
identify residues by these string keys.

* `MutationEngine` already receives and returns IMGT position integers for its
  public API.  Internally, convert to the string key immediately:
  `pos_key = str(position)`.
* DataFrame columns that hold position identifiers must use the IMGT string
  form (e.g. `"imgt_position"`), not 0-based indices.
* When displaying positions in the Streamlit UI, always label them as IMGT
  positions.

---

## CDR regions are frozen by default

CDR1, CDR2, and CDR3 (as defined in `sequence.IMGT_REGIONS`) must **not** be
mutated unless the user explicitly opts in.  The mutation engine already supports
an off-limits mechanism — ensure it defaults to freezing all CDR positions.

In the upcoming redesign:
* The default off-limits set must be computed from `VHHSequence.cdr_positions`.
* A UI toggle in the sidebar will allow the user to unlock individual CDRs.
* Any new mutation-proposal code must check the off-limits set before suggesting
  a substitution.

---

## combined_score is temporary

`combined_score` appears in library DataFrames and in the Streamlit results
table.  It is a weighted sum of normalised stability and nativeness scores and
exists **only for backward compatibility** with the current UI and CSV exports.

Rules:
* Do not build new ranking logic on top of `combined_score`.
* New code should read the individual axis columns (`stability_score`,
  `nativeness_score`, and — when available — `nanomelt_tm`, `esm2_pll`) and
  combine them at the ranking layer.
* When the multi-objective ranking layer is implemented, `combined_score` will
  be deprecated with a `DeprecationWarning` and eventually removed.

---

## Mutation engine design

`MutationEngine` is the largest module in the library.  Refactoring rules:

1. **One concern per PR.**  Do not mix scoring changes with strategy changes.
2. **Keep the three strategies.**  Exhaustive, random sampling, and iterative
   anchor-and-explore must all remain functional after every PR.
3. **Respect the scorer injection pattern.**  `MutationEngine.__init__` receives
   scorer instances; it must not import or instantiate them itself.
4. **PTM liability checks stay.**  `_introduces_ptm_liability` is a hard gate —
   do not relax it without explicit approval.
5. **Anchor threshold is user-controlled.**  The sidebar exposes
   `anchor_threshold`; the engine must not override it.

---

## Sequence representation

`VHHSequence` uses `__slots__` and `cached_property` for memory and speed.
Rules:

* Do not add new instance attributes without adding them to `__slots__`.
  (`__dict__` is in `__slots__` to support `cached_property`.)
* The fast-mutation class method (`VHHSequence.mutate`) bypasses ANARCI
  re-numbering.  This is intentional for batch mutagenesis performance.  Do not
  add validation logic to this path — validation belongs at the engine level
  before calling `mutate`.
* If you need to add new cached properties, follow the existing pattern in
  `regions`, `cdr_positions`, `framework_positions`.

---

## Streamlit app rules

* `app.py` must remain the single entry point (`streamlit run app.py`).
* Heavy objects are cached via `@st.cache_resource` in `load_scorers()`.
  Add new scorers there, not inline in page functions.
* Session-state keys must not be renamed or removed without a migration path in
  the same PR.
* All sidebar controls must have explicit `key=` arguments so Streamlit state
  survives reruns.

---

## Library output format

The library DataFrame returned by `MutationEngine.generate_library` must
include at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `variant_id` | str | Unique identifier |
| `aa_sequence` | str | Full amino-acid sequence |
| `mutations` | str | Comma-separated mutation strings (`"X1Y, A2B"`) |
| `stability_score` | float | Per-axis stability (0–1) |
| `nativeness_score` | float | Per-axis nativeness (0–1) |
| `combined_score` | float | Temporary weighted composite (0–1) |

Additional columns (`esm2_pll`, `predicted_tm`, `ptm_liabilities`, …) are
optional.  Do not remove existing columns without a deprecation cycle.

---

## Testing library-design code

* Unit-test mutation ranking with a mock scorer that returns deterministic
  scores, so tests are fast and reproducible.
* Test each library-generation strategy independently (exhaustive, random,
  iterative).
* Verify that CDR-frozen positions are never present in the `mutations` column
  of generated variants.
* Verify that `combined_score` equals the expected weighted sum of axis scores
  within floating-point tolerance.
