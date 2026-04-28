"""Main Streamlit application for VHH Biosimilar Library Generator."""

import json
import logging
import tempfile
import time
import traceback
import warnings

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
# Suppress noisy ``transformers`` lazy-module __path__ warnings triggered by
# Streamlit's file-watcher (see PR #50 context — ML backend noise suppression).
warnings.filterwarnings("ignore", message=r"Accessing `__path__`", category=FutureWarning)
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from vhh_library.background import (
    STATUS_IDLE,
    get_task_log,
    get_task_progress,
    get_task_status,
    is_task_running,
    make_progress_callback,
    make_progress_setter,
    recover_task,
    render_task_status,
    reset_task,
    submit_task,
)
from vhh_library.barcodes import BarcodeGenerator
from vhh_library.calibration import (
    load_calibration as _load_calibration,
)
from vhh_library.calibration import (
    reset_calibration as _reset_calibration,
)
from vhh_library.calibration import (
    run_calibration as _run_calibration,
)
from vhh_library.codon_optimizer import CodonOptimizer
from vhh_library.components.sequence_selector import imgt_key_int_part, sequence_selector
from vhh_library.developability import SurfaceHydrophobicityScorer
from vhh_library.library_manager import LibraryManager
from vhh_library.mutation_engine import MutationEngine, _total_grouped_combinations
from vhh_library.nativeness import NativenessScorer
from vhh_library.orthogonal_scoring import (
    ConsensusStabilityScorer,
)
from vhh_library.position_classifier import PositionClassifier
from vhh_library.position_policy import (
    DesignPolicy,
)
from vhh_library.position_policy import (
    PositionClass as _PositionClass,
)
from vhh_library.position_policy import (
    to_off_limits as _to_off_limits,
)
from vhh_library.runtime_config import (
    VALID_DEVICES,
    VALID_NATIVENESS_BACKENDS,
    VALID_STABILITY_BACKENDS,
    RuntimeConfig,
    resolve_device,
)
from vhh_library.sequence import IMGT_REGIONS, VHHSequence
from vhh_library.stability import (
    StabilityScorer,
    _esm2_pll_available,
)
from vhh_library.tags import TagManager
from vhh_library.utils import (
    AMINO_ACIDS as _ALL_AAS,
)
from vhh_library.utils import (
    DEFAULT_CONSERVATIVE_FALLBACK as _CONSERVATIVE_FALLBACK,
)
from vhh_library.utils import (
    SIMILAR_AA_GROUPS as _SIMILAR_AA_GROUPS,
)
from vhh_library.visualization import SequenceVisualizer

logger = logging.getLogger(__name__)

# Ensure vhh_library logs are visible in the terminal during Streamlit runs
# so users can see library-generation progress even when the UI progress bar
# callback is slow to update.
logging.basicConfig(format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.WARNING)
logging.getLogger("vhh_library").setLevel(logging.INFO)

# Explicit stderr handler guarantees terminal visibility even if Streamlit
# intercepts or redirects default log output.
_stderr_handler = logging.StreamHandler()
_stderr_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
_stderr_handler.setLevel(logging.INFO)
logging.getLogger("vhh_library").addHandler(_stderr_handler)

_ESM2_PLL_DEFAULT_TOP_N = 10

st.set_page_config(
    page_title="VHH Biosimilar Library Generator",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


def _build_runtime_config() -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` from sidebar session-state values.

    Falls back to environment variables, then to hardcoded defaults, so
    the app works identically to the old behaviour when no sidebar
    selection has been made yet.
    """
    device = st.session_state.get("cfg_device", "auto")
    stability_backend = st.session_state.get("cfg_stability_backend", "nanomelt")
    nativeness_backend = st.session_state.get("cfg_nativeness_backend", "abnativ")
    return RuntimeConfig(
        device=device,
        stability_backend=stability_backend,
        nativeness_backend=nativeness_backend,
    )


def _show_abnativ_weights_error(exc: FileNotFoundError) -> None:
    """Display a user-friendly Streamlit error for missing AbNatiV weights."""
    logger.warning("AbNatiV model weights missing: %s", exc)
    st.error(
        "⚠️ AbNatiV model weights not found. "
        "Please run `vhh-init` (or `abnativ init`) to download "
        "the pre-trained model, then restart the app."
    )


@st.cache_resource
def load_scorers(
    device: str = "auto",
    stability_backend: str = "esm2",
    _model_tier: str = "auto",
):
    """Cache heavy scorer initialization.

    Parameters are cache keys so that changing backend/device in the
    sidebar triggers a fresh load.
    """
    resolved = resolve_device(device)

    # Create ESM-2 scorer if needed and available
    esm_scorer = None
    if stability_backend in ("esm2", "both") and _esm2_pll_available():
        try:
            from vhh_library.esm_scorer import ESMStabilityScorer

            esm_scorer = ESMStabilityScorer(model_tier=_model_tier, device=resolved)
        except Exception as exc:
            logger.warning("ESM-2 scorer could not be initialised: %s", exc)

    # Create NanoMelt predictor if needed and available
    nanomelt_predictor = None
    if stability_backend in ("nanomelt", "both"):
        try:
            from vhh_library.predictors.nanomelt import NanoMeltPredictor

            nanomelt_predictor = NanoMeltPredictor(device=resolved)
            # Pre-load the ESM model and transfer it to GPU in the main
            # thread.  Library generation runs in a daemon thread where
            # first-time CUDA context initialisation can hang.
            nanomelt_predictor.warm_up()
        except ImportError:
            logger.warning("NanoMelt is not installed; falling back to ESM-2/heuristic scoring.")
        except Exception as exc:
            logger.warning("NanoMelt scorer could not be initialised: %s", exc)

    s = StabilityScorer(esm_scorer=esm_scorer, nanomelt_predictor=nanomelt_predictor)
    hydro = SurfaceHydrophobicityScorer()
    cons = ConsensusStabilityScorer()
    nativeness_scorer = NativenessScorer()
    return s, hydro, cons, esm_scorer, nativeness_scorer


def load_calibration_data() -> dict | None:
    """Load calibration data for display (not cached – reads fresh each time)."""
    return _load_calibration()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state():
    defaults = {
        "vhh_seq": None,
        "stability_scores": None,
        "nativeness_scores": None,
        "hydrophobicity_scores": None,
        "orthogonal_stability_scores": None,
        "ranked_mutations": None,
        "library": None,
        "construct": None,
        "constructs": [],
        "esm2_pll_scores": None,
        "esm2_top_n": _ESM2_PLL_DEFAULT_TOP_N,
        "library_manager": LibraryManager(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-restore from disk if critical state is missing but an auto-save
    # exists.  This guards against Streamlit session resets caused by
    # WebSocket disconnects, server restarts, or browser refreshes.
    if st.session_state.get("vhh_seq") is None:
        _try_auto_restore()

    # Recover orphaned background-task results that completed while the
    # WebSocket was disconnected (e.g. laptop sleep/wake cycle).  Each
    # task maps to a specific session-state key.
    _recover_background_tasks()

    # Clean up stale checkpoint files (>24 h) on each session start.
    try:
        from vhh_library.checkpoint import cleanup_stale_checkpoints

        cleanup_stale_checkpoints(Path(tempfile.gettempdir()))
    except Exception:
        logger.debug("Checkpoint cleanup failed", exc_info=True)


# ---------------------------------------------------------------------------
# Session auto-save / auto-restore
# ---------------------------------------------------------------------------

_AUTOSAVE_DIR = Path(tempfile.gettempdir()) / "vhh_autosave"

# Keys that are persisted to the auto-save file.
_AUTOSAVE_KEYS: list[str] = [
    "vhh_seq",
    "stability_scores",
    "nativeness_scores",
    "hydrophobicity_scores",
    "orthogonal_stability_scores",
    "ranked_mutations",
    "library",
    "constructs",
]


def serialize_session_data(state: dict) -> dict:
    """Convert session-state values to a JSON-serialisable dict.

    * ``VHHSequence`` → raw amino-acid string.
    * ``DataFrame`` → list of dicts (``orient="records"``).
    * Everything else is kept as-is (must already be JSON-serialisable).
    """
    out: dict = {}
    for key, val in state.items():
        if val is None:
            continue
        if isinstance(val, VHHSequence):
            out[key] = {"__type__": "VHHSequence", "sequence": val.sequence}
        elif isinstance(val, pd.DataFrame):
            out[key] = {"__type__": "DataFrame", "records": val.to_dict(orient="records")}
        else:
            out[key] = val
    return out


def deserialize_session_data(data: dict) -> dict:
    """Inverse of :func:`serialize_session_data`.

    Wraps raw strings back into ``VHHSequence`` and record-lists back into
    ``DataFrame`` objects.
    """
    out: dict = {}
    for key, val in data.items():
        if isinstance(val, dict) and "__type__" in val:
            type_tag = val["__type__"]
            if type_tag == "VHHSequence":
                try:
                    out[key] = VHHSequence(val["sequence"])
                except Exception:
                    logger.warning("Auto-restore: failed to reconstruct VHHSequence for key '%s'", key)
            elif type_tag == "DataFrame":
                out[key] = pd.DataFrame(val.get("records", []))
            else:
                logger.warning("Auto-restore: unknown type tag '%s' for key '%s'", type_tag, key)
        else:
            out[key] = val
    return out


def _auto_save_path() -> Path:
    """Return the path to the auto-save JSON file."""
    return _AUTOSAVE_DIR / "autosave.json"


def auto_save_session() -> None:
    """Persist critical session-state keys to disk.

    Called after each major operation (sequence analysis, mutation ranking,
    library generation) so that the session can be recovered automatically
    if Streamlit loses its server-side state.
    """
    snapshot: dict = {}
    for key in _AUTOSAVE_KEYS:
        val = st.session_state.get(key)
        if val is not None:
            snapshot[key] = val
    if not snapshot:
        return
    try:
        _AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        serialised = serialize_session_data(snapshot)
        _auto_save_path().write_text(json.dumps(serialised))
    except Exception:
        logger.warning("Auto-save failed", exc_info=True)


def _try_auto_restore() -> None:
    """Restore session state from the auto-save file if it exists.

    Only called when ``vhh_seq`` is ``None`` (i.e. the session appears
    fresh).  If the file is stale (>24 h) it is ignored and cleaned up.
    """
    path = _auto_save_path()
    if not path.is_file():
        return
    try:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > 86400:  # 24 hours
            path.unlink(missing_ok=True)
            return

        raw = json.loads(path.read_text())
        restored = deserialize_session_data(raw)
        if not restored:
            return

        for key, val in restored.items():
            st.session_state[key] = val

        st.session_state["_auto_restored"] = True
        logger.info("Auto-restored session state from %s", path)
    except Exception:
        logger.warning("Auto-restore failed", exc_info=True)


# Background task names and the session-state keys they populate.
_RECOVERABLE_TASKS: dict[str, str] = {
    "library_gen": "library",
    "construct_build": "constructs",
    "rank_mutations": "ranked_mutations",
}


def _recover_background_tasks() -> None:
    """Check for orphaned background-task results and restore them.

    Called during :func:`init_state`.  For each recoverable task, if a
    persisted result file exists on disk and the corresponding session-state
    key is still empty, the result is loaded into session state and a flag
    is set so the UI can inform the user.

    Only tasks with IDLE status are eligible for recovery.  A non-IDLE
    status (RUNNING, DONE, ERROR) means the task is still tracked in the
    current session and :func:`render_task_status` will handle it on
    this rerun.  Recovering here would race with that function — the
    recovery resets the task to IDLE before ``render_task_status`` can
    return the result to the caller, making library generation appear to
    hang with no feedback.
    """
    for task_name, state_key in _RECOVERABLE_TASKS.items():
        # Only recover tasks whose session-state status is IDLE.  Non-IDLE
        # means the current session is actively tracking the task and
        # render_task_status() will handle the result/error.
        if get_task_status(task_name) != STATUS_IDLE:
            continue

        # Skip if the session already has a result for this task.
        # DataFrames expose .empty; other types (lists, dicts) do not.
        existing = st.session_state.get(state_key)
        if existing is not None:
            if not hasattr(existing, "empty") or not existing.empty:
                continue

        result = recover_task(task_name)
        if result is not None:
            st.session_state[state_key] = result
            st.session_state["_bg_recovered"] = task_name
            auto_save_session()
            # Safe to clear the persisted file: the result is now held in
            # session_state and (best-effort) in the auto-save file.
            reset_task(task_name)
            logger.info("Recovered background task %r result into session key %r.", task_name, state_key)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_off_limit_csv(uploaded_file) -> dict[str, set[str]]:
    """Parse CSV with 2 columns: original_aa, forbidden_aas.

    Column1 = single AA letter, column2 = forbidden replacement AAs
    (e.g. "VIL" or "V,I,L"). Returns dict mapping AA -> set of forbidden AAs.
    """
    result: dict[str, set[str]] = {}
    content = uploaded_file.getvalue().decode("utf-8")
    for line in content.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) < 2:
            continue
        original_aa = parts[0].strip().upper()
        if len(original_aa) != 1:
            continue
        forbidden_str = parts[1].strip().upper().replace(",", "")
        forbidden = {ch for ch in forbidden_str if ch.isalpha()}
        if original_aa and forbidden:
            result.setdefault(original_aa, set()).update(forbidden)
    return result


def _aa_forbidden_to_position_forbidden(
    aa_forbidden: dict[str, set[str]],
    imgt_numbered: dict[str, str],
) -> dict[str, set[str]]:
    """Convert AA-level forbidden substitutions to position-level using IMGT numbering."""
    position_forbidden: dict[str, set[str]] = {}
    for pos, aa in imgt_numbered.items():
        if aa in aa_forbidden:
            position_forbidden[pos] = aa_forbidden[aa]
    return position_forbidden


def _library_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _library_to_fasta(df: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in df.iterrows():
        vid = row.get("variant_id", "variant")
        seq = row.get("aa_sequence", "")
        lines.append(f">{vid}")
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    return "\n".join(lines)


def _build_original_scores(
    stability_scores: dict | None,
    nativeness_scores: dict | None,
    hydrophobicity_scores: dict | None,
    engine: object | None = None,
) -> dict[str, float | None]:
    """Build a column-name → value mapping of the original sequence's scores.

    The returned keys match the library DataFrame column names so they can be
    used directly as reference-line values on the score-distribution
    histograms.
    """
    original: dict[str, float | None] = {}
    if stability_scores is not None:
        original["stability_score"] = stability_scores.get("composite_score")
        original["predicted_tm"] = stability_scores.get("predicted_tm")
        original["nanomelt_tm"] = stability_scores.get("nanomelt_tm")
    if nativeness_scores is not None:
        original["nativeness_score"] = nativeness_scores.get("composite_score")
    if hydrophobicity_scores is not None:
        original["surface_hydrophobicity_score"] = hydrophobicity_scores.get("composite_score")

    # Derive combined_score using the engine's weighting when available,
    # falling back to an equal-weight average.
    _stab = original.get("stability_score")
    _nat = original.get("nativeness_score")
    _sh = original.get("surface_hydrophobicity_score")
    if engine is not None and hasattr(engine, "_combined_score") and _stab is not None:
        _raw: dict[str, float] = {
            "stability": _stab,
            "nativeness": _nat if _nat is not None else 0.0,
            "surface_hydrophobicity": _sh if _sh is not None else 0.0,
        }
        original["combined_score"] = engine._combined_score(_raw)
    elif _stab is not None and _nat is not None:
        original["combined_score"] = (_stab + _nat) / 2.0
    elif _stab is not None:
        original["combined_score"] = _stab

    return original


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        # -- Backend & Device --
        st.subheader("Backend & Device")
        st.selectbox(
            "Stability backend",
            options=sorted(VALID_STABILITY_BACKENDS),
            index=sorted(VALID_STABILITY_BACKENDS).index("nanomelt"),
            key="cfg_stability_backend",
            help="nanomelt = NanoMelt Tm predictor (default), esm2 = ESM-2 PLL scoring, both = ensemble",
        )
        st.selectbox(
            "Nativeness backend",
            options=sorted(VALID_NATIVENESS_BACKENDS),
            index=0,
            key="cfg_nativeness_backend",
            help="AbNatiV VQ-VAE nativeness scoring",
        )
        st.selectbox(
            "Device",
            options=sorted(VALID_DEVICES),
            index=sorted(VALID_DEVICES).index("auto"),
            key="cfg_device",
            help="auto = detect CUDA/MPS/CPU; use 'cuda' on AWS GPU instances",
        )

        # Surface predictor availability
        resolved = resolve_device(st.session_state.get("cfg_device", "auto"))
        st.caption(f"Resolved device: **{resolved}**")

        stab_backend = st.session_state.get("cfg_stability_backend", "nanomelt")
        if stab_backend in ("nanomelt", "both"):
            try:
                import nanomelt  # noqa: F401

                st.success("✅ NanoMelt available (primary)")
            except ImportError:
                st.warning('⚠️ NanoMelt not installed — pip install ".[nanomelt]". Falling back to ESM-2/heuristic.')
        if stab_backend in ("esm2", "both"):
            if _esm2_pll_available():
                st.success("✅ ESM-2 available")
            else:
                st.warning("⚠️ ESM-2 unavailable — install torch + fair-esm. Falling back to heuristic scoring.")
        try:
            import abnativ  # noqa: F401

            st.success("✅ AbNatiV available")
        except ImportError:
            st.warning("⚠️ AbNatiV unavailable — run vhh-init to download model weights.")

        st.divider()

        # -- Scoring Weights --
        st.subheader("Scoring Weights")
        enable_stability = st.checkbox("Enable stability", value=True, key="enable_stability")
        st.slider(
            "Stability weight",
            0.0,
            1.0,
            0.70,
            0.05,
            disabled=not enable_stability,
            key="w_stability",
        )
        enable_hydrophobicity = st.checkbox(
            "Enable surface hydrophobicity",
            value=False,
            key="enable_hydrophobicity",
        )
        st.slider(
            "Surface hydrophobicity weight",
            0.0,
            1.0,
            0.00,
            0.05,
            disabled=not enable_hydrophobicity,
            key="w_hydrophobicity",
        )
        st.slider(
            "Nativeness weight (AbNatiV)",
            0.0,
            1.0,
            0.30,
            0.05,
            key="w_nativeness",
        )

        st.divider()

        # -- Library Generation --
        st.subheader("Library Generation")
        st.number_input(
            "Min mutations",
            1,
            20,
            1,
            key="min_mutations",
            help=(
                "Minimum number of mutations per variant. This is limited by the "
                "number of unique positions in the 'Top N mutations for library' "
                "selection — you cannot mutate more positions than are available."
            ),
        )
        st.number_input(
            "Max mutations (n_mutations)",
            1,
            20,
            3,
            key="n_mutations",
            help="Maximum number of mutations per variant.",
        )
        st.number_input(
            "Max variants",
            100,
            100_000,
            1000,
            step=100,
            key="max_variants",
            help=(
                "Target number of variants to generate. Actual count may be lower "
                "if the combinatorial search space (determined by available positions "
                "and mutation options) is smaller than this value."
            ),
        )
        st.number_input(
            "Max candidates per position",
            1,
            5,
            3,
            key="max_candidates_per_position",
            help="Number of amino acid options to consider at each IMGT position",
        )
        st.selectbox("Strategy", ["Auto", "Random", "Iterative"], key="strategy")
        st.slider("Anchor threshold", 0.0, 1.0, 0.6, 0.05, key="anchor_threshold")
        st.number_input("Max rounds", 1, 30, 15, key="max_rounds")
        st.number_input(
            "Rescore top N (iterative)",
            0,
            200,
            20,
            key="rescore_top_n",
            help=(
                "After each iterative round, re-score top N variants for accuracy. "
                "Uses the active stability backend (NanoMelt by default, ESM-2 full PLL when selected). "
                "Set to 0 to disable."
            ),
        )

        st.divider()

        # -- Codon Optimization --
        st.subheader("Codon Optimization")

        _organism_options = [
            "e_coli",
            "h_sapiens",
            "s_cerevisiae",
            "p_pastoris",
            "b_subtilis",
            "m_musculus",
            "d_melanogaster",
            "c_elegans",
        ]
        organism_choice = st.selectbox(
            "Host organism",
            _organism_options + ["Advanced: enter taxonomy ID"],
            key="host_organism_select",
        )
        if organism_choice == "Advanced: enter taxonomy ID":
            st.text_input("Taxonomy ID or organism name", value="", key="host_organism_custom")

        st.radio(
            "Codon strategy",
            ["most_frequent", "harmonized", "gc_balanced", "dnachisel_optimized"],
            key="codon_strategy",
        )

        # DnaChisel-specific options (visible when dnachisel_optimized is selected)
        if st.session_state.get("codon_strategy") == "dnachisel_optimized":
            st.markdown("**DnaChisel constraints**")
            st.checkbox("Avoid BamHI / EcoRI / HindIII / NdeI", value=True, key="dc_avoid_common_enzymes")
            st.checkbox("Avoid BsaI / BpiI (Golden Gate)", value=True, key="dc_avoid_golden_gate")
            st.checkbox("Enforce GC content window (30–65 %)", value=True, key="dc_gc_window")
            st.checkbox("Suppress repeats (UniquifyAllKmers k=9)", value=True, key="dc_uniquify")

        st.divider()

        # -- ESM-2 Prior (Optional) --
        with st.expander("ESM-2 Prior (Optional)", expanded=False):
            esm2_available = _esm2_pll_available()
            esm2_active = stab_backend in ("esm2", "both") and esm2_available
            st.caption("ESM-2 is a supplementary language-model prior. NanoMelt Tm is the primary stability signal.")
            st.selectbox(
                "Model tier",
                options=["auto", "t6_8M", "t12_35M", "t33_650M", "t36_3B"],
                index=0,
                key="esm2_model_tier",
                disabled=not esm2_active,
                help="auto = t6_8M on CPU, t33_650M on GPU",
            )

        st.divider()

        # -- Stability Calibration --
        with st.expander("🔧 Stability Calibration"):
            cal = load_calibration_data()
            if cal is not None:
                n_vhhs = len(cal.get("calibration_vhhs", []))
                created = cal.get("created_at", "unknown")
                params = cal.get("parameters", {})
                slope = params.get("pll_to_tm_slope", "?")
                intercept = params.get("pll_to_tm_intercept", "?")
                tm_min = params.get("tm_ideal_min", "?")
                tm_max = params.get("tm_ideal_max", "?")
                st.success(f"✅ Calibrated from {n_vhhs} VHHs ({created})")
                st.markdown(f"**Slope:** {slope}  \n**Intercept:** {intercept}")
                st.markdown(f"**Tm range:** {tm_min}–{tm_max} °C")
            else:
                st.warning(
                    "⚠️ Using default parameter estimates — upload reference VHHs with known Tm values to calibrate"
                )

            st.markdown("**Upload calibration CSV** (columns: `name` (optional), `sequence`, `experimental_tm`)")
            cal_csv = st.file_uploader(
                "Calibration CSV",
                type=["csv"],
                key="calibration_csv",
                label_visibility="collapsed",
            )

            if st.button("Run Calibration", key="btn_run_calibration", disabled=cal_csv is None):
                if cal_csv is not None:
                    try:
                        cal_df = pd.read_csv(cal_csv)
                        required = {"sequence", "experimental_tm"}
                        if not required.issubset(set(cal_df.columns)):
                            st.error(f"CSV must contain columns: {required}")
                        else:
                            seqs = cal_df["sequence"].tolist()
                            tms = cal_df["experimental_tm"].astype(float).tolist()
                            names = cal_df["name"].tolist() if "name" in cal_df.columns else None
                            with st.spinner("Running calibration…"):
                                result = _run_calibration(seqs, tms, names)
                            st.success(
                                f"Calibration complete! R²={result.r_squared:.3f}, "
                                f"n={result.n_samples}, slope={result.pll_to_tm_slope:.2f}, "
                                f"intercept={result.pll_to_tm_intercept:.2f}, "
                                f"Tm range={result.tm_ideal_min:.1f}–{result.tm_ideal_max:.1f} °C"
                            )
                            st.cache_resource.clear()
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Calibration failed: {exc}")
                        with st.expander("🔍 Error details"):
                            st.code(traceback.format_exc(), language="python")

            if st.button("Reset to Defaults", key="btn_reset_calibration"):
                _reset_calibration()
                st.cache_resource.clear()
                st.info("Calibration reset to defaults.")
                st.rerun()

        st.divider()


# ---------------------------------------------------------------------------
# Tab 1 – Input & Analysis
# ---------------------------------------------------------------------------


def tab_input(stability_scorer, nativeness_scorer, hydrophobicity_scorer, consensus_scorer, viz):
    st.header("🔬 Input & Analysis")

    raw_seq = st.text_area(
        "Paste your VHH amino acid sequence",
        height=120,
        placeholder="Paste VHH sequence here...",
        key="raw_seq_input",
    )

    if st.button("Analyse sequence", type="primary", key="btn_analyse"):
        seq_str = raw_seq.strip().replace("\n", "").replace(" ", "").upper()
        if not seq_str:
            st.error("Please enter a sequence.")
            return

        vhh = VHHSequence(seq_str)
        vr = vhh.validation_result

        if not vr["valid"]:
            for e in vr["errors"]:
                st.error(e)
            for w in vr["warnings"]:
                st.warning(w)
            return

        for w in vr["warnings"]:
            st.warning(w)

        st.session_state["vhh_seq"] = vhh

        with st.spinner("Scoring…"):
            st.session_state["stability_scores"] = stability_scorer.score(vhh)
            try:
                st.session_state["nativeness_scores"] = nativeness_scorer.score(vhh)
            except FileNotFoundError as exc:
                _show_abnativ_weights_error(exc)
                st.session_state["nativeness_scores"] = None
            st.session_state["hydrophobicity_scores"] = hydrophobicity_scorer.score(vhh)
            st.session_state["orthogonal_stability_scores"] = consensus_scorer.score(vhh)
        auto_save_session()

    # -- Display results --
    vhh = st.session_state.get("vhh_seq")
    if vhh is None:
        st.info("Enter a VHH sequence above and click **Analyse sequence**.")
        return

    s_scores = st.session_state["stability_scores"]
    nat_scores = st.session_state["nativeness_scores"]
    hy_scores = st.session_state["hydrophobicity_scores"]
    ort_s = st.session_state["orthogonal_stability_scores"]

    st.success(f"Sequence accepted – {vhh.length} residues")

    # Score bars
    st.subheader("Composite Scores")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(viz.render_score_bar(s_scores["composite_score"], "Stability", "#66BB6A"), unsafe_allow_html=True)
    with col2:
        nat_composite = nat_scores["composite_score"] if nat_scores else 0.0
        st.markdown(viz.render_score_bar(nat_composite, "Nativeness (AbNatiV)", "#AB47BC"), unsafe_allow_html=True)
    with col3:
        hydro_bar = viz.render_score_bar(
            hy_scores["composite_score"],
            "Surface Hydrophobicity",
            "#FFA726",
        )
        st.markdown(hydro_bar, unsafe_allow_html=True)

    # Orthogonal scores
    st.subheader("Orthogonal Scores")
    st.metric("Consensus Stability", f"{ort_s['composite_score']:.3f}")

    # Region track
    st.subheader("IMGT Region Map")
    st.markdown(viz.render_region_track(vhh), unsafe_allow_html=True)

    # Detailed breakdowns
    with st.expander("Stability Details"):
        detail_cols = [
            ("pI", "pI"),
            ("disulfide_score", "Disulfide"),
            ("vhh_hallmark_score", "VHH Hallmark"),
            ("aggregation_score", "Aggregation"),
            ("charge_balance_score", "Charge Balance"),
            ("hydrophobic_core_score", "Hydrophobic Core"),
        ]
        cols = st.columns(3)
        for idx, (key, label) in enumerate(detail_cols):
            with cols[idx % 3]:
                val = s_scores.get(key)
                if val is not None:
                    st.metric(label, f"{val:.3f}")
        st.write(f"**Scoring method:** {s_scores.get('scoring_method', 'legacy')}")
        if s_scores.get("predicted_tm") is not None:
            st.write(f"**Predicted Tm:** {s_scores['predicted_tm']:.1f} °C")
        if s_scores.get("warnings"):
            for w in s_scores["warnings"]:
                st.warning(w)

    with st.expander("Surface Hydrophobicity Details"):
        st.metric("Hydrophobic Patches", hy_scores.get("n_patches", 0))
        if hy_scores.get("warnings"):
            for w in hy_scores["warnings"]:
                st.warning(w)


# ---------------------------------------------------------------------------
# Tab 2 – Mutation Selection
# ---------------------------------------------------------------------------


def tab_mutations(stability_scorer):
    st.header("🎯 Mutation Selection")

    vhh = st.session_state.get("vhh_seq")
    if vhh is None:
        st.info("Please analyse a sequence first (Tab 1).")
        return

    # -- Region freeze checkboxes (seed the initial classification) --
    st.subheader("Region Freeze")
    checkbox_frozen: set[str] = set()
    region_cols = st.columns(len(IMGT_REGIONS))
    for idx, (region_name, (start, end)) in enumerate(IMGT_REGIONS.items()):
        with region_cols[idx]:
            default_off = region_name.startswith("CDR")
            if st.checkbox(region_name, value=default_off, key=f"ol_{region_name}"):
                for imgt_key in vhh.imgt_numbered:
                    if start <= imgt_key_int_part(imgt_key) <= end:
                        checkbox_frozen.add(imgt_key)

    # Build classifier-derived initial classes
    classifier = PositionClassifier()
    imgt_keys = list(vhh.imgt_numbered.keys())
    classifications = classifier.classify(imgt_keys)

    # Initial frozen/conservative sets from classifier
    classifier_frozen: set[str] = set()
    classifier_conservative: set[str] = set()
    for pos_key, clf in classifications.items():
        if clf.position_class is _PositionClass.FROZEN:
            classifier_frozen.add(pos_key)
        elif clf.position_class is _PositionClass.CONSERVATIVE:
            classifier_conservative.add(pos_key)

    # Merge checkbox state with classifier defaults.
    # Checkbox-frozen positions are added; CDR positions from unchecked
    # checkboxes that were frozen by the CDR rule become mutable (but NOT
    # structurally conserved Cys/Trp).
    initial_frozen = set(classifier_frozen)
    initial_frozen |= checkbox_frozen
    for pos_key in list(initial_frozen):
        if pos_key not in checkbox_frozen:
            clf = classifications.get(pos_key)
            if clf and clf.reason.rule == "cdr_freeze":
                initial_frozen.discard(pos_key)

    initial_conservative = set(classifier_conservative) - initial_frozen

    # Use session state to persist the selector's output across reruns.
    _state_key = "_position_classes"
    _checkbox_key = "_last_checkbox_frozen"

    prev_checkbox = st.session_state.get(_checkbox_key)
    checkboxes_changed = prev_checkbox is not None and prev_checkbox != checkbox_frozen
    st.session_state[_checkbox_key] = checkbox_frozen

    if _state_key not in st.session_state or checkboxes_changed:
        st.session_state[_state_key] = {
            "frozen": initial_frozen,
            "conservative": initial_conservative,
        }

    position_classes: dict[str, set[str]] = st.session_state[_state_key]
    frozen_positions: set[str] = position_classes.get("frozen", set())
    conservative_positions: set[str] = position_classes.get("conservative", set())

    # -- Forbidden substitutions CSV --
    st.subheader("Forbidden Substitutions")
    uploaded = st.file_uploader(
        "Upload CSV (original_aa, forbidden_aas)",
        type=["csv"],
        key="forbidden_csv",
    )
    aa_forbidden: dict[str, set[str]] = {}
    position_forbidden: dict[str, set[str]] = {}
    if uploaded is not None:
        aa_forbidden = _parse_off_limit_csv(uploaded)
        position_forbidden = _aa_forbidden_to_position_forbidden(aa_forbidden, vhh.imgt_numbered)
        st.write(f"Parsed {len(aa_forbidden)} AA-level rules → {len(position_forbidden)} position-level rules.")

    # -- Excluded target AAs --
    all_aas = sorted("ACDEFGHIKLMNPQRSTVWY")
    excluded_target_aas = st.multiselect(
        "Excluded target amino acids",
        all_aas,
        default=["C"],
        key="excluded_aas",
    )
    excluded_set = set(excluded_target_aas) if excluded_target_aas else None

    # -- Interactive selector --
    st.subheader("Interactive Position Selector")
    st.caption(
        "Click a residue to cycle: **mutable** → **frozen** → **conservative** → **mutable**. "
        "Drag to apply the same class to multiple positions."
    )
    selector_result = sequence_selector(
        sequence=vhh.sequence,
        imgt_numbered=vhh.imgt_numbered,
        frozen_positions=frozen_positions,
        conservative_positions=conservative_positions,
        key="seq_selector",
    )
    # The selector returns the authoritative classes when the user interacts.
    if selector_result is not None:
        new_frozen = set(selector_result.get("frozen", []))
        new_conservative = set(selector_result.get("conservative", []))
        if new_frozen != frozen_positions or new_conservative != conservative_positions:
            st.session_state[_state_key] = {
                "frozen": new_frozen,
                "conservative": new_conservative,
            }
            st.rerun()

    st.caption(
        f"Frozen: {len(frozen_positions)} · "
        f"Conservative: {len(conservative_positions)} · "
        f"Mutable: {len(imgt_keys) - len(frozen_positions) - len(conservative_positions)}"
    )

    # -- Position Policy Review/Edit (new design system) --
    with st.expander("📋 Position Policy (Advanced)", expanded=False):
        st.markdown(
            "Review and edit the position-level mutation policy. "
            "Positions are classified as **frozen** (no mutation), "
            "**conservative** (restricted AAs), or **mutable** (any AA). "
            "Toggle classes in the interactive selector above. "
            "The forbidden-substitutions CSV further restricts allowed AAs "
            "for non-frozen positions without changing their selector class."
        )

        # Build the policy from the selector's authoritative state.
        # The classifier is used only for allowed-AA sets and reason metadata.
        policy = DesignPolicy()
        for pos_key in imgt_keys:
            if pos_key in frozen_positions:
                policy.freeze([pos_key])
            elif pos_key in conservative_positions:
                # Use classifier's allowed AAs if available, else use
                # chemically-similar group based on the wild-type residue.
                clf = classifications.get(pos_key)
                if clf and clf.position_class is _PositionClass.CONSERVATIVE and clf.allowed_aas:
                    policy.restrict(pos_key, clf.allowed_aas)
                else:
                    wt_aa = vhh.imgt_numbered.get(pos_key, "A")
                    similar = _SIMILAR_AA_GROUPS.get(wt_aa, _CONSERVATIVE_FALLBACK)
                    policy.restrict(pos_key, similar)
            else:
                policy.make_mutable([pos_key])

        # The forbidden-substitutions CSV further restricts allowed AAs
        # for non-frozen positions.  It respects the user's interactive
        # selector choices: frozen positions stay frozen, conservative
        # positions have their allowed set narrowed, and mutable positions
        # become conservative only if the CSV removes some AAs.
        # The original (wild-type) AA is also excluded from the allowed
        # set since mutating to the same residue is not meaningful.
        if position_forbidden:
            for pos_key, forbidden_set in position_forbidden.items():
                existing = policy.get(pos_key)
                # Frozen positions from the selector stay frozen.
                if existing is not None and existing.is_frozen:
                    continue
                wt_aa = vhh.imgt_numbered.get(pos_key, "")
                if existing is not None and existing.is_conservative and existing.allowed_aas:
                    # Narrow the existing conservative set.
                    allowed = existing.allowed_aas - frozenset(forbidden_set) - frozenset({wt_aa})
                else:
                    # Mutable position: restrict from ALL_AAS.
                    allowed = _ALL_AAS - frozenset(forbidden_set) - frozenset({wt_aa})
                if allowed:
                    policy.restrict(pos_key, allowed)
                else:
                    policy.freeze([pos_key])

        # Store in session state for downstream use
        st.session_state["_design_policy"] = policy

        # Summary
        frozen_count = len(policy.frozen_positions())
        conservative_count = len(policy.conservative_positions())
        mutable_count = len(policy.mutable_positions())
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.metric("Frozen", frozen_count)
        with pc2:
            st.metric("Conservative", conservative_count)
        with pc3:
            st.metric("Mutable", mutable_count)

        # Policy detail table
        rows = []
        for pos_key in sorted(policy.policies, key=lambda k: (int("".join(c for c in k if c.isdigit()) or "0"), k)):
            pp = policy.policies[pos_key]
            wt_aa = vhh.imgt_numbered.get(pos_key, "?")
            reason = classifications.get(pos_key)
            reason_str = reason.reason.description if reason else ""
            rows.append(
                {
                    "IMGT Position": pos_key,
                    "WT AA": wt_aa,
                    "Class": pp.position_class.value,
                    "Allowed AAs": ", ".join(sorted(pp.allowed_aas)) if pp.allowed_aas else "",
                    "Reason": reason_str,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Import / Export
        st.markdown("---")
        ie1, ie2 = st.columns(2)
        with ie1:
            st.markdown("**Export policy**")
            export_fmt = st.radio(
                "Format",
                ["JSON", "YAML"],
                horizontal=True,
                key="policy_export_fmt",
            )
            policy_data = policy.to_dict()
            if export_fmt == "JSON":
                policy_bytes = json.dumps(policy_data, indent=2).encode("utf-8")
                st.download_button(
                    "📥 Download policy JSON",
                    data=policy_bytes,
                    file_name="position_policy.json",
                    mime="application/json",
                    key="dl_policy_json",
                )
            else:
                try:
                    import yaml

                    policy_bytes = yaml.dump(policy_data, default_flow_style=False).encode("utf-8")
                    st.download_button(
                        "📥 Download policy YAML",
                        data=policy_bytes,
                        file_name="position_policy.yaml",
                        mime="text/yaml",
                        key="dl_policy_yaml",
                    )
                except ImportError:
                    st.info("Install PyYAML for YAML export: pip install pyyaml")

        with ie2:
            st.markdown("**Import policy**")
            policy_file = st.file_uploader(
                "Upload JSON or YAML policy",
                type=["json", "yaml", "yml"],
                key="policy_upload",
            )
            if policy_file is not None:
                try:
                    content = policy_file.getvalue().decode("utf-8")
                    name = policy_file.name.lower()
                    if name.endswith(".json"):
                        imported_data = json.loads(content)
                    elif name.endswith((".yaml", ".yml")):
                        import yaml

                        imported_data = yaml.safe_load(content)
                    else:
                        st.error("Unsupported file format.")
                        imported_data = None

                    if imported_data is not None:
                        imported_policy = DesignPolicy.from_dict(imported_data)
                        st.session_state["_design_policy"] = imported_policy
                        st.success(
                            f"Imported policy: {len(imported_policy.frozen_positions())} frozen, "
                            f"{len(imported_policy.conservative_positions())} conservative, "
                            f"{len(imported_policy.mutable_positions())} mutable positions."
                        )
                except Exception as exc:
                    st.error(f"Failed to import policy: {exc}")
                    with st.expander("🔍 Error details"):
                        st.code(traceback.format_exc(), language="python")

    # -- Rank mutations --
    st.subheader("Rank Mutations")

    weights = {}
    enabled = {}
    if st.session_state.get("enable_stability", True):
        weights["stability"] = st.session_state.get("w_stability", 0.70)
        enabled["stability"] = True
    else:
        enabled["stability"] = False
    if st.session_state.get("enable_hydrophobicity", True):
        weights["surface_hydrophobicity"] = st.session_state.get("w_hydrophobicity", 0.00)
        enabled["surface_hydrophobicity"] = True
    else:
        enabled["surface_hydrophobicity"] = False
    weights["nativeness"] = st.session_state.get("w_nativeness", 0.30)
    enabled["nativeness"] = True

    cfg = _build_runtime_config()
    model_tier = st.session_state.get("esm2_model_tier", "auto")
    scorers = load_scorers(
        device=cfg.device,
        stability_backend=cfg.stability_backend,
        _model_tier=model_tier,
    )
    _, hydrophobicity_scorer, consensus_scorer, esm_scorer, nativeness_scorer = scorers

    _rank_running = is_task_running("rank_mutations")
    if st.button("Rank single mutations", type="primary", key="btn_rank", disabled=_rank_running):
        engine = MutationEngine(
            stability_scorer,
            nativeness_scorer,
            hydrophobicity_scorer=hydrophobicity_scorer if enabled.get("surface_hydrophobicity") else None,
            consensus_scorer=consensus_scorer,
            esm_scorer=esm_scorer,
            weights=weights,
            enabled_metrics=enabled,
        )
        # Derive legacy parameters from the design policy for backward
        # compatibility with rank_single_mutations.  The policy already
        # includes the CSV overrides (applied last), so these derived
        # parameters honour CSV as the ultimate arbiter.
        active_policy = st.session_state.get("_design_policy")
        if active_policy is not None:
            rank_off_limits, rank_forbidden = _to_off_limits(active_policy)
        else:
            rank_off_limits = frozen_positions
            rank_forbidden = {}

        _rank_progress_cb = make_progress_callback("rank_mutations")

        # Snapshot everything needed by the background thread
        _vhh = vhh
        _off_limits = rank_off_limits if rank_off_limits else None
        _forbidden = rank_forbidden if rank_forbidden else None
        _excluded = excluded_set
        _max_per_pos = st.session_state.get("max_candidates_per_position", 3)

        # Store the engine now so it is accessible when the background task
        # completes (which may be on a subsequent rerun outside this block).
        st.session_state["_mutation_engine"] = engine

        def _rank_work():
            return engine.rank_single_mutations(
                _vhh,
                off_limits=_off_limits,
                forbidden_substitutions=_forbidden,
                excluded_target_aas=_excluded,
                max_per_position=_max_per_pos,
                progress_callback=_rank_progress_cb,
            )

        print("[APP] Submitting rank_mutations task", flush=True)
        submit_task("rank_mutations", _rank_work)

    # Poll / display result for mutation ranking
    rank_result = render_task_status("rank_mutations", success_message="")

    # Diagnostic panel visible while ranking is running
    if is_task_running("rank_mutations"):
        with st.expander("🔧 Diagnostic Info", expanded=True):
            st.text(f"Task status: {get_task_status('rank_mutations')}")
            frac, text = get_task_progress("rank_mutations")
            st.text(f"Progress: {frac:.1%} — {text}")
            st.text(f"Log entries: {len(get_task_log('rank_mutations'))}")
            st.caption(
                "If this section shows 0% with no log entries for more than 30 seconds, "
                "check the terminal for [RANKING] and [TIMING] output."
            )

    if rank_result is not None:
        st.session_state["ranked_mutations"] = rank_result
        auto_save_session()
        reset_task("rank_mutations")
        st.success(f"Ranked {len(rank_result)} mutations.")

    ranked = st.session_state.get("ranked_mutations")
    if ranked is not None and not ranked.empty:
        # Highlight positions with multiple AA options.
        pos_counts = ranked["imgt_pos"].value_counts()
        multi_positions = set(pos_counts[pos_counts > 1].index)
        if multi_positions:
            st.caption(
                f"Positions with multiple candidates: "
                f"{', '.join(sorted(multi_positions, key=lambda p: int(''.join(c for c in p if c.isdigit()) or '0')))}"
            )
        st.dataframe(ranked, use_container_width=True, hide_index=True)

        top_n = st.slider(
            "Top N mutations for library",
            1,
            len(ranked),
            min(10, len(ranked)),
            key="top_n_muts",
            help=(
                "Number of top-ranked single-point mutations to include as "
                "candidates for library generation. Each mutation targets a "
                "specific IMGT position — the number of unique positions in "
                "this set determines the maximum mutations per variant. "
                "Increase this value to explore more positions and enable "
                "more mutations per variant and more total variants."
            ),
        )

        # Show user guidance about the search space.
        selected_top = ranked.head(top_n)
        n_unique_positions = selected_top["position"].nunique() if "position" in selected_top.columns else 0
        user_min_muts = st.session_state.get("min_mutations", 1)
        user_max_muts = st.session_state.get("n_mutations", 3)
        user_max_variants = st.session_state.get("max_variants", 1000)

        if n_unique_positions > 0:
            effective_k_max = min(user_max_muts, n_unique_positions)
            effective_k_min = min(user_min_muts, effective_k_max)

            if user_min_muts > n_unique_positions:
                st.warning(
                    f"⚠️ Min mutations ({user_min_muts}) exceeds available "
                    f"positions ({n_unique_positions}) in top {top_n} mutations. "
                    f"Each variant will have at most {n_unique_positions} mutations. "
                    f"Increase **Top N mutations** to include more positions.",
                    icon="⚠️",
                )

            # Estimate search space size using the shared utility.
            _position_groups: dict = {}
            for row in selected_top.itertuples(index=False):
                _position_groups.setdefault(int(row.position), []).append(row)
            _search_space = _total_grouped_combinations(_position_groups, effective_k_min, effective_k_max)

            if _search_space < user_max_variants:
                st.info(
                    f"ℹ️ The search space contains ~{_search_space:,} unique "
                    f"combinations ({n_unique_positions} positions, "
                    f"{effective_k_min}–{effective_k_max} mutations each), "
                    f"which is less than the requested {user_max_variants:,} variants. "
                    f"Increase **Top N mutations** to expand the search space.",
                    icon="ℹ️",
                )

            with st.expander("📈 Search space vs. Top N"):
                ns = []
                combos = []
                groups: dict[int, list] = {}
                for i, row in enumerate(ranked.itertuples(index=False), 1):
                    groups.setdefault(int(row.position), []).append(row)
                    n_pos = len(groups)
                    k_max = min(user_max_muts, n_pos)
                    k_min = min(user_min_muts, k_max)
                    combos.append(_total_grouped_combinations(groups, k_min, k_max))
                    ns.append(i)

                try:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(ns, combos, linewidth=1.5)
                    ax.axhline(
                        user_max_variants,
                        color="red",
                        linestyle="--",
                        label=f"Requested variants ({user_max_variants:,})",
                    )
                    ax.axvline(
                        top_n,
                        color="gray",
                        linestyle="--",
                        label=f"Current Top N ({top_n})",
                    )
                    ax.set_yscale("log")
                    ax.set_xlabel("Top N mutations")
                    ax.set_ylabel("Unique combinations")
                    ax.set_title("Search space size vs. Top N mutations")
                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception:
                    logger.warning("Failed to render search-space plot", exc_info=True)
                    st.warning("⚠️ Could not render search-space plot.")
                    with st.expander("🔍 Plot error details"):
                        st.code(traceback.format_exc(), language="python")

        _lib_running = is_task_running("library_gen")
        if st.button("Generate library", type="primary", key="btn_gen_lib", disabled=_lib_running):
            engine = st.session_state.get("_mutation_engine")
            if engine is None:
                st.error("Please rank mutations first.")
                return
            strategy_map = {"Auto": "auto", "Random": "random", "Iterative": "iterative"}
            strategy = strategy_map.get(st.session_state.get("strategy", "Auto"), "auto")

            _progress_cb = make_progress_callback("library_gen")
            _checkpoint_root = Path(tempfile.gettempdir())

            # Snapshot all session-state values now (in the main Streamlit
            # thread).  The background thread has no Streamlit script-run
            # context, so ``st.session_state`` is not accessible there.
            _n_mutations = st.session_state.get("n_mutations", 3)
            _max_variants = st.session_state.get("max_variants", 1000)
            _min_mutations = st.session_state.get("min_mutations", 1)
            _anchor_threshold = st.session_state.get("anchor_threshold", 0.6)
            _max_rounds = st.session_state.get("max_rounds", 15)
            _rescore_top_n = st.session_state.get("rescore_top_n", 20)
            _top_mutations = ranked.head(top_n)

            def _library_gen_work():
                return engine.generate_library(
                    vhh,
                    _top_mutations,
                    n_mutations=_n_mutations,
                    max_variants=_max_variants,
                    min_mutations=_min_mutations,
                    strategy=strategy,
                    anchor_threshold=_anchor_threshold,
                    max_rounds=_max_rounds,
                    rescore_top_n=_rescore_top_n,
                    progress_callback=_progress_cb,
                    checkpoint_dir=_checkpoint_root,
                )

            print(f"[APP] Submitting library_gen task (strategy={strategy})", flush=True)
            submit_task("library_gen", _library_gen_work)

        # Poll / display result for library generation
        library_result = render_task_status("library_gen", success_message="")

        # Diagnostic panel visible while library generation is running
        if is_task_running("library_gen"):
            with st.expander("🔧 Diagnostic Info", expanded=True):
                st.text(f"Task status: {get_task_status('library_gen')}")
                frac, text = get_task_progress("library_gen")
                st.text(f"Progress: {frac:.1%} — {text}")
                st.text(f"Log entries: {len(get_task_log('library_gen'))}")
                st.caption(
                    "If this section shows 0% with no log entries for more than 30 seconds, "
                    "check the terminal for [BG-THREAD] and [TIMING] output."
                )
        if library_result is not None:
            st.session_state["library"] = library_result
            st.session_state["esm2_pll_scores"] = None
            auto_save_session()
            reset_task("library_gen")
            _requested = st.session_state.get("max_variants", 1000)
            if len(library_result) < _requested:
                st.warning(
                    f"Generated {len(library_result)} of {_requested:,} requested variants. "
                    f"The search space was exhausted. Increase **Top N mutations for "
                    f"library** to expand the pool of candidate mutations and positions.",
                )
            else:
                st.success(f"Generated {len(library_result)} variants.")


# ---------------------------------------------------------------------------
# Tab 3 – Library Results
# ---------------------------------------------------------------------------


def tab_library(viz):
    st.header("📚 Library Results")

    library = st.session_state.get("library")
    if library is None or library.empty:
        st.info("Generate a library first (Tab 2).")
        return

    st.subheader("Library Overview")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Variants", len(library))
    with mc2:
        st.metric("Mean Combined Score", f"{library['combined_score'].mean():.3f}")
    with mc3:
        if "stability_score" in library.columns:
            st.metric("Mean Stability", f"{library['stability_score'].mean():.3f}")
    with mc4:
        if "nanomelt_tm" in library.columns and library["nanomelt_tm"].notna().any():
            st.metric("Mean NanoMelt Tm", f"{library['nanomelt_tm'].mean():.1f} °C")
        elif "predicted_tm" in library.columns and library["predicted_tm"].notna().any():
            st.metric("Mean Predicted Tm", f"{library['predicted_tm'].mean():.1f} °C")

    st.dataframe(library, use_container_width=True, hide_index=True)

    # -- Build original-sequence score lookup for reference lines --
    original_scores = _build_original_scores(
        stability_scores=st.session_state.get("stability_scores"),
        nativeness_scores=st.session_state.get("nativeness_scores"),
        hydrophobicity_scores=st.session_state.get("hydrophobicity_scores"),
        engine=st.session_state.get("_mutation_engine"),
    )

    # -- Distribution plots --
    st.subheader("Score Distributions")
    score_cols = [
        c
        for c in [
            "combined_score",
            "stability_score",
            "nativeness_score",
            "surface_hydrophobicity_score",
            "predicted_tm",
            "nanomelt_tm",
        ]
        if c in library.columns and library[c].notna().any()
    ]
    if score_cols:
        try:
            fig, axes = plt.subplots(1, len(score_cols), figsize=(4 * len(score_cols), 3))
            if len(score_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, score_cols):
                ax.hist(library[col].dropna(), bins=30, edgecolor="white", alpha=0.8)
                orig_val = original_scores.get(col)
                if orig_val is not None:
                    ax.axvline(
                        orig_val,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Original ({orig_val:.3f})",
                    )
                    ax.legend(fontsize=7)
                ax.set_title(col.replace("_", " ").title(), fontsize=10)
                ax.set_xlabel("Score")
                ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            logger.warning("Failed to render score distribution plots", exc_info=True)
            st.warning("⚠️ Could not render score distribution plots.")
            with st.expander("🔍 Plot error details"):
                st.code(traceback.format_exc(), language="python")

    # -- Correlation plot --
    if "nativeness_score" in library.columns and "stability_score" in library.columns:
        st.subheader("Nativeness vs Stability Correlation")
        n_vals = library["nativeness_score"].dropna()
        s_vals = library["stability_score"].dropna()
        common_idx = n_vals.index.intersection(s_vals.index)
        if len(common_idx) > 2:
            try:
                rho, pval = spearmanr(n_vals.loc[common_idx], s_vals.loc[common_idx])
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.scatter(n_vals.loc[common_idx], s_vals.loc[common_idx], alpha=0.4, s=10)
                orig_nat = original_scores.get("nativeness_score")
                orig_stab = original_scores.get("stability_score")
                if orig_nat is not None and orig_stab is not None:
                    ax2.scatter(
                        [orig_nat],
                        [orig_stab],
                        color="red",
                        marker="*",
                        s=200,
                        zorder=5,
                        label="Original",
                        edgecolors="black",
                        linewidths=0.5,
                    )
                    ax2.legend(fontsize=8)
                ax2.set_xlabel("Nativeness Score")
                ax2.set_ylabel("Stability Score")
                ax2.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.2e})")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception:
                logger.warning("Failed to render correlation plot", exc_info=True)
                st.warning("⚠️ Could not render nativeness vs stability correlation plot.")
                with st.expander("🔍 Plot error details"):
                    st.code(traceback.format_exc(), language="python")

    # -- Advanced Re-Ranking --
    st.subheader("Advanced Re-Ranking")

    # --- NanoMelt re-ranking (primary) ---
    st.markdown("**NanoMelt Tm Re-Ranking (Primary)**")
    if "nanomelt_tm" in library.columns and library["nanomelt_tm"].notna().any():
        st.success("✅ NanoMelt Tm scores are present in the library.")
        nm_cols = [c for c in ["variant_id", "aa_sequence", "combined_score", "nanomelt_tm"] if c in library.columns]
        st.dataframe(
            library[nm_cols].sort_values("nanomelt_tm", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "NanoMelt Tm scores are not yet available for this library. "
            "To include them, select **nanomelt** or **both** as the stability backend "
            "and regenerate the library."
        )
        top_n_nm = st.number_input(
            "Top N variants to score with NanoMelt",
            1,
            200,
            20,
            key="nanomelt_rerank_top_n",
        )
        _nm_running = is_task_running("nanomelt_rerank")
        if st.button("Score top variants with NanoMelt", key="btn_nanomelt_rerank", disabled=_nm_running):
            from vhh_library.stability import StabilityScorer

            subset = library.nlargest(top_n_nm, "combined_score")
            seqs = subset["aa_sequence"].tolist()
            variant_ids = subset["variant_id"].tolist()
            combined_scores = subset["combined_score"].tolist()

            # Capture a thread-safe progress setter — st.session_state is
            # not accessible from the background thread.
            _nm_set_progress = make_progress_setter("nanomelt_rerank")

            def _nanomelt_rerank_work():
                scorer = StabilityScorer(esm_scorer=None, device="auto")
                _nm_set_progress(0.1, "Computing NanoMelt Tm scores…")
                tm_scores = []
                for i, seq_str in enumerate(seqs):
                    vhh_tmp = VHHSequence(seq_str)
                    result = scorer.score(vhh_tmp)
                    tm_scores.append(result.get("nanomelt_tm"))
                    _nm_set_progress(
                        0.1 + 0.9 * (i + 1) / len(seqs),
                        f"Scoring variant {i + 1}/{len(seqs)}…",
                    )
                import pandas as _pd

                nm_df = _pd.DataFrame(
                    {
                        "variant_id": variant_ids,
                        "aa_sequence": seqs,
                        "combined_score": combined_scores,
                        "nanomelt_tm": tm_scores,
                    }
                )
                return nm_df.sort_values("nanomelt_tm", ascending=False)

            submit_task("nanomelt_rerank", _nanomelt_rerank_work)

        # Poll / display result for NanoMelt re-ranking
        nm_result = render_task_status("nanomelt_rerank", success_message="NanoMelt re-ranking complete.")
        if nm_result is not None:
            st.session_state["nanomelt_rerank_scores"] = nm_result
            reset_task("nanomelt_rerank")

        nm_df = st.session_state.get("nanomelt_rerank_scores")
        if nm_df is not None:
            st.dataframe(nm_df, use_container_width=True, hide_index=True)

    # --- ESM-2 re-ranking (supplementary) ---
    with st.expander("ESM-2 Re-Ranking (Supplementary)", expanded=False):
        st.caption(
            "ESM-2 PLL is an optional language-model plausibility check. "
            "NanoMelt Tm is the primary stability ranking signal. "
            "Scores below are for diagnostic purposes only and do not affect the primary ranking."
        )
        if _esm2_pll_available():
            # Show existing ESM-2 scores if already computed in the pipeline
            if "esm2_pll" in library.columns:
                st.success("✅ ESM-2 scores are integrated into the library scoring pipeline.")
                esm_cols = [
                    c
                    for c in [
                        "variant_id",
                        "aa_sequence",
                        "combined_score",
                        "esm2_pll",
                        "esm2_delta_pll",
                        "esm2_rank",
                    ]
                    if c in library.columns
                ]
                st.dataframe(
                    library[esm_cols].sort_values("esm2_pll", ascending=False).head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            elif "predicted_tm" in library.columns:
                st.info("ESM-2 is active in the stability scorer. Predicted Tm values are included above.")

            # User-controlled slider: up to the full library size
            lib_size = len(library)
            top_n_esm = st.slider(
                "Number of top variants to re-rank with ESM-2",
                min_value=1,
                max_value=max(lib_size, 1),
                value=min(st.session_state.get("esm2_top_n", _ESM2_PLL_DEFAULT_TOP_N), lib_size),
                key="esm2_top_n_slider",
                help=(
                    "Select how many top variants (ranked by combined score) to "
                    "re-rank with ESM-2 PLL. NanoMelt Tm is the primary stability "
                    "ranking signal; ESM-2 re-ranking is a supplementary diagnostic."
                ),
            )
            # Keep session-state key in sync for other consumers
            st.session_state["esm2_top_n"] = top_n_esm

            # Advanced: re-rank with a specific (potentially larger) model tier
            model_tier = st.session_state.get("esm2_model_tier", "auto")
            _esm_running = is_task_running("esm2_rerank")
            if st.button("Re-rank with ESM-2 (supplementary)", key="btn_esm2", disabled=_esm_running):
                from vhh_library.esm_scorer import ESMStabilityScorer

                subset = library.nlargest(top_n_esm, "combined_score")
                seqs = subset["aa_sequence"].tolist()
                variant_ids = subset["variant_id"].tolist()
                combined_scores = subset["combined_score"].tolist()

                # Capture a thread-safe progress setter — st.session_state
                # is not accessible from the background thread.
                _esm_set_progress = make_progress_setter("esm2_rerank")

                def _esm2_rerank_work():
                    _esm_set_progress(0.0, "Initialising ESM-2 model…")
                    scorer = ESMStabilityScorer(model_tier=model_tier, device="auto")
                    _esm_set_progress(0.1, "Computing ESM-2 PLL scores…")
                    pll_scores = scorer.score_batch(seqs)
                    _esm_set_progress(1.0, "Done!")
                    import pandas as _pd

                    pll_df = _pd.DataFrame(
                        {
                            "variant_id": variant_ids,
                            "aa_sequence": seqs,
                            "combined_score": combined_scores,
                            "esm2_pll": pll_scores,
                        }
                    )
                    return pll_df.sort_values("esm2_pll", ascending=False)

                submit_task("esm2_rerank", _esm2_rerank_work)

            # Poll / display result for ESM-2 re-ranking
            esm_result = render_task_status("esm2_rerank", success_message="ESM-2 re-ranking complete.")
            if esm_result is not None:
                st.session_state["esm2_pll_scores"] = esm_result
                reset_task("esm2_rerank")

            pll_df = st.session_state.get("esm2_pll_scores")
            if pll_df is not None:
                st.dataframe(pll_df, use_container_width=True, hide_index=True)

                # Spearman correlation between combined_score ranking and ESM-2 PLL ranking
                if "combined_score" in pll_df.columns and "esm2_pll" in pll_df.columns:
                    _cs = pll_df["combined_score"]
                    _pll = pll_df["esm2_pll"]
                    if len(_cs.dropna()) >= 3 and len(_pll.dropna()) >= 3:
                        try:
                            rho, pval = spearmanr(_cs, _pll)
                            st.markdown("#### Stability Ranking Correlation")
                            st.caption(
                                "Spearman rank correlation between the library's combined "
                                "stability score and the ESM-2 pseudo-log-likelihood (PLL). "
                                "A high ρ indicates the two ranking strategies agree on "
                                "which variants are most stable."
                            )
                            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                            ax_corr.scatter(_cs, _pll, alpha=0.6, edgecolors="white", linewidth=0.5)
                            ax_corr.set_xlabel("Combined Score (library ranking)")
                            ax_corr.set_ylabel("ESM-2 PLL")
                            ax_corr.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.2e})")
                            plt.tight_layout()
                            st.pyplot(fig_corr)
                            plt.close(fig_corr)
                        except Exception:
                            logger.warning("Failed to render ESM-2 correlation plot", exc_info=True)
        else:
            st.info("ESM-2 not available (torch / esm not found). Reinstall with: pip install -e .")

    # -- Downloads --
    st.subheader("Download Library")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 Download CSV",
            data=_library_to_csv(library),
            file_name="vhh_library.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "📥 Download FASTA",
            data=_library_to_fasta(library),
            file_name="vhh_library.fasta",
            mime="text/plain",
        )

    # -- Top 10 alignments --
    vhh = st.session_state.get("vhh_seq")
    if vhh is not None:
        st.subheader("Top 10 Variant Alignments")
        top10 = library.nlargest(10, "combined_score")
        for _, row in top10.iterrows():
            mut_seq = row.get("aa_sequence", "")
            vid = row.get("variant_id", "")
            mutations_str = row.get("mutations", "")
            mutation_info: dict[int, str] = {}
            if isinstance(mutations_str, str) and mutations_str:
                for m in mutations_str.split(";"):
                    m = m.strip()
                    if len(m) >= 3 and m[0].isalpha() and m[-1].isalpha():
                        try:
                            pos = int(m[1:-1])
                            mutation_info[pos] = m[-1]
                        except ValueError:
                            pass
            with st.expander(f"{vid} — score: {row.get('combined_score', 0):.3f}"):
                st.markdown(
                    viz.render_alignment(vhh, mut_seq, mutation_info),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab 4 – Barcoding
# ---------------------------------------------------------------------------


def tab_barcoding():
    st.header("🧬 Barcoding")

    library = st.session_state.get("library")
    if library is None or library.empty:
        st.info("Generate a library first (Tab 2).")
        return

    enable_bc = st.checkbox("Enable barcoding", value=False, key="enable_barcoding")
    if not enable_bc:
        st.info("Enable barcoding above to assign LC-MS/MS barcodes.")
        return

    st.subheader("Barcode Settings")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bc_top_n = st.number_input("Top N variants", 1, len(library), min(100, len(library)), key="bc_top_n")
    with bc2:
        bc_linker = st.text_input("Linker", value="GGS", key="bc_linker")
    with bc3:
        bc_tail = st.text_input("C-terminal tail", value="", key="bc_c_tail")

    if st.button("Assign barcodes", type="primary", key="btn_barcodes"):
        barcode_gen = BarcodeGenerator()
        with st.spinner("Assigning barcodes…"):
            barcoded = barcode_gen.assign_barcodes(
                library,
                top_n=bc_top_n,
                linker=bc_linker,
                c_terminal_tail=bc_tail,
            )
            ref_table = barcode_gen.generate_barcode_reference(barcoded)
        st.session_state["barcoded_library"] = barcoded
        st.session_state["barcode_ref"] = ref_table
        st.session_state["_barcode_gen"] = barcode_gen
        st.success(f"Assigned barcodes to {len(barcoded)} variants.")

    barcoded = st.session_state.get("barcoded_library")
    ref_table = st.session_state.get("barcode_ref")
    barcode_gen = st.session_state.get("_barcode_gen")

    if barcoded is not None:
        st.subheader("Barcoded Library")
        st.dataframe(barcoded, use_container_width=True, hide_index=True)

    if ref_table is not None:
        st.subheader("Barcode Reference Table")
        st.dataframe(ref_table, use_container_width=True, hide_index=True)

    if ref_table is not None and barcode_gen is not None:
        st.subheader("Barcode Distributions")
        try:
            fig = barcode_gen.plot_barcode_distributions(ref_table)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
        except Exception:
            logger.warning("Failed to render barcode distribution plots", exc_info=True)
            st.warning("⚠️ Could not render barcode distribution plots.")
            with st.expander("🔍 Plot error details"):
                st.code(traceback.format_exc(), language="python")

    # Downloads
    if barcoded is not None:
        st.subheader("Download Barcoded Data")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 Barcoded CSV",
                data=_library_to_csv(barcoded),
                file_name="vhh_barcoded_library.csv",
                mime="text/csv",
                key="dl_barcoded_csv",
            )
        with dl2:
            if barcode_gen is not None:
                fasta_str = barcode_gen.generate_barcoded_fasta(barcoded)
                st.download_button(
                    "📥 Barcoded FASTA",
                    data=fasta_str,
                    file_name="vhh_barcoded_library.fasta",
                    mime="text/plain",
                    key="dl_barcoded_fasta",
                )


# ---------------------------------------------------------------------------
# Tab 5 – Construct Builder
# ---------------------------------------------------------------------------


def tab_construct(optimizer, tag_manager):
    st.header("🔧 Construct Builder")

    vhh = st.session_state.get("vhh_seq")
    library = st.session_state.get("library")

    if vhh is None:
        st.info("Please analyse a sequence first (Tab 1).")
        return

    available_tags = tag_manager.get_available_tags()
    tag_names = ["None"] + list(available_tags.keys())

    # Source selection: use barcoded if available
    use_barcoded = False
    barcoded = st.session_state.get("barcoded_library")
    if barcoded is not None and not barcoded.empty:
        use_barcoded = st.checkbox("Use barcoded sequences", value=True, key="construct_use_bc")

    st.subheader("Tag Selection")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        n_tag = st.selectbox("N-terminal tag", tag_names, key="n_tag")
    with tc2:
        c_tag = st.selectbox("C-terminal tag", tag_names, key="c_tag")
    with tc3:
        linker = st.text_input("Linker sequence", value="GSGSGS", key="construct_linker")

    n_tag_val = n_tag if n_tag != "None" else None
    c_tag_val = c_tag if c_tag != "None" else None

    _construct_running = is_task_running("construct_build")
    if st.button("Build constructs", type="primary", key="btn_build_constructs", disabled=_construct_running):
        # Resolve host organism from sidebar widgets
        host_sel = st.session_state.get("host_organism_select", "e_coli")
        if host_sel == "Advanced: enter taxonomy ID":
            host = st.session_state.get("host_organism_custom", "e_coli") or "e_coli"
        else:
            host = host_sel
        codon_strat = st.session_state.get("codon_strategy", "most_frequent")

        # Build extra kwargs for dnachisel_optimized
        opt_kwargs: dict = {}
        if codon_strat == "dnachisel_optimized":
            enzymes: list[str] = []
            if st.session_state.get("dc_avoid_common_enzymes", True):
                enzymes.extend(["BamHI", "EcoRI", "HindIII", "NdeI"])
            if st.session_state.get("dc_avoid_golden_gate", True):
                enzymes.extend(["BsaI", "BpiI"])
            opt_kwargs["restriction_enzymes"] = enzymes
            if not st.session_state.get("dc_gc_window", True):
                opt_kwargs["gc_mini"] = 0.0
                opt_kwargs["gc_maxi"] = 1.0
            if not st.session_state.get("dc_uniquify", True):
                opt_kwargs["uniquify_kmers"] = None

        source_df = barcoded if (use_barcoded and barcoded is not None) else library

        # Snapshot rows for the background thread (avoid passing the full df)
        if source_df is not None and not source_df.empty:
            _rows = [
                (
                    row.get("variant_id", "variant"),
                    row.get("barcoded_sequence", row.get("aa_sequence", "")),
                )
                for _, row in source_df.iterrows()
            ]
        else:
            _rows = [("parent", vhh.sequence)]

        _is_library = source_df is not None and not source_df.empty

        # Capture a thread-safe progress setter — st.session_state is
        # not accessible from the background thread.
        _construct_set_progress = make_progress_setter("construct_build")

        def _construct_build_work():
            constructs_out: list[dict] = []
            total = len(_rows)
            for idx, (vid, aa_seq) in enumerate(_rows):
                opt = optimizer.optimize(aa_seq, host=host, strategy=codon_strat, **opt_kwargs)
                construct = tag_manager.build_construct(
                    aa_seq,
                    opt["dna_sequence"],
                    n_tag=n_tag_val,
                    c_tag=c_tag_val,
                    linker=linker,
                )
                constructs_out.append(
                    {
                        "variant_id": vid,
                        "aa_construct": construct["aa_construct"],
                        "dna_construct": construct["dna_construct"],
                        "schematic": construct["schematic"],
                        "gc_content": opt["gc_content"],
                        "cai": opt["cai"],
                    }
                )
                _construct_set_progress(
                    (idx + 1) / total,
                    f"Building construct {idx + 1}/{total}…",
                )
            return constructs_out

        submit_task("construct_build", _construct_build_work)

    # Poll / display result for construct building
    construct_result = render_task_status("construct_build", success_message="")
    if construct_result is not None:
        st.session_state["constructs"] = construct_result
        reset_task("construct_build")
        st.success(f"Built {len(construct_result)} construct(s).")

    constructs = st.session_state.get("constructs", [])
    if constructs:
        st.subheader("Construct Summary")
        summary_df = pd.DataFrame(constructs)
        st.dataframe(
            summary_df[["variant_id", "schematic", "gc_content", "cai"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Construct Details")
        for c in constructs:
            with st.expander(f"{c['variant_id']} — {c['schematic']}"):
                vid = c["variant_id"]
                st.text_area(
                    "AA construct",
                    c["aa_construct"],
                    height=80,
                    key=f"aa_{vid}",
                    disabled=True,
                )
                if c["dna_construct"]:
                    st.text_area(
                        "DNA construct",
                        c["dna_construct"],
                        height=80,
                        key=f"dna_{vid}",
                        disabled=True,
                    )
                st.write(f"**GC content:** {c['gc_content']:.2%} | **CAI:** {c['cai']:.3f}")

        st.subheader("Downloads")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 Constructs CSV",
                data=_library_to_csv(summary_df),
                file_name="vhh_constructs.csv",
                mime="text/csv",
                key="dl_constructs_csv",
            )
        with dl2:
            fasta_lines: list[str] = []
            for c in constructs:
                fasta_lines.append(f">{c['variant_id']}")
                seq = c["aa_construct"]
                for i in range(0, len(seq), 80):
                    fasta_lines.append(seq[i : i + 80])
            st.download_button(
                "📥 Constructs FASTA",
                data="\n".join(fasta_lines),
                file_name="vhh_constructs.fasta",
                mime="text/plain",
                key="dl_constructs_fasta",
            )


# ---------------------------------------------------------------------------
# Tab 6 – Session History
# ---------------------------------------------------------------------------


def tab_history():
    st.header("📁 Session History")

    lib_mgr: LibraryManager = st.session_state["library_manager"]

    # Save current session
    st.subheader("Save Current Session")
    if st.button("Save session", key="btn_save_session"):
        data: dict = {}
        for key in [
            "vhh_seq",
            "stability_scores",
            "nativeness_scores",
            "hydrophobicity_scores",
            "ranked_mutations",
            "library",
            "constructs",
        ]:
            val = st.session_state.get(key)
            if val is not None:
                if isinstance(val, VHHSequence):
                    data[key] = val.sequence
                elif isinstance(val, pd.DataFrame):
                    data[key] = val.to_dict(orient="records")
                else:
                    data[key] = val
        if data:
            result = lib_mgr.save_session(data)
            st.success(f"Session saved: {result.get('json', 'saved')}")
        else:
            st.warning("No data to save.")

    # Browse sessions
    st.subheader("Load Previous Session")
    sessions_dir = Path("sessions")
    if sessions_dir.exists():
        session_files = sorted(sessions_dir.glob("*.json"), reverse=True)
        if session_files:
            selected = st.selectbox(
                "Select session",
                session_files,
                format_func=lambda p: p.stem,
                key="session_select",
            )
            if st.button("Load session", key="btn_load_session"):
                loaded = lib_mgr.load_session(str(selected))
                if loaded:
                    # Restore sequence
                    if "vhh_seq" in loaded and isinstance(loaded["vhh_seq"], str):
                        st.session_state["vhh_seq"] = VHHSequence(loaded["vhh_seq"])
                    # Restore DataFrames
                    for key in ["ranked_mutations", "library"]:
                        if key in loaded and isinstance(loaded[key], list):
                            st.session_state[key] = pd.DataFrame(loaded[key])
                    # Restore dicts
                    for key in ["stability_scores", "nativeness_scores", "hydrophobicity_scores", "constructs"]:
                        if key in loaded:
                            st.session_state[key] = loaded[key]
                    st.success("Session loaded. Switch to other tabs to view results.")
                    st.rerun()
                else:
                    st.error("Failed to load session.")
        else:
            st.info("No saved sessions found.")
    else:
        st.info("No sessions directory found.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    init_state()

    # Show a one-time notification when session state was auto-restored.
    if st.session_state.pop("_auto_restored", False):
        st.toast("♻️ Session auto-restored from previous run.", icon="♻️")

    # Show a one-time notification when a background task result was recovered.
    recovered_task = st.session_state.pop("_bg_recovered", None)
    if recovered_task:
        _friendly = {"library_gen": "Library generation", "construct_build": "Construct building"}
        label = _friendly.get(recovered_task, recovered_task)
        st.toast(f"♻️ {label} result recovered after connection loss.", icon="♻️")

    # Sidebar must run first so that cfg_* session-state keys exist.
    sidebar()

    cfg = _build_runtime_config()
    model_tier = st.session_state.get("esm2_model_tier", "auto")
    scorers = load_scorers(
        device=cfg.device,
        stability_backend=cfg.stability_backend,
        _model_tier=model_tier,
    )
    (stability_scorer, hydrophobicity_scorer, consensus_scorer, esm_scorer, nativeness_scorer) = scorers
    optimizer = CodonOptimizer()
    tag_manager = TagManager()
    viz = SequenceVisualizer()

    tabs = st.tabs(
        [
            "🔬 Input & Analysis",
            "🎯 Mutation Selection",
            "📚 Library Results",
            "🧬 Barcoding",
            "🔧 Construct Builder",
            "📁 Session History",
        ]
    )
    with tabs[0]:
        tab_input(stability_scorer, nativeness_scorer, hydrophobicity_scorer, consensus_scorer, viz)
    with tabs[1]:
        tab_mutations(stability_scorer)
    with tabs[2]:
        tab_library(viz)
    with tabs[3]:
        tab_barcoding()
    with tabs[4]:
        tab_construct(optimizer, tag_manager)
    with tabs[5]:
        tab_history()


if __name__ == "__main__":
    main()
