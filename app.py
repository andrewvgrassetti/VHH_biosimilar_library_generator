"""Main Streamlit application for VHH Biosimilar Library Generator."""

import json
import logging

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

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
from vhh_library.visualization import SequenceVisualizer

logger = logging.getLogger(__name__)

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
    stability_backend = st.session_state.get("cfg_stability_backend", "esm2")
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
        "library_manager": LibraryManager(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
            index=sorted(VALID_STABILITY_BACKENDS).index("esm2"),
            key="cfg_stability_backend",
            help="esm2 = ESM-2 PLL scoring, nanomelt = NanoMelt Tm predictor, both = ensemble",
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

        stab_backend = st.session_state.get("cfg_stability_backend", "esm2")
        if stab_backend in ("esm2", "both"):
            if _esm2_pll_available():
                st.success("✅ ESM-2 available")
            else:
                st.warning("⚠️ ESM-2 unavailable — install torch + fair-esm. Falling back to heuristic scoring.")
        if stab_backend in ("nanomelt", "both"):
            try:
                import nanomelt  # noqa: F401

                st.success("✅ NanoMelt available")
            except ImportError:
                st.warning('⚠️ NanoMelt not installed — pip install ".[nanomelt]". Falling back to ESM-2/heuristic.')
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
            "Rescore top N (ESM-2 full PLL)",
            0,
            200,
            20,
            key="rescore_top_n",
            help="After each iterative round, re-score top N variants with full ESM-2 PLL for accuracy (0 = disabled)",
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

        # -- ESM-2 PLL --
        st.subheader("ESM-2 Model Settings")
        esm2_available = _esm2_pll_available()
        esm2_active = stab_backend in ("esm2", "both") and esm2_available
        st.selectbox(
            "Model tier",
            options=["auto", "t6_8M", "t12_35M", "t33_650M", "t36_3B"],
            index=0,
            key="esm2_model_tier",
            disabled=not esm2_active,
            help="auto = t6_8M on CPU, t33_650M on GPU",
        )
        st.slider(
            "Top N variants for advanced re-ranking",
            1,
            100,
            _ESM2_PLL_DEFAULT_TOP_N,
            key="esm2_top_n",
            disabled=not esm2_active,
            help="Number of top variants to re-rank with a larger ESM-2 model (Tab 3 advanced option)",
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
        placeholder="EVQLVESGGGLVQPGGSLRLSCAASGFTF...",
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

    # -- Off-limit regions --
    st.subheader("Off-Limit Regions")
    off_limit_positions: set[str] = set()
    region_cols = st.columns(len(IMGT_REGIONS))
    for idx, (region_name, (start, end)) in enumerate(IMGT_REGIONS.items()):
        with region_cols[idx]:
            default_off = region_name.startswith("CDR")
            if st.checkbox(region_name, value=default_off, key=f"ol_{region_name}"):
                # Collect all IMGT keys (including insertion codes) within this region.
                for imgt_key in vhh.imgt_numbered:
                    if start <= imgt_key_int_part(imgt_key) <= end:
                        off_limit_positions.add(imgt_key)

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
    selected_positions = sequence_selector(
        sequence=vhh.sequence,
        imgt_numbered=vhh.imgt_numbered,
        off_limit_positions=off_limit_positions,
        forbidden_substitutions=position_forbidden if position_forbidden else None,
        key="seq_selector",
    )
    if selected_positions:
        off_limit_positions.update(selected_positions)
    st.caption(f"Total off-limit positions: {len(off_limit_positions)}")

    # -- Position Policy Review/Edit (new design system) --
    with st.expander("📋 Position Policy (Advanced)", expanded=False):
        st.markdown(
            "Review and edit the position-level mutation policy. "
            "Positions are classified as **frozen** (no mutation), "
            "**conservative** (restricted AAs), or **mutable** (any AA)."
        )

        # Build the current policy from classifier + legacy controls
        classifier = PositionClassifier()
        imgt_keys = list(vhh.imgt_numbered.keys())
        classifications = classifier.classify(imgt_keys)
        policy = classifier.to_design_policy(imgt_keys)

        # Apply legacy off-limits on top
        if off_limit_positions:
            policy.freeze(off_limit_positions)
        if position_forbidden:
            from vhh_library.utils import AMINO_ACIDS as _ALL_AAS

            for pos_key, forbidden_set in position_forbidden.items():
                allowed = _ALL_AAS - frozenset(forbidden_set)
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

    if st.button("Rank single mutations", type="primary", key="btn_rank"):
        engine = MutationEngine(
            stability_scorer,
            nativeness_scorer,
            hydrophobicity_scorer=hydrophobicity_scorer if enabled.get("surface_hydrophobicity") else None,
            consensus_scorer=consensus_scorer,
            esm_scorer=esm_scorer,
            weights=weights,
            enabled_metrics=enabled,
        )
        with st.spinner("Ranking mutations…"):
            try:
                ranked = engine.rank_single_mutations(
                    vhh,
                    off_limits=off_limit_positions if off_limit_positions else None,
                    forbidden_substitutions=position_forbidden if position_forbidden else None,
                    excluded_target_aas=excluded_set,
                    max_per_position=st.session_state.get("max_candidates_per_position", 3),
                )
            except FileNotFoundError as exc:
                _show_abnativ_weights_error(exc)
                return
        st.session_state["ranked_mutations"] = ranked
        st.session_state["_mutation_engine"] = engine
        st.success(f"Ranked {len(ranked)} mutations.")

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
            min(50, len(ranked)),
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

        if st.button("Generate library", type="primary", key="btn_gen_lib"):
            engine = st.session_state.get("_mutation_engine")
            if engine is None:
                st.error("Please rank mutations first.")
                return
            strategy_map = {"Auto": "auto", "Random": "random", "Iterative": "iterative"}
            strategy = strategy_map.get(st.session_state.get("strategy", "Auto"), "auto")

            # Progress bar for iterative strategy
            progress_bar = st.progress(0, text="Generating library…")
            status_text = st.empty()

            def _on_progress(prog):
                frac = prog.round_number / max(prog.total_rounds, 1)
                progress_bar.progress(
                    min(frac, 1.0),
                    text=f"Phase: {prog.phase} — Round {prog.round_number}/{prog.total_rounds}",
                )
                status_text.caption(
                    f"🧬 {prog.population_size} variants | "
                    f"Best: {prog.best_score:.4f} | "
                    f"Anchors: {prog.n_anchors} | "
                    f"Diversity: {prog.diversity_entropy:.2f}"
                )

            with st.spinner("Generating library…"):
                try:
                    library = engine.generate_library(
                        vhh,
                        ranked.head(top_n),
                        n_mutations=st.session_state.get("n_mutations", 3),
                        max_variants=st.session_state.get("max_variants", 1000),
                        min_mutations=st.session_state.get("min_mutations", 1),
                        strategy=strategy,
                        anchor_threshold=st.session_state.get("anchor_threshold", 0.6),
                        max_rounds=st.session_state.get("max_rounds", 15),
                        rescore_top_n=st.session_state.get("rescore_top_n", 20),
                        progress_callback=_on_progress,
                    )
                except FileNotFoundError as exc:
                    _show_abnativ_weights_error(exc)
                    return
            progress_bar.progress(1.0, text="Complete!")
            st.session_state["library"] = library
            st.session_state["esm2_pll_scores"] = None
            _requested = st.session_state.get("max_variants", 1000)
            if len(library) < _requested:
                st.warning(
                    f"Generated {len(library)} of {_requested:,} requested variants. "
                    f"The search space was exhausted. Increase **Top N mutations for "
                    f"library** to expand the pool of candidate mutations and positions.",
                )
            else:
                st.success(f"Generated {len(library)} variants.")


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
        fig, axes = plt.subplots(1, len(score_cols), figsize=(4 * len(score_cols), 3))
        if len(score_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, score_cols):
            ax.hist(library[col].dropna(), bins=30, edgecolor="white", alpha=0.8)
            ax.set_title(col.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -- Correlation plot --
    if "nativeness_score" in library.columns and "stability_score" in library.columns:
        st.subheader("Nativeness vs Stability Correlation")
        n_vals = library["nativeness_score"].dropna()
        s_vals = library["stability_score"].dropna()
        common_idx = n_vals.index.intersection(s_vals.index)
        if len(common_idx) > 2:
            rho, pval = spearmanr(n_vals.loc[common_idx], s_vals.loc[common_idx])
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(n_vals.loc[common_idx], s_vals.loc[common_idx], alpha=0.4, s=10)
            ax2.set_xlabel("Nativeness Score")
            ax2.set_ylabel("Stability Score")
            ax2.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.2e})")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # -- ESM-2 Stability Scores --
    st.subheader("ESM-2 Stability Scoring")
    if _esm2_pll_available():
        # Show existing ESM-2 scores if already computed in the pipeline
        if "esm2_pll" in library.columns:
            st.success("✅ ESM-2 scores are integrated into the library scoring pipeline.")
            esm_cols = [
                c
                for c in ["variant_id", "aa_sequence", "combined_score", "esm2_pll", "esm2_delta_pll", "esm2_rank"]
                if c in library.columns
            ]
            st.dataframe(
                library[esm_cols].sort_values("esm2_pll", ascending=False).head(20),
                use_container_width=True,
                hide_index=True,
            )
        elif "predicted_tm" in library.columns:
            st.info("ESM-2 is active in the stability scorer. Predicted Tm values are included above.")

        # Advanced: re-rank with a specific (potentially larger) model tier
        with st.expander("Advanced: Re-rank top variants with a specific ESM-2 model"):
            model_tier = st.session_state.get("esm2_model_tier", "auto")
            top_n_esm = st.session_state.get("esm2_top_n", _ESM2_PLL_DEFAULT_TOP_N)
            if st.button("Re-rank with ESM-2", key="btn_esm2"):
                from vhh_library.esm_scorer import ESMStabilityScorer

                subset = library.nlargest(top_n_esm, "combined_score")
                seqs = subset["aa_sequence"].tolist()
                progress_bar = st.progress(0, text="Initialising ESM-2 model…")
                try:
                    scorer = ESMStabilityScorer(model_tier=model_tier, device="auto")
                    progress_bar.progress(10, text="Computing ESM-2 PLL scores…")
                    pll_scores = scorer.score_batch(seqs)
                    progress_bar.progress(100, text="Done!")
                except Exception as exc:
                    st.error(f"ESM-2 scoring failed: {exc}")
                    progress_bar.empty()
                    pll_scores = None

                if pll_scores is not None:
                    pll_df = subset[["variant_id", "aa_sequence", "combined_score"]].copy()
                    pll_df["esm2_pll"] = pll_scores
                    pll_df = pll_df.sort_values("esm2_pll", ascending=False)
                    st.session_state["esm2_pll_scores"] = pll_df
                    st.success("ESM-2 re-ranking complete.")

            pll_df = st.session_state.get("esm2_pll_scores")
            if pll_df is not None:
                st.dataframe(pll_df, use_container_width=True, hide_index=True)
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
        fig = barcode_gen.plot_barcode_distributions(ref_table)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)

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

    if st.button("Build constructs", type="primary", key="btn_build_constructs"):
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
        constructs: list[dict] = []

        if source_df is not None and not source_df.empty:
            with st.spinner("Building constructs…"):
                for _, row in source_df.iterrows():
                    aa_seq = row.get("barcoded_sequence", row.get("aa_sequence", ""))
                    vid = row.get("variant_id", "variant")
                    opt = optimizer.optimize(aa_seq, host=host, strategy=codon_strat, **opt_kwargs)
                    construct = tag_manager.build_construct(
                        aa_seq,
                        opt["dna_sequence"],
                        n_tag=n_tag_val,
                        c_tag=c_tag_val,
                        linker=linker,
                    )
                    constructs.append(
                        {
                            "variant_id": vid,
                            "aa_construct": construct["aa_construct"],
                            "dna_construct": construct["dna_construct"],
                            "schematic": construct["schematic"],
                            "gc_content": opt["gc_content"],
                            "cai": opt["cai"],
                        }
                    )
        else:
            # Single sequence mode
            with st.spinner("Optimizing codons…"):
                opt = optimizer.optimize(vhh.sequence, host=host, strategy=codon_strat, **opt_kwargs)
            construct = tag_manager.build_construct(
                vhh.sequence,
                opt["dna_sequence"],
                n_tag=n_tag_val,
                c_tag=c_tag_val,
                linker=linker,
            )
            constructs.append(
                {
                    "variant_id": "parent",
                    "aa_construct": construct["aa_construct"],
                    "dna_construct": construct["dna_construct"],
                    "schematic": construct["schematic"],
                    "gc_content": opt["gc_content"],
                    "cai": opt["cai"],
                }
            )
            if opt.get("warnings"):
                for w in opt["warnings"]:
                    st.warning(w)
            if opt.get("flagged_sites"):
                for f in opt["flagged_sites"]:
                    st.warning(f"Flagged site: {f}")

        st.session_state["constructs"] = constructs
        st.success(f"Built {len(constructs)} construct(s).")

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
# Tab 6 – Validation
# ---------------------------------------------------------------------------


def tab_validation(stability_scorer):
    st.header("📊 Validation & Benchmarking")
    st.markdown("Assess whether stability predictions correlate with real VHH thermal stability.")

    from vhh_library.benchmark import (
        compare_scoring_methods,
        load_benchmark_dataset,
        plot_correlation_scatter,
        plot_residuals,
        plot_scoring_comparison,
        run_benchmark,
        validate_library_predictions,
    )

    # --- Section 1: Benchmark on reference VHHs ---
    st.subheader("Benchmark on Reference VHHs")
    st.caption("Score the built-in benchmark VHH set and evaluate correlation with known Tm values.")

    cv_folds = st.slider("Cross-validation folds", min_value=2, max_value=10, value=5, key="bench_cv_folds")

    if st.button("Run benchmark on reference VHHs", key="btn_run_benchmark"):
        try:
            benchmark_vhhs = load_benchmark_dataset()
        except Exception as exc:
            st.error(f"Failed to load benchmark dataset: {exc}")
            return

        seqs = [v["sequence"] for v in benchmark_vhhs]
        exp_tms = [float(v["experimental_tm"]) for v in benchmark_vhhs]
        names = [v["name"] for v in benchmark_vhhs]

        # Compute scores for each benchmark VHH
        progress = st.progress(0, text="Scoring benchmark VHHs…")
        composite_scores = []
        predicted_tms = []
        esm_plls = []
        scoring_results: dict[str, list[float]] = {}

        for i, seq in enumerate(seqs):
            progress.progress((i + 1) / len(seqs), text=f"Scoring {names[i]}…")
            try:
                vhh = VHHSequence(seq)
                result = stability_scorer.score(vhh)
                composite_scores.append(result.get("composite_score", float("nan")))
                if "predicted_tm" in result:
                    predicted_tms.append(result["predicted_tm"])
                if "esm2_pll" in result:
                    esm_plls.append(result["esm2_pll"])
            except Exception:
                composite_scores.append(float("nan"))

        progress.progress(1.0, text="Done!")

        scoring_results["Composite Score"] = composite_scores
        if len(predicted_tms) == len(seqs):
            scoring_results["Predicted Tm"] = predicted_tms
        if len(esm_plls) == len(seqs):
            scoring_results["ESM-2 PLL"] = esm_plls

        # Compute per-residue PLLs from ESM data if available
        per_residue_plls = None
        if len(esm_plls) == len(seqs):
            per_residue_plls = [pll / max(len(seq), 1) for pll, seq in zip(esm_plls, seqs)]

        report = run_benchmark(
            benchmark_vhhs=benchmark_vhhs,
            per_residue_plls=per_residue_plls,
            composite_scores=composite_scores,
            cv_folds=cv_folds,
        )

        st.session_state["benchmark_report"] = report
        st.session_state["benchmark_scoring_results"] = scoring_results
        st.session_state["benchmark_exp_tms"] = exp_tms
        st.session_state["benchmark_composite_scores"] = composite_scores
        st.session_state["benchmark_predicted_tms"] = predicted_tms

    # Display results if available
    report = st.session_state.get("benchmark_report")
    if report is not None:
        m = report.correlation
        st.markdown("#### Correlation Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Spearman ρ", f"{m.spearman_rho:.3f}" if not _isnan(m.spearman_rho) else "N/A")
        with mc2:
            st.metric("Pearson r", f"{m.pearson_r:.3f}" if not _isnan(m.pearson_r) else "N/A")
        with mc3:
            st.metric("MAE", f"{m.mae:.2f}" if not _isnan(m.mae) else "N/A")
        with mc4:
            st.metric("RMSE", f"{m.rmse:.2f}" if not _isnan(m.rmse) else "N/A")

        mc5, mc6 = st.columns(2)
        with mc5:
            st.metric("Ranking Accuracy", f"{m.ranking_accuracy:.1%}" if not _isnan(m.ranking_accuracy) else "N/A")
        with mc6:
            st.metric("N Samples", m.n_samples)

        # Scatter plot
        exp_tms = st.session_state.get("benchmark_exp_tms", [])
        composite_scores = st.session_state.get("benchmark_composite_scores", [])
        predicted_tms = st.session_state.get("benchmark_predicted_tms", [])

        if predicted_tms and len(predicted_tms) == len(exp_tms):
            pred_for_plot, label = predicted_tms, "Predicted Tm (°C)"
        elif composite_scores and len(composite_scores) == len(exp_tms):
            pred_for_plot, label = composite_scores, "Composite Score"
        else:
            pred_for_plot, label = None, ""

        if pred_for_plot is not None:
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_correlation_scatter(
                    pred_for_plot,
                    exp_tms,
                    metrics=m,
                    xlabel=label,
                    ylabel="Experimental Tm (°C)",
                )
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                fig2 = plot_residuals(pred_for_plot, exp_tms)
                st.pyplot(fig2)
                plt.close(fig2)

        # Cross-validation
        if report.cross_validation is not None:
            cv = report.cross_validation
            st.markdown("#### Cross-Validation Results")
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                st.metric("Mean R²", f"{cv.mean_r2:.3f} ± {cv.std_r2:.3f}")
            with cv2:
                st.metric("Mean MAE", f"{cv.mean_mae:.2f} ± {cv.std_mae:.2f}")
            with cv3:
                st.metric("Folds", cv.k)

        # LOO predictions
        if report.loo_predictions:
            st.markdown("#### Leave-One-Out Predictions")
            loo_data = [
                {
                    "Name": p.name,
                    "Experimental Tm": f"{p.experimental_tm:.1f}",
                    "Predicted Tm": f"{p.predicted_tm:.1f}",
                    "Residual": f"{p.residual:.2f}",
                    "Calibration R²": f"{p.calibration_r2:.3f}" if not _isnan(p.calibration_r2) else "N/A",
                }
                for p in report.loo_predictions
            ]
            st.dataframe(pd.DataFrame(loo_data), use_container_width=True, hide_index=True)

        # Scoring comparison
        scoring_results = st.session_state.get("benchmark_scoring_results")
        if scoring_results and len(scoring_results) > 1 and exp_tms:
            st.markdown("#### Scoring Method Comparison")
            comparison = compare_scoring_methods(
                exp_tms,
                scoring_results,
            )
            if comparison:
                comp_data = [
                    {
                        "Method": name,
                        "Spearman ρ": f"{cm.spearman_rho:.3f}",
                        "Pearson r": f"{cm.pearson_r:.3f}",
                        "MAE": f"{cm.mae:.2f}",
                        "RMSE": f"{cm.rmse:.2f}",
                        "Ranking Accuracy": f"{cm.ranking_accuracy:.1%}",
                    }
                    for name, cm in comparison.items()
                ]
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
                fig3 = plot_scoring_comparison(comparison)
                st.pyplot(fig3)
                plt.close(fig3)

    # --- Section 2: Upload experimental results ---
    st.divider()
    st.subheader("Validate Library Against Experimental Results")
    st.caption(
        "Upload a CSV with `variant_id` and `experimental_tm` (or `ranking`) "
        "columns to validate your library's predictions."
    )

    library = st.session_state.get("library")
    exp_file = st.file_uploader(
        "Upload experimental results CSV",
        type=["csv"],
        key="exp_results_upload",
    )

    if exp_file is not None and library is not None and not library.empty:
        pred_col = st.selectbox(
            "Predicted score column",
            [c for c in library.columns if library[c].dtype in ("float64", "float32", "int64")],
            key="val_pred_col",
        )
        if st.button("Validate predictions", key="btn_validate_lib"):
            try:
                import io

                exp_csv = io.StringIO(exp_file.getvalue().decode("utf-8"))
                lib_metrics = validate_library_predictions(library, exp_csv, predicted_col=pred_col)
                st.session_state["lib_validation_metrics"] = lib_metrics
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

        lib_metrics = st.session_state.get("lib_validation_metrics")
        if lib_metrics is not None:
            st.markdown("#### Library Validation Results")
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.metric("Spearman ρ", f"{lib_metrics.spearman_rho:.3f}")
            with lc2:
                st.metric("Pearson r", f"{lib_metrics.pearson_r:.3f}")
            with lc3:
                st.metric("MAE", f"{lib_metrics.mae:.2f}")
            with lc4:
                st.metric("Ranking Accuracy", f"{lib_metrics.ranking_accuracy:.1%}")
    elif exp_file is not None and (library is None or library.empty):
        st.warning("Generate a library first (Tab 2) to validate against experimental results.")


def _isnan(value: float) -> bool:
    """Check if a float value is NaN."""
    import math

    return math.isnan(value) if isinstance(value, float) else False


# ---------------------------------------------------------------------------
# Tab 7 – Session History
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
            "📊 Validation",
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
        tab_validation(stability_scorer)
    with tabs[6]:
        tab_history()


if __name__ == "__main__":
    main()
