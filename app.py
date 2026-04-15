"""Main Streamlit application for VHH Biosimilar Library Generator."""


import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from vhh_library.barcodes import BarcodeGenerator
from vhh_library.codon_optimizer import CodonOptimizer
from vhh_library.components.sequence_selector import sequence_selector
from vhh_library.developability import SurfaceHydrophobicityScorer
from vhh_library.humanness import HumAnnotator
from vhh_library.library_manager import LibraryManager
from vhh_library.mutation_engine import MutationEngine
from vhh_library.orthogonal_scoring import (
    ConsensusStabilityScorer,
    HumanStringContentScorer,
    NanoMeltStabilityScorer,
)
from vhh_library.sequence import IMGT_REGIONS, VHHSequence
from vhh_library.stability import (
    StabilityScorer,
    _esm2_pll_available,
    _nanomelt_available,
)
from vhh_library.tags import TagManager
from vhh_library.visualization import SequenceVisualizer

_ESM2_PLL_DEFAULT_TOP_N = 10
_NANOMELT_TM_MIN = 40.0
_NANOMELT_TM_MAX = 90.0

st.set_page_config(
    page_title="VHH Biosimilar Library Generator",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_scorers():
    """Cache heavy scorer initialization."""
    h = HumAnnotator()
    s = StabilityScorer()
    hydro = SurfaceHydrophobicityScorer()
    hsc = HumanStringContentScorer()
    cons = ConsensusStabilityScorer()
    nano = NanoMeltStabilityScorer()
    return h, s, hydro, hsc, cons, nano


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "vhh_seq": None,
        "humanness_scores": None,
        "stability_scores": None,
        "hydrophobicity_scores": None,
        "orthogonal_humanness_scores": None,
        "orthogonal_stability_scores": None,
        "nanomelt_scores": None,
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

        # -- Scoring Weights --
        st.subheader("Scoring Weights")
        enable_humanness = st.checkbox("Enable humanness", value=True, key="enable_humanness")
        st.slider(
            "Humanness weight", 0.0, 1.0, 0.35, 0.05,
            disabled=not enable_humanness, key="w_humanness",
        )
        enable_stability = st.checkbox("Enable stability", value=True, key="enable_stability")
        st.slider(
            "Stability weight", 0.0, 1.0, 0.50, 0.05,
            disabled=not enable_stability, key="w_stability",
        )
        enable_hydrophobicity = st.checkbox(
            "Enable surface hydrophobicity", value=True, key="enable_hydrophobicity",
        )
        st.slider(
            "Surface hydrophobicity weight", 0.0, 1.0, 0.15, 0.05,
            disabled=not enable_hydrophobicity, key="w_hydrophobicity",
        )

        st.divider()

        # -- Library Generation --
        st.subheader("Library Generation")
        st.number_input("Min mutations", 1, 20, 1, key="min_mutations")
        st.number_input("Max mutations (n_mutations)", 1, 20, 3, key="n_mutations")
        st.number_input("Max variants", 100, 100_000, 1000, step=100, key="max_variants")
        st.selectbox("Strategy", ["Auto", "Random", "Iterative"], key="strategy")
        st.slider("Anchor threshold", 0.0, 1.0, 0.6, 0.05, key="anchor_threshold")
        st.number_input("Max rounds", 1, 20, 5, key="max_rounds")

        st.divider()

        # -- Codon Optimization --
        st.subheader("Codon Optimization")

        _organism_options = [
            "e_coli", "h_sapiens", "s_cerevisiae", "p_pastoris",
            "b_subtilis", "m_musculus", "d_melanogaster", "c_elegans",
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
        st.subheader("ESM-2 Stability Scoring")
        esm2_available = _esm2_pll_available()
        st.checkbox(
            "Enable ESM-2 stability scoring",
            value=False,
            disabled=not esm2_available,
            key="enable_esm2_pll",
            help=(
                "Uses ESM-2 protein language model for stability assessment. "
                "Computationally expensive – recommended for final library ranking."
            )
            if esm2_available
            else "Requires torch and fair-esm (included in default install)",
        )
        st.selectbox(
            "Model tier",
            options=["auto", "t6_8M", "t12_35M", "t33_650M", "t36_3B"],
            index=0,
            key="esm2_model_tier",
            disabled=not esm2_available,
            help="auto = t6_8M on CPU, t33_650M on GPU",
        )
        st.slider(
            "Top N variants for PLL",
            1, 100, _ESM2_PLL_DEFAULT_TOP_N, key="esm2_top_n",
            disabled=not esm2_available,
        )
        if not esm2_available:
            st.info("ESM-2 unavailable (torch / esm not installed). Reinstall with: pip install -e .")

        st.divider()

        # -- NanoMelt --
        st.subheader("NanoMelt Tm")
        if _nanomelt_available():
            st.success("NanoMelt is available ✅")
        else:
            st.warning("NanoMelt not installed – Tm predictions disabled.")


# ---------------------------------------------------------------------------
# Tab 1 – Input & Analysis
# ---------------------------------------------------------------------------

def tab_input(humanness_scorer, stability_scorer, hydrophobicity_scorer,
              hsc_scorer, consensus_scorer, nanomelt_scorer, viz):
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
            st.session_state["humanness_scores"] = humanness_scorer.score(vhh)
            st.session_state["stability_scores"] = stability_scorer.score(vhh)
            st.session_state["hydrophobicity_scores"] = hydrophobicity_scorer.score(vhh)
            st.session_state["orthogonal_humanness_scores"] = hsc_scorer.score(vhh)
            st.session_state["orthogonal_stability_scores"] = consensus_scorer.score(vhh)
            if nanomelt_scorer.is_available:
                st.session_state["nanomelt_scores"] = nanomelt_scorer.score(vhh)
            else:
                st.session_state["nanomelt_scores"] = None

    # -- Display results --
    vhh = st.session_state.get("vhh_seq")
    if vhh is None:
        st.info("Enter a VHH sequence above and click **Analyse sequence**.")
        return

    h_scores = st.session_state["humanness_scores"]
    s_scores = st.session_state["stability_scores"]
    hy_scores = st.session_state["hydrophobicity_scores"]
    ort_h = st.session_state["orthogonal_humanness_scores"]
    ort_s = st.session_state["orthogonal_stability_scores"]
    nano_s = st.session_state["nanomelt_scores"]

    st.success(f"Sequence accepted – {vhh.length} residues")

    # Score bars
    st.subheader("Composite Scores")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(viz.render_score_bar(h_scores["composite_score"], "Humanness", "#42A5F5"), unsafe_allow_html=True)
    with col2:
        st.markdown(viz.render_score_bar(s_scores["composite_score"], "Stability", "#66BB6A"), unsafe_allow_html=True)
    with col3:
        hydro_bar = viz.render_score_bar(
            hy_scores["composite_score"], "Surface Hydrophobicity", "#FFA726",
        )
        st.markdown(hydro_bar, unsafe_allow_html=True)

    # Orthogonal scores
    st.subheader("Orthogonal Scores")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.metric("Human String Content", f"{ort_h['composite_score']:.3f}")
    with oc2:
        st.metric("Consensus Stability", f"{ort_s['composite_score']:.3f}")
    with oc3:
        if nano_s:
            st.metric("NanoMelt Tm", f"{nano_s.get('predicted_tm', 0):.1f} °C")
        else:
            st.metric("NanoMelt Tm", "N/A")

    # Region track
    st.subheader("IMGT Region Map")
    st.markdown(viz.render_region_track(vhh), unsafe_allow_html=True)

    # Detailed breakdowns
    with st.expander("Humanness Details"):
        st.write(f"**Best germline:** {h_scores['best_germline']}")
        st.write(f"**Germline identity:** {h_scores['germline_identity']:.3f}")
        if h_scores.get("position_scores"):
            pos_df = pd.DataFrame(
                [{"IMGT position": k, "Score": v} for k, v in sorted(h_scores["position_scores"].items())],
            )
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

    with st.expander("Stability Details"):
        detail_cols = [
            ("pI", "pI"), ("disulfide_score", "Disulfide"),
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

def tab_mutations(humanness_scorer, stability_scorer):
    st.header("🎯 Mutation Selection")

    vhh = st.session_state.get("vhh_seq")
    if vhh is None:
        st.info("Please analyse a sequence first (Tab 1).")
        return

    # -- Off-limit regions --
    st.subheader("Off-Limit Regions")
    off_limit_positions: set[int] = set()
    region_cols = st.columns(len(IMGT_REGIONS))
    for idx, (region_name, (start, end)) in enumerate(IMGT_REGIONS.items()):
        with region_cols[idx]:
            default_off = region_name.startswith("CDR")
            if st.checkbox(region_name, value=default_off, key=f"ol_{region_name}"):
                off_limit_positions.update(range(start, end + 1))

    # -- Forbidden substitutions CSV --
    st.subheader("Forbidden Substitutions")
    uploaded = st.file_uploader(
        "Upload CSV (original_aa, forbidden_aas)", type=["csv"], key="forbidden_csv",
    )
    aa_forbidden: dict[str, set[str]] = {}
    position_forbidden: dict[int, set[str]] = {}
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

    # -- Rank mutations --
    st.subheader("Rank Mutations")

    weights = {}
    enabled = {}
    if st.session_state.get("enable_humanness", True):
        weights["humanness"] = st.session_state.get("w_humanness", 0.35)
        enabled["humanness"] = True
    else:
        enabled["humanness"] = False
    if st.session_state.get("enable_stability", True):
        weights["stability"] = st.session_state.get("w_stability", 0.50)
        enabled["stability"] = True
    else:
        enabled["stability"] = False
    if st.session_state.get("enable_hydrophobicity", True):
        weights["surface_hydrophobicity"] = st.session_state.get("w_hydrophobicity", 0.15)
        enabled["surface_hydrophobicity"] = True
    else:
        enabled["surface_hydrophobicity"] = False

    scorers = load_scorers()
    _, _, hydrophobicity_scorer, hsc_scorer, consensus_scorer, nanomelt_scorer = scorers

    if st.button("Rank single mutations", type="primary", key="btn_rank"):
        engine = MutationEngine(
            humanness_scorer,
            stability_scorer,
            hydrophobicity_scorer=hydrophobicity_scorer if enabled.get("surface_hydrophobicity") else None,
            hsc_scorer=hsc_scorer,
            consensus_scorer=consensus_scorer,
            nanomelt_scorer=nanomelt_scorer if nanomelt_scorer.is_available else None,
            weights=weights,
            enabled_metrics=enabled,
        )
        with st.spinner("Ranking mutations…"):
            ranked = engine.rank_single_mutations(
                vhh,
                off_limits=off_limit_positions if off_limit_positions else None,
                forbidden_substitutions=position_forbidden if position_forbidden else None,
                excluded_target_aas=excluded_set,
            )
        st.session_state["ranked_mutations"] = ranked
        st.session_state["_mutation_engine"] = engine
        st.success(f"Ranked {len(ranked)} mutations.")

    ranked = st.session_state.get("ranked_mutations")
    if ranked is not None and not ranked.empty:
        st.dataframe(ranked, use_container_width=True, hide_index=True)

        top_n = st.slider(
            "Top N mutations for library", 1,
            min(50, len(ranked)), min(10, len(ranked)), key="top_n_muts",
        )

        if st.button("Generate library", type="primary", key="btn_gen_lib"):
            engine = st.session_state.get("_mutation_engine")
            if engine is None:
                st.error("Please rank mutations first.")
                return
            strategy_map = {"Auto": "auto", "Random": "random", "Iterative": "iterative"}
            strategy = strategy_map.get(st.session_state.get("strategy", "Auto"), "auto")
            with st.spinner("Generating library…"):
                library = engine.generate_library(
                    vhh,
                    ranked.head(top_n),
                    n_mutations=st.session_state.get("n_mutations", 3),
                    max_variants=st.session_state.get("max_variants", 1000),
                    min_mutations=st.session_state.get("min_mutations", 1),
                    strategy=strategy,
                    anchor_threshold=st.session_state.get("anchor_threshold", 0.6),
                    max_rounds=st.session_state.get("max_rounds", 5),
                )
            st.session_state["library"] = library
            st.session_state["esm2_pll_scores"] = None
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
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Total Variants", len(library))
    with mc2:
        st.metric("Mean Combined Score", f"{library['combined_score'].mean():.3f}")
    with mc3:
        if "humanness_score" in library.columns:
            st.metric("Mean Humanness", f"{library['humanness_score'].mean():.3f}")

    st.dataframe(library, use_container_width=True, hide_index=True)

    # -- Distribution plots --
    st.subheader("Score Distributions")
    score_cols = [c for c in ["combined_score", "humanness_score", "stability_score",
                               "surface_hydrophobicity_score"] if c in library.columns]
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
    if "humanness_score" in library.columns and "stability_score" in library.columns:
        st.subheader("Humanness vs Stability Correlation")
        h_vals = library["humanness_score"].dropna()
        s_vals = library["stability_score"].dropna()
        common_idx = h_vals.index.intersection(s_vals.index)
        if len(common_idx) > 2:
            rho, pval = spearmanr(h_vals.loc[common_idx], s_vals.loc[common_idx])
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(h_vals.loc[common_idx], s_vals.loc[common_idx], alpha=0.4, s=10)
            ax2.set_xlabel("Humanness Score")
            ax2.set_ylabel("Stability Score")
            ax2.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.2e})")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # -- ESM-2 PLL rescoring --
    st.subheader("ESM-2 Stability Scoring")
    if st.session_state.get("enable_esm2_pll") and _esm2_pll_available():
        model_tier = st.session_state.get("esm2_model_tier", "auto")
        top_n_esm = st.session_state.get("esm2_top_n", _ESM2_PLL_DEFAULT_TOP_N)
        if st.button("Run ESM-2 Scoring", key="btn_esm2"):
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
                st.success("ESM-2 scoring complete.")

        pll_df = st.session_state.get("esm2_pll_scores")
        if pll_df is not None:
            st.dataframe(pll_df, use_container_width=True, hide_index=True)
    elif not _esm2_pll_available():
        st.info("ESM-2 not available (torch / esm not found). Reinstall with: pip install -e .")
    else:
        st.info("Enable ESM-2 stability scoring in the sidebar to rescore top variants.")

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
                        aa_seq, opt["dna_sequence"],
                        n_tag=n_tag_val, c_tag=c_tag_val, linker=linker,
                    )
                    constructs.append({
                        "variant_id": vid,
                        "aa_construct": construct["aa_construct"],
                        "dna_construct": construct["dna_construct"],
                        "schematic": construct["schematic"],
                        "gc_content": opt["gc_content"],
                        "cai": opt["cai"],
                    })
        else:
            # Single sequence mode
            with st.spinner("Optimizing codons…"):
                opt = optimizer.optimize(vhh.sequence, host=host, strategy=codon_strat, **opt_kwargs)
            construct = tag_manager.build_construct(
                vhh.sequence, opt["dna_sequence"],
                n_tag=n_tag_val, c_tag=c_tag_val, linker=linker,
            )
            constructs.append({
                "variant_id": "parent",
                "aa_construct": construct["aa_construct"],
                "dna_construct": construct["dna_construct"],
                "schematic": construct["schematic"],
                "gc_content": opt["gc_content"],
                "cai": opt["cai"],
            })
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
                vid = c['variant_id']
                st.text_area(
                    "AA construct", c["aa_construct"],
                    height=80, key=f"aa_{vid}", disabled=True,
                )
                if c["dna_construct"]:
                    st.text_area(
                        "DNA construct", c["dna_construct"],
                        height=80, key=f"dna_{vid}", disabled=True,
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
        for key in ["vhh_seq", "humanness_scores", "stability_scores",
                     "hydrophobicity_scores", "ranked_mutations", "library",
                     "constructs"]:
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
                    for key in ["humanness_scores", "stability_scores",
                                "hydrophobicity_scores", "constructs"]:
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
    scorers = load_scorers()
    humanness_scorer, stability_scorer, hydrophobicity_scorer, hsc_scorer, consensus_scorer, nanomelt_scorer = scorers
    optimizer = CodonOptimizer()
    tag_manager = TagManager()
    viz = SequenceVisualizer()

    sidebar()

    tabs = st.tabs([
        "🔬 Input & Analysis",
        "🎯 Mutation Selection",
        "📚 Library Results",
        "🧬 Barcoding",
        "🔧 Construct Builder",
        "📁 Session History",
    ])
    with tabs[0]:
        tab_input(humanness_scorer, stability_scorer, hydrophobicity_scorer,
                  hsc_scorer, consensus_scorer, nanomelt_scorer, viz)
    with tabs[1]:
        tab_mutations(humanness_scorer, stability_scorer)
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
