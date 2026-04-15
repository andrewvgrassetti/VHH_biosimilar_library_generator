"""Score VHH sequences for biophysical stability properties."""

from __future__ import annotations

import json
from pathlib import Path

from vhh_library.sequence import VHHSequence
from vhh_library.utils import AA_PROPERTIES, isoelectric_point, net_charge

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_HALLMARK_POSITIONS: list[int] = [37, 44, 45, 47]
_HYDROPHOBIC_AAS: frozenset[str] = frozenset("VILMFYW")
_DISULFIDE_POSITIONS: list[int] = [23, 104]

# Composite score weights (legacy)
_W_DISULFIDE: float = 0.25
_W_HALLMARK: float = 0.20
_W_AGGREGATION: float = 0.25
_W_CHARGE: float = 0.15
_W_HYDROPHOBIC: float = 0.15

# Nanomelt Tm normalisation parameters
_TM_BASELINE: float = 40.0
_TM_RANGE: float = 50.0
_NANOMELT_WEIGHT: float = 0.7
_LEGACY_WEIGHT: float = 0.3

# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------

_nanomelt_flag: bool | None = None


def _nanomelt_available() -> bool:
    global _nanomelt_flag
    if _nanomelt_flag is None:
        try:
            import nanomelt  # noqa: F401

            _nanomelt_flag = True
        except ImportError:
            _nanomelt_flag = False
    return _nanomelt_flag


_esm2_pll_flag: bool | None = None


def _esm2_pll_available() -> bool:
    global _esm2_pll_flag
    if _esm2_pll_flag is None:
        try:
            import esm  # noqa: F401
            import torch  # noqa: F401

            _esm2_pll_flag = True
        except ImportError:
            _esm2_pll_flag = False
    return _esm2_pll_flag


# ---------------------------------------------------------------------------
# ESM-2 pseudo-log-likelihood
# ---------------------------------------------------------------------------


def compute_esm2_pll(sequences: list[str]) -> list[float]:
    """Compute ESM-2 pseudo-log-likelihood for each sequence.

    Raises ``ImportError`` if *torch* or *esm* are not installed.
    """
    if not _esm2_pll_available():
        raise ImportError("torch and esm are required for ESM-2 PLL computation")

    import esm
    import torch

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    results: list[float] = []

    @torch.no_grad()
    def _pll_single(seq: str) -> float:
        tokens = batch_converter([("seq", seq)])[2]
        log_probs = torch.nn.functional.log_softmax(model(tokens)["logits"], dim=-1)
        pll = 0.0
        for i in range(1, len(seq) + 1):
            token_idx = tokens[0, i].item()
            pll += log_probs[0, i, token_idx].item()
        return pll

    for seq in sequences:
        results.append(_pll_single(seq))

    return results


# ---------------------------------------------------------------------------
# Stability scorer
# ---------------------------------------------------------------------------


class StabilityScorer:
    """Score VHH sequences for biophysical stability using legacy heuristics
    and optional nanomelt integration."""

    def __init__(self, use_nanomelt: bool = True) -> None:
        germline_path = _DATA_DIR / "vhh_germlines.json"
        with open(germline_path) as fh:
            self.germlines: list[dict] = json.load(fh)["germlines"]
        self.use_nanomelt: bool = use_nanomelt

    @property
    def nanomelt_active(self) -> bool:
        return self.use_nanomelt and _nanomelt_available()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, vhh: VHHSequence) -> dict:
        """Return a comprehensive stability score dictionary for *vhh*."""
        seq = vhh.sequence
        numbered = vhh.imgt_numbered
        warnings: list[str] = list(vhh.validation_result.get("warnings", []))

        disulfide = self._disulfide_score(numbered, warnings)
        hallmark = self._hallmark_score(numbered, warnings)
        aggregation = self._aggregation_score(seq)
        charge_balance = self._charge_balance_score(seq)
        hydrophobic_core = self._hydrophobic_core_score(numbered)

        charge = net_charge(seq)
        pi = isoelectric_point(seq)

        legacy = (
            _W_DISULFIDE * disulfide
            + _W_HALLMARK * hallmark
            + _W_AGGREGATION * aggregation
            + _W_CHARGE * charge_balance
            + _W_HYDROPHOBIC * hydrophobic_core
        )

        result: dict = {
            "disulfide_score": disulfide,
            "vhh_hallmark_score": hallmark,
            "aggregation_score": aggregation,
            "charge_balance_score": charge_balance,
            "hydrophobic_core_score": hydrophobic_core,
            "net_charge": charge,
            "pI": pi,
            "warnings": warnings,
        }

        if self.nanomelt_active:
            try:
                import nanomelt

                prediction = nanomelt.predict(seq)
                tm: float = prediction["Tm"]
                result["predicted_tm"] = tm
                nm_normalized = max(0.0, min(1.0, (tm - _TM_BASELINE) / _TM_RANGE))
                result["composite_score"] = _LEGACY_WEIGHT * legacy + _NANOMELT_WEIGHT * nm_normalized
                result["scoring_method"] = "nanomelt"
            except Exception:
                result["composite_score"] = legacy
                result["scoring_method"] = "legacy"
                warnings.append("nanomelt prediction failed; fell back to legacy scoring")
        else:
            result["composite_score"] = legacy
            result["scoring_method"] = "legacy"

        return result

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int | str, new_aa: str
    ) -> float:
        """Return the change in composite score when mutating *position* to *new_aa*."""
        parent_score = self.score(vhh)["composite_score"]
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        mutant_score = self.score(mutant)["composite_score"]
        return mutant_score - parent_score

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    @staticmethod
    def _disulfide_score(
        numbered: dict[str, str], warnings: list[str]
    ) -> float:
        cys_count = sum(
            1 for pos in _DISULFIDE_POSITIONS if numbered.get(str(pos)) == "C"
        )
        if cys_count == 2:
            return 1.0
        if cys_count == 1:
            warnings.append("Only one canonical Cys found for disulfide bond")
            return 0.5
        warnings.append("No canonical Cys residues found for disulfide bond")
        return 0.0

    def _hallmark_score(
        self, numbered: dict[str, str], warnings: list[str]
    ) -> float:
        allowed: dict[int, set[str]] = {}
        for gl in self.germlines:
            hp = gl.get("hallmark_positions", {})
            for pos_str, aa in hp.items():
                pos = int(pos_str)
                allowed.setdefault(pos, set()).add(aa)

        matches = 0
        for pos in _HALLMARK_POSITIONS:
            aa = numbered.get(str(pos))
            if aa is not None and aa in allowed.get(pos, set()):
                matches += 1
            else:
                warnings.append(
                    f"Non-canonical residue '{aa}' at VHH hallmark position {pos}"
                )

        return matches / len(_HALLMARK_POSITIONS) if _HALLMARK_POSITIONS else 0.0

    @staticmethod
    def _aggregation_score(sequence: str) -> float:
        window = 7
        if len(sequence) < window:
            return 1.0
        n_patches = 0
        for i in range(len(sequence) - window + 1):
            segment = sequence[i : i + window]
            avg_hydro = sum(
                AA_PROPERTIES.get(aa, {}).get("hydrophobicity", 0.0) for aa in segment
            ) / window
            if avg_hydro > 1.5:
                n_patches += 1
        return max(0.0, 1.0 - n_patches * 0.15)

    @staticmethod
    def _charge_balance_score(sequence: str) -> float:
        charge = net_charge(sequence)
        abs_charge = abs(charge)
        if abs_charge <= 2.0:
            return 1.0
        return max(0.0, 1.0 - (abs_charge - 2.0) * 0.1)

    @staticmethod
    def _hydrophobic_core_score(numbered: dict[str, str]) -> float:
        if not _HALLMARK_POSITIONS:
            return 0.0
        hits = sum(
            1
            for pos in _HALLMARK_POSITIONS
            if numbered.get(str(pos), "") in _HYDROPHOBIC_AAS
        )
        return hits / len(_HALLMARK_POSITIONS)
