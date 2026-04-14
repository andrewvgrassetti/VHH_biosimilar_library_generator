"""Developability scoring classes for VHH sequences."""

from __future__ import annotations

import re

from vhh_library.sequence import VHHSequence
from vhh_library.utils import AA_PROPERTIES, isoelectric_point

# ---------------------------------------------------------------------------
# PTM liability patterns
# ---------------------------------------------------------------------------

_PTM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"D[GSTD]"), "isomerization"),
    (re.compile(r"N[GSH]"), "deamidation"),
    (re.compile(r"N[^P][ST]"), "glycosylation"),
]

# ---------------------------------------------------------------------------
# Clearance risk constants
# ---------------------------------------------------------------------------

_PI_LOW: float = 6.0
_PI_HIGH: float = 9.0
_PI_MAX_DEVIATION: float = 4.0

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity lookup
# ---------------------------------------------------------------------------

_HYDRO: dict[str, float] = {aa: props["hydrophobicity"] for aa, props in AA_PROPERTIES.items()}


# ---------------------------------------------------------------------------
# PTM liability scorer
# ---------------------------------------------------------------------------


class PTMLiabilityScorer:
    """Detect post-translational modification liability motifs in VHH sequences."""

    def score(self, vhh: VHHSequence) -> dict:
        """Return liability hits and a composite score for *vhh*.

        Returns a dict with keys ``composite_score``, ``hits``, ``n_hits``,
        and ``warnings``.
        """
        seq = vhh.sequence
        warnings: list[str] = list(vhh.validation_result.get("warnings", []))
        hits: list[dict] = []

        for pattern, category in _PTM_PATTERNS:
            for match in pattern.finditer(seq):
                hits.append({
                    "position": match.start() + 1,
                    "motif": match.group(),
                    "category": category,
                })

        n_hits = len(hits)
        composite_score = max(0.0, 1.0 - n_hits * 0.1)

        if n_hits:
            warnings.append(f"{n_hits} PTM liability motif(s) detected")

        return {
            "composite_score": composite_score,
            "hits": hits,
            "n_hits": n_hits,
            "warnings": warnings,
        }

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int, new_aa: str
    ) -> float:
        """Return the change in composite score when mutating *position* to *new_aa*."""
        parent_score = self.score(vhh)["composite_score"]
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        return self.score(mutant)["composite_score"] - parent_score


# ---------------------------------------------------------------------------
# Clearance risk scorer
# ---------------------------------------------------------------------------


class ClearanceRiskScorer:
    """Score clearance risk based on isoelectric point deviation from the
    ideal VHH range (6.0–9.0)."""

    def score(self, vhh: VHHSequence) -> dict:
        """Return pI-based clearance risk score for *vhh*.

        Returns a dict with keys ``composite_score``, ``pI``, ``pI_deviation``,
        and ``warnings``.
        """
        warnings: list[str] = list(vhh.validation_result.get("warnings", []))
        pi = isoelectric_point(vhh.sequence)

        if pi < _PI_LOW:
            deviation = _PI_LOW - pi
        elif pi > _PI_HIGH:
            deviation = pi - _PI_HIGH
        else:
            deviation = 0.0

        composite_score = max(0.0, 1.0 - deviation / _PI_MAX_DEVIATION)

        if deviation > 0.0:
            warnings.append(f"pI {pi:.2f} outside ideal range ({_PI_LOW}–{_PI_HIGH})")

        return {
            "composite_score": composite_score,
            "pI": pi,
            "pI_deviation": deviation,
            "warnings": warnings,
        }

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int, new_aa: str
    ) -> float:
        """Return the change in composite score when mutating *position* to *new_aa*."""
        parent_score = self.score(vhh)["composite_score"]
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        return self.score(mutant)["composite_score"] - parent_score


# ---------------------------------------------------------------------------
# Surface hydrophobicity scorer
# ---------------------------------------------------------------------------


class SurfaceHydrophobicityScorer:
    """Detect surface-exposed hydrophobic patches using a CDR-weighted
    sliding window over the VHH sequence."""

    def __init__(self, window: int = 7, threshold: float = 1.5) -> None:
        self.window: int = window
        self.threshold: float = threshold

    def score(self, vhh: VHHSequence) -> dict:
        """Return hydrophobic-patch analysis for *vhh*.

        Returns a dict with keys ``composite_score``, ``n_patches``,
        ``max_patch_score``, and ``warnings``.
        """
        seq = vhh.sequence
        warnings: list[str] = list(vhh.validation_result.get("warnings", []))
        w = self.window

        if len(seq) < w:
            return {
                "composite_score": 1.0,
                "n_patches": 0,
                "max_patch_score": 0.0,
                "warnings": warnings,
            }

        cdr_pos = vhh.cdr_positions
        hydro_vals = [_HYDRO.get(aa, 0.0) for aa in seq]
        # CDR positions are 1-based IMGT; weight 2× for surface exposure
        weights = [2.0 if (i + 1) in cdr_pos else 1.0 for i in range(len(seq))]

        # Rolling sum initialisation
        w_sum = sum(hydro_vals[j] * weights[j] for j in range(w))
        w_weight = sum(weights[j] for j in range(w))

        n_patches = 0
        max_patch_score = 0.0

        avg = w_sum / w_weight if w_weight else 0.0
        if avg > self.threshold:
            n_patches += 1
            max_patch_score = avg

        for i in range(1, len(seq) - w + 1):
            # Roll: remove element leaving the window, add element entering
            w_sum -= hydro_vals[i - 1] * weights[i - 1]
            w_sum += hydro_vals[i + w - 1] * weights[i + w - 1]
            w_weight -= weights[i - 1]
            w_weight += weights[i + w - 1]

            avg = w_sum / w_weight if w_weight else 0.0
            if avg > self.threshold:
                n_patches += 1
                if avg > max_patch_score:
                    max_patch_score = avg

        composite_score = max(0.0, 1.0 - n_patches * 0.1)

        if n_patches:
            warnings.append(f"{n_patches} hydrophobic patch(es) detected")

        return {
            "composite_score": composite_score,
            "n_patches": n_patches,
            "max_patch_score": max_patch_score,
            "warnings": warnings,
        }

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int, new_aa: str
    ) -> float:
        """Return the change in composite score when mutating *position* to *new_aa*."""
        parent_score = self.score(vhh)["composite_score"]
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        return self.score(mutant)["composite_score"] - parent_score
