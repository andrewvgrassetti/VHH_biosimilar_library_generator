"""Score VHH sequences for humanness (similarity to human VH germlines)."""

from __future__ import annotations

import json
from pathlib import Path

from vhh_library.sequence import VHHSequence

_FR_KEYS = ("fr1", "fr2", "fr3", "fr4")


class HumAnnotator:

    def __init__(self) -> None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "human_vh_germlines.json"
        with open(data_path) as fh:
            data = json.load(fh)

        self.germlines: list[dict] = data["germlines"]
        self.position_freq: dict[str, dict[str, float]] = data["position_frequency_matrix"]

        self._germline_frameworks: list[tuple[str, str]] = [
            (g["name"], g["fr1"] + g["fr2"] + g["fr3"] + g["fr4"])
            for g in self.germlines
        ]

    def score(self, vhh: VHHSequence) -> dict:
        vhh_fw = "".join(
            vhh.regions[name][2]
            for name in ("FR1", "FR2", "FR3", "FR4")
        )

        best_identity = 0.0
        best_germline = ""
        for name, gfw in self._germline_frameworks:
            denom = max(len(gfw), len(vhh_fw))
            if denom == 0:
                continue
            matches = sum(a == b for a, b in zip(gfw, vhh_fw))
            identity = matches / denom
            if identity > best_identity:
                best_identity = identity
                best_germline = name

        position_scores: dict[str, float] = {}
        for pos, aa in vhh.imgt_numbered.items():
            freq_dict = self.position_freq.get(pos)
            if freq_dict is None:
                continue
            position_scores[pos] = freq_dict.get(aa, freq_dict.get("other", 0.0))

        pos_freq_score = (
            sum(position_scores.values()) / len(position_scores)
            if position_scores
            else 0.0
        )

        composite = 0.4 * best_identity + 0.6 * pos_freq_score

        return {
            "composite_score": composite,
            "germline_identity": best_identity,
            "best_germline": best_germline,
            "position_scores": position_scores,
        }

    def get_mutation_suggestions(
        self,
        vhh: VHHSequence,
        off_limits: set[int] | set[str],
        forbidden_substitutions: dict[int, set[str]] | dict[str, set[str]] | None = None,
        excluded_target_aas: set[str] | None = None,
        max_per_position: int = 1,
    ) -> list[dict]:
        cdr_positions = vhh.cdr_positions
        # Normalise off_limits to string keys for consistent comparison.
        off_limits_str = {str(p) for p in off_limits}
        # Normalise forbidden_substitutions keys to strings.
        forbidden_str: dict[str, set[str]] = {}
        if forbidden_substitutions:
            forbidden_str = {str(k): v for k, v in forbidden_substitutions.items()}
        suggestions: list[dict] = []

        for pos, aa in vhh.imgt_numbered.items():
            if pos in off_limits_str or pos in cdr_positions:
                continue

            freq_dict = self.position_freq.get(pos)
            if freq_dict is None:
                continue

            current_freq = freq_dict.get(aa, freq_dict.get("other", 0.0))

            candidates: list[tuple[str, float]] = []
            for candidate, freq in freq_dict.items():
                if candidate == "other" or candidate == aa:
                    continue
                if excluded_target_aas and candidate in excluded_target_aas:
                    continue
                if (
                    forbidden_str
                    and pos in forbidden_str
                    and candidate in forbidden_str[pos]
                ):
                    continue
                if freq > current_freq:
                    candidates.append((candidate, freq))

            # Sort by frequency descending and take top K
            candidates.sort(key=lambda c: c[1], reverse=True)
            for candidate_aa, freq in candidates[:max_per_position]:
                suggestions.append({
                    "position": pos,
                    "original_aa": aa,
                    "suggested_aa": candidate_aa,
                    "delta_humanness": freq - current_freq,
                    "reason": "Higher frequency in human germlines",
                })

        suggestions.sort(key=lambda s: s["delta_humanness"], reverse=True)
        return suggestions
