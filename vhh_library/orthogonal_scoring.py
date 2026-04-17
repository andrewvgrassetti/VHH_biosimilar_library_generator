"""Orthogonal (cross-validation) scoring methods for VHH sequences."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vhh_library.sequence import IMGT_REGIONS, VHHSequence

_FR_KEYS = ("fr1", "fr2", "fr3", "fr4")
_FR_REGION_NAMES = ("FR1", "FR2", "FR3", "FR4")


class ConsensusStabilityScorer:
    """Scores framework positions against a VHH germline consensus."""

    def __init__(self) -> None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "vhh_germlines.json"
        with open(data_path) as fh:
            data = json.load(fh)

        position_counts: dict[int, Counter[str]] = {}
        for g in data["germlines"]:
            for fr_key, fr_name in zip(_FR_KEYS, _FR_REGION_NAMES):
                start, _end = IMGT_REGIONS[fr_name]
                for offset, aa in enumerate(g[fr_key]):
                    pos = start + offset
                    if pos not in position_counts:
                        position_counts[pos] = Counter()
                    position_counts[pos][aa] += 1

        self._consensus: dict[int, str] = {
            pos: counts.most_common(1)[0][0] for pos, counts in position_counts.items()
        }

    def score(self, vhh: VHHSequence) -> dict:
        positions_evaluated = 0
        consensus_matches = 0
        for pos, consensus_aa in self._consensus.items():
            str_pos = str(pos)
            if str_pos in vhh.imgt_numbered:
                positions_evaluated += 1
                if vhh.imgt_numbered[str_pos] == consensus_aa:
                    consensus_matches += 1

        avg_conservation = (
            consensus_matches / positions_evaluated if positions_evaluated else 0.0
        )
        return {
            "composite_score": avg_conservation,
            "positions_evaluated": positions_evaluated,
            "consensus_matches": consensus_matches,
            "avg_conservation": avg_conservation,
        }

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int | str, new_aa: str
    ) -> float:
        if vhh.imgt_numbered.get(str(position)) == new_aa:
            return 0.0
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        return self.score(mutant)["composite_score"] - self.score(vhh)["composite_score"]
