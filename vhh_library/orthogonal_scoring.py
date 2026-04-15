"""Orthogonal (cross-validation) scoring methods for VHH sequences."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vhh_library.sequence import IMGT_REGIONS, VHHSequence

_NANOMELT_TM_MIN = 40.0
_NANOMELT_TM_MAX = 90.0

_FR_KEYS = ("fr1", "fr2", "fr3", "fr4")
_FR_REGION_NAMES = ("FR1", "FR2", "FR3", "FR4")


class HumanStringContentScorer:
    """Scores how much of the VHH sequence matches human germline k-mer content."""

    def __init__(self, kmer_size: int = 9) -> None:
        self._kmer_size = kmer_size

        data_path = Path(__file__).resolve().parent.parent / "data" / "human_vh_germlines.json"
        with open(data_path) as fh:
            data = json.load(fh)

        kmers: set[str] = set()
        for g in data["germlines"]:
            fw = g["fr1"] + g["fr2"] + g["fr3"] + g["fr4"]
            for i in range(len(fw) - kmer_size + 1):
                kmers.add(fw[i : i + kmer_size])

        self._human_kmers: frozenset[str] = frozenset(kmers)

    def score(self, vhh: VHHSequence) -> dict:
        seq = vhh.sequence
        k = self._kmer_size
        total = max(len(seq) - k + 1, 0)
        if total == 0:
            return {"composite_score": 0.0, "total_kmers": 0, "matched_kmers": 0}

        matched = sum(
            1 for i in range(total) if seq[i : i + k] in self._human_kmers
        )
        return {
            "composite_score": matched / total,
            "total_kmers": total,
            "matched_kmers": matched,
        }

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int | str, new_aa: str
    ) -> float:
        if vhh.imgt_numbered.get(str(position)) == new_aa:
            return 0.0
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        return self.score(mutant)["composite_score"] - self.score(vhh)["composite_score"]


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


class NanoMeltStabilityScorer:
    """Lazy-loading nanomelt Tm predictor."""

    def __init__(self) -> None:
        self._available: bool | None = None
        self._predict_fn = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            try:
                from nanomelt.predict import predict_tm  # type: ignore[import-untyped]

                self._predict_fn = predict_tm
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def score(self, vhh: VHHSequence) -> dict:
        if not self.is_available:
            raise ImportError("nanomelt is not installed")

        tm: float = self._predict_fn(vhh.sequence)
        raw = (tm - _NANOMELT_TM_MIN) / (_NANOMELT_TM_MAX - _NANOMELT_TM_MIN)
        composite_score = max(0.0, min(1.0, raw))
        return {"composite_score": composite_score, "predicted_tm": tm}

    def predict_mutation_effect(
        self, vhh: VHHSequence, position: int | str, new_aa: str
    ) -> float:
        if not self.is_available:
            return 0.0
        pos_int = int(position) if not isinstance(position, int) else position
        if pos_int < 1 or pos_int > vhh.length:
            return 0.0
        if vhh.imgt_numbered.get(str(position)) == new_aa:
            return 0.0
        mutant = VHHSequence.mutate(vhh, position, new_aa)
        return self.score(mutant)["composite_score"] - self.score(vhh)["composite_score"]
