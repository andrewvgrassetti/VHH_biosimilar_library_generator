"""Interactive sequence selector Streamlit component."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit.components.v1 as components

from vhh_library.sequence import IMGT_REGIONS

_FRONTEND_DIR = Path(__file__).parent / "frontend"

_component_func = components.declare_component(
    "sequence_selector",
    path=str(_FRONTEND_DIR),
)

_REGION_COLORS = {
    "FR1": "#E3F2FD", "CDR1": "#FFCDD2", "FR2": "#E8F5E9",
    "CDR2": "#FFCDD2", "FR3": "#E3F2FD", "CDR3": "#FFCDD2", "FR4": "#E8F5E9",
}
_REGION_LABEL_COLORS = {
    "FR1": "#1565C0", "CDR1": "#C62828", "FR2": "#2E7D32",
    "CDR2": "#C62828", "FR3": "#1565C0", "CDR3": "#C62828", "FR4": "#2E7D32",
}


def sequence_selector(
    sequence: str,
    imgt_numbered: Dict[int, str],
    off_limit_positions: Set[int],
    forbidden_substitutions: Optional[Dict[int, set]] = None,
    key: Optional[str] = None,
) -> List[int]:
    """Render the interactive sequence selector and return selected off-limit positions."""
    if forbidden_substitutions is None:
        forbidden_substitutions = {}

    regions = []
    for region_name in ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"):
        start, end = IMGT_REGIONS[region_name]
        regions.append({
            "name": region_name,
            "start": start,
            "end": end,
            "color": _REGION_COLORS[region_name],
            "labelColor": _REGION_LABEL_COLORS[region_name],
        })

    notable = {}
    for pos, label, bg, fg in [
        (23, "Cys (disulfide)", "#FFD600", "#000"),
        (104, "Cys (disulfide)", "#FFD600", "#000"),
        (47, "Trp (VHH hallmark)", "#AB47BC", "#FFF"),
        (118, "Trp (VHH hallmark)", "#AB47BC", "#FFF"),
    ]:
        if pos in imgt_numbered:
            notable[str(pos)] = {"label": label, "bg": bg, "fg": fg}

    forbidden_pos_list = [int(p) for p in forbidden_substitutions.keys()]

    result = _component_func(
        sequence=sequence,
        imgtNumbered={str(k): v for k, v in imgt_numbered.items()},
        regions=regions,
        offLimitPositions=sorted(off_limit_positions),
        notablePositions=notable,
        forbiddenPositions=forbidden_pos_list,
        default=sorted(off_limit_positions),
        key=key,
    )

    if result is None:
        return sorted(off_limit_positions)
    return result
