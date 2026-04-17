"""Interactive sequence selector Streamlit component."""

from __future__ import annotations

import re
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

_IMGT_KEY_INT_RE = re.compile(r"^(\d+)")


def imgt_key_int_part(key: str) -> int:
    """Extract the integer portion of an IMGT position key (e.g. '111A' → 111)."""
    m = _IMGT_KEY_INT_RE.match(key)
    return int(m.group(1)) if m else 0


def sequence_selector(
    sequence: str,
    imgt_numbered: Dict[str, str],
    off_limit_positions: Set[str],
    forbidden_substitutions: Optional[Dict[str, set]] = None,
    key: Optional[str] = None,
) -> List[str]:
    """Render the interactive sequence selector and return selected off-limit positions.

    Parameters
    ----------
    sequence:
        Raw amino-acid string (kept for backward compatibility but no longer
        used for rendering; the ordered *imgt_numbered* dict drives the display).
    imgt_numbered:
        Ordered ``dict[str, str]`` mapping IMGT position keys to amino acids.
    off_limit_positions:
        Set of IMGT position **string** keys that are off-limit.
    forbidden_substitutions:
        Optional mapping of IMGT position **string** keys to sets of forbidden AAs.
    key:
        Streamlit component key.

    Returns
    -------
    list[str]
        Sorted list of IMGT position string keys currently marked off-limit.
    """
    if forbidden_substitutions is None:
        forbidden_substitutions = {}

    # Build an ordered list of [imgt_key, amino_acid] pairs for the frontend.
    imgt_positions_list: list[list[str]] = [
        [k, v] for k, v in imgt_numbered.items()
    ]

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
        if str(pos) in imgt_numbered:
            notable[str(pos)] = {"label": label, "bg": bg, "fg": fg}

    forbidden_pos_list = [str(p) for p in forbidden_substitutions.keys()]

    sorted_off_limit = sorted(off_limit_positions, key=lambda k: (imgt_key_int_part(k), k))
    result = _component_func(
        imgtPositionsList=imgt_positions_list,
        regions=regions,
        offLimitPositions=sorted_off_limit,
        notablePositions=notable,
        forbiddenPositions=forbidden_pos_list,
        default=sorted_off_limit,
        key=key,
    )

    if result is None:
        return sorted_off_limit
    return result
