"""Interactive overlap range selector Streamlit component.

Renders the VHH sequence with a visual bracket/highlight layer that lets users
click residues to define N-terminal and C-terminal overlap boundaries for
yeast-display two-part assembly.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

from vhh_library.sequence import IMGT_REGIONS

_FRONTEND_DIR = Path(__file__).parent / "frontend_overlap"

_overlap_component_func = components.declare_component(
    "overlap_selector",
    path=str(_FRONTEND_DIR),
)

_REGION_COLORS = {
    "FR1": "#E3F2FD",
    "CDR1": "#FFCDD2",
    "FR2": "#E8F5E9",
    "CDR2": "#FFCDD2",
    "FR3": "#E3F2FD",
    "CDR3": "#FFCDD2",
    "FR4": "#E8F5E9",
}
_REGION_LABEL_COLORS = {
    "FR1": "#1565C0",
    "CDR1": "#C62828",
    "FR2": "#2E7D32",
    "CDR2": "#C62828",
    "FR3": "#1565C0",
    "CDR3": "#C62828",
    "FR4": "#2E7D32",
}


def overlap_selector(
    imgt_numbered: dict[str, str],
    n_boundary: str | None = None,
    c_boundary: str | None = None,
    key: str | None = None,
) -> dict[str, str | None] | None:
    """Render the interactive overlap range selector and return boundary positions.

    Users click residues to set the N-terminal and C-terminal boundaries of the
    PCR overlap region for yeast-display two-part assembly.  The overlap region
    is highlighted visually on the sequence.

    Parameters
    ----------
    imgt_numbered:
        Ordered ``dict[str, str]`` mapping IMGT position keys to amino acids.
    n_boundary:
        Current N-terminal boundary IMGT position key, or ``None``.
    c_boundary:
        Current C-terminal boundary IMGT position key, or ``None``.
    key:
        Streamlit component key.

    Returns
    -------
    dict[str, str | None] | None
        A dict with ``"n_boundary"`` and ``"c_boundary"`` keys, each mapping
        to an IMGT position string key or ``None``.  Returns ``None`` on the
        first render before user interaction.
    """
    imgt_positions_list: list[list[str]] = [[k, v] for k, v in imgt_numbered.items()]

    regions = []
    for region_name in ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"):
        start, end = IMGT_REGIONS[region_name]
        regions.append(
            {
                "name": region_name,
                "start": start,
                "end": end,
                "color": _REGION_COLORS[region_name],
                "labelColor": _REGION_LABEL_COLORS[region_name],
            }
        )

    notable = {}
    for pos, label, bg, fg in [
        (23, "Cys (disulfide)", "#FFD600", "#000"),
        (104, "Cys (disulfide)", "#FFD600", "#000"),
        (47, "Trp (VHH hallmark)", "#AB47BC", "#FFF"),
        (118, "Trp (VHH hallmark)", "#AB47BC", "#FFF"),
    ]:
        if str(pos) in imgt_numbered:
            notable[str(pos)] = {"label": label, "bg": bg, "fg": fg}

    default_value = {"n_boundary": n_boundary, "c_boundary": c_boundary}

    result = _overlap_component_func(
        imgtPositionsList=imgt_positions_list,
        regions=regions,
        notablePositions=notable,
        nBoundary=n_boundary,
        cBoundary=c_boundary,
        default=default_value,
        key=key,
    )

    if result is None:
        return default_value
    return result
