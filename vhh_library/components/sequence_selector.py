"""Interactive sequence selector Streamlit component."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Set

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
    off_limit_positions: Set[str] | None = None,
    forbidden_substitutions: Optional[Dict[str, set]] = None,
    frozen_positions: Set[str] | None = None,
    conservative_positions: Set[str] | None = None,
    key: Optional[str] = None,
) -> dict[str, list[str]] | None:
    """Render the interactive sequence selector and return position classes.

    The selector supports three states per position:

    * **frozen** — position must not be mutated (dark overlay).
    * **conservative** — only restricted amino acids allowed (orange dot).
    * **mutable** — any standard amino acid substitution may be proposed
      (default, no indicator).

    Clicking a residue cycles through: mutable → frozen → conservative → mutable.
    Dragging applies the same target class to all touched residues.

    Parameters
    ----------
    sequence:
        Raw amino-acid string (kept for backward compatibility but no longer
        used for rendering; the ordered *imgt_numbered* dict drives the display).
    imgt_numbered:
        Ordered ``dict[str, str]`` mapping IMGT position keys to amino acids.
    off_limit_positions:
        **Deprecated** — use *frozen_positions* instead.  Set of IMGT position
        **string** keys that are frozen.  When both *off_limit_positions* and
        *frozen_positions* are provided, they are merged (union).
    forbidden_substitutions:
        **Deprecated** — use *conservative_positions* instead.  Optional
        mapping of IMGT position **string** keys to sets of forbidden AAs.
        Positions listed here are shown as conservative.  When both
        *forbidden_substitutions* and *conservative_positions* are provided,
        they are merged.
    frozen_positions:
        Set of IMGT position **string** keys that are frozen.
    conservative_positions:
        Set of IMGT position **string** keys that are conservative.
    key:
        Streamlit component key.

    Returns
    -------
    dict[str, list[str]] | None
        A dict with ``"frozen"`` and ``"conservative"`` keys, each mapping to
        a sorted list of IMGT position string keys.  Positions not listed are
        mutable.  Returns ``None`` on the first render before user interaction.
    """
    # Merge legacy parameters with new parameters
    merged_frozen: set[str] = set(frozen_positions) if frozen_positions else set()
    if off_limit_positions:
        merged_frozen |= off_limit_positions

    merged_conservative: set[str] = set(conservative_positions) if conservative_positions else set()
    if forbidden_substitutions:
        for p in forbidden_substitutions:
            str_p = str(p)
            if str_p not in merged_frozen:
                merged_conservative.add(str_p)

    # Frozen takes precedence over conservative
    merged_conservative -= merged_frozen

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

    sorted_frozen = sorted(merged_frozen, key=lambda k: (imgt_key_int_part(k), k))
    sorted_conservative = sorted(merged_conservative, key=lambda k: (imgt_key_int_part(k), k))

    default_value = {"frozen": sorted_frozen, "conservative": sorted_conservative}

    result = _component_func(
        imgtPositionsList=imgt_positions_list,
        regions=regions,
        frozenPositions=sorted_frozen,
        conservativePositions=sorted_conservative,
        notablePositions=notable,
        default=default_value,
        key=key,
    )

    if result is None:
        return default_value
    return result
