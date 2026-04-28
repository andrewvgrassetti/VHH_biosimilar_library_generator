"""Two-part assembly utilities for yeast surface display library generation.

This module supports splitting a VHH construct into two halves at a user-defined
center position, generating variant parts independently for each half, and
producing a combinatorial library representing all Part 1 × Part 2 fusions.
This models a real-world PCR overlap-assembly workflow where N Part 1 oligos ×
M Part 2 oligos yield N×M full-length constructs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)


def lock_overlap_positions(
    split_position: str,
    overlap_width: int,
    imgt_positions: list[str],
) -> set[str]:
    """Return IMGT positions to freeze around the split point for PCR overlap homology.

    The overlap region spans *overlap_width* residues centred on *split_position*:
    ``overlap_width // 2`` positions on each side (including the split position
    itself on the left side).

    Parameters
    ----------
    split_position:
        IMGT position string key defining the split boundary.
    overlap_width:
        Total number of residues in the overlap region (default ~6).
    imgt_positions:
        Ordered list of all IMGT position string keys present in the sequence.

    Returns
    -------
    set[str]
        IMGT position keys that must be frozen.
    """
    if split_position not in imgt_positions:
        raise ValueError(f"Split position {split_position!r} not found in IMGT positions")

    split_idx = imgt_positions.index(split_position)
    half_left = overlap_width // 2
    half_right = overlap_width - half_left

    start = max(0, split_idx - half_left + 1)
    end = min(len(imgt_positions), split_idx + half_right + 1)

    locked = set(imgt_positions[start:end])
    logger.debug(
        "Overlap lock: split=%s, width=%d, locked=%s",
        split_position,
        overlap_width,
        sorted(locked, key=lambda k: imgt_positions.index(k)),
    )
    return locked


def split_mutations(
    top_mutations: pd.DataFrame,
    split_position: str,
    imgt_positions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition ranked mutations into Part 1 (≤ split) and Part 2 (> split).

    Uses the ordering of *imgt_positions* to determine which side of the
    split each mutation belongs to.  Mutations whose ``imgt_pos`` (or
    ``position`` converted to string) is at or before *split_position*
    go into Part 1; the rest go into Part 2.

    Parameters
    ----------
    top_mutations:
        Ranked mutations DataFrame (must have ``position`` or ``imgt_pos`` column).
    split_position:
        IMGT position string key for the split boundary.
    imgt_positions:
        Ordered list of all IMGT position string keys in the sequence.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(part1_mutations, part2_mutations)``
    """
    if split_position not in imgt_positions:
        raise ValueError(f"Split position {split_position!r} not found in IMGT positions")

    split_idx = imgt_positions.index(split_position)
    part1_keys = set(imgt_positions[: split_idx + 1])

    # Determine the position column to use for partitioning.
    if "imgt_pos" in top_mutations.columns:
        pos_col = "imgt_pos"
    elif "position" in top_mutations.columns:
        pos_col = "position"
    else:
        raise ValueError("top_mutations must contain 'imgt_pos' or 'position' column")

    mask = top_mutations[pos_col].astype(str).isin(part1_keys)
    part1 = top_mutations[mask].copy()
    part2 = top_mutations[~mask].copy()

    logger.info(
        "Split mutations at %s: Part 1 = %d mutations, Part 2 = %d mutations",
        split_position,
        len(part1),
        len(part2),
    )
    return part1, part2


def combine_parts(
    part1_variants: pd.DataFrame,
    part2_variants: pd.DataFrame,
    vhh_sequence: VHHSequence,
    split_position: str,
    overlap_width: int = 6,
) -> pd.DataFrame:
    """Build the N×M combinatorial DataFrame from independently generated parts.

    Each combination's full-length sequence is assembled by taking Part 1's
    mutated residues up to and including the overlap region, then Part 2's
    mutated residues after the overlap region.

    Parameters
    ----------
    part1_variants:
        DataFrame of Part 1 variants (must have ``aa_sequence``, ``mutations``, ``variant_id``).
    part2_variants:
        DataFrame of Part 2 variants (must have ``aa_sequence``, ``mutations``, ``variant_id``).
    vhh_sequence:
        The parent VHH sequence used for assembly reference.
    split_position:
        IMGT position string key for the split boundary.
    overlap_width:
        Number of residues in the overlap region.

    Returns
    -------
    pd.DataFrame
        Combinatorial DataFrame with columns: ``variant_id``, ``part1_id``,
        ``part2_id``, ``part1_mutations``, ``part2_mutations``, ``mutations``,
        ``n_mutations``, ``aa_sequence``, plus scoring placeholder columns.
    """
    imgt_positions = list(vhh_sequence.imgt_numbered.keys())

    if split_position not in imgt_positions:
        raise ValueError(f"Split position {split_position!r} not found in IMGT positions")

    # Determine the sequence-index boundary for splitting.
    # Part 1 residues: all positions up to and including the overlap right edge.
    split_idx = imgt_positions.index(split_position)
    half_right = overlap_width - overlap_width // 2
    # The overlap right boundary in IMGT position list index space
    overlap_right_idx = min(len(imgt_positions) - 1, split_idx + half_right)

    # Map IMGT positions to 0-based sequence indices for slicing.
    pos_to_seq = vhh_sequence._pos_to_seq_idx

    # Find the raw sequence cut point: Part 1 covers seq[:cut], Part 2 covers seq[cut:]
    # The cut point is right after the overlap region.
    overlap_right_key = imgt_positions[overlap_right_idx]
    cut_seq_idx = pos_to_seq[overlap_right_key] + 1

    rows: list[dict] = []
    counter = 1

    for _, p1_row in part1_variants.iterrows():
        p1_seq = p1_row["aa_sequence"]
        p1_id = p1_row["variant_id"]
        p1_muts = p1_row.get("mutations", "")

        for _, p2_row in part2_variants.iterrows():
            p2_seq = p2_row["aa_sequence"]
            p2_id = p2_row["variant_id"]
            p2_muts = p2_row.get("mutations", "")

            # Assemble: Part 1's N-terminal half + Part 2's C-terminal half.
            # The overlap region comes from Part 1 (it's frozen, so both parts
            # have the same overlap residues as the parent).
            full_seq = p1_seq[:cut_seq_idx] + p2_seq[cut_seq_idx:]

            # Combine mutation labels.
            all_muts_parts = []
            if p1_muts:
                all_muts_parts.append(p1_muts)
            if p2_muts:
                all_muts_parts.append(p2_muts)
            all_muts = ", ".join(all_muts_parts)

            # Count total mutations.
            n_muts = 0
            if p1_muts:
                n_muts += len(p1_muts.split(", "))
            if p2_muts:
                n_muts += len(p2_muts.split(", "))

            rows.append(
                {
                    "variant_id": f"V{counter:06d}",
                    "part1_id": p1_id,
                    "part2_id": p2_id,
                    "part1_mutations": p1_muts,
                    "part2_mutations": p2_muts,
                    "mutations": all_muts,
                    "n_mutations": n_muts,
                    "aa_sequence": full_seq,
                    # Scoring placeholders — filled by the caller.
                    "stability_score": 0.0,
                    "nativeness_score": 0.0,
                    "combined_score": 0.0,
                }
            )
            counter += 1

    logger.info(
        "Combined %d Part 1 × %d Part 2 = %d total combinations",
        len(part1_variants),
        len(part2_variants),
        len(rows),
    )
    return pd.DataFrame(rows)
