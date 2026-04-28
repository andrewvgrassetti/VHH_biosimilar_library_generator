"""Sequence diversity analysis for VHH variant libraries.

Pure-library module (no Streamlit imports) providing functions for encoding
mutation matrices, computing UMAP embeddings, and generating frequency /
co-occurrence data suitable for heatmap visualisation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Canonical amino-acid ordering used for integer encoding (1-indexed).
_AA_ORDER: list[str] = sorted("ACDEFGHIKLMNPQRSTVWY")
_AA_TO_INT: dict[str, int] = {aa: i + 1 for i, aa in enumerate(_AA_ORDER)}

# Regex for mutation strings like "A10S", "G42K", "L111AF" (IMGT positions may
# contain insertion codes such as "111A").
_MUT_RE = re.compile(r"^([A-Z])(\d+[A-Z]?)([A-Z])$")


def encode_mutation_matrix(
    library_df: pd.DataFrame,
    wt_sequence: str,
) -> tuple[np.ndarray, list[str]]:
    """Build an integer-encoded mutation matrix from a library DataFrame.

    Parameters
    ----------
    library_df : pd.DataFrame
        Library DataFrame with a ``mutations`` column containing comma-separated
        mutation strings in ``"{OrigAA}{IMGTPos}{NewAA}"`` format.
    wt_sequence : str
        Wild-type amino-acid sequence (unused beyond validation; position
        identity is parsed directly from the mutation string).

    Returns
    -------
    matrix : np.ndarray
        Integer matrix of shape ``(n_variants, n_mutable_positions)`` where
        0 = wild-type at that position and 1–20 = a specific substitution
        amino acid (index into ``_AA_ORDER``).
    positions : list[str]
        Ordered list of IMGT position labels corresponding to the columns.
    """
    # Collect all (variant_idx, imgt_pos, new_aa) triples.
    records: list[tuple[int, str, str]] = []
    all_positions: set[str] = set()

    mutations_col = library_df["mutations"]
    for row_idx, mut_str in enumerate(mutations_col):
        if not isinstance(mut_str, str) or not mut_str.strip():
            continue
        for token in mut_str.split(","):
            token = token.strip()
            m = _MUT_RE.match(token)
            if m is None:
                # Try semicolon-separated fallback (some pipelines use ";")
                continue
            _orig_aa, imgt_pos, new_aa = m.group(1), m.group(2), m.group(3)
            records.append((row_idx, imgt_pos, new_aa))
            all_positions.add(imgt_pos)

    if not all_positions:
        return np.zeros((len(library_df), 0), dtype=np.int8), []

    # Sort positions: numeric part first, then insertion code.
    def _sort_key(pos: str) -> tuple[int, str]:
        digits = []
        suffix = ""
        for ch in pos:
            if ch.isdigit():
                digits.append(ch)
            else:
                suffix += ch
        return (int("".join(digits)) if digits else 0, suffix)

    sorted_positions = sorted(all_positions, key=_sort_key)
    pos_to_col: dict[str, int] = {p: i for i, p in enumerate(sorted_positions)}

    n_variants = len(library_df)
    n_positions = len(sorted_positions)
    matrix = np.zeros((n_variants, n_positions), dtype=np.int8)

    for row_idx, imgt_pos, new_aa in records:
        col = pos_to_col[imgt_pos]
        matrix[row_idx, col] = _AA_TO_INT.get(new_aa, 0)

    return matrix, sorted_positions


def compute_umap_embedding(
    mutation_matrix: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute a 2-D UMAP embedding of the mutation matrix.

    Parameters
    ----------
    mutation_matrix : np.ndarray
        Integer-encoded matrix of shape ``(n_variants, n_positions)``.
    n_neighbors : int
        Number of neighbours for UMAP (default 30).
    min_dist : float
        Minimum distance for UMAP (default 0.1).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_variants, 2)`` with UMAP coordinates.
    """
    import umap  # lazy import — heavy dependency

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, mutation_matrix.shape[0] - 1),
        min_dist=min_dist,
        metric="hamming",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(mutation_matrix)
    return np.asarray(embedding, dtype=np.float64)


def mutation_frequency_matrix(
    library_df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Compute a position × amino-acid frequency matrix.

    Parameters
    ----------
    library_df : pd.DataFrame
        Library DataFrame with a ``mutations`` column.
    top_n : int or None
        If set, restrict to the top *N* variants by ``combined_score``.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape ``(n_positions, 20)`` with amino-acid columns and
        IMGT position index.  Values are frequencies in [0, 1].
    """
    df = library_df
    if top_n is not None and "combined_score" in df.columns:
        df = df.nlargest(top_n, "combined_score")

    n_variants = len(df)
    if n_variants == 0:
        return pd.DataFrame()

    counts: dict[str, dict[str, int]] = {}

    for mut_str in df["mutations"]:
        if not isinstance(mut_str, str) or not mut_str.strip():
            continue
        for token in mut_str.split(","):
            token = token.strip()
            m = _MUT_RE.match(token)
            if m is None:
                continue
            imgt_pos, new_aa = m.group(2), m.group(3)
            if imgt_pos not in counts:
                counts[imgt_pos] = {}
            counts[imgt_pos][new_aa] = counts[imgt_pos].get(new_aa, 0) + 1

    if not counts:
        return pd.DataFrame()

    # Sort positions consistently.
    def _sort_key(pos: str) -> tuple[int, str]:
        digits = []
        suffix = ""
        for ch in pos:
            if ch.isdigit():
                digits.append(ch)
            else:
                suffix += ch
        return (int("".join(digits)) if digits else 0, suffix)

    sorted_positions = sorted(counts.keys(), key=_sort_key)
    aa_list = _AA_ORDER

    freq = np.zeros((len(sorted_positions), len(aa_list)), dtype=np.float64)
    for i, pos in enumerate(sorted_positions):
        for j, aa in enumerate(aa_list):
            freq[i, j] = counts[pos].get(aa, 0) / n_variants

    return pd.DataFrame(freq, index=sorted_positions, columns=aa_list)


def pairwise_cooccurrence_matrix(
    library_df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Compute a symmetric position-pair co-occurrence matrix.

    Parameters
    ----------
    library_df : pd.DataFrame
        Library DataFrame with a ``mutations`` column.
    top_n : int or None
        If set, restrict to the top *N* variants by ``combined_score``.

    Returns
    -------
    pd.DataFrame
        Symmetric DataFrame where rows and columns are IMGT position labels and
        values are co-occurrence counts (number of variants in which both
        positions are mutated).
    """
    df = library_df
    if top_n is not None and "combined_score" in df.columns:
        df = df.nlargest(top_n, "combined_score")

    # Collect per-variant mutated positions.
    variant_positions: list[list[str]] = []
    all_positions: set[str] = set()

    for mut_str in df["mutations"]:
        positions_in_variant: list[str] = []
        if isinstance(mut_str, str) and mut_str.strip():
            for token in mut_str.split(","):
                token = token.strip()
                m = _MUT_RE.match(token)
                if m is None:
                    continue
                positions_in_variant.append(m.group(2))
        variant_positions.append(positions_in_variant)
        all_positions.update(positions_in_variant)

    if not all_positions:
        return pd.DataFrame()

    def _sort_key(pos: str) -> tuple[int, str]:
        digits = []
        suffix = ""
        for ch in pos:
            if ch.isdigit():
                digits.append(ch)
            else:
                suffix += ch
        return (int("".join(digits)) if digits else 0, suffix)

    sorted_positions = sorted(all_positions, key=_sort_key)
    pos_to_idx = {p: i for i, p in enumerate(sorted_positions)}
    n = len(sorted_positions)

    cooc = np.zeros((n, n), dtype=np.int64)
    for positions_in_variant in variant_positions:
        idxs = [pos_to_idx[p] for p in positions_in_variant]
        for i_idx in range(len(idxs)):
            for j_idx in range(i_idx, len(idxs)):
                cooc[idxs[i_idx], idxs[j_idx]] += 1
                if i_idx != j_idx:
                    cooc[idxs[j_idx], idxs[i_idx]] += 1

    return pd.DataFrame(cooc, index=sorted_positions, columns=sorted_positions)
