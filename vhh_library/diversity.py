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


def _imgt_sort_key(pos: str) -> tuple[int, str]:
    """Sort key for IMGT position strings (numeric part, then insertion code)."""
    digits = []
    suffix = ""
    for ch in pos:
        if ch.isdigit():
            digits.append(ch)
        else:
            suffix += ch
    return (int("".join(digits)) if digits else 0, suffix)


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
    sorted_positions = sorted(all_positions, key=_imgt_sort_key)
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
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """Compute a 2-D UMAP embedding of the mutation matrix.

    Parameters
    ----------
    mutation_matrix : np.ndarray
        Integer-encoded matrix of shape ``(n_variants, n_positions)`` or
        a dense float embedding matrix of shape ``(n_variants, n_features)``.
    n_neighbors : int
        Number of neighbours for UMAP (default 15).
    min_dist : float
        Minimum distance for UMAP (default 0.3).
    metric : str
        Distance metric for UMAP (default ``"euclidean"``).  Use
        ``"hamming"`` when passing the integer mutation matrix directly.
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
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(mutation_matrix)
    return np.asarray(embedding, dtype=np.float64)


# Module-level cache for the ESM-2 model and tokenizer to avoid re-loading on
# every call.  Keys are (model_name, device_string) tuples.
_ESM2_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def compute_esm2_embeddings(
    sequences: list[str],
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """Compute mean-pooled ESM-2 embeddings for a list of amino-acid sequences.

    Uses the lightweight ``facebook/esm2_t6_8M_UR50D`` model for fast
    inference.  Requires ``torch`` and ``transformers`` to be installed.

    The model and tokenizer are loaded lazily on first call and cached for
    subsequent calls.

    Parameters
    ----------
    sequences : list[str]
        Amino-acid sequences (one-letter codes, no spaces).
    batch_size : int
        Number of sequences per inference batch (default 32).
    device : str
        Compute device.  ``"auto"`` (default) selects CUDA → MPS → CPU
        automatically.  Pass ``"cpu"``, ``"cuda"``, or ``"mps"`` to
        override.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_sequences, hidden_dim)`` containing
        mean-pooled per-sequence embeddings.
    """
    import torch  # lazy import
    from transformers import AutoModel, AutoTokenizer  # lazy import

    # Resolve device string.
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    model_name = "facebook/esm2_t6_8M_UR50D"
    # The cache key uses (model_name, device) only. batch_size is a processing
    # parameter that does not affect the model weights, so it is intentionally
    # excluded from the key.
    cache_key = (model_name, resolved_device)
    if cache_key not in _ESM2_MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(resolved_device)
        _ESM2_MODEL_CACHE[cache_key] = (tokenizer, model)

    tokenizer, model = _ESM2_MODEL_CACHE[cache_key]

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        # ESM tokenizer expects spaces between residues
        spaced = [" ".join(s) for s in batch]
        tokens = tokenizer(spaced, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(resolved_device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        # Mean-pool over residue positions weighted by the attention mask
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_position_frequencies(
    library_df: pd.DataFrame,
    wt_sequence: str,
    imgt_numbered: dict[str, str],
) -> pd.DataFrame:
    """Compute per-IMGT-position amino-acid frequencies across library variants.

    Parameters
    ----------
    library_df : pd.DataFrame
        Library DataFrame containing an ``aa_sequence`` column with the
        full amino-acid sequence for each variant.
    wt_sequence : str
        Wild-type amino-acid sequence (used as a fallback reference; the
        canonical WT per-position identity is taken from ``imgt_numbered``).
    imgt_numbered : dict[str, str]
        Mapping of IMGT position label → wild-type amino acid, e.g.
        ``{"1": "E", "2": "V", ..., "111A": "G"}``.

    Returns
    -------
    pd.DataFrame
        DataFrame with IMGT position labels as the index, one-letter amino
        acids as columns, and frequency values in ``[0, 1]``.  An
        additional ``wt_aa`` column contains the wild-type residue at each
        position.  Returns an empty DataFrame if ``imgt_numbered`` is empty
        or the ``aa_sequence`` column is absent.
    """
    if not imgt_numbered or "aa_sequence" not in library_df.columns:
        return pd.DataFrame()

    sorted_positions = sorted(imgt_numbered.keys(), key=_imgt_sort_key)
    n_positions = len(sorted_positions)

    sequences = library_df["aa_sequence"].tolist()
    n_variants = len(sequences)
    if n_variants == 0:
        return pd.DataFrame()

    aa_list = _AA_ORDER
    aa_to_idx: dict[str, int] = {aa: i for i, aa in enumerate(aa_list)}

    counts = np.zeros((n_positions, len(aa_list)), dtype=np.int64)
    valid_count = np.zeros(n_positions, dtype=np.int64)

    for seq in sequences:
        if not isinstance(seq, str):
            continue
        for i in range(min(n_positions, len(seq))):
            aa = seq[i]
            idx = aa_to_idx.get(aa)
            if idx is not None:
                counts[i, idx] += 1
                valid_count[i] += 1

    freq = np.where(valid_count[:, None] > 0, counts / valid_count[:, None], 0.0)

    result = pd.DataFrame(freq, index=sorted_positions, columns=aa_list)
    result["wt_aa"] = [imgt_numbered.get(pos, "") for pos in sorted_positions]
    return result


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
    sorted_positions = sorted(counts.keys(), key=_imgt_sort_key)
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

    sorted_positions = sorted(all_positions, key=_imgt_sort_key)
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
