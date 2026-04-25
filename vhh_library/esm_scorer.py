"""ESM-2 pseudo-log-likelihood stability scorer with efficient batch architecture.

This module provides :class:`ESMStabilityScorer`, which wraps Facebook's ESM-2
protein language models and exposes several scoring strategies optimised for
large variant libraries:

* **Full PLL** (``score_single`` / ``score_batch``): standard masked-marginal
  pseudo-log-likelihood for individual sequences.
* **Delta PLL** (``score_delta``): wild-type logits are cached; only mutated
  positions are re-evaluated, dramatically reducing forward passes.
* **Progressive funnel** (``score_library_progressive``): a multi-stage filter
  that applies fast legacy heuristics first, then ESM-2 delta scoring, and
  optionally re-scores the survivors with a larger model.
  **Deprecated** — NanoMelt Tm is now the primary stability ranking signal.
  Use ESM-2 delta PLL only as an optional prior filter.

Scores are persisted in a lightweight SQLite cache so repeated runs skip
redundant GPU/CPU work.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model tier registry
# ---------------------------------------------------------------------------

_MODEL_TIERS: dict[str, str] = {
    "t6_8M": "esm2_t6_8M_UR50D",
    "t12_35M": "esm2_t12_35M_UR50D",
    "t33_650M": "esm2_t33_650M_UR50D",
    "t36_3B": "esm2_t36_3B_UR50D",
}

_CPU_DEFAULT_TIER = "t6_8M"
_GPU_DEFAULT_TIER = "t33_650M"
_CPU_DEFAULT_BATCH = 64
_GPU_DEFAULT_BATCH = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq_hash(model_name: str, sequence: str) -> str:
    """Return a deterministic hash key for *(model_name, sequence)*."""
    return hashlib.sha256(f"{model_name}:{sequence}".encode()).hexdigest()


def _check_ml_deps() -> None:
    """Raise a clear ``ImportError`` when torch / esm are missing."""
    try:
        import esm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ESM-2 scoring requires PyTorch and fair-esm. Install them with:  pip install torch fair-esm"
        ) from exc


# ---------------------------------------------------------------------------
# SQLite result cache
# ---------------------------------------------------------------------------


class _ScoreCache:
    """Thin wrapper around an SQLite database for PLL score persistence."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # Lazy connect so the file is only created when actually needed.
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("CREATE TABLE IF NOT EXISTS scores (seq_hash TEXT PRIMARY KEY, score REAL)")
            self._conn.commit()
        return self._conn

    def get(self, seq_hash: str) -> float | None:
        cur = self._connect().execute("SELECT score FROM scores WHERE seq_hash = ?", (seq_hash,))
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, seq_hash: str, score: float) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO scores (seq_hash, score) VALUES (?, ?)",
            (seq_hash, score),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class ESMStabilityScorer:
    """ESM-2 pseudo-log-likelihood scorer with caching and delta-scoring.

    Parameters
    ----------
    model_tier:
        One of ``"auto"``, ``"t6_8M"``, ``"t12_35M"``, ``"t33_650M"``, or
        ``"t36_3B"``.  ``"auto"`` selects *t6_8M* on CPU and *t33_650M* when a
        CUDA GPU is detected.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
    batch_size:
        Mini-batch size for batched inference.  ``None`` picks a sensible
        default (64 on CPU, 256 on GPU).
    cache_dir:
        Directory for the SQLite score cache.  ``None`` disables caching.
    """

    def __init__(
        self,
        model_tier: str = "auto",
        device: str = "auto",
        batch_size: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        _check_ml_deps()

        import torch

        # Device selection
        if device == "auto":
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device_str = device

        # Model tier selection
        if model_tier == "auto":
            self._tier = _GPU_DEFAULT_TIER if self._device_str == "cuda" else _CPU_DEFAULT_TIER
        else:
            if model_tier not in _MODEL_TIERS:
                raise ValueError(f"Unknown model_tier {model_tier!r}. Choose from: {', '.join(_MODEL_TIERS)}")
            self._tier = model_tier

        self._model_name = _MODEL_TIERS[self._tier]

        # Batch size
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            self._batch_size = _GPU_DEFAULT_BATCH if self._device_str == "cuda" else _CPU_DEFAULT_BATCH

        # Model state (lazy)
        self._model = None
        self._alphabet = None
        self._batch_converter = None

        # Disk cache
        self._cache: _ScoreCache | None = None
        self._cache_dir: str | None = cache_dir
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"esm2_{self._tier}_scores.sqlite"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = _ScoreCache(cache_path)

        # Wild-type logits cache for delta scoring
        self._wt_logits_cache: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import esm

        loader = getattr(esm.pretrained, self._model_name)
        model, alphabet = loader()
        model = model.to(self._device_str)
        model.eval()

        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        logger.info("Loaded ESM-2 model %s on %s", self._model_name, self._device_str)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached(self, seq_hash: str) -> float | None:
        if self._cache is None:
            return None
        return self._cache.get(seq_hash)

    def _put_cache(self, seq_hash: str, score: float) -> None:
        if self._cache is not None:
            self._cache.put(seq_hash, score)

    # ------------------------------------------------------------------
    # Full PLL scoring
    # ------------------------------------------------------------------

    def score_single(self, sequence: str) -> float:
        """Return the pseudo-log-likelihood for *sequence*."""
        return self.score_batch([sequence])[0]

    def score_batch(self, sequences: list[str]) -> list[float]:
        """Compute PLL for a list of sequences with batched inference."""
        import torch

        self._load_model()

        results: list[float | None] = [None] * len(sequences)
        to_compute: list[tuple[int, str]] = []

        # Check cache first
        for i, seq in enumerate(sequences):
            h = _seq_hash(self._model_name, seq)
            cached = self._get_cached(h)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, seq))

        # Batch inference for uncached sequences
        for batch_start in range(0, len(to_compute), self._batch_size):
            batch = to_compute[batch_start : batch_start + self._batch_size]
            data = [(f"seq_{idx}", seq) for idx, seq in batch]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self._device_str)

            with torch.no_grad():
                logits = self._model(tokens)["logits"]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            for j, (orig_idx, seq) in enumerate(batch):
                pll = 0.0
                for pos in range(1, len(seq) + 1):
                    token_idx = tokens[j, pos].item()
                    pll += log_probs[j, pos, token_idx].item()
                results[orig_idx] = pll
                h = _seq_hash(self._model_name, seq)
                self._put_cache(h, pll)

        return [r for r in results]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Delta scoring (wild-type caching)
    # ------------------------------------------------------------------

    def _get_wt_logits(self, parent_sequence: str):
        """Get or compute cached wild-type log-probabilities tensor."""
        import torch

        if parent_sequence in self._wt_logits_cache:
            return self._wt_logits_cache[parent_sequence]

        self._load_model()
        _, _, tokens = self._batch_converter([("wt", parent_sequence)])
        tokens = tokens.to(self._device_str)

        with torch.no_grad():
            logits = self._model(tokens)["logits"]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Keep on device for subsequent delta lookups
        self._wt_logits_cache[parent_sequence] = (tokens, log_probs)
        return tokens, log_probs

    def score_delta(
        self,
        parent_sequence: str,
        variants: list[tuple[list[int], list[str]]],
    ) -> list[float]:
        """Efficient delta scoring using cached wild-type logits.

        For each variant, only the mutated positions are evaluated instead of
        running a full forward pass.

        Parameters
        ----------
        parent_sequence:
            The wild-type (parent) amino-acid sequence.
        variants:
            Each element is ``(positions_0indexed, new_aas)`` where
            *positions_0indexed* lists the 0-based residue indices that are
            mutated and *new_aas* lists the corresponding replacement amino
            acids.

        Returns
        -------
        list[float]
            Approximate ΔPLL for each variant (positive = improved).
        """

        self._load_model()
        wt_tokens, wt_log_probs = self._get_wt_logits(parent_sequence)

        results: list[float] = []
        alphabet = self._alphabet

        for positions, new_aas in variants:
            delta = 0.0
            for pos_0, new_aa in zip(positions, new_aas):
                # ESM tokens are offset by +1 (position 0 in seq → token index 1)
                tok_pos = pos_0 + 1
                old_aa = parent_sequence[pos_0]

                old_tok = alphabet.get_idx(old_aa)
                new_tok = alphabet.get_idx(new_aa)

                old_logp = wt_log_probs[0, tok_pos, old_tok].item()
                new_logp = wt_log_probs[0, tok_pos, new_tok].item()
                delta += new_logp - old_logp
            results.append(delta)

        return results

    # ------------------------------------------------------------------
    # Progressive funnel
    # ------------------------------------------------------------------

    def score_library_progressive(
        self,
        parent,  # VHHSequence
        library_df: pd.DataFrame,
        stage1_top_frac: float = 0.2,
        stage2_top_frac: float = 0.25,
        stage3: bool = False,
    ) -> pd.DataFrame:
        """Multi-stage progressive filter with ESM-2 scoring.

        .. deprecated::
            NanoMelt Tm is now the primary stability ranking signal.
            Use ESM-2 delta PLL only as an optional prior filter.
            This method will be removed in a future version.

        Stage 1: Keep top *stage1_top_frac* by the existing ``combined_score``
        (legacy heuristic + humanness composite).

        Stage 2: Score survivors with ESM-2 delta PLL and keep top
        *stage2_top_frac*.

        Stage 3 (optional): Re-score with the larger *t33_650M* model for
        final ranking.

        Adds columns ``esm2_pll``, ``esm2_delta_pll``, and ``esm2_rank`` to
        the returned DataFrame.
        """
        warnings.warn(
            "score_library_progressive() is deprecated. NanoMelt Tm is now the "
            "primary stability ranking signal. Use ESM-2 delta PLL only as an "
            "optional prior filter. This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if library_df.empty:
            library_df["esm2_pll"] = pd.Series(dtype=float)
            library_df["esm2_delta_pll"] = pd.Series(dtype=float)
            library_df["esm2_rank"] = pd.Series(dtype=int)
            return library_df

        df = library_df.copy()

        # Stage 1 — fast heuristic filter
        n_stage1 = max(1, int(len(df) * stage1_top_frac))
        df = df.nlargest(n_stage1, "combined_score").reset_index(drop=True)
        logger.info("Progressive stage 1: kept %d / %d variants", len(df), len(library_df))

        # Stage 2 — ESM-2 delta scoring
        logger.info("Progressive stage 2: computing ESM-2 delta PLL for %d variants…", len(df))
        parent_seq = parent.sequence
        variants: list[tuple[list[int], list[str]]] = []
        for _, row in df.iterrows():
            seq = row["aa_sequence"]
            positions: list[int] = []
            new_aas: list[str] = []
            for i, (p_aa, v_aa) in enumerate(zip(parent_seq, seq)):
                if p_aa != v_aa:
                    positions.append(i)
                    new_aas.append(v_aa)
            variants.append((positions, new_aas))

        delta_scores = self.score_delta(parent_seq, variants)
        df["esm2_delta_pll"] = delta_scores
        logger.info("Progressive stage 2: delta PLL scoring complete")

        # Also compute full PLL for the stage-2 survivors
        n_stage2 = max(1, int(len(df) * stage2_top_frac))
        df = df.nlargest(n_stage2, "esm2_delta_pll").reset_index(drop=True)
        logger.info("Progressive stage 2: computing full PLL for top %d variants…", len(df))

        seqs = df["aa_sequence"].tolist()
        pll_scores = self.score_batch(seqs)
        df["esm2_pll"] = pll_scores
        logger.info("Progressive stage 2: full PLL scoring complete, kept %d variants", len(df))

        # Stage 3 (optional) — re-score with larger model
        if stage3 and self._tier != "t33_650M":
            logger.info("Progressive stage 3: re-scoring with t33_650M")
            stage3_scorer = ESMStabilityScorer(
                model_tier="t33_650M",
                device=self._device_str,
                batch_size=self._batch_size,
                cache_dir=self._cache_dir,
            )
            pll_stage3 = stage3_scorer.score_batch(seqs)
            df["esm2_pll"] = pll_stage3

        df["esm2_rank"] = df["esm2_pll"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("esm2_pll", ascending=False).reset_index(drop=True)
        return df
