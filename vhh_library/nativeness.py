"""AbNatiV-based nativeness scorer for VHH sequences.

This module wraps the AbNatiV library (https://github.com/Exscientia/AbNatiV)
to score VHH sequences for nativeness — how closely they resemble natural
camelid nanobody repertoires.  AbNatiV uses a VQ-VAE model that produces
interpretable nativeness scores approaching 1 for highly native sequences
(0.8 is the recommended threshold).

The scorer follows the same interface as other scorers in this project
(:class:`~vhh_library.developability.SurfaceHydrophobicityScorer`,
:class:`~vhh_library.stability.StabilityScorer`):

* ``score(vhh)`` → ``dict`` with at least ``"composite_score"``
* ``predict_mutation_effect(vhh, position, new_aa)`` → ``float``

Acceleration
~~~~~~~~~~~~
For batch scoring of variant libraries derived from a single parent sequence,
:meth:`NativenessScorer.score_batch_prealigned` aligns the parent once via
ANARCI and constructs pre-aligned AHo strings for all variants.  This bypasses
the per-sequence ANARCI alignment bottleneck (~0.3–0.5 s each), reducing
alignment cost from O(n_variants) to O(1).
"""

from __future__ import annotations

import functools
import logging
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)

# Expected length of an AHo-aligned antibody variable domain string.
_AHO_ALIGNED_LENGTH = 149


def _check_abnativ_deps() -> None:
    """Raise a clear ``ImportError`` when abnativ is missing."""
    # Apply the Windows compatibility patch *before* importing abnativ,
    # because ``abnativ.init`` calls ``os.uname()`` at module level.
    from vhh_library._abnativ_compat import patch_abnativ_platform

    patch_abnativ_platform()

    try:
        import abnativ  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Nativeness scoring requires the AbNatiV package. "
            "Install it with:  pip install abnativ\n"
            "Then download model weights with:  abnativ init\n"
            "See https://github.com/Exscientia/AbNatiV for details."
        ) from exc


class NativenessScorer:
    """Score VHH sequences for nativeness using AbNatiV.

    AbNatiV scores approach 1.0 for highly native sequences.  The output
    is already normalised to [0, 1], so no additional normalisation is
    applied.

    Parameters
    ----------
    model_type : str
        AbNatiV model variant to use.  ``"VHH"`` (v1) or ``"VHH2"`` (v2).
        Default is ``"VHH"`` for broad compatibility.
    batch_size : int
        Batch size for AbNatiV inference.
    cache_maxsize : int
        Maximum number of single-sequence score results to cache.  Set to
        0 to disable caching.
    """

    def __init__(
        self,
        model_type: str = "VHH",
        batch_size: int = 128,
        cache_maxsize: int = 128,
        ncpus: int = 1,
    ) -> None:
        _check_abnativ_deps()
        self._model_type = model_type
        self._batch_size = batch_size
        self._ncpus = ncpus
        self._scoring_fn = None  # lazy-loaded
        # Cache for parent AHo alignment — avoids re-aligning the same parent.
        self._aho_cache: dict[str, tuple[str, dict[int, int]]] = {}

        # Build a per-instance LRU cache keyed by amino acid sequence string.
        # Using a wrapper + functools.lru_cache on a nested function gives us
        # a proper per-instance cache without making the class unhashable.
        if cache_maxsize > 0:

            @functools.lru_cache(maxsize=cache_maxsize)
            def _cached_score(sequence: str) -> dict:
                scores = self._score_sequences([sequence])
                return {"composite_score": scores[0]}

        else:
            # Caching disabled — always score from scratch.
            def _cached_score(sequence: str) -> dict:
                scores = self._score_sequences([sequence])
                return {"composite_score": scores[0]}

        self._cached_score = _cached_score

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_scoring_fn(self):
        """Lazy-load the AbNatiV scoring function.

        Raises
        ------
        FileNotFoundError
            If the AbNatiV pre-trained model weights have not been
            downloaded.  The error message explains how to fix this.
        """
        if self._scoring_fn is None:
            # Verify that model weights exist before attempting to load
            # the scoring function.  This gives a clear, actionable error
            # instead of an opaque FileNotFoundError deep inside AbNatiV.
            try:
                from abnativ.init import PRETRAINED_MODELS_DIR
            except ImportError:
                PRETRAINED_MODELS_DIR = None  # pragma: no cover

            if PRETRAINED_MODELS_DIR is not None:
                if not os.path.isdir(PRETRAINED_MODELS_DIR):
                    raise FileNotFoundError(
                        f"AbNatiV model weights not found at "
                        f"{PRETRAINED_MODELS_DIR!r}.\n"
                        "Download them by running:  vhh-init\n"
                        "(or equivalently:  abnativ init)"
                    )

            from abnativ.model.scoring_functions import abnativ_scoring

            self._scoring_fn = abnativ_scoring
            logger.info("AbNatiV scoring function loaded (model_type=%s)", self._model_type)
        return self._scoring_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_seq_records(sequences: list[str]):
        """Convert plain amino acid strings to BioPython SeqRecords."""
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        return [SeqRecord(Seq(seq), id=f"seq_{i}", description="") for i, seq in enumerate(sequences)]

    def _score_sequences(self, sequences: list[str]) -> list[float]:
        """Score a list of amino acid sequences and return nativeness scores.

        Returns a list of floats in [0, 1], one per input sequence.
        """
        scoring_fn = self._load_scoring_fn()
        seq_records = self._make_seq_records(sequences)

        # Suppress verbose ANARCI log output that AbNatiV produces during
        # its internal alignment step.
        anarci_logger = logging.getLogger("anarci")
        previous_level = anarci_logger.level
        anarci_logger.setLevel(logging.WARNING)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                scores_df, _ = scoring_fn(
                    model_type=self._model_type,
                    seq_records=seq_records,
                    batch_size=self._batch_size,
                    mean_score_only=True,
                    do_align=True,
                    is_VHH=True,
                    output_dir=tmpdir,
                    output_id="nativeness",
                    run_parall_al=self._ncpus if self._ncpus > 1 else 1,
                    verbose=False,
                )
        finally:
            anarci_logger.setLevel(previous_level)

        return self._extract_scores(scores_df, len(sequences))

    # ------------------------------------------------------------------
    # Score extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scores(scores_df: "pd.DataFrame", expected_count: int) -> list[float]:
        """Extract nativeness scores from an AbNatiV output DataFrame.

        Handles varying column names across AbNatiV versions and pads
        with a neutral score (0.5) when AbNatiV returns fewer rows than
        expected (e.g. ANARCI silently rejects a sequence).
        """
        # The scores DataFrame has a 'score' column (or similar).
        # Extract the nativeness score for each sequence.
        if "score" in scores_df.columns:
            raw_scores = scores_df["score"].tolist()
        elif "nativeness_score" in scores_df.columns:
            raw_scores = scores_df["nativeness_score"].tolist()
        else:
            # Fall back to the last numeric column
            numeric_cols = scores_df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                raw_scores = scores_df[numeric_cols[-1]].tolist()
            else:
                logger.warning("Could not find score column in AbNatiV output; defaulting to 0.5")
                raw_scores = [0.5] * expected_count

        # Guard against AbNatiV returning fewer rows than input sequences
        # (e.g. its internal ANARCI alignment silently rejects a sequence).
        if len(raw_scores) != expected_count:
            logger.warning(
                "AbNatiV returned %d scores for %d input sequences; missing scores will default to 0.5",
                len(raw_scores),
                expected_count,
            )
            # Pad with neutral score for any sequences AbNatiV failed to score
            raw_scores.extend([0.5] * (expected_count - len(raw_scores)))

        # AbNatiV scores are already in [0, 1] (approaching 1 for native).
        # Clamp to be safe.
        return [max(0.0, min(1.0, float(s))) for s in raw_scores]

    # ------------------------------------------------------------------
    # Pre-aligned scoring (ANARCI bypass for variant libraries)
    # ------------------------------------------------------------------

    def _align_parent(self, sequence: str) -> tuple[str, dict[int, int]]:
        """Align *sequence* once with ANARCI and cache the AHo alignment.

        Results are cached per instance keyed by the raw amino acid string,
        so repeated calls with the same parent (e.g. across chunked batches)
        are essentially free.  The cache is **not** thread-safe; concurrent
        callers should use separate scorer instances.

        Returns
        -------
        tuple[str, dict[int, int]]
            ``(aho_aligned, seq_idx_to_aho_idx)`` where *aho_aligned* is
            the 149-character AHo-aligned string and *seq_idx_to_aho_idx*
            maps 0-based raw-sequence indices to their position in the
            aligned string.
        """
        if sequence in self._aho_cache:
            return self._aho_cache[sequence]

        scoring_fn = self._load_scoring_fn()
        seq_records = self._make_seq_records([sequence])

        anarci_logger = logging.getLogger("anarci")
        previous_level = anarci_logger.level
        anarci_logger.setLevel(logging.WARNING)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                scores_df, _ = scoring_fn(
                    model_type=self._model_type,
                    seq_records=seq_records,
                    batch_size=1,
                    mean_score_only=True,
                    do_align=True,
                    is_VHH=True,
                    output_dir=tmpdir,
                    output_id="parent_align",
                    run_parall_al=False,
                    verbose=False,
                )
        finally:
            anarci_logger.setLevel(previous_level)

        if scores_df.empty or "aligned_seq" not in scores_df.columns:
            raise ValueError("Failed to obtain AHo alignment for parent sequence")

        aho_aligned: str = scores_df["aligned_seq"].iloc[0]

        if len(aho_aligned) != _AHO_ALIGNED_LENGTH:
            raise ValueError(
                f"Parent AHo alignment has unexpected length {len(aho_aligned)} (expected {_AHO_ALIGNED_LENGTH})"
            )

        # Build mapping: raw sequence 0-based index → AHo aligned string index.
        seq_idx_to_aho_idx: dict[int, int] = {}
        seq_idx = 0
        for aho_idx, char in enumerate(aho_aligned):
            if char != "-":
                seq_idx_to_aho_idx[seq_idx] = aho_idx
                seq_idx += 1

        self._aho_cache[sequence] = (aho_aligned, seq_idx_to_aho_idx)
        logger.info(
            "Cached AHo alignment for parent (%d residues → %d AHo positions)",
            len(sequence),
            len(seq_idx_to_aho_idx),
        )
        return aho_aligned, seq_idx_to_aho_idx

    def score_batch_prealigned(
        self,
        parent_seq: str,
        variant_seqs: list[str],
    ) -> list[float]:
        """Score variants by reusing the parent's AHo alignment — O(1) ANARCI.

        For variant libraries derived from a single parent, this method
        eliminates the per-sequence ANARCI alignment bottleneck:

        1. Align the parent sequence **once** via ANARCI to obtain its
           149-character AHo-aligned representation.
        2. For each variant (which differs by only a few amino acids),
           construct the AHo-aligned string by swapping mutated positions
           in the parent alignment.
        3. Score all pre-aligned variants in a single ``abnativ_scoring``
           call with ``do_align=False``.

        Falls back to :meth:`score_batch` if pre-alignment fails (e.g.
        parent alignment error, length mismatch).

        Parameters
        ----------
        parent_seq : str
            Raw amino acid string of the parent / wild-type sequence.
        variant_seqs : list[str]
            Raw amino acid strings of the variant sequences.

        Returns
        -------
        list[float]
            Nativeness scores in [0, 1] for each variant.
        """
        if not variant_seqs:
            return []

        # Step 1: obtain parent AHo alignment
        try:
            aho_parent, seq_to_aho = self._align_parent(parent_seq)
        except Exception:
            logger.warning(
                "Pre-alignment of parent failed; falling back to standard batch scoring",
                exc_info=True,
            )
            return self._score_sequences(variant_seqs)

        aho_parent_list = list(aho_parent)

        # Step 2: build pre-aligned AHo strings for each variant
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        prealigned_records: list[tuple[int, SeqRecord]] = []
        fallback_indices: list[int] = []

        for i, variant_seq in enumerate(variant_seqs):
            if len(variant_seq) != len(parent_seq):
                # Length differs — alignment may change, fall back
                fallback_indices.append(i)
                continue

            variant_aho = aho_parent_list.copy()
            ok = True
            for seq_idx, (p_aa, v_aa) in enumerate(zip(parent_seq, variant_seq)):
                if p_aa != v_aa:
                    aho_idx = seq_to_aho.get(seq_idx)
                    if aho_idx is None:
                        ok = False
                        break
                    variant_aho[aho_idx] = v_aa

            if not ok:
                fallback_indices.append(i)
                continue

            aho_str = "".join(variant_aho)
            if len(aho_str) != _AHO_ALIGNED_LENGTH:
                fallback_indices.append(i)
                continue

            prealigned_records.append((i, SeqRecord(Seq(aho_str), id=f"var_{i}", description="")))

        # Step 3: score pre-aligned variants (no ANARCI)
        scores: list[float] = [0.5] * len(variant_seqs)

        if prealigned_records:
            scoring_fn = self._load_scoring_fn()
            ordered_records = [rec for _, rec in prealigned_records]
            ordered_indices = [idx for idx, _ in prealigned_records]

            anarci_logger = logging.getLogger("anarci")
            previous_level = anarci_logger.level
            anarci_logger.setLevel(logging.WARNING)
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    result_df, _ = scoring_fn(
                        model_type=self._model_type,
                        seq_records=ordered_records,
                        batch_size=self._batch_size,
                        mean_score_only=True,
                        do_align=False,
                        is_VHH=True,
                        output_dir=tmpdir,
                        output_id="prealigned",
                        run_parall_al=False,
                        verbose=False,
                    )
            except Exception:
                logger.warning(
                    "Pre-aligned scoring failed; falling back to standard batch",
                    exc_info=True,
                )
                return self._score_sequences(variant_seqs)
            finally:
                anarci_logger.setLevel(previous_level)

            raw = self._extract_scores(result_df, len(ordered_records))
            for idx, score in zip(ordered_indices, raw):
                scores[idx] = score

        # Step 4: handle fallback sequences (different length, mapping issues)
        if fallback_indices:
            fallback_seqs = [variant_seqs[i] for i in fallback_indices]
            logger.info(
                "Scoring %d fallback sequences with ANARCI alignment",
                len(fallback_seqs),
            )
            fallback_scores = self._score_sequences(fallback_seqs)
            for idx, score in zip(fallback_indices, fallback_scores):
                scores[idx] = score

        return scores

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, vhh: "VHHSequence") -> dict:
        """Score a single VHH sequence for nativeness.

        Results are cached by amino acid sequence string so repeated calls
        with the same sequence (e.g. the parent in
        :meth:`predict_mutation_effect`) are essentially free.

        Returns
        -------
        dict
            At least ``{"composite_score": float}`` where the score is in
            [0, 1] (higher = more native).
        """
        return self._cached_score(vhh.sequence)

    def predict_mutation_effect(self, vhh: "VHHSequence", position: int | str, new_aa: str) -> float:
        """Return the change in nativeness when mutating *position* to *new_aa*.

        A positive delta means the mutation *increases* nativeness.

        The parent score is cached, so calling this method repeatedly for
        different mutations on the same parent sequence does not re-invoke
        AbNatiV for the parent.
        """
        from vhh_library.sequence import VHHSequence as _VHHSequence

        parent_score = self.score(vhh)["composite_score"]
        mutant = _VHHSequence.mutate(vhh, position, new_aa)
        mutant_score = self.score(mutant)["composite_score"]
        return mutant_score - parent_score

    def score_batch(self, sequences: list[str]) -> list[float]:
        """Score multiple amino acid sequences in batch for efficiency.

        This method bypasses the single-sequence cache and sends all
        sequences to AbNatiV in a single batch call for throughput.

        Parameters
        ----------
        sequences : list[str]
            Raw amino acid strings.

        Returns
        -------
        list[float]
            Nativeness scores in [0, 1] for each sequence.
        """
        if not sequences:
            return []
        scores = self._score_sequences(sequences)
        # Defense-in-depth: _score_sequences already pads, but callers may
        # subclass or monkey-patch the method, so guard here too.
        if len(scores) != len(sequences):
            logger.warning(
                "score_batch: expected %d scores but got %d; padding with 0.5",
                len(sequences),
                len(scores),
            )
            scores.extend([0.5] * (len(sequences) - len(scores)))
        return scores
