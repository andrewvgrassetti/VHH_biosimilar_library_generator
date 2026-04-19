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
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vhh_library.sequence import VHHSequence

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        model_type: str = "VHH",
        batch_size: int = 128,
    ) -> None:
        _check_abnativ_deps()
        self._model_type = model_type
        self._batch_size = batch_size
        self._scoring_fn = None  # lazy-loaded

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
                run_parall_al=1,  # AbNatiV's API parameter name (upstream typo)
            )

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
                raw_scores = [0.5] * len(sequences)

        # Guard against AbNatiV returning fewer rows than input sequences
        # (e.g. its internal ANARCI alignment silently rejects a sequence).
        if len(raw_scores) != len(sequences):
            logger.warning(
                "AbNatiV returned %d scores for %d input sequences; missing scores will default to 0.5",
                len(raw_scores),
                len(sequences),
            )
            # Pad with neutral score for any sequences AbNatiV failed to score
            raw_scores.extend([0.5] * (len(sequences) - len(raw_scores)))

        # AbNatiV scores are already in [0, 1] (approaching 1 for native).
        # Clamp to be safe.
        return [max(0.0, min(1.0, float(s))) for s in raw_scores]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, vhh: "VHHSequence") -> dict:
        """Score a single VHH sequence for nativeness.

        Returns
        -------
        dict
            At least ``{"composite_score": float}`` where the score is in
            [0, 1] (higher = more native).
        """
        scores = self._score_sequences([vhh.sequence])
        return {"composite_score": scores[0]}

    def predict_mutation_effect(self, vhh: "VHHSequence", position: int | str, new_aa: str) -> float:
        """Return the change in nativeness when mutating *position* to *new_aa*.

        A positive delta means the mutation *increases* nativeness.
        """
        from vhh_library.sequence import VHHSequence as _VHHSequence

        parent_score = self.score(vhh)["composite_score"]
        mutant = _VHHSequence.mutate(vhh, position, new_aa)
        mutant_score = self.score(mutant)["composite_score"]
        return mutant_score - parent_score

    def score_batch(self, sequences: list[str]) -> list[float]:
        """Score multiple amino acid sequences in batch for efficiency.

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
