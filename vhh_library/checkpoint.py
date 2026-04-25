"""Disk-based checkpointing for long-running library generation.

Provides helpers to save, load, and clean up intermediate DataFrames so
that iterative library generation can resume after a Streamlit timeout or
WebSocket disconnect.

All functions are backend-agnostic (no Streamlit imports) and operate on
plain :class:`~pandas.DataFrame` objects and filesystem paths.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default subdirectory name inside the caller-supplied checkpoint root.
_CHECKPOINT_SUBDIR = "vhh_checkpoints"

# Stale-checkpoint age threshold in seconds (24 hours).
_STALE_AGE_SECONDS = 86_400


# ------------------------------------------------------------------
# Run-identity helpers
# ------------------------------------------------------------------


def compute_run_id(
    sequence: str,
    *,
    n_mutations: int = 0,
    max_variants: int = 0,
    min_mutations: int = 0,
    strategy: str = "",
    extra: str = "",
) -> str:
    """Return a short hex digest that uniquely identifies a generation run.

    The digest is derived from the VHH amino-acid *sequence* and the
    key parameters that affect the output.  Two runs with the same inputs
    will produce the same ``run_id`` so that a checkpoint written by a
    previous attempt can be detected and resumed.
    """
    parts = [
        sequence,
        str(n_mutations),
        str(max_variants),
        str(min_mutations),
        strategy,
        extra,
    ]
    blob = "|".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------


def checkpoint_dir(root: Path) -> Path:
    """Return (and lazily create) the checkpoint subdirectory under *root*."""
    d = root / _CHECKPOINT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(root: Path, run_id: str) -> Path:
    """Return the Parquet path for a given *run_id*."""
    return checkpoint_dir(root) / f"vhh_checkpoint_{run_id}.parquet"


def result_path(root: Path, run_id: str) -> Path:
    """Return the Parquet path for the *final* result of a given *run_id*."""
    return checkpoint_dir(root) / f"vhh_result_{run_id}.parquet"


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


def save_checkpoint(
    root: Path,
    run_id: str,
    df: pd.DataFrame,
    *,
    completed_rounds: int = 0,
) -> Path:
    """Write *df* to disk as a Parquet checkpoint.

    Parameters
    ----------
    root:
        Base directory (e.g. ``tempfile.gettempdir()``).
    run_id:
        Identifier returned by :func:`compute_run_id`.
    df:
        Current state of the library DataFrame.
    completed_rounds:
        Number of iterative rounds already finished.  Stored as Parquet
        metadata so the engine knows where to resume.

    Returns
    -------
    Path to the written file.
    """
    path = checkpoint_path(root, run_id)
    # Store completed_rounds in the Parquet file metadata.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df)
    meta = table.schema.metadata or {}
    meta[b"completed_rounds"] = str(completed_rounds).encode()
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)
    logger.debug("Checkpoint saved: %s (%d rows, round %d)", path, len(df), completed_rounds)
    return path


def load_checkpoint(root: Path, run_id: str) -> tuple[pd.DataFrame, int] | None:
    """Load a checkpoint for *run_id*, returning ``(df, completed_rounds)``.

    Returns ``None`` if no checkpoint exists.
    """
    path = checkpoint_path(root, run_id)
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta = table.schema.metadata or {}
        completed_rounds = int(meta.get(b"completed_rounds", b"0"))
        df = table.to_pandas()
        logger.info("Checkpoint loaded: %s (%d rows, round %d)", path, len(df), completed_rounds)
        return df, completed_rounds
    except Exception:
        logger.warning("Failed to load checkpoint %s — starting fresh", path, exc_info=True)
        return None


def save_result(root: Path, run_id: str, df: pd.DataFrame) -> Path:
    """Persist the final library DataFrame to a Parquet file.

    Returns the path to the written file.
    """
    path = result_path(root, run_id)
    df.to_parquet(path, index=False)
    logger.info("Final result saved: %s (%d rows)", path, len(df))
    return path


def load_result(root: Path, run_id: str) -> pd.DataFrame | None:
    """Load a previously saved final result, or ``None`` if absent."""
    path = result_path(root, run_id)
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
        logger.info("Result loaded from disk: %s (%d rows)", path, len(df))
        return df
    except Exception:
        logger.warning("Failed to load result %s", path, exc_info=True)
        return None


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------


def remove_checkpoint(root: Path, run_id: str) -> None:
    """Delete the checkpoint file for *run_id* (idempotent)."""
    path = checkpoint_path(root, run_id)
    if path.is_file():
        path.unlink()
        logger.debug("Checkpoint removed: %s", path)


def cleanup_stale_checkpoints(root: Path, *, max_age_seconds: int = _STALE_AGE_SECONDS) -> int:
    """Remove checkpoint and result files older than *max_age_seconds*.

    Returns the number of files deleted.
    """
    d = root / _CHECKPOINT_SUBDIR
    if not d.is_dir():
        return 0
    now = time.time()
    removed = 0
    for f in d.iterdir():
        if not f.is_file():
            continue
        if f.suffix != ".parquet":
            continue
        age = now - f.stat().st_mtime
        if age > max_age_seconds:
            f.unlink()
            logger.debug("Stale file removed: %s (age %.0fs)", f, age)
            removed += 1
    return removed
