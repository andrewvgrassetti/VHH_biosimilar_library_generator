"""Reusable background-task utility for Streamlit apps.

Runs heavy callables in daemon threads and stores status / progress / result
in ``st.session_state`` under namespaced keys so the UI can poll and display
live updates without blocking the Streamlit rerun cycle.

Thread-safety note
------------------
Streamlit's ``session_state`` is a dict-like object living in server memory.
Writes from a background thread to ``session_state`` are *de-facto* safe in
the current Streamlit architecture (single Python process, GIL-protected dict
operations).  This module relies on that assumption.  If Streamlit ever moves
to multi-process sessions the approach will need a ``threading.Lock`` or a
shared-memory store.

Disk persistence
----------------
When a background task completes, its result (and status) are persisted to a
temporary file on disk.  This allows recovery after WebSocket disconnects
caused by e.g. laptop sleep/wake cycles.  On the next session init, the app
can call :func:`recover_task` to check for an orphaned result and restore it
into the fresh ``session_state``.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import streamlit as st

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-state key helpers
# ---------------------------------------------------------------------------

_PREFIX = "bg_"


def _key(task_name: str, suffix: str) -> str:
    return f"{_PREFIX}{task_name}_{suffix}"


# Possible status values
STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


# ---------------------------------------------------------------------------
# Disk persistence helpers
# ---------------------------------------------------------------------------

_TASK_PERSIST_DIR = Path(tempfile.gettempdir()) / "vhh_bg_tasks"

# How long persisted results stay valid (1 hour).
_PERSIST_MAX_AGE_SECONDS = 3600


def _persist_path(task_name: str) -> Path:
    """Return the on-disk path for a persisted task result."""
    return _TASK_PERSIST_DIR / f"{task_name}.json"


def _persist_result(task_name: str, *, status: str, result: Any = None, error: str | None = None) -> None:
    """Write a completed task's outcome to disk so it survives session loss.

    The serialisation format mirrors the auto-save helpers in ``app.py``:
    DataFrames are stored as ``{"__type__": "DataFrame", "records": [...]}``,
    and all other values pass through ``json.dumps`` directly.
    """
    import pandas as pd

    def _serialize(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return {"__type__": "DataFrame", "records": obj.to_dict(orient="records")}
        # Lists of dicts (e.g. construct results) are already JSON-friendly.
        return obj

    payload = {
        "status": status,
        "result": _serialize(result),
        "error": error,
        "timestamp": time.time(),
    }
    try:
        _TASK_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _persist_path(task_name).write_text(json.dumps(payload))
        logger.info("Persisted result for background task %r to disk.", task_name)
    except Exception:
        logger.warning("Failed to persist background task %r result to disk.", task_name, exc_info=True)


def _load_persisted(task_name: str) -> dict | None:
    """Load a persisted task result from disk, or ``None`` if unavailable/stale."""
    import pandas as pd

    path = _persist_path(task_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())

        # Use the timestamp stored in the payload for reliable age tracking,
        # falling back to file mtime if the field is absent.
        ts = payload.get("timestamp")
        age = time.time() - (ts if ts is not None else path.stat().st_mtime)
        if age > _PERSIST_MAX_AGE_SECONDS:
            path.unlink(missing_ok=True)
            logger.info("Removed stale persisted result for task %r (%.0fs old).", task_name, age)
            return None

        # Deserialise DataFrames
        raw_result = payload.get("result")
        if isinstance(raw_result, dict) and raw_result.get("__type__") == "DataFrame":
            payload["result"] = pd.DataFrame(raw_result.get("records", []))

        return payload
    except Exception:
        logger.warning("Failed to load persisted result for task %r.", task_name, exc_info=True)
        return None


def _clear_persisted(task_name: str) -> None:
    """Remove the on-disk persisted result for *task_name*."""
    try:
        _persist_path(task_name).unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_task(
    name: str,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Launch *func* in a daemon thread and track it under *name*.

    Returns ``True`` if the task was submitted, ``False`` if a task with the
    same name is already running (duplicate guard).
    """
    status = st.session_state.get(_key(name, "status"), STATUS_IDLE)
    if status == STATUS_RUNNING:
        logger.warning("Task %r is already running — ignoring duplicate submission.", name)
        return False

    # Initialise state keys
    st.session_state[_key(name, "status")] = STATUS_RUNNING
    st.session_state[_key(name, "progress")] = 0.0
    st.session_state[_key(name, "progress_text")] = ""
    st.session_state[_key(name, "result")] = None
    st.session_state[_key(name, "error")] = None

    # Clear any stale persisted result from a previous run.
    _clear_persisted(name)

    # Capture the session_state reference so the background thread can write to
    # the *same* dict even though ``st.session_state`` is thread-local in
    # Streamlit's runtime.  We grab the underlying dict-like object once here.
    state = st.session_state

    def _worker() -> None:
        try:
            result = func(*args, **kwargs)
            # Write to session_state (may be a dead dict if session was lost).
            try:
                state[_key(name, "result")] = result
                state[_key(name, "status")] = STATUS_DONE
            except Exception:
                logger.warning(
                    "Could not write result to session_state for task %r (session may have been lost).",
                    name,
                    exc_info=True,
                )
            # Always persist to disk so the result can be recovered.
            _persist_result(name, status=STATUS_DONE, result=result)
        except Exception as exc:
            error_msg = str(exc)
            try:
                state[_key(name, "error")] = error_msg
                state[_key(name, "status")] = STATUS_ERROR
            except Exception:
                logger.warning("Could not write error to session_state for task %r.", name, exc_info=True)
            _persist_result(name, status=STATUS_ERROR, error=error_msg)
            logger.exception("Background task %r failed", name)

    thread = threading.Thread(target=_worker, daemon=True, name=f"bg-{name}")
    thread.start()
    return True


def get_task_status(name: str) -> str:
    """Return the current status string for *name*."""
    return st.session_state.get(_key(name, "status"), STATUS_IDLE)


def get_task_progress(name: str) -> tuple[float, str]:
    """Return ``(fraction, text)`` for the running task."""
    frac = st.session_state.get(_key(name, "progress"), 0.0)
    text = st.session_state.get(_key(name, "progress_text"), "")
    return frac, text


def get_task_result(name: str) -> Any:
    """Return the result stored by a completed task (or ``None``)."""
    return st.session_state.get(_key(name, "result"))


def get_task_error(name: str) -> str | None:
    """Return the error message if the task failed."""
    return st.session_state.get(_key(name, "error"))


def reset_task(name: str) -> None:
    """Reset all state keys for *name* back to idle."""
    st.session_state[_key(name, "status")] = STATUS_IDLE
    st.session_state[_key(name, "progress")] = 0.0
    st.session_state[_key(name, "progress_text")] = ""
    st.session_state[_key(name, "result")] = None
    st.session_state[_key(name, "error")] = None
    _clear_persisted(name)


def recover_task(name: str) -> Any | None:
    """Attempt to recover a completed background-task result from disk.

    If a persisted result file exists for *name* and is not stale, restore it
    into ``session_state`` and return the result (or the error string for failed
    tasks).  Returns ``None`` when there is nothing to recover.

    This should be called during session initialisation so that results
    produced while the WebSocket was disconnected (e.g. laptop sleep) are
    not lost.
    """
    payload = _load_persisted(name)
    if payload is None:
        return None

    persisted_status = payload.get("status")
    if persisted_status == STATUS_DONE:
        result = payload.get("result")
        st.session_state[_key(name, "status")] = STATUS_DONE
        st.session_state[_key(name, "result")] = result
        st.session_state[_key(name, "progress")] = 1.0
        st.session_state[_key(name, "progress_text")] = ""
        st.session_state[_key(name, "error")] = None
        logger.info("Recovered completed result for background task %r from disk.", name)
        return result

    if persisted_status == STATUS_ERROR:
        error = payload.get("error", "Unknown error")
        st.session_state[_key(name, "status")] = STATUS_ERROR
        st.session_state[_key(name, "error")] = error
        st.session_state[_key(name, "result")] = None
        st.session_state[_key(name, "progress")] = 0.0
        st.session_state[_key(name, "progress_text")] = ""
        logger.info("Recovered error for background task %r from disk.", name)
        return None

    return None


def is_task_running(name: str) -> bool:
    """Return ``True`` if the named task is currently running."""
    return get_task_status(name) == STATUS_RUNNING


# ---------------------------------------------------------------------------
# Progress callback factory
# ---------------------------------------------------------------------------


def make_progress_callback(task_name: str) -> Callable[..., None]:
    """Return a callback compatible with ``engine.generate_library(progress_callback=…)``.

    The returned callable accepts a single *prog* argument (the
    ``ProgressInfo`` named-tuple emitted by the mutation engine) and writes
    fractional progress + descriptive text into ``session_state``.
    """
    state = st.session_state

    # Phases reported during non-iterative strategies (exhaustive / random).
    _simple_phases = frozenset(
        {
            "generating_variants",
            "scoring_stability",
            "scoring_nativeness",
            "esm2_scoring",
        }
    )

    def _callback(prog: Any) -> None:
        frac = prog.round_number / max(prog.total_rounds, 1)
        state[_key(task_name, "progress")] = min(frac, 1.0)

        if prog.phase in _simple_phases:
            # Show the human-readable message for non-iterative phases.
            state[_key(task_name, "progress_text")] = prog.message or f"Step {prog.round_number}/{prog.total_rounds}"
        else:
            state[_key(task_name, "progress_text")] = (
                f"Phase: {prog.phase} — Round {prog.round_number}/{prog.total_rounds} | "
                f"🧬 {prog.population_size} variants | Best: {prog.best_score:.4f} | "
                f"Anchors: {prog.n_anchors} | Diversity: {prog.diversity_entropy:.2f}"
            )

    return _callback


def set_progress(task_name: str, fraction: float, text: str = "") -> None:
    """Directly set progress for a task (useful for simple loops)."""
    st.session_state[_key(task_name, "progress")] = min(max(fraction, 0.0), 1.0)
    st.session_state[_key(task_name, "progress_text")] = text


# ---------------------------------------------------------------------------
# UI rendering helper
# ---------------------------------------------------------------------------


def render_task_status(
    name: str,
    *,
    poll_interval: float = 3.0,
    success_message: str = "",
    show_progress: bool = True,
) -> Any | None:
    """Render live progress / result UI for the named background task.

    Call this in your Streamlit page function.  It will:

    * Show a progress bar + status text while the task is running via an
      ``@st.fragment(run_every=poll_interval)`` that reruns only the
      progress UI — **not** the entire app script.
    * Trigger a full app rerun (``st.rerun(scope="app")``) once the task
      finishes so that calling code can pick up the result.
    * Show a success message (or error) when the task finishes.
    * Return the task result when status is ``done``, or ``None`` otherwise.
    """
    status = get_task_status(name)

    if status == STATUS_RUNNING:

        @st.fragment(run_every=poll_interval)
        def _poll_fragment() -> None:
            current = get_task_status(name)
            if current == STATUS_RUNNING:
                frac, text = get_task_progress(name)
                if show_progress:
                    st.progress(min(frac, 1.0), text=text or "Working…")
            else:
                # Task completed or errored — trigger a full app rerun so
                # the caller picks up the result on the next execution.
                st.rerun(scope="app")

        _poll_fragment()
        return None

    if status == STATUS_DONE:
        result = get_task_result(name)
        if success_message:
            st.success(success_message)
        return result

    if status == STATUS_ERROR:
        err = get_task_error(name)
        st.error(f"Task failed: {err}")
        return None

    # STATUS_IDLE — nothing to show
    return None


# ---------------------------------------------------------------------------
# Disk-based result persistence
# ---------------------------------------------------------------------------


def save_result_to_disk(
    run_id: str,
    df: pd.DataFrame,
    checkpoint_root: Path | None = None,
) -> Path | None:
    """Save *df* to a Parquet file on disk for crash recovery.

    Parameters
    ----------
    run_id:
        Run identifier (from :func:`vhh_library.checkpoint.compute_run_id`).
    df:
        Final library DataFrame to persist.
    checkpoint_root:
        Base directory for checkpoint / result files.  When ``None``,
        ``tempfile.gettempdir()`` is used.

    Returns
    -------
    Path to the written file, or ``None`` if saving failed.
    """
    from vhh_library.checkpoint import save_result

    if checkpoint_root is None:
        import tempfile

        checkpoint_root = Path(tempfile.gettempdir())
    try:
        return save_result(checkpoint_root, run_id, df)
    except Exception:
        logger.warning("Failed to save result to disk", exc_info=True)
        return None


def load_result_from_disk(
    run_id: str,
    checkpoint_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load a previously saved library DataFrame from disk.

    Returns ``None`` if no result file exists or loading fails.
    """
    from vhh_library.checkpoint import load_result

    if checkpoint_root is None:
        import tempfile

        checkpoint_root = Path(tempfile.gettempdir())
    try:
        return load_result(checkpoint_root, run_id)
    except Exception:
        logger.warning("Failed to load result from disk", exc_info=True)
        return None
