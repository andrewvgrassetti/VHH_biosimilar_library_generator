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
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import streamlit as st

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

    # Capture the session_state reference so the background thread can write to
    # the *same* dict even though ``st.session_state`` is thread-local in
    # Streamlit's runtime.  We grab the underlying dict-like object once here.
    state = st.session_state

    def _worker() -> None:
        try:
            result = func(*args, **kwargs)
            state[_key(name, "result")] = result
            state[_key(name, "status")] = STATUS_DONE
        except Exception as exc:
            state[_key(name, "error")] = str(exc)
            state[_key(name, "status")] = STATUS_ERROR
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

    def _callback(prog: Any) -> None:
        frac = prog.round_number / max(prog.total_rounds, 1)
        state[_key(task_name, "progress")] = min(frac, 1.0)
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

    * Show a progress bar + status text while the task is running, then
      ``time.sleep(poll_interval); st.rerun()`` to keep the UI refreshing.
    * Show a success message (or error) when the task finishes.
    * Return the task result when status is ``done``, or ``None`` otherwise.
    """
    status = get_task_status(name)

    if status == STATUS_RUNNING:
        frac, text = get_task_progress(name)
        if show_progress:
            st.progress(min(frac, 1.0), text=text or "Working…")
        time.sleep(poll_interval)
        st.rerun()

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
