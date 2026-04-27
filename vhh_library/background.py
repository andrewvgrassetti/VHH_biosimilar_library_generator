"""Reusable background-task utility for Streamlit apps.

Runs heavy callables in daemon threads and stores status / progress / result
in a module-level plain Python dict (keyed by Streamlit session ID) so the
UI can poll and display live updates without blocking the Streamlit rerun
cycle.

Thread-safety note
------------------
A module-level ``_shared_state`` dict is used as the communication channel
between background threads and the Streamlit UI.  This bypasses the
``st.session_state`` proxy lifecycle: Streamlit creates a new proxy object on
each script-run / fragment-rerun, so writes from a background thread to a
captured proxy reference may not be visible to subsequent fragment reruns that
acquire a fresh proxy.  A plain Python dict stored in ``_shared_state`` is
always the same object regardless of which thread or rerun context accesses it.

Compound read-modify-write operations (e.g. appending to ``log_entries``) are
protected by ``_shared_state_lock`` to prevent data races between the
background thread and the UI polling fragment.

Disk persistence
----------------
When a background task completes, its result (and status) are persisted to a
temporary file on disk.  This allows recovery after WebSocket disconnects
caused by e.g. laptop sleep/wake cycles.  On the next session init, the app
can call :func:`recover_task` to check for an orphaned result and restore it
into the fresh ``session_state``.
"""

from __future__ import annotations

import datetime
import json
import logging
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_script_run_ctx
except ImportError:  # pragma: no cover — only missing if Streamlit is not installed
    _get_script_run_ctx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level shared state (bypasses st.session_state proxy lifecycle)
# ---------------------------------------------------------------------------

# Maps Streamlit session_id -> per-session task-state dict.
# Background threads write here; fragment reruns read from here.
# Both operations always access the same plain Python dict object,
# regardless of which thread or rerun context they originate from.
#
# Memory note: entries accumulate for the lifetime of the process.  Each
# entry is a small dict (~7 keys) so growth is negligible in practice;
# call ``_cleanup_session_store(session_id)`` after a session ends if you
# need to reclaim memory in long-running deployments.
_shared_state: dict[str, dict[str, Any]] = {}

# Protects compound read-modify-write operations (e.g. appending to
# log_entries) and the initial creation of per-session sub-dicts.
_shared_state_lock = threading.Lock()


def _get_session_store() -> dict[str, Any]:
    """Return the per-session plain-dict store for the current Streamlit session.

    When called from the main script-run context (including ``@st.fragment``
    reruns), this returns a stable plain dict keyed by the Streamlit session
    ID.  Both the background worker (which captures this reference at
    submission time) and the fragment polling loop (which calls this function
    anew on each rerun) will receive the **same** dict object — making
    background-thread writes immediately visible to the fragment.

    Falls back to ``st.session_state`` when no Streamlit script-run context
    is available.  This covers two cases:
    - **Background threads**: no context exists in the worker thread; the
      captured reference passed at submission time is used directly, so the
      fallback is only triggered if this function is called unexpectedly from
      a background thread without a pre-captured store.
    - **Test environments**: no Streamlit runtime is running, so
      ``get_script_run_ctx()`` returns ``None`` and the mock
      ``st.session_state`` (a plain dict) is used instead.  This keeps all
      existing tests working without modification.
    """
    if _get_script_run_ctx is None:
        return st.session_state  # type: ignore[return-value]
    try:
        ctx = _get_script_run_ctx()
        if ctx is None:
            # No active Streamlit script-run context (background thread or
            # test environment) — fall back to st.session_state.
            return st.session_state  # type: ignore[return-value]
        session_id = ctx.session_id
    except Exception:
        return st.session_state  # type: ignore[return-value]

    # Fast path: sub-dict already exists (common case).
    store = _shared_state.get(session_id)
    if store is not None:
        return store

    # Slow path: create sub-dict under the lock to avoid a TOCTOU race.
    with _shared_state_lock:
        store = _shared_state.get(session_id)
        if store is None:
            store = {}
            _shared_state[session_id] = store
    return store


def _cleanup_session_store(session_id: str) -> None:
    """Remove the shared-state entry for *session_id*.

    Call this when a Streamlit session ends to reclaim memory in long-running
    deployments.  Safe to call even if the session_id is not present.
    """
    with _shared_state_lock:
        _shared_state.pop(session_id, None)


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


def _persist_result(
    task_name: str,
    *,
    status: str,
    result: Any = None,
    error: str | None = None,
    error_traceback: str | None = None,
) -> None:
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
        "error_traceback": error_traceback,
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
    store = _get_session_store()
    status = store.get(_key(name, "status"), STATUS_IDLE)
    if status == STATUS_RUNNING:
        logger.warning("Task %r is already running — ignoring duplicate submission.", name)
        return False

    # Initialise state keys in the shared store.
    store[_key(name, "status")] = STATUS_RUNNING
    store[_key(name, "progress")] = 0.0
    store[_key(name, "progress_text")] = ""
    store[_key(name, "log_entries")] = []
    store[_key(name, "result")] = None
    store[_key(name, "error")] = None
    store[_key(name, "error_traceback")] = None

    # Clear any stale persisted result from a previous run.
    _clear_persisted(name)

    # Capture the session store reference (a plain Python dict, NOT the
    # st.session_state proxy).  The background thread writes directly to
    # this dict; fragment reruns read from the same dict via _get_session_store().
    captured_store = store

    def _worker() -> None:
        print(f"[BG-THREAD] Background thread started for task {name!r}", flush=True)
        try:
            result = func(*args, **kwargs)
            print(f"[BG-THREAD] Task {name!r} completed successfully", flush=True)
            # Write to the shared store captured at submission time.
            try:
                captured_store[_key(name, "result")] = result
                captured_store[_key(name, "status")] = STATUS_DONE
            except Exception:
                logger.warning(
                    "Could not write result to shared store for task %r.",
                    name,
                    exc_info=True,
                )
            # Always persist to disk so the result can be recovered.
            _persist_result(name, status=STATUS_DONE, result=result)
        except Exception as exc:
            print(f"[BG-THREAD] Task {name!r} FAILED: {exc}", flush=True)
            traceback.print_exc()
            error_msg = str(exc)
            tb_str = traceback.format_exc()
            try:
                captured_store[_key(name, "error")] = error_msg
                captured_store[_key(name, "error_traceback")] = tb_str
                captured_store[_key(name, "status")] = STATUS_ERROR
            except Exception:
                logger.warning("Could not write error to shared store for task %r.", name, exc_info=True)
            _persist_result(name, status=STATUS_ERROR, error=error_msg, error_traceback=tb_str)
            logger.exception("Background task %r failed", name)

    thread = threading.Thread(target=_worker, daemon=True, name=f"bg-{name}")
    thread.start()
    return True


def get_task_status(name: str) -> str:
    """Return the current status string for *name*."""
    return _get_session_store().get(_key(name, "status"), STATUS_IDLE)


def get_task_progress(name: str) -> tuple[float, str]:
    """Return ``(fraction, text)`` for the running task."""
    store = _get_session_store()
    frac = store.get(_key(name, "progress"), 0.0)
    text = store.get(_key(name, "progress_text"), "")
    return frac, text


def get_task_result(name: str) -> Any:
    """Return the result stored by a completed task (or ``None``)."""
    return _get_session_store().get(_key(name, "result"))


def get_task_error(name: str) -> str | None:
    """Return the error message if the task failed."""
    return _get_session_store().get(_key(name, "error"))


def get_task_traceback(name: str) -> str | None:
    """Return the full traceback string if the task failed, or ``None``."""
    return _get_session_store().get(_key(name, "error_traceback"))


def get_task_log(name: str) -> list[tuple[float, str]]:
    """Return the accumulated log entries for *name*.

    Each entry is a ``(timestamp, message)`` tuple.  Returns an empty list
    when no entries exist.
    """
    return list(_get_session_store().get(_key(name, "log_entries"), []))


def append_log_entry(task_name: str, message: str, *, _state: Any = None) -> None:
    """Append a timestamped message to the task's activity log.

    Parameters
    ----------
    task_name:
        Background task name.
    message:
        Human-readable log line.
    _state:
        Optional dict-like override (used by background threads which capture
        the session store reference at submit time).
    """
    state = _state if _state is not None else _get_session_store()
    key = _key(task_name, "log_entries")
    with _shared_state_lock:
        entries = state.get(key)
        if entries is None:
            entries = []
            state[key] = entries
        entries.append((time.time(), message))


def reset_task(name: str) -> None:
    """Reset all state keys for *name* back to idle."""
    _idle_state = {
        _key(name, "status"): STATUS_IDLE,
        _key(name, "progress"): 0.0,
        _key(name, "progress_text"): "",
        _key(name, "log_entries"): [],
        _key(name, "result"): None,
        _key(name, "error"): None,
        _key(name, "error_traceback"): None,
    }
    # Clear from the shared store (the real inter-thread channel).
    _get_session_store().update(_idle_state)
    # Also clear from st.session_state for backward compatibility with
    # any code that reads directly from the Streamlit session (e.g. recover_task).
    try:
        for k, v in _idle_state.items():
            st.session_state[k] = v
    except Exception:
        pass
    _clear_persisted(name)


def recover_task(name: str) -> Any | None:
    """Attempt to recover a completed background-task result from disk.

    If a persisted result file exists for *name* and is not stale, restore it
    into the shared session store (and ``st.session_state`` for backward
    compatibility) and return the result (or the error string for failed
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
        _done_state = {
            _key(name, "status"): STATUS_DONE,
            _key(name, "result"): result,
            _key(name, "progress"): 1.0,
            _key(name, "progress_text"): "",
            _key(name, "log_entries"): [],
            _key(name, "error"): None,
            _key(name, "error_traceback"): None,
        }
        # Write to the shared store so fragment reruns can see the result.
        _get_session_store().update(_done_state)
        # Also write to st.session_state for backward compatibility with
        # app code that reads directly from Streamlit session state.
        for k, v in _done_state.items():
            st.session_state[k] = v
        logger.info("Recovered completed result for background task %r from disk.", name)
        return result

    if persisted_status == STATUS_ERROR:
        error = payload.get("error", "Unknown error")
        error_tb = payload.get("error_traceback")
        _error_state = {
            _key(name, "status"): STATUS_ERROR,
            _key(name, "error"): error,
            _key(name, "error_traceback"): error_tb,
            _key(name, "result"): None,
            _key(name, "progress"): 0.0,
            _key(name, "progress_text"): "",
            _key(name, "log_entries"): [],
        }
        # Write to the shared store so fragment reruns can see the error.
        _get_session_store().update(_error_state)
        # Also write to st.session_state for backward compatibility.
        for k, v in _error_state.items():
            st.session_state[k] = v
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
    fractional progress + descriptive text into the session store.

    Each callback invocation also appends a timestamped entry to the task's
    activity log so the UI can display a scrollable history of every step.
    """
    # Capture the session store reference (a plain dict) at creation time so
    # the returned callback is safe to call from background threads.
    state = _get_session_store()

    # Phases reported during non-iterative strategies (exhaustive / random)
    # and sub-step phases reported from within _batch_fill_* methods.
    _simple_phases = frozenset(
        {
            "initializing",
            "generating_variants",
            "sampling_variants",
            "scoring_stability",
            "scoring_nativeness",
            "esm2_scoring",
            "scoring_stability_start",
            "scoring_stability_done",
            "scoring_stability_progress",
            "scoring_nativeness_start",
            "scoring_nativeness_progress",
            "scoring_nativeness_done",
            "esm2_scoring_stage1",
            "esm2_scoring_stage2",
        }
    )

    _first_call = True

    def _callback(prog: Any) -> None:
        nonlocal _first_call
        if _first_call:
            print(
                f"[PROGRESS] First progress callback for {task_name!r}: phase={prog.phase}",
                flush=True,
            )
            _first_call = False

        frac = prog.round_number / max(prog.total_rounds, 1)
        state[_key(task_name, "progress")] = min(frac, 1.0)

        if prog.phase in _simple_phases:
            # Show the human-readable message for non-iterative phases.
            text = prog.message or f"Step {prog.round_number}/{prog.total_rounds}"
            state[_key(task_name, "progress_text")] = text
        else:
            text = (
                f"Phase: {prog.phase} — Round {prog.round_number}/{prog.total_rounds} | "
                f"🧬 {prog.population_size} variants | Best: {prog.best_score:.4f} | "
                f"Anchors: {prog.n_anchors} | Diversity: {prog.diversity_entropy:.2f}"
            )
            state[_key(task_name, "progress_text")] = text

        # Append to the activity log so the UI can show a scrollable history.
        append_log_entry(task_name, text, _state=state)

    return _callback


def set_progress(task_name: str, fraction: float, text: str = "", *, _state: dict[str, Any] | None = None) -> None:
    """Directly set progress for a task (useful for simple loops).

    Parameters
    ----------
    _state:
        Optional dict-like override for the session store.  When called from
        a background thread, pass the store reference captured at submit time.
    """
    state = _state if _state is not None else _get_session_store()
    state[_key(task_name, "progress")] = min(max(fraction, 0.0), 1.0)
    state[_key(task_name, "progress_text")] = text


def make_progress_setter(task_name: str) -> Callable[[float, str], None]:
    """Return a thread-safe progress setter for *task_name*.

    Unlike :func:`set_progress` which accesses the session store at call time,
    the returned callable captures the session store reference at creation time
    and is safe to call from daemon threads launched by :func:`submit_task`.

    Usage::

        _set_progress = make_progress_setter("my_task")
        def _worker():
            _set_progress(0.5, "Half done…")
        submit_task("my_task", _worker)
    """
    # Capture the session store reference (plain dict) at creation time.
    state = _get_session_store()
    progress_key = _key(task_name, "progress")
    text_key = _key(task_name, "progress_text")

    def _setter(fraction: float, text: str = "") -> None:
        state[progress_key] = min(max(fraction, 0.0), 1.0)
        state[text_key] = text

    return _setter


# ---------------------------------------------------------------------------
# UI rendering helper
# ---------------------------------------------------------------------------

# Maximum number of log entries displayed in the activity log widget to
# keep the DOM from growing unboundedly during very long runs.
_ACTIVITY_LOG_MAX_LINES = 200


def _render_activity_log(name: str, *, max_lines: int = _ACTIVITY_LOG_MAX_LINES) -> None:
    """Render a scrollable activity-log container for a running task.

    Shows the most recent *max_lines* log entries inside a styled
    container so the user can watch processing steps stream in without
    the page jumping around.
    """
    entries = get_task_log(name)
    if not entries:
        return

    # Limit to the most recent entries.
    display = entries[-max_lines:]

    lines: list[str] = []
    for ts, msg in display:
        dt = datetime.datetime.fromtimestamp(ts)
        stamp = dt.strftime("%H:%M:%S")
        lines.append(f"[{stamp}]  {msg}")

    log_text = "\n".join(lines)

    with st.expander("📋 Activity Log", expanded=True):
        st.code(log_text, language=None)


def render_task_status(
    name: str,
    *,
    poll_interval: float = 3.0,
    success_message: str = "",
    show_progress: bool = True,
    show_activity_log: bool = True,
) -> Any | None:
    """Render live progress / result UI for the named background task.

    Call this in your Streamlit page function.  It will:

    * Show a progress bar + status text while the task is running via an
      ``@st.fragment(run_every=poll_interval)`` that reruns only the
      progress UI — **not** the entire app script.
    * Optionally show a scrollable activity log (``show_activity_log``)
      beneath the progress bar so the user can see every processing step
      as it happens.
    * Trigger a full app rerun (``st.rerun(scope="app")``) once the task
      finishes so that calling code can pick up the result.
    * Show a success message (or error) when the task finishes.
    * Return the task result when status is ``done``, or ``None`` otherwise.
    """
    status = get_task_status(name)
    print(f"[UI-POLL] render_task_status({name!r}): status={status}", flush=True)

    if status == STATUS_RUNNING:

        @st.fragment(run_every=poll_interval)
        def _poll_fragment() -> None:
            current = get_task_status(name)
            if current == STATUS_RUNNING:
                frac, text = get_task_progress(name)
                if show_progress:
                    st.progress(min(frac, 1.0), text=text or "Working…")
                if show_activity_log:
                    _render_activity_log(name)
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
        tb = get_task_traceback(name)
        if tb:
            with st.expander("🔍 Full error traceback", expanded=False):
                st.code(tb, language="python")
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
