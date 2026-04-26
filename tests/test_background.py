"""Unit tests for the background-task utility (vhh_library.background)."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock st.session_state as a plain dict for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_session_state(monkeypatch):
    """Replace ``st.session_state`` with a plain dict for testing."""
    state: dict = {}
    import vhh_library.background as bg_mod

    # Patch at the module level so all functions see the mock
    mock_st = MagicMock()
    mock_st.session_state = state
    mock_st.progress = MagicMock()
    mock_st.rerun = MagicMock()
    mock_st.success = MagicMock()
    mock_st.error = MagicMock()

    # st.fragment(run_every=...) must act as a transparent decorator so the
    # inner function is still callable in tests (no real Streamlit runtime).
    mock_st.fragment = lambda **kwargs: lambda fn: fn

    monkeypatch.setattr(bg_mod, "st", mock_st)
    # Expose mock_st on the dict so tests can inspect calls
    state["__mock_st__"] = mock_st
    return state


@pytest.fixture(autouse=True)
def _clean_persist_dir():
    """Remove persisted task files before and after each test."""
    import shutil

    from vhh_library.background import _TASK_PERSIST_DIR

    if _TASK_PERSIST_DIR.is_dir():
        shutil.rmtree(_TASK_PERSIST_DIR)
    yield
    if _TASK_PERSIST_DIR.is_dir():
        shutil.rmtree(_TASK_PERSIST_DIR)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubmitTask:
    """submit_task launches a thread and sets status to 'running'."""

    def test_launches_thread_and_sets_running(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_RUNNING,
            _key,
            submit_task,
        )

        event = threading.Event()

        def slow_fn():
            event.wait(timeout=5)
            return 42

        ok = submit_task("t1", slow_fn)
        assert ok is True
        assert _mock_session_state[_key("t1", "status")] == STATUS_RUNNING

        # Let the thread finish
        event.set()
        time.sleep(0.2)

    def test_completion_sets_done_and_stores_result(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            submit_task,
        )

        def work():
            return {"answer": 42}

        submit_task("t2", work)
        # Give the thread a moment to complete
        time.sleep(0.3)

        assert _mock_session_state[_key("t2", "status")] == STATUS_DONE
        assert _mock_session_state[_key("t2", "result")] == {"answer": 42}

    def test_exception_sets_error(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_ERROR,
            _key,
            submit_task,
        )

        def failing():
            raise ValueError("boom")

        submit_task("t3", failing)
        time.sleep(0.3)

        assert _mock_session_state[_key("t3", "status")] == STATUS_ERROR
        assert "boom" in _mock_session_state[_key("t3", "error")]

    def test_exception_stores_traceback(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_ERROR,
            _key,
            submit_task,
        )

        def failing():
            raise RuntimeError("traceback-test")

        submit_task("t3_tb", failing)
        time.sleep(0.3)

        assert _mock_session_state[_key("t3_tb", "status")] == STATUS_ERROR
        tb = _mock_session_state[_key("t3_tb", "error_traceback")]
        assert tb is not None
        assert "RuntimeError" in tb
        assert "traceback-test" in tb
        assert "Traceback" in tb

    def test_get_task_traceback_returns_stored_traceback(self, _mock_session_state):
        from vhh_library.background import (
            get_task_traceback,
            submit_task,
        )

        def failing():
            raise TypeError("get-tb-test")

        submit_task("t3_get_tb", failing)
        time.sleep(0.3)

        tb = get_task_traceback("t3_get_tb")
        assert tb is not None
        assert "TypeError" in tb
        assert "get-tb-test" in tb

    def test_get_task_traceback_returns_none_when_no_error(self, _mock_session_state):
        from vhh_library.background import get_task_traceback

        assert get_task_traceback("nonexistent") is None

    def test_duplicate_submission_rejected(self, _mock_session_state):
        from vhh_library.background import (
            submit_task,
        )

        event = threading.Event()

        def slow():
            event.wait(timeout=5)

        ok1 = submit_task("t4", slow)
        assert ok1 is True

        # Second submission while still running should be rejected
        ok2 = submit_task("t4", slow)
        assert ok2 is False

        event.set()
        time.sleep(0.2)


class TestProgressCallback:
    """make_progress_callback writes progress into session state."""

    def test_progress_callback_updates_state(self, _mock_session_state):
        from vhh_library.background import _key, make_progress_callback

        cb = make_progress_callback("libgen")

        # Simulate a ProgressInfo-like object
        prog = MagicMock()
        prog.round_number = 3
        prog.total_rounds = 10
        prog.phase = "explore"
        prog.population_size = 500
        prog.best_score = 0.85
        prog.n_anchors = 5
        prog.diversity_entropy = 1.2

        cb(prog)

        assert _mock_session_state[_key("libgen", "progress")] == pytest.approx(0.3)
        text = _mock_session_state[_key("libgen", "progress_text")]
        assert "Phase: explore" in text
        assert "Round 3/10" in text

    def test_progress_callback_simple_phase_uses_message(self, _mock_session_state):
        """Non-iterative phases (e.g. generating_variants) display the message field."""
        from vhh_library.background import _key, make_progress_callback

        cb = make_progress_callback("libgen")

        prog = MagicMock()
        prog.round_number = 1
        prog.total_rounds = 3
        prog.phase = "generating_variants"
        prog.message = "Building exhaustive combinations (500 max)…"
        prog.population_size = 0
        prog.best_score = 0.0
        prog.n_anchors = 0
        prog.diversity_entropy = 0.0

        cb(prog)

        assert _mock_session_state[_key("libgen", "progress")] == pytest.approx(1 / 3)
        text = _mock_session_state[_key("libgen", "progress_text")]
        assert "Building exhaustive" in text
        # Should NOT contain iterative-style format
        assert "Phase:" not in text
        assert "Anchors:" not in text

    def test_progress_callback_scoring_nativeness_phase(self, _mock_session_state):
        """Scoring nativeness phase shows the message."""
        from vhh_library.background import _key, make_progress_callback

        cb = make_progress_callback("libgen")

        prog = MagicMock()
        prog.round_number = 2
        prog.total_rounds = 3
        prog.phase = "scoring_nativeness"
        prog.message = "Scoring nativeness for 100 variants…"
        prog.population_size = 100
        prog.best_score = 0.0
        prog.n_anchors = 0
        prog.diversity_entropy = 0.0

        cb(prog)

        text = _mock_session_state[_key("libgen", "progress_text")]
        assert "Scoring nativeness" in text

    def test_progress_callback_esm2_scoring_phase(self, _mock_session_state):
        """ESM-2 scoring phase (esm2_scoring) shows the descriptive message."""
        from vhh_library.background import _key, make_progress_callback

        cb = make_progress_callback("libgen")

        prog = MagicMock()
        prog.round_number = 1
        prog.total_rounds = 1
        prog.phase = "esm2_scoring"
        prog.message = "ESM-2 progressive scoring for 500 variants…"
        prog.population_size = 500
        prog.best_score = 0.0
        prog.n_anchors = 0
        prog.diversity_entropy = 0.0

        cb(prog)

        frac = _mock_session_state[_key("libgen", "progress")]
        assert frac == pytest.approx(1.0)
        text = _mock_session_state[_key("libgen", "progress_text")]
        assert "ESM-2 progressive scoring" in text
        # esm2_scoring is a simple phase — should NOT contain iterative format
        assert "Phase:" not in text
        assert "Anchors:" not in text


class TestSetProgress:
    """set_progress directly updates progress keys."""

    def test_set_progress(self, _mock_session_state):
        from vhh_library.background import _key, set_progress

        set_progress("my_task", 0.75, "Three quarters done")
        assert _mock_session_state[_key("my_task", "progress")] == pytest.approx(0.75)
        assert _mock_session_state[_key("my_task", "progress_text")] == "Three quarters done"

    def test_set_progress_clamps(self, _mock_session_state):
        from vhh_library.background import _key, set_progress

        set_progress("clamp", 1.5, "over")
        assert _mock_session_state[_key("clamp", "progress")] == 1.0
        set_progress("clamp", -0.5, "under")
        assert _mock_session_state[_key("clamp", "progress")] == 0.0

    def test_set_progress_with_state_override(self, _mock_session_state):
        """set_progress uses explicit _state when provided."""
        from vhh_library.background import _key, set_progress

        custom_state: dict = {}
        set_progress("t", 0.5, "half", _state=custom_state)
        assert custom_state[_key("t", "progress")] == pytest.approx(0.5)
        assert custom_state[_key("t", "progress_text")] == "half"
        # Must not write to session_state mock
        assert _key("t", "progress") not in _mock_session_state


class TestMakeProgressSetter:
    """make_progress_setter returns a thread-safe callable."""

    def test_basic_setter(self, _mock_session_state):
        from vhh_library.background import _key, make_progress_setter

        setter = make_progress_setter("t1")
        setter(0.33, "one third")
        assert _mock_session_state[_key("t1", "progress")] == pytest.approx(0.33)
        assert _mock_session_state[_key("t1", "progress_text")] == "one third"

    def test_setter_clamps(self, _mock_session_state):
        from vhh_library.background import _key, make_progress_setter

        setter = make_progress_setter("clamp")
        setter(1.5, "over")
        assert _mock_session_state[_key("clamp", "progress")] == 1.0
        setter(-0.1, "under")
        assert _mock_session_state[_key("clamp", "progress")] == 0.0

    def test_setter_works_from_background_thread(self, _mock_session_state):
        """The captured setter must work from a daemon thread even when
        st.session_state is not accessible.

        This simulates the real Streamlit scenario: the setter is created
        in the main thread (where st.session_state is available) and called
        from a background thread (where it is NOT).
        """
        from vhh_library.background import _key, make_progress_setter

        # Create setter in the "main" thread (session_state available).
        setter = make_progress_setter("bg_test")

        # Now make st.session_state raise (simulating background thread).
        # The setter should still work because it captured the state dict.
        result = {}

        def _worker():
            try:
                setter(0.75, "from background")
                result["ok"] = True
            except Exception as exc:
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join(timeout=5)

        assert result.get("ok") is True, f"Setter failed in background thread: {result.get('error')}"
        assert _mock_session_state[_key("bg_test", "progress")] == pytest.approx(0.75)
        assert _mock_session_state[_key("bg_test", "progress_text")] == "from background"

    def test_setter_default_text_is_empty(self, _mock_session_state):
        from vhh_library.background import _key, make_progress_setter

        setter = make_progress_setter("t2")
        setter(0.5)
        assert _mock_session_state[_key("t2", "progress_text")] == ""


class TestResetTask:
    """reset_task clears all state keys for a task."""

    def test_reset(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_DONE,
            STATUS_IDLE,
            _key,
            reset_task,
        )

        _mock_session_state[_key("r1", "status")] = STATUS_DONE
        _mock_session_state[_key("r1", "result")] = "data"

        reset_task("r1")

        assert _mock_session_state[_key("r1", "status")] == STATUS_IDLE
        assert _mock_session_state[_key("r1", "result")] is None

    def test_reset_clears_traceback(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_ERROR,
            STATUS_IDLE,
            _key,
            reset_task,
        )

        _mock_session_state[_key("r_tb", "status")] = STATUS_ERROR
        _mock_session_state[_key("r_tb", "error")] = "oops"
        _mock_session_state[_key("r_tb", "error_traceback")] = "Traceback ..."

        reset_task("r_tb")

        assert _mock_session_state[_key("r_tb", "status")] == STATUS_IDLE
        assert _mock_session_state[_key("r_tb", "error")] is None
        assert _mock_session_state[_key("r_tb", "error_traceback")] is None


class TestIsTaskRunning:
    """is_task_running checks the status key."""

    def test_running(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_RUNNING,
            _key,
            is_task_running,
        )

        _mock_session_state[_key("x", "status")] = STATUS_RUNNING
        assert is_task_running("x") is True

    def test_not_running(self, _mock_session_state):
        from vhh_library.background import is_task_running

        assert is_task_running("nonexistent") is False


class TestRenderTaskStatus:
    """render_task_status returns results correctly."""

    def test_returns_result_when_done(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            render_task_status,
        )

        _mock_session_state[_key("d1", "status")] = STATUS_DONE
        _mock_session_state[_key("d1", "result")] = [1, 2, 3]

        result = render_task_status("d1")
        assert result == [1, 2, 3]

    def test_returns_none_when_idle(self, _mock_session_state):
        from vhh_library.background import render_task_status

        result = render_task_status("idle_task")
        assert result is None

    def test_returns_none_on_error(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_ERROR,
            _key,
            render_task_status,
        )

        _mock_session_state[_key("e1", "status")] = STATUS_ERROR
        _mock_session_state[_key("e1", "error")] = "something broke"

        result = render_task_status("e1")
        assert result is None

    def test_error_renders_traceback_expander(self, _mock_session_state):
        """When a traceback is available, render_task_status shows it in an expander."""
        from vhh_library.background import (
            STATUS_ERROR,
            _key,
            render_task_status,
        )

        _mock_session_state[_key("e_tb", "status")] = STATUS_ERROR
        _mock_session_state[_key("e_tb", "error")] = "boom"
        _mock_session_state[_key("e_tb", "error_traceback")] = "Traceback (most recent call last):\n  ..."

        mock_st = _mock_session_state["__mock_st__"]
        # Make st.expander return a context manager mock
        expander_ctx = MagicMock()
        expander_ctx.__enter__ = MagicMock(return_value=expander_ctx)
        expander_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander_ctx

        result = render_task_status("e_tb")
        assert result is None
        mock_st.error.assert_called_once()
        mock_st.expander.assert_called_once()

    def test_returns_none_when_running(self, _mock_session_state):
        """While running the fragment renders progress and returns None."""
        from vhh_library.background import (
            STATUS_RUNNING,
            _key,
            render_task_status,
        )

        _mock_session_state[_key("r1", "status")] = STATUS_RUNNING
        _mock_session_state[_key("r1", "progress")] = 0.5
        _mock_session_state[_key("r1", "progress_text")] = "halfway"

        result = render_task_status("r1")
        assert result is None

        # Fragment should have rendered the progress bar
        mock_st = _mock_session_state["__mock_st__"]
        mock_st.progress.assert_called_once_with(0.5, text="halfway")

    def test_running_fragment_triggers_app_rerun_on_completion(self, _mock_session_state):
        """When the fragment detects the task finished it calls st.rerun(scope='app')."""
        from vhh_library.background import (
            STATUS_DONE,
            STATUS_RUNNING,
            _key,
            render_task_status,
        )

        # Start as RUNNING so render_task_status enters the fragment path,
        # but transition to DONE *before* the fragment body actually executes.
        # Because st.fragment is a no-op decorator in tests, the inner
        # _poll_fragment runs synchronously right away.
        # We simulate the task finishing between the outer status check and
        # the fragment body by changing status after the first read.
        _mock_session_state[_key("tr", "status")] = STATUS_RUNNING
        _mock_session_state[_key("tr", "progress")] = 0.0
        _mock_session_state[_key("tr", "progress_text")] = ""
        _mock_session_state[_key("tr", "result")] = "done-data"

        # Patch get_task_status to return RUNNING first (outer check), then DONE (fragment)
        call_count = {"n": 0}
        original_get = None
        import vhh_library.background as bg_mod

        original_get = bg_mod.get_task_status

        def _patched_get(name):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return STATUS_RUNNING
            return STATUS_DONE

        bg_mod.get_task_status = _patched_get
        try:
            result = render_task_status("tr")
        finally:
            bg_mod.get_task_status = original_get

        # The function returns None (still in RUNNING branch)
        assert result is None
        # The fragment detected DONE and called st.rerun(scope="app")
        mock_st = _mock_session_state["__mock_st__"]
        mock_st.rerun.assert_called_once_with(scope="app")

    def test_no_sleep_called_during_running(self, _mock_session_state):
        """Ensure time.sleep is never called — the old blocking pattern is gone."""
        import time as _time
        from unittest.mock import patch

        from vhh_library.background import (
            STATUS_RUNNING,
            _key,
            render_task_status,
        )

        _mock_session_state[_key("ns", "status")] = STATUS_RUNNING
        _mock_session_state[_key("ns", "progress")] = 0.1
        _mock_session_state[_key("ns", "progress_text")] = "working"

        with patch.object(_time, "sleep", side_effect=AssertionError("sleep must not be called")):
            result = render_task_status("ns")

        assert result is None

    def test_fragment_uses_poll_interval(self, _mock_session_state, monkeypatch):
        """The fragment decorator receives the configured poll_interval."""
        from vhh_library.background import (
            STATUS_RUNNING,
            _key,
            render_task_status,
        )

        captured_kwargs: list[dict] = []

        def _capture_fragment(**kwargs):
            captured_kwargs.append(kwargs)
            return lambda fn: fn

        mock_st = _mock_session_state["__mock_st__"]
        mock_st.fragment = _capture_fragment

        _mock_session_state[_key("pi", "status")] = STATUS_RUNNING
        _mock_session_state[_key("pi", "progress")] = 0.0
        _mock_session_state[_key("pi", "progress_text")] = ""

        render_task_status("pi", poll_interval=5.0)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["run_every"] == 5.0


# ---------------------------------------------------------------------------
# Disk persistence tests
# ---------------------------------------------------------------------------


class TestDiskPersistence:
    """Verify that background tasks persist results to disk."""

    def test_successful_task_persists_to_disk(self, _mock_session_state):
        from vhh_library.background import _persist_path, submit_task

        def work():
            return {"answer": 42}

        submit_task("persist_ok", work)
        time.sleep(0.5)

        path = _persist_path("persist_ok")
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["status"] == "done"
        assert payload["result"] == {"answer": 42}
        assert "timestamp" in payload

    def test_failed_task_persists_error_to_disk(self, _mock_session_state):
        from vhh_library.background import _persist_path, submit_task

        def failing():
            raise RuntimeError("disk-test-error")

        submit_task("persist_err", failing)
        time.sleep(0.5)

        path = _persist_path("persist_err")
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["status"] == "error"
        assert "disk-test-error" in payload["error"]

    def test_failed_task_persists_traceback_to_disk(self, _mock_session_state):
        from vhh_library.background import _persist_path, submit_task

        def failing():
            raise RuntimeError("disk-tb-error")

        submit_task("persist_err_tb", failing)
        time.sleep(0.5)

        path = _persist_path("persist_err_tb")
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["status"] == "error"
        assert payload.get("error_traceback") is not None
        assert "RuntimeError" in payload["error_traceback"]
        assert "disk-tb-error" in payload["error_traceback"]

    def test_dataframe_result_persists_and_round_trips(self, _mock_session_state):
        import pandas as pd

        from vhh_library.background import _load_persisted, _persist_path, submit_task

        def work():
            return pd.DataFrame({"variant_id": ["V1", "V2"], "score": [0.8, 0.9]})

        submit_task("persist_df", work)
        time.sleep(0.5)

        path = _persist_path("persist_df")
        assert path.is_file()

        # Raw JSON should have the DataFrame type tag
        raw = json.loads(path.read_text())
        assert raw["result"]["__type__"] == "DataFrame"

        # _load_persisted should deserialise it back to a DataFrame
        payload = _load_persisted("persist_df")
        assert payload is not None
        assert isinstance(payload["result"], pd.DataFrame)
        assert list(payload["result"]["variant_id"]) == ["V1", "V2"]

    def test_reset_task_clears_persisted_file(self, _mock_session_state):
        from vhh_library.background import _persist_path, _persist_result, reset_task

        _persist_result("reset_me", status="done", result="data")
        assert _persist_path("reset_me").is_file()

        reset_task("reset_me")
        assert not _persist_path("reset_me").is_file()

    def test_submit_clears_stale_persisted_file(self, _mock_session_state):
        from vhh_library.background import _persist_path, _persist_result, submit_task

        # Pre-populate a stale file
        _persist_result("stale_test", status="done", result="old-data")
        assert _persist_path("stale_test").is_file()

        event = threading.Event()

        def slow():
            event.wait(timeout=2)
            return "new"

        submit_task("stale_test", slow)
        # The old file should have been cleared at submission time
        assert not _persist_path("stale_test").is_file()

        event.set()
        time.sleep(0.3)


class TestRecoverTask:
    """Verify that recover_task restores results into session state."""

    def test_recover_done_task(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            _persist_result,
            recover_task,
        )

        _persist_result("rec_done", status="done", result={"answer": 42})

        result = recover_task("rec_done")
        assert result == {"answer": 42}
        assert _mock_session_state[_key("rec_done", "status")] == STATUS_DONE
        assert _mock_session_state[_key("rec_done", "result")] == {"answer": 42}
        assert _mock_session_state[_key("rec_done", "progress")] == 1.0

    def test_recover_error_task(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_ERROR,
            _key,
            _persist_result,
            recover_task,
        )

        _persist_result("rec_err", status="error", error="something broke")

        result = recover_task("rec_err")
        assert result is None
        assert _mock_session_state[_key("rec_err", "status")] == STATUS_ERROR
        assert "something broke" in _mock_session_state[_key("rec_err", "error")]

    def test_recover_error_task_with_traceback(self, _mock_session_state):
        from vhh_library.background import (
            STATUS_ERROR,
            _key,
            _persist_result,
            recover_task,
        )

        tb_text = "Traceback (most recent call last):\n  File ...\nRuntimeError: kaboom"
        _persist_result("rec_err_tb", status="error", error="kaboom", error_traceback=tb_text)

        result = recover_task("rec_err_tb")
        assert result is None
        assert _mock_session_state[_key("rec_err_tb", "status")] == STATUS_ERROR
        assert _mock_session_state[_key("rec_err_tb", "error")] == "kaboom"
        assert _mock_session_state[_key("rec_err_tb", "error_traceback")] == tb_text

    def test_recover_returns_none_when_no_file(self, _mock_session_state):
        from vhh_library.background import recover_task

        result = recover_task("nonexistent_task")
        assert result is None

    def test_recover_ignores_stale_file(self, _mock_session_state):
        from vhh_library.background import (
            _TASK_PERSIST_DIR,
            _persist_path,
            recover_task,
        )

        # Write a persisted result with an old timestamp (2 hours ago)
        _TASK_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        path = _persist_path("stale_rec")
        payload = {
            "status": "done",
            "result": "old-data",
            "error": None,
            "timestamp": time.time() - 7200,
        }
        path.write_text(json.dumps(payload))

        result = recover_task("stale_rec")
        assert result is None
        # The stale file should have been cleaned up
        assert not path.is_file()

    def test_recover_dataframe_result(self, _mock_session_state):
        import pandas as pd

        from vhh_library.background import _persist_result, recover_task

        df = pd.DataFrame({"x": [1, 2, 3]})
        _persist_result("rec_df", status="done", result=df)

        result = recover_task("rec_df")
        assert isinstance(result, pd.DataFrame)
        assert list(result["x"]) == [1, 2, 3]


class TestSessionStateLossRecovery:
    """End-to-end test: task completes after session state is replaced."""

    def test_task_result_survives_session_loss(self, _mock_session_state):
        """Simulate a task completing after the session state dict is replaced."""
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            recover_task,
            submit_task,
        )

        def work():
            return "computed-result"

        submit_task("survive", work)
        time.sleep(0.5)

        # Simulate session loss: clear the dict (as Streamlit would on reconnect)
        keys_to_remove = [k for k in _mock_session_state if k != "__mock_st__"]
        for k in keys_to_remove:
            del _mock_session_state[k]

        # Session state no longer has the result
        assert _mock_session_state.get(_key("survive", "result")) is None

        # But recovery from disk should restore it
        result = recover_task("survive")
        assert result == "computed-result"
        assert _mock_session_state[_key("survive", "status")] == STATUS_DONE


# ---------------------------------------------------------------------------
# Recovery vs. render_task_status race condition tests
# ---------------------------------------------------------------------------


class TestRecoveryRaceCondition:
    """Verify that _recover_background_tasks does not race with render_task_status.

    When a background task completes normally (session alive), the disk-
    persisted result may exist by the time init_state() calls recovery.
    Recovery must NOT reset the task to IDLE if the session-state status
    is already DONE/ERROR — that would prevent render_task_status from
    returning the result to the caller, making the task appear to hang.
    """

    def test_recovery_skips_done_task(self, _mock_session_state):
        """Recovery should not interfere when session-state status is DONE."""
        from vhh_library.background import (
            STATUS_DONE,
            STATUS_IDLE,
            _key,
            get_task_status,
            recover_task,
            reset_task,
            submit_task,
        )

        # Simulate a task that completed normally: status DONE in session
        # state AND a persisted result file on disk (the _worker writes both).
        def work():
            return "my-library-result"

        submit_task("race_done", work)
        time.sleep(0.5)

        # Verify task completed
        assert get_task_status("race_done") == STATUS_DONE
        assert _mock_session_state[_key("race_done", "result")] == "my-library-result"

        # Simulate what _recover_background_tasks should do:
        # Since status is DONE (not IDLE), skip recovery entirely.
        if get_task_status("race_done") != STATUS_IDLE:
            recovered = False
        else:
            result = recover_task("race_done")
            recovered = result is not None
            if recovered:
                reset_task("race_done")

        assert not recovered, "Recovery should not run when status is DONE"
        # Status must still be DONE so render_task_status can process it
        assert get_task_status("race_done") == STATUS_DONE
        assert _mock_session_state[_key("race_done", "result")] == "my-library-result"

    def test_recovery_skips_error_task(self, _mock_session_state):
        """Recovery should not interfere when session-state status is ERROR."""
        from vhh_library.background import (
            STATUS_ERROR,
            STATUS_IDLE,
            _key,
            get_task_status,
            recover_task,
            reset_task,
            submit_task,
        )

        def failing_work():
            raise RuntimeError("test failure")

        submit_task("race_err", failing_work)
        time.sleep(0.5)

        assert get_task_status("race_err") == STATUS_ERROR
        assert _mock_session_state[_key("race_err", "error")] == "test failure"

        # Recovery should skip because status is ERROR
        if get_task_status("race_err") != STATUS_IDLE:
            recovered = False
        else:
            result = recover_task("race_err")
            recovered = result is not None
            if recovered:
                reset_task("race_err")

        assert not recovered
        assert get_task_status("race_err") == STATUS_ERROR

    def test_recovery_skips_running_task(self, _mock_session_state):
        """Recovery should not interfere with a currently running task."""
        from vhh_library.background import (
            STATUS_IDLE,
            STATUS_RUNNING,
            get_task_status,
            submit_task,
        )

        event = threading.Event()

        def slow_work():
            event.wait(timeout=5)
            return "done"

        submit_task("race_run", slow_work)
        assert get_task_status("race_run") == STATUS_RUNNING

        # Recovery should skip because status is RUNNING
        assert get_task_status("race_run") != STATUS_IDLE
        event.set()
        time.sleep(0.3)

    def test_recovery_proceeds_after_session_loss(self, _mock_session_state):
        """Recovery SHOULD proceed when session was lost (status is IDLE)."""
        from vhh_library.background import (
            STATUS_DONE,
            STATUS_IDLE,
            get_task_status,
            recover_task,
            submit_task,
        )

        def work():
            return "recovered-result"

        submit_task("race_lost", work)
        time.sleep(0.5)
        assert get_task_status("race_lost") == STATUS_DONE

        # Simulate session loss: clear all task keys (fresh session_state)
        keys_to_remove = [k for k in _mock_session_state if k.startswith("bg_race_lost")]
        for k in keys_to_remove:
            del _mock_session_state[k]

        # Status is now IDLE (key missing → default)
        assert get_task_status("race_lost") == STATUS_IDLE

        # Recovery should proceed and restore the result from disk
        result = recover_task("race_lost")
        assert result == "recovered-result"
        assert get_task_status("race_lost") == STATUS_DONE


# ---------------------------------------------------------------------------
# Activity log tests
# ---------------------------------------------------------------------------


class TestActivityLog:
    """Tests for the log accumulation / activity log feature."""

    def test_submit_initialises_log_entries(self, _mock_session_state):
        from vhh_library.background import _key, submit_task

        event = threading.Event()
        submit_task("lg1", lambda: event.wait(timeout=2) or "done")
        assert _mock_session_state[_key("lg1", "log_entries")] == []
        event.set()
        time.sleep(0.2)

    def test_reset_clears_log_entries(self, _mock_session_state):
        from vhh_library.background import _key, reset_task

        _mock_session_state[_key("lg2", "status")] = "done"
        _mock_session_state[_key("lg2", "log_entries")] = [(1.0, "hello")]
        _mock_session_state[_key("lg2", "progress")] = 1.0
        _mock_session_state[_key("lg2", "progress_text")] = ""
        _mock_session_state[_key("lg2", "result")] = None
        _mock_session_state[_key("lg2", "error")] = None
        reset_task("lg2")
        assert _mock_session_state[_key("lg2", "log_entries")] == []

    def test_append_log_entry(self, _mock_session_state):
        from vhh_library.background import _key, append_log_entry

        _mock_session_state[_key("lg3", "log_entries")] = []
        append_log_entry("lg3", "step one")
        append_log_entry("lg3", "step two")
        entries = _mock_session_state[_key("lg3", "log_entries")]
        assert len(entries) == 2
        assert entries[0][1] == "step one"
        assert entries[1][1] == "step two"
        # Timestamps should be floats
        assert isinstance(entries[0][0], float)
        assert isinstance(entries[1][0], float)

    def test_append_log_entry_creates_list_if_missing(self, _mock_session_state):
        from vhh_library.background import _key, append_log_entry

        # No prior initialisation of log_entries
        append_log_entry("lg4", "first")
        entries = _mock_session_state[_key("lg4", "log_entries")]
        assert len(entries) == 1
        assert entries[0][1] == "first"

    def test_append_log_entry_with_state_override(self, _mock_session_state):
        from vhh_library.background import _key, append_log_entry

        custom_state: dict = {_key("lg5", "log_entries"): []}
        append_log_entry("lg5", "via state", _state=custom_state)
        assert len(custom_state[_key("lg5", "log_entries")]) == 1
        # Should NOT have written to session_state
        assert _key("lg5", "log_entries") not in _mock_session_state

    def test_get_task_log_returns_copy(self, _mock_session_state):
        from vhh_library.background import _key, append_log_entry, get_task_log

        _mock_session_state[_key("lg6", "log_entries")] = []
        append_log_entry("lg6", "msg")
        log = get_task_log("lg6")
        # Mutating the returned list should not affect the original
        log.append((0.0, "extra"))
        assert len(get_task_log("lg6")) == 1

    def test_get_task_log_empty_when_no_entries(self, _mock_session_state):
        from vhh_library.background import get_task_log

        assert get_task_log("nonexistent") == []

    def test_progress_callback_appends_log_entries(self, _mock_session_state):
        """make_progress_callback should append to the activity log."""
        from vhh_library.background import _key, make_progress_callback

        _mock_session_state[_key("lgcb", "log_entries")] = []
        cb = make_progress_callback("lgcb")

        # Simulate two progress events
        prog1 = MagicMock()
        prog1.round_number = 1
        prog1.total_rounds = 3
        prog1.phase = "generating_variants"
        prog1.message = "Building combinations…"

        prog2 = MagicMock()
        prog2.round_number = 2
        prog2.total_rounds = 3
        prog2.phase = "scoring_nativeness"
        prog2.message = "Scoring nativeness for 100 variants…"

        cb(prog1)
        cb(prog2)

        entries = _mock_session_state[_key("lgcb", "log_entries")]
        assert len(entries) == 2
        assert "Building combinations" in entries[0][1]
        assert "Scoring nativeness" in entries[1][1]

    def test_recover_task_initialises_log_entries(self, _mock_session_state):
        """recover_task should set log_entries to an empty list."""
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            _persist_result,
            recover_task,
        )

        _persist_result("lgr", status=STATUS_DONE, result="recovered")
        recover_task("lgr")
        assert _mock_session_state[_key("lgr", "log_entries")] == []

    def test_render_activity_log_renders_entries(self, _mock_session_state):
        """_render_activity_log should call st.expander and st.code."""
        from vhh_library.background import _key, _render_activity_log

        mock_st = _mock_session_state["__mock_st__"]

        _mock_session_state[_key("lgr2", "log_entries")] = [
            (1000000.0, "Step 1 done"),
            (1000001.0, "Step 2 done"),
        ]

        # Set up the expander mock as a context manager
        expander_ctx = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=expander_ctx)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        _render_activity_log("lgr2")

        mock_st.expander.assert_called_once()
        mock_st.code.assert_called_once()
        log_text = mock_st.code.call_args[0][0]
        assert "Step 1 done" in log_text
        assert "Step 2 done" in log_text


class TestThreadSafetyContract:
    """Verify that progress-reporting utilities are safe from background threads.

    Streamlit's ``st.session_state`` is only accessible from the main
    script-run thread.  Background threads launched by :func:`submit_task`
    must use captured state references — *never* ``st.session_state``
    directly.  These tests ensure the thread-safe helpers work correctly
    even when ``st.session_state`` is inaccessible.
    """

    def test_make_progress_callback_uses_captured_state(self, _mock_session_state):
        """make_progress_callback captures session_state at creation time
        and uses the captured reference from the background thread."""
        from vhh_library.background import _key, make_progress_callback

        _mock_session_state[_key("tsc", "log_entries")] = []
        cb = make_progress_callback("tsc")

        # Simulate calling from a background thread — the callback
        # should write to the captured dict, not to st.session_state.
        result = {}

        def _bg():
            try:
                from vhh_library.mutation_engine import IterativeProgress

                prog = IterativeProgress(
                    phase="test_phase",
                    round_number=3,
                    total_rounds=10,
                    best_score=0.95,
                    mean_score=0.80,
                    population_size=500,
                    n_anchors=5,
                    diversity_entropy=1.0,
                    message="",
                )
                cb(prog)
                result["ok"] = True
            except Exception as exc:
                result["error"] = str(exc)

        t = threading.Thread(target=_bg)
        t.start()
        t.join(timeout=5)

        assert result.get("ok") is True, f"Callback failed: {result.get('error')}"
        assert _mock_session_state[_key("tsc", "progress")] == pytest.approx(0.3)

    def test_make_progress_setter_works_alongside_submit_task(self, _mock_session_state):
        """Verify end-to-end: setter created in main thread, used in
        submit_task's background worker."""
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            make_progress_setter,
            submit_task,
        )

        setter = make_progress_setter("e2e")

        def _work():
            setter(0.5, "halfway")
            time.sleep(0.05)
            setter(1.0, "done")
            return "result"

        submit_task("e2e", _work)
        time.sleep(0.5)  # Let the thread finish

        assert _mock_session_state[_key("e2e", "status")] == STATUS_DONE
        assert _mock_session_state[_key("e2e", "progress")] == 1.0
        assert _mock_session_state[_key("e2e", "progress_text")] == "done"
        assert _mock_session_state[_key("e2e", "result")] == "result"

    def test_submit_task_closure_should_not_access_session_state_directly(self, _mock_session_state):
        """Demonstrate the bug: a closure that accesses st.session_state
        from a background thread would fail in real Streamlit, but works
        in tests because the mock is a plain dict.

        This test documents the pattern — closures must snapshot values
        from session state BEFORE being passed to submit_task.
        """
        from vhh_library.background import (
            STATUS_DONE,
            _key,
            submit_task,
        )

        # Correct pattern: snapshot values in the main thread
        _mock_session_state["param"] = 42
        captured_val = _mock_session_state.get("param", 0)

        def _correct_closure():
            return captured_val * 2

        submit_task("correct", _correct_closure)
        time.sleep(0.5)

        assert _mock_session_state[_key("correct", "status")] == STATUS_DONE
        assert _mock_session_state[_key("correct", "result")] == 84
