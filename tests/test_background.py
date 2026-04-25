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
