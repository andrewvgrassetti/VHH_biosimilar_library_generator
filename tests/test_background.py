"""Unit tests for the background-task utility (vhh_library.background)."""

from __future__ import annotations

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
    monkeypatch.setattr(bg_mod, "st", mock_st)
    return state


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
