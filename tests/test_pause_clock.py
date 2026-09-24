"""A pause is the operator's time, not the model's.

`meta.duration_seconds` is the run-level wall clock, and it used to absorb
however long the user held the run with 'p' — so a coffee break read as a slow
model against the runtime budget the bank is calibrated to. Per-task
`duration_seconds` never had the bug (the gate is awaited before
`_execute_task` starts its timer), which is why these tests pin the controller's
accounting rather than the task path.
"""

from __future__ import annotations

import pytest

from small_llm_bench.runner import PauseController


class _SilentConsole:
    def print(self, *args: object, **kwargs: object) -> None:
        pass


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test advances by hand."""
    now = {"t": 1000.0}
    monkeypatch.setattr("small_llm_bench.runner.time.monotonic",
                        lambda: now["t"])
    return now


@pytest.fixture
def pause(monkeypatch, clock):
    """A controller whose keypress reader always yields 'p'."""
    monkeypatch.setattr("small_llm_bench.runner.os.read",
                        lambda *_: b"p")
    controller = PauseController(_SilentConsole())
    controller._fd = 0
    return controller


def test_unpaused_run_reports_no_held_time(pause, clock):
    clock["t"] += 300.0
    assert pause.paused_seconds == 0.0


def test_one_pause_resume_cycle_is_excluded(pause, clock):
    pause._on_key()             # pause
    assert not pause.running.is_set()
    clock["t"] += 120.0
    pause._on_key()             # resume
    assert pause.running.is_set()
    assert pause.paused_seconds == pytest.approx(120.0)


def test_held_time_accumulates_across_cycles(pause, clock):
    for held in (30.0, 45.0, 5.5):
        pause._on_key()
        clock["t"] += held
        pause._on_key()
        clock["t"] += 10.0      # running time between pauses is not counted
    assert pause.paused_seconds == pytest.approx(80.5)


def test_time_spent_running_is_never_counted_as_held(pause, clock):
    clock["t"] += 60.0          # running
    pause._on_key()
    clock["t"] += 20.0          # held
    pause._on_key()
    clock["t"] += 60.0          # running
    assert pause.paused_seconds == pytest.approx(20.0)


def test_a_pause_still_open_is_counted(pause, clock):
    pause._on_key()
    clock["t"] += 75.0
    # Never resumed — the run was aborted while held. The total is read after
    # the `with` block exits, so an open interval has to be visible.
    assert pause.paused_seconds == pytest.approx(75.0)


def test_exit_closes_an_open_pause_without_double_counting(pause, clock):
    pause._on_key()
    clock["t"] += 75.0
    pause.__exit__()
    clock["t"] += 500.0         # time after the run ends must not accrue
    assert pause.paused_seconds == pytest.approx(75.0)


def test_keys_other_than_p_do_not_pause(monkeypatch, clock):
    monkeypatch.setattr("small_llm_bench.runner.os.read", lambda *_: b"x")
    controller = PauseController(_SilentConsole())
    controller._fd = 0
    controller._on_key()
    clock["t"] += 90.0
    assert controller.running.is_set()
    assert controller.paused_seconds == 0.0


def test_non_tty_exit_still_closes_the_clock(pause, clock):
    """`__exit__` returns early when there are no terminal settings to restore;
    the interval has to be closed before that return, not after it."""
    assert pause._old_settings is None
    pause._on_key()
    clock["t"] += 40.0
    pause.__exit__()
    assert pause._paused_at is None
    assert pause.paused_seconds == pytest.approx(40.0)
