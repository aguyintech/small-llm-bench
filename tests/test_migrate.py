"""Bringing stored runs in line with a changed bank.

`rescore` re-grades trials; `migrate` changes which trials exist. A task pruned
from the bank has to leave no trace, or every aggregator keeps counting it — and
a task that merely changed module has to survive, or a bank reshuffle strands
every run already on disk.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from small_llm_bench.migrate import migrate_bench
from small_llm_bench.models import (BenchMeta, BenchResult, TaskResult,
                                    TurnRecord, results_task_set_hash,
                                    task_content_hash)
from small_llm_bench.modules.base import load_tasks

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"


def _bank_task(module: str, task_id: str) -> Task:
    """A task by id, from the shipped bank or from the v0.15 retirees.

    The v0.15 cut took the bank from 59 tasks to 40. Nineteen of those tasks
    were pinning scorer behaviours here, so their definitions moved unchanged
    to tests/fixtures/retired_bank/ and this looks there second. A test keeps
    asserting exactly what it asserted before; only the task's membership of
    the shipped bank changed.
    """
    for tasks_dir in (None, RETIRED_BANK):
        for task in load_tasks(module, profile="full", tasks_dir=tasks_dir):
            if task.id == task_id:
                return task
    raise LookupError(f"{module}:{task_id} is in neither bank")


def _bench(results: list[TaskResult], duration: float = 100.0) -> BenchResult:
    return BenchResult(
        meta=BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                       duration_seconds=duration, bench_version="0.9.0",
                       trials=3, task_count=52, task_set_hash="stale"),
        results=results)


def _task(task_id: str):
    return _bank_task("tools", task_id)


def _trial(task, *, module: str | None = None, **kwargs) -> TaskResult:
    fields = {"det_score": 1.0, "success": True, "det_success": True, **kwargs}
    return TaskResult(task_id=task.id, module=module or task.module,
                      prompt=task.prompt, task_hash=task_content_hash(task),
                      **fields)


class TestPruning:
    def test_a_removed_task_leaves_no_trace(self):
        keeper = _trial(_task("tl_02"))
        gone = TaskResult(task_id="ds_05", module="tool_discovery", prompt="p",
                          det_score=1.0, success=True, duration_seconds=13.6)
        bench = _bench([keeper, gone])
        report = migrate_bench(bench)
        assert [r.task_id for r in bench.results] == ["tl_02"]
        assert report.dropped == ["tool_discovery:ds_05"]

    def test_meta_is_restamped_so_the_file_looks_native(self):
        keeper = _trial(_task("tl_02"))
        gone = TaskResult(task_id="ds_08", module="tool_discovery", prompt="p",
                          duration_seconds=17.7)
        bench = _bench([keeper, gone], duration=100.0)
        migrate_bench(bench)
        assert bench.meta.task_count == 1
        assert bench.meta.task_set_hash == results_task_set_hash(bench.results)
        # the dropped trial's wall clock goes with it
        assert bench.meta.duration_seconds == pytest.approx(82.3)

    def test_nothing_is_regraded(self):
        trial = _trial(_task("tl_02"), det_score=0.42, success=False,
                       det_success=False)
        migrate_bench(_bench([trial]))
        assert (trial.det_score, trial.success) == (0.42, False)


class TestModuleMerge:
    @pytest.mark.parametrize("old", ["tool_simple", "tool_loop", "tool_state",
                                     "tool_discovery"])
    def test_a_predecessor_module_is_carried_over(self, old):
        task = _task("tl_02")
        trial = _trial(task, module=old)
        report = migrate_bench(_bench([trial]))
        assert trial.module == "tools"
        assert report.relabelled == 1
        assert report.dropped == []

    def test_the_axis_is_filled_in_from_the_bank(self):
        trial = _trial(_task("tst_03"), module="tool_state")
        migrate_bench(_bench([trial]))
        assert trial.axis == "state"

    def test_an_unrecognised_module_move_is_refused(self):
        """Guessing at an unknown bank edit is worse than stopping."""
        trial = _trial(_task("tl_02"), module="something_else")
        with pytest.raises(ValueError, match="known predecessor"):
            migrate_bench(_bench([trial]))


class TestHashHandling:
    def test_a_grading_only_change_refreshes_the_hash(self):
        task = _task("tl_02")
        trial = _trial(task, module="tool_loop")
        trial.task_hash = "deadbeefdeadbeef"
        report = migrate_bench(_bench([trial]))
        assert trial.task_hash == task_content_hash(task)
        assert report.rehashed == 1
        assert report.kept_stale_hash == []

    def test_a_changed_stimulus_keeps_its_old_hash(self):
        """That hash is the only record that the trial answered another prompt;
        losing it would silently make it poolable with future runs."""
        task = _task("tl_02")
        trial = _trial(task, module="tool_loop")
        trial.task_hash = "deadbeefdeadbeef"
        trial.prompt = "an older wording"
        report = migrate_bench(_bench([trial]))
        assert trial.task_hash == "deadbeefdeadbeef"
        assert report.kept_stale_hash == ["tools:tl_02"]

    def test_a_retired_injected_error_counts_as_changed_stimulus(self):
        task = _task("tl_26")
        trial = _trial(task, module="tool_loop")
        trial.task_hash = "deadbeefdeadbeef"
        trial.turns = [TurnRecord(role="tool", content=json.dumps(
            {"error": "rejected: titles must start with 'ACME-'"}))]
        report = migrate_bench(_bench([trial]), tasks_dir=RETIRED_BANK)
        assert report.kept_stale_hash == ["tools:tl_26"]
