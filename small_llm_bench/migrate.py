"""Bring stored result files in line with the current task bank.

Unlike ``rescore``, which re-grades trials, this changes which trials *exist*:
a task removed from the bank should leave no trace in the files, or every
aggregator keeps counting it. It also carries trials across a module rename —
the four tool_* modules merged into ``tools`` in v0.10 — so the runs already on
disk stay usable instead of being stranded on a shape the bank no longer has.

What it deliberately does NOT do is re-score anything. A migrated file holds the
same verdicts it held before, minus the trials whose tasks are gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import BenchResult, Task, TaskResult, results_task_set_hash, task_content_hash
from .modules.base import load_tasks
from .rescore import _stimulus_changed

# Modules a task may legitimately have come from, keyed by where it lives now.
# A stored trial naming a predecessor is not an unknown task — it is the same
# task under its old label. v0.10 merged the four tool_* modules into `tools`;
# v0.13 dissolved `data_extract` and `tool_arg_typing` into `format`, except
# `tat_03`, which is a multi-turn tool episode and went to `tools` instead —
# hence `tool_arg_typing` appearing under both.
_MODULE_PREDECESSORS: dict[str, frozenset[str]] = {
    "tools": frozenset(
        {"tool_simple", "tool_loop", "tool_state", "tool_discovery",
         "tool_arg_typing"}),
    "format": frozenset({"data_extract", "tool_arg_typing"}),
}


class MigrateReport:
    """What a migration pass changed, per file."""

    def __init__(self) -> None:
        self.kept = 0
        self.dropped: list[str] = []
        self.relabelled = 0
        self.rehashed = 0
        self.kept_stale_hash: list[str] = []


def _bank_index(tasks_dir: Path | None) -> dict[str, Task]:
    """Every task in the current bank, keyed by task id.

    Keyed by id alone, not (module, id): the point is to recognise a task that
    changed module. Ids are unique across the bank (pinned by
    tests/test_tasks_schema.py).
    """
    modules = sorted(p.stem for p in (tasks_dir or _default_tasks_dir()).glob("*.yaml"))
    index: dict[str, Task] = {}
    for module in modules:
        for task in load_tasks(module, profile="full", tasks_dir=tasks_dir):
            index[task.id] = task
    return index


def _default_tasks_dir() -> Path:
    from .modules.base import find_tasks_dir
    return find_tasks_dir()


def migrate_bench(bench: BenchResult, *, tasks_dir: Path | None = None) -> MigrateReport:
    """Prune, relabel and re-stamp ``bench`` in place against the current bank."""
    report = MigrateReport()
    bank = _bank_index(tasks_dir)
    kept: list[TaskResult] = []
    dropped_duration = 0.0

    for result in bench.results:
        task = bank.get(result.task_id)
        if task is None:
            report.dropped.append(f"{result.module}:{result.task_id}")
            dropped_duration += result.duration_seconds or 0.0
            continue
        if result.module != task.module:
            if result.module not in _MODULE_PREDECESSORS.get(task.module, ()):
                # A task that moved somewhere unexpected is a bank edit this
                # tool doesn't understand; refuse rather than guess.
                raise ValueError(
                    f"{result.task_id}: stored module {result.module!r} is not a "
                    f"known predecessor of {task.module!r}"
                )
            result.module = task.module
            report.relabelled += 1
        if task.axis and result.axis != task.axis:
            result.axis = task.axis
        # The hash records which version of the task the trial answered. Refresh
        # it only where the model saw the same thing it would see today —
        # otherwise the drift flag that marks a trial un-poolable would be lost.
        if result.task_hash and result.task_hash != task_content_hash(task):
            if _stimulus_changed(task, result):
                report.kept_stale_hash.append(f"{task.module}:{result.task_id}")
            else:
                result.task_hash = task_content_hash(task)
                report.rehashed += 1
        kept.append(result)

    bench.results = kept
    report.kept = len(kept)
    bench.meta.task_count = len({(r.module, r.task_id) for r in kept})
    bench.meta.task_set_hash = results_task_set_hash(kept)
    # Wall clock of a run that no longer includes the dropped trials. Not
    # measured — reconstructed from the per-trial durations — but leaving the
    # original would overstate the cost of the pruned bank.
    if dropped_duration:
        bench.meta.duration_seconds = round(
            max(0.0, bench.meta.duration_seconds - dropped_duration), 2)
    return report


def migrate_file(path: Path, *, out_path: Path, tasks_dir: Path | None = None,
                 write: bool = True) -> tuple[MigrateReport, dict[str, Any]]:
    """Migrate one stored file; returns the report and the new meta summary."""
    from .reporter import load_results, save_results

    bench = load_results(path)
    report = migrate_bench(bench, tasks_dir=tasks_dir)
    if write:
        save_results(bench, out_path)
    return report, {"task_count": bench.meta.task_count,
                    "task_set_hash": bench.meta.task_set_hash,
                    "trials": len(bench.results)}
