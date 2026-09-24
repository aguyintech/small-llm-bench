"""Re-score stored trials with the current deterministic scorer.

A scorer improvement is worthless if it can only be observed by re-running
every model: `sllmb score` re-prints stored numbers, and `score_task` otherwise
only runs inside a live benchmark. Everything the scorer reads (turns,
final_state, response_raw, truncated, attempts_used) is already persisted per
trial, so a stored run can be graded again offline for free.

What this deliberately refuses to do is re-score a trial whose *stimulus*
changed: a different prompt or different tool behavior means the stored
response answers a question that no longer exists, and grading it under the new
rules would silently mix two banks.

``task_hash`` alone is too blunt to decide that — it covers grading fields too,
so adding a constraint to a task (which changes nothing the model saw) reads as
drift. So a hash mismatch is triaged against what the trial itself recorded: the
prompt the model was given and the tools it was offered. If those still match
and the task injects no tool behavior, the change was grading-only and the
trial is safely re-gradable; otherwise it is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__
from .judge import apply_judge_verdict, judge_anchor_is_stale
from .models import BenchResult, Task, TaskResult, task_content_hash
from .modules.base import load_tasks
from .scorer import score_task, truncation_class

# Needs a live sandbox to re-execute candidate code, so it is skipped unless the
# caller passes one explicitly.
_SANDBOX_MODULES = frozenset({"code"})


class RescoreReport:
    """Counts and per-trial notes from a rescore pass."""

    def __init__(self) -> None:
        self.rescored = 0
        self.changed: list[str] = []
        # (module, task_id) pairs whose deterministic grade moved at all — the
        # set worth re-judging, since the judge prompt anchors on det_score.
        self.moved_keys: set[tuple[str, str]] = set()
        # Trials whose stored judge verdict was dropped as stale: its anchor,
        # the deterministic score, moved underneath it. They are unjudged until
        # --judge re-asks.
        self.stale_judge: list[str] = []
        self.regraded_only: list[str] = []
        # Generation failures: nothing was produced, so nothing is re-gradable.
        self.skipped_error: list[str] = []
        self.skipped_drift: list[str] = []
        self.skipped_sandbox: list[str] = []
        self.skipped_unknown: list[str] = []

    @property
    def skipped(self) -> int:
        """Total trials left exactly as they were found."""
        return (len(self.skipped_drift) + len(self.skipped_sandbox)
                + len(self.skipped_unknown) + len(self.skipped_error))


def _task_index(modules: set[str],
                tasks_dir: Path | None) -> dict[tuple[str, str], Task]:
    """Map (module, task_id) to the task as the bank defines it today."""
    index: dict[tuple[str, str], Task] = {}
    for module in sorted(modules):
        for task in load_tasks(module, profile="full", tasks_dir=tasks_dir):
            index[(module, task.id)] = task
    return index


def rescore_bench(bench: BenchResult, *, tasks_dir: Path | None = None,
                  sandbox: dict[str, Any] | None = None,
                  allow_task_drift: bool = False) -> RescoreReport:
    """Re-grade every trial in ``bench`` in place with the current scorer.

    A stored ``llm_score`` is preserved and the final verdict is re-derived
    through ``apply_judge_verdict``, so a judged file stays judged: the judge's
    opinion is data, the rules that turn it into a pass are code.
    """
    report = RescoreReport()
    index = _task_index({r.module for r in bench.results}, tasks_dir)
    # The file now carries scores produced by THIS scorer, and bench_version is
    # a leaderboard comparability field precisely because a scorer change moves
    # scores without moving a single task hash.
    bench.meta.bench_version = __version__

    for result in bench.results:
        key = (result.module, result.task_id)
        task = index.get(key)
        label = f"{result.module}:{result.task_id}"
        if task is None:
            report.skipped_unknown.append(label)
            continue
        if result.module in _SANDBOX_MODULES and sandbox is None:
            report.skipped_sandbox.append(label)
            continue
        if result.error and not result.turns:
            # A generation failure left nothing to re-grade — no turns, no final
            # state. Grading that anyway pays the trial for the dimensions
            # inaction earns: two stored 500s (the server refusing a tool call
            # the model malformed, which `82e90dc` ruled a MODEL failure, so
            # `counted` keeps them) came back 0.0 -> 0.2 on efficiency and
            # loop-avoidance alone. The runner's verdict is the only honest one.
            # Narrow on purpose: a trial whose SCORING crashed still has its
            # turns, and re-grading is exactly what fixes it.
            report.skipped_error.append(label)
            continue
        current_hash = task_content_hash(task)
        if result.task_hash and result.task_hash != current_hash:
            if _stimulus_changed(task, result) and not allow_task_drift:
                report.skipped_drift.append(label)
                continue
            if not _stimulus_changed(task, result):
                report.regraded_only.append(label)
            result.task_hash = current_hash
        _rescore_one(task, result, sandbox, report, label)
    return report


def _stimulus_changed(task: Task, result: TaskResult) -> bool:
    """True if what the model was shown or handed differs from today's task.

    Compares against what the trial itself recorded rather than a stored
    stimulus hash, so old result files work. Three things were model-visible:
    the prompt, the tools offered, and — for tasks that inject tool behavior —
    the error text the mocks returned. All three are persisted, the last one
    inside the stored tool turns, so injected wording is checked directly
    instead of treating every task with ``tool_overrides`` as drifted.

    Deliberately NOT drift: a changed ``expected``, ``constraints`` or
    ``content_checks``. The model never sees those, so a change there is a
    grading change and the stored trajectory can be re-graded under it.

    The system prompt is model-visible and therefore drift, but it was not
    persisted before v0.15. On an older file it cannot be compared, so a task
    that HAS one fails closed: unverifiable is treated as changed rather than
    as unchanged. Silently assuming otherwise is how fm_17's rules were
    re-graded against responses written under the old rules.
    """
    if result.prompt and result.prompt != task.prompt:
        return True
    if task.system_prompt is not None and result.system_prompt != task.system_prompt:
        return True
    offered = {(t.get("function") or {}).get("name")
               for t in result.tools_schema}
    offered.discard(None)
    if offered and offered != set(task.tools):
        return True
    return _injected_errors_changed(task, result)


def _injected_errors_changed(task: Task, result: TaskResult) -> bool:
    """True if the trial saw an injected error the task can no longer produce.

    The mocks' own failures ("unknown tool", "invalid arguments for ...") are
    ignored — those come from the registry, not the task, and reproduce on a
    re-grade. Only text the task itself supplies is compared.
    """
    if not task.tool_overrides:
        return False
    producible = set()
    for override in task.tool_overrides.values():
        for key in ("error_message", "require_args_error"):
            if override.get(key):
                producible.add(str(override[key]).strip())
    seen = set()
    for turn in result.turns:
        if getattr(turn, "role", "") != "tool":
            continue
        payload = turn.content or ""
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            seen.add(parsed["error"].strip())
    seen -= {e for e in seen
             if e.startswith(("unknown tool", "invalid arguments for",
                              "unknown first_call_behavior"))}
    return bool(seen - producible)


def _rescore_one(task: Task, result: TaskResult, sandbox: dict[str, Any] | None,
                 report: RescoreReport, label: str) -> None:
    """Re-grade one trial, recording whether its pass flag moved."""
    # Reporting metadata, restamped from the bank the way the runner does at
    # run time. None of it is stimulus and none of it is graded, but ALL of it
    # is read downstream: `band` drives the band-weighted headline and the
    # anchor branch in analysis.classify_task, `tier` drives the capability
    # split. Before v0.15 rescore left them frozen at whatever the task said on
    # the day it ran, so re-banding a task moved nothing until every model was
    # re-run — fm_08 was still stored as `hard` and fm_22 as `frontier` long
    # after both became anchors.
    result.difficulty = task.difficulty
    result.tier = task.tier
    result.band = task.band
    # `expected` is not stimulus and is not what grading reads here (that comes
    # from `task`), but it IS what the judge is shown, so a stale copy would
    # hide a rubric change — including `not_graded`, which exists precisely to
    # stop the judge reimposing a criterion the task dropped.
    result.expected = dict(task.expected)
    was_success = result.success
    was_det_score, was_det_success = result.det_score, result.det_success
    scored = score_task(task, result, sandbox)
    result.det_score = scored.score
    result.det_breakdown = scored.breakdown
    # Re-grading re-measures the SANDBOX verdict, so the sandbox-derived infra
    # flag is re-derived with it — that is what lets a file whose code module
    # was lost to a missing image come back with real scores instead of
    # fifteen zeros. A generation failure (`result.error`) is not something
    # re-grading can re-measure, so that flag stands untouched.
    if not result.error:
        result.infra_error = scored.infra_error
    result.truncation_class = truncation_class(result)
    result.success = scored.success
    result.det_success = scored.success
    moved = (abs(scored.score - was_det_score) > 1e-9
             or scored.success != was_det_success)
    if result.llm_score is not None:
        # Stamp the anchor for a trial judged before the field existed: this is
        # the only moment the score the judge actually saw is still known. Files
        # written since carry it already, and re-stamping would point it at the
        # score the judge never saw — so an existing anchor is left alone.
        if moved and result.judge_anchor_det is None:
            result.judge_anchor_det = was_det_score
        # The delta-credit fix dropped tst_57 from 0.978 to 0.400 against a
        # stored 0.978 and turned 14 deterministic failures across 7 models into
        # passes. The opinion is still data and is still displayed; what it
        # loses, until --judge re-asks against the score that now exists, is the
        # power to overturn a failure.
        apply_judge_verdict(result, task.module)
        if judge_anchor_is_stale(result):
            report.stale_judge.append(label)
    report.rescored += 1
    if moved:
        report.moved_keys.add((task.module, result.task_id))
    if result.success != was_success:
        direction = "pass" if result.success else "fail"
        report.changed.append(f"{label} -> {direction}")
