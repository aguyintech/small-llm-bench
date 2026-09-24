"""Re-scoring stored runs offline.

A scorer fix that can only be seen by re-running six models is a fix nobody
applies. Everything the scorer reads is persisted per trial, so `sllmb rescore`
grades stored runs again for free — while refusing, by default, to grade a
trial whose task itself changed underneath it.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from small_llm_bench.models import (BenchMeta, BenchResult, Task, TaskResult,
                                    TurnRecord, task_content_hash)
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.rescore import rescore_bench

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"


def _bench(results: list[TaskResult]) -> BenchResult:
    return BenchResult(
        meta=BenchMeta(model="m", endpoint="http://x/v1",
                       timestamp="2026-08-22T00:00:00Z", duration_seconds=1.0,
                       bench_version="0.9.0"),
        results=results)


def _lc_task() -> Task:
    """A real bank task, so the hash logic is exercised against real data."""
    return next(t for t in load_tasks("long_context", profile="full")
                if t.id == "lc_08")


def _trial(task: Task, response: str, **kwargs) -> TaskResult:
    return TaskResult(task_id=task.id, module=task.module, prompt=task.prompt,
                      response_raw=response, task_hash=task_content_hash(task),
                      det_score=1.0, success=True, det_success=True, **kwargs)


class TestRescore:
    def test_stale_pass_is_regraded_against_the_current_scorer(self):
        task = _lc_task()
        trial = _trial(task, "The current code is 7284, hope that helps!")
        bench = _bench([trial])
        report = rescore_bench(bench)
        assert report.rescored == 1
        # ends_with_number now graded: right answer, wrong placement.
        assert trial.success is False
        assert trial.det_score == pytest.approx(0.85)
        assert report.changed == ["long_context:lc_08 -> fail"]
        assert report.moved_keys == {("long_context", "lc_08")}

    def test_a_still_correct_trial_is_left_alone(self):
        task = _lc_task()
        trial = _trial(task, "The current Oslo access code is 7284")
        rescore_bench(_bench([trial]))
        assert (trial.success, trial.det_score) == (True, 1.0)

    def test_a_changed_prompt_is_skipped_by_default(self):
        """The stored response answers a question that no longer exists."""
        task = _lc_task()
        trial = _trial(task, "nonsense")
        trial.prompt = "an older wording of this task"
        trial.task_hash = "deadbeefdeadbeef"
        report = rescore_bench(_bench([trial]))
        assert report.rescored == 0
        assert report.skipped_drift == ["long_context:lc_08"]
        assert trial.success is True  # untouched

    def test_overrides_alone_are_not_drift(self):
        """A task can inject tool errors and still be re-gradable: what matters
        is whether the wording the model saw is still producible."""
        task = next(t for t in load_tasks("tools", profile="full", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_26")
        trial = _trial(task, "done")
        trial.task_hash = "deadbeefdeadbeef"
        trial.turns = [TurnRecord(role="tool", content=json.dumps(
            {"error": task.tool_overrides["create_ticket"]["error_message"]}))]
        report = rescore_bench(_bench([trial]), tasks_dir=RETIRED_BANK)
        assert report.skipped_drift == []
        assert report.regraded_only == ["tools:tl_26"]

    def test_a_retired_injected_error_is_drift(self):
        """The model was told something the task no longer says — that trial
        answered a different question and cannot be pooled with new runs."""
        task = next(t for t in load_tasks("tools", profile="full", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_26")
        trial = _trial(task, "done")
        trial.task_hash = "deadbeefdeadbeef"
        trial.turns = [TurnRecord(role="tool", content=json.dumps(
            {"error": "rejected: titles must start with 'ACME-'"}))]
        report = rescore_bench(_bench([trial]), tasks_dir=RETIRED_BANK)
        assert report.skipped_drift == ["tools:tl_26"]
        assert report.rescored == 0

    def test_registry_errors_are_not_drift(self):
        """'unknown tool' comes from the mock registry, not the task."""
        task = next(t for t in load_tasks("tools", profile="full", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_26")
        trial = _trial(task, "done")
        trial.task_hash = "deadbeefdeadbeef"
        trial.turns = [TurnRecord(role="tool", content=json.dumps(
            {"error": "unknown tool: nope"})),
            TurnRecord(role="tool", content="not json at all")]
        report = rescore_bench(_bench([trial]))
        assert report.skipped_drift == []

    def test_changed_expected_alone_is_not_drift(self):
        """`expected` is never shown to the model, so it is a grading change."""
        task = next(t for t in load_tasks("tools", profile="full", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_23")
        trial = _trial(task, "done")
        trial.task_hash = "deadbeefdeadbeef"
        trial.expected = {"goal_tool": "post_update", "min_calls": 7}
        report = rescore_bench(_bench([trial]))
        assert report.skipped_drift == []

    def test_a_grading_only_change_is_regraded(self):
        """Adding a constraint changes nothing the model saw."""
        task = _lc_task()
        trial = _trial(task, "The current code is 7284, hope that helps!")
        trial.task_hash = "deadbeefdeadbeef"
        report = rescore_bench(_bench([trial]))
        assert report.rescored == 1
        assert report.regraded_only == ["long_context:lc_08"]
        assert trial.success is False
        assert trial.task_hash == task_content_hash(task)

    def test_drift_can_be_forced(self):
        task = _lc_task()
        trial = _trial(task, "nonsense")
        trial.prompt = "an older wording of this task"
        trial.task_hash = "deadbeefdeadbeef"
        report = rescore_bench(_bench([trial]), allow_task_drift=True)
        assert report.rescored == 1
        assert trial.success is False
        assert trial.task_hash == task_content_hash(task)

    def test_code_needs_a_sandbox(self):
        code_task = next(t for t in load_tasks("code", profile="full"))
        trial = _trial(code_task, "def f():\n    return 1\n")
        report = rescore_bench(_bench([trial]))
        assert report.rescored == 0
        assert report.skipped_sandbox == [f"code:{code_task.id}"]

    def test_unknown_task_is_reported_not_crashed(self):
        trial = TaskResult(task_id="gone_99", module="long_context",
                           prompt="p", response_raw="x")
        report = rescore_bench(_bench([trial]))
        assert report.skipped_unknown == ["long_context:gone_99"]
        assert report.rescored == 0

    def test_stored_judge_score_is_preserved_and_reapplied(self):
        task = _lc_task()
        trial = _trial(task, "The current code is 7284, hope that helps!")
        trial.llm_score, trial.llm_reasoning = 0.95, "found the needle"
        rescore_bench(_bench([trial]))
        assert trial.llm_reasoning == "found the needle"  # opinion is data
        assert trial.det_success is False
        # long_context is display-only: the judge's score is shown as emitted
        # and cannot move success in either direction.
        assert trial.success is False
        assert trial.llm_score == pytest.approx(0.95)

    def test_judge_echo_no_longer_demotes_on_rescore(self):
        task = next(t for t in load_tasks("tools", profile="full")
                    if t.id == "tl_02")
        trial = TaskResult(task_id=task.id, module=task.module,
                           prompt=task.prompt, response_raw="done",
                           task_hash=task_content_hash(task),
                           det_score=0.8333, success=True, det_success=True)
        trial.llm_score = 0.8333
        rescore_bench(_bench([trial]))
        assert trial.success is False  # no calls stored -> det fail on regrade
        assert trial.det_success is False


class TestVerdictIsNotDoubleApplied:
    """A re-judge after a rescore must not freeze the judge's own answer in as
    the deterministic verdict (it did: ornith mt_07 came back det_success True
    on a trial whose per-turn fractions were 0.5/1.0/0.8/0.67/1.0)."""

    def test_stale_verdict_is_cleared_before_rejudging(self):
        from small_llm_bench.judge import apply_judge_verdict

        task = _lc_task()
        trial = _trial(task, "The current code is 7284, hope that helps!")
        trial.llm_score = 1.0
        rescore_bench(_bench([trial]))
        assert (trial.det_success, trial.success) == (False, False)

        # What the CLI does before handing the trial back to the judge.
        trial.success = trial.det_success
        trial.llm_score, trial.llm_reasoning = None, None
        trial.llm_score = 0.85
        apply_judge_verdict(trial, trial.module)
        assert trial.det_success is False  # not the judge's own answer


def test_rescore_stamps_the_current_bench_version():
    """Deterministic scores moved without any task hash moving, so the version
    is the only signal that two rows were graded by different rules."""
    from small_llm_bench import __version__

    task = _lc_task()
    bench = _bench([_trial(task, "the code is 7284")])
    bench.meta.bench_version = "0.8.0"
    rescore_bench(bench)
    assert bench.meta.bench_version == __version__


class TestStaleJudgeAnchor:
    """A stored judge verdict is only meaningful against the deterministic
    score it was shown.

    The judge prompt tells the model to treat det as the default and keep it
    where it agrees, so an echoed score is agreement. Re-score that trial to a
    lower det and the echo turns into a large positive delta — which reads as a
    rescue. Applying the v1.0 delta-credit fix to the stored bank turned 14
    deterministic failures across 7 models into passes exactly this way.
    """

    def _moved_tools_trial(self) -> tuple[Task, TaskResult]:
        """tst_57 with an untouched world: det 0.978 under the old absolute
        rule, 0.400 under delta credit."""
        task = next(t for t in load_tasks("tools", profile="full")
                    if t.id == "tst_57")
        trial = TaskResult(task_id=task.id, module=task.module,
                           prompt=task.prompt, response_raw="",
                           task_hash=task_content_hash(task),
                           final_state=dict(task.initial_state),
                           det_score=0.9784, success=False, det_success=False)
        return task, trial

    def test_an_echo_on_a_moved_trial_does_not_become_a_rescue(self):
        task, trial = self._moved_tools_trial()
        trial.llm_score, trial.llm_reasoning = 0.9784, "agreed with det"
        rescore_bench(_bench([trial]))
        assert trial.det_score < 0.5          # delta credit applied
        assert trial.det_success is False
        assert trial.success is False          # the rescue is refused
        assert trial.judge_blocked_by is not None

    def test_the_judge_opinion_is_still_kept_as_data(self):
        task, trial = self._moved_tools_trial()
        trial.llm_score, trial.llm_reasoning = 0.9784, "agreed with det"
        report = rescore_bench(_bench([trial]))
        assert trial.llm_score == pytest.approx(0.9784)
        assert trial.llm_reasoning == "agreed with det"
        assert report.stale_judge == ["tools:tst_57"]

    def test_a_verdict_judged_against_the_current_score_can_still_rescue(self):
        """The guard is the anchor, not the fact that a rescore ran."""
        from small_llm_bench.judge import apply_judge_verdict

        task, trial = self._moved_tools_trial()
        trial.det_score, trial.success, trial.det_success = 0.4, False, False
        trial.llm_score, trial.judge_anchor_det = 1.0, 0.4
        apply_judge_verdict(trial, "tools")
        assert trial.success is True

    def test_the_block_survives_a_second_rescore(self):
        """Staleness lives on the trial, not in the pass that noticed it.

        The first rescore moves the score and stamps the anchor; the second
        sees a settled score and no longer knows anything moved. Without the
        stored anchor it re-derived the verdict from the stale judge score and
        handed back every rescue the first pass had refused — 17 of them.
        """
        task, trial = self._moved_tools_trial()
        trial.llm_score, trial.llm_reasoning = 0.9784, "agreed with det"
        rescore_bench(_bench([trial]))
        assert trial.success is False
        assert trial.judge_anchor_det == pytest.approx(0.9784)

        report = rescore_bench(_bench([trial]))   # nothing moves this time
        assert trial.success is False
        assert report.stale_judge == ["tools:tst_57"]

    def test_re_judging_clears_the_anchor_with_the_verdict(self):
        """A fresh verdict is anchored on the score the judge actually saw."""
        from small_llm_bench.judge import apply_judge_verdict

        task, trial = self._moved_tools_trial()
        trial.llm_score = 0.9784
        rescore_bench(_bench([trial]))
        # What the CLI does before handing the trial back to the judge.
        trial.success = trial.det_success
        trial.llm_score = trial.llm_reasoning = trial.judge_anchor_det = None
        trial.llm_score, trial.judge_anchor_det = 1.0, trial.det_score
        apply_judge_verdict(trial, "tools")
        assert trial.success is True


class TestGenerationFailuresAreNotRegraded:
    """A trial the run never got a response for has nothing to re-grade.

    Two stored 500s — the server rejecting a tool call the model malformed,
    which `82e90dc` ruled a MODEL failure so `counted` keeps them — came back
    from a rescore at 0.2, collected from `efficiency` and `no_loop_detected`:
    making no calls reads as maximally efficient and non-looping. The runner
    had recorded 0.0.
    """

    def _failed_trial(self) -> tuple[Task, TaskResult]:
        task = next(t for t in load_tasks("tools", profile="full")
                    if t.id == "tst_57")
        trial = TaskResult(task_id=task.id, module=task.module,
                           prompt=task.prompt, response_raw="",
                           task_hash=task_content_hash(task),
                           turns=[], final_state={},
                           det_score=0.0, success=False, det_success=False,
                           error="HTTPStatusError: Server error '500 ...'")
        return task, trial

    def test_it_is_skipped_not_scored(self):
        task, trial = self._failed_trial()
        report = rescore_bench(_bench([trial]))
        assert report.rescored == 0
        assert report.skipped_error == ["tools:tst_57"]
        assert trial.det_score == 0.0
        assert trial.det_breakdown == {}

    def test_a_scoring_crash_is_still_regraded(self):
        """Narrow on purpose: those keep their turns, and re-grading fixes them."""
        task, trial = self._failed_trial()
        trial.error = "scoring failed: KeyError: 'files'"
        trial.turns = [TurnRecord(role="assistant", content="done")]
        report = rescore_bench(_bench([trial]))
        assert report.rescored == 1
        assert report.skipped_error == []
