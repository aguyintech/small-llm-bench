"""v0.15: anchor tasks are not dead weight, and a changed system prompt is drift.

Three defects, all in the same family — reporting and rescore reading a task's
metadata as if it could never move:

1. `classify_task` labelled every saturated task `dead_easy`, including
   `band: anchor` tasks whose whole job is to be passed by everything.
2. `rescore` compared only the user prompt and the tool list when deciding
   whether a stored response still answered today's task, so editing a
   `system_prompt` read as a grading-only change and the trial was re-graded
   against rules the model was never shown.
3. `rescore` never restamped `band`/`tier`/`difficulty`, so re-banding a task
   moved nothing in the reports until every model was re-run.
"""

from __future__ import annotations

import pytest

from small_llm_bench.analysis import classify_task
from small_llm_bench.models import Task, TaskResult
from small_llm_bench.rescore import _stimulus_changed, rescore_bench
from small_llm_bench.models import BenchMeta, BenchResult


def _row(**kw):
    row = {"task_id": "x", "module": "format", "band": "mid", "pass_rate": 1.0,
           "discrimination": 0.0, "flaky_models": 0, "n_models": 10}
    row.update(kw)
    return row


class TestAnchorClassification:
    def test_saturated_anchor_is_not_dead_easy(self):
        assert classify_task(_row(band="anchor"), 10) == "anchor"

    def test_saturated_non_anchor_is_still_dead_easy(self):
        assert classify_task(_row(band="hard"), 10) == "dead_easy"

    def test_missing_band_is_still_dead_easy(self):
        row = _row()
        del row["band"]
        assert classify_task(row, 10) == "dead_easy"

    def test_anchor_nobody_passes_is_still_dead_hard(self):
        # An anchor at 0.00 is a broken anchor, and must not be excused.
        assert classify_task(_row(band="anchor", pass_rate=0.0), 10) == "dead_hard"

    def test_anchor_that_discriminates_is_judged_on_its_merits(self):
        row = _row(band="anchor", pass_rate=0.7, discrimination=0.9, flaky_models=2)
        assert classify_task(row, 10) == "discriminating"


class TestSystemPromptIsStimulus:
    def _pair(self, task_sys, result_sys):
        task = Task(id="fm_x", module="format", prompt="p", system_prompt=task_sys)
        result = TaskResult(task_id="fm_x", module="format", prompt="p",
                            system_prompt=result_sys)
        return task, result

    def test_changed_system_prompt_is_drift(self):
        task, result = self._pair("follow rule A", "follow rule A and rule B")
        assert _stimulus_changed(task, result) is True

    def test_identical_system_prompt_is_not_drift(self):
        task, result = self._pair("follow rule A", "follow rule A")
        assert _stimulus_changed(task, result) is False

    def test_unrecorded_system_prompt_fails_closed(self):
        # Files written before v0.15 stored no system prompt. It cannot be
        # verified, so it counts as changed rather than as unchanged.
        task, result = self._pair("follow rule A", None)
        assert _stimulus_changed(task, result) is True

    def test_task_without_system_prompt_is_unaffected(self):
        task, result = self._pair(None, None)
        assert _stimulus_changed(task, result) is False


class TestRescoreRestampsBand:
    def test_band_tier_difficulty_refreshed_from_the_bank(self, monkeypatch):
        task = Task(id="tsp_01", module="tools", prompt="p", band="anchor",
                    tier="baseline", difficulty="hard",
                    expected={"tool_name": "get_weather"})
        result = TaskResult(task_id="tsp_01", module="tools", prompt="p",
                            band="hard", tier="hard", difficulty="medium",
                            system_prompt=None)
        bench = BenchResult(meta=BenchMeta(model="m", endpoint="http://x", timestamp="2026-08-30", duration_seconds=0.0, bench_version="test", trials=1), results=[result])

        monkeypatch.setattr("small_llm_bench.rescore._task_index",
                            lambda modules, tasks_dir: {("tools", "tsp_01"): task})
        rescore_bench(bench)

        assert result.band == "anchor"
        assert result.tier == "baseline"
        assert result.difficulty == "hard"


class TestNotGradedReachesTheJudge:
    """A task can narrow its rubric; the judge has to be told, or it widens it back.

    The first judged run over the 37-task bank demoted a qwen3.6-27b `pf_01`
    trial for a stale frontmatter `updated:` date — the exact check that task
    removed after ten models showed it measured a per-model habit rather than
    capability. The judge re-derived the requirement from the prompt, because
    nothing told it the omission was deliberate.
    """

    def _rendered(self, expected):
        from small_llm_bench.judge import render_judge_prompt
        r = TaskResult(task_id="pf_01", module="tools", prompt="p",
                       expected=expected, response_raw="x", det_score=1.0)
        return render_judge_prompt("tools", [r])

    def test_the_note_is_rendered_when_declared(self):
        out = self._rendered({"not_graded": ["The frontmatter date."]})
        assert "NOT GRADED" in out
        assert "The frontmatter date." in out

    def test_no_note_without_the_field(self):
        assert "NOT GRADED" not in self._rendered({"goal_tool": "x"})

    def test_the_shipped_pf_01_declares_it(self):
        from small_llm_bench.modules.base import load_tasks
        task = next(t for t in load_tasks("tools", profile="full")
                    if t.id == "pf_01")
        notes = " ".join(task.expected.get("not_graded", []))
        assert "frontmatter" in notes.lower()
        # And the check itself really is absent — the note must not paper over
        # a check that is still running.
        checks = task.expected["file_checks"]["people/dana.md"]
        assert not any(c.get("type") == "frontmatter" for c in checks)


class TestRescoreRestampsExpected:
    def test_expected_is_refreshed_from_the_bank(self, monkeypatch):
        """`expected` is what the judge is shown, so a stale copy hides a rubric
        change — including `not_graded`, which exists to stop the judge
        reimposing a dropped criterion."""
        task = Task(id="pf_01", module="tools", prompt="p",
                    expected={"tool_name": "get_weather",
                              "not_graded": ["the frontmatter date"]})
        result = TaskResult(task_id="pf_01", module="tools", prompt="p",
                            expected={"tool_name": "get_weather"})
        bench = BenchResult(meta=BenchMeta(model="m", endpoint="http://x",
                                           timestamp="2026-08-30",
                                           duration_seconds=0.0,
                                           bench_version="test", trials=1),
                            results=[result])
        monkeypatch.setattr("small_llm_bench.rescore._task_index",
                            lambda modules, tasks_dir: {("tools", "pf_01"): task})
        rescore_bench(bench)
        assert result.expected["not_graded"] == ["the frontmatter date"]


class TestMt25GradesTheCorrection:
    """Turn 6 used to grade only the bullet count, so a model could carry
    mobile's stale status forward and still score 1.00 — the correction the
    task exists to test went ungraded."""

    def _turn6(self):
        from small_llm_bench.modules.base import load_tasks
        task = next(t for t in load_tasks("multi_turn_if", profile="full")
                    if t.id == "mt_25")
        return task.conversation[5]["constraints"]

    def test_turn_six_checks_the_status_was_updated(self):
        assert any(c.get("id") == "mobile-updated" for c in self._turn6())

    def test_the_pattern_catches_the_stale_status_only(self):
        from small_llm_bench.scorer import _check_constraints
        check = [c for c in self._turn6() if c.get("id") == "mobile-updated"]
        stale = "STANDUP:\n- API shipped caching\n- Mobile is waiting on review\n"
        for updated in ("STANDUP:\n- Mobile review is complete\n",
                        "STANDUP:\n- Mobile review came through this morning\n",
                        "STANDUP:\n- Mobile review cleared; mobile can proceed\n",
                        "STANDUP:\n- Design is waiting on infra for migration\n"):
            assert _check_constraints(check, updated)[0] == 1.0, updated
        assert _check_constraints(check, stale)[0] == 0.0
