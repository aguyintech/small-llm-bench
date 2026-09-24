"""Deterministic coverage the LLM judge used to carry alone.

Across six judged v0.8 runs the judge disagreed with the deterministic scorer on
15 of 936 trials. Four causes were deterministic gaps rather than semantics, and
each is pinned here: a forbidden tool riding in on a valid goal call, an
injected error that lied to a model whose argument was already correct, a
prompt-stated answer placement nothing graded, and a judge "demote" that only
echoed the deterministic score back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from small_llm_bench.judge import (JUDGE_PASS_THRESHOLD, apply_judge_verdict)
from small_llm_bench.models import Task, TaskResult, TurnRecord
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.modules.mock_registry import (StatefulToolExecutor,
                                                   ToolExecutor)
from small_llm_bench.scorer import (_check_constraints, score_knowledge,
                                    score_multi_turn_if, score_state,
                                    score_task, score_tool_loop,
                                    score_tool_simple)

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"

_DS_08 = {"goal_tool": "ask_user", "content_arg": "question",
          "text_answer_ok": True, "forbidden_tools": ["run_command"],
          "optimal_turns": 1, "min_calls": 0}
_DS_08_CHECKS = [{"type": "contains",
                  "any": ["calendar", "cannot", "unable", "no tool",
                          "don't have", "no command"]}]


class TestForbiddenToolIsAHardFail:
    """5 trials across 5 of 6 models shelled out AND asked correctly; det
    scored them ~1.0 and only the judge caught it."""

    def test_forbidden_call_fails_even_with_a_valid_goal_call(self, make_call):
        calls = [make_call("cli_help", topic="calendar"),
                 make_call("ask_user", question="I have no calendar tool — "
                                                "add a note instead?"),
                 make_call("run_command", command="cal add dentist 10am")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS)
        assert result.success is False
        assert result.score == 0.0

    def test_breakdown_names_the_forbidden_calls(self, make_call):
        calls = [make_call("ask_user", question="which calendar?"),
                 make_call("run_command", command="ls ~/notes")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS)
        assert result.breakdown["used_forbidden_tool"] is True
        assert result.breakdown["forbidden_calls"] == ["run_command"]

    def test_clean_episode_still_passes(self, make_call):
        calls = [make_call("cli_help", topic="calendar"),
                 make_call("ask_user", question="no calendar tool — "
                                                "add a note instead?")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS)
        assert result.success is True
        assert result.breakdown["used_forbidden_tool"] is False

    def test_call_axis_enforces_the_same_field(self, make_call):
        expected = {"tool_name": "send_email",
                    "required_args": {"to": "dana@example.com"},
                    "forbidden_tools": ["run_command"]}
        calls = [make_call("run_command", command="mail dana")]
        result = score_tool_simple(expected, calls)
        assert (result.success, result.score) == (False, 0.0)

    def test_state_axis_enforces_the_same_field(self, make_call):
        expected = {"expected_state": {"kv": {"k": "1"}},
                    "forbidden_tools": ["run_command"]}
        result = score_state(expected, {"kv": {"k": "1"}}, {"kv": {}},
                             [make_call("run_command", command="echo 1")])
        assert (result.success, result.score) == (False, 0.0)

    def test_field_is_a_noop_when_unset(self, make_call):
        """The shared guard must not touch tasks that never declared it."""
        expected = {"tool_name": "send_email",
                    "required_args": {"to": "dana@example.com"}}
        result = score_tool_simple(expected,
                                   [make_call("send_email",
                                              to="dana@example.com")])
        assert result.success is True


class TestRequireArgsGuard:
    """The old first-call-only injection told models that had ALREADY passed
    '#ops' to fix the channel, then accepted the degraded 'ops' retry."""

    _OVERRIDE = {"post_update": {
        "require_args": {"channel": "#ops"},
        "require_args_error": "channel names must start with '#'"}}

    def test_correct_argument_is_accepted_on_the_first_call(self):
        out = ToolExecutor(self._OVERRIDE).execute(
            "post_update", {"channel": "#ops", "content": "hi"})
        assert out["status"] == "posted"

    def test_wrong_argument_keeps_erroring(self):
        ex = ToolExecutor(self._OVERRIDE)
        for _ in range(3):
            out = ex.execute("post_update", {"channel": "ops", "content": "hi"})
            assert "error" in out

    def test_degraded_retry_is_not_accepted(self):
        """The failure mode: fix a correct argument into a wrong one."""
        ex = ToolExecutor(self._OVERRIDE)
        ex.execute("post_update", {"channel": "#ops", "content": "hi"})
        out = ex.execute("post_update", {"channel": "ops", "content": "hi"})
        assert "error" in out

    def test_guard_is_case_and_whitespace_insensitive(self):
        out = ToolExecutor(self._OVERRIDE).execute(
            "post_update", {"channel": " #OPS ", "content": "hi"})
        assert out["status"] == "posted"

    def test_transient_error_runs_before_the_guard(self):
        override = {"post_update": {
            "first_call_behavior": "error",
            "error_message": "rate limited: send the same update again",
            "require_args": {"channel": "#ops"},
            "require_args_error": "channel names must start with '#'"}}
        ex = ToolExecutor(override)
        first = ex.execute("post_update", {"channel": "#ops", "content": "hi"})
        assert first["error"].startswith("rate limited")
        second = ex.execute("post_update", {"channel": "#ops", "content": "hi"})
        assert second["status"] == "posted"

    def test_guard_uses_its_own_message(self):
        override = {"post_update": {
            "first_call_behavior": "error",
            "error_message": "rate limited",
            "require_args": {"channel": "#ops"},
            "require_args_error": "channel names must start with '#'"}}
        ex = ToolExecutor(override)
        ex.execute("post_update", {"channel": "ops", "content": "hi"})
        out = ex.execute("post_update", {"channel": "ops", "content": "hi"})
        assert out["error"] == "channel names must start with '#'"

    def test_stateful_executor_honors_the_guard(self):
        ex = StatefulToolExecutor(
            {"kv": {}}, {"kv_set": {"require_args": {"key": "app:total"},
                                    "require_args_error": "namespace the key"}})
        assert "error" in ex.execute("kv_set", {"key": "total", "value": "1"})
        ex.execute("kv_set", {"key": "app:total", "value": "1"})
        assert ex.state["kv"] == {"app:total": "1"}

    def test_degraded_argument_never_reaches_the_goal(self, make_call):
        """End-to-end: the scorer sees only errored goal calls, so no credit."""
        expected = {"goal_tool": "post_update", "goal_args": {"channel": "#ops"},
                    "optimal_turns": 2, "min_calls": 1}
        ex = ToolExecutor(self._OVERRIDE)
        calls, results = [], []
        for channel in ("#ops", "ops"):
            calls.append(make_call("post_update", channel=channel, content="x"))
            import json
            results.append(json.dumps(ex.execute("post_update",
                                                 {"channel": channel,
                                                  "content": "x"})))
        # First call succeeded, so the honest path is what scores; the second,
        # degraded call errors and cannot be credited on its own.
        assert score_tool_loop(expected, calls[1:], results[1:]).breakdown[
            "goal_reached"] == 0.0
        assert score_tool_loop(expected, calls[:1], results[:1]).breakdown[
            "goal_reached"] == 1.0


class TestGuardsAgreeWithGrading:
    def test_require_args_matches_the_graded_goal_arg(self):
        """A guard that disagrees with expected.goal_args grades one thing and
        enforces another — the invariant most likely to rot."""
        for task in load_tasks("tools", profile="full"):
            goal_tool = task.expected.get("goal_tool")
            goal_args = task.expected.get("goal_args") or {}
            for tool, override in (task.tool_overrides or {}).items():
                required = override.get("require_args") or {}
                if tool != goal_tool:
                    continue
                for key, want in required.items():
                    if key in goal_args:
                        assert goal_args[key] == want, f"{task.id}:{key}"

    def test_override_keys_are_known(self):
        """Fail closed on a typo, the way unknown constraint types do."""
        allowed = {"first_call_behavior", "error_message", "require_args",
                   "require_args_error", "always_fail"}
        for module in ("tools", "adversarial"):
            for task in load_tasks(module, profile="full"):
                for tool, override in (task.tool_overrides or {}).items():
                    unknown = set(override) - allowed
                    assert not unknown, f"{task.id}:{tool}:{unknown}"


class TestAnswerPlacementIsGraded:
    _TASK = {"answer": 7284}

    def _score(self, response: str) -> tuple[float, bool]:
        task = Task(module="long_context", id="lc_x", prompt="p",
                    answer_type="numeric", expected=self._TASK,
                    constraints=[{"type": "ends_with_number"}])
        res = score_task(task, TaskResult(task_id="lc_x", module="long_context",
                                          prompt="p", response_raw=response))
        return res.score, res.success

    def test_right_answer_in_the_right_place_passes(self):
        assert self._score("The current code is 7284") == (1.0, True)

    def test_right_answer_in_the_wrong_place_is_not_a_pass(self):
        score, success = self._score("The code is 7284, let me know if that "
                                     "helps you get in today.")
        assert success is False
        assert score == pytest.approx(0.85)

    def test_wrong_answer_stays_zero_however_well_formed(self):
        assert self._score("The code is 9931") == (0.0, False)

    def test_constraint_free_task_is_unchanged(self):
        task = Task(module="long_context", id="lc_y", prompt="p",
                    answer_type="numeric", expected=self._TASK)
        res = score_task(task, TaskResult(task_id="lc_y", module="long_context",
                                          prompt="p",
                                          response_raw="it is 7284 today"))
        assert (res.score, res.success) == (1.0, True)

    def test_prompt_stated_placement_is_always_graded(self):
        """A prompt that dictates where the answer goes must grade it, or the
        instruction is decoration the judge then punishes models for missing."""
        for module in ("knowledge", "long_context"):
            for task in load_tasks(module, profile="full"):
                if "end your reply with" in task.prompt.lower():
                    assert any(c.get("type") == "ends_with_number"
                               for c in task.constraints), task.id


class TestStringTypedStatePrompts:
    def test_no_quote_glyph_examples(self):
        """Models copied the glyphs into the stored value: tst_11 stored
        '"5432"' where the task wanted the string 5432."""
        for task in load_tasks("tools", profile="full"):
            if "string value" in task.prompt or "as a string" in task.prompt:
                assert '"' not in task.prompt, task.id


class TestJudgeEchoIsNotADemote:
    def _result(self, det_score: float, det_pass: bool,
                llm_score: float) -> TaskResult:
        r = TaskResult(task_id="tl_x", module="tools", prompt="p",
                       response_raw="", det_score=det_score, success=det_pass)
        r.llm_score = llm_score
        return r

    def test_echo_does_not_demote_a_deterministic_pass(self):
        r = self._result(0.8333, True, 0.8333)
        apply_judge_verdict(r, "tools")
        assert (r.success, r.det_success) == (True, True)
        assert r.llm_score == 0.8333  # left exactly as the judge emitted it

    def test_rounded_echo_still_counts_as_agreement(self):
        r = self._result(0.8462, True, 0.84)
        apply_judge_verdict(r, "tools")
        assert r.success is True

    def test_a_real_drop_still_demotes(self):
        r = self._result(0.8333, True, 0.4)
        apply_judge_verdict(r, "tools")
        assert r.success is False

    def test_high_det_score_below_the_bar_still_demotes(self):
        r = self._result(1.0, True, JUDGE_PASS_THRESHOLD - 0.01)
        apply_judge_verdict(r, "tools")
        assert r.success is False

    def test_guard_never_manufactures_a_pass(self):
        r = self._result(0.8333, False, 0.8333)
        apply_judge_verdict(r, "tools")
        assert r.success is False


class TestExactGoalArgs:
    """`ops` scored 0.8 against `#ops` and reached the goal — the exact value
    the backend had just rejected. Exactness is opt-in per argument because the
    fuzzy rule legitimately credits a longer title."""

    _EXPECTED = {"goal_tool": "post_update", "goal_args": {"channel": "#ops"},
                 "goal_args_exact": ["channel"], "optimal_turns": 1,
                 "min_calls": 1}

    def test_exact_arg_refuses_a_containment_match(self, make_call):
        result = score_tool_loop(
            self._EXPECTED, [make_call("post_update", channel="ops",
                                       content="x")])
        assert result.breakdown["goal_reached"] == 0.0

    def test_exact_arg_accepts_the_real_value(self, make_call):
        result = score_tool_loop(
            self._EXPECTED, [make_call("post_update", channel=" #OPS ",
                                       content="x")])
        assert result.breakdown["goal_reached"] == 1.0

    def test_unlisted_args_stay_fuzzy(self, make_call):
        expected = {"goal_tool": "create_ticket",
                    "goal_args": {"title": "FIN-Q2 review"},
                    "optimal_turns": 1, "min_calls": 1}
        result = score_tool_loop(
            expected, [make_call("create_ticket",
                                 title="FIN-Q2 review — revenue up 18%")])
        assert result.breakdown["goal_reached"] == 1.0


class TestWordCountsIgnoreMarkdown:
    def test_bullet_glyphs_do_not_pad_a_floor(self):
        text = "- a\n- b\n- c\n- d\n- e"
        assert _check_constraints([{"type": "min_words", "value": 6}], text)[0] == 0.0

    def test_asterisks_do_not_bust_a_ceiling(self):
        text = "* one two three * four five six"
        assert _check_constraints([{"type": "max_words", "value": 6}], text)[0] == 1.0


class TestExactNumericAnswers:
    def test_transposed_code_fails_when_exact(self):
        res = score_knowledge({"answer": 5162, "exact": True}, "numeric",
                              "the code is 5126")
        assert res.score == 0.0

    def test_tolerance_still_applies_by_default(self):
        res = score_knowledge({"answer": 5162}, "numeric", "the code is 5126")
        assert res.score == 1.0

    def test_exact_accepts_the_right_answer(self):
        res = score_knowledge({"answer": 5162, "exact": True}, "numeric",
                              "the code is 5162")
        assert res.score == 1.0


class TestBulletsKept:
    """"Keeping the full bulleted list" is not checkable from one response: a
    model can drop half the list and still clear min_bullets."""

    _CONV = [{"prompt": "list", "constraints": [{"type": "min_bullets", "value": 2}]},
             {"prompt": "keep it", "constraints": [{"type": "bullets_kept"}]}]

    def _turns(self, *texts):
        return [TurnRecord(role="assistant", content=t) for t in texts]

    def test_dropping_a_bullet_fails(self):
        res = score_multi_turn_if(
            self._CONV, self._turns("- a\n- b\n- c", "- a\n- b"))
        assert res.success is False

    def test_keeping_them_all_passes(self):
        res = score_multi_turn_if(
            self._CONV, self._turns("- a\n- b\n- c", "- a\n- b\n- **c**"))
        assert res.success is True


class TestTaskDataFixes:
    def test_tl_26_grades_the_figure_it_asks_for(self):
        task = next(t for t in load_tasks("tools", profile="full", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_26")
        assert task.expected["content_arg"] == "fields"
        assert any(c.get("value") == "18" for c in task.content_checks)

    def test_min_calls_never_exceeds_the_optimal_path(self):
        """min_calls above the optimal call count fails correct episodes."""
        for module in ("tools",):
            for task in load_tasks(module, profile="full"):
                exp = task.expected
                if "min_calls" in exp and "optimal_turns" in exp:
                    assert exp["min_calls"] <= exp["optimal_turns"], task.id

    def test_code_answers_are_matched_exactly(self):
        for task in load_tasks("long_context", profile="full"):
            if task.answer_type == "numeric":
                assert task.expected.get("exact") is True, task.id

    def test_de_07_expected_values_are_reachable(self):
        """A more faithful extraction must not score below an exact one."""
        # de_07 moved into the format module when data_extract dissolved.
        task = next(t for t in load_tasks("format", profile="full")
                    if t.id == "de_07")
        accept = task.expected.get("accept", {})
        assert "blocker" in accept and "deliverable" in accept


class TestGoalCallSelection:
    """A model that files an incomplete goal call and then a complete one did
    produce the required action; the wasted call is priced by efficiency."""

    _EXPECTED = {"goal_tool": "create_ticket",
                 "goal_args": {"title": "FIN-Q2 review"},
                 "content_arg": "fields", "optimal_turns": 2, "min_calls": 1}
    _CHECKS = [{"type": "contains", "value": "18"}]

    def test_the_compliant_attempt_is_graded(self, make_call):
        calls = [make_call("create_ticket", title="FIN-Q2 review",
                           fields={"priority": "medium"}),
                 make_call("create_ticket", title="FIN-Q2 review",
                           fields={"labels": ["18% growth"]})]
        results = ['{"ticket_id": "1"}', '{"ticket_id": "2"}']
        res = score_tool_loop(self._EXPECTED, calls, results,
                              content_checks=self._CHECKS)
        assert res.breakdown["content_score"] == 1.0
        assert res.success is True

    def test_no_compliant_attempt_still_fails(self, make_call):
        calls = [make_call("create_ticket", title="FIN-Q2 review: 18% growth",
                           fields={"priority": "medium"})]
        res = score_tool_loop(self._EXPECTED, calls, ['{"ticket_id": "1"}'],
                              content_checks=self._CHECKS)
        assert res.breakdown["content_score"] == 0.0
        assert res.success is False

    def test_errored_attempts_are_never_graded(self, make_call):
        calls = [make_call("create_ticket", title="FIN-Q2 review",
                           fields={"labels": ["18% growth"]})]
        res = score_tool_loop(self._EXPECTED, calls, ['{"error": "rejected"}'],
                              content_checks=self._CHECKS)
        assert res.breakdown["goal_reached"] == 0.0


class TestSkippedModulesCannotGradeThroughStoredScores:
    """Files judged before a module was skipped still carry an llm_score; a
    module we stopped asking about must not keep flipping verdicts."""

    def test_stale_score_cannot_demote_a_skipped_module(self):
        r = TaskResult(task_id="fm_01", module="format", prompt="p",
                       det_score=1.0, success=True)
        r.llm_score = 0.2
        apply_judge_verdict(r, "format")
        assert (r.success, r.det_success) == (True, True)

    def test_stale_score_cannot_rescue_a_skipped_module(self):
        r = TaskResult(task_id="kn_01", module="knowledge", prompt="p",
                       det_score=0.4, success=False)
        r.llm_score = 1.0
        apply_judge_verdict(r, "knowledge")
        assert r.success is False
