"""Regression tests for the v0.6 bank changes.

Covers the tasks added in v0.6 and the ds_08 text-answer path, where a plain
prose reply is an equally valid way to report that no available tool fits.
"""

from __future__ import annotations

from small_llm_bench.models import TaskResult, ToolCall, TurnRecord
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.scorer import score_task, score_tool_loop
from pathlib import Path

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"

# --- ds_08: no calendar tool exists; prose or ask_user both count -------------

_DS_08 = {"goal_tool": "ask_user", "content_arg": "question",
          "text_answer_ok": True, "forbidden_tools": ["run_command"],
          "optimal_turns": 1, "min_calls": 0}
_DS_08_CHECKS = [{"type": "contains",
                  "any": ["calendar", "cannot", "unable", "no tool",
                          "don't have", "no command"]}]
_REFUSAL = ("I don't have a calendar tool available. The topics I can reach are "
            "deployer, notes and ticket. How would you like to proceed?")


class TestAskAxisTextAnswer:
    def test_ask_user_call_still_passes(self, make_call):
        calls = [make_call("cli_help", topic="calendar"),
                 make_call("ask_user", question="I have no calendar tool — "
                                                "should I add a note instead?")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS)
        assert result.success is True
        assert result.breakdown["answered_in_text"] is False

    def test_prose_refusal_after_probing_passes(self, make_call):
        calls = [make_call("cli_help", topic="calendar")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS,
                                 answer_text=_REFUSAL)
        assert result.success is True
        assert result.breakdown["answered_in_text"] is True

    def test_prose_refusal_without_probing_passes(self):
        """The tool list itself shows no calendar tool, so probing is optional."""
        result = score_tool_loop(_DS_08, [], content_checks=_DS_08_CHECKS,
                                 answer_text=_REFUSAL)
        assert result.success is True

    def test_shell_spelunking_fails_even_if_it_answers(self, make_call):
        calls = [make_call("cli_help", topic="calendar"),
                 make_call("run_command", command="which calendar")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS,
                                 answer_text=_REFUSAL)
        assert result.success is False
        assert result.breakdown["used_forbidden_tool"] is True

    def test_faking_the_write_fails(self, make_call):
        calls = [make_call("run_command", command="cal add dentist 10am")]
        result = score_tool_loop(
            _DS_08, calls, content_checks=_DS_08_CHECKS,
            answer_text="Done — added the dentist appointment to your calendar.")
        assert result.success is False

    def test_silence_fails(self, make_call):
        """Burning the turn budget without ever answering is not a pass."""
        calls = [make_call("cli_help", topic="calendar")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS,
                                 answer_text="")
        assert result.success is False

    def test_off_topic_answer_fails_content_checks(self, make_call):
        calls = [make_call("cli_help", topic="calendar")]
        result = score_tool_loop(_DS_08, calls, content_checks=_DS_08_CHECKS,
                                 answer_text="Sure, all set.")
        assert result.success is False

    def test_truncated_answer_does_not_count_as_answered(self, tasks_dir):
        task = next(t for t in load_tasks("tools", tasks_dir=RETIRED_BANK)
                    if t.id == "ds_11")
        result = TaskResult(
            task_id="ds_11", module="tools", prompt=task.prompt,
            turns=[TurnRecord(role="assistant", content=_REFUSAL)],
            truncated=True)
        assert score_task(task, result).success is False

    def test_task_yaml_wires_the_text_path(self, tasks_dir):
        task = next(t for t in load_tasks("tools", tasks_dir=RETIRED_BANK)
                    if t.id == "ds_11")
        assert task.expected["text_answer_ok"] is True
        assert task.expected["forbidden_tools"] == ["run_deploy"]

    def test_every_ask_task_accepts_prose_and_guards_the_guess(self, tasks_dir):
        """v0.8: asking is the capability; the mechanism is reported, not
        scored. Every ask_user task takes the prose path, and every one of them
        names the tool that would act on an unconfirmed guess as forbidden —
        otherwise the escape hatch would also excuse guessing."""
        for task in load_tasks("tools", tasks_dir=tasks_dir):
            # Only the tasks where asking IS the goal. Its contrast partner
            # (ds_15) expects the model to go look instead, and forbids asking.
            if task.expected.get("goal_tool") != "ask_user":
                continue
            assert task.expected["text_answer_ok"] is True, task.id
            assert task.expected["forbidden_tools"], task.id
            assert task.expected["min_calls"] == 0, task.id


class TestTextAnswerPathIsOptIn:
    def test_no_opt_in_means_prose_never_substitutes_for_the_goal(self, make_call):
        """tl_20-style task: an answer in text is not a posted update."""
        expected = {"goal_tool": "post_update", "goal_args": {"channel": "#ops"},
                    "content_arg": "content", "optimal_turns": 4, "min_calls": 4}
        calls = [make_call("read_file", path="/data/incident.txt")]
        result = score_tool_loop(expected, calls,
                                 content_checks=[{"type": "contains",
                                                  "value": "INC-4471"}],
                                 answer_text="INC-4471 was owned by Dana.")
        assert result.success is False
        assert result.breakdown["goal_reached"] == 0.0
        assert "answered_in_text" not in result.breakdown


# --- the five new v0.6 tasks are wired up as intended ------------------------

class TestNewTaskWiring:
    def test_bands_are_assigned_from_measurement(self, tasks_dir):
        """v0.13 retagged every band against the published thresholds.

        43 of 55 tasks were in the wrong one, and five carried `frontier`
        (<30% measured pass) while sitting between 0.77 and 1.00 — fm_22 was
        labelled frontier at a pass rate of exactly 1.00. Bands no longer set
        the ranking (see reporter.headline_overall, scheme "module"), so a
        wrong band is now a reporting error rather than a scoring one, but it
        is still a claim about difficulty and has to be earned.

        The frontier band is deliberately EMPTY: nothing in the bank has been
        measured below 30%. Filling it means writing a harder task, not
        relabelling an easier one.
        """
        modules = ("tools", "code", "knowledge", "format", "long_context",
                   "multi_turn_if", "adversarial")
        bands = {}
        for m in modules:
            for t in load_tasks(m, tasks_dir=tasks_dir):
                bands.setdefault(t.band, set()).add(t.id)
        assert set(bands) <= {"anchor", "mid", "hard", "frontier"}
        assert "frontier" not in bands
        # One labelled anchor per module, as the harness-health canary.
        assert len(bands["anchor"]) >= 5

    def test_lc_22_decoy_sits_after_the_real_binding(self, tasks_dir):
        from small_llm_bench.modules.long_context import build_haystack
        task = next(t for t in load_tasks("long_context", tasks_dir=RETIRED_BANK)
                    if t.id == "lc_22")
        doc = build_haystack(task.haystack)
        chain = task.haystack["chain"]
        depths = [doc.index(line) for line in chain]
        assert depths == sorted(depths), "chain must appear in definition order"
        assert task.expected["answer"] == 106

    def test_tl_23_every_stage_fails_once(self, tasks_dir):
        task = next(t for t in load_tasks("tools", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_23")
        assert set(task.tool_overrides) == {"read_file", "write_file",
                                            "post_update"}
        assert all(o["first_call_behavior"] == "error"
                   for o in task.tool_overrides.values())

    def test_tst_23_protected_kv_keys_are_in_the_goal_state(self, tasks_dir):
        task = next(t for t in load_tasks("tools", tasks_dir=tasks_dir)
                    if t.id == "tst_23")
        kv = task.expected["expected_state"]["kv"]
        assert kv["refund_policy"] == task.initial_state["kv"]["refund_policy"]
        assert kv["audit_lock"] == task.initial_state["kv"]["audit_lock"]
        # The seeded keys above stay quoted because the world seeded them that
        # way. The two the MODEL produces are plain integers since v0.12: the
        # prompt no longer asks for a string, because kv_set's schema declares
        # `value` as "any type" and both encodings are valid there. Demanding
        # one split the fleet by vendor rather than capability.
        assert (kv["shipped_count"], kv["refund_total"]) == (2, 32)


# --- judge rubric matches the scorer on the text-answer path -----------------

class TestJudgeTextAnswerNote:
    def _rendered(self, tasks_dir, task_id):
        """Look in the shipped bank first, then the v0.15 retirees.

        This class contrasts a task that opts into the text-answer path with
        one that does not, and the opting-in side (ds_11) left the bank with
        the rest of the discovery axis in v0.15.
        """
        from small_llm_bench.judge import render_judge_prompt
        task = next((t for d in (tasks_dir, RETIRED_BANK)
                     for t in load_tasks("tools", tasks_dir=d)
                     if t.id == task_id), None)
        assert task is not None, f"tools:{task_id} is in neither bank"
        result = TaskResult(task_id=task_id, module="tools",
                            prompt=task.prompt, expected=task.expected,
                            response_raw="I don't have a calendar tool.",
                            det_score=0.85)
        return render_judge_prompt("tools", [result])

    def test_ask_task_prompt_tells_the_judge_prose_is_correct(self, tasks_dir):
        """Without this the judge demotes a correct prose reply for not calling
        the goal_tool it can see in the expected blob."""
        assert "NOTE for this task" in self._rendered(tasks_dir, "ds_11")

    def test_tasks_without_the_opt_in_get_no_note(self, tasks_dir):
        """tl_23 wants an actual posted update — prose is not a substitute, and
        the judge must not be told otherwise."""
        from small_llm_bench.judge import render_judge_prompt
        task = next(t for t in load_tasks("tools", tasks_dir=RETIRED_BANK)
                    if t.id == "tl_23")
        result = TaskResult(task_id="tl_23", module="tools",
                            prompt=task.prompt, expected=task.expected,
                            response_raw="Posted.", det_score=0.5)
        assert "NOTE for this task" not in render_judge_prompt("tools",
                                                               [result])
