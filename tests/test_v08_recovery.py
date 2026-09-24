"""v0.8: recovery-aware scoring.

Covers the code execute-and-fix loop and its decay ladder, the reported-only
``tool_mechanism`` sub-score, context-overflow reporting, and the long-context
ladder up to 64k.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

# aliased: pytest tries to collect any module-level name starting with Test
from small_llm_bench.models import Task, TaskResult
from small_llm_bench.models import TestCase as CaseSpec
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.modules.code import CodeModule, failure_report
from small_llm_bench.modules.long_context import build_haystack, estimate_tokens
from small_llm_bench.reporter import recovery_stats, structured_call_rate
from small_llm_bench.runner import _is_context_overflow, _is_transient
from small_llm_bench.scorer import _parse_outcomes, score_code, score_tool_loop

SANDBOX = {"backend": "rlimit", "allow_unsandboxed": True}
_CASES = [CaseSpec(args=[2], expected=4), CaseSpec(args=[3], expected=9)]
_GOOD = "```python\ndef sq(n):\n    return n * n\n```"
_BAD = "```python\ndef sq(n):\n    return n + n\n```"
_HALF = "```python\ndef sq(n):\n    return 4 if n == 2 else 0\n```"


class StubClient:
    """A ChatClient stand-in that replays a fixed list of replies."""

    save_responses = False

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def chat(self, messages, tools=None, max_tokens=None):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0) if self.replies else self.replies[-1]
        return {"choices": [{"message": {"content": reply},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 20}}


def _code_task(**kw) -> Task:
    base = dict(id="t", module="code", prompt="Write sq(n).",
                function_name="sq", test_cases=_CASES)
    return Task(**{**base, **kw})


def _run(task: Task, replies: list[str], sandbox=SANDBOX) -> TaskResult:
    return asyncio.run(CodeModule().run_task(StubClient(replies), task,
                                             sandbox=sandbox))


# --- the decay ladder --------------------------------------------------------

class TestRepairDecay:
    @pytest.mark.parametrize("attempts, expected", [(1, 1.0), (2, 0.85), (3, 0.70)])
    def test_passing_scores_by_attempt(self, attempts, expected):
        result = score_code(_GOOD, "sq", _CASES, attempts_used=attempts, **SANDBOX)
        assert result.score == expected

    @pytest.mark.parametrize("attempts", [1, 2, 3, 9])
    def test_a_pass_is_a_pass_however_many_attempts(self, attempts):
        """The decay lives in the soft score only. If success tracked the score
        instead, every recovered task would read as a failure in pass^k and the
        repair loop would be measuring nothing."""
        result = score_code(_GOOD, "sq", _CASES, attempts_used=attempts, **SANDBOX)
        assert result.success is True

    def test_decay_bottoms_out_rather_than_running_negative(self):
        result = score_code(_GOOD, "sq", _CASES, attempts_used=99, **SANDBOX)
        assert result.score == 0.70

    def test_partial_credit_is_capped_below_the_worst_pass(self):
        """Half the cases passing must not approach a three-attempt pass."""
        result = score_code(_HALF, "sq", _CASES, **SANDBOX)
        assert result.score == 0.5
        assert result.success is False

    def test_breakdown_carries_the_attempt_provenance(self):
        first = score_code(_GOOD, "sq", _CASES, attempts_used=1, **SANDBOX)
        later = score_code(_GOOD, "sq", _CASES, attempts_used=2, **SANDBOX)
        assert first.breakdown["one_shot"] is True
        assert later.breakdown["one_shot"] is False
        assert later.breakdown["attempts_used"] == 2


# --- the loop ----------------------------------------------------------------

class TestRepairLoop:
    def test_default_is_still_one_shot(self):
        result = _run(_code_task(), [_BAD, _GOOD])
        assert result.attempts_used == 1
        assert result.response_raw == _BAD

    def test_second_attempt_sees_the_failure_and_fixes_it(self):
        client = StubClient([_BAD, _GOOD])
        task = _code_task(repair_attempts=3)
        result = asyncio.run(CodeModule().run_task(client, task, sandbox=SANDBOX))
        assert result.attempts_used == 2
        assert "sq(3) -> returned 6, which is wrong" in client.prompts[1]
        assert score_code(result.response_raw, "sq", _CASES,
                          attempts_used=result.attempts_used, **SANDBOX).score == 0.85

    def test_stops_as_soon_as_it_passes(self):
        result = _run(_code_task(repair_attempts=3), [_GOOD, _BAD, _BAD])
        assert result.attempts_used == 1

    def test_repeating_the_same_code_ends_the_loop(self):
        """A model that resubmits its own output has nothing left to give; the
        remaining attempts would only cost tokens."""
        result = _run(_code_task(repair_attempts=3), [_BAD, _BAD, _GOOD])
        assert result.attempts_used == 2
        assert result.response_raw == _BAD

    def test_tokens_are_summed_across_attempts(self):
        result = _run(_code_task(repair_attempts=3), [_BAD, _GOOD])
        assert result.completion_tokens == 40

    def test_no_sandbox_means_no_repair(self):
        result = _run(_code_task(repair_attempts=3), [_BAD, _GOOD], sandbox=None)
        assert result.attempts_used == 1

    def test_transcript_keeps_the_feedback_turn(self):
        result = _run(_code_task(repair_attempts=2), [_BAD, _GOOD])
        assert [t.role for t in result.turns] == ["assistant", "user", "assistant"]


class TestFailureReport:
    def test_syntax_error_is_named(self):
        report = failure_report(_code_task(), "def sq(n:\n    pass", SANDBOX)
        assert "Syntax error" in report

    def test_missing_code_block_is_named(self):
        report = failure_report(_code_task(), "", SANDBOX)
        assert "No Python code block" in report

    def test_import_error_shows_the_traceback_tail(self):
        report = failure_report(_code_task(), "import nope_not_a_module\n"
                                "def sq(n): return n", SANDBOX)
        assert "Import/setup error" in report
        assert "nope_not_a_module" in report

    def test_wrong_answers_show_the_input_and_what_came_back(self):
        report = failure_report(_code_task(), "def sq(n):\n    return n + n",
                                SANDBOX)
        assert "sq(2)" not in report          # 2+2 == 4, passes
        assert "sq(3) -> returned 6, which is wrong" in report

    def test_the_report_never_names_the_expected_value(self):
        """The report is a failing-test signal, not the answer key.

        cd_15 has five test cases and three repair attempts, so printing
        `expected` let a model pass by transcribing the reported pairs rather
        than fixing the function — and _REPAIR_DECAY prices attempts, not
        leakage.
        """
        report = failure_report(_code_task(), "def sq(n):\n    return n + n",
                                SANDBOX)
        assert "expected" not in report
        assert "9" not in report

    def test_raised_exceptions_are_shown_as_raised(self):
        report = failure_report(_code_task(), "def sq(n):\n    return n / 0",
                                SANDBOX)
        assert "raised ZeroDivisionError" in report

    def test_a_pass_produces_nothing_to_say(self):
        assert failure_report(_code_task(), "def sq(n):\n    return n * n",
                              SANDBOX) is None


class TestHarnessOutput:
    def test_parses_the_object_form(self):
        parsed = _parse_outcomes('[{"ok": true, "got": "4", "error": null}]')
        assert parsed == [{"ok": True, "got": "4", "error": None}]

    @pytest.mark.parametrize("stdout", ["", "not json", "[1, 2]", '{"ok": true}'])
    def test_unusable_output_is_none_not_a_crash(self, stdout):
        assert _parse_outcomes(stdout) is None


# --- ask_user: prose passes, mechanism is reported only ----------------------

_ASK = {"goal_tool": "ask_user", "content_arg": "question",
        "text_answer_ok": True, "forbidden_tools": ["run_deploy"],
        "optimal_turns": 1, "min_calls": 0}
_ASK_CHECKS = [{"type": "contains", "any": ["which", "env"]}]


class TestAskMechanismIsReportedNotScored:
    def test_prose_question_passes(self, make_call):
        result = score_tool_loop(_ASK, [], content_checks=_ASK_CHECKS,
                                 answer_text="Which environment should I deploy to?")
        assert result.success is True
        assert result.breakdown["tool_mechanism"] == 0.0

    def test_structured_question_passes_identically(self, make_call):
        call = make_call("ask_user", question="Which environment?")
        result = score_tool_loop(_ASK, [call], content_checks=_ASK_CHECKS)
        assert result.success is True
        assert result.breakdown["tool_mechanism"] == 1.0

    def test_both_paths_score_the_same(self, make_call):
        prose = score_tool_loop(_ASK, [], content_checks=_ASK_CHECKS,
                                answer_text="Which environment should I deploy to?")
        tool = score_tool_loop(_ASK, [make_call("ask_user",
                                                question="Which environment?")],
                               content_checks=_ASK_CHECKS)
        assert prose.score == tool.score

    def test_acting_on_the_guess_still_fails(self, make_call):
        """The escape hatch covers asking, not deploying to prod on a hunch."""
        call = make_call("run_deploy", service="reports", env="prod")
        result = score_tool_loop(_ASK, [call], content_checks=_ASK_CHECKS,
                                 answer_text="Deployed to prod.")
        assert result.success is False

    def test_tasks_without_the_opt_in_report_no_mechanism(self, make_call):
        expected = {"goal_tool": "post_update", "optimal_turns": 1, "min_calls": 1}
        result = score_tool_loop(expected, [make_call("post_update",
                                                      channel="#ops")])
        assert "tool_mechanism" not in result.breakdown


# --- context overflow --------------------------------------------------------

def _status_error(body: str, code: int = 400) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://x/chat/completions")
    response = httpx.Response(code, text=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class TestContextOverflow:
    def test_detects_the_usual_wording(self):
        exc = _status_error('{"error": "This model\'s maximum context length '
                            'is 8192 tokens"}')
        assert _is_context_overflow(exc) is True

    def test_is_not_treated_as_infra(self):
        """Excluding it would hide a real limitation; it stays a scored zero."""
        exc = _status_error('{"error": "prompt is too long"}')
        assert _is_transient(exc) is False

    def test_an_unrelated_400_is_not_overflow(self):
        assert _is_context_overflow(_status_error('{"error": "bad model"}')) is False

    def test_a_timeout_is_not_overflow(self):
        assert _is_context_overflow(httpx.TimeoutException("slow")) is False


# --- the long-context ladder -------------------------------------------------

class TestLongContextLadder:
    def test_the_ladder_spans_its_declared_range(self, tasks_dir):
        """The ladder tops out at 32k as of v0.15.

        lc_64 (the 64k rung) went in v0.10: 130 s per model to separate exactly
        one 2.6B model, at zero discrimination. lc_45 (48k, a five-link chain)
        went in the second v0.15 cut — it read +0.78 on ten models but fell to
        +0.58 once the panel included two MoEs, and at 57 s/trial it was the
        module's most expensive item.

        Losing it costs no measured length coverage: qwen3.5-4b cleared the
        48k chain 3/3, so length was never what that rung was testing. What it
        does mean is that nothing in the bank now runs past 32k, so a context
        regression above that would go unseen.
        """
        tasks = load_tasks("long_context", tasks_dir=tasks_dir)
        sizes = sorted(t.haystack["filler_tokens"] for t in tasks)
        assert max(sizes) == 32000
        assert {4000, 16000, 32000} <= set(sizes)

    def test_every_haystack_lands_near_its_declared_size(self, tasks_dir):
        for task in load_tasks("long_context", tasks_dir=tasks_dir):
            target = task.haystack["filler_tokens"]
            actual = estimate_tokens(build_haystack(task.haystack))
            assert 0.8 <= actual / target <= 1.2, task.id

    def test_every_needle_is_present_exactly_once(self, tasks_dir):
        """Pinned across every live needle-based task, not one id.

        This named lc_31 until v1.0 and broke when lc_31 was retired. The
        invariant is about `build_haystack`, not about one task, so asserting
        it over whatever the bank currently holds keeps the coverage and
        survives the next retirement. multi_hop tasks carry a `chain` instead
        of a needle and are skipped.
        """
        seen = 0
        for task in load_tasks("long_context", tasks_dir=tasks_dir):
            needle = task.haystack.get("needle")
            if not needle:
                continue
            seen += 1
            doc = build_haystack(task.haystack, salt=task.id)
            assert doc.count(needle) == 1, task.id
        assert seen >= 2, "expected at least two needle-based tasks"

    def test_distractors_all_survive_into_the_document(self, tasks_dir):
        seen = 0
        for task in load_tasks("long_context", tasks_dir=tasks_dir):
            distractors = task.haystack.get("distractors") or []
            if not distractors:
                continue
            seen += 1
            doc = build_haystack(task.haystack, salt=task.id)
            for distractor in distractors:
                assert distractor in doc, f"{task.id}: {distractor[:60]}"
        assert seen >= 2, "expected at least two tasks with distractors"


# --- reporting ---------------------------------------------------------------

class TestRecoveryReporting:
    def _code(self, success: bool, attempts: int) -> TaskResult:
        return TaskResult(task_id="c", module="code", prompt="p",
                          det_success=success, attempts_used=attempts)

    def test_splits_one_shot_from_recovered(self):
        stats = recovery_stats([self._code(True, 1), self._code(True, 3),
                                self._code(False, 3)])
        assert stats == {"solved": 2, "one_shot": 1, "recovered": 1}

    def test_ignores_other_modules(self):
        other = TaskResult(task_id="x", module="format", prompt="p",
                           det_success=True)
        assert recovery_stats([other])["solved"] == 0

    def test_structured_call_rate_counts_only_opt_in_tasks(self):
        rows = [
            TaskResult(task_id="a", module="tools", prompt="p",
                       det_breakdown={"tool_mechanism": 1.0}),
            TaskResult(task_id="b", module="tools", prompt="p",
                       det_breakdown={"tool_mechanism": 0.0}),
            TaskResult(task_id="c", module="tools", prompt="p",
                       det_breakdown={"goal_reached": 1.0}),
        ]
        assert structured_call_rate(rows) == (1, 2)
