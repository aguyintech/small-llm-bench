"""Tests for all deterministic scorer functions."""

from __future__ import annotations

import pytest

from small_llm_bench.models import Task, TaskResult
from small_llm_bench.models import TestCase as CodeCase
from small_llm_bench.modules.base import hit_length_cap
from small_llm_bench.scorer import (BAND_WEIGHTS, band_weighted, extract_code,
                                    score_code, score_format, score_knowledge,
                                    score_state, score_task, score_tool_loop,
                                    score_tool_simple, repetition_ratio, loop_ratio,
                                    truncation_class, counted, scorable,
                                    _REPETITION_THRESHOLD)

SIMPLE_EXPECTED = {
    "tool_name": "get_weather",
    "required_args": {"city": "Tokyo"},
}

LOOP_EXPECTED = {
    "goal_tool": "send_message",
    "goal_args": {"recipient": "alice@example.com"},
    "optimal_turns": 2,
    "min_calls": 2,
}


class TestScoreToolSimple:
    def test_perfect_call(self, make_call):
        result = score_tool_simple(SIMPLE_EXPECTED, [make_call("get_weather", city="Tokyo")])
        assert result.score == 1.0
        assert result.breakdown["tool_name_correct"] == 1.0

    def test_no_call(self, make_call):
        result = score_tool_simple(SIMPLE_EXPECTED, [])
        assert result.score == 0.0
        assert result.breakdown["tool_called"] == 0.0

    def test_wrong_tool_name(self, make_call):
        result = score_tool_simple(SIMPLE_EXPECTED, [make_call("search_web", query="Tokyo")])
        assert result.breakdown["tool_name_correct"] == 0.0
        assert result.score < 1.0

    def test_missing_required_arg(self, make_call):
        result = score_tool_simple(SIMPLE_EXPECTED, [make_call("get_weather")])
        assert result.breakdown["required_args_present"] == 0.0

    def test_fuzzy_arg_value_case_insensitive(self, make_call):
        result = score_tool_simple(SIMPLE_EXPECTED, [make_call("get_weather", city="tokyo")])
        assert result.score == 1.0


class TestScoreToolSimpleBFCL:
    def test_no_call_correct(self, make_call):
        result = score_tool_simple({"no_call": True}, [],
                                   answer_text="A hash map stores key-value pairs.")
        assert result.score == 1.0

    def test_no_call_violated(self, make_call):
        result = score_tool_simple({"no_call": True},
                                   [make_call("search_web", query="x")],
                                   answer_text="Looking that up now.")
        assert result.score == 0.0

    def test_no_call_but_empty_answer_fails(self, make_call):
        """Correctly avoiding the tool call is not enough if the model then
        gives the user nothing back (regression: a model burned its full token
        budget on hidden reasoning and returned empty content, det scorer
        still marked it a pass since no tool was called)."""
        result = score_tool_simple({"no_call": True}, [], answer_text="")
        assert result.score == 0.0
        assert result.success is False
        assert result.breakdown["answered"] is False

    def test_parallel_both_calls_correct(self, make_call):
        specs = [
            {"tool_name": "get_weather", "required_args": {"city": "Tokyo"}},
            {"tool_name": "get_weather", "required_args": {"city": "Berlin"}},
        ]
        calls = [make_call("get_weather", city="Tokyo"),
                 make_call("get_weather", city="Berlin")]
        result = score_tool_simple({}, calls, parallel=specs)
        assert result.score == 1.0

    def test_parallel_missing_one_call(self, make_call):
        specs = [
            {"tool_name": "get_weather", "required_args": {"city": "Tokyo"}},
            {"tool_name": "get_weather", "required_args": {"city": "Berlin"}},
        ]
        result = score_tool_simple({}, [make_call("get_weather", city="Tokyo")],
                                   parallel=specs)
        assert 0.0 < result.score < 1.0

    def test_parallel_penalizes_extra_calls(self, make_call):
        specs = [{"tool_name": "get_weather", "required_args": {"city": "Tokyo"}}]
        calls = [make_call("get_weather", city="Tokyo"),
                 make_call("get_weather", city="Paris"),
                 make_call("get_weather", city="London")]
        result = score_tool_simple({}, calls, parallel=specs)
        assert result.score < 1.0


class TestScoreToolLoopContent:
    EXPECTED = {
        "goal_tool": "write_file",
        "goal_args": {"path": "/data/stats.json"},
        "content_arg": "content",
        "optimal_turns": 3,
        "min_calls": 3,
    }
    CHECKS = [
        {"type": "valid_json"},
        {"type": "json_has_keys", "keys": ["sum", "count"]},
        {"type": "json_value", "path": "sum", "value": 100},
    ]

    def test_content_checks_all_pass(self, make_call):
        calls = [
            make_call("read_file", path="/data/numbers.txt"),
            make_call("calculate", expression="10+20+30+40"),
            make_call("write_file", path="/data/stats.json",
                      content='{"sum": 100, "count": 4}'),
        ]
        result = score_tool_loop(self.EXPECTED, calls, content_checks=self.CHECKS)
        assert result.breakdown["content_score"] == 1.0
        assert result.score == 1.0

    def test_content_checks_partial(self, make_call):
        calls = [
            make_call("read_file", path="/data/numbers.txt"),
            make_call("calculate", expression="x"),
            make_call("write_file", path="/data/stats.json",
                      content='{"sum": 999}'),
        ]
        result = score_tool_loop(self.EXPECTED, calls, content_checks=self.CHECKS)
        assert 0.0 < result.breakdown["content_score"] < 1.0
        assert result.score < 1.0

    def test_content_checks_invalid_json(self, make_call):
        calls = [
            make_call("read_file", path="/data/numbers.txt"),
            make_call("calculate", expression="x"),
            make_call("write_file", path="/data/stats.json", content="not json"),
        ]
        result = score_tool_loop(self.EXPECTED, calls, content_checks=self.CHECKS)
        assert result.breakdown["content_score"] == 0.0


class TestScoreFormat:
    def test_json_all_constraints_pass(self):
        constraints = [
            {"type": "json_has_keys", "keys": ["name", "age"]},
            {"type": "json_value", "path": "name", "value": "Alex"},
            {"type": "json_value", "path": "age", "value": 30},
        ]
        result = score_format(constraints, "json", '{"name": "Alex", "age": 30}')
        assert result.score == 1.0

    def test_json_in_code_fence(self):
        constraints = [{"type": "json_has_keys", "keys": ["reply"]}]
        result = score_format(constraints, "json",
                              '```json\n{"reply": "hi"}\n```')
        assert result.score == 1.0

    def test_json_invalid_scores_zero(self):
        constraints = [{"type": "json_has_keys", "keys": ["name"]}]
        result = score_format(constraints, "json", "name is Alex, age 30")
        assert result.score == 0.0

    def test_json_nested_path_type(self):
        constraints = [
            {"type": "json_path_type", "path": "meta.year", "expected": "integer"},
            {"type": "json_value", "path": "meta.year", "value": 2020},
            {"type": "json_array_min", "path": "authors", "value": 2},
        ]
        response = '{"authors": ["a", "b"], "meta": {"year": 2020}}'
        result = score_format(constraints, "json", response)
        assert result.score == 1.0

    def test_top_level_array(self):
        constraints = [
            {"type": "json_array_min", "path": "", "value": 3},
            {"type": "json_path_type", "path": "0.id", "expected": "integer"},
        ]
        response = '[{"id": 1, "status": "open"}, {"id": 2, "status": "closed"}, {"id": 3, "status": "open"}]'
        result = score_format(constraints, "json", response)
        assert result.score == 1.0

    def test_markdown_structure(self):
        constraints = [
            {"type": "min_headings", "value": 1},
            {"type": "has_table"},
            {"type": "min_bullets", "value": 2},
        ]
        response = "# Title\n\n| Name | Score |\n| --- | --- |\n| A | 1 |\n\n- one\n- two"
        result = score_format(constraints, "markdown", response)
        assert result.score == 1.0

    def test_code_fence_check(self):
        constraints = [{"type": "has_code_fence"}, {"type": "contains", "value": "print"}]
        response = "## Demo\n\n```python\nprint('hi')\n```"
        result = score_format(constraints, "markdown", response)
        assert result.score == 1.0

    def test_ifeval_word_and_affix_constraints(self):
        constraints = [
            {"type": "starts_with", "value": "Introducing"},
            {"type": "contains", "value": "eco-friendly"},
            {"type": "ends_with", "value": "!"},
            {"type": "min_words", "value": 3},
        ]
        response = "Introducing our eco-friendly bottle that keeps drinks cold!"
        result = score_format(constraints, "constraint", response)
        assert result.score == 1.0

    def test_not_contains_and_uppercase(self):
        constraints = [{"type": "uppercase"}, {"type": "contains", "value": "PARIS"}]
        assert score_format(constraints, "constraint", "PARIS").score == 1.0
        assert score_format(constraints, "constraint", "Paris").score < 1.0

    def test_system_adherence_partial(self):
        constraints = [
            {"type": "max_words", "value": 20},
            {"type": "not_contains", "value": "z"},
            {"type": "ends_with", "value": "Done."},
        ]
        result = score_format(constraints, "system_adherence",
                              "The ocean is vast and deep. Done.")
        assert result.score == 1.0

    def test_no_constraints_scores_zero(self):
        assert score_format([], "constraint", "anything").score == 0.0


class TestScoreToolLoop:
    def test_optimal_run(self, make_call):
        calls = [
            make_call("get_contacts", name="Alice"),
            make_call("send_message", recipient="alice@example.com", message="hi"),
        ]
        result = score_tool_loop(LOOP_EXPECTED, calls)
        assert result.score == 1.0
        assert result.breakdown["goal_reached"] == 1.0

    def test_loop_detected(self, make_call):
        calls = [
            make_call("get_contacts", name="Alice"),
            make_call("get_contacts", name="Alice"),
            make_call("send_message", recipient="alice@example.com", message="hi"),
        ]
        result = score_tool_loop(LOOP_EXPECTED, calls)
        assert result.breakdown["no_loop_detected"] == 0.0

    def test_premature_stop(self, make_call):
        calls = [make_call("get_contacts", name="Alice")]
        result = score_tool_loop(LOOP_EXPECTED, calls)
        assert result.breakdown["no_premature_stop"] == 0.0
        assert result.breakdown["goal_reached"] == 0.0

    def test_no_calls_at_all(self, make_call):
        result = score_tool_loop(LOOP_EXPECTED, [])
        assert result.breakdown["goal_reached"] == 0.0
        assert result.breakdown["turn_efficiency"] == 0.0
        assert result.score == pytest.approx(0.2)

    def test_retry_after_error_not_flagged_as_loop(self, make_call):
        calls = [
            make_call("get_contacts", name="Alice"),
            make_call("get_contacts", name="Alice"),
            make_call("send_message", recipient="alice@example.com", message="hi"),
        ]
        results = ['{"error": "temporary failure, please retry"}',
                   '[{"name": "Alice"}]', '{"status": "sent"}']
        result = score_tool_loop(LOOP_EXPECTED, calls, results=results)
        assert result.breakdown["no_loop_detected"] == 1.0

    def test_repeat_after_success_still_flagged(self, make_call):
        calls = [
            make_call("get_contacts", name="Alice"),
            make_call("get_contacts", name="Alice"),
        ]
        results = ['[{"name": "Alice"}]', '[{"name": "Alice"}]']
        result = score_tool_loop(LOOP_EXPECTED, calls, results=results)
        assert result.breakdown["no_loop_detected"] == 0.0

    def test_inefficient_but_goal_reached(self, make_call):
        calls = [
            make_call("get_contacts", name="Alice"),
            make_call("get_contacts", name="alice"),
            make_call("get_contacts", name="Alice Smith"),
            make_call("send_message", recipient="alice@example.com", message="hi"),
        ]
        result = score_tool_loop(LOOP_EXPECTED, calls)
        assert result.breakdown["goal_reached"] == 1.0
        assert result.breakdown["turn_efficiency"] == 0.5

    def test_loop_fails_success_even_with_goal_reached(self, make_call):
        """A model that spins on a repeated call before finally reaching the
        goal must not pass — no_loop_detected feeds the score but previously
        wasn't gated into success (regression: real transcript looped 6x on
        the same ask_user call, det scorer still marked it a pass)."""
        calls = [
            make_call("get_contacts", name="Alice"),
            make_call("get_contacts", name="Alice"),
            make_call("send_message", recipient="alice@example.com", message="hi"),
        ]
        result = score_tool_loop(LOOP_EXPECTED, calls)
        assert result.breakdown["goal_reached"] == 1.0
        assert result.success is False

    def test_goal_call_that_errored_does_not_count_as_reached(self, make_call):
        """A goal call whose arguments fuzzy-match but whose tool result is an
        error did not actually accomplish the goal (regression: model called
        post_update with a slightly wrong channel name, got rejected, and gave
        up — det scorer still marked goal_reached=1 from the fuzzy arg match)."""
        calls = [make_call("get_contacts", name="Alice"),
                 make_call("send_message", recipient="alice@example.com", message="hi")]
        results = ['[{"name": "Alice"}]', '{"error": "recipient not found"}']
        result = score_tool_loop(LOOP_EXPECTED, calls, results=results)
        assert result.breakdown["goal_reached"] == 0.0
        assert result.success is False

    def test_goal_call_retried_after_error_still_counts(self, make_call):
        """If the model retries the goal call after an error and the retry
        succeeds, the goal is reached — only a call that never recovers should
        be penalised."""
        calls = [make_call("get_contacts", name="Alice"),
                 make_call("send_message", recipient="alice@example.com", message="hi"),
                 make_call("send_message", recipient="alice@example.com", message="hi")]
        results = ['[{"name": "Alice"}]',
                   '{"error": "temporary failure, please retry"}',
                   '{"status": "sent"}']
        result = score_tool_loop(LOOP_EXPECTED, calls, results=results)
        assert result.breakdown["goal_reached"] == 1.0


class TestScoreCode:
    CASES = [
        CodeCase(args=["hello world"], expected="world hello"),
        CodeCase(args=["a b"], expected="b a"),
        CodeCase(args=[""], expected=""),
    ]
    # Force the rlimit fallback + opt-in so tests execute without Docker in CI.
    SANDBOX = {"backend": "rlimit", "allow_unsandboxed": True}

    def test_all_pass(self):
        code = "```python\ndef rev(s):\n    return ' '.join(reversed(s.split()))\n```"
        result = score_code(code, "rev", self.CASES, **self.SANDBOX)
        assert result.score == 1.0
        assert result.breakdown["passed"] == 3

    def test_partial_pass(self):
        code = "```python\ndef rev(s):\n    return s\n```"
        result = score_code(code, "rev", self.CASES, **self.SANDBOX)
        assert 0.0 < result.score < 1.0

    def test_syntax_error(self):
        result = score_code("```python\ndef rev(s:\n    pass\n```", "rev",
                            self.CASES, **self.SANDBOX)
        assert result.score == 0.0
        assert "syntax error" in result.breakdown["error"]

    def test_no_code_in_response(self):
        result = score_code("I cannot write code.", "rev", self.CASES, **self.SANDBOX)
        assert result.score == 0.0

    def test_timeout(self):
        code = "```python\ndef rev(s):\n    while True:\n        pass\n```"
        result = score_code(code, "rev", self.CASES[:1], timeout=1.0, **self.SANDBOX)
        assert result.score == 0.0
        assert "timed out" in result.breakdown["error"]

    def test_skipped_when_no_real_sandbox(self):
        code = "```python\ndef rev(s):\n    return s\n```"
        result = score_code(code, "rev", self.CASES,
                            backend="rlimit", allow_unsandboxed=False)
        assert result.score == 0.0
        assert result.breakdown["status"] == "skipped_no_sandbox"

    def test_extract_bare_function(self):
        assert extract_code("def f(x):\n    return x").startswith("def f")


class TestScoreKnowledge:
    def test_numeric_exact(self):
        result = score_knowledge({"answer": 270}, "numeric", "The total is 270 km, so 270")
        assert result.score == 1.0

    def test_numeric_within_tolerance(self):
        result = score_knowledge({"answer": 1000}, "numeric", "Roughly 1005")
        assert result.score == 1.0

    def test_numeric_wrong(self):
        result = score_knowledge({"answer": 270}, "numeric", "It is 300")
        assert result.score == 0.0

    def test_numeric_no_number(self):
        result = score_knowledge({"answer": 270}, "numeric", "I think it is far.")
        assert result.score == 0.0

    def test_factual_substring(self):
        result = score_knowledge({"answer": "Canberra"}, "factual",
                                 "The capital of Australia is Canberra.")
        assert result.score == 1.0

    def test_factual_accept_alias(self):
        result = score_knowledge({"answer": "tungsten", "accept": ["wolfram"]},
                                 "factual", "The element W is Wolfram.")
        assert result.score == 1.0

    def test_factual_wrong(self):
        result = score_knowledge({"answer": "Canberra"}, "factual",
                                 "The capital of Australia is Sydney.")
        assert result.score == 0.0

    def test_calibration_uncertain(self):
        result = score_knowledge({"answer": "uncertainty"}, "calibration",
                                 "That is impossible to know exactly.")
        assert result.score == 1.0

    def test_calibration_markdown_bold_phrase(self):
        result = score_knowledge(
            {"answer": "uncertainty"}, "calibration",
            "That number is **impossible** to determine exactly.")
        assert result.score == 1.0

    def test_calibration_impossible_to_provide(self):
        result = score_knowledge(
            {"answer": "uncertainty"}, "calibration",
            "It is impossible to provide an exact number; no one can count them.")
        assert result.score == 1.0

    def test_calibration_confidently_wrong(self):
        result = score_knowledge({"answer": "uncertainty"}, "calibration",
                                 "There are exactly 7 quintillion grains.")
        assert result.score == 0.0


class TestBandWeighted:
    def test_all_bands_present_matches_weighted_sum(self):
        scores = {"anchor": 1.0, "mid": 0.8, "hard": 0.5, "frontier": 0.2}
        expected = sum(scores[b] * BAND_WEIGHTS[b] for b in scores)
        assert band_weighted(scores) == pytest.approx(expected)

    def test_missing_band_renormalizes(self):
        # no frontier tasks yet: remaining weights (anchor/mid/hard) renormalize.
        scores = {"anchor": 1.0, "mid": 1.0, "hard": 0.0}
        w = BAND_WEIGHTS["anchor"] + BAND_WEIGHTS["mid"] + BAND_WEIGHTS["hard"]
        expected = (1.0 * BAND_WEIGHTS["anchor"] + 1.0 * BAND_WEIGHTS["mid"]
                   + 0.0 * BAND_WEIGHTS["hard"]) / w
        assert band_weighted(scores) == pytest.approx(expected)

    def test_empty_scores_is_zero(self):
        assert band_weighted({}) == 0.0


class TestTruncationGate:
    """A response cut off at the token cap never delivered a final answer, so
    text-extraction scorers must not award credit for values that only appear
    inside the incomplete reasoning dump (regression: lc_08/de_01/kn_13 — a
    thinking model's truncated CoT contained the expected answer and det
    scored 1.0 while the judge correctly scored 0.0)."""

    @staticmethod
    def _result(module: str, response: str, truncated: bool = True) -> TaskResult:
        return TaskResult(task_id="t", module=module, prompt="x",
                          response_raw=response, truncated=truncated)

    def test_numeric_answer_inside_truncated_cot_scores_zero(self):
        task = Task(id="t", module="knowledge", prompt="x",
                    answer_type="numeric", expected={"answer": 1440})
        cot = "Thinking: 1200 - 360 + 600 = 1440. So 1440. Wait, revised plan:"
        scored = score_task(task, self._result("knowledge", cot))
        assert scored.score == 0.0
        assert scored.success is False
        assert scored.breakdown["truncated"] is True

    def test_same_response_untruncated_still_scores(self):
        task = Task(id="t", module="knowledge", prompt="x",
                    answer_type="numeric", expected={"answer": 1440})
        text = "1200 - 360 + 600 = 1440"
        scored = score_task(task, self._result("knowledge", text, truncated=False))
        assert scored.score == 1.0

    def test_data_extract_json_inside_truncated_cot_scores_zero(self):
        task = Task(id="t", module="data_extract", prompt="x",
                    expected={"extracted": {"person": "Emma Larsson"}})
        cot = 'Draft: ```json\n{"person": "Emma Larsson"}\n``` Now checking'
        scored = score_task(task, self._result("data_extract", cot))
        assert scored.score == 0.0

    def test_format_truncated_scores_zero(self):
        task = Task(id="t", module="format", prompt="x", answer_type="constraint",
                    constraints=[{"type": "contains", "value": "paris"}])
        scored = score_task(task, self._result("format", "Paris is the capital"))
        assert scored.score == 0.0

    def test_adversarial_no_call_truncated_cot_not_an_answer(self, make_call):
        task = Task(id="t", module="adversarial", prompt="x", tools=["search_web"],
                    expected={"no_call": True})
        scored = score_task(task, self._result("adversarial", "Wait, I'll write it."))
        assert scored.success is False
        assert scored.breakdown["answered"] is False

    def test_code_not_gated_execution_self_verifies(self):
        task = Task(id="t", module="code", prompt="x", function_name="f",
                    test_cases=[CodeCase(args=[1], expected=2)])
        response = "```python\ndef f(x):\n    return x + 1\n```"
        scored = score_task(task, self._result("code", response),
                            sandbox={"allow_unsandboxed": True})
        assert scored.breakdown.get("error") != "response truncated at token cap"

    def test_hit_length_cap_reads_finish_reason(self):
        assert hit_length_cap({"choices": [{"finish_reason": "length"}]}) is True
        assert hit_length_cap({"choices": [{"finish_reason": "stop"}]}) is False
        assert hit_length_cap({}) is False


class TestScalarTypeTolerance:
    """Mixed-type scalar comparison is by VALUE unless a task opts into
    strictness with ``expected["strict_types"]``.

    This inverts the v0.4-v0.11 rule. That rule was introduced because
    tst_11/tst_20/tst_21 ask the model to store a value "as a string" and it
    passed an int, scoring 1.0 by stringified comparison. The v0.11 audit then
    measured the rule across four vendor families and found it sorted models by
    tool-call serialization convention rather than capability: mean pass rate
    on those tasks was gemma 0.93, LFM 0.67, qwen 0.04, ornith 0.00, with no
    relationship to size — a 2.6B beat a 35B 0.80 to 0.00, and a 0.8B and a 27B
    of the same family failed identically. Because those five tasks sit in the
    module carrying 48% of the weight, deleting them flipped the #1 model.

    Typing discipline is still real and still graded — under `strict_types`,
    on its own axis, where it cannot decide the board."""

    def test_state_string_expected_int_actual_passes_by_value(self):
        state = {"kv": {"db_port": 5432}}
        r = score_state({"expected_state": {"kv": {"db_port": "5432"}}},
                        state, {}, [])
        assert r.breakdown["per_key"]["kv"] == 1.0
        assert r.success is True

    def test_strict_types_still_rejects_the_int(self):
        state = {"kv": {"db_port": 5432}}
        r = score_state({"expected_state": {"kv": {"db_port": "5432"}},
                         "strict_types": True}, state, {}, [])
        assert r.breakdown["per_key"]["kv"] == 0.0
        assert r.success is False

    def test_state_string_expected_string_actual_passes(self):
        state = {"kv": {"db_port": "5432"}}
        r = score_state({"expected_state": {"kv": {"db_port": "5432"}}},
                        state, {}, [])
        assert r.breakdown["per_key"]["kv"] == 1.0
        assert r.success is True

    def test_numeric_expected_numeric_actual_still_lenient_on_int_float(self):
        state = {"kv": {"count": 5.0}}
        r = score_state({"expected_state": {"kv": {"count": 5}}}, state, {}, [])
        assert r.breakdown["per_key"]["kv"] == 1.0

    def test_tolerance_does_not_rescue_a_different_number(self):
        """Coercion compares numerically, so "1" must not satisfy 10 — the old
        stringify-and-contain path would have credited that at 0.8."""
        state = {"kv": {"count": 10}}
        r = score_state({"expected_state": {"kv": {"count": "1"}}},
                        state, {}, [])
        assert r.breakdown["per_key"]["kv"] == 0.0
        assert r.success is False

    def test_strictness_reaches_values_nested_in_a_dict(self):
        """The five affected tasks all put their quoted numerics inside `kv`,
        so the flag has to survive _fuzzy_value_match's dict recursion."""
        state = {"kv": {"a": 1, "b": "2"}}
        expected = {"expected_state": {"kv": {"a": "1", "b": "2"}},
                    "strict_types": True}
        r = score_state(expected, state, {}, [])
        assert r.breakdown["per_key"]["kv"] == 0.5


def test_overall_score_fallback_keeps_unjudged_module_in_denominator():
    """Without a fallback, a module the judge never scored leaves the weight
    denominator, so LOSING the judge on a weak module raises the judged score.

    Regression: qwen3.6-35b-a3b-v2 had three tool modules go unjudged on a
    Gemini 503 burst and came out with judged det 0.884 > det 0.864.
    """
    from small_llm_bench.scorer import overall_score
    weights = {"code": 0.5, "tool_loop": 0.5}
    modules = {
        "code": {"det_score": 1.0, "llm_score": 1.0},
        "tool_loop": {"det_score": 0.0, "llm_score": None},   # judge failed here
    }
    # old behavior: the weak module vanishes and the judged score reads perfect
    assert overall_score(modules, weights, key="llm_score") == 1.0
    # with a fallback it contributes its deterministic score instead
    assert overall_score(modules, weights, key="llm_score",
                         fallback_key="det_score") == 0.5
    # a module absent from the run entirely still drops out
    assert overall_score({"code": modules["code"]}, weights, key="llm_score",
                         fallback_key="det_score") == 1.0


class TestRepetitionAndTruncationClass:
    """A truncated trial is either a model failure or a harness artefact, and
    the two must not be scored the same way.

    The v0.11 audit measured this across 11 runs: qwen3.5-0.8b's truncations
    are degenerate loops (53 of 61), while LFM2.5-8B-A1B's are coherent and
    merely verbose (0 of 27 looping). Scoring both as wrong answers cost the
    latter 0.12 overall and three board positions, for trials that measured
    _MODULE_MAX_TOKENS rather than the model.
    """

    def test_clean_prose_has_no_repetition(self):
        text = ("The database listens on port 5432 and the cache uses 6379. "
                "Each service writes its own log file under /var/log.")
        assert repetition_ratio(text) == 0.0

    def test_a_looping_response_scores_high(self):
        line = "This is a factual question that requires a specific answer. "
        assert repetition_ratio(line * 20) > 0.9

    def test_text_shorter_than_two_windows_is_not_a_loop(self):
        assert repetition_ratio("too short to judge") == 0.0

    def test_untruncated_trial_has_no_class(self):
        r = TaskResult(task_id="t", module="knowledge", prompt="p",
                       response_raw="a b c " * 40, truncated=False)
        assert truncation_class(r) == ""

    def test_looping_truncation_is_degenerate(self):
        r = TaskResult(task_id="t", module="knowledge", prompt="p",
                       response_raw="the same sentence over and over again. " * 20,
                       truncated=True)
        assert truncation_class(r) == "degenerate"

    def test_verbose_truncation_is_incomplete(self):
        words = " ".join(f"distinct token number {i}" for i in range(200))
        r = TaskResult(task_id="t", module="knowledge", prompt="p",
                       response_raw=words, truncated=True)
        assert truncation_class(r) == "incomplete"

    def test_a_late_onset_loop_is_degenerate(self):
        """A model that reasons coherently and then cycles until the cap is
        looping, not unfinished. Shaped after the fm_72 probe: qwen3.5-4b wrote
        real self-checking prose, then repeated one couplet to 8192 tokens.
        Whole-response ratio 0.285 read as `incomplete` and the trial was
        discarded, hiding the failure the task exists to catch."""
        head = " ".join(f"considering option {i} carefully on its merits" for i in range(120))
        loop = "Wait, I need to check the word step. No. Okay. " * 60
        r = TaskResult(task_id="t", module="format", prompt="p",
                       response_raw=head + " " + loop, truncated=True)
        assert repetition_ratio(head + " " + loop) < _REPETITION_THRESHOLD
        assert truncation_class(r) == "degenerate"

    def test_loop_ratio_keeps_the_whole_response_reading(self):
        """max(), not replacement: a response looping from its first token must
        still classify even though its tail is no worse than its head."""
        text = "the same sentence over and over again. " * 40
        assert loop_ratio(text) >= repetition_ratio(text)
        assert loop_ratio(text) > _REPETITION_THRESHOLD

    def test_a_short_response_is_not_a_loop_via_its_tail(self):
        """The tail window inherits repetition_ratio's two-window floor, so a
        brief answer cannot be called degenerate on a handful of tail words."""
        assert loop_ratio("a short and entirely unremarkable answer here") == 0.0

    def test_verbose_truncation_with_a_clean_tail_stays_incomplete(self):
        """The rule must not reclassify the trials it was checked against: all
        15 `incomplete` trials on disk keep their class."""
        words = " ".join(f"distinct token number {i}" for i in range(400))
        r = TaskResult(task_id="t", module="knowledge", prompt="p",
                       response_raw=words, truncated=True)
        assert truncation_class(r) == "incomplete"

    def test_a_truncation_with_nothing_to_inspect_is_incomplete(self):
        """No text and no calls means nothing was observed, so nothing can be
        concluded — it must not be charged to the model as a wrong answer."""
        r = TaskResult(task_id="t", module="tools", prompt="p",
                       response_raw="", truncated=True)
        assert truncation_class(r) == "incomplete"

    def test_degenerate_fails_and_still_counts(self):
        r = TaskResult(task_id="t", module="knowledge", prompt="p",
                       response_raw="round and round it goes forever now. " * 20,
                       truncated=True, truncation_class="degenerate")
        assert counted(r) is True

    def test_incomplete_is_scored_not_excluded(self):
        """v1.0 reversed the v0.11 exclusion.

        Excluding `incomplete` deleted a task from the denominator instead of
        failing it, which on the v1.0 sweep moved nothing at or above 4B and
        lifted only the bottom four — LFM2.5-8B-A1B, the model the exclusion
        was written to protect, by 27% relative. With the caps raised
        (tools 2048 -> 4096, tat_03 6144 -> 12288) a trial that still runs out
        of room is a model that could not finish.
        """
        r = TaskResult(task_id="t", module="knowledge", prompt="p",
                       truncated=True, truncation_class="incomplete")
        assert scorable(r) is True      # not an infra failure
        assert counted(r) is True       # and a failure the model earned

    def test_only_infra_failures_are_uncounted(self):
        """`counted` and `scorable` now differ in name only. The pair is kept
        because the coverage report reads them apart, and because a future
        exclusion class would land in `counted` rather than `scorable`."""
        r = TaskResult(task_id="t", module="tools", prompt="p",
                       infra_error=True)
        assert scorable(r) is False
        assert counted(r) is False

    def test_an_incomplete_trial_that_passed_is_still_counted(self):
        """The exclusion is about uncertainty, not about truncation itself: a
        cut-off answer cannot be told apart from a wrong one. A passing
        deterministic verdict removes the uncertainty — the grader reached the
        answer before the cap did.

        Observed: LFM2.5-2.6B solved cd_34 in 21,444 characters, ran into the
        cap afterwards, and lost the trial anyway — leaving the task on one
        usable trial out of three, which pass^3 then drops silently.
        """
        r = TaskResult(task_id="t", module="code", prompt="p",
                       truncated=True, truncation_class="incomplete",
                       det_success=True, success=True)
        assert counted(r) is True

    def test_a_judge_demotion_does_not_uncount_a_passing_truncated_trial(self):
        """det_success is what the gate reads, not success: the deterministic
        grader is what establishes that the answer was reached, and a later
        judge demotion is a verdict on that answer rather than evidence the cap
        prevented one."""
        r = TaskResult(task_id="t", module="code", prompt="p",
                       truncated=True, truncation_class="incomplete",
                       det_success=True, success=False)
        assert counted(r) is True

    def test_degenerate_is_a_hard_fail_even_in_an_exempt_module(self):
        """tools is exempt from the extraction gate, but a model that looped
        until the cap did not do the task whatever state it left behind."""
        task = Task(id="x", module="tools", prompt="p",
                    expected={"expected_state": {"kv": {"a": "1"}}})
        result = TaskResult(task_id="x", module="tools", prompt="p",
                            response_raw="calling the tool again and again now. " * 20,
                            final_state={"kv": {"a": "1"}}, truncated=True)
        scored = score_task(task, result)
        assert scored.success is False
        assert scored.breakdown["truncation_class"] == "degenerate"
