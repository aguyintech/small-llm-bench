"""Tests for the new hard axes: stateful tools, memory, MMLU-Pro, and needles."""

from __future__ import annotations

import json
import re

import httpx

from small_llm_bench.models import Task, TurnRecord
from small_llm_bench.models import TestCase as CodeCase
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.modules.code import CodeModule
from small_llm_bench.modules.long_context import LongContextModule, build_haystack
from small_llm_bench.modules.mock_registry import StatefulToolExecutor
from small_llm_bench.modules.tools import ToolsModule
from small_llm_bench.runner import ChatClient
from small_llm_bench.scorer import (score_code, score_knowledge,
                                    score_multi_turn_if, score_state,
                                    score_task, score_tool_loop)

# rlimit + opt-in so code executes without Docker in CI (matches test_scorer).
_SANDBOX = {"backend": "rlimit", "allow_unsandboxed": True}

ENDPOINT = "http://testserver/v1"


def _client(handler, **kwargs) -> ChatClient:
    return ChatClient(endpoint=ENDPOINT, model="test-model",
                      transport=httpx.MockTransport(handler), **kwargs)


def _tool_call(name, **arguments):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


# --- stateful executor --------------------------------------------------------

class TestStatefulExecutor:
    def test_create_and_cancel_order(self):
        ex = StatefulToolExecutor({"orders": []})
        ex.execute("create_order", {"customer": "Acme", "total": 120})
        ex.execute("cancel_order", {"order_id": 101})
        assert ex.state["orders"][0]["status"] == "cancelled"

    def test_refund_under_limit_allowed(self):
        ex = StatefulToolExecutor({"balance": 500, "refunds": []})
        out = ex.execute("issue_refund", {"order_id": 1, "amount": 40})
        assert out["status"] == "refunded"
        assert ex.state["balance"] == 460

    def test_refund_over_limit_refused_leaves_state(self):
        ex = StatefulToolExecutor({"balance": 500, "refunds": []})
        out = ex.execute("issue_refund", {"order_id": 1, "amount": 200})
        assert "error" in out
        assert ex.state["balance"] == 500
        assert ex.state["refunds"] == []

    def test_refund_over_limit_with_code_allowed(self):
        ex = StatefulToolExecutor({"balance": 500, "refunds": []})
        out = ex.execute("issue_refund",
                         {"order_id": 1, "amount": 90, "manager_code": "MGR-9"})
        assert out["status"] == "refunded"

    def test_memory_write_then_recall(self):
        ex = StatefulToolExecutor({"memory": []})
        ex.execute("memory_write", {"fact": "The database listens on port 5432."})
        ex.execute("memory_write", {"fact": "The cache listens on port 6379."})
        hits = ex.execute("memory_recall", {"query": "database"})
        assert hits["matches"] == ["The database listens on port 5432."]

    def test_initial_state_is_deep_copied(self):
        seed = {"orders": []}
        ex = StatefulToolExecutor(seed)
        ex.execute("create_order", {"customer": "X", "total": 1})
        assert seed["orders"] == []  # seed untouched


# --- score_state --------------------------------------------------------------

class TestScoreState:
    def test_perfect_match(self, make_call):
        expected = {"optimal_turns": 1,
                    "expected_state": {"kv": {"a": "1"}}}
        result = score_state(expected, {"kv": {"a": "1"}}, {"kv": {}},
                             [make_call("kv_set", key="a", value="1")])
        assert result.score == 1.0

    def test_wrong_final_state(self, make_call):
        expected = {"expected_state": {"kv": {"a": "1"}}}
        result = score_state(expected, {"kv": {"a": "2"}}, {"kv": {}},
                             [make_call("kv_set", key="a", value="2")])
        assert result.breakdown["goal_state_match"] == 0.0

    def test_policy_refusal_scores_high(self):
        """No calls, state unchanged: a correct refusal should score 1.0."""
        expected = {"unchanged": ["balance"],
                    "expected_state": {"balance": 500, "refunds": []}}
        state = {"balance": 500, "refunds": []}
        result = score_state(expected, state, state, [])
        assert result.score == 1.0
        assert result.breakdown["no_side_effects"] == 1.0


# --- multiple choice ----------------------------------------------------------

class TestMultipleChoice:
    def test_answer_cue(self):
        result = score_knowledge({"answer": "C"}, "multiple_choice",
                                 "After reasoning, the answer is C.")
        assert result.score == 1.0

    def test_trailing_letter_fallback(self):
        result = score_knowledge({"answer": "B"}, "multiple_choice", "I pick B")
        assert result.score == 1.0

    def test_wrong_letter(self):
        result = score_knowledge({"answer": "A"}, "multiple_choice",
                                 "The answer is D.")
        assert result.score == 0.0

    def test_cue_beats_stray_letters_in_explanation(self):
        # Regression: "answer is B" cue must win over stray standalone letters
        # in the trailing explanation ("A catalyst...") — the old [^A-J] gap
        # was blocked by the "is" and the fallback grabbed the last letter.
        result = score_knowledge(
            {"answer": "B"}, "multiple_choice",
            "The correct answer is B.\n\nA catalyst lowers activation energy.")
        assert result.breakdown["chosen"] == "B"
        assert result.score == 1.0


# --- needle in a haystack -----------------------------------------------------

class TestHaystack:
    SPEC = {"filler_tokens": 2000, "needle": "The code is 4827.", "position": 0.5}

    def test_needle_present(self):
        doc = build_haystack(self.SPEC)
        assert "The code is 4827." in doc

    def test_deterministic(self):
        assert build_haystack(self.SPEC) == build_haystack(self.SPEC)

    def test_reaches_target_size(self):
        doc = build_haystack({"filler_tokens": 4000, "needle": "X", "position": 0.5})
        assert len(doc.split()) * 1.3 >= 4000

    def test_needle_roughly_centered(self):
        doc = build_haystack(self.SPEC)
        lines = doc.splitlines()
        idx = next(i for i, ln in enumerate(lines) if ln == "The code is 4827.")
        assert 0.3 < idx / len(lines) < 0.7

    def test_multi_key_includes_target_and_distractors(self):
        doc = build_haystack({"type": "multi_key", "filler_tokens": 1000,
                              "needle": "London code is 5391.",
                              "distractors": ["Berlin code is 1122."]})
        assert "London code is 5391." in doc and "Berlin code is 1122." in doc

    def test_multi_hop_inserts_chain(self):
        doc = build_haystack({"type": "multi_hop", "filler_tokens": 1000,
                              "chain": ["A is 1.", "B copies A.", "C copies B."]})
        for line in ("A is 1.", "B copies A.", "C copies B."):
            assert line in doc

    def test_aggregation_injects_counts(self):
        doc = build_haystack({"type": "aggregation", "filler_tokens": 1000,
                              "inject": {"saturn": 5, "mars": 2}})
        assert doc.count("saturn") == 5 and doc.count("mars") == 2


# --- module integration -------------------------------------------------------

async def test_state_axis_end_to_end():
    """Two create_order calls then a stop; final state has both orders."""
    turns = []

    def handler(request: httpx.Request) -> httpx.Response:
        turns.append(1)
        if len(turns) == 1:
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [_tool_call("create_order", customer="Acme", total=120),
                                  _tool_call("create_order", customer="Globex", total=75)]}
        else:
            msg = {"role": "assistant", "content": "Done."}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    task = Task(id="tst_x", module="tools",
                prompt="create two orders", tools=["create_order"],
                initial_state={"orders": []},
                expected={"optimal_turns": 2, "expected_state": {"orders": [
                    {"id": 101, "customer": "Acme", "total": 120, "status": "open"},
                    {"id": 102, "customer": "Globex", "total": 75, "status": "open"}]}})
    client = _client(handler)
    result = await ToolsModule().run_task(client, task)
    await client.close()
    assert len(result.final_state["orders"]) == 2
    assert score_task(task, result).score == 1.0


async def test_code_module_includes_buggy_code():
    """Repair tasks must show the buggy_code to the model."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        msg = {"role": "assistant", "content": "```python\ndef f(): pass\n```"}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    task = Task(id="cd_x", module="code", prompt="fix f",
                function_name="f", buggy_code="def f():\n    return 0")
    client = _client(handler)
    await CodeModule().run_task(client, task)
    await client.close()
    sent = captured["body"]["messages"][0]["content"]
    assert "return 0" in sent


class TestToolDiscovery:
    DISCOVER = {"goal_tool": "run_command", "content_arg": "command",
                "optimal_turns": 2, "min_calls": 1}
    CHECKS = [{"type": "contains", "value": v}
              for v in ("notes", "append", "vault", "work", "Shipped v2")]

    def test_correct_constructed_command(self, make_call):
        calls = [make_call("cli_help", topic="notes"),
                 make_call("run_command",
                           command='notes append --vault work --text "Shipped v2"')]
        r = score_tool_loop(self.DISCOVER, calls, content_checks=self.CHECKS)
        assert r.success is True

    def test_wrong_command_missing_flag_fails(self, make_call):
        calls = [make_call("run_command", command='notes append "Shipped v2"')]
        r = score_tool_loop(self.DISCOVER, calls, content_checks=self.CHECKS)
        assert r.success is False  # missing --vault work

    def test_missing_param_asks(self, make_call):
        expected = {"goal_tool": "ask_user", "content_arg": "question",
                    "optimal_turns": 1, "min_calls": 1}
        checks = [{"type": "contains", "any": ["vault", "which", "where"]}]
        calls = [make_call("ask_user", question="Which vault should I append to?")]
        r = score_tool_loop(expected, calls, content_checks=checks)
        assert r.success is True

    def test_missing_param_hallucinating_fails(self, make_call):
        expected = {"goal_tool": "ask_user", "content_arg": "question",
                    "optimal_turns": 1, "min_calls": 1}
        checks = [{"type": "contains", "any": ["vault", "which", "where"]}]
        # model fabricated a command instead of asking -> goal not reached
        calls = [make_call("run_command", command="notes append --vault guess")]
        r = score_tool_loop(expected, calls, content_checks=checks)
        assert r.success is False


class TestLongContextCoding:
    def test_support_code_lets_function_reference_externals(self):
        code = "```python\ndef f(x):\n    return x * RATE\n```"
        r = score_code(code, "f", [CodeCase(args=[10], expected=20)],
                       support_code="RATE = 2",
                       regression_cases=[CodeCase(args=[5], expected=10)], **_SANDBOX)
        assert r.score == 1.0
        assert r.breakdown["pass_to_pass"] == [1, 1]
        assert r.breakdown["wellformed"] is True

    def test_regression_gate_fails_a_breaking_fix(self):
        # passes FAIL_TO_PASS (x=10) but breaks PASS_TO_PASS (x=5)
        code = "```python\ndef f(x):\n    return 20\n```"
        r = score_code(code, "f", [CodeCase(args=[10], expected=20)],
                       regression_cases=[CodeCase(args=[5], expected=10)], **_SANDBOX)
        assert r.score < 1.0
        assert r.breakdown["pass_to_pass"] == [0, 1]

    def test_wellformed_false_when_no_code(self):
        r = score_code("I cannot do this.", "f", [CodeCase(args=[1], expected=1)])
        assert r.breakdown["wellformed"] is False


async def test_code_module_includes_context_files():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        msg = {"role": "assistant", "content": "```python\ndef f(): pass\n```"}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    task = Task(id="cd_y", module="code", prompt="fix f", function_name="f",
                context_files={"constants.py": "SECRET = 42"},
                buggy_code="def f():\n    return 0")
    client = _client(handler)
    await CodeModule().run_task(client, task)
    await client.close()
    assert "SECRET = 42" in captured["body"]["messages"][0]["content"]


async def test_long_context_module_builds_and_asks():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "4827"}}]})

    task = Task(id="lc_x", module="long_context", answer_type="numeric",
                prompt="What is the code?",
                haystack={"filler_tokens": 1000, "needle": "code is 4827", "position": 0.5},
                expected={"answer": 4827})
    client = _client(handler)
    result = await LongContextModule().run_task(client, task)
    await client.close()
    assert "code is 4827" in captured["body"]["messages"][0]["content"]
    assert score_task(task, result).score == 1.0


# --- cross-file coding (context_files become importable modules) --------------

class TestCrossFileImport:
    def test_correct_import_resolves(self):
        ctx = {"pricing/constants.py": "RATE = 0.5\n"}
        sol = "```python\nfrom pricing.constants import RATE\n" \
              "def f(x):\n    return x * RATE\n```"
        res = score_code(sol, "f", [CodeCase(args=[10], expected=5.0)],
                         context_files=ctx, **_SANDBOX)
        assert res.score == 1.0
        assert res.breakdown["wellformed"] is True

    def test_package_submodule_import(self):
        ctx = {"shop/taxes.py": "def rate(r):\n    return {'EU': 0.2}.get(r, 0.0)\n"}
        sol = "```python\nfrom shop.taxes import rate\n" \
              "def total(a, r):\n    return round(a + a * rate(r), 2)\n```"
        res = score_code(sol, "total",
                         [CodeCase(args=[100, "EU"], expected=120.0),
                          CodeCase(args=[100, "XX"], expected=100.0)],
                         context_files=ctx, **_SANDBOX)
        assert res.score == 1.0

    def test_relative_import_resolves_inside_the_context_package(self):
        ctx = {"shipping/rates.py": "RATE = 0.5\n",
               "shipping/utils.py": "def half(x):\n    return x / 2\n"}
        for imports in ("from .rates import RATE",
                        "from . import rates\nRATE = rates.RATE"):
            sol = f"```python\n{imports}\ndef f(x):\n    return x * RATE\n```"
            res = score_code(sol, "f", [CodeCase(args=[10], expected=5.0)],
                             context_files=ctx, **_SANDBOX)
            assert res.score == 1.0, (imports, res.breakdown)

    def test_no_package_is_guessed_across_several(self):
        ctx = {"a/x.py": "X = 1\n", "b/y.py": "Y = 2\n"}
        sol = "```python\nfrom .x import X\ndef f():\n    return X\n```"
        res = score_code(sol, "f", [CodeCase(args=[], expected=1)],
                         context_files=ctx, **_SANDBOX)
        assert res.score == 0.0

    def test_context_tasks_name_a_target_inside_their_package(self):
        for task in load_tasks("code", profile="full"):
            if not task.context_files:
                continue
            packages = {p.rpartition("/")[0] for p in task.context_files
                        if p.endswith(".py")}
            assert len(packages) == 1, task.id
            package = packages.pop()
            targets = [p for p in re.findall(r"[\w/]+\.py", task.prompt)
                       if p not in task.context_files]
            assert targets and all(p.rpartition("/")[0] == package
                                   for p in targets), (task.id, targets)

    def test_ignoring_layout_fails(self):
        ctx = {"shop/taxes.py": "def rate(r):\n    return 0.2\n"}
        bad = "```python\ndef total(a, r):\n    return a\n```"  # ignores tax module
        res = score_code(bad, "total",
                         [CodeCase(args=[100, "EU"], expected=120.0)],
                         context_files=ctx, **_SANDBOX)
        assert res.score == 0.0


# --- scalar state coercion (a port stored as int vs expected str) ------------

class TestScalarStateMatch:
    def test_int_actual_passes_str_expected(self):
        """Type is NOT part of the spec by default (v0.12). The strict rule
        this test used to pin was measured across four vendor families and
        found to sort models by tool-call serialization convention rather than
        capability — see TestScalarTypeTolerance in test_scorer.py for the
        numbers. A task that genuinely grades typing opts in with
        ``strict_types``."""
        res = score_state({"expected_state": {"kv": {"db_port": "5432"}}},
                          {"kv": {"db_port": 5432}}, {}, [])
        assert res.success is True
        assert res.score == 1.0

    def test_int_actual_fails_str_expected_under_strict_types(self):
        res = score_state({"expected_state": {"kv": {"db_port": "5432"}},
                           "strict_types": True},
                          {"kv": {"db_port": 5432}}, {}, [])
        assert res.success is False
        assert res.score < 1.0

    def test_genuinely_different_scalar_still_fails(self):
        res = score_state({"expected_state": {"kv": {"floor": "4"}}},
                          {"kv": {"floor": "4th"}}, {}, [])
        assert res.success is False


# --- multi_turn_if (accumulating constraints, instruction forgetting) ---------

class TestMultiTurnIf:
    CONV = [
        {"prompt": "p1", "constraints": [{"type": "lowercase"}]},
        {"prompt": "p2", "constraints": [{"type": "contains", "value": "because"}]},
        {"prompt": "p3", "constraints": [{"type": "min_words", "value": 3}]},
    ]

    def _turns(self, *contents):
        return [TurnRecord(role="assistant", content=c) for c in contents]

    def test_all_turns_pass(self):
        turns = self._turns("hello there",
                            "i like it because reasons",
                            "another lowercase reply because words")
        res = score_multi_turn_if(self.CONV, turns)
        assert res.success is True
        assert res.score == 1.0
        assert res.breakdown["forgetting"] == 0.0

    def test_forgets_lowercase_by_turn_three(self):
        turns = self._turns("hello there",
                            "i like it because reasons",
                            "SHOUTING because LOUD now")  # breaks turn-1 lowercase
        res = score_multi_turn_if(self.CONV, turns)
        assert res.success is False
        assert res.breakdown["per_turn"][0] == 1.0
        assert res.breakdown["per_turn"][2] < 1.0
        assert res.breakdown["forgetting"] > 0.0

    def test_missing_turn_is_failure(self):
        res = score_multi_turn_if(self.CONV, self._turns("hello", "x because y"))
        assert res.success is False
        assert res.breakdown["responses_seen"] == 2

    def test_tasks_are_hard_tier(self, tasks_dir):
        tasks = load_tasks("multi_turn_if", tasks_dir=tasks_dir)
        assert tasks and all(t.tier == "hard" for t in tasks)
        assert all(t.conversation for t in tasks)
