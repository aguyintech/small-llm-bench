"""A truncation must be classified on evidence, not on what happened to be kept.

``incomplete`` is the default branch of ``truncation_class``, which makes it
worth only as much as the text the classifier was given to read. Two holes let
it be returned without anything having been concluded, both found on one
spark-x2.5-4b run (39 tasks, 3 trials, 14 `incomplete`):

- The agentic-loop modules persisted only ``message["content"]``, and
  ``response_raw`` was the FINAL turn's content — empty precisely when that
  turn hit the cap. Four `tools` episodes were filed `incomplete` on zero
  characters of recorded assistant text; tst_63 had spent 16,629 completion
  tokens getting there, all of it in the reasoning channel and discarded.
- ``repetition_ratio`` is whitespace-tokenised, so a loop emitting no
  whitespace is a single token to it. cd_31 closed with 4,690 characters of
  "20s20s20s…" and scored 0.024.

Each also hid a per-turn question the episode totals cannot answer: which turn
ran long, and which one was cut.
"""

from __future__ import annotations

import json

import httpx
import pytest

from small_llm_bench.models import Task, TaskResult, TurnRecord
from small_llm_bench.modules.tools import ToolsModule
from small_llm_bench.reporter import coverage_report, cut_turn_evidence
from small_llm_bench.runner import ChatClient
from small_llm_bench.scorer import (_CHAR_LOOP_MIN_RUN, _REPETITION_THRESHOLD,
                                    char_loop_ratio, loop_ratio,
                                    truncation_class)

ENDPOINT = "http://testserver/v1"


def _client(*payloads) -> ChatClient:
    remaining = list(payloads)

    def handler(_request):
        return httpx.Response(200, json=remaining.pop(0))

    return ChatClient(endpoint=ENDPOINT, model="test-model",
                      transport=httpx.MockTransport(handler))


def _reply(content=None, tool_calls=None, finish_reason="stop",
           reasoning=None, tokens=7) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"completion_tokens": tokens}}


def _call(name, **arguments):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def _loop_task() -> Task:
    return Task(id="t", module="tools", axis="loop", prompt="do the thing",
                tools=["kv_get"], max_turns=3,
                expected={"tool_sequence": ["kv_get"]})


class TestTheCutTurnIsRecorded:
    """The thinking that spent the budget has to survive to the file."""

    @pytest.mark.asyncio
    async def test_reasoning_is_persisted_on_a_loop_turn(self):
        """tst_63: an episode whose whole token spend was invisible."""
        client = _client(
            _reply(tool_calls=[_call("kv_get", key="a")]),
            _reply(content="", finish_reason="length",
                   reasoning="Let me reconsider. " * 20),
        )
        result = await ToolsModule().run_task(client, _loop_task())
        cut = [t for t in result.turns if t.truncated]
        assert len(cut) == 1
        assert cut[0].reasoning.startswith("Let me reconsider.")

    @pytest.mark.asyncio
    async def test_per_turn_tokens_and_cut_are_recorded(self):
        """The episode total is a sum, so it cannot name the long turn."""
        client = _client(
            _reply(tool_calls=[_call("kv_get", key="a")], tokens=10),
            _reply(content="", finish_reason="length", tokens=90),
        )
        result = await ToolsModule().run_task(client, _loop_task())
        assistant = [t for t in result.turns if t.role == "assistant"]
        assert [t.completion_tokens for t in assistant] == [10, 90]
        assert [t.truncated for t in assistant] == [False, True]
        assert result.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_a_mid_episode_cut_is_attributed_to_its_own_turn(self):
        """``TaskResult.truncated`` ORs the episode; the turn flag localises it."""
        client = _client(
            _reply(tool_calls=[_call("kv_get", key="a")], finish_reason="length"),
            _reply(content="done here"),
        )
        result = await ToolsModule().run_task(client, _loop_task())
        assert result.truncated is True
        assistant = [t for t in result.turns if t.role == "assistant"]
        assert [t.truncated for t in assistant] == [True, False]


class TestLoopingInTheReasoningChannelIsDegenerate:
    def test_a_loop_recorded_only_as_reasoning_classifies(self):
        """Before this, response_raw was "" and the class was `incomplete` —
        not because the model was unfinished but because nothing was read."""
        result = TaskResult(
            task_id="tst_63", module="tools", prompt="p", response_raw="",
            truncated=True,
            turns=[TurnRecord(role="assistant", content="",
                              reasoning="Wait, let me check the order again. " * 40,
                              truncated=True)])
        assert truncation_class(result) == "degenerate"

    def test_an_unfinished_reasoning_dump_is_still_incomplete(self):
        """The new reading must not call every silent turn a loop."""
        reasoning = " ".join(f"step {i} considers a different order" for i in range(200))
        result = TaskResult(
            task_id="tst_23", module="tools", prompt="p", response_raw="",
            truncated=True,
            turns=[TurnRecord(role="assistant", content="", reasoning=reasoning,
                              truncated=True)])
        assert truncation_class(result) == "incomplete"

    def test_turns_are_read_separately_not_stitched(self):
        """An agent that restates its progress across turns is reporting, not
        looping: concatenating the episode would invent the 8-gram repeat."""
        progress = ("The refund for order 401 succeeded, but order 403 needs a "
                    "manager code before it can be issued at all. ")
        result = TaskResult(
            task_id="tst_23", module="tools", prompt="p", response_raw="",
            truncated=True,
            turns=[TurnRecord(role="assistant", content=progress),
                   TurnRecord(role="assistant", content=progress),
                   TurnRecord(role="assistant", content="", truncated=True)])
        assert truncation_class(result) == "incomplete"

    def test_an_empty_episode_is_still_incomplete(self):
        """Nothing observed, nothing concluded — it must not become a verdict."""
        result = TaskResult(
            task_id="t", module="tools", prompt="p", response_raw="",
            truncated=True,
            turns=[TurnRecord(role="assistant", content="", truncated=True)])
        assert truncation_class(result) == "incomplete"


class TestCharacterLevelLoops:
    def test_a_whitespace_free_cycle_is_degenerate(self):
        """cd_31: 4,690 characters of "20s" is one word-level token, so the
        n-gram ratio cannot see it at all."""
        run = "20s" * 400
        text = "Let me test the pattern against the input: " + run
        assert loop_ratio(text) < _REPETITION_THRESHOLD
        assert char_loop_ratio(text) > 0.99
        result = TaskResult(task_id="cd_31", module="code", prompt="p",
                            response_raw=text, truncated=True)
        assert truncation_class(result) == "degenerate"

    def test_a_single_repeated_character_is_a_cycle(self):
        assert char_loop_ratio("x" + "a" * _CHAR_LOOP_MIN_RUN) > 0.99

    def test_ordinary_prose_has_nothing_to_measure(self):
        """The check only looks inside runs long enough to be anomalous, which
        prose never has — 25,692 untruncated trials on disk hold two such runs
        between them, and both are real decode loops."""
        prose = " ".join(f"sentence {i} says something specific" for i in range(300))
        assert char_loop_ratio(prose) == 0.0

    def test_a_long_run_that_does_not_cycle_is_not_a_loop(self):
        """Hashes, base64 and long identifiers are long without being periodic."""
        import hashlib
        blob = "".join(hashlib.sha256(str(i).encode()).hexdigest()
                       for i in range(20))
        assert len(blob) >= _CHAR_LOOP_MIN_RUN
        assert char_loop_ratio(blob) < 0.9


def _incomplete(task_id, module, success, **turn) -> TaskResult:
    return TaskResult(task_id=task_id, module=module, prompt="p",
                      truncated=True, truncation_class="incomplete",
                      success=success, det_success=success,
                      turns=[TurnRecord(role="assistant", truncated=True, **turn)])


class TestTheCoverageLineMatchesTheFile:
    """"Scored as a failure" is the policy, not the outcome."""

    def test_a_truncated_pass_is_not_counted_as_a_failure(self):
        """minicpm5-2b's cd_31: written code that executed clean, then rambled
        to the cap. Twice, det 0.85, both passes — and both were named in a
        line that said SCORED AS FAILURES and in the count the operator is
        told to raise a cap over."""
        cov = coverage_report([
            _incomplete("cd_31", "code", True, content="def f(): pass" * 400),
            _incomplete("cd_31", "code", True, content="def f(): pass" * 400),
            _incomplete("tst_23", "tools", False, reasoning="deliberating " * 900),
        ])
        assert cov["incomplete"] == 3
        assert cov["incomplete_failed"] == 1
        assert cov["per_module"]["code"]["incomplete_failed"] == 0
        assert cov["per_module"]["tools"]["incomplete_failed"] == 1

    def test_a_passing_module_is_left_out_of_the_cap_hint(self):
        """The hint is about trials that lost a verdict. `code` grades a
        truncated response by executing it, so it can truncate and pass."""
        cov = coverage_report([_incomplete("cd_31", "code", True, content="x" * 99)])
        assert cov["incomplete"] == 1
        assert cov["incomplete_failed"] == 0
        assert cov["per_module"]["code"]["incomplete"] == 1
        assert cov["per_module"]["code"]["incomplete_failed"] == 0


class TestTheReportAnswersItsOwnQuestion:
    """The line used to tell the operator to go and check the cut themselves —
    the only honest option while the cut turn's text was unreadable."""

    def test_a_cap_bound_turn_is_reported_with_what_it_produced(self):
        rows = cut_turn_evidence([
            _incomplete("tst_23", "tools", False,
                        completion_tokens=4096, reasoning="x " * 8000)])
        assert len(rows) == 1
        assert rows[0]["tokens"] == 4096
        assert rows[0]["content_chars"] == 0
        assert rows[0]["calls"] == 0
        assert rows[0]["reasoning_chars"] > 10000

    def test_a_truncated_pass_is_not_evidence_for_a_cap_raise(self):
        rows = cut_turn_evidence([
            _incomplete("cd_31", "code", True,
                        completion_tokens=12288, content="y" * 46261)])
        assert rows == []

    def test_an_old_file_asserts_nothing_about_the_cut(self):
        """Without per-turn flags the cut cannot be located, and a turn with no
        flag is not evidence of a clean turn."""
        stale = TaskResult(task_id="tst_41", module="tools", prompt="p",
                           truncated=True, truncation_class="incomplete",
                           turns=[TurnRecord(role="assistant", content="")])
        assert cut_turn_evidence([stale]) == []

    def test_multiple_cut_turns_report_the_last_and_say_how_many(self):
        r = TaskResult(
            task_id="cd_31", module="code", prompt="p", truncated=True,
            truncation_class="incomplete", success=False,
            turns=[TurnRecord(role="assistant", truncated=True,
                              completion_tokens=12288, content="a" * 10),
                   TurnRecord(role="user", content="test failed"),
                   TurnRecord(role="assistant", truncated=True,
                              completion_tokens=12288, content="b" * 20)])
        rows = cut_turn_evidence([r])
        assert rows[0]["cut_turns"] == 2
        assert rows[0]["content_chars"] == 20
