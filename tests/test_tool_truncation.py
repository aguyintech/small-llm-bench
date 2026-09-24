"""The tool_* modules must record truncation like every other module.

They originally never called ``hit_length_cap``, so every tool_* trial reported
``truncated=False`` regardless of where the completion actually stopped. That
made a real failure mode invisible (a reasoning model spending its whole budget
on the trace and returning empty content read as "the model declined to
answer") and it silently disabled the truncated-trial retry path in
``plan_work``, which excludes truncated results from reuse.
"""

from __future__ import annotations

import json

import httpx
import pytest

from small_llm_bench.models import Task
from small_llm_bench.modules.base import _MODULE_MAX_TOKENS
from small_llm_bench.modules.tools import ToolsModule
from small_llm_bench.runner import ChatClient

ENDPOINT = "http://testserver/v1"


def _client(handler) -> ChatClient:
    return ChatClient(endpoint=ENDPOINT, model="test-model",
                      transport=httpx.MockTransport(handler))


def _reply(content=None, tool_calls=None, finish_reason="stop") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _tool_call(name, **arguments):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def _replies(*payloads):
    """MockTransport handler that returns each payload in turn."""
    remaining = list(payloads)

    def handler(_request):
        return httpx.Response(200, json=remaining.pop(0))

    return handler


class TestCallAxisTruncation:
    @pytest.mark.asyncio
    async def test_length_finish_marks_truncated(self):
        """The ts_16 failure: budget burned on reasoning, content empty."""
        client = _client(_replies(_reply(content="", finish_reason="length")))
        task = Task(id="t", module="tools", prompt="explain hash maps",
                    expected={"no_call": True})
        result = await ToolsModule().run_task(client, task)
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_normal_finish_is_not_truncated(self):
        client = _client(_replies(_reply(content="A hash map is …")))
        task = Task(id="t", module="tools", prompt="explain hash maps",
                    expected={"no_call": True})
        result = await ToolsModule().run_task(client, task)
        assert result.truncated is False


class TestLoopTruncation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("axis", ["loop", "discovery"])
    async def test_truncation_on_a_middle_turn_is_recorded(self, axis):
        """A cap hit mid-loop taints the episode, not only a final-turn hit."""
        client = _client(_replies(
            _reply(tool_calls=[_tool_call("cli_help", topic="notes")],
                   finish_reason="length"),
            _reply(content="done"),
        ))
        task = Task(id="t", module="tools", axis=axis, prompt="append a line",
                    tools=["cli_help", "ask_user"], max_turns=4)
        result = await ToolsModule().run_task(client, task)
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_clean_loop_is_not_truncated(self):
        client = _client(_replies(
            _reply(tool_calls=[_tool_call("cli_help", topic="notes")]),
            _reply(content="done"),
        ))
        task = Task(id="t", module="tools", prompt="append a line",
                    tools=["cli_help", "ask_user"], max_turns=4)
        result = await ToolsModule().run_task(client, task)
        assert result.truncated is False


class TestStateAxisTruncation:
    @pytest.mark.asyncio
    async def test_truncation_on_a_middle_turn_is_recorded(self):
        client = _client(_replies(
            _reply(tool_calls=[_tool_call("create_order", customer="Acme",
                                          total=10)],
                   finish_reason="length"),
            _reply(content="done"),
        ))
        task = Task(id="t", module="tools", prompt="place an order",
                    tools=["create_order"], max_turns=4,
                    initial_state={"orders": []})
        result = await ToolsModule().run_task(client, task)
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_clean_run_is_not_truncated(self):
        client = _client(_replies(
            _reply(tool_calls=[_tool_call("create_order", customer="Acme",
                                          total=10)]),
            _reply(content="done"),
        ))
        task = Task(id="t", module="tools", prompt="place an order",
                    tools=["create_order"], max_turns=4,
                    initial_state={"orders": []})
        result = await ToolsModule().run_task(client, task)
        assert result.truncated is False


class TestCapHeadroom:
    def test_cap_leaves_room_for_a_reasoning_trace(self):
        """768 was below the trace length of an ordinary reasoning model, so a
        one-sentence no_call answer could never be reached."""
        assert _MODULE_MAX_TOKENS["tools"] >= 2048
