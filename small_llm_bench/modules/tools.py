"""Every tool-using axis in one module.

Four modules until v0.10 — tool_simple, tool_loop, tool_state, tool_discovery —
which left `call` and `discovery` holding a single task each while still carrying
0.11 and 0.09 of the balanced headline. Merging them puts one weight on the axis
family and keeps the four as reported sub-rows.

Execution still differs per task shape, because that is what the tasks are:

* a single-turn call (``expected.no_call`` / ``tool_name`` / ``parallel``) is one
  request with tool schemas attached and no loop;
* a stateful episode (``initial_state`` / ``expected.expected_state``) runs the
  agentic loop against a mutable backend and is graded on the FINAL state;
* everything else is the ordinary agentic loop, graded on the goal call.

``scorer.score_task`` dispatches on the same three shapes, so grading is
unchanged from the four-module layout.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import (BaseModule, build_tools_schema, completion_tokens,
                   hit_length_cap, message_reasoning, parse_tool_calls)
from .mock_registry import StatefulToolExecutor, ToolExecutor

if TYPE_CHECKING:
    from ..runner import ChatClient


def is_single_call(task: Task) -> bool:
    """True if the task is graded on one call rather than an episode."""
    return bool(task.expected.get("no_call") or task.expected.get("tool_name")
                or task.parallel)


def is_stateful(task: Task) -> bool:
    """True if the task is graded on the backend state it leaves behind."""
    return bool(task.initial_state or task.expected.get("expected_state"))


class ToolsModule(BaseModule):
    """Tool calling, agentic loops, stateful episodes, and discovery."""

    name = "tools"

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Run the task the way its shape requires."""
        if is_single_call(task):
            return await self._run_single(client, task)
        return await self._run_loop(client, task)

    async def _run_single(self, client: "ChatClient", task: Task) -> TaskResult:
        """One request with tool schemas attached; record the call it made."""
        tools = build_tools_schema(task.tools)
        response = await client.chat(
            [{"role": "user", "content": task.prompt}], tools=tools,
            max_tokens=task.max_tokens,
        )
        message = response["choices"][0]["message"]
        # Deliberately the raw `content` field, not the reasoning_content
        # fallback other modules use: a no_call task needs to tell "answered
        # the user" apart from "burned the token budget on reasoning and
        # returned nothing" — reasoning_content being non-empty doesn't mean
        # the model ever delivered a final answer.
        turn = TurnRecord(
            role="assistant",
            content=message.get("content"),
            reasoning=message_reasoning(message),
            tool_calls=parse_tool_calls(message),
            completion_tokens=completion_tokens(response),
            truncated=hit_length_cap(response),
        )
        return TaskResult(
            task_id=task.id,
            module=self.name,
            axis=task.axis,
            prompt=task.prompt,
            tools_schema=tools,
            expected=task.expected,
            response_raw=json.dumps(message),
            turns=[turn],
            truncated=hit_length_cap(response),
            completion_tokens=completion_tokens(response),
            raw_api_responses=[response] if client.save_responses else None,
        )

    async def _run_loop(self, client: "ChatClient", task: Task) -> TaskResult:
        """Run the agentic loop until the model stops or max_turns is hit."""
        tools = build_tools_schema(task.tools)
        stateful = is_stateful(task)
        executor: Any = (StatefulToolExecutor(task.initial_state, task.tool_overrides)
                         if stateful else ToolExecutor(task.tool_overrides))
        messages: list[dict[str, Any]] = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        turns: list[TurnRecord] = []
        raw_responses: list[dict[str, Any]] = []
        final_content = ""
        tokens = 0
        # Any turn cut off at the cap taints the episode, not just the last
        # one: a truncated mid-loop turn drops the tool call that would have
        # advanced the goal, so the remaining turns run off a broken history.
        truncated = False

        for _ in range(task.max_turns):
            response = await client.chat(messages, tools=tools,
                                         max_tokens=task.max_tokens)
            raw_responses.append(response)
            turn_cut = hit_length_cap(response)
            turn_tokens = completion_tokens(response)
            truncated = truncated or turn_cut
            tokens += turn_tokens
            message = response["choices"][0]["message"]
            calls = parse_tool_calls(message)
            turns.append(TurnRecord(
                role="assistant", content=message.get("content"), tool_calls=calls,
                reasoning=message_reasoning(message),
                completion_tokens=turn_tokens, truncated=turn_cut,
            ))
            if not calls:
                final_content = message.get("content") or ""
                break
            messages.append(assistant_message(message))
            for index, call in enumerate(calls):
                result = executor.execute(call.name, call.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id(message, index),
                    "content": json.dumps(result),
                })
                turns.append(TurnRecord(role="tool", content=json.dumps(result)))

        return TaskResult(
            task_id=task.id,
            module=self.name,
            axis=task.axis,
            prompt=task.prompt,
            tools_schema=tools,
            expected=task.expected,
            response_raw=final_content,
            turns=turns,
            final_state=executor.state if stateful else {},
            truncated=truncated,
            completion_tokens=tokens,
            raw_api_responses=raw_responses if client.save_responses else None,
        )


def assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Build the assistant message to append back into the conversation."""
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }


def call_id(message: dict[str, Any], index: int) -> str:
    """Get the tool_call_id for the index-th call, with a stable fallback."""
    raw_calls = message.get("tool_calls") or []
    if index < len(raw_calls) and raw_calls[index].get("id"):
        return raw_calls[index]["id"]
    return f"call_{index}"
