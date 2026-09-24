"""Adversarial robustness module: prompt injection, policy, and ambiguity tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import (BaseModule, build_tools_schema, completion_tokens,
                   hit_length_cap, message_reasoning, message_text,
                   parse_tool_calls)
from .mock_registry import ToolExecutor

if TYPE_CHECKING:
    from ..runner import ChatClient


class AdversarialModule(BaseModule):
    """Tests robustness against conflicting instructions and adversarial prompts.

    Text-only tasks (no tools): single-turn with optional system prompt, scored
    on verifiable text constraints. Tool tasks: agentic loop scored on whether
    the model makes the correct tool choice (e.g. ask_user before acting).
    """

    name = "adversarial"

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Run the task as text-only or agentic depending on whether tools are set."""
        if task.tools:
            return await self._run_tool_task(client, task)
        return await self._run_text_task(client, task)

    async def _run_text_task(self, client: "ChatClient", task: Task) -> TaskResult:
        """Single-turn text response, optionally under a system prompt."""
        messages: list[dict[str, Any]] = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        response = await client.chat(messages, max_tokens=task.max_tokens)
        message = response["choices"][0]["message"]
        content = message_text(message)
        return TaskResult(
            task_id=task.id,
            module=self.name,
            prompt=task.prompt,
            expected={"answer_type": task.answer_type, "constraints": task.constraints},
            response_raw=content,
            turns=[TurnRecord(role="assistant", content=content,
                              completion_tokens=completion_tokens(response),
                              truncated=hit_length_cap(response))],
            completion_tokens=completion_tokens(response),
            truncated=hit_length_cap(response),
            raw_api_responses=[response] if client.save_responses else None,
        )

    async def _run_tool_task(self, client: "ChatClient", task: Task) -> TaskResult:
        """Agentic loop (like tool_loop) with optional system prompt."""
        tools = build_tools_schema(task.tools)
        executor = ToolExecutor(task.tool_overrides)
        messages: list[dict[str, Any]] = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        turns: list[TurnRecord] = []
        raw_responses: list[dict[str, Any]] = []
        final_content = ""
        tokens = 0
        # OR across every turn, not just the last: a truncated mid-loop turn
        # drops the tool call that would have advanced the goal, so the
        # remaining turns run off a broken history. This read only
        # raw_responses[-1] until v1.0, which is the identical loop to
        # tools.py with a weaker rule.
        truncated = False

        for _ in range(task.max_turns):
            response = await client.chat(messages, tools=tools, max_tokens=task.max_tokens)
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
            messages.append(_assistant_message(message))
            for index, call in enumerate(calls):
                result = executor.execute(call.name, call.arguments)
                call_id = _call_id(message, index)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                })
                turns.append(TurnRecord(role="tool", content=json.dumps(result)))

        return TaskResult(
            task_id=task.id,
            module=self.name,
            prompt=task.prompt,
            tools_schema=tools,
            expected=task.expected,
            response_raw=final_content,
            turns=turns,
            completion_tokens=tokens,
            truncated=truncated,
            raw_api_responses=raw_responses if client.save_responses else None,
        )


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }


def _call_id(message: dict[str, Any], index: int) -> str:
    raw_calls = message.get("tool_calls") or []
    if index < len(raw_calls) and raw_calls[index].get("id"):
        return raw_calls[index]["id"]
    return f"call_{index}"
