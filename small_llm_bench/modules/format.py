"""Formatting / instruction-following module (IFEval + JSON-mode style).

Single-turn: the model is asked to produce output under verifiable constraints
(strict JSON, Markdown structure, length/keyword rules, or adherence to a long
system prompt). Scoring is deterministic via the constraint checks in the task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import BaseModule, completion_tokens, hit_length_cap, message_text

if TYPE_CHECKING:
    from ..runner import ChatClient


class FormatModule(BaseModule):
    """Tests structured output and instruction following under verifiable rules."""

    name = "format"

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Send the prompt (optionally under a system prompt) and record the answer."""
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
            # task.expected is carried through as well as the constraint pair:
            # the extraction tasks that moved in from `data_extract` in v0.13
            # keep their per-field expectations there.
            expected={**task.expected, "answer_type": task.answer_type,
                      "constraints": task.constraints},
            response_raw=content,
            turns=[TurnRecord(role="assistant", content=content,
                              completion_tokens=completion_tokens(response),
                              truncated=hit_length_cap(response))],
            completion_tokens=completion_tokens(response),
            truncated=hit_length_cap(response),
            raw_api_responses=[response] if client.save_responses else None,
        )
