"""Knowledge module: reasoning, factual recall, and calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import BaseModule, completion_tokens, hit_length_cap, message_text

if TYPE_CHECKING:
    from ..runner import ChatClient


class KnowledgeModule(BaseModule):
    """Tests reasoning, factual recall, and knowing-what-you-don't-know."""

    name = "knowledge"

    def recorded_expected(self, task: Task) -> dict[str, Any]:
        return {**task.expected, "answer_type": task.answer_type}

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Send the question and record the model's plain-text answer."""
        response = await client.chat([{"role": "user", "content": task.prompt}],
                                     max_tokens=task.max_tokens)
        message = response["choices"][0]["message"]
        content = message_text(message)
        return TaskResult(
            task_id=task.id,
            module=self.name,
            prompt=task.prompt,
            expected=self.recorded_expected(task),
            response_raw=content,
            turns=[TurnRecord(role="assistant", content=content,
                              completion_tokens=completion_tokens(response),
                              truncated=hit_length_cap(response))],
            completion_tokens=completion_tokens(response),
            truncated=hit_length_cap(response),
            raw_api_responses=[response] if client.save_responses else None,
        )
