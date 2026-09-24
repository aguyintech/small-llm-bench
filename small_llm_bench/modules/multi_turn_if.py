"""Multi-turn instruction-following module (Multi-IF / IFEval over a dialogue).

A 3-turn conversation where verifiable constraints *accumulate*: each turn adds
new rules, and every later turn's response must still satisfy all earlier rules.
This exposes instruction-forgetting — the small-model failure where turn 1 is
obeyed but the constraint is dropped two turns later (context rot). Graded
deterministically by the same verifiable-constraint engine as the format module;
success is binary (every turn satisfies its full accumulated constraint set).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import (BaseModule, completion_tokens, hit_length_cap,
                   message_text)

if TYPE_CHECKING:
    from ..runner import ChatClient


class MultiTurnIfModule(BaseModule):
    """Tests instruction following across an accumulating multi-turn dialogue."""

    name = "multi_turn_if"

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Walk the conversation turn by turn, keeping full history each call."""
        messages: list[dict[str, Any]] = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        turns: list[TurnRecord] = []
        raw_responses: list[dict[str, Any]] = []
        tokens = 0
        last_content = ""
        # Any turn cut off at the cap taints the episode, not just the last
        # one. This module went without the flag until v1.0, and the absence
        # read as evidence: a truncated turn was graded as a constraint
        # failure, and a turn cut short enough to be empty VACUOUSLY PASSES
        # every negative constraint (not_contains, max_words, max_bullets), so
        # the cap could move a score in either direction with nothing recording
        # that it had been hit at all.
        truncated = False

        for spec in task.conversation:
            messages.append({"role": "user", "content": spec.get("prompt", "")})
            response = await client.chat(messages, max_tokens=task.max_tokens)
            raw_responses.append(response)
            turn_cut = hit_length_cap(response)
            turn_tokens = completion_tokens(response)
            truncated = truncated or turn_cut
            tokens += turn_tokens
            message = response["choices"][0]["message"]
            content = message_text(message)
            messages.append({"role": "assistant", "content": content})
            turns.append(TurnRecord(role="assistant", content=content,
                                    completion_tokens=turn_tokens,
                                    truncated=turn_cut))
            last_content = content

        return TaskResult(
            task_id=task.id,
            module=self.name,
            prompt=task.prompt,
            expected={"conversation": task.conversation},
            response_raw=last_content,
            turns=turns,
            truncated=truncated,
            completion_tokens=tokens,
            raw_api_responses=raw_responses if client.save_responses else None,
        )
