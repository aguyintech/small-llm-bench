"""Shared fixtures for the small-llm-bench test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from small_llm_bench.models import ToolCall

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tasks_dir() -> Path:
    """Path to the repository tasks/ directory."""
    return PROJECT_ROOT / "tasks"


@pytest.fixture
def retired_bank_dir() -> Path:
    """Task bank holding ONLY the tasks retired in v0.15.

    A large part of the scorer suite pinned its behaviours to specific bank
    tasks, so the v0.15 cut (59 -> 40) orphaned 60 tests. The definitions are
    kept here unchanged and those tests load this directory, which preserves
    every assertion while letting the shipped bank shrink. Nothing here is run
    against a model or counted in any score.
    """
    return PROJECT_ROOT / "tests" / "fixtures" / "retired_bank"


@pytest.fixture
def make_call():
    """Factory for ToolCall objects."""

    def _make(tool: str, **arguments) -> ToolCall:
        return ToolCall(name=tool, arguments=arguments)

    return _make


@pytest.fixture
def chat_completion():
    """Factory for a minimal OpenAI-style chat completion payload."""

    def _make(content: str | None = None, tool_calls: list | None = None) -> dict:
        message: dict = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        return {"choices": [{"message": message}]}

    return _make
