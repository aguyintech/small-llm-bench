"""Module base class, task loading, and response parsing helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..models import Task, TaskResult, ToolCall, strip_reasoning
from .mock_registry import ALL_TOOL_SCHEMAS

# Modules whose tasks are one-shot "baseline" capability (anyone-competent passes);
# everything else defaults to the "hard" wheat/chaff tier.
_BASELINE_MODULES = {"knowledge", "format"}

# Per-module default completion-token caps, set from observed p95/p99 usage
# across 22 reference models (see the `items` command). v0.4 capped modules
# whose usage sits well below the global default — bounding worst-case
# latency without truncating legitimate output. These are per-task ``max_tokens``
# (Task.max_tokens, set from this table unless the task YAML overrides it) and
# win over BenchSettings.max_tokens in modules/*.py's
# ``client.chat(..., max_tokens=task.max_tokens)`` calls — so raising the
# default in config.py/.env does NOT raise these.
#
# An EXPLICIT --max-tokens does override them (v0.12), in both directions, and
# warns which caps it moved. Before that it was silently ignored here, which
# made a floor-model calibration run impossible to configure: a run started
# with --max-tokens 16384 still truncated knowledge at exactly 4096 on 21/21
# trials. Note also that the p95/p99 figures below were measured on models
# large enough never to truncate, so they are a latency bound fitted to strong
# models, not a difficulty setting — small models legitimately need more room.
# long_context and adversarial were left out of this table until v0.13, on the
# grounds that long_context completions are short despite a huge prompt
# (prefill, not decode, dominates its cost) and adversarial's p95 sits under
# its neighbours' caps. Leaving them out did not mean uncapped — it meant they
# silently inherited BENCH_MAX_TOKENS, which is a fallback nobody chose for
# either. Measured, long_context needed the most room of any module and
# adversarial the least, so the fallback was too small for one and too large
# for the other. Both are explicit now.
# multi_turn_if is raised well above the others: its multi-turn accumulated
# context pushed p99 to ~8600 tokens, well past the global default — a low cap
# there causes truncation-driven false failures, not a real runtime saving.
# format/knowledge/code/data_extract were raised again in the same direction
# (from 1536/2048/2048/1536) after auditing 23 models' saved results: these are
# the prose-generating modules, and thinking models spend a chunk of the same
# completion budget on the reasoning trace before ever reaching the answer —
# they accounted for the overwhelming majority (176/200) of all truncated
# trials across every past run.
# The four tool_* modules merged into `tools` in v0.10 at a single 2048 cap; the
# loop axis moves up from 1536, which is the safe direction (a budget only needs
# to be at least as generous as before) and cannot cause truncation failures.
# The tool_* modules looked immune to that only because they never recorded
# truncation: none of them called hit_length_cap, so every trial reported
# truncated=False no matter where it stopped. With that fixed, tool_simple's
# 768 turned out to be the tightest cap in the table and the one a reasoning
# model actually hits — it can spend the whole budget drafting a one-sentence
# answer and return empty content, which a no_call task scores as "never
# answered". Raised to 2048, in line with its tool_* neighbours.
# v0.13 refit the whole table against 10 models x 3 trials run at
# --max-tokens 16384 (2,812 trials), which is the first corpus large enough to
# fit them on. That corpus also revealed why it was overdue: every reference
# run to date had been made with an explicit 16384, so the defaults below had
# never actually been measured against this fleet. The first run that used them
# lost three tasks to truncation immediately.
#
# Share of single-request replies that fit, at the OLD cap for each module:
#
#     adversarial    8192 (inherited)   94.7%
#     code           4096               98.6%
#     format         4096               84.7%
#     knowledge      4096               85.7%
#     long_context   8192 (inherited)   84.7%
#
# format and knowledge were truncating one reply in seven. These are the
# prose-generating modules, and a model without a separate thinking channel
# spends the same completion budget on its reasoning trace before reaching the
# answer — three tasks were lost that way with the model reasoning correctly
# and being cut off before it could say so.
#
# Every value below now covers 100% of that corpus except knowledge (99.5%; its
# tail is a single 15k runaway). A cap is a ceiling, not a target — the median
# reply is 400-1100 tokens — so raising one costs wall-clock only on a model
# that actually runs long, and a model that runs long by looping is already
# scored zero by the degenerate-truncation check rather than paid for.
#
# tools and multi_turn_if sum completion_tokens across an episode's turns, so
# the pre-v1.0 corpus could not give a per-request figure for them.
#
# "Neither has ever recorded a truncation" stood here until v1.0 and was the
# same missing-instrument fallacy called out for the tool_* modules twenty
# lines up: multi_turn_if never called hit_length_cap, so it reported
# truncated=False no matter where a turn stopped. Fixed in v1.0. Any figure
# for these two modules has to come from a corpus recorded after that fix.
#
# v1.0 sweep, 507 tools trials over 13 models, is that corpus. The 47 trials
# that made exactly one call give an uncensored per-request read:
#
#     p50 228   p90 794   p95 2048   p99 2048   max 6144
#
# p95 and p99 sitting ON the cap is the tell: the distribution is right-
# censored, so the tail cannot be read off it at all. What the corpus does
# show is who the cap binds. Every truncation is a model under 8B or a
# reasoning-only MoE -- LFM2.5-8B-A1B lost five tasks, tst_35 all three
# trials, leaving it scored on 34 of 39; qwen3.5-0.8b reached det 0.60 on
# tst_41 three times running and was cut mid-answer three times running.
# Nothing at or above 12B ever touches 2048.
#
# 2048 was therefore not measuring tool use, it was measuring whether a model
# thinks in the completion channel before it calls. Doubled to 4096. A cap is
# a ceiling, not a target -- p50 is 228 -- so this bills wall-clock only on the
# trials that were being cut, and a model that runs long by looping is still
# scored zero by the degenerate check rather than paid for the room.
_MODULE_MAX_TOKENS: dict[str, int] = {
    "tools": 4096,
    "adversarial": 6144,
    "code": 6144,
    "multi_turn_if": 6144,
    "format": 8192, "knowledge": 8192,
    "long_context": 12288,
}


def infer_tier(module_name: str, raw: dict[str, Any]) -> str:
    """Default a task's tier when the YAML doesn't set one explicitly.

    code from-scratch is baseline; code repair (ships buggy_code) is hard.
    """
    if "tier" in raw:
        return raw["tier"]
    if module_name in _BASELINE_MODULES:
        return "baseline"
    if module_name == "tools":
        # One module, four axes: a single tool call is one-shot baseline
        # capability, while loop/state/discovery episodes compound.
        return "baseline" if raw.get("axis") == "call" else "hard"
    if module_name == "code":
        return "hard" if raw.get("buggy_code") else "baseline"
    return "hard"


_DIFFICULTY_BAND_FALLBACK = {"easy": "anchor", "medium": "mid", "hard": "hard"}


def infer_band(difficulty: str, raw: dict[str, Any]) -> str:
    """Default a task's calibration band when the YAML doesn't set one
    explicitly. Bands are assigned empirically from measured pass rates (see
    the `items` CLI command); until a task has been calibrated, fall back to
    a coarse mapping from its curated `difficulty` — not `tier`, which
    defaults to "baseline" for whole modules (e.g. format) regardless of how
    hard an individual task actually is."""
    if "band" in raw:
        return raw["band"]
    return _DIFFICULTY_BAND_FALLBACK.get(difficulty, "mid")

if TYPE_CHECKING:
    from ..runner import ChatClient


def find_tasks_dir() -> Path:
    """Locate the tasks/ directory (cwd first, then repo layout)."""
    candidates = [
        Path.cwd() / "tasks",
        Path(__file__).resolve().parents[2] / "tasks",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate the tasks/ directory")


def load_tasks(module_name: str, fast: bool = False, profile: str | None = None,
               tasks_dir: Path | None = None) -> list[Task]:
    """Load tasks for a module from its YAML file, filtering by profile.

    ``profile`` is one of "fast" | "full"; the legacy ``fast`` bool is kept
    for back-compat and is equivalent to ``profile="fast"``.
    """
    path = (tasks_dir or find_tasks_dir()) / f"{module_name}.yaml"
    data = yaml.safe_load(path.read_text())
    tasks = []
    for raw in data["tasks"]:
        tier = infer_tier(module_name, raw)
        band = infer_band(raw.get("difficulty", "medium"), raw)
        max_tokens = raw.get("max_tokens", _MODULE_MAX_TOKENS.get(module_name))
        tasks.append(Task(module=module_name, tier=tier, band=band,
                          max_tokens=max_tokens,
                          **{k: v for k, v in raw.items()
                             if k not in ("tier", "band", "max_tokens")}))
    resolved = profile or ("fast" if fast else "full")
    if resolved == "fast":
        tasks = [t for t in tasks if t.fast]
    return tasks


def build_tools_schema(tool_names: list[str]) -> list[dict[str, Any]]:
    """Build the OpenAI tools payload for the given mock tool names."""
    return [
        {"type": "function", "function": ALL_TOOL_SCHEMAS[name]}
        for name in tool_names
    ]


def completion_tokens(response: dict[str, Any]) -> int:
    """Read completion_tokens from an OpenAI-style usage block (0 if absent)."""
    usage = response.get("usage") or {}
    try:
        return int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _num(value: Any) -> float:
    """Coerce a JSON number to float, 0.0 for anything unusable."""
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def _cached_tokens(usage: dict[str, Any]) -> int:
    """Prompt tokens served from a prefix cache, per OpenAI's usage details."""
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    return int(_num(details.get("cached_tokens")))


def call_timings(response: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one response's server-reported timing into
    ``{prompt_tokens, cached_prompt_tokens, prefill_seconds,
    generation_seconds, source}``, or None when the response carries nothing.

    Three shapes, in order of precedence:

    * llama.cpp server puts a ``timings`` block beside ``usage`` on every
      non-streaming /chat/completions response: ``prompt_n`` (tokens actually
      evaluated), ``cache_n`` (reused from the prefix cache), ``prompt_ms``,
      ``predicted_ms``.
    * oMLX extends ``usage`` with ``prompt_eval_duration`` /
      ``generation_duration`` (seconds) — but only on the streaming usage
      chunk; its non-streaming responses fall through to the third case.
    * Anything else OpenAI-compatible: token counts only, no split.
    """
    timings = response.get("timings")
    if isinstance(timings, dict):
        prompt_n = int(_num(timings.get("prompt_n")))
        cache_n = int(_num(timings.get("cache_n")))
        if prompt_n or cache_n or timings.get("prompt_ms") is not None:
            return {
                "prompt_tokens": prompt_n,
                "cached_prompt_tokens": cache_n,
                "prefill_seconds": _num(timings.get("prompt_ms")) / 1000.0,
                "generation_seconds": _num(timings.get("predicted_ms")) / 1000.0,
                "source": "llamacpp",
            }

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    cached = _cached_tokens(usage)
    total_prompt = int(_num(usage.get("prompt_tokens")))
    if usage.get("prompt_eval_duration") is not None or \
            usage.get("generation_duration") is not None:
        # oMLX counts cached tokens inside prompt_tokens; subtract them so the
        # prefill rate reflects work actually done.
        return {
            "prompt_tokens": max(0, total_prompt - cached),
            "cached_prompt_tokens": cached,
            "prefill_seconds": _num(usage.get("prompt_eval_duration")),
            "generation_seconds": _num(usage.get("generation_duration")),
            "source": "omlx",
        }

    if not total_prompt and not cached:
        return None
    return {
        "prompt_tokens": max(0, total_prompt - cached),
        "cached_prompt_tokens": cached,
        "prefill_seconds": 0.0,
        "generation_seconds": 0.0,
        "source": "usage",
    }


def hit_length_cap(response: dict[str, Any]) -> bool:
    """True when the completion was cut off at the token cap.

    A truncated response never delivered a final answer, so text-scored
    modules must not award credit for values that only appear inside the
    (incomplete) reasoning dump.
    """
    choices = response.get("choices") or [{}]
    return choices[0].get("finish_reason") == "length"


def message_text(message: dict[str, Any]) -> str:
    """Extract a model's answer text, tolerant of reasoning ('thinking') models.

    Falls back to ``reasoning_content`` when ``content`` is empty (some servers,
    e.g. llama.cpp with --reasoning-format deepseek, put the final answer there
    or exhaust the budget on reasoning) and strips any inline reasoning block,
    including the unbalanced forms — see ``strip_reasoning``.
    """
    content = message.get("content") or ""
    if not content.strip():
        content = message.get("reasoning_content") or ""
    return strip_reasoning(content)


def message_reasoning(message: dict[str, Any]) -> str | None:
    """The server's separate reasoning channel, when it sent one.

    For the modules that score `message_text`, this is already folded into the
    answer (it is that function's fallback). It is recorded separately by the
    agentic-loop modules, which deliberately keep `content` raw and would
    otherwise persist nothing at all for a turn that spent its whole budget
    thinking — the exact turn whose text a truncation has to be classified on.
    """
    return message.get("reasoning_content") or None


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from an OpenAI-style assistant message."""
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function", {})
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))
    return calls


class BaseModule(ABC):
    """Base class for benchmark modules."""

    name: str

    def load(self, fast: bool = False, profile: str | None = None,
             tasks_dir: Path | None = None) -> list[Task]:
        """Load this module's tasks, filtered by profile (fast|full).

        ``tasks_dir`` overrides bank discovery, so a run can be pointed at a
        scratch bank holding a single candidate task without touching
        ``tasks/`` (see the ``probe`` workflow).
        """
        return load_tasks(self.name, fast=fast, profile=profile,
                          tasks_dir=tasks_dir)

    @abstractmethod
    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Execute one task against the model and return an unscored result."""
