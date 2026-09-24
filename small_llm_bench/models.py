"""Pydantic data models shared across the benchmark."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """One input/output pair used to verify a generated code function."""

    args: list[Any]
    expected: Any = None


class Task(BaseModel):
    """One benchmark task loaded from a YAML task file."""

    id: str
    module: str
    # Sub-axis within a module, for reporting only — never for grading. The
    # tools module spans four (call | loop | state | discovery) and prints them
    # as unweighted sub-rows so merging them into one weight didn't cost the
    # diagnostic breakdown.
    axis: str | None = None
    prompt: str
    fast: bool = False
    difficulty: str = "medium"
    tier: str = "baseline"  # baseline (one-shot) | hard (agentic/long/compounding)
    band: str = "mid"  # anchor | mid | hard | frontier — empirical pass-rate band
    max_tokens: int | None = None  # per-task completion cap; falls back to the run default
    tools: list[str] = Field(default_factory=list)
    expected: dict[str, Any] = Field(default_factory=dict)
    function_name: str | None = None
    test_cases: list[TestCase] = Field(default_factory=list)
    regression_cases: list[TestCase] = Field(default_factory=list)
    context_files: dict[str, str] = Field(default_factory=dict)
    support_code: str | None = None
    answer_type: str | None = None
    max_turns: int = 10
    # Code module only: how many execute-and-fix attempts the model gets. 1 is
    # the historical one-shot behavior; >1 feeds real test failures back and
    # applies the repair decay in score_code().
    repair_attempts: int = 1
    tool_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    parallel: list[dict[str, Any]] = Field(default_factory=list)
    content_checks: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    system_prompt: str | None = None
    buggy_code: str | None = None
    choices: list[str] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    haystack: dict[str, Any] = Field(default_factory=dict)
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    what_this_tests: str | None = None
    failure_mode: str | None = None


def world_content_hash() -> str:
    """Stable hash of the mock tool world (registry source).

    Deliberately coarse — one hash for the whole registry — because the thing it
    guards against is a silent mismatch, and per-tool granularity cannot see
    module-level data anyway: `db_query`'s source does not change when the table
    it reads gains rows. Coarse means an unrelated registry edit invalidates
    tool trials that could not have been affected; `--ignore-world-hash` is the
    escape hatch for when the author knows that, and the reuse report says which
    trials were dropped for this reason rather than folding them in silently.
    """
    from pathlib import Path
    source = Path(__file__).parent / "modules" / "mock_registry.py"
    try:
        payload = source.read_bytes()
    except OSError:  # pragma: no cover - only if the install is broken
        return ""
    return hashlib.sha256(payload).hexdigest()[:16]


# Bumped whenever the long-context haystack generator changes what it builds
# from an unchanged spec. The document is generated at runtime and so never
# reaches task_content_hash; without this, a generator change would silently
# leave stored trials looking reusable while the document underneath them had
# moved — the same failure world_content_hash exists to catch for tool tasks.
_HAYSTACK_GENERATOR = "salted-v2"


def task_world_hash(task: Task) -> str:
    """The world hash that applies to a task: the mock tool world for tool
    tasks, the haystack generator for long-context ones, empty otherwise."""
    if task.tools:
        return world_content_hash()
    if task.haystack:
        payload = f"{_HAYSTACK_GENERATOR}:{task.id}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]
    return ""


def task_content_hash(task: Task) -> str:
    """Stable hash of a task's semantic content, excluding the presentation-only
    `fast` flag — which profile a task appears in doesn't change what the task
    tests."""
    payload = task.model_dump_json(exclude={"fast"})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ToolCall(BaseModel):
    """A parsed tool call extracted from a model response."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TurnRecord(BaseModel):
    """One conversation turn recorded during task execution."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # The server's separate reasoning channel for this turn, when it sent one.
    # The agentic-loop modules record `content` raw rather than running it
    # through `message_text` (see tools.py), so that "answered the user" stays
    # distinguishable from "spent the budget thinking and returned nothing" —
    # which left the thinking itself unrecorded, and on a turn cut off at the
    # cap that is the only text there is. spark-x2.5-4b's tst_63 spent 16,629
    # completion tokens across an episode whose persisted assistant text was
    # zero characters long: `truncation_class` was handed an empty string and
    # returned `incomplete` because it had nothing to read, not because it had
    # read anything. None on files written before per-turn evidence existed.
    reasoning: str | None = None
    # Completion tokens this one turn spent. TaskResult.completion_tokens sums
    # the episode, so it cannot say which turn ran long. None on older files.
    completion_tokens: int | None = None
    # True when THIS turn is the one that hit the cap. The episode-level flag
    # is an OR across turns and so cannot point at the cut, which is the turn
    # whose text decides whether the model was looping or unfinished.
    # False on files written before this field existed — absence is not
    # evidence there, so the classifier falls back to reading every
    # assistant turn.
    truncated: bool = False


class ScorerResult(BaseModel):
    """Output of a deterministic scorer: a soft score in [0, 1], a strict
    binary success flag (drives pass^k reliability), plus a breakdown."""

    score: float
    success: bool = False
    breakdown: dict[str, Any] = Field(default_factory=dict)
    # True when the harness never got to measure the model: the sandbox could
    # not run the candidate at all. Scoring such a trial 0.0 grades the runner,
    # not the model — v1.0 recorded 15 of these as a flat 0/15 code module for
    # one model because a missing image read as fifteen wrong answers.
    # Propagated to TaskResult.infra_error, which `counted()` drops.
    infra_error: bool = False


class TaskResult(BaseModel):
    """Full record of a single executed task, including scores."""

    task_id: str
    module: str
    # Reporting sub-axis, copied from the task (see Task.axis). Empty on files
    # written before v0.10.
    axis: str | None = None
    prompt: str
    # The system prompt the model was actually shown. Model-visible stimulus,
    # so `rescore._stimulus_changed` has to be able to compare it — before
    # v0.15 it was not persisted, and a task whose system_prompt changed read
    # as a grading-only edit and was silently re-graded against responses
    # written for the old rules. None on files written before v0.15, which is
    # why that check fails closed rather than assuming "unchanged".
    system_prompt: str | None = None
    difficulty: str = "medium"
    tier: str = "baseline"
    band: str = "mid"
    tools_schema: list[dict[str, Any]] = Field(default_factory=list)
    expected: dict[str, Any] = Field(default_factory=dict)
    response_raw: str = ""
    turns: list[TurnRecord] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    det_score: float = 0.0
    success: bool = False
    det_success: bool = False  # deterministic pass, frozen before any judge override
    det_breakdown: dict[str, Any] = Field(default_factory=dict)
    completion_tokens: int = 0
    # Server-reported timing, summed over every call this task made (0 when the
    # backend returns none). Durations rather than tok/s: rates are derived at
    # report time so aggregation stays token-weighted instead of averaging
    # ratios. `prompt_tokens` counts tokens the server actually evaluated;
    # `cached_prompt_tokens` counts those served from its prefix cache, which
    # would otherwise inflate the prefill rate.
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    generation_seconds: float = 0.0
    timing_source: str = ""  # "" | "llamacpp" | "omlx" | "usage"
    truncated: bool = False  # response hit the token cap (finish_reason == "length")
    # Why it truncated: "" (not truncated) | "incomplete" (cut off mid-answer,
    # excluded from the pass rate as a coverage gap) | "degenerate" (cut off
    # while looping — a hard failure). Derived by the scorer from data already
    # stored, so `rescore` can recompute it for files written before this
    # existed; the "" default keeps those files loading.
    truncation_class: str = ""
    # Code module: how many execute-and-fix attempts were spent (1 == one-shot).
    attempts_used: int = 1
    # The endpoint rejected the prompt for exceeding its context window. Still a
    # failure (score 0), but reported apart from a wrong answer: it says the
    # deployment couldn't hold the input, not that the model read it and missed.
    context_overflow: bool = False
    # The server refused to parse the tool-call arguments the model emitted
    # (llama.cpp: raw newline inside a JSON string). Answered as a 500, which
    # would otherwise read as a server fault and EXCLUDE the trial — but the
    # request reached the model and the model's own output was rejected, so it
    # is a failure it earned. See runner._is_malformed_tool_call.
    malformed_tool_call: bool = False
    duration_seconds: float = 0.0
    llm_score: float | None = None
    llm_reasoning: str | None = None
    # The deterministic score the judge was shown when it formed `llm_score`.
    # The judge prompt anchors on that number and is told to keep it where it
    # agrees, so a verdict only means anything against it: once `det_score`
    # moves underneath (a re-score), the judge's AGREEMENT with the old number
    # reads as disagreement with the new one, which is the shape of a rescue.
    # Staleness has to live on the trial, not in the re-scoring pass that
    # noticed it — a second pass sees a settled score and no longer knows.
    judge_anchor_det: float | None = None
    # Set when the judge wanted to rescue this trial and was refused because
    # the deterministic record says the episode broke a hard rule (called a
    # forbidden tool, blew the turn budget, looped, was cut off mid-loop).
    # Recorded rather than dropped: a judge that keeps trying to rescue the
    # same gate is telling you something about the task, not about the model.
    judge_blocked_by: str | None = None
    error: str | None = None
    infra_error: bool = False
    raw_api_responses: list[dict[str, Any]] | None = None
    task_hash: str = ""
    # Hash of the mock world this trial ran against; empty for tasks that use no
    # tools. Separate from task_hash on purpose: a task can be untouched while
    # the world underneath it changes, and that used to be invisible. adv_11
    # went a whole sweep reusing trials from before its table existed, so every
    # model "correctly" reported nothing to delete and the task scored -0.500
    # discrimination on data that described a world that no longer existed.
    world_hash: str = ""

    @property
    def timed_out(self) -> bool:
        """True when this task failed due to an httpx timeout."""
        return bool(self.error) and "Timeout" in self.error.split(":", 1)[0]


class BenchMeta(BaseModel):
    """Metadata about a benchmark run."""

    model: str
    endpoint: str
    timestamp: str
    duration_seconds: float
    bench_version: str
    fast: bool = False
    profile: str = "full"  # fast | full
    trials: int = 1
    thinking: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # What the sampler actually was. `temperature: None` means "the server
    # applied its own default", which is not a value and must not be read as
    # one — `sampling` says so in words so a stored file is self-describing.
    # A run whose sampler is unknown is not reproducible and its pass^k is not
    # comparable to another run's, so this is a comparability field.
    seed: int | None = None
    sampling: str = ""
    # Which sandbox actually executed the code tasks. docker/podman run
    # python:3.12-slim; bwrap/sandbox-exec/rlimit run the host interpreter, so
    # two runs can disagree on a code task for reasons that are not the model.
    sandbox_backend: str = ""
    python_version: str = ""
    # True when this run was started with an explicit --max-tokens, which
    # overrides the per-module caps. Without it `max_tokens` is ambiguous
    # between "8192 default, tools actually ran at 2048" and "8192 forced
    # everywhere", and _config_matches needs to tell those apart.
    max_tokens_override: bool = False
    # Requests in flight during the run. Decode speed under a batching server
    # depends on it, so timing numbers from two runs only compare at equal
    # concurrency. 0 on files written before v0.9.
    concurrency: int = 0
    # Identity of the task bank this run was scored against, so two runs can be
    # checked for comparability without diffing every per-result hash. Empty on
    # files written before v0.7 — recompute with results_task_set_hash().
    # Per-request context window the endpoint reported at run start, 0 when
    # it could not be discovered. llama.cpp divides --ctx-size across
    # --parallel slots, so two servers started from the same --ctx-size can
    # serve wildly different windows; a run against a small one fails long
    # tasks for reasons that have nothing to do with the model. Comparability
    # field for that reason.
    served_context: int = 0
    task_set_hash: str = ""
    task_count: int = 0


def results_task_set_hash(results: list["TaskResult"]) -> str:
    """Stable hash of the task bank a run actually covered.

    Derived from the per-result ``task_hash`` rather than from the selected
    task list, so it stays correct under ``--only-new``/``--add-trials`` (where
    the file carries passthrough results too) and can be recomputed for older
    result files that predate ``BenchMeta.task_set_hash``.
    """
    ids = sorted({f"{r.module}:{r.task_id}:{r.task_hash}" for r in results})
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


class BenchResult(BaseModel):
    """A complete benchmark run: metadata plus all task results."""

    meta: BenchMeta
    results: list[TaskResult]


# A reasoning block the server left inline in `content`. Both halves are
# optional in practice, which is the whole reason this is not one regex:
# llama.cpp with `--reasoning-format none` emits the pair, some servers strip
# the opener and forward only the closer, and a turn cut at the token cap ends
# mid-thought with an opener and no closer at all.
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove an inline reasoning block, balanced or not.

    The balanced form was handled from the start; the unbalanced ones were not,
    and `<think>.*?</think>` silently matches neither. A stray closer means the
    text before it is thinking (the opener was stripped upstream) and a stray
    opener means the text after it is thinking (the turn was cut mid-thought).
    Getting either wrong hands a constraint checker a reasoning dump to grade,
    and a model reasoning out loud recites the rules it is tracking.
    """
    text = _THINK_PAIR_RE.sub("", text)
    close = _THINK_CLOSE_RE.search(text)
    if close:
        text = text[close.end():]
    open_ = _THINK_OPEN_RE.search(text)
    if open_:
        text = text[:open_.start()]
    return text.strip()
