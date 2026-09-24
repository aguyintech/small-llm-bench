"""Async task executor: concurrency control, retry, progress reporting."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import platform
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn)

from rich.console import Console

from . import __version__
from .config import BenchSettings
from .models import (BenchMeta, BenchResult, Task, TaskResult,
                     results_task_set_hash, task_content_hash, task_world_hash)
from .modules.adversarial import AdversarialModule
from .modules.base import BaseModule, call_timings
from .modules.code import CodeModule
from .modules.format import FormatModule
from .modules.knowledge import KnowledgeModule
from .modules.long_context import LongContextModule
from .modules.multi_turn_if import MultiTurnIfModule
from .modules.tools import ToolsModule
from .scorer import score_task, truncation_class


class DeadEndpointError(RuntimeError):
    """Every one of the first ``sanity_check_after`` trials produced nothing.

    Raised to abort a run early rather than spend the whole budget against a
    misconfigured endpoint (wrong model id, no tool support, a server that
    answers 200 with an empty completion).
    """


def checkpoint_path_for(output: Path) -> Path:
    """Sidecar path for a run writing to ``output``."""
    return output.with_suffix(output.suffix + ".partial.jsonl")


def _checkpoint_fingerprint(settings: BenchSettings, profile: str,
                            temperature: float | None,
                            base_seed: int | None,
                            max_tokens_explicit: bool) -> dict[str, Any]:
    """Everything that must match for recovered trials to be blendable.

    The same fields `_config_matches` guards for `--only-new`, plus the profile
    and seed. A recovered trial is a trial like any other: it may only rejoin a
    run that would have produced it.
    """
    return {"model": settings.model, "endpoint": settings.endpoint,
            "profile": profile, "thinking": settings.thinking,
            "temperature": temperature, "seed": base_seed,
            "max_tokens": settings.max_tokens,
            "max_tokens_override": max_tokens_explicit,
            "bench_version": __version__}


class TrialCheckpoint:
    """Appends each finished trial to a JSONL sidecar as the run proceeds.

    A run wrote its results exactly once, at the end. A 79-minute sweep
    interrupted at minute 78 — Ctrl-C, an OOM kill, a laptop suspending, the
    endpoint dying — produced nothing at all, and the fix was to run the whole
    thing again.

    One JSON object per line, flushed per trial, so the file is readable after
    any kind of death: a torn final line is discarded on load and everything
    before it survives. The first line is a header carrying the run's
    fingerprint, which is what makes recovery safe — trials are only rejoined
    to a run configured the way the one that produced them was.

    Deleted once the real results file is written. Its presence means a run
    did not finish.
    """

    def __init__(self, path: Path, fingerprint: dict[str, Any]) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self._handle: Any = None

    def open(self) -> "TrialCheckpoint":
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        if fresh:
            self._write({"__checkpoint__": self.fingerprint})
        return self

    def _write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(payload) + "\n")
        # flush() alone leaves the line in the OS page cache, which survives a
        # killed process but not a power loss or a hard suspend. fsync is a few
        # ms against trials that take seconds to minutes.
        self._handle.flush()
        try:
            os.fsync(self._handle.fileno())
        except OSError:  # pragma: no cover - not every filesystem supports it
            pass

    def append(self, result: TaskResult) -> None:
        self._write(result.model_dump(mode="json"))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def discard(self) -> None:
        """Drop the sidecar — the real results file now holds everything."""
        self.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "TrialCheckpoint":
        return self.open()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def load_checkpoint(path: Path,
                    fingerprint: dict[str, Any]) -> list[TaskResult]:
    """Trials recovered from an interrupted run, or [] if there are none.

    Returns nothing rather than raising for every reason a sidecar might not
    apply — absent, empty, headerless, written by a different config or a
    different bench version. Recovery is an optimisation; a wrong recovery is
    a corrupted result file, so every ambiguity resolves to "start over".
    """
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if not lines:
        return []
    try:
        header = json.loads(lines[0])
    except ValueError:
        return []
    if header.get("__checkpoint__") != fingerprint:
        return []
    recovered = []
    for line in lines[1:]:
        try:
            recovered.append(TaskResult.model_validate_json(line))
        except ValueError:
            # A torn last line is the expected shape of an interrupted write.
            # Everything before it is intact, so keep it and stop here.
            break
    return recovered


def _resume_meta(settings: BenchSettings, profile: str,
                 temperature: float | None, base_seed: int | None,
                 max_tokens_explicit: bool, trials: int) -> BenchMeta:
    """A stand-in meta for recovered trials on a run with no previous file.

    `_plan_work` reads only the results; this exists so `previous` is a
    well-formed BenchResult. It is never persisted — the meta written at the
    end is built fresh from the finished run.
    """
    return BenchMeta(
        model=settings.model, endpoint=settings.endpoint,
        timestamp=datetime.now(timezone.utc).isoformat(), duration_seconds=0.0,
        bench_version=__version__, profile=profile, trials=trials,
        thinking=settings.thinking, temperature=temperature, seed=base_seed,
        max_tokens=settings.max_tokens,
        max_tokens_override=max_tokens_explicit)


class UndersizedContextError(RuntimeError):
    """The endpoint's served context cannot hold tasks this run would send.

    Raised BEFORE any work, because the alternative is what v1.0 measured: a
    prompt longer than the window is rejected by the server, scored 0, and
    published as a capability result for a model that never saw the task.
    """


# Where an OpenAI-compatible server reports the context it was started with.
# llama.cpp answers /props directly; llama-swap fronts many llama-servers and
# routes per model under /upstream/<model>/. Anything else answers neither and
# the preflight simply does not run — an unknown window is not an error.
_PROPS_PATHS = ("/props", "/upstream/{model}/props")


async def discover_served_context(client: httpx.AsyncClient, base_url: str,
                                  model: str) -> int | None:
    """The per-request context window the endpoint serves, or None if unknown.

    llama.cpp reports the PER-SEQUENCE window here, which is what a single
    request may actually use: with `--parallel N` the KV cache is statically
    partitioned and each slot holds `--ctx-size / N`. That division is the
    whole reason this function exists. On the v1.0 sweep qwen3.5-4b ran
    `--parallel 4` against `--ctx-size 65536` and served 16,384 per request
    while twelve other models ran `--parallel 1` and served 65,536. Nothing in
    a result file distinguished those two deployments, so the board compared a
    model given a quarter of the context against twelve that were not.

    Best-effort by design: every failure path returns None, because refusing to
    run against a server that merely declines to introspect would be worse than
    the problem.
    """
    root = base_url.rstrip("/")
    for suffix in ("/v1", "/v1/"):          # the chat path, not the server root
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    for path in _PROPS_PATHS:
        try:
            response = await client.get(root + path.format(model=model),
                                        timeout=httpx.Timeout(180.0, connect=10.0))
            if response.status_code != 200:
                continue
            settings = response.json().get("default_generation_settings") or {}
            n_ctx = settings.get("n_ctx")
            if isinstance(n_ctx, int) and n_ctx > 0:
                return n_ctx
        except (httpx.HTTPError, ValueError, AttributeError):
            continue
    return None


def oversized_tasks(selected: list[tuple[BaseModule, Task]],
                    served_context: int) -> list[tuple[str, int]]:
    """Tasks whose prompt plus generation budget cannot fit ``served_context``.

    The prompt half is known only for haystack tasks, where the bank declares
    ``filler_tokens`` — and those are the only tasks big enough to matter; the
    rest of the bank's prompts are a few hundred tokens against caps that this
    check already counts in full.

    ``filler_tokens`` is a target the generator hits approximately and every
    tokenizer disagrees about by up to ~26% across this fleet, so the estimate
    is deliberately not padded: it is a floor. A task flagged here cannot fit.
    A task not flagged still might not.
    """
    over = []
    for _module, task in selected:
        filler = int(task.haystack.get("filler_tokens") or 0)
        needed = filler + int(task.max_tokens or 0)
        if needed > served_context:
            over.append((task.id, needed))
    return sorted(over, key=lambda pair: -pair[1])


def _is_dead_trial(result: TaskResult) -> bool:
    """True when a trial produced nothing any scorer could read.

    Not the same as a failing trial. A wrong answer is signal, and so is a
    context overflow — long_context deliberately grades a model whose window is
    too small as a failure — so neither counts here. This is the shape a broken
    endpoint produces on every task regardless of what was asked: a transport or
    parse error, or an empty completion with no tool calls.
    """
    if result.context_overflow:
        return False
    if result.error:
        return True
    if any(turn.tool_calls for turn in result.turns):
        return False
    return not result.response_raw.strip()


def all_modules() -> list[BaseModule]:
    """Instantiate all benchmark modules in canonical order."""
    return [ToolsModule(), CodeModule(),
            KnowledgeModule(), FormatModule(), LongContextModule(),
            MultiTurnIfModule(), AdversarialModule()]


# HTTP statuses that signal a transient server/infra problem (not the model's
# fault): rate limiting and the standard 5xx gateway/server errors.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Statuses that mean the request was never going to work: a model name the
# server does not serve, a wrong base path, missing or rejected credentials.
# Deliberately NOT in _TRANSIENT_STATUS — retrying cannot help, and
# `_is_retryable` would otherwise spend three attempts and two backoffs to
# re-learn it. They are still outside the model's scope, so they are excluded
# from scoring rather than counted as a failure: a run that ended against a
# swapped-out model recorded two 404s as two wrong answers.
_CONFIG_STATUS = {401, 403, 404}


# Phrases every common OpenAI-compatible server uses when the prompt exceeds
# the served context window. Matching them keeps "your deployment can't hold
# this input" visibly distinct from "the model read it and answered wrong" —
# both fail, but only one of them is about the model.
#
# "context size" was added in v1.0: llama.cpp's own wording is "the request
# exceeds the available context size", which matched none of the others and
# would have been filed as a plain unexplained error.
_CONTEXT_OVERFLOW_PATTERNS = ("context length", "context window", "maximum context",
                              "context size", "too long", "n_ctx",
                              "exceeds the model")


def _is_context_overflow(exc: BaseException) -> bool:
    """True when the endpoint rejected the prompt for being longer than the
    context window it was served with."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    try:
        body = exc.response.text.lower()
    except (UnicodeDecodeError, httpx.ResponseNotRead):
        return False
    return any(p in body for p in _CONTEXT_OVERFLOW_PATTERNS)


# A 500 whose body says the server could not PARSE what the model emitted.
# llama.cpp raises this when a tool call's arguments are not valid JSON.
_MALFORMED_TOOL_CALL_PATTERNS = ("failed to parse tool call",
                                 "tool call arguments as json",
                                 "parse error at line")


def _is_malformed_tool_call(exc: BaseException) -> bool:
    """True when the server rejected the MODEL'S OUTPUT, not the request.

    Measured on LFM2.5-8B-A1B, 2026-09-09. It emitted a `write_file` call whose
    `content` argument held a raw newline inside a JSON string:

        Failed to parse tool call arguments as JSON: [json.exception.
        parse_error.101] parse error at line 2, column 0: invalid string:
        control character U+000A (LF) must be escaped to \u000A or \n;
        last read: '"# 2026-08-24<U+000A>'

    That is invalid JSON, so llama.cpp answers 500 — and a 500 is otherwise a
    server fault, which had the trial excluded from scoring entirely. Three
    tasks (tst_35, tst_51, tst_57) vanished from that model's bank across two
    separate runs, always the same three, always the same byte, each retried
    three times. Nothing else on the same endpoint produced it.

    Emitting well-formed arguments IS the capability the tool modules measure,
    so this is a failure the model earned. Treating it as infra flattered
    exactly the model that could not do the task — the same defect the
    `incomplete`-truncation exclusion had.

    Deliberately narrow: a bare 500 stays transient. Only a body naming a
    parse failure counts, because only that says the request reached the model
    and the model's own output was the problem.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code != 500:
        return False
    try:
        body = exc.response.text.lower()
    except (UnicodeDecodeError, httpx.ResponseNotRead):
        return False
    return any(p in body for p in _MALFORMED_TOOL_CALL_PATTERNS)


def _is_config_error(exc: BaseException) -> bool:
    """True when the endpoint could never have served this request.

    A 404 is the one that actually happened: a run whose model was swapped off
    the port mid-sweep took the last two trials as 404s, and both were recorded
    as the model answering wrong. Unlike `_is_transient` this decides EXCLUSION
    ONLY — see `_is_retryable`, which these statuses must stay out of, because
    no number of retries conjures a model the server is not serving.
    """
    return (isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code in _CONFIG_STATUS)


def _error_text(exc: BaseException) -> str:
    """`Type: message`, with the server's body appended for HTTP errors.

    Without the body an HTTPStatusError records only "Server error '500'" and
    a link to MDN, which is exactly nothing: the v1.0 sweep hit three
    reproducible 500s whose cause could only be found by reading llama.cpp's
    own log on another machine. The body is where the server says why.
    """
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.text.strip()
        except (UnicodeDecodeError, httpx.ResponseNotRead):
            body = ""
        if body:
            text += f"\nserver said: {body[:600]}"
    return text


def _is_transient(exc: BaseException) -> bool:
    """True for errors outside the model's scope: timeouts, connection loss,
    and transient server statuses. Such failures are excluded from scoring
    rather than counted as a model failure.

    Note this decides EXCLUSION, not retry — see ``_is_retryable``. A read
    timeout is still not the model's fault, so it still belongs out of the
    score; it just must not be attempted again."""
    if _is_malformed_tool_call(exc):
        return False        # the model's output, not the server's health
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _is_retryable(exc: BaseException) -> bool:
    """True for failures a second identical request might actually survive.

    Everything ``_is_transient`` covers EXCEPT a read timeout. A read timeout
    means the generation did not finish inside the wall clock, and generation
    length is a property of the model and the prompt, not of the network — so
    the retry runs the same work and blows the same budget. Measured
    2026-09-08: two probe trials cost exactly 546s each, which is
    3 x 180s timeout + 2s + 4s backoff, to arrive at a failure the first
    attempt had already established in 180.
    """
    if isinstance(exc, httpx.ReadTimeout):
        return False
    return _is_transient(exc)   # a malformed tool call is not transient either:
                                # the model re-emits the same bytes every time.


class TimingAccumulator:
    """Sums per-call server-reported timing over one task's LLM calls.

    Aggregates only: a task that makes five calls with growing prefixes reports
    one prefill total, not a per-call curve. ``source`` records which backend
    shape produced the numbers, or "mixed" if a task somehow saw two.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.prefill_seconds = 0.0
        self.generation_seconds = 0.0
        self.source = ""

    def add(self, response: dict[str, Any]) -> None:
        """Fold one response's timing in; a response without any is ignored."""
        timing = call_timings(response)
        if timing is None:
            return
        self.prompt_tokens += timing["prompt_tokens"]
        self.cached_prompt_tokens += timing["cached_prompt_tokens"]
        self.prefill_seconds += timing["prefill_seconds"]
        self.generation_seconds += timing["generation_seconds"]
        source = timing["source"]
        self.source = source if self.source in ("", source) else "mixed"

    def apply(self, result: TaskResult) -> None:
        """Copy the totals onto a finished (or failed) task result."""
        result.prompt_tokens = self.prompt_tokens
        result.cached_prompt_tokens = self.cached_prompt_tokens
        result.prefill_seconds = round(self.prefill_seconds, 4)
        result.generation_seconds = round(self.generation_seconds, 4)
        result.timing_source = self.source


# Set per task in _execute_task. asyncio.gather hands each child coroutine its
# own copy of the context, so concurrent tasks accumulate independently without
# locking and without threading an argument through every module.
_TIMINGS: contextvars.ContextVar[TimingAccumulator | None] = \
    contextvars.ContextVar("bench_timings", default=None)

# The seed for the trial currently executing, set the same way and for the same
# reason: gather() gives each child its own copy of the context, so trials can
# carry different seeds without threading an argument through every module's
# run_task signature. None means "send no seed".
_SEED: contextvars.ContextVar[int | None] = \
    contextvars.ContextVar("bench_seed", default=None)


class ChatClient:
    """Async client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, endpoint: str, model: str, timeout: float = 60.0,
                 api_key: str = "", save_responses: bool = False,
                 max_tokens: int = 4096, temperature: float | None = None,
                 thinking: bool | None = None,
                 max_attempts: int = 3, retry_backoff: float = 1.0,
                 max_tokens_override: bool = False,
                 connect_timeout: float = 10.0,
                 min_generation_tok_s: float = 20.0,
                 prefill_allowance: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Create a client; `transport` allows injection of mocks in tests."""
        self.timeout_floor = timeout
        self.connect_timeout = connect_timeout
        self.min_generation_tok_s = max(1.0, min_generation_tok_s)
        self.prefill_allowance = prefill_allowance
        self.model = model
        self.save_responses = save_responses
        self.max_tokens = max_tokens
        self.max_tokens_override = max_tokens_override
        self.temperature = temperature
        self.thinking = thinking
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff = retry_backoff
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            timeout=self._timeout_for(max_tokens),
            headers=headers, transport=transport,
        )

    def _timeout_for(self, cap: int) -> httpx.Timeout:
        """Per-request timeout, derived from the token cap it was sent with.

        Connect stays short so a dead endpoint fails fast; read scales with
        how much generation the request actually permits. One global value
        cannot do both, and choosing it wrong in either direction is
        expensive: too short kills legitimate long generations and EXCLUDES
        them from the score (biasing in favour of the slow model), too long
        makes a misconfigured endpoint take tens of minutes to detect.
        """
        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout_for(cap),
            write=30.0, pool=self.connect_timeout,
        )

    def read_timeout_for(self, cap: int) -> float:
        """Seconds to wait for a completion of at most ``cap`` tokens."""
        return max(self.timeout_floor,
                   cap / self.min_generation_tok_s + self.prefill_allowance)

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None,
                   max_tokens: int | None = None) -> dict[str, Any]:
        """POST a chat completion; retry transient errors with exponential
        backoff, then raise. Non-transient errors (4xx, parse) raise at once.

        ``max_tokens`` overrides the client default for this call (a task's
        per-task cap, when set) — unless ``max_tokens_override`` is set, which
        is how an EXPLICIT --max-tokens wins. The per-module caps in
        modules/base._MODULE_MAX_TOKENS are calibrated from measured p95/p99
        usage and are the right default, but they were fitted on models large
        enough never to truncate; a floor-model calibration run needs to raise
        them, and before v0.12 the flag to do that was silently ignored by the
        six capped modules."""
        cap = (self.max_tokens if self.max_tokens_override
               else (max_tokens or self.max_tokens))
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "max_tokens": cap}
        seed = _SEED.get()
        if seed is not None:
            payload["seed"] = seed
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if tools:
            payload["tools"] = tools
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(
                    "/chat/completions", json=payload,
                    timeout=self._timeout_for(cap))
                response.raise_for_status()
                body = response.json()
                accumulator = _TIMINGS.get()
                if accumulator is not None:
                    accumulator.add(body)
                return body
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                if attempt == self.max_attempts - 1 or not _is_retryable(exc):
                    raise
                await asyncio.sleep(self.retry_backoff * 2 ** attempt)

        raise httpx.TimeoutException("unreachable")  # pragma: no cover

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _config_matches(meta: BenchMeta, settings: BenchSettings,
                    max_tokens_explicit: bool = False) -> bool:
    """True when a previous run's recorded config matches this run's.

    ``max_tokens`` only needs to be at least as generous as before — a bigger
    budget can't invalidate trials that already completed within the smaller
    one, so it's still safe to reuse them (this is what lets ``--only-new``
    retry old truncated trials against a raised default without discarding
    every other already-good trial in the file).

    That invariant needs one guard now that --max-tokens can override the
    per-module caps. An overriding run's trials ran at the global cap in EVERY
    module; a non-overriding run's ran at the tighter per-module cap. Pooling
    the first into the second reuses trials that had a more generous budget
    than anything this run will get — the exact direction the rule above
    forbids. The reverse stays fine.
    """
    if meta.max_tokens_override and not max_tokens_explicit:
        return False
    temp_matches = (meta.temperature is None and settings.temperature is None) or (
        meta.temperature is not None and settings.temperature is not None
        and abs(meta.temperature - settings.temperature) < 1e-9)
    return (meta.model == settings.model and meta.endpoint == settings.endpoint
            and meta.thinking == settings.thinking and temp_matches
            and meta.max_tokens is not None
            and meta.max_tokens <= settings.max_tokens)


def _stale_world_tasks(selected: list[tuple[BaseModule, Task]],
                       previous: BenchResult | None,
                       skipped: bool) -> set[str]:
    """Task ids whose stored trials ran against a different mock world."""
    if previous is None or skipped:
        return set()
    current = {(m.name, t.id): task_world_hash(t) for m, t in selected}
    return {r.task_id for r in previous.results
            if r.world_hash
            and (r.module, r.task_id) in current
            and r.world_hash != current[(r.module, r.task_id)]}


def _sampling_description(temperature: float | None,
                          seed: int | None) -> str:
    """One human-readable line saying what sampler produced a result file.

    `temperature: None` in a stored file is ambiguous between "greedy" and
    "whatever this server defaults to", and pass^k is a statement about a
    sampler. Recording the words removes the ambiguity for a reader who was
    not at the machine.
    """
    temp = ("server-default" if temperature is None
            else f"temperature={temperature:g}")
    return f"{temp}, " + ("no seed sent" if seed is None else f"seed={seed}+trial")


def _add_trials_target(previous: BenchResult, add_trials: int) -> int:
    """The uniform per-task trial count an ``--add-trials`` run aims for."""
    return (previous.meta.trials or 0) + add_trials


def _observed_trials(results: list[TaskResult], requested: int) -> int:
    """The trial count this file was actually run at, capped at ``requested``.

    A file records one ``trials`` for the whole bank and `pass_hat_k` skips any
    task holding fewer than k scorable trials, so this number decides which
    tasks reach the headline at all.

    The MODE of the per-task attempt count, for two failures on either side of
    it:

    * The minimum of SCORABLE trials was used until v1.0, and one dead request
      took the whole file down a rung: a single ReadTimeout on de_07 left that
      task 2 scorable trials, relabelled all 39 tasks k=2, and — pass^2 > pass^3
      always — lifted qwen3.6-27b from 0.912 to 0.930 and 4th place to 2nd.
      A board position became a function of the network.
    * The minimum of ATTEMPTS has the mirror failure: one task holding a single
      record (a filtered top-up, a partial file) drags k to 1 and hands the
      whole bank pass^1.

    The mode is unmoved by either. A run of 39 tasks at 3 trials records 3 even
    with a task short by one, and that task alone is dropped by `pass_hat_k`
    and reported in the row's ``tasks_excluded`` — a coverage gap confined to
    the task that has it. A file where every task really does hold 2 still
    records 2, which is the case that stops a genuinely short file scoring 0.0.

    Ties break HIGH, because the two errors are not symmetric. Recording a k
    that is too high drops the tasks that cannot honour it and says so in
    ``tasks_excluded``; recording one that is too low is an invisible,
    systematic lift across every task in the file. Only an unambiguous
    majority of short tasks lowers k, which is the case that stops a genuinely
    short file scoring 0.0.
    """
    per_task: dict[tuple[str, str], int] = {}
    for r in results:
        per_task[(r.module, r.task_id)] = per_task.get((r.module, r.task_id), 0) + 1
    if not per_task:
        return requested
    counts: dict[int, int] = {}
    for n in per_task.values():
        counts[n] = counts.get(n, 0) + 1
    mode = max(counts, key=lambda n: (counts[n], n))
    return max(1, min(requested, mode))


def _plan_work(selected: list[tuple[BaseModule, Task]],
              previous: BenchResult | None, trials: int,
              reusable: bool, skip_hash_check: bool = False,
              add_trials: int | None = None,
              skip_world_check: bool = False,
              ) -> tuple[list[TaskResult], list[tuple[BaseModule, Task]],
                        list[TaskResult]]:
    """Split ``selected`` tasks into (kept, work, passthrough).

    ``kept``: previously-recorded trials reused as-is (not infra-errored, not
    truncated), capped at ``trials`` per task. Unchanged task content is also
    required unless ``skip_hash_check`` (``--ignore-task-hash``) — that mode
    trusts ``(module, task_id)`` identity alone, so a task-bank-wide change
    that doesn't affect this task's semantics (e.g. bumping a module's
    ``max_tokens`` cap in `_MODULE_MAX_TOKENS`, which is baked into every
    task's content hash) doesn't force every task in the module to re-run —
    only the ones that were actually truncated or missing.
    ``work``: (module, task) pairs still needing execution to reach ``trials``
    — or, with ``add_trials`` set, to reach one UNIFORM target of
    ``previous.meta.trials + add_trials`` for every task.

    That target used to be per-task (``len(matches) + add_trials``), which
    looked additive and was quietly destructive: a task that had lost a trial
    to truncation ended below the ``meta.trials`` the run went on to record,
    and ``pass_hat_k`` drops any task with ``n < k`` outright. One truncated
    trial therefore deleted a whole task from the headline, and a file where
    every task was short scored 0.0. Topping up to a common k also re-runs the
    truncated trial, which is what a user asking for another trial wants.
    ``passthrough``: previous results for tasks outside ``selected`` — carried
    forward untouched so accumulated coverage isn't lost across partial runs.
    """
    selected_keys = {(module.name, task.id) for module, task in selected}
    prior_by_task: dict[tuple[str, str], list[TaskResult]] = {}
    if previous is not None:
        for r in previous.results:
            prior_by_task.setdefault((r.module, r.task_id), []).append(r)

    target = (_add_trials_target(previous, add_trials)
              if add_trials is not None and previous is not None else trials)

    kept: list[TaskResult] = []
    work: list[tuple[BaseModule, Task]] = []
    for module, task in selected:
        current_hash = task_content_hash(task)
        # A task can be untouched while the world underneath it changes, and
        # that used to be invisible: adv_11 reused a whole sweep of trials from
        # before its table existed. An empty stored world_hash is a pre-v0.11.2
        # file, which we cannot check and do not punish.
        current_world = task_world_hash(task)
        matches = [
            r for r in prior_by_task.get((module.name, task.id), [])
            if (skip_hash_check or r.task_hash == current_hash)
            and (skip_world_check or not r.world_hash
                 or r.world_hash == current_world)
            and not r.infra_error and not r.truncated
        ] if reusable else []
        reuse = matches[:target]
        kept.extend(reuse)
        for _ in range(target - len(reuse)):
            work.append((module, task))

    passthrough = []
    if previous is not None:
        passthrough = [r for r in previous.results
                       if (r.module, r.task_id) not in selected_keys]
    return kept, work, passthrough


def _select_tasks(profile: str, task_filter: str | None,
                  modules: list[str] | None,
                  tasks_dir: Path | None = None
                  ) -> list[tuple[BaseModule, Task]]:
    """Resolve which (module, task) pairs a run covers.

    ``modules``, if given, restricts to those exact module names — AND-combined
    with ``task_filter``'s substring match on task id/module name. Unlike
    ``task_filter`` alone, an exact module list doesn't also pull in modules
    sharing a name prefix (e.g. ``tool_``).

    ``tasks_dir`` overrides bank discovery. Only modules named in ``modules``
    are loaded, so a scratch bank needs just that module's yaml, not all seven.
    """
    module_set = set(modules) if modules else None
    if module_set:
        known = {m.name for m in all_modules()}
        unknown = module_set - known
        if unknown:
            raise ValueError(f"unknown module(s): {', '.join(sorted(unknown))}")

    selected = [
        (module, task)
        for module in all_modules()
        if module_set is None or module.name in module_set
        for task in module.load(profile=profile, tasks_dir=tasks_dir)
        if task_filter is None
        or task_filter in task.id or task_filter in module.name
    ]
    if not selected:
        raise ValueError(f"no tasks match filter: modules={modules}, "
                         f"filter={task_filter!r}")
    return selected


class PauseController:
    """Lets the user pause/resume the run by pressing 'p' in the terminal.

    In-flight tasks are unaffected by a pause; only tasks that haven't yet
    started work are held back. A no-op when stdin isn't a tty (piped or
    non-interactive runs), so headless usage is unaffected.

    Tracks how long the run spent held so `meta.duration_seconds` can exclude
    it. Per-task `duration_seconds` needs no such correction: the gate is
    awaited BEFORE `_execute_task` starts its timer, so no task is ever
    running while paused. The run-level figure is wall clock, though, and a
    coffee break used to land in it — which made a paused run look like a slow
    model against the runtime budget the bank is calibrated to.
    """

    def __init__(self, console: Console) -> None:
        self.running = asyncio.Event()
        self.running.set()
        self._console = console
        self._fd: int | None = None
        self._old_settings: list | None = None
        self._paused_at: float | None = None
        self._paused_total = 0.0

    @property
    def paused_seconds(self) -> float:
        """Total time held, including a pause still open right now."""
        open_interval = (0.0 if self._paused_at is None
                         else time.monotonic() - self._paused_at)
        return self._paused_total + open_interval

    def _on_key(self) -> None:
        try:
            data = os.read(self._fd, 1)
        except OSError:
            return
        if data != b"p":
            return
        if self.running.is_set():
            self.running.clear()
            self._paused_at = time.monotonic()
            self._console.print("\n[yellow bold]paused[/] — press p to resume")
        else:
            self.running.set()
            self._close_interval()
            self._console.print("[green bold]resuming[/]")

    def _close_interval(self) -> None:
        if self._paused_at is not None:
            self._paused_total += time.monotonic() - self._paused_at
            self._paused_at = None

    def __enter__(self) -> "PauseController":
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        asyncio.get_event_loop().add_reader(self._fd, self._on_key)
        self._console.print("[dim]press p to pause/resume[/]")
        return self

    def __exit__(self, *exc: object) -> None:
        # Before the early return: a run aborted while paused still has an open
        # interval, and the total is read after this block exits.
        self._close_interval()
        if self._old_settings is None:
            return
        asyncio.get_event_loop().remove_reader(self._fd)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


async def run_bench(settings: BenchSettings, profile: str = "full",
                    save_responses: bool = False, verbose: bool = False,
                    task_filter: str | None = None,
                    modules: list[str] | None = None,
                    trials: int | None = None,
                    only_new: bool = False,
                    ignore_task_hash: bool = False,
                    ignore_world_hash: bool = False,
                    add_trials: int | None = None,
                    max_tokens_explicit: bool = False,
                    previous: BenchResult | None = None,
                    tasks_dir: Path | None = None,
                    allow_undersized_context: bool = False,
                    checkpoint_path: Path | None = None,
                    resume: bool = True) -> BenchResult:
    """Run all modules against the configured endpoint and score results.

    Each task runs ``trials`` times (for pass^k reliability). Temperature and
    thinking are only sent to the model when explicitly set via settings/CLI;
    otherwise the model's own defaults apply (e.g. when MLX/Ollama already
    control these server-side).

    When ``only_new`` and ``previous`` are given, already-recorded trials for a
    task are reused (and not re-executed) when the task's content is unchanged
    and ``previous.meta``'s model/endpoint/temperature/max_tokens/thinking
    match this run's config. Infra-errored trials are never counted as done.

    ``ignore_task_hash`` skips that content-hash requirement, trusting
    ``(module, task_id)`` identity alone — useful both for retrying only
    truncated/missing trials after a module-wide ``max_tokens`` bump (which
    changes every task's content hash in that module, so the strict check
    would otherwise force the whole module to re-run) and for combining with
    ``add_trials`` when the same bump would otherwise make it think a task's
    existing good trials no longer exist.

    ``add_trials`` raises the per-task target to ``previous.meta.trials +
    add_trials`` (for a statistically firmer read on a model that looks off),
    rather than topping up to an absolute ``trials`` target. Every task gets
    the same k: an uneven bank is not a cheaper bank, it is a smaller one,
    because ``pass_hat_k`` drops any task holding fewer than k trials. It
    requires ``previous`` to exist and its recorded config to match this run's
    — unlike ``only_new``/``ignore_task_hash``, a mismatch is a hard error here
    rather than a silent fallback to a fresh run, since that fallback would
    quietly replace the existing trials with just the ``add_trials`` new ones
    instead of adding to them.

    ``modules`` selects exact module names — see ``_select_tasks``.
    """
    trials = max(1, trials if trials is not None else settings.trials)
    temperature = settings.temperature
    # -1 disables seeding entirely, for a server that errors on the field.
    base_seed = None if settings.seed < 0 else settings.seed
    console = Console()

    if add_trials is not None:
        if previous is None:
            raise ValueError("--add-trials needs an existing results file to "
                             "add to; run without --add-trials first")
        if not _config_matches(previous.meta, settings, max_tokens_explicit):
            raise ValueError(
                "--add-trials: previous results were run with a different "
                "(or untracked) model/endpoint/temperature/max_tokens/"
                "thinking config — refusing to blend trials from mismatched "
                "configs. Match the original run's config (--reuse-params "
                "can help) or start a fresh file instead.")
        reusable = True
        # One uniform target for every task. Whatever `--trials` said is
        # irrelevant here; the run is defined relative to the stored file.
        trials = _add_trials_target(previous, add_trials)
    else:
        reusable = (only_new or ignore_task_hash) and previous is not None
        if reusable and not _config_matches(previous.meta, settings,
                                            max_tokens_explicit):
            console.print("[yellow]--only-new/--ignore-task-hash: previous "
                          "results were run with a different (or untracked) "
                          "model/endpoint/temperature/max_tokens/thinking "
                          "config — running fresh.[/]")
            reusable = False

    selected = _select_tasks(profile, task_filter, modules, tasks_dir)

    if max_tokens_explicit:
        # The per-module caps normally win (see ChatClient.chat). An explicit
        # --max-tokens overrides them, and saying so matters in both
        # directions: raising a cap makes the run non-comparable with default
        # runs, and lowering one manufactures truncation.
        caps = {t.module: t.max_tokens for _, t in selected if t.max_tokens}
        raised = {m: c for m, c in caps.items() if c < settings.max_tokens}
        lowered = {m: c for m, c in caps.items() if c > settings.max_tokens}
        if raised:
            console.print(
                f"[yellow]--max-tokens {settings.max_tokens} raises the "
                f"per-module cap for:[/] "
                + ", ".join(f"{m} ({c})" for m, c in sorted(raised.items()))
                + "\n[dim]those caps are calibrated from measured p95/p99 "
                  "usage; raising them increases worst-case latency and makes "
                  "this run non-comparable with default-cap runs.[/]")
        if lowered:
            console.print(
                f"[red]--max-tokens {settings.max_tokens} is BELOW the "
                f"calibrated cap for:[/] "
                + ", ".join(f"{m} ({c})" for m, c in sorted(lowered.items()))
                + "\n[dim]this will manufacture truncation in those modules.[/]")

    client = ChatClient(
        endpoint=settings.endpoint, model=settings.model,
        timeout=settings.timeout, api_key=settings.api_key,
        save_responses=save_responses, max_tokens=settings.max_tokens,
        temperature=temperature, thinking=settings.thinking,
        max_attempts=settings.max_attempts,
        retry_backoff=settings.retry_backoff,
        max_tokens_override=max_tokens_explicit,
        connect_timeout=settings.connect_timeout,
        min_generation_tok_s=settings.min_generation_tok_s,
        prefill_allowance=settings.prefill_allowance,
    )

    fingerprint = _checkpoint_fingerprint(settings, profile, temperature,
                                          base_seed, max_tokens_explicit)
    checkpoint = (TrialCheckpoint(checkpoint_path, fingerprint)
                  if checkpoint_path is not None else None)
    if checkpoint is not None and resume:
        recovered = load_checkpoint(checkpoint_path, fingerprint)
        if recovered:
            console.print(
                f"[green]resuming: recovered {len(recovered)} trial(s) from an "
                f"interrupted run[/] [dim]({checkpoint_path.name})[/]")
            # Recovered trials join as prior results, which is exactly what
            # they are. The fingerprint has already established the config
            # match that `_config_matches` guards for --only-new, so reuse is
            # safe here even on a run that did not ask for it.
            base = list(previous.results) if previous is not None else []
            previous = BenchResult(
                meta=(previous.meta if previous is not None
                      else _resume_meta(settings, profile, temperature,
                                        base_seed, max_tokens_explicit,
                                        trials)),
                results=base + recovered)
            reusable = True

    # getattr, not attribute access: a stubbed ChatClient (tests, and any
    # future non-httpx backend) has no transport to introspect, and an
    # undiscoverable window is the documented None case rather than an error.
    _http = getattr(client, "_client", None)
    served_context = (
        await discover_served_context(_http, settings.endpoint, settings.model)
        if _http is not None else None)
    if served_context:
        console.print(f"[dim]served context: {served_context} tokens "
                      f"per request[/]")
        oversized = oversized_tasks(selected, served_context)
        if oversized and not allow_undersized_context:
            await client.close()
            raise UndersizedContextError(
                f"the endpoint serves {served_context} tokens per request, "
                f"which cannot hold {len(oversized)} task(s) in this run:\n"
                + "\n".join(f"  {tid}: needs at least {n}"
                             for tid, n in oversized[:8])
                + (f"\n  … and {len(oversized) - 8} more"
                   if len(oversized) > 8 else "")
                + "\n\nThe server rejects these prompts outright, and a "
                  "rejected prompt scores 0 — a capability result for a task "
                  "the model never saw.\n"
                  "llama.cpp divides --ctx-size across --parallel slots, so "
                  "`-c 65536 --parallel 4` serves 16384 per request. This "
                  "bench runs one request at a time; --parallel 1 gives the "
                  "whole window.\n"
                  "Pass --allow-undersized-context to run anyway and score "
                  "those tasks 0.")
        if oversized:
            console.print(
                f"[red]--allow-undersized-context: {len(oversized)} task(s) "
                f"exceed the {served_context}-token window and will score 0:[/] "
                + ", ".join(tid for tid, _ in oversized[:8]))

    kept, work, passthrough = _plan_work(selected, previous, trials, reusable,
                                         skip_hash_check=ignore_task_hash,
                                         add_trials=add_trials,
                                         skip_world_check=ignore_world_hash)
    if only_new or ignore_task_hash or add_trials is not None:
        label = ("--add-trials" if add_trials is not None else
                 "--ignore-task-hash" if ignore_task_hash else "--only-new")
        console.print(f"[dim]{label}: reused {len(kept)} trial(s), "
                      f"running {len(work)} new trial(s).[/]")
        stale_world = _stale_world_tasks(selected, previous, ignore_world_hash)
        if stale_world:
            # Named, never silent: this is the failure mode the hash exists for.
            console.print(
                f"[yellow]the world underneath {len(stale_world)} task(s) "
                f"changed since they last ran — re-running them:[/] "
                + ", ".join(sorted(stale_world)[:8])
                + (" …" if len(stale_world) > 8 else "")
                + "\n[dim]pass --ignore-world-hash to reuse them anyway, if you "
                  "know the registry change cannot affect them.[/]")

    semaphore = asyncio.Semaphore(settings.concurrency)
    sandbox = {
        "timeout": settings.code_timeout,
        "memory_mb": settings.sandbox_memory_mb,
        "backend": settings.sandbox_backend,
        "allow_unsandboxed": settings.allow_unsandboxed,
    }
    _log_sandbox_backend(settings)
    started = time.monotonic()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(),
        TextColumn("[green]P:{task.fields[passed]}[/]/"
                   "[yellow]Pa:{task.fields[partial]}[/]/"
                   "[red]F:{task.fields[failed]}[/]/"
                   "[magenta]T:{task.fields[timed_out]}[/] of {task.total:.0f}"),
    )
    tally = {"passed": 0, "partial": 0, "failed": 0, "timed_out": 0}
    # Sanity gate: if the first N trials to finish ALL came back empty, the
    # endpoint is misconfigured and the remaining ~30 minutes measure nothing.
    # Gated on empty output rather than on score, so a genuinely bad model is
    # still allowed to fail every task. 0 disables.
    sanity_after = max(0, settings.sanity_check_after)
    checked = 0
    dead: list[TaskResult] = []
    aborted: list[DeadEndpointError] = []
    if checkpoint is not None:
        checkpoint.open()
    with progress, PauseController(console) as pause:
        bar = progress.add_task("running", total=len(work), **tally)

        def _sanity_check(module: BaseModule, result: TaskResult) -> None:
            """Count this result toward the opening sanity window; raise if dead."""
            nonlocal checked
            if not sanity_after or checked >= sanity_after:
                return
            checked += 1
            if _is_dead_trial(result):
                dead.append(result)
            if checked < sanity_after or len(dead) < sanity_after:
                return
            detail = dead[0].error or "empty completion (no text, no tool calls)"
            aborted.append(DeadEndpointError(
                f"the first {sanity_after} trials all returned nothing — "
                f"aborting before the rest of the run.\n"
                f"first failure: {module.name}/{dead[0].task_id}: {detail}\n"
                f"endpoint={settings.endpoint} model={settings.model}\n"
                "check the model id, that the server is up, and that it "
                "supports the tool/chat API this bench needs. Set "
                "BENCH_SANITY_CHECK_AFTER=0 to disable this guard."))
            raise aborted[0]

        # Each (module, task) appears in `work` once per trial still owed, so
        # enumerating per task gives every trial of a task a distinct seed
        # while keeping the assignment stable across runs: trial 2 of tst_63
        # gets the same seed today and next month. Without the offset the k
        # trials of a task would be k identical samples on any server that
        # honours the field, and pass^k would measure nothing.
        seq: dict[tuple[str, str], int] = {}
        seeds: list[int | None] = []
        for _m, _t in work:
            index = seq[(_m.name, _t.id)] = seq.get((_m.name, _t.id), -1) + 1
            seeds.append(None if base_seed is None else base_seed + index)

        async def run_one(module: BaseModule, task: Task,
                          seed: int | None = None) -> TaskResult:
            await pause.running.wait()
            async with semaphore:
                # Checked here, not only via gather(): cancellation is
                # scheduler-dependent, but nothing acquires the semaphore after
                # the verdict lands, so no further trial can start.
                if aborted:
                    raise aborted[0]
                await pause.running.wait()
                progress.update(bar, description=f"{module.name}/{task.id}")
                result = await _execute_task(module, client, task, sandbox,
                                             seed=seed)
                if checkpoint is not None:
                    checkpoint.append(result)
                tally[_outcome(result)] += 1
                progress.update(bar, advance=1, **tally)
                if verbose:
                    _print_verbose(progress, module, result)
                _sanity_check(module, result)
                return result

        pending = [asyncio.create_task(run_one(m, t, s))
                   for (m, t), s in zip(work, seeds)]
        try:
            results = await asyncio.gather(*pending)
        except DeadEndpointError:
            # gather() leaves siblings running; stop them before the client goes.
            for job in pending:
                job.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await client.close()
            if checkpoint is not None:
                checkpoint.close()
            raise
    await client.close()
    if checkpoint is not None:
        checkpoint.close()

    all_results = passthrough + kept + list(results)
    # `meta.trials` is the k every later pass^k read is computed at, so it has
    # to be a k the file can actually honour. Recording the requested target
    # when some task fell short of it (a trial that truncated, an infra error
    # on the retry) silently deleted those tasks from the headline. The
    # observed per-task minimum is the largest k for which no task is dropped.
    recorded_trials = _observed_trials(all_results, trials)
    meta = BenchMeta(
        model=settings.model, endpoint=settings.endpoint,
        timestamp=datetime.now(timezone.utc).isoformat(),
        # Wall clock MINUS time the user held the run: a pause is the
        # operator's, not the model's. max(0.0, ...) because the two clocks are
        # read at different instants and a pause closed microseconds before
        # this line could otherwise round to a negative duration.
        duration_seconds=round(
            max(0.0, time.monotonic() - started - pause.paused_seconds), 2),
        bench_version=__version__, fast=(profile == "fast"), profile=profile,
        trials=recorded_trials,
        thinking=settings.thinking, temperature=temperature,
        seed=base_seed, sampling=_sampling_description(temperature, base_seed),
        sandbox_backend=(sandbox or {}).get("backend", ""),
        python_version=platform.python_version(),
        max_tokens=settings.max_tokens,
        max_tokens_override=max_tokens_explicit,
        concurrency=settings.concurrency,
        served_context=served_context or 0,
        task_set_hash=results_task_set_hash(all_results),
        task_count=len({(r.module, r.task_id) for r in all_results}),
    )
    return BenchResult(meta=meta, results=all_results)


_OUTCOME_STYLES = {"passed": ("green", "PASS"), "partial": ("yellow", "PART"),
                   "failed": ("red", "FAIL"), "timed_out": ("magenta", "TIME")}


def fmt_det(score: float, success: bool) -> str:
    """Format a trial's det score without ever rounding a failure up to 1.00.

    Byte-exact state tasks fail on a near-miss: one wrong line in a 90-line file
    scores ~0.996, which two decimals render as `1.00` next to a FAIL. The
    2.6B did exactly that on tst_35b. Failing near-misses get three decimals and
    a `~` so the number cannot be read as a pass.
    """
    if not success and score >= 0.995:
        return f"~{score:.3f}"
    return f"{score:.2f}"


def _print_verbose(progress: Progress, module: BaseModule,
                   result: TaskResult) -> None:
    """Print a per-task debug line above the progress bar."""
    color, label = _OUTCOME_STYLES[_outcome(result)]
    head = (f"[{color}]{label}[/] {module.name}/{result.task_id} "
            f"det={fmt_det(result.det_score, result.success)}")
    progress.console.print(head)
    if result.error:
        progress.console.print(f"  [dim]error:[/] {result.error}")
        return
    if result.det_score < 1.0:
        progress.console.print(f"  [dim]expected:[/] {result.expected}")
        progress.console.print(f"  [dim]breakdown:[/] {result.det_breakdown}")
        snippet = result.response_raw.replace("\n", " ")[:200]
        progress.console.print(f"  [dim]response:[/] {snippet}")


def _outcome(result: TaskResult) -> str:
    """Classify a task result for the live progress tally."""
    if result.error:
        return "timed_out" if result.timed_out else "failed"
    if result.det_score == 1.0:
        return "passed"
    return "partial" if result.det_score >= 0.7 else "failed"


def _log_sandbox_backend(settings: BenchSettings) -> None:
    """Resolve and announce the code-execution sandbox backend once per run."""
    from rich.console import Console

    from .sandbox import detect_backend, is_real_sandbox

    backend = detect_backend(settings.sandbox_backend)
    console = Console()
    if is_real_sandbox(backend):
        console.print(f"[dim]sandbox backend: {backend}[/]")
    elif backend == "rlimit" and settings.allow_unsandboxed:
        console.print("[yellow]sandbox: rlimit only (no fs/net isolation) "
                      "— running unsandboxed by request[/]")
    else:
        console.print("[yellow]sandbox: none available — code tasks will be "
                      "SKIPPED (use --allow-unsandboxed to override)[/]")


async def _execute_task(module: BaseModule, client: ChatClient,
                        task: Task, sandbox: dict | None = None,
                        seed: int | None = None) -> TaskResult:
    """Run and score one task, converting failures into error results."""
    _SEED.set(seed)
    task_started = time.monotonic()
    task_hash = task_content_hash(task)
    world_hash = task_world_hash(task)
    timings = TimingAccumulator()
    _TIMINGS.set(timings)
    try:
        result = await module.run_task(client, task, sandbox=sandbox)
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        # A KeyError or IndexError here is a TASK DEFINITION bug — a haystack
        # spec with no `needle`, a conversation entry with no `prompt` — not
        # something the model did. It used to be recorded with
        # infra_error=False, so `counted()` kept it and a malformed task read
        # as every model failing it. Excluded and named instead; the coverage
        # report surfaces it, and the run still finishes.
        authoring_bug = isinstance(exc, (KeyError, IndexError))
        failed = TaskResult(
            task_id=task.id, module=module.name, prompt=task.prompt,
            system_prompt=task.system_prompt,
            difficulty=task.difficulty, tier=task.tier, band=task.band, expected=task.expected,
            duration_seconds=round(time.monotonic() - task_started, 3),
            error=(f"task definition error: {_error_text(exc)}"
                   if authoring_bug else _error_text(exc)),
            infra_error=(_is_transient(exc) or authoring_bug
                         or _is_config_error(exc)),
            context_overflow=_is_context_overflow(exc),
            malformed_tool_call=_is_malformed_tool_call(exc),
            task_hash=task_hash,
            world_hash=world_hash,
        )
        # Calls that landed before the failure still measured something.
        timings.apply(failed)
        return failed
    result.difficulty = task.difficulty
    result.tier = task.tier
    result.band = task.band
    result.system_prompt = task.system_prompt
    result.task_hash = task_hash
    result.world_hash = world_hash
    result.duration_seconds = round(time.monotonic() - task_started, 3)
    timings.apply(result)
    # Scoring is guarded separately from generation. It used to sit outside
    # any try, and asyncio.gather re-raises, so a single malformed check in a
    # task YAML (a `section_contains` with no value, say) destroyed a
    # 45-minute sweep and wrote no file at all — the run's whole output lost
    # to a one-line authoring mistake. A scorer crash is now one unscorable
    # trial: it is marked infra_error so `counted()` drops it rather than
    # reading as a model failure the model never had a chance to cause.
    try:
        scored = score_task(task, result, sandbox)
    except Exception as exc:                        # noqa: BLE001 - see above
        result.error = f"scoring failed: {type(exc).__name__}: {exc}"
        result.infra_error = True
        result.det_score = 0.0
        result.success = False
        result.det_success = False
        result.truncation_class = truncation_class(result)
        return result
    result.det_score = scored.score
    result.success = scored.success
    result.det_success = scored.success
    result.det_breakdown = scored.breakdown
    # A scorer that could not run the candidate (no sandbox, no image) reports
    # it here; never cleared, since generation may already have failed too.
    result.infra_error = result.infra_error or scored.infra_error
    result.truncation_class = truncation_class(result)
    return result
