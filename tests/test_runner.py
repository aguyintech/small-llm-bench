"""Tests for the async runner: retry on timeout, error handling, parsing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from small_llm_bench.config import BenchSettings
from small_llm_bench.models import BenchMeta, BenchResult, Task, TaskResult, task_content_hash
from small_llm_bench.modules.base import parse_tool_calls
from small_llm_bench.modules.knowledge import KnowledgeModule
from small_llm_bench.reporter import aggregate_by_task
from small_llm_bench.runner import (ChatClient, _config_matches, _execute_task,
                                    _outcome, _plan_work, _select_tasks, run_bench)
from small_llm_bench.scorer import aggregate_module_scores

ENDPOINT = "http://testserver/v1"


def _client(handler, **kwargs) -> ChatClient:
    """Build a ChatClient backed by an httpx MockTransport."""
    return ChatClient(endpoint=ENDPOINT, model="test-model",
                      transport=httpx.MockTransport(handler), **kwargs)


@pytest.fixture
def knowledge_task() -> Task:
    return Task(id="kn_test", module="knowledge", prompt="What is 2+2?",
                answer_type="numeric", expected={"answer": 4})


async def test_chat_returns_payload(chat_completion):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=chat_completion(content="4"))

    client = _client(handler)
    response = await client.chat([{"role": "user", "content": "2+2?"}])
    assert response["choices"][0]["message"]["content"] == "4"
    await client.close()


async def test_chat_thinking_injects_template_kwargs(chat_completion):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_completion(content="ok"))

    on = _client(handler, thinking=True)
    await on.chat([{"role": "user", "content": "hi"}])
    await on.close()
    assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": True}

    off = _client(handler)
    await off.chat([{"role": "user", "content": "hi"}])
    await off.close()
    assert "chat_template_kwargs" not in bodies[-1]


async def test_chat_does_not_retry_a_read_timeout():
    """A read timeout means the generation outran the wall clock.

    Generation length is a property of the model and the prompt, not of the
    network, so a second identical request runs the same work and blows the
    same budget. Measured 2026-09-08: two probe trials cost exactly 546s each
    (3 x 180s timeout + 2s + 4s backoff) to reach a failure the first attempt
    had established in 180.
    """
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler, max_attempts=3, retry_backoff=0)
    with pytest.raises(httpx.ReadTimeout):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1
    await client.close()


async def test_chat_still_retries_a_connect_timeout(chat_completion):
    """Connecting is a network act and a second try can genuinely succeed."""
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("no route", request=request)
        return httpx.Response(200, json=chat_completion(content="ok"))

    client = _client(handler, retry_backoff=0)
    response = await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 2
    assert response["choices"][0]["message"]["content"] == "ok"
    await client.close()


async def test_chat_raises_after_exhausting_attempts():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectTimeout("no route", request=request)

    client = _client(handler, max_attempts=3, retry_backoff=0)
    with pytest.raises(httpx.TimeoutException):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 3
    await client.close()


async def test_a_read_timeout_is_still_excluded_from_scoring():
    """Not retried, but still not the model's fault."""
    from small_llm_bench.runner import _is_retryable, _is_transient

    exc = httpx.ReadTimeout("slow", request=httpx.Request("POST", "http://x"))
    assert _is_retryable(exc) is False
    assert _is_transient(exc) is True


def test_read_timeout_scales_with_the_token_cap():
    """One global value cannot serve a 16k cap and a 21-264 tok/s fleet.

    Measured 2026-09-08: the long_context 12288 cap needs 182s on
    gemma-4-12b, and BENCH_TIMEOUT was 180 — the two were set independently
    and landed two seconds apart, killing roughly one trial in five.
    """
    client = ChatClient("http://x", "m", timeout=180.0,
                        min_generation_tok_s=20.0, prefill_allowance=120.0)
    # 12288 / 20 + 120 = 734.4
    assert client.read_timeout_for(12288) == pytest.approx(734.4)
    # the configured value is a FLOOR, so a small cap does not shrink below it
    assert client.read_timeout_for(512) == 180.0
    # connect stays short regardless, so a dead endpoint fails fast
    assert client._timeout_for(12288).connect == 10.0


async def test_chat_retries_500_then_succeeds(chat_completion):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=chat_completion(content="ok"))

    client = _client(handler, retry_backoff=0)
    response = await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 2
    assert response["choices"][0]["message"]["content"] == "ok"
    await client.close()


async def test_chat_retries_connect_error_then_succeeds(chat_completion):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection lost", request=request)
        return httpx.Response(200, json=chat_completion(content="ok"))

    client = _client(handler, retry_backoff=0)
    response = await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 2
    await client.close()


async def test_chat_raises_after_persistent_500():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="boom")

    client = _client(handler, max_attempts=3, retry_backoff=0)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 3
    await client.close()


async def test_chat_does_not_retry_400():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, text="bad request")

    client = _client(handler, max_attempts=3, retry_backoff=0)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1
    await client.close()


async def test_execute_task_persistent_500_marks_infra_error(knowledge_task):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client(handler, max_attempts=2, retry_backoff=0)
    result = await _execute_task(KnowledgeModule(), client, knowledge_task)
    assert result.error is not None
    assert result.infra_error is True
    assert result.det_score == 0.0
    await client.close()


def test_infra_errored_trials_excluded_from_scoring():
    base = dict(task_id="t", module="knowledge", prompt="p", tier="baseline")
    good = TaskResult(**base, det_score=1.0, success=True)
    infra = TaskResult(**base, det_score=0.0, error="500", infra_error=True)
    mod = aggregate_module_scores([good, infra])
    # only the real trial counts: 1.0, not (1.0 + 0.0) / 2
    assert mod["knowledge"]["det_score"] == 1.0
    assert mod["knowledge"]["count"] == 1
    by_task = aggregate_by_task([good, infra])
    assert by_task["t"]["n"] == 1


async def test_execute_task_records_timeout_as_failed(knowledge_task):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler)
    result = await _execute_task(KnowledgeModule(), client, knowledge_task)
    assert result.error is not None
    assert "Timeout" in result.error
    assert result.det_score == 0.0
    await client.close()


async def test_execute_task_scores_success(knowledge_task, chat_completion):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_completion(content="The answer is 4"))

    client = _client(handler)
    result = await _execute_task(KnowledgeModule(), client, knowledge_task)
    assert result.error is None
    assert result.det_score == 1.0
    await client.close()


async def test_execute_task_uses_per_task_max_tokens_override(chat_completion):
    task = Task(id="kn_capped", module="knowledge", prompt="What is 2+2?",
               answer_type="numeric", expected={"answer": 4}, max_tokens=111)
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_completion(content="4"))

    client = _client(handler, max_tokens=4096)
    await _execute_task(KnowledgeModule(), client, task)
    await client.close()
    assert bodies[-1]["max_tokens"] == 111


async def test_execute_task_falls_back_to_client_default_max_tokens(knowledge_task,
                                                                     chat_completion):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_completion(content="4"))

    client = _client(handler, max_tokens=4096)
    await _execute_task(KnowledgeModule(), client, knowledge_task)
    await client.close()
    assert bodies[-1]["max_tokens"] == 4096


def test_parse_tool_calls_handles_malformed_arguments():
    message = {"tool_calls": [
        {"function": {"name": "get_weather", "arguments": "not json"}},
        {"function": {"name": "calculate",
                      "arguments": json.dumps({"expression": "1+1"})}},
    ]}
    calls = parse_tool_calls(message)
    assert calls[0].arguments == {}
    assert calls[1].arguments == {"expression": "1+1"}


def test_task_result_timed_out_classification():
    base = dict(task_id="t", module="m", prompt="p")
    assert TaskResult(**base, error="ReadTimeout: timed out").timed_out
    assert TaskResult(**base, error="TimeoutException: x").timed_out
    assert not TaskResult(**base, error="KeyError: 'choices'").timed_out
    assert not TaskResult(**base, error="ValueError: Timeout mentioned late").timed_out
    assert not TaskResult(**base).timed_out


def test_outcome_classification():
    base = dict(task_id="t", module="m", prompt="p")
    assert _outcome(TaskResult(**base, error="KeyError: x")) == "failed"
    assert _outcome(TaskResult(**base, error="ReadTimeout: x")) == "timed_out"
    assert _outcome(TaskResult(**base, det_score=1.0)) == "passed"
    assert _outcome(TaskResult(**base, det_score=0.8)) == "partial"
    assert _outcome(TaskResult(**base, det_score=0.7)) == "partial"
    assert _outcome(TaskResult(**base, det_score=0.5)) == "failed"


# --- --only-new: task content hashing, reuse planning, config matching ---

def test_task_content_hash_changes_on_edit(knowledge_task):
    original = task_content_hash(knowledge_task)
    edited = knowledge_task.model_copy(update={"prompt": "What is 3+3?"})
    assert task_content_hash(edited) != original


def test_task_content_hash_ignores_fast_flag(knowledge_task):
    fast_variant = knowledge_task.model_copy(update={"fast": True})
    assert task_content_hash(knowledge_task) == task_content_hash(fast_variant)


def _module(name: str):
    return SimpleNamespace(name=name)


def test_plan_work_reuses_unchanged_task_up_to_trials(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h) for _ in range(3)],
    )
    kept, work, passthrough = _plan_work([(module, knowledge_task)], previous,
                                         trials=3, reusable=True)
    assert len(kept) == 3
    assert work == []
    assert passthrough == []


def test_plan_work_tops_up_missing_trials(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h)],
    )
    kept, work, passthrough = _plan_work([(module, knowledge_task)], previous,
                                         trials=3, reusable=True)
    assert len(kept) == 1
    assert len(work) == 2
    assert work == [(module, knowledge_task)] * 2


def test_plan_work_reruns_when_task_content_changed(knowledge_task):
    module = _module("knowledge")
    stale_hash = task_content_hash(knowledge_task) + "stale"
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=stale_hash)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True)
    assert kept == []
    assert work == [(module, knowledge_task)]


def test_plan_work_skip_hash_check_reuses_despite_content_change(knowledge_task):
    module = _module("knowledge")
    stale_hash = task_content_hash(knowledge_task) + "stale"
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=stale_hash)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True, skip_hash_check=True)
    assert len(kept) == 1
    assert work == []


def test_plan_work_skip_hash_check_still_excludes_truncated(knowledge_task):
    module = _module("knowledge")
    stale_hash = task_content_hash(knowledge_task) + "stale"
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=stale_hash, truncated=True)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True, skip_hash_check=True)
    assert kept == []
    assert work == [(module, knowledge_task)]


def test_plan_work_excludes_infra_errored_trials_from_reuse(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h, error="500", infra_error=True)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True)
    assert kept == []
    assert work == [(module, knowledge_task)]


def test_plan_work_excludes_truncated_trials_from_reuse(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h, truncated=True)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True)
    assert kept == []
    assert work == [(module, knowledge_task)]


def test_plan_work_add_trials_is_additive_on_top_of_existing(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096, trials=3),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h) for _ in range(3)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True, add_trials=2)
    assert len(kept) == 3
    assert work == [(module, knowledge_task)] * 2


def test_plan_work_add_trials_tops_a_short_task_up_to_the_same_k(knowledge_task):
    """A task that lost a trial is topped up to the SAME k as everyone else.

    This used to be `len(matches) + add_trials`, i.e. 2 + 2 = 4 while the run
    recorded meta.trials = 5. `pass_hat_k` drops any task with n < k, so the
    "additive per task" reading deleted the task from the headline outright.
    """
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096, trials=3),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h),
                 TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                           task_hash=h, truncated=True),
                 TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                           task_hash=h)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=True, add_trials=2)
    # 2 reusable (1 was truncated), target 3 + 2 = 5, so 3 new — the truncated
    # trial gets re-run rather than leaving the task one short forever.
    assert len(kept) == 2
    assert work == [(module, knowledge_task)] * 3


def test_observed_trials_never_overstates_what_a_task_holds():
    """meta.trials must be a k every task can honour, or tasks vanish."""
    from small_llm_bench.runner import _observed_trials

    def task(tid, n, **kw):
        return [TaskResult(task_id=tid, module="knowledge", prompt="p", **kw)
                for _ in range(n)]

    assert _observed_trials(task("a", 3), requested=3) == 3
    # Never above what was asked for, even if some task somehow holds more.
    assert _observed_trials(task("a", 3), requested=2) == 2

    # The modal attempt count, not the minimum. One short task among many is
    # dropped by pass_hat_k and reported as a coverage gap; it must not pull
    # every other task down to a weaker exponent.
    many = task("a", 3) + task("b", 3) + task("c", 3) + task("d", 2)
    assert _observed_trials(many, requested=3) == 3

    # An infra-errored trial is an ATTEMPT. Skipping it was how one dead
    # request relabelled a whole 39-task file from k=3 to k=2 and moved
    # qwen3.6-27b from 4th to 2nd on pass^2 > pass^3 alone.
    with_infra = (task("a", 3) + task("b", 3)
                  + task("c", 2) + task("c", 1, infra_error=True))
    assert _observed_trials(with_infra, requested=3) == 3

    # A genuinely short file still records what it holds, or pass^3 would drop
    # every task and score it 0.0.
    all_short = task("a", 2) + task("b", 2) + task("c", 2)
    assert _observed_trials(all_short, requested=3) == 2
    assert _observed_trials([], requested=3) == 3


async def test_run_bench_add_trials_requires_previous():
    settings = BenchSettings(model="m", endpoint="e")
    with pytest.raises(ValueError, match="add-trials"):
        await run_bench(settings, add_trials=2, previous=None)


async def test_run_bench_add_trials_errors_on_config_mismatch():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=4096)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                      bench_version="0", max_tokens=8192, trials=3),
        results=[],
    )
    with pytest.raises(ValueError, match="mismatched"):
        await run_bench(settings, add_trials=2, previous=previous)


def test_plan_work_not_reusable_runs_everything_fresh(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[TaskResult(task_id="kn_test", module="knowledge", prompt="p",
                            task_hash=h)],
    )
    kept, work, _ = _plan_work([(module, knowledge_task)], previous,
                              trials=1, reusable=False)
    assert kept == []
    assert work == [(module, knowledge_task)]


def test_plan_work_passes_through_unselected_tasks(knowledge_task):
    module = _module("knowledge")
    h = task_content_hash(knowledge_task)
    other = TaskResult(task_id="other_task", module="knowledge", prompt="p", task_hash=h)
    previous = BenchResult(
        meta=BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                       bench_version="0", temperature=0.0, max_tokens=4096),
        results=[other],
    )
    kept, work, passthrough = _plan_work([(module, knowledge_task)], previous,
                                         trials=1, reusable=True)
    assert kept == []
    assert work == [(module, knowledge_task)]
    assert passthrough == [other]


def test_select_tasks_modules_restricts_to_exact_names():
    selected = _select_tasks(profile="fast", task_filter=None,
                             modules=["tools", "code"])
    names = {module.name for module, _ in selected}
    assert names == {"tools", "code"}


def test_select_tasks_modules_does_not_prefix_match():
    selected = _select_tasks(profile="fast", task_filter=None, modules=["tools"])
    names = {module.name for module, _ in selected}
    assert names == {"tools"}
    assert "code" not in names and "knowledge" not in names


def test_select_tasks_modules_and_filter_combine():
    all_state = _select_tasks(profile="fast", task_filter=None, modules=["tools"])
    narrowed = _select_tasks(profile="fast", task_filter="tst_", modules=["tools"])
    assert len(narrowed) <= len(all_state)
    assert narrowed  # state-axis task ids are prefixed tst_


def test_select_tasks_rejects_unknown_module():
    with pytest.raises(ValueError, match="unknown module"):
        _select_tasks(profile="fast", task_filter=None, modules=["not_a_module"])


def test_config_matches_identical_settings():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=4096,
                             thinking=False, temperature=0.5)
    meta = BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                     bench_version="0", temperature=0.5, max_tokens=4096, thinking=False)
    assert _config_matches(meta, settings)


def test_config_matches_false_on_temperature_change():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=4096, temperature=0.9)
    meta = BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                     bench_version="0", temperature=0.5, max_tokens=4096)
    assert not _config_matches(meta, settings)


def test_config_matches_true_when_neither_sets_temperature():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=4096)
    meta = BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                     bench_version="0", max_tokens=4096)  # temperature/thinking at None defaults
    assert _config_matches(meta, settings)


def test_config_matches_true_when_max_tokens_raised():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=8192)
    meta = BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                     bench_version="0", max_tokens=4096)
    assert _config_matches(meta, settings)


def test_config_matches_false_when_max_tokens_lowered():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=4096)
    meta = BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                     bench_version="0", max_tokens=8192)
    assert not _config_matches(meta, settings)


def test_config_matches_false_when_meta_predates_tracking():
    settings = BenchSettings(model="m", endpoint="e", max_tokens=4096)
    meta = BenchMeta(model="m", endpoint="e", timestamp="", duration_seconds=0,
                     bench_version="0", max_tokens=None)
    assert not _config_matches(meta, settings)


def test_results_task_set_hash_is_order_insensitive_and_content_sensitive():
    """The hash identifies the task bank a run covered, so two runs over the
    same bank must match regardless of result order, and a single changed task
    must break the match (regression: qwen3.8-27b silently ran a different
    revision of ds_08 than the six runs it was ranked against)."""
    from small_llm_bench.models import TaskResult, results_task_set_hash

    def _r(task_id, task_hash):
        return TaskResult(task_id=task_id, module="code", prompt="p",
                          task_hash=task_hash)

    a = [_r("cd_01", "h1"), _r("cd_02", "h2")]
    assert results_task_set_hash(a) == results_task_set_hash(list(reversed(a)))
    # extra trials of the same tasks don't change the bank identity
    assert results_task_set_hash(a + a) == results_task_set_hash(a)
    changed = [_r("cd_01", "h1"), _r("cd_02", "CHANGED")]
    assert results_task_set_hash(changed) != results_task_set_hash(a)


# --- explicit --max-tokens overrides the per-module caps ---------------------
#
# Before v0.12 a per-task cap always won, so `--max-tokens 16384` was silently
# ignored by the capped modules. Observed on a floor-model run: knowledge still
# truncated at exactly 4096 on 21/21 trials and format at 4096. Only the
# modules missing from the cap table took the flag — and in v0.13 there are
# none of those left, so without this the flag would do nothing at all.

async def test_explicit_max_tokens_raises_a_per_task_cap(chat_completion):
    task = Task(id="kn_capped", module="knowledge", prompt="What is 2+2?",
                answer_type="numeric", expected={"answer": 4}, max_tokens=111)
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_completion(content="4"))

    client = _client(handler, max_tokens=16384, max_tokens_override=True)
    await _execute_task(KnowledgeModule(), client, task)
    await client.close()
    assert bodies[-1]["max_tokens"] == 16384


async def test_explicit_max_tokens_also_lowers_a_per_task_cap(chat_completion):
    """The override wins in both directions — lowering it is how a cheap probe
    run bounds its cost, and it must not be silently ignored either."""
    task = Task(id="kn_capped", module="knowledge", prompt="What is 2+2?",
                answer_type="numeric", expected={"answer": 4}, max_tokens=4096)
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_completion(content="4"))

    client = _client(handler, max_tokens=100, max_tokens_override=True)
    await _execute_task(KnowledgeModule(), client, task)
    await client.close()
    assert bodies[-1]["max_tokens"] == 100


async def test_without_the_flag_the_task_cap_still_wins(chat_completion):
    """Only an explicit flag changes precedence; the calibrated per-module caps
    remain the default, which is what keeps ordinary runs comparable."""
    task = Task(id="kn_capped", module="knowledge", prompt="What is 2+2?",
                answer_type="numeric", expected={"answer": 4}, max_tokens=111)
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_completion(content="4"))

    client = _client(handler, max_tokens=16384)
    await _execute_task(KnowledgeModule(), client, task)
    await client.close()
    assert bodies[-1]["max_tokens"] == 111


def test_overriding_runs_trials_are_not_reused_by_a_default_run():
    """An overriding run's trials ran at the global cap in every module. Pooling
    them into a default run reuses trials with a more generous budget than
    anything that run will get — the direction the max_tokens rule forbids."""
    settings = BenchSettings(model="m", endpoint=ENDPOINT, max_tokens=8192)
    meta = BenchMeta(model="m", endpoint=ENDPOINT, timestamp="t",
                     duration_seconds=0.0, bench_version="0.11.0",
                     max_tokens=8192, max_tokens_override=True)
    assert _config_matches(meta, settings, max_tokens_explicit=False) is False
    assert _config_matches(meta, settings, max_tokens_explicit=True) is True


def test_a_default_runs_trials_are_still_reusable_by_an_overriding_run():
    settings = BenchSettings(model="m", endpoint=ENDPOINT, max_tokens=8192)
    meta = BenchMeta(model="m", endpoint=ENDPOINT, timestamp="t",
                     duration_seconds=0.0, bench_version="0.11.0",
                     max_tokens=8192, max_tokens_override=False)
    assert _config_matches(meta, settings, max_tokens_explicit=True) is True
