"""Tests for server-reported prefill/generation timing capture.

Covers the three response shapes the parser accepts (llama.cpp's `timings`
block, oMLX's `usage` extension, plain `usage`), the per-task accumulator, and
the token-weighted report aggregation.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from small_llm_bench.models import BenchResult, TaskResult
from small_llm_bench.modules.base import call_timings
from small_llm_bench.reporter import aggregate_timings
from small_llm_bench.runner import ChatClient, TimingAccumulator, _TIMINGS

# A llama.cpp /v1/chat/completions response: 236 of 1236 prompt tokens came
# from the prefix cache, so only prompt_n was actually evaluated.
LLAMACPP = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1236, "completion_tokens": 35},
    "timings": {
        "cache_n": 236, "prompt_n": 1000, "prompt_ms": 500.0,
        "prompt_per_second": 2000.0, "predicted_n": 35,
        "predicted_ms": 700.0, "predicted_per_second": 50.0,
    },
}

# oMLX only fills these on the streaming usage chunk; cached tokens are counted
# inside prompt_tokens there, unlike llama.cpp.
OMLX_STREAM = {
    "usage": {
        "prompt_tokens": 1200, "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 200},
        "time_to_first_token": 0.5, "total_time": 1.3,
        "prompt_eval_duration": 0.5, "generation_duration": 0.8,
        "prompt_tokens_per_second": 2400.0,
        "generation_tokens_per_second": 50.0,
    },
}

# What oMLX actually returns non-streaming: tokens and a total, no split.
OMLX_NON_STREAM = {
    "usage": {
        "prompt_tokens": 1200, "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 200},
        "total_time": 1.3,
    },
}


def test_llamacpp_timings_block():
    t = call_timings(LLAMACPP)
    assert t == {"prompt_tokens": 1000, "cached_prompt_tokens": 236,
                 "prefill_seconds": 0.5, "generation_seconds": 0.7,
                 "source": "llamacpp"}


def test_omlx_streaming_usage_extension():
    t = call_timings(OMLX_STREAM)
    # Cached tokens are excluded so the prefill rate reflects real work.
    assert t["prompt_tokens"] == 1000
    assert t["cached_prompt_tokens"] == 200
    assert t["prefill_seconds"] == 0.5
    assert t["generation_seconds"] == 0.8
    assert t["source"] == "omlx"


def test_omlx_non_streaming_gives_tokens_but_no_split():
    t = call_timings(OMLX_NON_STREAM)
    assert t["prompt_tokens"] == 1000
    assert t["cached_prompt_tokens"] == 200
    assert t["prefill_seconds"] == 0.0
    assert t["generation_seconds"] == 0.0
    assert t["source"] == "usage"


def test_no_usage_at_all():
    assert call_timings({"choices": []}) is None
    assert call_timings({"usage": {"completion_tokens": 5}}) is None


def test_garbage_values_do_not_raise():
    t = call_timings({"timings": {"prompt_n": "x", "prompt_ms": None,
                                  "cache_n": 4, "predicted_ms": "nope"}})
    assert t == {"prompt_tokens": 0, "cached_prompt_tokens": 4,
                 "prefill_seconds": 0.0, "generation_seconds": 0.0,
                 "source": "llamacpp"}


def test_accumulator_sums_calls_with_growing_prefixes():
    acc = TimingAccumulator()
    for prompt_n, prompt_ms in ((100, 50.0), (600, 300.0), (1400, 700.0)):
        acc.add({"timings": {"prompt_n": prompt_n, "cache_n": 0,
                             "prompt_ms": prompt_ms, "predicted_ms": 200.0}})
    result = TaskResult(task_id="t", module="tools", prompt="p")
    acc.apply(result)
    assert result.prompt_tokens == 2100
    assert result.prefill_seconds == pytest.approx(1.05)
    assert result.generation_seconds == pytest.approx(0.6)
    assert result.timing_source == "llamacpp"


def test_accumulator_ignores_timingless_responses():
    acc = TimingAccumulator()
    acc.add(LLAMACPP)
    acc.add({"choices": []})
    result = TaskResult(task_id="t", module="code", prompt="p")
    acc.apply(result)
    assert result.prompt_tokens == 1000
    assert result.timing_source == "llamacpp"


def test_accumulator_flags_mixed_sources():
    acc = TimingAccumulator()
    acc.add(LLAMACPP)
    acc.add(OMLX_STREAM)
    result = TaskResult(task_id="t", module="code", prompt="p")
    acc.apply(result)
    assert result.timing_source == "mixed"


def test_client_records_into_active_accumulator():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=LLAMACPP)

    client = ChatClient(endpoint="http://x", model="m",
                        transport=httpx.MockTransport(handler))
    acc = TimingAccumulator()
    token = _TIMINGS.set(acc)
    try:
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    finally:
        _TIMINGS.reset(token)
    assert acc.prompt_tokens == 1000
    assert acc.prefill_seconds == pytest.approx(0.5)


def _r(**kw) -> TaskResult:
    base = dict(task_id="t", module="code", prompt="p")
    base.update(kw)
    return TaskResult(**base)


def test_aggregate_timings_is_token_weighted():
    # One long-context task must dominate the prefill rate rather than being
    # averaged away by many short ones.
    results = [
        _r(module="long_context", prompt_tokens=20000, prefill_seconds=10.0,
           completion_tokens=50, generation_seconds=1.0),
        _r(module="format", prompt_tokens=20, prefill_seconds=0.001,
           completion_tokens=100, generation_seconds=2.0),
    ]
    agg = aggregate_timings(results)
    assert agg["long_context"]["prefill_tok_s"] == pytest.approx(2000.0)
    assert agg["__global__"]["prefill_tok_s"] == pytest.approx(20020 / 10.001)
    assert agg["__global__"]["gen_tok_s"] == pytest.approx(150 / 3.0)


def test_aggregate_timings_none_without_a_split():
    # The oMLX non-streaming case: tokens known, durations zero.
    results = [_r(prompt_tokens=1000, completion_tokens=40)]
    agg = aggregate_timings(results)
    assert agg["code"]["prefill_tok_s"] is None
    assert agg["code"]["gen_tok_s"] is None


def test_aggregate_timings_skips_errors():
    results = [
        _r(error="Timeout: x", prompt_tokens=999, prefill_seconds=9.0),
        _r(prompt_tokens=100, prefill_seconds=0.1),
    ]
    # The errored task contributes nothing; the module entry reflects only the
    # task that completed. A module with no completed task gets no entry.
    assert aggregate_timings(results)["code"]["prefill_tok_s"] == pytest.approx(1000.0)
    assert "code" not in aggregate_timings(results[:1])


def test_pre_change_results_file_still_loads(tmp_path):
    """Results written before these fields existed must validate and read as
    "no timing data", not fail."""
    payload = {
        "meta": {"model": "m", "endpoint": "e", "timestamp": "t",
                 "duration_seconds": 1.0, "bench_version": "0.8.0"},
        "results": [{"task_id": "t", "module": "code", "prompt": "p",
                     "completion_tokens": 10, "duration_seconds": 1.0}],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload))
    bench = BenchResult.model_validate(json.loads(path.read_text()))
    assert bench.meta.concurrency == 0
    assert bench.results[0].prompt_tokens == 0
    assert bench.results[0].timing_source == ""
    assert aggregate_timings(bench.results)["code"]["prefill_tok_s"] is None
