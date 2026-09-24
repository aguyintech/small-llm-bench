"""Tests for the LLM judge: prompt rendering, batch parsing, result merging."""

from __future__ import annotations

import json

import pytest

import small_llm_bench.judge as judge_module
from small_llm_bench.config import JudgeSettings
from small_llm_bench.judge import (default_judge_output, judge_results,
                                   parse_judge_batch, render_judge_prompt)
from small_llm_bench.models import BenchMeta, BenchResult, TaskResult


@pytest.fixture
def bench() -> BenchResult:
    meta = BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                     duration_seconds=1.0, bench_version="0.1.0")
    results = [
        TaskResult(task_id="ts_01", module="tools", prompt="p1",
                   det_score=0.5),
        TaskResult(task_id="ts_02", module="tools", prompt="p2",
                   det_score=0.7),
        TaskResult(task_id="kn_01", module="tools", prompt="p3",
                   det_score=1.0),
    ]
    return BenchResult(meta=meta, results=results)


class FakeClient:
    """Stand-in ChatClient returning a canned batched judge reply."""

    # uids are per-trial (task_id#index of that trial within its own task):
    # ts_01 and ts_02 are distinct tasks, each on their first (0th) trial.
    reply = json.dumps([
        {"task_id": "ts_01#0", "score": 0.8, "reasoning": "Solid."},
        {"task_id": "ts_02#0", "score": 0.6, "reasoning": "Okay."},
        {"task_id": "kn_01#0", "score": 1.0, "reasoning": "Correct."},
    ])

    def __init__(self, **kwargs) -> None:
        pass

    async def chat(self, messages, tools=None):
        return {"choices": [{"message": {"content": self.reply}}]}

    async def close(self) -> None:
        pass


def test_parse_valid_judge_batch():
    scores = parse_judge_batch(
        '[{"task_id": "a", "score": 0.7, "reasoning": "ok"}]'
    )
    assert scores == {"a": (0.7, "ok")}


def test_parse_batch_with_surrounding_text():
    scores = parse_judge_batch(
        'Here: [{"task_id": "a", "score": 0.9, "reasoning": "x"}] done.'
    )
    assert scores["a"][0] == 0.9


def test_parse_malformed_judge_batch():
    assert parse_judge_batch("not json at all") == {}
    assert parse_judge_batch("[not valid json]") == {}
    assert parse_judge_batch('{"task_id": "a", "score":}') == {}


def test_parse_salvages_entries_when_array_unparseable():
    # A truncated/broken array (unterminated final reasoning string) must not
    # discard the well-formed verdicts before it — salvage them per-object.
    text = ('[{"task_id": "a", "score": 0.5, "reasoning": "ok"}, '
            '{"task_id": "b", "score": 0.9, "reasoning": "good"}, '
            '{"task_id": "c", "score": 0.2, "reasoning": "trunca')
    scores = parse_judge_batch(text)
    assert scores["a"] == (0.5, "ok")
    assert scores["b"] == (0.9, "good")
    assert "c" not in scores  # the truncated entry is dropped, not fatal

    # A single bare valid object (no array) is recovered too.
    assert parse_judge_batch('{"task_id": "z", "score": 1, "reasoning": "y"}') \
        == {"z": (1.0, "y")}


def test_parse_skips_malformed_entries():
    scores = parse_judge_batch(json.dumps([
        {"task_id": "good", "score": 0.5, "reasoning": "ok"},
        {"task_id": "bad", "score": "high"},
        {"score": 0.4},
        "not a dict",
    ]))
    assert set(scores) == {"good"}


def test_parse_clamps_out_of_range_score():
    scores = parse_judge_batch(
        '[{"task_id": "a", "score": 1.7, "reasoning": "x"}]'
    )
    assert scores["a"][0] == 1.0


def test_render_judge_prompt_lists_all_tasks(bench):
    results = [r for r in bench.results if r.module == "tools"]
    prompt = render_judge_prompt("tools", results)
    assert "tools" in prompt
    assert "ts_01" in prompt and "ts_02" in prompt
    assert '"task_id"' in prompt


def test_default_judge_output_never_equals_input(tmp_path):
    source = tmp_path / "raw_results.json"
    output = default_judge_output(source)
    assert output != source
    assert output.name == "raw_results_judged.json"


async def test_judge_results_adds_scores(bench, monkeypatch):
    monkeypatch.setattr(judge_module, "ChatClient", FakeClient)
    judged = await judge_results(bench, JudgeSettings())
    by_id = {r.task_id: r for r in judged.results}
    assert by_id["ts_01"].llm_score == 0.8
    assert by_id["ts_02"].llm_reasoning == "Okay."
    assert by_id["kn_01"].llm_score == 1.0


async def test_judge_scores_trials_of_same_task_independently(monkeypatch):
    """Two trials sharing a task_id must each get their own verdict, not one
    score smeared across both (regression: judge keyed only by task_id)."""
    meta = BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                     duration_seconds=1.0, bench_version="0.1.0")
    bench = BenchResult(meta=meta, results=[
        TaskResult(task_id="ts_01", module="tools", prompt="p", det_score=1.0),
        TaskResult(task_id="ts_01", module="tools", prompt="p", det_score=0.4),
    ])

    class TrialClient(FakeClient):
        reply = json.dumps([
            {"task_id": "ts_01#0", "score": 1.0, "reasoning": "perfect trial"},
            {"task_id": "ts_01#1", "score": 0.4, "reasoning": "buggy trial"},
        ])

    monkeypatch.setattr(judge_module, "ChatClient", TrialClient)
    judged = await judge_results(bench, JudgeSettings())
    assert [r.llm_score for r in judged.results] == [1.0, 0.4]
    assert judged.results[0].success is True
    assert judged.results[1].success is False


def test_judge_cannot_rescue_a_recall_module_failure():
    """long_context/data_extract are graded by string and field matching, and
    the judge proved measurably worse at it than the code — it called a reply
    ending `**7284**.` "a complete sentence" and a fenced JSON object invalid.
    Its verdict is display-only there; a tool module stays overridable."""
    lc = TaskResult(task_id="lc_08", module="long_context", prompt="p",
                    det_score=0.85, success=False)
    lc.llm_score = 1.0
    judge_module.apply_judge_verdict(lc, "long_context")
    assert lc.success is False

    tl = TaskResult(task_id="tl_01", module="tools", prompt="p",
                    det_score=0.5, success=False)
    tl.llm_score = 1.0
    judge_module.apply_judge_verdict(tl, "tools")
    assert tl.llm_score == 1.0 and tl.success is True


async def test_judge_cannot_rescue_by_echoing_det_score(monkeypatch):
    """A judge reply that keeps the deterministic score is agreement, not a
    verdict, so it must not flip a det failure to a pass.

    Regression: JUDGE_PASS_THRESHOLD (0.85) sat below the deterministic success
    bar (~0.999), so every det near-miss in between was rescued by an echo — up
    to +11 points of headline on models whose scorers land just under the bar.
    """
    meta = BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                     duration_seconds=1.0, bench_version="0.1.0")
    bench = BenchResult(meta=meta, results=[
        TaskResult(task_id="tl_15", module="tools", prompt="p",
                   det_score=0.9, success=False),      # exact echo
        TaskResult(task_id="tst_11", module="tools", prompt="p",
                   det_score=0.88, success=False),     # nudged, still < 0.95
        TaskResult(task_id="tl_23", module="tools", prompt="p",
                   det_score=0.9, success=False),      # genuine raise
    ])

    class EchoClient(FakeClient):
        reply = json.dumps([
            {"task_id": "tl_15#0", "score": 0.9,
             "reasoning": "Keeping default deterministic score."},
            {"task_id": "tst_11#0", "score": 0.9, "reasoning": "close enough"},
            {"task_id": "tl_23#0", "score": 1.0, "reasoning": "harmless variation"},
        ])

    monkeypatch.setattr(judge_module, "ChatClient", EchoClient)
    judged = await judge_results(bench, JudgeSettings())
    echo, nudged, raised = judged.results
    # echo: no disagreement at all -> stays failed. The score is left exactly
    # as the judge emitted it; clamping it to det (v1.0 removed that) made the
    # stored score no longer the judge's answer, so `rescore` re-deriving the
    # verdict from it read a number the judge never gave.
    assert echo.success is False and echo.llm_score == 0.9
    # raised, but not to near-correct -> still not a rescue
    assert nudged.success is False and nudged.llm_score == 0.9
    # genuine raise to fully-correct -> rescued
    assert raised.success is True and raised.llm_score == 1.0
    # det_success stays frozen in every case
    assert [r.det_success for r in judged.results] == [False, False, False]


async def test_judge_demotion_unaffected_by_rescue_gate(monkeypatch):
    """The rescue gate must not touch the demote path: a det pass scored below
    JUDGE_PASS_THRESHOLD still fails, even though the judge lowered the score
    (i.e. the score is not above det)."""
    meta = BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                     duration_seconds=1.0, bench_version="0.1.0")
    bench = BenchResult(meta=meta, results=[
        TaskResult(task_id="tl_01", module="tools", prompt="p",
                   det_score=1.0, success=True),
    ])

    class DemoteClient(FakeClient):
        reply = json.dumps(
            [{"task_id": "tl_01#0", "score": 0.4, "reasoning": "would fail live"}])

    monkeypatch.setattr(judge_module, "ChatClient", DemoteClient)
    judged = await judge_results(bench, JudgeSettings())
    assert judged.results[0].success is False
    assert judged.results[0].det_success is True


def test_judge_can_still_demote_an_agentic_module():
    """Where the judge is still consulted, a det pass it scores clearly lower
    means the deterministic check was too lenient — worth surfacing."""
    tst = TaskResult(task_id="tst_01", module="tools", prompt="p",
                     det_score=1.0, success=True)
    tst.llm_score = 0.2
    judge_module.apply_judge_verdict(tst, "tools")
    assert tst.success is False


async def test_judge_cannot_override_adversarial_success(monkeypatch):
    """Adversarial (prompt-injection) tasks are deterministic-authoritative: the
    judge is fed the injection and misreads it as the user's goal, so it must
    never flip success — neither demote a correct refusal nor rescue a failure
    (regression: judge scored a correct French-only refusal 0.0)."""
    meta = BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                     duration_seconds=1.0, bench_version="0.1.0")
    bench = BenchResult(meta=meta, results=[
        TaskResult(task_id="adv_01", module="adversarial", prompt="p",
                   det_score=1.0, success=True),
        TaskResult(task_id="adv_02", module="adversarial", prompt="p",
                   det_score=0.0, success=False),
    ])

    class FlipClient(FakeClient):
        reply = json.dumps([
            {"task_id": "adv_01#0", "score": 0.0, "reasoning": "ignored user"},
            {"task_id": "adv_02#0", "score": 1.0, "reasoning": "sounds fine"},
        ])

    monkeypatch.setattr(judge_module, "ChatClient", FlipClient)
    judged = await judge_results(bench, JudgeSettings())
    a1, a2 = judged.results
    # llm scores recorded for display, but success stays deterministic
    assert a1.llm_score == 0.0 and a1.success is True
    assert a2.llm_score == 1.0 and a2.success is False


async def test_judge_results_does_not_mutate_original(bench, monkeypatch):
    monkeypatch.setattr(judge_module, "ChatClient", FakeClient)
    await judge_results(bench, JudgeSettings())
    assert all(r.llm_score is None for r in bench.results)


async def test_judge_handles_malformed_json(bench, monkeypatch):
    class BadClient(FakeClient):
        reply = "I think these all deserve good scores!"

    monkeypatch.setattr(judge_module, "ChatClient", BadClient)
    judged = await judge_results(bench, JudgeSettings())
    assert all(r.llm_score is None for r in judged.results)
    assert all(r.llm_reasoning is None for r in judged.results)


async def test_judge_handles_omitted_task(bench, monkeypatch):
    class PartialClient(FakeClient):
        reply = json.dumps([
            {"task_id": "ts_01#0", "score": 0.8, "reasoning": "Solid."},
            {"task_id": "kn_01#0", "score": 1.0, "reasoning": "Correct."},
        ])

    monkeypatch.setattr(judge_module, "ChatClient", PartialClient)
    judged = await judge_results(bench, JudgeSettings())
    by_id = {r.task_id: r for r in judged.results}
    assert by_id["ts_01"].llm_score == 0.8
    assert by_id["ts_02"].llm_score is None


async def test_judge_retries_a_fully_unjudged_module(bench, monkeypatch):
    """A module whose batched call failed transport-side gets a second pass.

    Regression: a burst of Gemini 503s left three tool modules entirely
    unjudged, which then dropped them from the judged-score denominator and
    pushed the judged score ABOVE the deterministic one.
    """
    calls: list[int] = []

    class FlakyClient(FakeClient):
        async def chat(self, messages, tools=None):
            calls.append(1)
            if len(calls) == 1:  # first module call blows up
                raise RuntimeError("503 Service Unavailable")
            return {"choices": [{"message": {"content": FakeClient.reply}}]}

    monkeypatch.setattr(judge_module, "ChatClient", FlakyClient)
    monkeypatch.setattr(judge_module.asyncio, "sleep", _noop_sleep)
    settings = JudgeSettings(concurrency=1)
    judged = await judge_results(bench, settings)
    # tool_simple failed first, then the stranded-module pass recovered it
    by_id = {r.task_id: r for r in judged.results}
    assert by_id["ts_01"].llm_score == 0.8
    assert by_id["ts_02"].llm_score == 0.6


async def _noop_sleep(_seconds):
    return None
