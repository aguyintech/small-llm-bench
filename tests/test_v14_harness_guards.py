"""v0.14 harness guards: dead-endpoint abort, per-task haystack salting, and
`final_text_checks` on stateful tasks."""

from __future__ import annotations

import pytest

from small_llm_bench.config import BenchSettings
from small_llm_bench.models import Task, TaskResult, task_world_hash
from small_llm_bench.modules.base import BaseModule
from small_llm_bench.modules.long_context import (_filler_lines, _salt_offset,
                                                  build_haystack)
from small_llm_bench.runner import (DeadEndpointError, _is_dead_trial,
                                    run_bench)


# --- dead-endpoint abort ---------------------------------------------------

def _result(**kwargs) -> TaskResult:
    base = {"task_id": "t", "module": "knowledge", "prompt": "p"}
    return TaskResult(**{**base, **kwargs})


def test_is_dead_trial_flags_errors_and_empty_completions():
    assert _is_dead_trial(_result(error="TransportError: connect refused"))
    assert _is_dead_trial(_result(response_raw="   "))


def test_is_dead_trial_spares_real_signal(make_call):
    """A wrong answer, a context overflow, and a tool-only turn are all data."""
    assert not _is_dead_trial(_result(response_raw="42", det_score=0.0))
    assert not _is_dead_trial(
        _result(error="context length exceeded", context_overflow=True))
    assert not _is_dead_trial(_result(turns=[
        {"role": "assistant", "content": None,
         "tool_calls": [make_call("read_file", path="/a")]},
    ]))


@pytest.fixture
def dead_execute(monkeypatch):
    """Make every task come back with nothing, counting the attempts."""
    calls: list[str] = []

    async def fake(module: BaseModule, client, task: Task, sandbox=None,
                   seed=None):
        calls.append(task.id)
        return _result(task_id=task.id, module=module.name, prompt=task.prompt)

    monkeypatch.setattr("small_llm_bench.runner._execute_task", fake)
    return calls


async def test_run_bench_aborts_when_the_opening_trials_are_all_dead(
        dead_execute, tasks_dir):
    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=3)
    with pytest.raises(DeadEndpointError, match="first 3 trials"):
        await run_bench(settings, modules=["knowledge"], trials=1,
                        tasks_dir=tasks_dir)
    # The guard is the point: it must stop well before the module is done.
    assert len(dead_execute) < 6


async def test_run_bench_sanity_gate_can_be_disabled(dead_execute, tasks_dir):
    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=0)
    bench = await run_bench(settings, modules=["knowledge"], trials=1,
                            tasks_dir=tasks_dir)
    assert len(bench.results) == len(dead_execute) > 3


async def test_run_bench_runs_to_completion_for_a_merely_bad_model(
        monkeypatch, tasks_dir):
    """Failing every task is not the same as returning nothing."""
    async def fake(module: BaseModule, client, task: Task, sandbox=None,
                   seed=None):
        return _result(task_id=task.id, module=module.name, prompt=task.prompt,
                       response_raw="I don't know", det_score=0.0)

    monkeypatch.setattr("small_llm_bench.runner._execute_task", fake)
    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=3)
    bench = await run_bench(settings, modules=["knowledge"], trials=1,
                            tasks_dir=tasks_dir)
    assert len(bench.results) > 3


# --- haystack salting ------------------------------------------------------

SPEC = {"filler_tokens": 2000, "needle": "The code is 4242.", "position": 0.5}


def test_salted_haystacks_share_no_prefix():
    a = build_haystack(SPEC, salt="lc_08")
    b = build_haystack(SPEC, salt="lc_31")
    assert a != b
    assert a.split("\n")[0] != b.split("\n")[0]


def test_salt_is_deterministic_across_processes():
    assert build_haystack(SPEC, salt="lc_08") == build_haystack(SPEC, salt="lc_08")
    # sha256-derived, so it cannot move with PYTHONHASHSEED.
    assert _salt_offset("lc_08") == _salt_offset("lc_08") != _salt_offset("lc_31")


def test_salt_does_not_change_the_document_size():
    """Only the line numbering moves; the needle and line count must not."""
    plain = build_haystack(SPEC)
    salted = build_haystack(SPEC, salt="lc_08")
    assert len(plain.split("\n")) == len(salted.split("\n"))
    assert "The code is 4242." in salted
    assert len(_filler_lines(2000)) == len(_filler_lines(2000, salt="lc_08"))


def test_unsalted_build_is_unchanged():
    """Tests and tooling that call build_haystack(spec) keep the old document."""
    assert build_haystack(SPEC).startswith("Log entry 1:")


def test_long_context_tasks_carry_a_world_hash():
    """Otherwise a generator change leaves stale trials looking reusable."""
    task = Task(id="lc_08", module="long_context", prompt="?",
                answer_type="numeric", expected={"answer": 1}, haystack=SPEC)
    other = Task(id="lc_31", module="long_context", prompt="?",
                 answer_type="numeric", expected={"answer": 1}, haystack=SPEC)
    assert task_world_hash(task)
    assert task_world_hash(task) != task_world_hash(other)


def test_plain_tasks_still_have_no_world_hash():
    task = Task(id="kn_01", module="knowledge", prompt="?",
                answer_type="numeric", expected={"answer": 1})
    assert task_world_hash(task) == ""


# --- final_text_checks on stateful tasks -----------------------------------

STATE_EXPECT = {
    "expected_state": {"kv": {"port": "5432"}},
    "final_text_checks": [{"type": "contains", "value": "already"}],
}
GOOD_STATE = {"kv": {"port": "5432"}}


def test_state_task_honours_final_text_checks():
    """Until v0.14 this block lived only in score_tool_loop, so a task with both
    `expected_state` and `final_text_checks` had the text half silently dropped
    — tst_55 shipped that way."""
    from small_llm_bench.scorer import score_state

    said = score_state(STATE_EXPECT, GOOD_STATE, {}, [],
                       answer_text="it was already set, no change needed")
    assert said.success is True
    assert said.breakdown["final_text_score"] == 1.0

    silent = score_state(STATE_EXPECT, GOOD_STATE, {}, [], answer_text="done")
    assert silent.success is False
    assert silent.breakdown["final_text_score"] == 0.0
    # Proportional deduction, same 0.80 + 0.20*frac shape the loop scorer uses.
    assert silent.score == round(said.score * 0.80, 4)


def test_state_task_without_final_text_checks_is_untouched():
    """The common case must not gain a breakdown key or lose a point."""
    from small_llm_bench.scorer import score_state

    r = score_state({"expected_state": {"kv": {"port": "5432"}}},
                    GOOD_STATE, {}, [], answer_text="")
    assert r.success is True
    assert r.score == 1.0
    assert "final_text_score" not in r.breakdown


def test_state_final_text_checks_cannot_rescue_a_wrong_state():
    from small_llm_bench.scorer import score_state

    r = score_state(STATE_EXPECT, {"kv": {"port": "9999"}}, {}, [],
                    answer_text="it was already set, no change needed")
    assert r.success is False


def test_score_task_feeds_the_closing_turn_to_a_state_task():
    """End to end through score_task, which is where the wiring was missing."""
    from small_llm_bench.models import Task, TaskResult, TurnRecord
    from small_llm_bench.scorer import score_task

    task = Task(id="tst_x", module="tools", prompt="p",
                initial_state={"kv": {}}, expected=STATE_EXPECT)
    result = TaskResult(
        task_id="tst_x", module="tools", prompt="p", final_state=GOOD_STATE,
        turns=[TurnRecord(role="assistant",
                          content="it was already set, no change needed")])
    assert score_task(task, result).breakdown["final_text_score"] == 1.0


def test_probe_disables_the_dead_endpoint_guard(monkeypatch, tmp_path):
    """A probe is one task at three trials, so "all three empty" is the verdict
    being screened for — not a broken endpoint. Leaving the guard armed made a
    hard-failing weak model raise DeadEndpointError instead of scoring 0/3."""
    import asyncio

    from small_llm_bench.config import ProbeSettings
    from small_llm_bench.models import BenchMeta, BenchResult
    from small_llm_bench.probe import _run_one_model

    seen = {}

    async def fake_run_bench(settings, **kwargs):
        seen["sanity_check_after"] = settings.sanity_check_after
        return BenchResult(
            meta=BenchMeta(model=settings.model, endpoint="e", timestamp="",
                           duration_seconds=0, bench_version="0"),
            results=[])

    monkeypatch.setattr("small_llm_bench.probe.run_bench", fake_run_bench)
    asyncio.run(_run_one_model(
        "m", module="tools", task_id="tst_x", tasks_dir=tmp_path / "tasks",
        out_path=tmp_path / "runs" / "out.json", probe=ProbeSettings(),
        reprobe=False))
    assert seen["sanity_check_after"] == 0


# --- probe verdict reason ordering -----------------------------------------

def test_mid_dip_reports_inversion_not_saturation():
    """ds_61 probed 1.00 / 0.33 / 1.00 and printed "saturated: every model
    passes" while the 12B passed one trial in three. The saturated branch used
    to fire on the weak slot alone, before the monotonicity check."""
    from small_llm_bench.analysis import probe_verdict

    v = probe_verdict((3, 3), (1, 3), (3, 3))
    assert v.verdict == "REJECT"
    assert v.reason == "inverted"


def test_all_three_perfect_is_still_saturated():
    from small_llm_bench.analysis import probe_verdict

    v = probe_verdict((3, 3), (3, 3), (3, 3))
    assert (v.verdict, v.reason) == ("REJECT", "saturated")


def test_a_discriminating_task_still_accepts():
    from small_llm_bench.analysis import probe_verdict

    v = probe_verdict((0, 3), (2, 3), (3, 3))
    assert v.verdict == "ACCEPT"


# --- near-miss score display ------------------------------------------------

def test_a_failing_near_miss_never_renders_as_1_00():
    """One wrong line in a 90-line byte-exact file scores ~0.996. At two
    decimals that prints `1.00` beside a FAIL, which reads as a pass — the
    2.6B did exactly this on tst_35b."""
    from small_llm_bench.runner import fmt_det

    assert fmt_det(0.9963, False) == "~0.996"
    assert fmt_det(0.9999, False) == "~1.000"


def test_ordinary_scores_keep_two_decimals():
    from small_llm_bench.runner import fmt_det

    assert fmt_det(1.0, True) == "1.00"
    assert fmt_det(0.87, False) == "0.87"
    assert fmt_det(0.0, False) == "0.00"


# --- structural file checks -------------------------------------------------

FILE = """---
updated: 2026-08-14
---

## Threads

- Gateway migration
  - waiting on the fix

## Meetings

| Meeting | Summary | Date | Notes |
| --- | --- | --- | --- |
| Vendor review | Renewed. | 2026-08-07 | [[2026-08-07]] |
| Q2 retro | Latency. | 2026-06-19 | [[2026-06-19]] |
"""

CHECKS = {"f.md": [
    {"type": "frontmatter", "key": "updated", "value": "2026-08-24"},
    {"type": "table_wellformed", "section": "## Meetings"},
    {"type": "table_rows_preserved", "section": "## Meetings", "key_column": 0},
    {"type": "table_row_position", "section": "## Meetings",
     "contains": "Payments review", "index": 0},
    {"type": "bullets_wellformed", "section": "## Threads"},
]}
NEW_ROW = "| Payments review | Delayed. | 2026-08-21 | [[2026-08-21]] |"


def _state(text):
    return {"files": {"f.md": text}}


def _ideal():
    return (FILE.replace("updated: 2026-08-14", "updated: 2026-08-24")
            .replace("| --- | --- | --- | --- |",
                     "| --- | --- | --- | --- |\n" + NEW_ROW))


def test_file_checks_pass_on_a_correct_edit():
    from small_llm_bench.scorer import score_file_checks

    score, detail = score_file_checks(CHECKS, _state(_ideal()), _state(FILE))
    assert score == 1.0
    assert all(detail.values())


def test_file_checks_tolerate_rewording_and_extra_content():
    """The whole point: a broad instruction has many correct renderings. A model
    that also reworks a summary or adds a bullet is doing the job, and byte-exact
    grading failed all three models on pf_01 for exactly that."""
    from small_llm_bench.scorer import score_file_checks

    looser = (_ideal().replace("Renewed.", "Renewed for twelve months.")
              .replace("  - waiting on the fix",
                       "  - waiting on the fix\n  - delayed until it lands"))
    assert score_file_checks(CHECKS, _state(looser), _state(FILE))[0] == 1.0


def test_file_checks_catch_each_real_failure():
    from small_llm_bench.scorer import score_file_checks

    def fails(text):
        _, detail = score_file_checks(CHECKS, _state(text), _state(FILE))
        return {k.split(":")[1] for k, ok in detail.items() if not ok}

    bottom = FILE.replace("updated: 2026-08-14", "updated: 2026-08-24").replace(
        "| Q2 retro | Latency. | 2026-06-19 | [[2026-06-19]] |",
        "| Q2 retro | Latency. | 2026-06-19 | [[2026-06-19]] |\n" + NEW_ROW)
    assert fails(bottom) == {"table_row_position"}
    assert fails(_ideal().replace("updated: 2026-08-24",
                                  "updated: 2026-08-21")) == {"frontmatter"}
    assert fails(_ideal().replace(
        "| Q2 retro | Latency. | 2026-06-19 | [[2026-06-19]] |\n", "")) == {
        "table_rows_preserved"}
    assert fails(_ideal().replace(NEW_ROW, "| Payments review | 2026-08-21 |")) == {
        "table_wellformed"}
    assert fails(_ideal().replace("  - waiting on the fix",
                                  "     - waiting on the fix")) == {
        "bullets_wellformed"}


def test_repeated_check_types_do_not_collide():
    """Two bullets_wellformed checks on different sections must be reported
    separately; keying the detail map on path+type alone let the second silently
    overwrite the first."""
    from small_llm_bench.scorer import score_file_checks

    spec = {"f.md": [
        {"type": "bullets_wellformed", "section": "## Threads"},
        {"type": "bullets_wellformed", "section": "## Meetings"},
    ]}
    _, detail = score_file_checks(spec, _state(FILE), _state(FILE))
    assert len(detail) == 2


def test_unknown_check_type_fails_closed():
    from small_llm_bench.scorer import score_file_checks

    spec = {"f.md": [{"type": "not_a_real_check"}]}
    assert score_file_checks(spec, _state(FILE), _state(FILE))[0] == 0.0
