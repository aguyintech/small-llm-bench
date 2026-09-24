"""Tests for reporter aggregations (difficulty tiers and tok/s speed)."""

from __future__ import annotations

import json

import pytest

from small_llm_bench.models import TaskResult
from small_llm_bench.reporter import (aggregate_by_task,
                                      aggregate_difficulty_scores,
                                      aggregate_speed, headline_overall,
                                      judge_coverage, load_results,
                                      module_success)


def _r(**kw) -> TaskResult:
    base = dict(task_id="t", module="code", prompt="p")
    base.update(kw)
    return TaskResult(**base)


def test_speed_per_module_and_global():
    results = [
        _r(module="code", completion_tokens=300, duration_seconds=4.0),
        _r(module="code", completion_tokens=100, duration_seconds=1.0),
        _r(module="knowledge", completion_tokens=200, duration_seconds=2.0),
    ]
    speeds = aggregate_speed(results)
    assert speeds["code"] == (300 + 100) / (4.0 + 1.0)
    assert speeds["knowledge"] == 100.0
    assert speeds["__global__"] == (300 + 100 + 200) / (4.0 + 1.0 + 2.0)


def test_speed_ignores_errors_and_zero_usage():
    results = [
        _r(completion_tokens=0, duration_seconds=0.0, error="Timeout: x"),
        _r(completion_tokens=0, duration_seconds=3.0),
        _r(completion_tokens=120, duration_seconds=0.0),
    ]
    speeds = aggregate_speed(results)
    assert speeds.get("code") is None
    assert speeds.get("__global__") is None


def test_difficulty_aggregation():
    results = [
        _r(task_id="e1", difficulty="easy", det_score=1.0),
        _r(task_id="h1", difficulty="hard", det_score=0.5),
        _r(task_id="h2", difficulty="hard", det_score=1.0),
    ]
    tiers = aggregate_difficulty_scores(results)
    assert tiers["easy"]["count"] == 1
    assert tiers["hard"]["count"] == 2
    assert tiers["hard"]["det_score"] == 0.75


def test_difficulty_aggregation_counts_tasks_not_trials():
    """A task run with multiple trials counts once in the difficulty table,
    matching how the module/overall tables count tasks (not per-trial rows)."""
    results = [
        _r(task_id="h1", difficulty="hard", det_score=1.0),
        _r(task_id="h1", difficulty="hard", det_score=0.0),
        _r(task_id="h1", difficulty="hard", det_score=1.0),
    ]
    tiers = aggregate_difficulty_scores(results)
    assert tiers["hard"]["count"] == 1
    assert tiers["hard"]["det_score"] == pytest.approx(2 / 3)


def test_pass_breakdown_det_vs_judge():
    """aggregate_by_task tracks det pass and judge-adjusted pass separately,
    and module_success exposes both via use_det."""
    results = [
        # judge rescued a det failure (success True, det_success False)
        _r(task_id="a", module="tools", det_success=False, success=True),
        # judge demoted a det pass (success False, det_success True)
        _r(task_id="b", module="tools", det_success=True, success=False),
    ]
    by_task = aggregate_by_task(results)
    assert by_task["a"]["det_passes"] == 0 and by_task["a"]["passes"] == 1
    assert by_task["b"]["det_passes"] == 1 and by_task["b"]["passes"] == 0
    # one det pass and one judge pass across the two tasks → 0.5 each
    assert module_success(results, use_det=True)["tools"] == 0.5
    assert module_success(results)["tools"] == 0.5


def test_headline_band_scheme_weights_by_band():
    results = [
        _r(task_id="a1", band="anchor", success=True, det_success=True),
        _r(task_id="m1", band="mid", success=True, det_success=True),
        _r(task_id="h1", band="hard", success=False, det_success=False),
        _r(task_id="f1", band="frontier", success=False, det_success=False),
    ]
    band_score = headline_overall(results, scheme="band")
    legacy_score = headline_overall(results, scheme="legacy")
    # every band/tier has exactly one task at pass/fail, so the two schemes
    # both reduce to a weighted average, but with different weights —
    # they should generally disagree given the different weight tables.
    assert 0.0 < band_score < 1.0
    assert 0.0 < legacy_score < 1.0
    assert band_score != legacy_score


def test_v03_result_file_without_band_field_still_loads(tmp_path):
    """A result file saved before the `band` field existed (v0.3) must still
    load and score under the legacy scheme; band-less tasks default to "mid"
    and fall out of the band scheme's frontier/anchor/hard buckets."""
    payload = {
        "meta": {"model": "old-model", "endpoint": "http://x", "timestamp": "t",
                 "duration_seconds": 1.0, "bench_version": "0.3.0", "trials": 1},
        "results": [
            {"task_id": "b1", "module": "knowledge", "prompt": "p",
             "tier": "baseline", "success": True, "det_success": True,
             "det_score": 1.0},
        ],
    }
    path = tmp_path / "old_raw_results.json"
    path.write_text(json.dumps(payload))
    bench = load_results(path)
    assert bench.results[0].band == "mid"
    assert headline_overall(bench.results, scheme="legacy") == 1.0


def test_judge_coverage_reports_unjudged_and_partial_modules():
    results = [
        # fully judged
        _r(task_id="cd_01", module="code", llm_score=1.0),
        # partly judged
        _r(task_id="tst_01", module="tools", llm_score=0.9),
        _r(task_id="tst_02", module="tools"),
        # not judged at all (the Gemini-503 case)
        _r(task_id="lc_01", module="long_context"),
        _r(task_id="lc_02", module="long_context"),
        # deliberately never judged — must not read as a coverage gap
        _r(task_id="fm_01", module="format"),
        _r(task_id="kn_01", module="knowledge"),
    ]
    cov = judge_coverage(results)
    assert cov["judged"] == 2 and cov["total"] == 5
    assert cov["unjudged_modules"] == ["long_context"]
    assert cov["partial_modules"] == ["tools"]
    assert cov["per_module"]["tools"]["coverage"] == 0.5
    assert cov["per_module"]["code"]["coverage"] == 1.0
    assert "format" not in cov["per_module"]
    assert "knowledge" not in cov["per_module"]


def test_judge_coverage_excludes_infra_errors():
    """Infra-errored trials are out of every other aggregation (scorable), so
    counting them here would report coverage gaps that don't exist."""
    results = [
        _r(task_id="cd_01", module="code", llm_score=1.0),
        _r(task_id="cd_02", module="code", error="connection reset",
           infra_error=True),
    ]
    cov = judge_coverage(results)
    assert cov["total"] == 1 and cov["coverage"] == 1.0
    assert cov["unjudged_modules"] == [] and cov["partial_modules"] == []
