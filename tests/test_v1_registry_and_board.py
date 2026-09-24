"""v1.0: the board says how big each model is, and what it was scored on.

Nothing in a result file recorded model size, so the leaderboard ranked a 12B
above a 35B with no way to notice, and every size claim in the CHANGELOG was
inferred from the model's own name string. Nothing recorded how many tasks a
row was actually scored on either, so two models silently scored on 33 and 38
tasks sat in one ranked table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from small_llm_bench.leaderboard import (_size_cell, build_leaderboard,
                                         load_model_registry,
                                         render_leaderboard_html)
from small_llm_bench.models import BenchMeta, BenchResult, TaskResult
from small_llm_bench.reporter import save_results

_REPO_ROOT = Path(__file__).parent.parent


class TestRegistryFile:
    def test_the_shipped_registry_parses(self):
        entries = load_model_registry(_REPO_ROOT)
        assert entries, "models.yaml should ship with the fleet in it"
        for name, entry in entries.items():
            assert entry["name"] == name
            assert entry.get("params_b") is not None, f"{name} has no size"

    def test_every_moe_declares_active_params_below_total(self):
        for name, entry in load_model_registry(_REPO_ROOT).items():
            active = entry.get("active_b")
            if active is not None:
                assert active < entry["params_b"], name

    def test_variants_point_at_a_model_that_exists(self):
        entries = load_model_registry(_REPO_ROOT)
        for name, entry in entries.items():
            parent = entry.get("variant_of")
            if parent:
                assert parent in entries, f"{name} -> unknown {parent}"
                assert parent != name

    def test_a_missing_registry_is_not_an_error(self, tmp_path):
        # A user's checkout may not have one, and a model absent from it still
        # runs and still scores.
        assert isinstance(load_model_registry(tmp_path), dict)


class TestSizeCell:
    def test_dense_model_shows_total_and_sorts_by_it(self):
        cell = _size_cell({"params_b": 12, "family": "gemma"})
        assert cell["label"] == "12B" and cell["sort_b"] == 12

    def test_moe_sorts_by_active_params(self):
        # The bank has always ranked MoEs this way: gemma-4-26b-a4b landed
        # below gemma-4-12b on its 4B active. Printing 26B beside that row
        # would read as an inversion that is not one.
        cell = _size_cell({"params_b": 8, "active_b": 1})
        assert cell["label"] == "8B · A1B" and cell["sort_b"] == 1
        assert cell["is_sparse"] is True

    def test_unknown_model_is_a_question_mark_not_a_crash(self):
        cell = _size_cell(None)
        assert cell["label"] == "?" and cell["sort_b"] is None
        assert cell["is_sparse"] is False and cell["bucket"] is None

    def test_sparsity_is_the_grouping_and_arch_is_only_the_label(self):
        """`arch` defaults to MoE because every sparse model here routes
        experts except one. gemma-4-e2b-it reaches its active count through
        per-layer embeddings, and a badge reading MoE there would be false."""
        assert _size_cell({"params_b": 12})["arch"] is None
        assert _size_cell({"params_b": 8, "active_b": 1})["arch"] == "moe"
        ple = _size_cell({"params_b": 5.1, "active_b": 2.3, "arch": "ple"})
        assert ple["arch"] == "ple" and ple["is_sparse"] is True

    def test_buckets_cut_on_total_params_not_active(self):
        """A 35B MoE has to be loaded whole, so it is a 24B+ footprint even
        though it ranks by its 3B active. The two answer different questions
        and the board shows both."""
        moe = _size_cell({"params_b": 35, "active_b": 3})
        assert moe["bucket"] == "xl" and moe["sort_b"] == 3


def _write(tmp_path: Path, model: str, results: list[TaskResult],
           trials: int = 3) -> Path:
    meta = BenchMeta(model=model, endpoint="http://x", timestamp="now",
                     duration_seconds=1.0, bench_version="1.0.0",
                     trials=trials)
    path = tmp_path / f"{model}_raw_results.json"
    save_results(BenchResult(meta=meta, results=results), path)
    return path


def _trials(task_id: str, n: int, ok: bool = True, seconds: float = 60.0):
    return [TaskResult(task_id=task_id, module="knowledge", prompt="p",
                       success=ok, det_success=ok, det_score=1.0 if ok else 0.0,
                       duration_seconds=seconds) for _ in range(n)]


class TestBoardReportsWhatItScored:
    def test_a_short_task_is_counted_as_excluded_not_absorbed(self, tmp_path):
        # Two of three trials on `b` were unusable, so pass^3 drops the task
        # entirely and the row is scored on one task, not two.
        _write(tmp_path, "m", _trials("a", 3) + _trials("b", 1))
        row = build_leaderboard(tmp_path)["rows"][0]
        assert row["n_tasks"] == 1
        assert row["tasks_excluded"] == 1

    def test_a_complete_row_excludes_nothing(self, tmp_path):
        _write(tmp_path, "m", _trials("a", 3) + _trials("b", 3))
        row = build_leaderboard(tmp_path)["rows"][0]
        assert row["n_tasks"] == 2 and row["tasks_excluded"] == 0

    def test_run_time_is_summed_trial_time(self, tmp_path):
        # meta.duration_seconds reports only the last session's wall clock, so
        # a file assembled with --only-new understated its own cost 3-30x.
        _write(tmp_path, "m", _trials("a", 3, seconds=120.0))
        assert build_leaderboard(tmp_path)["rows"][0]["trial_minutes"] == 6.0

    def test_sampling_and_sandbox_are_comparability_fields(self, tmp_path):
        _write(tmp_path, "m", _trials("a", 3))
        settings = build_leaderboard(tmp_path)["rows"][0]["run_settings"]
        assert "sampling" in settings and "sandbox_backend" in settings


class TestTemplateColumnsLineUp:
    """A header with no cell shifts every column to its right."""

    def test_header_count_matches_emitted_cell_count(self, tmp_path):
        # `first_try_rate` had a header and no <td> until v1.0, so the speed
        # columns all rendered one place left of their own headings.
        _write(tmp_path, "m", _trials("a", 3))
        html = render_leaderboard_html(build_leaderboard(tmp_path))
        columns = re.search(r"const columns = (.*?);\n\n", html, re.S).group(1)
        body = re.search(r"let html = `<td>.*?tr\.innerHTML", html, re.S).group(0)
        assert columns.count("{ key:") == body.count("<td")
