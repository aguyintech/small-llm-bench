"""Tests for the candidate-task probe loop (v0.14).

The loop exists because task authoring was open-loop: v0.13 shipped 16 new
tasks of which 6 separated nobody and none separated the 4B from the 12B. The
screen is only worth having if its verdict is trustworthy, so the truth table
below is the load-bearing test in this file.
"""

from __future__ import annotations

import json

import pytest
import typer
import yaml

from small_llm_bench.analysis import (DISCRIMINATION_FLOOR, collect_item_stats,
                                      probe_verdict)
from small_llm_bench.config import BenchSettings, ProbeSettings
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.probe import (JUDGE_DECIDES, assert_not_results_dir,
                                   candidate_fingerprint, cycles_for,
                                   probe_paths, read_expectation)
from small_llm_bench.runner import _select_tasks

FULL = 3


class TestProbeVerdictTruthTable:
    """Every row of the documented verdict table, at k=3."""

    @pytest.mark.parametrize("weak,mid,strong,expect_verdict,expect_reason", [
        # Nobody fails: zero signal, the v0.13 fm_22 shape.
        ((3, 3), (3, 3), (3, 3), "REJECT", "saturated"),
        # The strongest model cannot pass either — ambiguous prompt and
        # genuinely-hard look identical from the bottom, and only the first is
        # actionable, so this must not read as "hard".
        ((0, 3), (0, 3), (1, 3), "REJECT", "broken"),
        # Flat in the middle of the range.
        ((2, 3), (2, 3), (2, 3), "REJECT", "no_signal"),
        # Monotone, live, two trials of separation.
        ((0, 3), (1, 3), (3, 3), "ACCEPT", "discriminates"),
        ((0, 3), (0, 3), (3, 3), "ACCEPT", "discriminates"),
        ((1, 3), (3, 3), (3, 3), "ACCEPT", "discriminates"),
        # One trial wide: real, but 3 trials cannot resolve it.
        ((1, 3), (1, 3), (2, 3), "REPROBE", "narrow"),
        ((2, 3), (3, 3), (3, 3), "REPROBE", "narrow"),
    ])
    def test_rows(self, weak, mid, strong, expect_verdict, expect_reason):
        v = probe_verdict(weak, mid, strong)
        assert (v.verdict, v.reason) == (expect_verdict, expect_reason)

    def test_adv_09_inversion_is_rejected_and_named(self):
        """adv_09's real numbers: LFM 1.00, qwen 0.00, gemma 0.00.

        This is the case the whole declared-order design exists for. It must
        reject, and the reason must say *inverted* — reporting it as "broken"
        would hide that the task ranks models backwards.
        """
        v = probe_verdict((3, 3), (0, 3), (0, 3))
        assert v.verdict == "REJECT"
        assert v.reason == "inverted"
        assert v.gap == -1.0

    def test_collect_item_stats_would_have_accepted_the_inversion(self):
        """Why collect_item_stats cannot be reused as the verdict.

        It ranks models into thirds by a headline computed from the same run,
        so on a one-task probe "top" is whichever model scored best and the
        discrimination gap is always >= 0. It literally cannot see an
        inverted task, which is exactly the defect the probe must catch.
        """
        rows = _fake_item_stats_for_inverted_task()
        assert rows[0]["discrimination"] >= DISCRIMINATION_FLOOR
        # ...whereas the probe rejects it.
        assert probe_verdict((3, 3), (0, 3), (0, 3)).verdict == "REJECT"

    def test_partial_inversion_in_the_middle_rejects(self):
        v = probe_verdict((1, 3), (0, 3), (3, 3))
        assert (v.verdict, v.reason) == ("REJECT", "inverted")

    def test_ties_are_not_inversions(self):
        assert probe_verdict((0, 3), (0, 3), (3, 3)).verdict == "ACCEPT"

    def test_splits_name_the_separating_pair(self):
        assert probe_verdict((0, 3), (0, 3), (3, 3)).splits == ("m|s",)
        assert probe_verdict((0, 3), (3, 3), (3, 3)).splits == ("w|m",)
        assert probe_verdict((0, 3), (1, 3), (3, 3)).splits == ("w|m", "m|s")

    def test_floor_sits_between_the_two_bands(self):
        """The 1/3 grid is why the probe cannot just use DISCRIMINATION_FLOOR."""
        assert 1 / 3 < DISCRIMINATION_FLOOR < 2 / 3


class TestProbeVerdictInvalid:
    """A short trial count means unreadable, never zero."""

    def test_short_trials_are_invalid_not_failure(self):
        v = probe_verdict((0, 2), (1, 3), (3, 3))
        assert v.verdict == "INVALID"
        assert "weak" in v.detail

    def test_zero_counted_trials_is_invalid(self):
        """A code candidate with no sandbox backend yields an empty aggregate.

        counted() drops skipped and incomplete trials, so this must key on the
        trial count rather than on a missing key, or it reads as a silent 0/3.
        """
        assert probe_verdict((0, 0), (0, 0), (0, 0)).verdict == "INVALID"

    def test_infra_error_is_invalid(self):
        v = probe_verdict((0, 3), (1, 3), (3, 3), infra_error=True)
        assert v.verdict == "INVALID"

    def test_reprobe_raises_the_bar_at_k5(self):
        """At k=5 resolution is 1/5, so the smallest gap clearing the floor
        is 0.4 — one trial of separation is no longer enough."""
        assert probe_verdict((3, 5), (3, 5), (4, 5), trials=5).verdict == "REPROBE"
        assert probe_verdict((1, 5), (2, 5), (4, 5), trials=5).verdict == "ACCEPT"


class TestTasksDirIsolation:
    """A scratch bank must run without touching tasks/."""

    def test_select_tasks_reads_the_scratch_bank(self, tmp_path):
        (tmp_path / "tools.yaml").write_text(yaml.safe_dump({
            "module": "tools",
            "tasks": [{"id": "probe_01", "difficulty": "hard",
                       "prompt": "hi", "tools": ["get_weather"],
                       "expected": {"tool": "get_weather"}}],
        }))
        selected = _select_tasks("full", None, ["tools"], tmp_path)
        assert [t.id for _, t in selected] == ["probe_01"]

    def test_scratch_bank_needs_only_the_one_module_yaml(self, tmp_path):
        """_select_tasks only loads modules named in --modules, so a candidate
        never requires stubbing the other six banks."""
        (tmp_path / "knowledge.yaml").write_text(yaml.safe_dump({
            "module": "knowledge",
            "tasks": [{"id": "kn_probe", "difficulty": "hard",
                       "prompt": "2+2?", "expected": {"answer": 4}}],
        }))
        assert len(list(tmp_path.iterdir())) == 1
        selected = _select_tasks("full", None, ["knowledge"], tmp_path)
        assert [t.id for _, t in selected] == ["kn_probe"]

    def test_real_bank_still_loads_when_tasks_dir_is_none(self, tasks_dir):
        assert len(load_tasks("tools", tasks_dir=tasks_dir)) >= 6

    def test_probe_forces_full_profile(self, tmp_path):
        """A candidate with no `fast:` key vanishes under profile=fast, which
        would surface as a confusing 'no tasks match filter'."""
        (tmp_path / "knowledge.yaml").write_text(yaml.safe_dump({
            "module": "knowledge",
            "tasks": [{"id": "kn_probe", "difficulty": "hard",
                       "prompt": "2+2?", "expected": {"answer": 4}}],
        }))
        assert _select_tasks("full", None, ["knowledge"], tmp_path)
        with pytest.raises(ValueError, match="no tasks match"):
            _select_tasks("fast", None, ["knowledge"], tmp_path)


class TestClobberGuard:
    """A probe file in results/ would destroy a sweep AND poison items."""

    def test_rejects_a_path_inside_the_results_dir(self, tmp_path):
        settings = BenchSettings()
        settings.output_dir = tmp_path / "results"
        settings.output_dir.mkdir()
        with pytest.raises(ValueError, match="refusing to write"):
            assert_not_results_dir(
                settings.output_dir / "qwen_raw_results.json", settings)

    def test_rejects_a_nested_path(self, tmp_path):
        settings = BenchSettings()
        settings.output_dir = tmp_path / "results"
        settings.output_dir.mkdir()
        with pytest.raises(ValueError):
            assert_not_results_dir(
                settings.output_dir / "deep" / "x_raw_results.json", settings)

    def test_allows_the_scratch_path(self, tmp_path):
        settings = BenchSettings()
        settings.output_dir = tmp_path / "results"
        settings.output_dir.mkdir()
        assert_not_results_dir(tmp_path / "probe" / "x.json", settings)

    def test_run_requires_output_with_tasks_dir(self, tmp_path):
        """The CLI-level half of the guard."""
        from small_llm_bench import cli
        with pytest.raises(typer.BadParameter, match="requires --output"):
            cli.run(model="m", endpoint=None, fast=False, profile=None,
                    output=None, concurrency=None, save_responses=False,
                    verbose=False, task_filter=None, modules="tools",
                    sandbox=None, allow_unsandboxed=False, sandbox_memory=None,
                    trials=None, max_tokens=None, temperature=None,
                    thinking=False, only_new=False, reuse_params=False,
                    ignore_task_hash=False, ignore_world_hash=False,
                    add_trials=None, tasks_dir=tmp_path)


class TestCycleCap:
    """Soft norms are what failed in v0.13; the cap is a refusal."""

    def _log(self, path, task_id, verdict):
        with path.open("a") as fh:
            fh.write(json.dumps({"task_id": task_id, "verdict": verdict}) + "\n")

    def test_counts_only_scoring_cycles(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self._log(log, "x_01", "REJECT")
        self._log(log, "x_01", "REPROBE")
        assert cycles_for(log, "x_01") == 2

    def test_invalid_cycles_do_not_spend_the_budget(self, tmp_path):
        """Infra errors and missing sandboxes say nothing about the task."""
        log = tmp_path / "log.jsonl"
        self._log(log, "x_01", "INVALID")
        self._log(log, "x_01", "INVALID")
        self._log(log, "x_01", "REJECT")
        assert cycles_for(log, "x_01") == 1

    def test_other_tasks_do_not_count(self, tmp_path):
        log = tmp_path / "log.jsonl"
        self._log(log, "y_01", "REJECT")
        assert cycles_for(log, "x_01") == 0

    def test_missing_log_is_zero(self, tmp_path):
        assert cycles_for(tmp_path / "nope.jsonl", "x_01") == 0

    def test_default_cap_is_three(self):
        assert ProbeSettings().max_cycles == FULL


class TestPreRegistration:
    """The one bias control that produces evidence rather than a vibe."""

    def _write(self, tmp_path, expect):
        raw = {"id": "p_01", "difficulty": "hard", "prompt": "x",
               "expected": {"answer": 1}}
        if expect is not None:
            raw["probe_expect"] = expect
        (tmp_path / "knowledge.yaml").write_text(
            yaml.safe_dump({"module": "knowledge", "tasks": [raw]}))

    def test_reads_the_declared_expectation(self, tmp_path):
        self._write(tmp_path, {"weak": "fail", "mid": "fail", "strong": "pass"})
        assert read_expectation(tmp_path, "knowledge", "p_01")["strong"] == "pass"

    def test_absent_expectation_is_none(self, tmp_path):
        self._write(tmp_path, None)
        assert read_expectation(tmp_path, "knowledge", "p_01") is None

    def test_probe_expect_is_inert_to_the_runner(self, tmp_path):
        """A probe-time annotation only: it must never reach the model or enter
        task_content_hash, or a pre-registration would change the task."""
        self._write(tmp_path, {"weak": "fail"})
        task = load_tasks("knowledge", tasks_dir=tmp_path)[0]
        assert not hasattr(task, "probe_expect")
        assert "probe_expect" not in task.model_dump_json(exclude={"fast"})

    def test_fingerprint_changes_when_the_candidate_is_edited(self, tmp_path):
        self._write(tmp_path, {"weak": "fail"})
        first = candidate_fingerprint(tmp_path, "knowledge")
        self._write(tmp_path, {"weak": "pass"})
        assert candidate_fingerprint(tmp_path, "knowledge") != first


class TestJudgeDivergence:
    """Measured on the first real tools probe, not hypothesised.

    Probing a copy of tst_35 (an indentation-exact in-place edit) against the
    trio produced det 0/3, 1/3, 2/3 — monotone, gap 2/3, a clean ACCEPT — while
    the judge rescued the weakest model's trials (llm 1.00 over det_score 0.85
    and 0.87) to 2/3, 1/3, 2/3, an inversion. The judge cannot see structural
    damage in a file, which is exactly what that task grades, so the probe must
    report both rather than silently pick one.
    """

    DET = ((0, 3), (1, 3), (2, 3))
    JUDGED = ((2, 3), (1, 3), (2, 3))

    def test_deterministic_grading_accepts(self):
        v = probe_verdict(*self.DET)
        assert v.verdict == "ACCEPT"
        assert v.gap == pytest.approx(2 / 3)

    def test_judged_grading_inverts(self):
        v = probe_verdict(*self.JUDGED)
        assert (v.verdict, v.reason) == ("REJECT", "inverted")

    def test_the_two_disagree(self):
        """The regression this guards: if a future change makes the probe
        report only one number, this task's real signal disappears."""
        assert (probe_verdict(*self.DET).verdict
                != probe_verdict(*self.JUDGED).verdict)

    def test_divergence_is_only_possible_where_the_judge_decides(self):
        assert JUDGE_DECIDES == {"tools", "multi_turn_if"}


class TestProbeWiring:

    def test_judge_decides_only_where_a_verdict_can_move_success(self):
        """format/knowledge are never judged; adversarial/code/long_context are
        judged for display only. Skipping those is most of the loop's speed."""
        assert JUDGE_DECIDES == {"tools", "multi_turn_if"}

    def test_paths_all_live_under_the_scratch_root(self):
        settings = ProbeSettings()
        for path in probe_paths(settings, "x_01"):
            assert settings.dir in path.parents or path.parent == settings.dir

    def test_scratch_root_is_gitignored(self):
        from pathlib import Path
        ignored = Path(".gitignore").read_text()
        assert ".scratch/" in ignored
        assert str(ProbeSettings().dir).startswith(".scratch/")

    def test_declared_trio_is_weak_to_strong(self):
        """The 2.6B floor slot was dropped in v0.14: three of the four ACCEPTs
        the old trio produced split only 2.6B-vs-4B, certifying tasks that say
        nothing about the 4B-vs-27B gap the bank lacks."""
        assert ProbeSettings().model_list() == [
            "qwen3.5-4b", "gemma-4-12b", "qwen3.6-27b"]


def _fake_item_stats_for_inverted_task():
    """collect_item_stats over three synthetic single-task 'models'.

    Built to mirror adv_09: the weakest model passes, the two stronger ones
    fail. Returned rows are used to show the metric reports a healthy positive
    discrimination for a task that ranks models backwards.
    """
    from small_llm_bench.models import BenchMeta, BenchResult, TaskResult
    from small_llm_bench.reporter import save_results
    import tempfile
    from pathlib import Path

    paths = []
    tmp = Path(tempfile.mkdtemp())
    for name, passed in (("weak", True), ("mid", False), ("strong", False)):
        results = [
            TaskResult(task_id="adv_09", module="adversarial", tier="hard",
                       band="mid", prompt="x", response_raw="y",
                       det_score=1.0 if passed else 0.0,
                       success=passed, det_success=passed)
            for _ in range(FULL)
        ]
        meta = BenchMeta(model=name, endpoint="e", timestamp="t",
                         duration_seconds=1.0, bench_version="test")
        bench = BenchResult(meta=meta, results=results)
        path = tmp / f"{name}_raw_results.json"
        save_results(bench, path)
        paths.append(path)
    return collect_item_stats(paths)
