"""Tests for cross-model item analysis (analysis.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from small_llm_bench.analysis import (collect_item_stats, find_result_files,
                                      pairwise_separability, sign_test_p)
from small_llm_bench.models import BenchMeta, BenchResult, TaskResult
from small_llm_bench.reporter import save_results


def _r(**kw) -> TaskResult:
    base = dict(task_id="t", module="code", prompt="p", tier="baseline")
    base.update(kw)
    return TaskResult(**base)


def _meta(model: str) -> BenchMeta:
    return BenchMeta(model=model, endpoint="http://x", timestamp="now",
                     duration_seconds=1.0, bench_version="0.4.0", trials=1)


def _write(tmp_path: Path, name: str, results: list[TaskResult]) -> Path:
    path = tmp_path / f"{name}_raw_results.json"
    save_results(BenchResult(meta=_meta(name), results=results), path)
    return path


def test_saturated_task_has_zero_discrimination(tmp_path):
    # "sat" always passes for every model: no discrimination, pass_rate 1.0.
    strong = _write(tmp_path, "strong", [
        _r(task_id="sat", success=True, det_success=True, det_score=1.0),
        _r(task_id="hard", success=True, det_success=True, det_score=1.0),
    ])
    weak = _write(tmp_path, "weak", [
        _r(task_id="sat", success=True, det_success=True, det_score=1.0),
        _r(task_id="hard", success=False, det_success=False, det_score=0.0),
    ])
    rows = {r["task_id"]: r for r in collect_item_stats([strong, weak])}
    assert rows["sat"]["pass_rate"] == 1.0
    assert rows["sat"]["discrimination"] == 0.0
    assert rows["hard"]["pass_rate"] == 0.5


def test_discriminating_task_separates_top_from_bottom(tmp_path):
    files = []
    # 3 models: strong passes everything, mid passes the discriminator half the
    # time, weak never passes it. Ranking is by each model's own headline
    # (driven by many "other" tasks it always passes/fails).
    for name, hard_pass in (("strong", True), ("mid", True), ("weak", False)):
        results = [_r(task_id="disc", success=hard_pass, det_success=hard_pass,
                      det_score=1.0 if hard_pass else 0.0)]
        # Filler tasks to separate model headline ranking cleanly.
        n_other_pass = {"strong": 5, "mid": 2, "weak": 0}[name]
        for i in range(5):
            ok = i < n_other_pass
            results.append(_r(task_id=f"other_{i}", success=ok, det_success=ok,
                              det_score=1.0 if ok else 0.0))
        files.append(_write(tmp_path, name, results))

    rows = {r["task_id"]: r for r in collect_item_stats(files)}
    assert rows["disc"]["pass_rate"] < 1.0
    assert rows["disc"]["discrimination"] > 0.0


def test_flaky_counts_partial_trial_passes(tmp_path):
    path = _write(tmp_path, "m", [
        _r(task_id="flaky", success=True, det_success=True, det_score=1.0),
        _r(task_id="flaky", success=False, det_success=False, det_score=0.0),
    ])
    rows = {r["task_id"]: r for r in collect_item_stats([path])}
    assert rows["flaky"]["flaky_models"] == 1


def test_find_result_files_excludes_judged_by_default(tmp_path):
    raw = tmp_path / "a_raw_results.json"
    judged = tmp_path / "a_raw_results_judged.json"
    raw.touch()
    judged.touch()
    assert find_result_files(tmp_path) == [raw]
    assert find_result_files(tmp_path, judged=True) == [judged]


def test_empty_input_returns_empty_list():
    assert collect_item_stats([]) == []


def test_wilson_interval_bounds():
    from small_llm_bench.analysis import wilson_interval

    low, high = wilson_interval(9, 10)
    assert 0.55 < low < 0.60 and 0.97 < high <= 1.0
    # never escapes [0, 1], which the normal approximation would at p near 1
    assert wilson_interval(45, 45)[1] == 1.0
    assert wilson_interval(0, 45)[0] == 0.0
    # no division by zero on an empty bank
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_classify_task_labels_saturated_and_discriminating():
    from small_llm_bench.analysis import classify_task

    def _row(pass_rate, discrimination=0.0, flaky=0, n_models=7, task_id="x"):
        return {"pass_rate": pass_rate, "discrimination": discrimination,
                "flaky_models": flaky, "n_models": n_models,
                "task_id": task_id}

    assert classify_task(_row(1.0), 7) == "dead_easy"
    assert classify_task(_row(0.0), 7) == "dead_hard"
    # a task everyone passes but flakily still carries signal
    assert classify_task(_row(1.0, flaky=4), 7) == "flaky"
    assert classify_task(_row(0.5, discrimination=0.6), 7) == "discriminating"
    assert classify_task(_row(0.5), 7) == "weak"
    # too few models to call a task dead
    assert classify_task(_row(1.0, n_models=2), 2) == "weak"


def test_restraint_tasks_are_not_judged_by_discrimination():
    """adv_09 is a tool-description injection test that 7 of 10 reference models
    obey, so it scores -0.89 discrimination. Read as a capability item that
    looks like an anti-signal task to delete; it is the opposite — stronger
    models follow tool metadata more faithfully, and that is the finding.
    """
    from small_llm_bench.analysis import _RESTRAINT_TASKS, classify_task

    def _row(task_id, pass_rate=0.4, discrimination=-0.9):
        return {"pass_rate": pass_rate, "discrimination": discrimination,
                "flaky_models": 0, "n_models": 7, "task_id": task_id}

    for task_id in _RESTRAINT_TASKS:
        assert classify_task(_row(task_id), 7) == "restraint"
    # an ordinary task with the same numbers is still judged normally
    assert classify_task(_row("tl_26"), 7) == "weak"
    # ...and a restraint task nobody fails is still dead weight
    assert classify_task(_row("ds_15", pass_rate=1.0, discrimination=0.0),
                         7) == "dead_easy"


def test_sample_size_paired_is_cheaper_than_independent():
    from small_llm_bench.analysis import (sample_size_independent,
                                          sample_size_paired)

    indep = sample_size_independent(0.78, 0.05)
    paired = sample_size_paired(0.20, 0.05)
    assert 1000 < indep < 1150          # ~1075 tasks for a 5-point gap
    assert 600 < paired < 660           # ~630 on a shared bank
    assert paired < indep
    # tightening the gap costs quadratically
    assert sample_size_independent(0.78, 0.10) < indep / 3
    assert sample_size_independent(0.78, 0.0) == 0


class TestPairwiseSeparability:
    """Sorting rows by headline prints a rank order whether or not one exists.
    Over the six models measured at v0.10 almost every pair was a tie, so a
    reader taking the sort at face value read findings that were not there.
    """

    def _files(self, tmp_path, models: dict[str, list[bool]]):
        """One saved file per model; `models` maps name -> per-task pass flags."""
        paths = []
        for name, flags in models.items():
            results = [
                TaskResult(task_id=f"t{i}", module="tools", prompt="p",
                           response_raw="r", det_score=1.0 if ok else 0.0,
                           success=ok, det_success=ok)
                for i, ok in enumerate(flags)
            ]
            bench = BenchResult(meta=_meta(name), results=results)
            path = tmp_path / f"{name}_raw_results.json"
            save_results(bench, path)
            paths.append(path)
        return paths

    def test_a_clean_sweep_is_separable(self, tmp_path):
        paths = self._files(tmp_path, {
            "strong": [True] * 20,
            "weak": [False] * 20,
        })
        row = pairwise_separability(paths)[0]
        assert row["better"] == "strong"
        assert (row["wins"], row["losses"]) == (20, 0)
        assert row["separable"] is True

    def test_a_narrow_edge_is_a_tie(self, tmp_path):
        """4-1 on discordant tasks is a 6-point headline gap and still noise."""
        paths = self._files(tmp_path, {
            "a": [True] * 15 + [False] * 5,
            "b": [True] * 11 + [False] * 4 + [True] * 1 + [False] * 4,
        })
        row = pairwise_separability(paths)[0]
        assert row["separable"] is False
        assert row["tasks_needed"] > 0

    def test_identical_models_are_a_tie_with_no_discordance(self, tmp_path):
        paths = self._files(tmp_path, {"a": [True, False] * 10,
                                       "b": [True, False] * 10})
        row = pairwise_separability(paths)[0]
        assert row["discordant"] == 0
        assert row["p_value"] == 1.0
        assert row["separable"] is False

    def test_holm_correction_demotes_a_borderline_pair(self, tmp_path):
        """With many pairs on the board an uncorrected 0.05 promotes ties by
        chance. A 5-0 sweep is p=0.0625 alone; it must not become a finding
        just because other comparisons are running alongside it."""
        models = {f"m{i}": [True] * 10 + [False] * 10 for i in range(5)}
        models["edge"] = [True] * 15 + [False] * 5
        rows = pairwise_separability(self._files(tmp_path, models))
        for row in rows:
            assert row["p_adjusted"] >= row["p_value"]
        assert all(row["p_adjusted"] <= 1.0 for row in rows)

    def test_adjacent_pairs_are_flagged(self, tmp_path):
        paths = self._files(tmp_path, {
            "top": [True] * 20,
            "mid": [True] * 10 + [False] * 10,
            "low": [False] * 20,
        })
        rows = {(r["better"], r["worse"]): r
                for r in pairwise_separability(paths)}
        assert rows[("top", "mid")]["adjacent"] is True
        assert rows[("mid", "low")]["adjacent"] is True
        assert rows[("top", "low")]["adjacent"] is False


class TestSignTest:
    def test_two_sided_exact_values(self):
        assert sign_test_p(0, 0) == 1.0
        assert sign_test_p(5, 0) == pytest.approx(0.0625)
        assert sign_test_p(6, 0) == pytest.approx(0.03125)
        assert sign_test_p(3, 3) == 1.0

    def test_direction_does_not_matter(self):
        assert sign_test_p(7, 2) == sign_test_p(2, 7)


class TestBandRestrictedReading:
    """The panel-wide discrimination stat cannot see the band the bank is tuned
    for. With twelve models, top-third minus bottom-third is top four against
    BOTTOM four, and this fleet's bottom four are all sub-3B-active — so a task
    that separates a 0.8B from a 27B scores +1.00 while telling you nothing
    about 4B versus 27B. Eleven probed candidates and a 0/6 band cohort are what
    that blind spot cost."""

    def test_band_discrimination_reads_the_declared_cohort(self, tmp_path):
        # `floor` splits weak-from-strong only OUTSIDE the cohort; `band` splits
        # inside it. The old stat cannot tell them apart.
        files = [
            _write(tmp_path, "tiny", [
                _r(task_id="floor", success=False, det_success=False, det_score=0.0),
                _r(task_id="band", success=False, det_success=False, det_score=0.0)]),
            _write(tmp_path, "weak", [
                _r(task_id="floor", success=True, det_success=True, det_score=1.0),
                _r(task_id="band", success=False, det_success=False, det_score=0.0)]),
            _write(tmp_path, "strong", [
                _r(task_id="floor", success=True, det_success=True, det_score=1.0),
                _r(task_id="band", success=True, det_success=True, det_score=1.0)]),
        ]
        rows = {r["task_id"]: r
                for r in collect_item_stats(files, cohort=["weak", "strong"])}
        assert rows["floor"]["band_discrimination"] == 0.0
        assert rows["band"]["band_discrimination"] == 1.0

    def test_a_cohort_model_that_never_ran_reports_none_not_zero(self, tmp_path):
        """Absent is not "no signal" — collapsing the two would let a task that
        was never run read as measured-and-flat."""
        files = [_write(tmp_path, "weak", [
            _r(task_id="only", success=True, det_success=True, det_score=1.0)])]
        rows = {r["task_id"]: r
                for r in collect_item_stats(files, cohort=["weak", "absent"])}
        assert rows["only"]["band_discrimination"] is None

    def test_band_reading_keeps_declared_order_so_inversion_stays_visible(self, tmp_path):
        """Never re-sorted by observed score. Sorting would make the gap >= 0 by
        construction and hide a task that ranks the band backwards — the defect
        `probe_verdict` exists to catch (adv_09)."""
        files = [
            _write(tmp_path, "weak", [
                _r(task_id="inv", success=True, det_success=True, det_score=1.0)]),
            _write(tmp_path, "strong", [
                _r(task_id="inv", success=False, det_success=False, det_score=0.0)]),
        ]
        rows = {r["task_id"]: r
                for r in collect_item_stats(files, cohort=["weak", "strong"])}
        assert rows["inv"]["band_discrimination"] == -1.0

    def test_floor_only_is_not_dead_easy_and_not_discriminating(self):
        from small_llm_bench.analysis import classify_task
        row = {"pass_rate": 0.6, "discrimination": 1.0, "flaky_models": 1,
               "n_models": 12, "task_id": "x", "band": "mid",
               "band_discrimination": 0.0, "band_fracs": [1.0, 1.0, 1.0]}
        # +1.00 on the panel stat, zero signal in the band.
        assert classify_task(row, 12) == "floor_only"

    def test_an_anchor_is_never_relabelled_floor_only(self):
        """Saturation is what an anchor is FOR; it carries 0.10 of the
        band-weighted headline as a floor sentinel."""
        from small_llm_bench.analysis import classify_task
        row = {"pass_rate": 1.0, "discrimination": 0.0, "flaky_models": 0,
               "n_models": 12, "task_id": "a", "band": "anchor",
               "band_discrimination": 0.0, "band_fracs": [1.0, 1.0, 1.0]}
        assert classify_task(row, 12) == "anchor"

    def test_without_a_cohort_classification_is_unchanged(self):
        from small_llm_bench.analysis import classify_task
        row = {"pass_rate": 0.6, "discrimination": 1.0, "flaky_models": 1,
               "n_models": 12, "task_id": "x", "band": "mid"}
        assert classify_task(row, 12) == "discriminating"


class TestCohortSeparability:
    """Holm's threshold is set by how many pairs are on the board, and k
    consistently-won tasks give an exact two-sided sign p of 2^(1-k). Correcting
    a 4B-vs-27B question across 66 pairs — most of them sub-3B comparisons
    nobody asked about — is what makes the band unanswerable."""

    def test_cohort_restricts_the_pairs_and_therefore_the_correction(self, tmp_path):
        files = [_write(tmp_path, name, [
            _r(task_id=f"t{i}", success=(name != "weak"),
               det_success=(name != "weak"), det_score=0.0 if name == "weak" else 1.0)
            for i in range(8)])
            for name in ("weak", "mid", "strong", "extra")]
        assert len(pairwise_separability(files)) == 6          # 4 models
        assert len(pairwise_separability(files, cohort=["weak", "strong"])) == 1

    def test_a_cohort_naming_an_absent_model_just_drops_it(self, tmp_path):
        files = [_write(tmp_path, name, [
            _r(task_id="t", success=True, det_success=True, det_score=1.0)])
            for name in ("weak", "strong")]
        rows = pairwise_separability(files, cohort=["weak", "strong", "ghost"])
        assert len(rows) == 1
