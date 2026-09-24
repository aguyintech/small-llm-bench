"""v0.17: reading the cohort rung by rung, and the fleet for inversions.

Three additions, one motivation — endpoint-minus-endpoint arithmetic hid the
leg the bank is actually being tuned for:

1. ``band_legs`` / ``band_monotonic`` split the declared cohort into rungs, so
   a task whose weak->mid leg carries the whole number cannot report itself as
   evidence for mid->strong.
2. ``classify_task`` names that case ``inverted_leg`` instead of folding it
   into ``discriminating``.
3. ``fleet_monotonicity`` answers the same question one level up: which
   modules rank a declared-larger model below a smaller one.

``mean_soft`` rides along as a per-model diagnostic — recorded here so the
"soft scoring does not rescue the mid-vs-large gap" finding stays falsifiable.
"""

from __future__ import annotations

from pathlib import Path

from small_llm_bench.analysis import (classify_task, collect_item_stats,
                                      collect_model_stats, fleet_monotonicity)
from small_llm_bench.config import ProbeSettings
from small_llm_bench.models import BenchMeta, BenchResult, TaskResult
from small_llm_bench.reporter import save_results


def _r(task_id: str, ok: bool, module: str = "tools", **kw) -> TaskResult:
    base = dict(task_id=task_id, module=module, prompt="p", tier="baseline",
                success=ok, det_success=ok, det_score=1.0 if ok else 0.0)
    base.update(kw)
    return TaskResult(**base)


def _write(tmp_path: Path, model: str, results: list[TaskResult],
           trials: int = 1) -> Path:
    meta = BenchMeta(model=model, endpoint="http://x", timestamp="now",
                     duration_seconds=1.0, bench_version="0.17.0",
                     trials=trials)
    path = tmp_path / f"{model}_raw_results.json"
    save_results(BenchResult(meta=meta, results=results), path)
    return path


def _row(**kw) -> dict:
    row = {"task_id": "x", "module": "tools", "band": "mid", "pass_rate": 0.6,
           "discrimination": 0.5, "flaky_models": 0, "n_models": 10}
    row.update(kw)
    return row


class TestBandLegs:
    """The cohort read rung by rung, not endpoint minus endpoint."""

    def _cohort_files(self, tmp_path, weak_ok, mid_ok, strong_ok):
        return [
            _write(tmp_path, "weak", [_r("t", weak_ok)]),
            _write(tmp_path, "mid", [_r("t", mid_ok)]),
            _write(tmp_path, "strong", [_r("t", strong_ok)]),
        ]

    def test_legs_are_consecutive_differences(self, tmp_path):
        files = self._cohort_files(tmp_path, False, False, True)
        row = collect_item_stats(files, cohort=["weak", "mid", "strong"])[0]
        assert row["band_fracs"] == [0.0, 0.0, 1.0]
        assert row["band_legs"] == [0.0, 1.0]
        assert row["band_monotonic"] is True

    def test_inverted_middle_is_visible_in_legs_not_in_the_gap(self, tmp_path):
        # The pf_01 shape: the endpoints say +1.00 while the mid->strong leg
        # runs backwards. band_discrimination cannot see it; band_legs can.
        files = [
            _write(tmp_path, "weak", [_r("t", False), _r("t", False)]),
            _write(tmp_path, "mid", [_r("t", True), _r("t", True)]),
            _write(tmp_path, "strong", [_r("t", True), _r("t", False)]),
        ]
        row = collect_item_stats(files, cohort=["weak", "mid", "strong"])[0]
        assert row["band_discrimination"] == 0.5
        assert row["band_legs"] == [1.0, -0.5]
        assert row["band_monotonic"] is False

    def test_absent_cohort_model_leaves_monotonic_unknown(self, tmp_path):
        # Absent is not "monotone": the mid rung never ran, so no verdict.
        files = [_write(tmp_path, "weak", [_r("t", False)]),
                 _write(tmp_path, "strong", [_r("t", True)])]
        row = collect_item_stats(files, cohort=["weak", "mid", "strong"])[0]
        assert row["band_fracs"] == [0.0, None, 1.0]
        assert row["band_legs"] == [None, None]
        assert row["band_monotonic"] is None

    def test_no_cohort_declared_yields_no_legs(self, tmp_path):
        files = [_write(tmp_path, "solo", [_r("t", True)])]
        row = collect_item_stats(files)[0]
        assert row["band_legs"] == []
        assert row["band_monotonic"] is None


class TestInvertedLegClass:
    """A task that ranks a rung backwards is not `discriminating`."""

    def test_inverted_leg_beats_discriminating(self):
        row = _row(band_monotonic=False)
        assert classify_task(row, 10) == "inverted_leg"

    def test_monotone_task_stays_discriminating(self):
        assert classify_task(_row(band_monotonic=True), 10) == "discriminating"

    def test_unmeasured_cohort_stays_discriminating(self):
        # None means "not read", which must not be treated as inverted.
        assert classify_task(_row(band_monotonic=None), 10) == "discriminating"
        row = _row()
        row.pop("band_monotonic", None)
        assert classify_task(row, 10) == "discriminating"

    def test_saturation_still_wins_over_inversion(self):
        # A task everybody passes is dead weight regardless of leg shape.
        row = _row(pass_rate=1.0, discrimination=0.0, band_monotonic=False)
        assert classify_task(row, 10) == "dead_easy"

    def test_class_has_a_table_abbreviation(self):
        # The items table is width-bound; every class needs a short form or
        # the row wraps and the column alignment the table relies on breaks.
        from small_llm_bench.cli import _CLASS_ABBR
        assert "inverted_leg" in _CLASS_ABBR
        assert len(_CLASS_ABBR["inverted_leg"]) <= 6


class TestFleetMonotonicity:
    """Which modules rank a declared-larger model below a smaller one."""

    def _fleet(self, tmp_path):
        # tools runs the declared way; multi_turn_if runs backwards.
        small = _write(tmp_path, "small", [
            _r("a", False, "tools"), _r("b", False, "tools"),
            _r("m1", True, "multi_turn_if"), _r("m2", True, "multi_turn_if"),
        ])
        big = _write(tmp_path, "big", [
            _r("a", True, "tools"), _r("b", True, "tools"),
            _r("m1", False, "multi_turn_if"), _r("m2", False, "multi_turn_if"),
        ])
        return [small, big]

    def test_reports_the_inverted_module_only(self, tmp_path):
        rows = fleet_monotonicity(self._fleet(tmp_path), ["small", "big"])
        assert [r["module"] for r in rows] == ["multi_turn_if"]
        assert rows[0]["smaller"] == "small" and rows[0]["larger"] == "big"
        assert rows[0]["delta"] == 1.0

    def test_declared_order_is_never_re_sorted(self, tmp_path):
        # Passing the fleet backwards must flip which module reads inverted.
        # An order derived from observed scores would report nothing at all,
        # which is exactly the failure this check exists to avoid.
        rows = fleet_monotonicity(self._fleet(tmp_path), ["big", "small"])
        assert [r["module"] for r in rows] == ["tools"]

    def test_min_delta_filters_single_flaky_trials(self, tmp_path):
        rows = fleet_monotonicity(self._fleet(tmp_path), ["small", "big"],
                                  min_delta=1.0)
        assert rows == []

    def test_models_absent_from_the_fleet_are_skipped(self, tmp_path):
        files = self._fleet(tmp_path)
        rows = fleet_monotonicity(files, ["small", "ghost", "big"])
        assert [r["module"] for r in rows] == ["multi_turn_if"]

    def test_empty_fleet_reports_nothing(self, tmp_path):
        assert fleet_monotonicity(self._fleet(tmp_path), []) == []


class TestFleetOrderSetting:
    def test_default_is_empty_so_a_stale_ladder_cannot_ship(self):
        assert ProbeSettings(fleet_order="").fleet_list() == []

    def test_parses_and_strips(self):
        s = ProbeSettings(fleet_order=" a , b ,, c ")
        assert s.fleet_list() == ["a", "b", "c"]


class TestMeanSoft:
    def test_mean_soft_averages_scorable_trials(self, tmp_path):
        path = _write(tmp_path, "m", [
            _r("a", True, det_score=1.0),
            _r("b", False, det_score=0.5),
        ])
        stats = collect_model_stats([path])[0]
        assert stats["mean_soft"] == 0.75

    def test_mean_soft_ignores_uncounted_trials(self, tmp_path):
        # An infra error measured nothing and must not be averaged in as 0.
        path = _write(tmp_path, "m", [
            _r("a", True, det_score=1.0),
            _r("b", False, det_score=0.0, infra_error=True),
        ])
        stats = collect_model_stats([path])[0]
        assert stats["mean_soft"] == 1.0
