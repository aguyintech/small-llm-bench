"""Tests for leaderboard row assembly: judge coverage and run comparability."""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

from small_llm_bench.analysis import pairwise_separability
from small_llm_bench.leaderboard import (assign_size_bucket, assign_tier,
                                         build_leaderboard,
                                         load_model_registry,
                                         render_leaderboard_html)
from small_llm_bench.models import (BenchMeta, BenchResult, TaskResult,
                                    results_task_set_hash)


def _bench(model: str, *, judged_modules: set[str] | None = None,
           endpoint: str = "http://gpu:8092/v1", task_hash: str = "aaa",
           trials: int = 3) -> BenchResult:
    meta = BenchMeta(model=model, endpoint=endpoint, timestamp="t",
                     duration_seconds=1.0, bench_version="0.7.0",
                     trials=trials, thinking=True, max_tokens=8192)
    results = []
    for module, det in (("code", 1.0), ("tool_loop", 0.5)):
        for _ in range(trials):
            r = TaskResult(task_id=f"{module}_01", module=module, prompt="p",
                           det_score=det, success=det >= 0.999,
                           det_success=det >= 0.999, task_hash=task_hash)
            if judged_modules and module in judged_modules:
                r.llm_score = det
            results.append(r)
    return BenchResult(meta=meta, results=results)


def _write(tmp_path, bench: BenchResult, judged: bool) -> None:
    stem = f"{bench.meta.model}_raw_results"
    (tmp_path / f"{stem}.json").write_text(json.dumps(bench.model_dump()))
    if judged:
        (tmp_path / f"{stem}_judged.json").write_text(
            json.dumps(bench.model_dump()))


def test_row_flags_incomplete_judge_coverage(tmp_path):
    """A run whose judge failed on one module is flagged, not silently ranked:
    its judged columns fall back to det on the missing module."""
    _write(tmp_path, _bench("partial", judged_modules={"code"}), judged=True)
    _write(tmp_path, _bench("complete", judged_modules={"code", "tool_loop"}),
           judged=True)
    rows = {r["model"]: r for r in build_leaderboard(tmp_path)["rows"]}

    assert rows["partial"]["judge_complete"] is False
    assert rows["partial"]["judge_coverage"] == 0.5
    assert rows["partial"]["unjudged_modules"] == ["tool_loop"]
    assert rows["complete"]["judge_complete"] is True
    assert rows["complete"]["unjudged_modules"] == []
    # the unjudged module still carries weight via its deterministic score,
    # so a judge failure can't inflate the judged column
    assert rows["partial"]["overall_llm"] == rows["complete"]["overall_llm"]


def test_row_flags_run_setting_mismatches(tmp_path):
    """A task-bank divergence makes two rows unrankable against each other, so
    it is named on the offending row. A different endpoint does not: it is
    where the server was, and flagging it marked every row a user added to the
    shipped reference panel."""
    _write(tmp_path, _bench("a"), judged=False)
    _write(tmp_path, _bench("b"), judged=False)
    _write(tmp_path, _bench("other_bank", task_hash="zzz"), judged=False)
    _write(tmp_path, _bench("other_host", endpoint="http://localhost:8000/v1"),
           judged=False)
    data = build_leaderboard(tmp_path)
    rows = {r["model"]: r for r in data["rows"]}

    assert rows["a"]["comparability_mismatch"] == []
    assert rows["b"]["comparability_mismatch"] == []
    assert rows["other_bank"]["comparability_mismatch"] == ["task_set_hash"]
    assert rows["other_host"]["comparability_mismatch"] == []
    assert "endpoint" not in data["comparability"]["modal"]


def test_comparability_skipped_for_a_single_row(tmp_path):
    _write(tmp_path, _bench("only"), judged=False)
    data = build_leaderboard(tmp_path)
    assert data["comparability"] == {}
    assert "comparability_mismatch" not in data["rows"][0]


def test_task_set_hash_recomputed_for_pre_v07_files(tmp_path):
    """Files written before BenchMeta.task_set_hash existed still take part in
    the comparability check — the hash is derived from per-result task_hash."""
    bench = _bench("legacy")
    bench.meta.task_set_hash = ""
    _write(tmp_path, bench, judged=False)
    row = build_leaderboard(tmp_path)["rows"][0]
    assert row["run_settings"]["task_set_hash"] == \
        results_task_set_hash(bench.results)


def test_html_renders_coverage_markup(tmp_path):
    _write(tmp_path, _bench("partial", judged_modules={"code"}), judged=True)
    _write(tmp_path, _bench("other_host", endpoint="http://localhost:8000/v1"),
           judged=False)
    html = render_leaderboard_html(build_leaderboard(tmp_path))
    assert "cov-badge" in html


def test_the_config_mismatch_badge_is_gone_but_the_data_is_not(tmp_path):
    """The badge fired on any run-config difference, `task_set_hash` included,
    so growing the bank made it appear on every row — a badge on all rows says
    nothing. The per-row list stays in the JSON as the machine-readable record.
    """
    _write(tmp_path, _bench("a"), judged=False)
    _write(tmp_path, _bench("other_bank", task_hash="zzz"), judged=False)
    data = build_leaderboard(tmp_path)
    assert data["rows"][0]["comparability_mismatch"] is not None

    html = render_leaderboard_html(data)
    assert "&ne;" not in html
    assert "mismatchDetail" not in html
    assert "config mismatch" not in html


# --- the per-row tie badge is gone (v0.15) -----------------------------------

def _graded(model: str, per_task: dict[str, float], trials: int = 3) -> BenchResult:
    """A run with an explicit pass fraction per task, so ties are controllable."""
    meta = BenchMeta(model=model, endpoint="http://gpu:8092/v1", timestamp="t",
                     duration_seconds=1.0, bench_version="0.7.0",
                     trials=trials, thinking=True, max_tokens=8192)
    results = []
    for task_id, frac in per_task.items():
        passes = round(frac * trials)
        for i in range(trials):
            ok = i < passes
            results.append(TaskResult(
                task_id=task_id, module="tools", prompt="p",
                det_score=1.0 if ok else 0.0, success=ok, det_success=ok,
                task_hash="aaa"))
    return BenchResult(meta=meta, results=results)


def test_separable_pair_count_replaces_the_per_row_badge(tmp_path):
    """One honest count, instead of a `=` on nearly every row.

    On the real 11-model board the badge fired on 10 of 10 adjacent pairs,
    which marks the whole table and so tells a reader nothing.
    """
    tasks = [f"t{i:02d}" for i in range(20)]
    _write(tmp_path, _graded("strong", {t: 1.0 for t in tasks}), False)
    _write(tmp_path, _graded("weak", {t: 0.0 for t in tasks}), False)
    data = build_leaderboard(tmp_path)
    assert data["total_pairs"] == 1
    assert data["separable_pairs"] == 1


def test_indistinguishable_models_report_zero_separable_pairs(tmp_path):
    tasks = [f"t{i:02d}" for i in range(20)]
    for name in ("a", "b", "c"):
        _write(tmp_path, _graded(name, {t: 0.5 for t in tasks}), False)
    data = build_leaderboard(tmp_path)
    assert data["total_pairs"] == 3
    assert data["separable_pairs"] == 0
    # The per-row verdicts are still computed — they just do not decorate rows.
    assert all(r.get("tied_with_above") for r in data["rows"][1:])


def test_the_html_carries_no_tie_badge(tmp_path):
    """The per-row `=` badge is gone; the same fact is stated once as a count.

    On a 10-model board it fired on 8 of 9 rows and on 11 it fired on 10 of 10,
    which marks the whole table and reads as an apology for the ranking rather
    than information.
    """
    tasks = [f"t{i:02d}" for i in range(20)]
    for name in ("a", "b", "c"):
        _write(tmp_path, _graded(name, {t: 0.5 for t in tasks}), False)
    html = render_leaderboard_html(build_leaderboard(tmp_path))
    assert "tie-badge" not in html
    assert "pairs separable" in html


def test_score_bands_ship_but_never_replace_the_separability_count(tmp_path):
    """The board has letter tiers again, and they are a different object.

    What was rejected was tiers cut where the sign test stopped resolving rows:
    a bucket spanning a 4B and a 31B presents a limitation OF THE BANK as a
    claim ABOUT the models. `_TIER_BANDS` instead cuts at fixed thresholds on
    the score, so an `S` says only that a model scored above .90.

    Those two facts have to travel together, so this asserts the pair: the
    tiers are rendered AND the separable-pair count is still on the page. A
    change that drops the count while keeping the tiers is the failure mode
    this guards.
    """
    tasks = [f"t{i:02d}" for i in range(20)]
    for name in ("a", "b", "c"):
        _write(tmp_path, _graded(name, {t: 0.5 for t in tasks}), False)
    html = render_leaderboard_html(build_leaderboard(tmp_path))
    assert "tier-chip" in html
    assert "pairs separable" in html
    # The claim the letters are NOT allowed to make.
    assert "not a claim that" in html or "not</em> a claim" in html


def test_assign_tier_cuts_at_the_documented_floors():
    """A score exactly on a floor takes the higher band."""
    assert assign_tier(1.0) == "S"
    assert assign_tier(0.90) == "S"
    assert assign_tier(0.8999) == "A"
    assert assign_tier(0.80) == "A"
    assert assign_tier(0.70) == "B"
    assert assign_tier(0.60) == "C"
    assert assign_tier(0.45) == "D"
    assert assign_tier(0.30) == "E"
    assert assign_tier(0.2999) == "F"
    assert assign_tier(0.0) == "F"


def test_the_board_ranks_on_the_number_it_displays(tmp_path):
    """Ranking on `overall_pass` while leading with the judged score let a model
    hold a row on a number the judge had already taken away from it."""
    tasks = [f"t{i:02d}" for i in range(20)]
    # `high_raw` outscores `steady` before the judge and loses to it after.
    raw = _graded("high_raw", {t: 0.8 for t in tasks})
    for r in raw.results:
        r.llm_score = 0.0 if r.success else 0.0
    _write(tmp_path, raw, judged=True)
    steady = _graded("steady", {t: 0.75 for t in tasks})
    for r in steady.results:
        r.llm_score = 1.0 if r.success else 0.0
    _write(tmp_path, steady, judged=True)

    rows = build_leaderboard(tmp_path)["rows"]
    assert [r["headline"] for r in rows] == sorted(
        (r["headline"] for r in rows), reverse=True)
    for row in rows:
        assert row["headline"] == row["overall_judge_pass"]


def test_an_unjudged_row_falls_back_and_says_so(tmp_path):
    """A model with no judge file still ranks and still gets a tier — flagged,
    rather than quietly dropping out of the charts and shrinking the board."""
    _write(tmp_path, _bench("nojudge"), judged=False)
    _write(tmp_path, _bench("judged", judged_modules={"code", "tool_loop"}),
           judged=True)
    rows = {r["model"]: r for r in build_leaderboard(tmp_path)["rows"]}

    assert rows["nojudge"]["headline_judged"] is False
    assert rows["nojudge"]["headline"] == rows["nojudge"]["overall_pass"]
    assert rows["nojudge"]["tier"] in {"S", "A", "B", "C", "D", "E", "F"}
    assert rows["judged"]["headline_judged"] is True


def test_every_row_lands_in_exactly_one_tier(tmp_path):
    tasks = [f"t{i:02d}" for i in range(20)]
    for name, frac in (("top", 1.0), ("mid", 0.6), ("low", 0.0)):
        _write(tmp_path, _graded(name, {t: frac for t in tasks}), False)
    data = build_leaderboard(tmp_path)

    assert sum(t["count"] for t in data["tiers"]) == len(data["rows"])
    assert [t["letter"] for t in data["tiers"]] == list("SABCDEF")
    # Tiers never improve as you walk down a board sorted by score.
    order = "SABCDEF"
    seen = [order.index(r["tier"]) for r in data["rows"]]
    assert seen == sorted(seen)


def test_the_payload_carries_the_weight_values_not_just_the_preset_name(tmp_path):
    """Without these nothing downstream can say why `tools` moves the headline
    five times as much as `knowledge` does."""
    _write(tmp_path, _bench("m"), judged=False)
    data = build_leaderboard(tmp_path)

    assert data["weights"] == "balanced"
    assert data["module_weights"]["tools"] == 0.30
    assert abs(sum(data["module_weights"].values()) - 1.0) < 1e-9


def test_the_html_ships_four_views_and_a_radar(tmp_path):
    _write(tmp_path, _bench("m"), judged=False)
    html = render_leaderboard_html(build_leaderboard(tmp_path))

    for panel in ("panel-leaderboard", "panel-by-tier", "panel-by-params",
                  "panel-detailed"):
        assert f'id="{panel}"' in html
    assert html.count('role="tab" ') == 4
    assert html.count('role="tabpanel"') == 4
    assert "arch-badge" in html
    assert 'class="radar"' in html
    # The radar plots pass^k, not the det score: det sits between .80 and 1.00
    # for nearly every model and every radar comes out the same shape.
    assert "cell.judge_pass" in html and "cell.pass" in html


# --- footprint buckets (v1.1) ------------------------------------------------

def test_size_buckets_cut_on_total_params():
    assert assign_size_bucket(100) == "xl"
    assert assign_size_bucket(24) == "xl"
    assert assign_size_bucket(23.9) == "l"
    assert assign_size_bucket(7) == "l"
    assert assign_size_bucket(6.9) == "m"
    assert assign_size_bucket(3) == "m"
    assert assign_size_bucket(2.9) == "s"
    assert assign_size_bucket(0.8) == "s"


def test_an_unregistered_model_has_no_bucket_rather_than_the_smallest():
    """No recorded size is not the same fact as being tiny. Filing an unknown
    model under the bottom bucket would assert something the registry does not
    know."""
    assert assign_size_bucket(None) is None


def test_ornith_is_a_sparse_model_not_a_dense_35b(tmp_path):
    """Recorded as dense 35B until 2026-09-19, which put it top of the params
    column and made an A-tier score read as a large model underperforming
    rather than a 3B-active MoE holding its own."""
    registry = load_model_registry(_REPO_ROOT)
    entry = registry["ornith-1.5-35b"]
    assert entry["active_b"] == 3 and entry["params_b"] == 35

    _write(tmp_path, _bench("ornith-1.5-35b"), judged=False)
    data = build_leaderboard(tmp_path, scheme="module")
    row = next(r for r in data["rows"] if r["model"] == "ornith-1.5-35b")
    assert row["size"]["is_sparse"] is True
    assert row["size"]["sort_b"] == 3        # ranks by active
    assert row["size"]["bucket"] == "xl"     # but is loaded whole


def test_bucket_counts_account_for_every_sized_row(tmp_path):
    # One model per bucket side that matters, plus one the registry has never
    # heard of, which must stay out of every bucket rather than land in one.
    for model in ("ornith-1.5-35b", "gemma-4-31b", "gemma-4-12b",
                  "qwen3.5-4b", "qwen3.5-0.8b", "not-in-the-registry"):
        _write(tmp_path, _bench(model), judged=False)
    data = build_leaderboard(tmp_path, scheme="module")
    counted = sum(b["dense"] + b["sparse"] for b in data["size_buckets"])
    sized = [r for r in data["rows"] if r["size"]["bucket"] is not None]
    assert counted == len(sized)
    assert [b["key"] for b in data["size_buckets"]] == ["xl", "l", "m", "s"]


def test_the_sticky_column_offsets_are_measured_not_hardcoded(tmp_path):
    """They were hand-computed constants matched to fixed column widths, so
    anything that changed a cell's contents broke them silently: the tier chip
    and the MoE badge grew columns 1 and 2 to 264px and 163px against declared
    230px and 108px, and columns 2-5 overlapped once the table scrolled."""
    _write(tmp_path, _bench("ornith-1.5-35b"), judged=False)
    html = render_leaderboard_html(build_leaderboard(tmp_path))
    assert "applyStickyOffsets" in html
    for stale in ("left: 230px", "left: 338px", "left: 446px", "left: 554px"):
        assert stale not in html


def test_the_params_column_sorts_on_a_dotted_key(tmp_path):
    """`get` only understood the `modules.` prefix, so `size.sort_b` read an
    undefined property and the comparator's null branch fired on every row —
    clicking Params left the order untouched."""
    _write(tmp_path, _bench("ornith-1.5-35b"), judged=False)
    html = render_leaderboard_html(build_leaderboard(tmp_path))
    assert 'key: "size.sort_b"' in html
    assert 'key.split(".").reduce(' in html


def test_a_judged_file_without_its_raw_sibling_is_a_full_row(tmp_path):
    """The shipped reference panel is judged files only. A judged file holds
    everything its raw one does, so the board must read it rather than start
    from raw files and render nothing."""
    stem = "judged-only_raw_results"
    (tmp_path / f"{stem}_judged.json").write_text(
        json.dumps(_bench("judged-only").model_dump()))
    _write(tmp_path, _bench("both", judged_modules={"code"}), judged=True)
    _write(tmp_path, _bench("raw-only"), judged=False)
    models = sorted(r["model"] for r in build_leaderboard(tmp_path)["rows"])
    assert models == ["both", "judged-only", "raw-only"]
