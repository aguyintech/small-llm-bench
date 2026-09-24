"""Cross-model leaderboard: static HTML + JSON, models as rows, modules as columns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from .analysis import find_result_files
from .models import results_task_set_hash
from .reporter import (aggregate_by_task, aggregate_module_scores, aggregate_speed,
                       aggregate_timings, headline_overall,
                       judge_coverage, load_results, module_success,
                       recovery_stats)
from .analysis import pairwise_separability
from .scorer import MODULE_WEIGHT_PRESETS, overall_score

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_REGISTRY_NAME = "models.yaml"

# Presentational bands on the headline score. These cut points are an EDITORIAL
# choice, not a statistical one. A tier says "this model scored in the .80s"; it
# does NOT say "the bank can prove this model beats the one below it". That
# second question is answered by `separable_pairs`, which is why the board shows
# the separability count beside the tiers rather than instead of them — see
# `_mark_ties` for the distinction and the history behind it.
#
# Why these numbers. The headline is a module-weighted pass^k rate, and on the
# v1.0 sweep it runs .95 down to .09 with no natural gaps to cut at — so any
# boundary is chosen, and the property that matters is that it stays put between
# runs. Round tenths do that and survive being read aloud on a recording.
#
# The D and E floors are the two exceptions, at .45 and .30 rather than .40 and
# .20. The bottom of the field bunches: four models sit between .45 and .50 and
# two more at .38-.39. Tenths there would put six models in one band and leave
# the next nearly empty, which tells a viewer less than the split does.
_TIER_BANDS: list[tuple[str, float]] = [
    ("S", 0.90), ("A", 0.80), ("B", 0.70), ("C", 0.60),
    ("D", 0.45), ("E", 0.30), ("F", 0.0),
]


# Footprint classes, cut on TOTAL parameters, ordered high->low.
#
# Total, not active, and the difference is deliberate — the board does both and
# they disagree in public, so the reason has to be written down. `sort_b` ranks
# a model by its ACTIVE params, which is the v0.15 decision and answers "how
# fast": ornith-1.5-35b is a 3B-active MoE and sorts among the 3B rows. These
# buckets answer the other question, "will it fit", and there the total is what
# counts. Google's Gemma 4 documentation puts it plainly for 26B-A4B: "all 26
# billion parameters must be loaded into memory ... making its baseline memory
# requirement much closer to a dense 26B model than a 4B model." So a 35B MoE
# belongs beside the 27B dense models here and beside the 3B models there, and
# both placements are correct.
#
# The cuts themselves follow how the local-model community already shops:
# whether a quantised model clears a 24GB card, a 12-16GB card, an 8GB card, or
# has to run on CPU/phone.
_SIZE_BUCKETS: list[tuple[str, str, float]] = [
    ("xl", "24B and above", 24.0),
    ("l", "7B – 23B", 7.0),
    ("m", "3B – 6B", 3.0),
    ("s", "Under 3B", 0.0),
]


def assign_size_bucket(params_b: float | None) -> str | None:
    """Footprint class for a total parameter count. ``None`` when unknown.

    An unregistered model has no size at all, and that is not the same as
    being small — it gets ``None`` and is held out of the by-params view
    rather than silently filed under the bottom bucket.
    """
    if params_b is None:
        return None
    for key, _label, floor in _SIZE_BUCKETS:
        if params_b >= floor:
            return key
    return _SIZE_BUCKETS[-1][0]


def assign_tier(headline: float) -> str:
    """The letter band a headline score falls in.

    The single place the cut points are applied. A score sitting exactly on a
    floor takes the higher band (.90 is S, not A).
    """
    for letter, floor in _TIER_BANDS:
        if headline >= floor:
            return letter
    return _TIER_BANDS[-1][0]


def load_model_registry(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read ``models.yaml``, keyed by model name. Missing file -> empty.

    Nothing in a result file records how big a model is, so without this the
    board ranks a 12B above a 35B with no way to notice. Absent models are not
    an error: they score normally and show `?` for size.
    """
    path = (root or Path.cwd()) / _REGISTRY_NAME
    if not path.exists():
        path = Path(__file__).parent.parent / _REGISTRY_NAME
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {entry["name"]: entry for entry in data.get("models", [])
            if entry.get("name")}


def _size_cell(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Params for the board: total, and active when they differ.

    Ranked by ACTIVE parameters, which is how this bank has always read MoEs —
    gemma-4-26b-a4b lands below gemma-4-12b on its 4B active. Printing the
    total beside that row would read as an inversion that is not one.
    """
    if not entry:
        return {"params_b": None, "active_b": None, "sort_b": None,
                "label": "?", "family": "", "variant_of": None,
                "is_sparse": False, "arch": None, "bucket": None}
    total = entry.get("params_b")
    active = entry.get("active_b")
    label = ("?" if total is None
             else f"{total:g}B" if active is None
             else f"{total:g}B · A{active:g}B")
    # Sparsity is the grouping concept — does a forward pass use everything you
    # had to load? `arch` only names the mechanism, for the badge. It defaults
    # to MoE because every sparse model here routes experts except
    # gemma-4-e2b-it, which reaches its active count through per-layer
    # embeddings; printing "MoE" on that row would be a false claim.
    is_sparse = active is not None
    return {"params_b": total, "active_b": active,
            "sort_b": active if active is not None else total,
            "label": label, "family": entry.get("family", ""),
            "variant_of": entry.get("variant_of"),
            "is_sparse": is_sparse,
            "arch": (entry.get("arch") or "moe") if is_sparse else None,
            "bucket": assign_size_bucket(total)}


def _judged_sibling(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_path.stem}_judged.json")


# Run settings that must agree for two rows to be worth comparing. A row that
# differs on any of these is still shown — silently ranking it against the rest
# is what this guards against.
# ``bench_version`` is in here because a scorer change can move deterministic
# scores without touching a single task: the task_set_hash then still matches
# and two rows graded by different rules would rank against each other in
# silence. The version is the only thing that moves in that case.
# `sampling` and `sandbox_backend` joined in v1.0. pass^k is a statement about
# a sampler, so two rows drawn from different samplers are not comparable
# however identical the bank; and docker/podman run python:3.12-slim while
# bwrap/sandbox-exec/rlimit run the host interpreter, so a code task can
# disagree between rows for a reason that is not the model.
# `endpoint` left after 1.0.0. It says where a server was, not how the model was
# run: every settings difference that matters is checked above on its own, and
# against the shipped reference panel it flagged every row a user added,
# because nobody else's server has the maintainer's address.
_COMPARABILITY_FIELDS = ["task_set_hash", "bench_version", "trials",
                         "thinking", "max_tokens", "profile", "sampling",
                         "sandbox_backend"]

# `served_context` is deliberately NOT above. It is recorded on every row and
# it matters enormously — a window too small rejects prompts unread — but
# EQUALITY is the wrong predicate for it. The bank's largest single request is
# lc_57 at ~44k, so 65536, 98304 and 262144 are all equally sufficient and
# flagging them as a mismatch is noise that trains a reader to ignore the
# comparability column. What matters is whether a row's window FITS, and that
# is enforced where it can still be acted on: `runner.oversized_tasks` refuses
# the run before it starts.


def _flag_comparability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare each row's run settings against the modal value across rows.

    Records the mismatching field names on each row and returns the modal
    settings for display. With fewer than two rows there is nothing to compare.
    """
    if len(rows) < 2:
        return {}
    modal: dict[str, Any] = {}
    for field in _COMPARABILITY_FIELDS:
        counts: dict[Any, int] = {}
        for row in rows:
            value = row["run_settings"].get(field)
            counts[value] = counts.get(value, 0) + 1
        modal[field] = max(counts, key=lambda v: (counts[v], str(v)))
    for row in rows:
        row["comparability_mismatch"] = [
            field for field in _COMPARABILITY_FIELDS
            if row["run_settings"].get(field) != modal[field]
        ]
    return {"modal": modal, "fields": _COMPARABILITY_FIELDS}


def _module_row(entry: dict[str, Any], det_pass: float, judge_pass: float | None,
                coverage: float | None = None) -> dict[str, Any]:
    """Build one module cell: raw det score, judge delta (if any), raw pass."""
    det = entry["det_score"]
    llm = entry["llm_score"]
    cell: dict[str, Any] = {"det": round(det, 4), "pass": round(det_pass, 4)}
    if coverage is not None and coverage < 1.0:
        cell["judge_coverage"] = round(coverage, 4)
    if llm is not None:
        cell["llm"] = round(llm, 4)
        cell["delta"] = round(llm - det, 4)
        cell["judge_pass"] = round(judge_pass, 4) if judge_pass is not None else None
    return cell


def build_leaderboard(results_dir: Path, weights_name: str = "balanced",
                      scheme: str = "band") -> dict[str, Any]:
    """Load every model's saved results and aggregate into leaderboard rows.

    For each model, the judged file (if present) is loaded instead of the raw
    one — it already carries both the deterministic scores and the judge
    annotations, exactly what ``print_report`` uses for the single-model table.
    """
    weights = MODULE_WEIGHT_PRESETS[weights_name]
    raw_files = find_result_files(results_dir, judged=False)
    registry = load_model_registry()

    rows: list[dict[str, Any]] = []
    modules_present: dict[str, None] = {}
    # The tie test has to read the same files the rows are scored from, or the
    # badge describes a different run than the number beside it.
    sources: list[Path] = []
    loaded: list[tuple[Path, Any]] = []
    for raw_path in raw_files:
        judged_path = _judged_sibling(raw_path)
        source = judged_path if judged_path.exists() else raw_path
        sources.append(source)
    # A judged file holds everything its raw file does, so one without a raw
    # sibling is a complete row, not an orphan. The shipped reference panel is
    # judged files only; skipping them rendered it as an empty board.
    raw_names = {p.name for p in raw_files}
    sources += [p for p in find_result_files(results_dir, judged=True)
                if p.name.replace("_judged.json", ".json") not in raw_names]
    sources.sort()
    for source in sources:
        loaded.append((source, load_results(source)))

    # One exponent for the whole board. pass^k is not comparable across
    # different k — pass^2 > pass^3 always — so ranking a row scored at k=2
    # against rows scored at k=3 hands the short row a free lift over every
    # task it holds. Observed on the v1.0 sweep before `_observed_trials` was
    # fixed: two files recorded k=2 off a single dead request each, and one of
    # them ranked 2nd on a number that was 4th at the fleet's k.
    #
    # The max, not the min. Taking the min would make one short file lower the
    # exponent for twelve sound ones; taking the max costs the short file only
    # the individual tasks that cannot honour k, which `pass_hat_k` drops and
    # the row reports as `tasks_excluded`. A coverage gap belongs to the row
    # that has it. Legacy files are covered too, with no rescore.
    fleet_k = max((b.meta.trials for _, b in loaded if b.meta.trials), default=1)

    for source, bench in loaded:
        results = bench.results
        cov = judge_coverage(results)
        has_llm = cov["judged"] > 0

        modules = aggregate_module_scores(results)
        det_pass = module_success(results, use_det=has_llm)
        judge_pass = module_success(results) if has_llm else {}

        module_cells: dict[str, Any] = {}
        for name in weights:
            entry = modules.get(name)
            if entry is None:
                continue
            modules_present[name] = None
            module_cells[name] = _module_row(
                entry, det_pass.get(name, 0.0), judge_pass.get(name),
                cov["per_module"].get(name, {}).get("coverage") if has_llm else None)

        overall_det = overall_score(modules, weights)
        overall_pass = headline_overall(results, fleet_k,
                                        use_det=has_llm, scheme=scheme,
                                        weights=weights)
        # Cross-model view of the repair loop. Per-model it was already
        # printed (recovery_stats); the spread only becomes readable side by
        # side — one-shot rates ran 1.00 to 0.53 across six models while every
        # code trial counted as a pass.
        rec = recovery_stats(results)
        first_try = (rec["one_shot"] / rec["solved"]) if rec["solved"] else None
        tok_per_s = aggregate_speed(results).get("__global__")
        split = aggregate_timings(results).get("__global__") or {}
        # Tasks holding fewer than k scorable trials are dropped by pass^k, so
        # a model that truncates its way out of a task is scored on a smaller
        # bank than one that answers it wrong. That is the largest single lever
        # in the framework and it was visible only in the terminal.
        by_task = aggregate_by_task(results)
        n_scorable = sum(1 for t in by_task.values() if t["n"] >= fleet_k)
        # Summed trial time, not meta.duration_seconds: a run assembled with
        # --only-new or --add-trials reports only the wall time of its last
        # session, which understated the cost of every stored file by 3-30x.
        trial_seconds = sum(r.duration_seconds or 0.0 for r in results)
        row: dict[str, Any] = {
            "model": bench.meta.model,
            "size": _size_cell(registry.get(bench.meta.model)),
            "profile": bench.meta.profile,
            "n_tasks": n_scorable,
            "tasks_excluded": len(by_task) - n_scorable,
            "trial_minutes": round(trial_seconds / 60.0, 1),
            "has_judge": has_llm,
            "modules": module_cells,
            "overall_det": round(overall_det, 4),
            "overall_pass": round(overall_pass, 4),
            "first_try_rate": (round(first_try, 4)
                               if first_try is not None else None),
            "overall_llm": None,
            "overall_judge_pass": None,
            "tok_per_s": round(tok_per_s, 1) if tok_per_s is not None else None,
            # None whenever the backend reported no prefill/decode split.
            "prefill_tok_s": (round(split["prefill_tok_s"], 1)
                              if split.get("prefill_tok_s") is not None else None),
            "gen_tok_s": (round(split["gen_tok_s"], 1)
                          if split.get("gen_tok_s") is not None else None),
            # Recomputed rather than read from meta so pre-v0.7 files (which
            # have no task_set_hash) still participate in the check.
            "run_settings": {
                "task_set_hash": (bench.meta.task_set_hash
                                  or results_task_set_hash(results)),
                "bench_version": bench.meta.bench_version,
                "endpoint": bench.meta.endpoint,
                "trials": bench.meta.trials,
                "thinking": bench.meta.thinking,
                "max_tokens": bench.meta.max_tokens,
                "profile": bench.meta.profile,
                "sampling": bench.meta.sampling,
                "sandbox_backend": bench.meta.sandbox_backend,
                # 0 means "not discoverable", which is not the same as "the
                # same as everyone else" — two rows at 0 are unverified, not
                # verified equal.
                "served_context": bench.meta.served_context,
            },
        }
        if has_llm:
            overall_llm = overall_score(modules, weights, key="llm_score",
                                        fallback_key="det_score")
            overall_judge_pass = headline_overall(results, fleet_k,
                                                  scheme=scheme, weights=weights)
            row["overall_llm"] = round(overall_llm, 4)
            row["overall_judge_pass"] = round(overall_judge_pass, 4)
            # A row the judge only partly covered still shows its judged
            # columns, but flagged: unjudged modules fall back to their det
            # score, so the number is a blend, not a judged score.
            row["judge_coverage"] = round(cov["coverage"], 4)
            row["judge_complete"] = cov["coverage"] >= 1.0
            row["unjudged_modules"] = cov["unjudged_modules"]
            row["partial_modules"] = cov["partial_modules"]
        # The one number the board ranks and tiers on: the judged pass^k where a
        # judge has seen the run, the raw one where it has not. A model with no
        # judged file still ranks — flagged, via `headline_judged` — rather than
        # dropping out of the charts, which would quietly shrink the board.
        row["headline"] = (row["overall_judge_pass"]
                           if row["overall_judge_pass"] is not None
                           else row["overall_pass"])
        row["headline_judged"] = row["overall_judge_pass"] is not None
        row["tier"] = assign_tier(row["headline"])
        rows.append(row)

    comparability = _flag_comparability(rows)
    # Ranked on the number the board actually displays. This used to sort on
    # `overall_pass` while the page led with the judged score, so a model whose
    # judge run had pulled it down still held the higher row. On the v1.0 sweep
    # that was one pair — ornith-1.5-35b sat above qwen3.6-35b-a3b on a .8275 it
    # no longer had once judged (.8044).
    rows.sort(key=lambda r: r["headline"], reverse=True)
    separable = _mark_ties(rows, sources)
    total_pairs = len(rows) * (len(rows) - 1) // 2
    module_order = [name for name in weights if name in modules_present]
    tier_counts: dict[str, int] = {}
    for row in rows:
        tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1
    tiers = [{"letter": letter, "floor": floor,
              "count": tier_counts.get(letter, 0)}
             for letter, floor in _TIER_BANDS]
    size_buckets = []
    for key, label, floor in _SIZE_BUCKETS:
        members = [r for r in rows if r["size"]["bucket"] == key]
        size_buckets.append({
            "key": key, "label": label, "floor": floor,
            "dense": sum(1 for r in members if not r["size"]["is_sparse"]),
            "sparse": sum(1 for r in members if r["size"]["is_sparse"]),
        })
    return {"weights": weights_name, "scheme": scheme,
           # The weight VALUES, not just the preset name: without them nothing
           # downstream can say why `tools` moves the headline five times as
           # much as `knowledge` does.
           "module_weights": dict(weights),
           "modules": module_order, "rows": rows,
           "tiers": tiers, "size_buckets": size_buckets,
           "comparability": comparability,
           # The single exponent every row's pass^k was computed at. A row
           # whose own `run_settings.trials` is below this was scored on fewer
           # tasks, not on an easier k — see `tasks_excluded`.
           "headline_k": fleet_k,
           # How much of the board's ordering the bank can actually defend.
           # Stated once, as a count, instead of a badge on every tied row.
           "separable_pairs": separable, "total_pairs": total_pairs}


def _mark_ties(rows: list[dict[str, Any]], paths: list[Path]) -> int:
    """Record which rows the bank cannot separate, and return how many pairs it can.

    This used to render a `=` on every row it could not separate from the one
    above. On a 10-model board that fired on 8 of 9 rows and on 11 it fired on
    10 of 10 — marking the whole table, which tells a reader nothing per row
    and reads as an apology for the ranking rather than information.

    The per-row data is still computed and still ships in the JSON, because it
    is real; it just no longer decorates every line. The board states the same
    fact once, as a count of how many of the pairs on it are actually
    separable.

    On tiers, which the board now has and once did not. What was tried and
    rejected here was SEPARABILITY-DERIVED tiers — buckets cut where the sign
    test stopped resolving adjacent rows. Those were wrong for a reason worth
    keeping written down: a "tier 1" drawn that way and spanning a 4B and a 31B
    presents a limitation OF THE BANK as a claim ABOUT the models, and it
    re-letters the whole board whenever a model is added, because the Holm
    correction is over the pair count.

    `_TIER_BANDS` is a different object. It cuts on the headline score at fixed,
    stated thresholds, so an S is a claim about a number the row actually
    scored and nothing more; it makes no assertion that the bank can separate
    an S from an A. Those two facts have to travel together, which is why the
    separable-pair count returned here stays on the board beside the tiers
    rather than being replaced by them. A tier boundary is not evidence of a
    gap; this count is the only thing on the page that speaks to that.
    """
    if len(rows) < 2:
        return 0
    verdicts = pairwise_separability(paths)
    lookup = {}
    for pair in verdicts:
        lookup[(pair["better"], pair["worse"])] = pair
        lookup[(pair["worse"], pair["better"])] = pair
    for above, row in zip(rows, rows[1:]):
        pair = lookup.get((above["model"], row["model"]))
        if pair is None:
            continue
        row["tied_with_above"] = not pair["separable"]
        row["tie_p_adjusted"] = pair["p_adjusted"]
        row["tie_tasks_needed"] = pair["tasks_needed"]
    return sum(1 for p in verdicts if p["separable"])


def render_leaderboard_html(data: dict[str, Any]) -> str:
    """Render the static leaderboard HTML with the data embedded inline."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
    template = env.get_template("leaderboard.html.j2")
    data_json = json.dumps(data).replace("</script>", "<\\/script>")
    return template.render(data_json=data_json)
