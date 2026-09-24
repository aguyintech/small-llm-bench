"""Terminal reporting, JSON persistence, and run comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .judge import JUDGE_SKIP_MODULES
from .models import BenchResult, TaskResult
from .scorer import (BAND_WEIGHTS, MODULE_WEIGHT_PRESETS, TIER_BASELINE_WEIGHT,
                     counted, aggregate_module_scores, band_weighted, overall_score,
                     pass_hat_k, scorable, tier_weighted, weighted_by_module)

_TIER_ORDER = ["easy", "medium", "hard"]
_CAPABILITY_TIERS = ["baseline", "hard"]
_BAND_ORDER = ["anchor", "mid", "hard", "frontier"]

_console = Console()


def save_results(bench: BenchResult, path: Path) -> None:
    """Write a benchmark result to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bench.model_dump(), indent=2))


def load_results(path: Path) -> BenchResult:
    """Load a benchmark result from a JSON file."""
    return BenchResult.model_validate(json.loads(path.read_text()))


def print_report(bench: BenchResult, weights_name: str = "balanced",
                 scheme: str = "band") -> None:
    """Print the per-module scorecard and weighted overall score."""
    weights = MODULE_WEIGHT_PRESETS[weights_name]
    modules = aggregate_module_scores(bench.results)
    speeds = aggregate_speed(bench.results)
    timings = aggregate_timings(bench.results)
    by_task = aggregate_by_task(bench.results)
    has_llm = any(r.llm_score is not None for r in bench.results)
    # With a judge, the Pass column shows the deterministic (pre-judge) pass and
    # the LLM section adds the judge-adjusted pass + delta. Without a judge,
    # det == success, so Pass is just the success rate.
    succ = module_success(bench.results, use_det=has_llm)
    succ_judge = module_success(bench.results)

    table = Table(title=f"{bench.meta.model} — small-llm-bench v{bench.meta.bench_version}")
    table.add_column("Module")
    table.add_column("Tasks", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Det score", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("tok/s (wall)", justify="right")
    table.add_column("pp tok/s", justify="right")
    table.add_column("tg tok/s", justify="right")
    if has_llm:
        table.add_column("LLM judge", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Pass (LLM)", justify="right")
        table.add_column("Pass Δ", justify="right")

    axes = axis_breakdown(bench.results, use_det=has_llm)
    n_tasks_per_module: dict[str, int] = {}
    for t in by_task.values():
        n_tasks_per_module[t["module"]] = n_tasks_per_module.get(t["module"], 0) + 1

    for name, weight in weights.items():
        entry = modules.get(name)
        if entry is None:
            continue
        row = [name, str(n_tasks_per_module.get(name, 0)), f"{weight:.2f}",
               f"{entry['det_score']:.3f}", f"{succ.get(name, 0.0):.3f}",
               _speed_cell(speeds.get(name)),
               _speed_cell((timings.get(name) or {}).get("prefill_tok_s")),
               _speed_cell((timings.get(name) or {}).get("gen_tok_s"))]
        if has_llm:
            row += _llm_cells(entry)
            det_pass = succ.get(name, 0.0)
            judge_pass = succ_judge.get(name, 0.0)
            row += [f"{judge_pass:.3f}", f"{judge_pass - det_pass:+.3f}"]
        table.add_row(*row)
        # Unweighted sub-rows: the module carries the weight, the axes carry the
        # diagnosis (see axis_breakdown).
        for label, axis_entry in sorted(axes.items()):
            if not label.startswith(f"{name}/"):
                continue
            table.add_row(f"└ {label.split('/', 1)[1]}",
                          str(axis_entry["n_tasks"]), "—",
                          f"{axis_entry['det_score']:.3f}",
                          f"{axis_entry['pass']:.3f}", "", "", "",
                          *([""] * 4 if has_llm else []),
                          style="dim")

    table.add_section()
    det_pass_overall = headline_overall(bench.results, bench.meta.trials,
                                        use_det=has_llm, scheme=scheme,
                                        weights=weights)
    overall_row = ["overall", str(len(by_task)), "",
                   f"{overall_score(modules, weights):.3f}",
                   f"{det_pass_overall:.3f}",
                   _speed_cell(speeds.get("__global__")),
                   _speed_cell((timings.get("__global__") or {}).get("prefill_tok_s")),
                   _speed_cell((timings.get("__global__") or {}).get("gen_tok_s"))]
    if has_llm:
        llm_overall = overall_score(modules, weights, key="llm_score",
                                    fallback_key="det_score")
        det_overall = overall_score(modules, weights)
        judge_pass_overall = headline_overall(bench.results, bench.meta.trials,
                                              scheme=scheme, weights=weights)
        overall_row += [f"{llm_overall:.3f}", f"{llm_overall - det_overall:+.3f}",
                        f"{judge_pass_overall:.3f}",
                        f"{judge_pass_overall - det_pass_overall:+.3f}"]
    table.add_row(*overall_row, style="bold")

    _console.print(table)
    if scheme == "legacy":
        _print_capability_tier_table(bench.results, bench.meta.trials)
    else:
        _print_capability_band_table(bench.results, bench.meta.trials)
    _print_difficulty_table(bench.results)
    _print_coverage_lines(bench.results, bench.meta.trials)
    skipped = [r for r in bench.results
               if r.det_breakdown.get("status") == "skipped_no_sandbox"]
    if skipped:
        _console.print(
            f"[yellow]{len(skipped)} code task(s) skipped: no sandbox available "
            f"(pass --allow-unsandboxed to run):[/] "
            + ", ".join(r.task_id for r in skipped))
    _print_recovery_lines(bench.results)
    if has_llm:
        cov = judge_coverage(bench.results)
        if cov["unjudged_modules"] or cov["partial_modules"]:
            _console.print(
                f"[red]judge covered {cov['judged']}/{cov['total']} trials "
                f"({cov['coverage']:.1%}) — judged columns are NOT comparable "
                f"to a fully-judged run:[/]")
            if cov["unjudged_modules"]:
                _console.print("  unjudged modules: "
                               + ", ".join(cov["unjudged_modules"]))
            if cov["partial_modules"]:
                _console.print("  partially judged: "
                               + ", ".join(cov["partial_modules"]))
    errors = [r for r in bench.results if r.error]
    if errors:
        infra = [r for r in errors if r.infra_error]
        others = [r for r in errors if not r.infra_error]
        total = len(bench.results)
        if infra:
            _console.print(
                f"[yellow]{len(infra)} trial(s) excluded from scoring — "
                f"server/network errors outside the model's scope "
                f"({len(infra) / total:.1%} of all trials):[/] "
                + ", ".join(r.task_id for r in infra))
        if others:
            overflow = [r for r in others if r.context_overflow]
            malformed = [r for r in others if r.malformed_tool_call]
            plain = [r for r in others
                     if not r.context_overflow and not r.malformed_tool_call]
            if malformed:
                # Named apart from a wrong answer AND from a server fault: the
                # request reached the model, and the server refused what the
                # model wrote back.
                _console.print(
                    f"[yellow]{len(malformed)} trial(s) emitted tool-call "
                    f"arguments the server could not parse (scored 0):[/] "
                    + ", ".join(r.task_id for r in malformed)
                    + "\n[dim]llama.cpp rejects a raw newline inside a JSON "
                      "string, so multi-line file content has to be escaped. "
                      "Producing well-formed arguments is the capability these "
                      "tasks measure — this is a model failure, not an "
                      "endpoint one.[/]")
            if overflow:
                # Scored 0 like any other failure, but named: the prompt never
                # reached the model, so this says nothing about its ability.
                _console.print(
                    f"[yellow]{len(overflow)} task(s) exceeded context — the "
                    f"endpoint rejected the prompt as longer than its served "
                    f"context window (scored 0):[/] "
                    + ", ".join(r.task_id for r in overflow))
            if plain:
                _console.print(
                    f"[yellow]{len(plain)} task(s) failed with errors "
                    f"({len(plain) / total:.1%} of all tasks):[/] "
                    + ", ".join(r.task_id for r in plain))


def recovery_stats(results: list[TaskResult]) -> dict[str, int]:
    """Count how code tasks were solved: first try vs after seeing an error.

    ``solved`` counts passing trials, ``one_shot`` those that passed on the
    first attempt, ``recovered`` those that needed the execution feedback.
    The pair is the point of the repair loop — a model that reliably arrives
    after one correction is a usable coding assistant; a model that never
    arrives is not, and the single "solved" number hides the difference.
    """
    code = [r for r in results if r.module == "code" and r.det_success]
    one_shot = sum(1 for r in code if r.attempts_used == 1)
    return {"solved": len(code), "one_shot": one_shot,
            "recovered": len(code) - one_shot}


def structured_call_rate(results: list[TaskResult]) -> tuple[int, int]:
    """(used the structured call, total) across tasks where prose was equally
    accepted. Reported, never scored — see ``tool_mechanism`` in the scorer."""
    rows = [r for r in results if "tool_mechanism" in r.det_breakdown]
    return sum(1 for r in rows if r.det_breakdown["tool_mechanism"]), len(rows)


def coverage_report(results: list[TaskResult], trials: int = 1) -> dict:
    """What the score did NOT measure, and what that cost.

    ``counted`` drops infra failures from every aggregate — those measured the
    server, not the model — and an invisible exclusion is how a model that
    never answered ends up looking confidently mid-table. This names every
    dropped trial and every task that lost coverage as a result.

    ``incomplete`` truncations are still counted HERE, and still reported, but
    since v1.0 they are no longer excluded from scoring: a trial that ran out
    of room under a calibrated cap is a model that could not finish (see
    ``scorer.counted``). They stay in this report because a cap that is
    binding on one model and not on twelve others is a fact about the harness
    worth seeing, and because it is the evidence a cap needs raising.

    ``incomplete_failed`` is the subset that actually lost its verdict, and it
    is the number to act on. "Scored as a failure" is the policy, not the
    outcome: the modules outside ``scorer._TRUNCATION_GATED`` still grade a
    truncated response, and `code` grades it by EXECUTING it, so a model that
    writes working code and then rambles to the cap passes while truncating.
    minicpm5-2b did exactly that on cd_31 — twice, det 0.85, both passes —
    while the coverage line called them failures and put them in the count the
    operator is told to go raise a cap over.

    ``tasks_fully_excluded`` is the one that matters most. In the v0.11 audit
    qwen3.5-0.8b had 13 tasks where all three trials truncated, so the model was
    scored only on the subset it happened to finish. With the exclusion gone it
    can now only fire on infra failures.
    """
    infra = [r for r in results if r.infra_error]
    incomplete = [r for r in results
                  if not r.infra_error and r.truncation_class == "incomplete"]
    degenerate = [r for r in results if r.truncation_class == "degenerate"]
    per_module: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = per_module.setdefault(
            r.module, {"trials": 0, "infra": 0, "incomplete": 0,
                       "incomplete_failed": 0, "degenerate": 0})
        bucket["trials"] += 1
        if r.infra_error:
            bucket["infra"] += 1
        elif r.truncation_class == "incomplete":
            bucket["incomplete"] += 1
            if not r.success:
                bucket["incomplete_failed"] += 1
        if r.truncation_class == "degenerate":
            bucket["degenerate"] += 1
    kept: dict[str, int] = {}
    total: dict[str, int] = {}
    for r in results:
        total[r.task_id] = total.get(r.task_id, 0) + 1
        kept[r.task_id] = kept.get(r.task_id, 0) + (1 if counted(r) else 0)
    return {
        "trials": len(results),
        "infra": len(infra),
        "incomplete": len(incomplete),
        "incomplete_failed": sum(1 for r in incomplete if not r.success),
        "degenerate": len(degenerate),
        "per_module": per_module,
        "tasks_fully_excluded": sorted(t for t, n in kept.items() if n == 0),
        "tasks_under_trials": {t: n for t, n in sorted(kept.items())
                               if 0 < n < total[t]},
    }


def cut_turn_evidence(results: list[TaskResult]) -> list[dict]:
    """What the turn that hit the cap actually produced, per failed trial.

    The coverage line used to end by telling the operator to "check whether the
    cap or the model ran out", which was the only honest thing it could say
    while the evidence was unreadable: the agentic-loop modules persisted no
    text for a turn that spent its budget in the reasoning channel. With
    ``TurnRecord.reasoning``/``completion_tokens``/``truncated`` recorded, the
    question is answerable from the file, so the report answers it.

    The shape of the answer is the diagnosis. minicpm5-2b's three `tools`
    incompletes each ended on a turn that spent exactly 4096 tokens producing
    16-18k characters of coherent reasoning, no content and no call — a cap
    binding on a model mid-work. Its two cd_31 trials ended on turns that spent
    12288 producing 41-46k characters of content, having already written code
    that passed: same class, opposite meaning.

    Only trials that lost their verdict are reported; a truncation that still
    passed is not something to go raise a cap over. Empty for files written
    before per-turn flags existed, which is the honest answer for them — the
    cut cannot be located, so nothing is asserted about it.
    """
    rows = []
    for r in results:
        if r.infra_error or r.success or r.truncation_class != "incomplete":
            continue
        cut = [t for t in r.turns if t.truncated]
        if not cut:
            continue
        last = cut[-1]
        rows.append({
            "task_id": r.task_id,
            "module": r.module,
            "tokens": last.completion_tokens,
            "reasoning_chars": len(last.reasoning or ""),
            "content_chars": len(last.content or ""),
            "calls": len(last.tool_calls),
            "cut_turns": len(cut),
        })
    return rows


# Enough rows to show the shape of a cap problem without burying the lines
# under it; qwen3.5-0.8b would otherwise print 13.
_CUT_EVIDENCE_ROWS = 6


def _print_coverage_lines(results: list[TaskResult], trials: int = 1) -> None:
    """Print the coverage gaps, loudly, and only when there are any."""
    cov = coverage_report(results, trials)
    if not (cov["incomplete"] or cov["degenerate"] or cov["infra"]):
        return
    if cov["degenerate"]:
        _console.print(
            f"[red]{cov['degenerate']} trial(s) degenerate[/] — truncated while "
            f"repeating themselves. Scored as failures: the model ran out of "
            f"ideas, not budget.")
    if cov["incomplete"]:
        failed = cov["incomplete_failed"]
        # The failing modules, not every module that truncated: the cap hint
        # below is about trials that lost a verdict, and `code` grades a
        # truncated response by executing it, so it can truncate and pass.
        modules = ", ".join(
            f"{m} {b['incomplete_failed']}/{b['trials']}"
            for m, b in sorted(cov["per_module"].items())
            if b["incomplete_failed"])
        passed = cov["incomplete"] - failed
        note = (f", {passed} still passed (a truncated response can still "
                f"grade correct)" if passed else "")
        _console.print(
            f"[yellow]{cov['incomplete']} trial(s) incomplete[/] — cut off at the "
            f"token cap without looping. {failed} scored as failures{note}"
            + (f": {modules}" if modules else ""))
        for row in (evidence := cut_turn_evidence(results))[:_CUT_EVIDENCE_ROWS]:
            _console.print(
                f"[dim]  {row['module']} {row['task_id']}: cut at "
                f"{row['tokens']} tok with {row['reasoning_chars']:,} ch "
                f"reasoning, {row['content_chars']:,} ch content, "
                f"{row['calls']} call(s)"
                + (f" ({row['cut_turns']} turns cut)"
                   if row["cut_turns"] > 1 else "") + "[/]")
        if len(evidence) > _CUT_EVIDENCE_ROWS:
            _console.print(f"[dim]  … and {len(evidence) - _CUT_EVIDENCE_ROWS} "
                           f"more cut turn(s)[/]")
        if failed:
            _console.print(
                f"[dim]a cut turn with no content and no call did not finish; "
                f"one still deliberating at the cap is the cap binding. If it "
                f"is, raise it in modules/base._MODULE_MAX_TOKENS (or the "
                f"task's own max_tokens) and re-run with --only-new. Do not "
                f"use --max-tokens for this — it overrides EVERY module's cap "
                f"at once and makes the run non-comparable.[/]")
    if cov["tasks_fully_excluded"]:
        _console.print(
            f"[red]{len(cov['tasks_fully_excluded'])} task(s) have NO usable "
            f"trial[/] — this model was not measured on them at all, and its "
            f"score covers only the rest of the bank: "
            + ", ".join(cov["tasks_fully_excluded"]))
    if cov["tasks_under_trials"]:
        _console.print(
            f"[yellow]{len(cov['tasks_under_trials'])} task(s) scored on fewer "
            f"than {trials} trials[/] — pass^k above k drops them silently: "
            + ", ".join(f"{t} ({n})"
                        for t, n in cov["tasks_under_trials"].items()))


def _print_recovery_lines(results: list[TaskResult]) -> None:
    """Print the solved-vs-one-shot and structured-call lines when they apply."""
    rec = recovery_stats(results)
    if rec["solved"]:
        _console.print(
            f"[dim]code: {rec['solved']} solved — {rec['one_shot']} one-shot, "
            f"{rec['recovered']} after execution feedback[/]")
    used, total = structured_call_rate(results)
    if total:
        _console.print(
            f"[dim]clarifying questions: {used}/{total} routed through the "
            f"structured tool call (not scored)[/]")


def print_comparison(benches: list[BenchResult],
                     weights_name: str = "balanced", scheme: str = "band") -> None:
    """Print a side-by-side comparison table across multiple runs."""
    weights = MODULE_WEIGHT_PRESETS[weights_name]
    table = Table(title="small-llm-bench comparison")
    table.add_column("Module")
    for bench in benches:
        table.add_column(bench.meta.model, justify="right")

    aggregates = [aggregate_module_scores(b.results) for b in benches]
    for name in weights:
        row = [name]
        for modules in aggregates:
            entry = modules.get(name)
            row.append(f"{entry['det_score']:.3f}" if entry else "—")
        table.add_row(*row)

    table.add_section()
    overall_cells = [f"{headline_overall(b.results, b.meta.trials, scheme=scheme, weights=weights):.3f}"
                     for b in benches]
    table.add_row("overall", *overall_cells, style="bold")
    _console.print(table)


def aggregate_speed(results: list[TaskResult]) -> dict[str, float | None]:
    """Average generation speed (completion tokens / wall time) per module plus
    a "__global__" entry. Display-only; never feeds the score."""
    totals: dict[str, list[float]] = {}
    for result in results:
        if result.error or result.duration_seconds <= 0 or result.completion_tokens <= 0:
            continue
        for key in (result.module, "__global__"):
            tok, dur = totals.setdefault(key, [0.0, 0.0])
            totals[key] = [tok + result.completion_tokens, dur + result.duration_seconds]
    return {key: (tok / dur if dur > 0 else None) for key, (tok, dur) in totals.items()}


def aggregate_timings(results: list[TaskResult]) -> dict[str, dict[str, float | None]]:
    """Token-weighted prefill and generation speed per module plus a
    "__global__" entry, from the server-reported split.

    Weighted by tokens, not averaged over tasks: one long-context task with a
    20k-token prefill must dominate the prefill rate, not count the same as a
    20-token one. Values are None where the backend reported no split (any
    non-streaming oMLX run, today) so callers can render a dash rather than a
    misleading zero. Display-only; never feeds the score."""
    totals: dict[str, list[float]] = {}
    for result in results:
        if result.error:
            continue
        for key in (result.module, "__global__"):
            acc = totals.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            acc[0] += result.prompt_tokens
            acc[1] += result.prefill_seconds
            acc[2] += result.completion_tokens
            acc[3] += result.generation_seconds
            acc[4] += result.cached_prompt_tokens
    return {
        key: {
            "prefill_tok_s": (ptok / psec) if psec > 0 else None,
            "gen_tok_s": (ctok / gsec) if gsec > 0 else None,
            "prefill_seconds": psec,
            "cached_prompt_tokens": cached,
        }
        for key, (ptok, psec, ctok, gsec, cached) in totals.items()
    }


def _speed_cell(tok_per_s: float | None) -> str:
    """Format a tok/s cell, or a dash when no usage data was returned."""
    return f"{tok_per_s:.1f}" if tok_per_s else "—"


def aggregate_by_task(results: list[TaskResult]) -> dict[str, dict]:
    """Group trials by task_id: passing/total trials, tier, module, soft scores."""
    tasks: dict[str, dict] = {}
    for r in results:
        if not counted(r):
            continue
        entry = tasks.setdefault(r.task_id, {"module": r.module, "tier": r.tier,
                                             "band": r.band,
                                             "passes": 0, "det_passes": 0,
                                             "n": 0, "det": []})
        entry["n"] += 1
        entry["passes"] += 1 if r.success else 0
        entry["det_passes"] += 1 if r.det_success else 0
        entry["det"].append(r.det_score)
    return tasks


def capability_tier_scores(results: list[TaskResult],
                           use_det: bool = False) -> dict[str, dict]:
    """Per capability tier (baseline/hard): task count and (passes,n) per task.

    ``use_det`` counts the deterministic pass (pre-judge) instead of the
    judge-adjusted ``success``."""
    key = "det_passes" if use_det else "passes"
    by_task = aggregate_by_task(results)
    out: dict[str, dict] = {}
    for tier in _CAPABILITY_TIERS:
        counts = [(t[key], t["n"]) for t in by_task.values() if t["tier"] == tier]
        if counts:
            out[tier] = {"count": len(counts), "counts": counts,
                         "pass1": pass_hat_k(counts, 1)}
    return out


def capability_band_scores(results: list[TaskResult],
                           use_det: bool = False) -> dict[str, dict]:
    """Per calibration band (anchor/mid/hard/frontier): task count and
    (passes, n) per task. ``use_det`` mirrors ``capability_tier_scores``."""
    key = "det_passes" if use_det else "passes"
    by_task = aggregate_by_task(results)
    out: dict[str, dict] = {}
    for band in _BAND_ORDER:
        counts = [(t[key], t["n"]) for t in by_task.values() if t["band"] == band]
        if counts:
            out[band] = {"count": len(counts), "counts": counts,
                         "pass1": pass_hat_k(counts, 1)}
    return out


def module_task_scores(results: list[TaskResult],
                       use_det: bool = False) -> dict[str, dict]:
    """Per module: task count and (passes, n) per task, for pass^k."""
    key = "det_passes" if use_det else "passes"
    by_task = aggregate_by_task(results)
    out: dict[str, dict] = {}
    for entry in by_task.values():
        out.setdefault(entry["module"], {"count": 0, "counts": []})
        out[entry["module"]]["count"] += 1
        out[entry["module"]]["counts"].append((entry[key], entry["n"]))
    for stats in out.values():
        stats["pass1"] = pass_hat_k(stats["counts"], 1)
    return out


def headline_overall(results: list[TaskResult], k: int = 1,
                     use_det: bool = False, scheme: str = "module",
                     weights: dict[str, float] | None = None) -> float:
    """Overall headline score at pass^k.

    ``k`` defaults to 1 (ordinary success rate); pass ``k = trials`` so the
    headline counts a task only if it passes every trial — a model that flakes
    2 of 3 scores 0 on that task. ``use_det`` uses the deterministic pass.

    ``scheme`` selects the aggregation. "module" (default, v0.13+) weights the
    modules via ``weights``, falling back to the balanced preset — the score
    then reflects the construct the benchmark claims to measure, and a reader
    can dispute the weights on their merits.

    "band" (v0.4–v0.12) weights the four calibration bands via ``BAND_WEIGHTS``.
    It was the default until v0.13, and it is kept for comparing older files —
    but difficulty-weighting is circular (it weights by the thing being
    measured) and destabilises at low task counts: retagging the v0.12 bank
    honestly left four tasks carrying 0.40 of the score, where one task flip
    moved the headline ten points. Difficulty now drives item *selection*, and
    bands are a reporting axis.

    "legacy" (v0.3) weights the two capability tiers via
    ``TIER_BASELINE_WEIGHT``. Kept for comparing v0.3 result files.
    """
    if scheme == "module":
        modules = module_task_scores(results, use_det=use_det)
        scores = {name: pass_hat_k(entry["counts"], k)
                  for name, entry in modules.items()}
        return weighted_by_module(
            scores, weights or MODULE_WEIGHT_PRESETS["balanced"])
    if scheme == "legacy":
        tiers = capability_tier_scores(results, use_det=use_det)
        b = tiers.get("baseline")
        h = tiers.get("hard")
        bk = pass_hat_k(b["counts"], k) if b else None
        hk = pass_hat_k(h["counts"], k) if h else None
        return tier_weighted(bk, hk)
    bands = capability_band_scores(results, use_det=use_det)
    scores = {band: pass_hat_k(entry["counts"], k) for band, entry in bands.items()}
    return band_weighted(scores)


def judge_coverage(results: list[TaskResult]) -> dict:
    """How much of a run the judge actually scored, overall and per module.

    Coverage is not cosmetic: ``overall_score`` drops a module with no judged
    trials from its weight denominator, so a run whose judge calls failed on
    its WEAK modules scores *higher* on the judged columns than a fully-judged
    one. Callers use this to flag such rows instead of ranking them.

    Modules the judge is deliberately not asked about (``JUDGE_SKIP_MODULES``)
    are left out of the accounting entirely — counting them as gaps would make
    every complete run look incomplete.
    """
    per_module: dict[str, dict] = {}
    for r in results:
        if not counted(r) or r.module in JUDGE_SKIP_MODULES:
            continue
        bucket = per_module.setdefault(r.module, {"judged": 0, "total": 0})
        bucket["total"] += 1
        if r.llm_score is not None:
            bucket["judged"] += 1
    for bucket in per_module.values():
        bucket["coverage"] = bucket["judged"] / bucket["total"]
    judged = sum(b["judged"] for b in per_module.values())
    total = sum(b["total"] for b in per_module.values())
    return {
        "judged": judged,
        "total": total,
        "coverage": judged / total if total else 0.0,
        "per_module": per_module,
        "unjudged_modules": sorted(m for m, b in per_module.items()
                                   if b["judged"] == 0),
        "partial_modules": sorted(m for m, b in per_module.items()
                                  if 0 < b["judged"] < b["total"]),
    }


def module_success(results: list[TaskResult],
                   use_det: bool = False) -> dict[str, float]:
    """Mean strict success rate per module (averaged per task, not per trial).

    ``use_det`` uses the deterministic pass (pre-judge) instead of the
    judge-adjusted ``success``."""
    key = "det_passes" if use_det else "passes"
    by_task = aggregate_by_task(results)
    mods: dict[str, list[float]] = {}
    for t in by_task.values():
        mods.setdefault(t["module"], []).append(t[key] / t["n"])
    return {m: sum(v) / len(v) for m, v in mods.items()}


def axis_breakdown(results: list[TaskResult],
                   use_det: bool = False) -> dict[str, dict[str, Any]]:
    """Per-axis det score, pass rate and task count, keyed ``module/axis``.

    Merging the four tool modules into one traded four weighted rows for one.
    These sub-rows keep the diagnosis: a model that aces single calls and falls
    over on stateful episodes has to stay legible, or the merge cost real
    information.
    """
    key = "det_passes" if use_det else "passes"
    per_task = aggregate_by_task(results)
    axis_of = {r.task_id: r.axis for r in results if r.axis}
    passes: dict[str, list[float]] = {}
    for task_id, entry in per_task.items():
        axis = axis_of.get(task_id)
        if axis:
            passes.setdefault(f"{entry['module']}/{axis}", []).append(
                entry[key] / entry["n"])
    scores: dict[str, list[float]] = {}
    for r in results:
        if r.axis and counted(r):
            scores.setdefault(f"{r.module}/{r.axis}", []).append(r.det_score)
    return {
        label: {"pass": sum(v) / len(v),
                "det_score": (sum(scores[label]) / len(scores[label])
                              if scores.get(label) else 0.0),
                "n_tasks": len(v)}
        for label, v in passes.items()
    }


def aggregate_difficulty_scores(results: list[TaskResult]) -> dict[str, dict]:
    """Average the deterministic score per difficulty tier, one entry per task
    (not per trial) — a task's det score is averaged across its own trials
    first, matching how the module/overall tables count tasks."""
    by_task = aggregate_by_task(results)
    tiers: dict[str, list[float]] = {}
    task_difficulty = {
        r.task_id: r.difficulty for r in results if counted(r)
    }
    for task_id, entry in by_task.items():
        difficulty = task_difficulty.get(task_id, "medium")
        det_mean = sum(entry["det"]) / len(entry["det"]) if entry["det"] else 0.0
        tiers.setdefault(difficulty, []).append(det_mean)
    return {
        tier: {"count": len(scores),
               "det_score": sum(scores) / len(scores) if scores else 0.0}
        for tier, scores in tiers.items()
    }


def _print_capability_tier_table(results: list[TaskResult], trials: int) -> None:
    """Print the baseline/hard tier table with the pass^k reliability curve.

    pass^1 is the success rate; with trials>1, pass^2..pass^N columns expose
    flaky tasks (a task that passes c of n trials decays toward 0 as k grows).
    """
    tiers = capability_tier_scores(results)
    if not tiers:
        return
    ks = list(range(1, trials + 1))
    table = Table(title="Capability tiers (weighted overall = "
                        f"{TIER_BASELINE_WEIGHT:.2f}·baseline + "
                        f"{1 - TIER_BASELINE_WEIGHT:.2f}·hard)")
    table.add_column("Tier")
    table.add_column("Tasks", justify="right")
    for k in ks:
        table.add_column(f"pass^{k}", justify="right")
    for tier in _CAPABILITY_TIERS:
        entry = tiers.get(tier)
        if entry is None:
            continue
        cells = [f"{pass_hat_k(entry['counts'], k):.3f}" for k in ks]
        table.add_row(tier, str(entry["count"]), *cells)
    table.add_section()
    overall_cells = []
    for k in ks:
        b = tiers.get("baseline")
        h = tiers.get("hard")
        bk = pass_hat_k(b["counts"], k) if b else None
        hk = pass_hat_k(h["counts"], k) if h else None
        overall_cells.append(f"{tier_weighted(bk, hk):.3f}")
    table.add_row("overall", str(sum(t["count"] for t in tiers.values())),
                  *overall_cells, style="bold")
    _console.print(table)


def _print_capability_band_table(results: list[TaskResult], trials: int) -> None:
    """Print the anchor/mid/hard/frontier band table with the pass^k curve.

    Same shape as ``_print_capability_tier_table`` but grouped by the
    empirical calibration band instead of the coarse baseline/hard tier.
    """
    bands = capability_band_scores(results)
    if not bands:
        return
    ks = list(range(1, trials + 1))
    # When a band has no tasks (e.g. no frontier tier yet), band_weighted()
    # renormalizes the remaining weights to sum to 1 rather than using the raw
    # BAND_WEIGHTS — show the renormalized values so the header formula
    # actually reproduces the printed overall.
    present = [b for b in _BAND_ORDER if b in bands]
    weight_sum = sum(BAND_WEIGHTS[b] for b in present)
    weight_str = " + ".join(
        f"{BAND_WEIGHTS[b] / weight_sum:.2f}·{b}" for b in present)
    table = Table(title=f"Capability bands (weighted overall = {weight_str})")
    table.add_column("Band")
    table.add_column("Tasks", justify="right")
    for k in ks:
        table.add_column(f"pass^{k}", justify="right")
    for band in _BAND_ORDER:
        entry = bands.get(band)
        if entry is None:
            continue
        cells = [f"{pass_hat_k(entry['counts'], k):.3f}" for k in ks]
        table.add_row(band, str(entry["count"]), *cells)
    table.add_section()
    overall_cells = []
    for k in ks:
        scores = {band: pass_hat_k(entry["counts"], k) for band, entry in bands.items()}
        overall_cells.append(f"{band_weighted(scores):.3f}")
    table.add_row("overall", str(sum(t["count"] for t in bands.values())),
                  *overall_cells, style="bold")
    _console.print(table)


def _print_difficulty_table(results: list[TaskResult]) -> None:
    """Print the per-difficulty-tier scorecard: where the model falls off."""
    tiers = aggregate_difficulty_scores(results)
    if len(tiers) < 2:
        return
    table = Table(title="By difficulty")
    table.add_column("Tier")
    table.add_column("Tasks", justify="right")
    table.add_column("Det score", justify="right")
    ordered = [t for t in _TIER_ORDER if t in tiers]
    ordered += [t for t in tiers if t not in _TIER_ORDER]
    for tier in ordered:
        entry = tiers[tier]
        table.add_row(tier, str(entry["count"]), f"{entry['det_score']:.3f}")
    _console.print(table)


def _llm_cells(entry: dict) -> list[str]:
    """Format the LLM-judge and delta cells for one module row."""
    if entry["llm_score"] is None:
        return ["—", "—"]
    delta = entry["llm_score"] - entry["det_score"]
    omitted = entry.get("omitted", 0)
    suffix = f" ({omitted} omitted)" if omitted else ""
    return [f"{entry['llm_score']:.3f}{suffix}", f"{delta:+.3f}"]
