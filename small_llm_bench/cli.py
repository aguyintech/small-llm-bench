"""Typer CLI: run, score, judge, compare."""

from __future__ import annotations

import asyncio
import re
import webbrowser
from pathlib import Path
from typing import Optional

import json as _json

import typer
from rich.console import Console
from rich.table import Table

from .analysis import (collect_item_stats, collect_model_stats,
                       find_result_files, fleet_monotonicity,
                       pairwise_separability, sample_size_independent)
from .config import BenchSettings, JudgeSettings, ProbeSettings
from .judge import default_judge_output, judge_results
from .leaderboard import (build_leaderboard, load_model_registry,
                          render_leaderboard_html)
from .models import BenchResult
from .reporter import (judge_coverage, load_results, print_comparison,
                       print_report, save_results)
from .migrate import migrate_file
from .probe import JUDGE_DECIDES, run_probe
from .rescore import rescore_bench
from .runner import (DeadEndpointError, UndersizedContextError,
                     checkpoint_path_for, run_bench)
from .scorer import MODULE_WEIGHT_PRESETS

# Aggregations the headline can be computed under. "module" is the v0.13+
# default; the other two exist so older result files stay comparable.
_HEADLINE_SCHEMES = ("module", "band", "legacy")

app = typer.Typer(help="Benchmark small LLMs on tool calling, agentic loops, "
                       "coding, and knowledge.", no_args_is_help=True)
_console = Console()


def apply_reuse_params(settings: BenchSettings, previous: BenchResult | None, *,
                       endpoint: str | None, temperature: float | None,
                       thinking: bool) -> bool:
    """Fill temperature/thinking from ``previous.meta`` when the matching CLI
    flag wasn't passed explicitly (``previous`` is None unless
    ``--reuse-params`` was given).

    The endpoint is never adopted. It is not part of a run's identity (see
    ``runner._config_matches``), and a recorded address is exactly the thing
    that goes stale: adopting it pointed a top-up at a server that had moved.
    It comes from ``--endpoint`` or the environment like any other run.

    Returns True when the previous run's ``max_tokens`` override was adopted,
    which the caller must pass on as ``max_tokens_explicit``.

    ``max_tokens`` used to be excluded here on the reasoning that a raised
    default should never inherit the old, smaller budget. That is right for a
    plain run and wrong for this flag: ``_config_matches`` refuses to pool a
    run recorded with ``max_tokens_override`` into one without it, so
    ``--only-new --reuse-params`` against a file run with an explicit
    ``--max-tokens`` silently reused nothing and swept the whole bank — a
    3-trial top-up turning into 177 trials. Reproducing the recorded config is
    what the flag is FOR, so an override is adopted unless --max-tokens says
    otherwise. A non-overriding previous run still keeps today's budget, so
    the raise-and-retry path the old comment protected is untouched.
    """
    if endpoint:
        settings.endpoint = endpoint
    if temperature is not None:
        settings.temperature = temperature
    elif previous is not None:
        settings.temperature = previous.meta.temperature
    if thinking:
        settings.thinking = True
    elif previous is not None and previous.meta.thinking is not None:
        settings.thinking = previous.meta.thinking
    if (previous is not None and previous.meta.max_tokens_override
            and previous.meta.max_tokens):
        settings.max_tokens = previous.meta.max_tokens
        return True
    return False


@app.command()
def run(
    model: Optional[str] = typer.Option(None, help="Model name to benchmark."),
    endpoint: Optional[str] = typer.Option(None, help="OpenAI-compatible endpoint URL."),
    fast: bool = typer.Option(False, help="Deprecated alias for --profile fast."),
    profile: Optional[str] = typer.Option(None, "--profile",
                                          help="Task subset: fast (17 tasks, quick "
                                               "smoke test) | full (entire bank, "
                                               "39 tasks). Default: full."),
    output: Optional[Path] = typer.Option(None, help="Output results file."),
    concurrency: Optional[int] = typer.Option(None, help="Concurrent requests."),
    save_responses: bool = typer.Option(False, "--save-responses",
                                        help="Save raw API responses in results."),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                 help="Print per-task outcome, score breakdown, "
                                      "and response snippet for failures."),
    task_filter: Optional[str] = typer.Option(None, "--filter", "-k",
                                              help="Only run tasks whose id or "
                                                   "module contains this substring."),
    modules: Optional[str] = typer.Option(None, "--modules",
                                          help="Comma-separated list of exact "
                                               "module names to run, e.g. "
                                               "tools,code. AND-"
                                               "combined with --filter; unlike "
                                               "--filter's substring match, "
                                               "this won't also pull in "
                                               "modules sharing a name prefix."),
    sandbox: Optional[str] = typer.Option(None, "--sandbox",
                                          help="Code sandbox backend: "
                                               "auto|docker|podman|bwrap|"
                                               "sandbox-exec|rlimit."),
    allow_unsandboxed: bool = typer.Option(False, "--allow-unsandboxed",
                                           help="Run code tasks even when only the "
                                                "rlimit fallback (no fs/net "
                                                "isolation) is available."),
    sandbox_memory: Optional[int] = typer.Option(None, "--sandbox-memory",
                                                 help="Sandbox memory cap (MB)."),
    trials: Optional[int] = typer.Option(None, "--trials",
                                         help="Run each task N times for pass^k "
                                              "reliability (default 1)."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens",
                                             help="Max completion tokens per call."),
    temperature: Optional[float] = typer.Option(None, "--temperature",
                                                help="Sampling temperature. "
                                                     "Omit to let the model/server "
                                                     "apply its own default."),
    seed: Optional[int] = typer.Option(None, "--seed",
                                       help="Base sampling seed, offset by "
                                            "trial index so k trials stay k "
                                            "samples. -1 sends no seed. "
                                            "Recorded in the results file."),
    thinking: bool = typer.Option(False, "--thinking",
                                  help="Enable model reasoning via llama.cpp "
                                       "chat_template_kwargs (requires server "
                                       "--jinja). Omit to let the model/server "
                                       "apply its own default."),
    allow_undersized_context: bool = typer.Option(
        False, "--allow-undersized-context",
        help="Run even when the endpoint's served context cannot hold every "
             "task. Those tasks score 0."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume",
        help="Pick up trials recorded by an interrupted run from the "
             "<output>.partial.jsonl sidecar. The sidecar is written as the "
             "run goes and deleted once the results file lands."),
    only_new: bool = typer.Option(False, "--only-new",
                                  help="Reuse already-recorded trials from the "
                                       "output file for tasks whose content and "
                                       "run config (model/temperature/"
                                       "max_tokens/thinking) are unchanged; only "
                                       "run what's missing. Pair with "
                                       "--reuse-params so you don't have to "
                                       "restate the original temperature/"
                                       "thinking by hand."),
    reuse_params: bool = typer.Option(False, "--reuse-params",
                                      help="Adopt temperature/thinking "
                                           "from the existing output file's "
                                           "recorded config for any of those not "
                                           "passed explicitly here, plus a "
                                           "recorded --max-tokens override "
                                           "(without which those trials cannot "
                                           "be reused at all). Works with or "
                                           "without --only-new. A previous run "
                                           "that did NOT override keeps today's "
                                           "budget, so a raised "
                                           "default can retry old "
                                           "truncated trials instead of "
                                           "inheriting the smaller budget that "
                                           "caused the truncation."),
    ignore_task_hash: bool = typer.Option(False, "--ignore-task-hash",
                                          help="Reuse a previous trial by "
                                               "(module, task_id) alone "
                                               "instead of also requiring its "
                                               "content hash to match. Two "
                                               "uses: (1) alone, like "
                                               "--only-new but only re-runs "
                                               "trials that were truncated, "
                                               "infra-errored, or missing — "
                                               "use this instead of --only-new "
                                               "after bumping a module's "
                                               "max_tokens cap, since that "
                                               "changes every task's content "
                                               "hash in that module and "
                                               "--only-new would re-run all of "
                                               "them, not just the truncated "
                                               "ones; (2) combined with "
                                               "--add-trials, so a task's "
                                               "still-good old trials aren't "
                                               "mistaken for gone just because "
                                               "its content hash shifted."),
    ignore_world_hash: bool = typer.Option(False, "--ignore-world-hash",
                                           help="Reuse trials that ran against "
                                                "a different version of the "
                                                "mock tool world (the registry "
                                                "of simulated tools and the "
                                                "data they read). Off by "
                                                "default: a task can be "
                                                "untouched while the world "
                                                "underneath it changes, which "
                                                "silently reused a whole sweep "
                                                "of adv_11 trials from before "
                                                "its table existed. Pass this "
                                                "when you know the registry "
                                                "change cannot affect the "
                                                "tasks in question."),
    add_trials: Optional[int] = typer.Option(None, "--add-trials",
                                             help="Add this many MORE trials "
                                                  "per task on top of what's "
                                                  "already in the output file "
                                                  "(e.g. 3 existing + "
                                                  "--add-trials 2 = 5), for a "
                                                  "statistically firmer read "
                                                  "on a model that looks off. "
                                                  "Requires an existing file "
                                                  "with a matching recorded "
                                                  "config (--reuse-params can "
                                                  "supply it) — errors instead "
                                                  "of silently running fresh "
                                                  "on a mismatch, unlike "
                                                  "--only-new/--ignore-task-hash. "
                                                  "Pair with --ignore-task-hash "
                                                  "if the file has trials from "
                                                  "before a max_tokens bump."),
    tasks_dir: Optional[Path] = typer.Option(None, "--tasks-dir",
                                             help="Load tasks from this "
                                                  "directory instead of the "
                                                  "discovered tasks/ bank. "
                                                  "Only the modules named by "
                                                  "--modules are read, so a "
                                                  "scratch bank needs just "
                                                  "those yaml files. Requires "
                                                  "--output, so a scratch run "
                                                  "can never overwrite a "
                                                  "full-bank results file."),
) -> None:
    """Run the benchmark against an endpoint and save raw results."""
    resolved_profile = profile or ("fast" if fast else "full")
    if resolved_profile not in ("fast", "full"):
        raise typer.BadParameter("profile must be fast or full")
    if tasks_dir is not None:
        if not tasks_dir.is_dir():
            raise typer.BadParameter(f"--tasks-dir not a directory: {tasks_dir}")
        if output is None:
            # Without --output the path defaults to
            # results/<model>_raw_results.json, so a scratch run would silently
            # overwrite that model's full-bank sweep AND poison items/
            # leaderboard, which glob results/*_raw_results*.json.
            raise typer.BadParameter(
                "--tasks-dir requires --output: a scratch bank must not write "
                "to the default results/<model>_raw_results.json path.")
    settings = BenchSettings()
    if model:
        settings.model = model
    # `is not None`, not truthiness: --max-tokens 0 and --concurrency 0 are
    # explicit values, and taking the falsy branch for --max-tokens 0 while
    # `max_tokens_explicit` below reads `max_tokens is not None` set the
    # override flag with the default cap still in place, silently changing the
    # per-module cap regime for the whole run.
    if concurrency is not None:
        settings.concurrency = concurrency
    if max_tokens is not None:
        settings.max_tokens = max_tokens
    if seed is not None:
        settings.seed = seed
    if sandbox:
        settings.sandbox_backend = sandbox
    if allow_unsandboxed:
        settings.allow_unsandboxed = True
    if sandbox_memory:
        settings.sandbox_memory_mb = sandbox_memory

    path = output or settings.output_dir / f"{_slug(settings.model)}_raw_results.json"
    checkpoint = checkpoint_path_for(path)
    previous = (load_results(path)
               if (only_new or ignore_task_hash or reuse_params
                   or add_trials is not None) and path.exists()
               else None)
    if reuse_params and previous is None:
        _console.print("[yellow]--reuse-params: no existing results file at "
                       f"{path}, nothing to reuse — using normal defaults.[/]")

    # --max-tokens on the command line always wins; apply_reuse_params only
    # fills in what was left unset, so it must not clobber an explicit value.
    reused_override = apply_reuse_params(
        settings, previous if (reuse_params and max_tokens is None) else None,
        endpoint=endpoint, temperature=temperature, thinking=thinking)
    if reused_override:
        _console.print("[dim]--reuse-params: adopting the recorded "
                       f"--max-tokens {settings.max_tokens} override, without "
                       "which these trials could not be reused.[/]")

    module_list = ([m.strip() for m in modules.split(",") if m.strip()]
                  if modules else None)
    try:
        bench = asyncio.run(run_bench(settings, profile=resolved_profile,
                                      save_responses=save_responses,
                                      verbose=verbose, task_filter=task_filter,
                                      modules=module_list,
                                      trials=trials, only_new=only_new,
                                      ignore_task_hash=ignore_task_hash,
                                      ignore_world_hash=ignore_world_hash,
                                      add_trials=add_trials,
                                      max_tokens_explicit=(max_tokens is not None
                                                           or reused_override),
                                      previous=previous
                                      if (only_new or ignore_task_hash
                                          or add_trials is not None)
                                      else None,
                                      tasks_dir=tasks_dir,
                                      allow_undersized_context=
                                          allow_undersized_context,
                                      checkpoint_path=checkpoint,
                                      resume=resume))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except UndersizedContextError as exc:
        # The deployment cannot hold the bank. Refusing beats publishing a
        # capability score for tasks the server rejected unread.
        _console.print(f"[red]run refused:[/] {exc}")
        raise typer.Exit(code=2) from exc
    except DeadEndpointError as exc:
        # Not a bad parameter: the config parsed fine, the endpoint is dead.
        # Nothing is saved — a file of empty trials is worse than no file.
        # The checkpoint sidecar is deliberately LEFT in place here: the
        # trials in it are real, and --resume can pick them up once the
        # endpoint is back.
        _console.print(f"[red]run aborted:[/] {exc}")
        raise typer.Exit(code=2) from exc
    save_results(bench, path)
    # Everything the sidecar held is now in the results file.
    checkpoint.unlink(missing_ok=True)
    print_report(bench)
    _console.print(f"\nResults saved to [bold]{path}[/]")


@app.command()
def score(
    results: Path = typer.Option(..., help="Saved results JSON file."),
    weights: str = typer.Option("balanced", help="Weight preset: "
                                + "|".join(MODULE_WEIGHT_PRESETS)),
    scheme: str = typer.Option("module", help="Headline scheme: module "
                               "(v0.13+, module-weighted) | band (v0.4-v0.12, "
                               "anchor/mid/hard/frontier) | legacy (v0.3, "
                               "baseline/hard tier)."),
) -> None:
    """Re-print the scorecard for a saved results file with a weight preset."""
    if weights not in MODULE_WEIGHT_PRESETS:
        raise typer.BadParameter(f"unknown preset: {weights}")
    if scheme not in _HEADLINE_SCHEMES:
        raise typer.BadParameter(
            "scheme must be one of " + ", ".join(_HEADLINE_SCHEMES))
    print_report(load_results(results), weights_name=weights, scheme=scheme)


@app.command()
def migrate(
    results: Optional[Path] = typer.Option(None, help="One saved results file."),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir",
                                               help="Migrate every raw and "
                                                    "judged file in this "
                                                    "directory (top level "
                                                    "only)."),
    tasks_dir: Optional[Path] = typer.Option(None, "--tasks-dir",
                                             help="Task bank to migrate against."),
    in_place: bool = typer.Option(False, "--in-place",
                                  help="Overwrite the input files."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Report what would change; write nothing."),
) -> None:
    """Bring stored result files in line with the current task bank.

    Drops trials whose task no longer exists, carries trials across a module
    rename, and re-stamps `task_count` and `task_set_hash` so a pruned file
    looks like a native run of the current bank. Nothing is re-graded: every
    surviving trial keeps the verdict it already had.

    A trial whose task changed what the MODEL saw keeps its old `task_hash`, so
    it stays visibly un-poolable with future runs of that task.
    """
    if bool(results) == bool(results_dir):
        raise typer.BadParameter("pass either --results or --results-dir")
    if in_place and dry_run:
        raise typer.BadParameter("--dry-run writes nothing; drop --in-place")

    if results_dir:
        files = (find_result_files(results_dir)
                 + find_result_files(results_dir, judged=True))
    else:
        files = [results]
    if not files:
        raise typer.BadParameter("no result files found")

    total_dropped = 0
    for path in sorted(files):
        out_path = path if in_place else path.with_name(
            path.name.replace(".json", "_migrated.json"))
        report, meta = migrate_file(path, out_path=out_path, tasks_dir=tasks_dir,
                                    write=not dry_run)
        total_dropped += len(report.dropped)
        dropped_ids = sorted({d.split(":", 1)[1] for d in report.dropped})
        _console.print(
            f"[bold]{path.name}[/]: kept {report.kept}, dropped "
            f"{len(report.dropped)} trial(s) across {len(dropped_ids)} task(s)"
            + (f" ({', '.join(dropped_ids)})" if dropped_ids else "")
        )
        _console.print(
            f"  relabelled {report.relabelled}, re-hashed {report.rehashed}, "
            f"task_count {meta['task_count']}, hash {meta['task_set_hash']}"
        )
        if report.kept_stale_hash:
            stale = sorted(set(report.kept_stale_hash))
            _console.print(f"  [yellow]kept the old task_hash on "
                           f"{len(stale)} task(s) whose stimulus changed:[/] "
                           f"{', '.join(s.split(':', 1)[1] for s in stale)}")
    verb = "would drop" if dry_run else "dropped"
    _console.print(f"\n{verb} {total_dropped} trial(s) across "
                   f"{len(files)} file(s).")
    if dry_run:
        _console.print("[dim]--dry-run: nothing written[/]")


@app.command()
def rescore(
    results: Path = typer.Option(..., help="Saved results JSON file."),
    output: Optional[Path] = typer.Option(None, help="Output file (default: "
                                                     "<input>_rescored.json)."),
    in_place: bool = typer.Option(False, "--in-place",
                                  help="Overwrite the input file instead of "
                                       "writing a new one."),
    tasks_dir: Optional[Path] = typer.Option(None, "--tasks-dir",
                                             help="Task bank to grade against."),
    allow_task_drift: bool = typer.Option(False, "--allow-task-drift",
                                          help="Re-score trials whose task "
                                               "content changed since the run "
                                               "(NOT comparable — the stored "
                                               "response answers the old task)."),
    sandbox: Optional[str] = typer.Option(None, "--sandbox",
                                          help="Sandbox backend; required to "
                                               "re-score the code module, "
                                               "which re-executes candidates."),
    allow_unsandboxed: bool = typer.Option(False, "--allow-unsandboxed",
                                           help="Permit code re-scoring with "
                                                "no real sandbox."),
    judge_changed: bool = typer.Option(False, "--judge",
                                       help="Re-judge only the trials whose "
                                            "deterministic grade moved."),
    judge_endpoint: Optional[str] = typer.Option(None, help="Judge endpoint URL."),
    judge_model: Optional[str] = typer.Option(None, help="Judge model name."),
) -> None:
    """Re-grade saved trials with the current deterministic scorer.

    Everything the scorer reads is already stored per trial, so a scorer fix can
    be applied to past runs without spending a single model call. Trials whose
    task content changed since the run are skipped by default: the stored
    response answers a question that no longer exists.

    A stored judge score is preserved and the final verdict is re-derived from
    it. With --judge, the trials whose deterministic grade actually moved are
    sent back to the judge model — its previous opinion was anchored on the old
    deterministic score, so those are exactly the stale ones.
    """
    if in_place and output:
        raise typer.BadParameter("pass either --in-place or --output, not both")
    bench = load_results(results)
    out_path = results if in_place else (output or results.with_name(
        results.name.replace("_judged", "").replace(".json", "")
        + "_rescored.json"))
    if out_path.resolve() == results.resolve() and not in_place:
        raise typer.BadParameter("output must differ from the input results "
                                 "file (or pass --in-place)")

    sandbox_cfg = None
    if sandbox or allow_unsandboxed:
        settings = BenchSettings()
        sandbox_cfg = {
            "timeout": settings.code_timeout,
            "memory_mb": settings.sandbox_memory_mb,
            "backend": sandbox or settings.sandbox_backend,
            "allow_unsandboxed": allow_unsandboxed or settings.allow_unsandboxed,
        }

    report = rescore_bench(bench, tasks_dir=tasks_dir, sandbox=sandbox_cfg,
                           allow_task_drift=allow_task_drift)

    if judge_changed and report.moved_keys:
        judge_settings = JudgeSettings()
        if judge_endpoint:
            judge_settings.endpoint = judge_endpoint
        if judge_model:
            judge_settings.model = judge_model
        _console.print(f"[dim]re-judging {len(report.moved_keys)} task(s) whose "
                       f"deterministic grade moved[/]")
        # Clear the stale verdict on exactly those trials first: it was formed
        # against the old deterministic score, and apply_judge_verdict reads
        # `success` as the deterministic input — re-deriving on top of an
        # already-overridden success would freeze the judge's own answer in as
        # det_success.
        for result in bench.results:
            if (result.module, result.task_id) in report.moved_keys:
                result.success = result.det_success
                result.llm_score = None
                result.llm_reasoning = None
                result.judge_anchor_det = None
                result.judge_blocked_by = None
        bench = asyncio.run(judge_results(bench, judge_settings,
                                          only=report.moved_keys))
    elif judge_changed:
        _console.print("[dim]nothing to re-judge: no deterministic grade "
                       "moved[/]")

    save_results(bench, out_path)
    print_report(bench)

    _console.print(f"\nRe-scored {report.rescored} trial(s); "
                   f"{len(report.changed)} verdict(s) changed.")
    for line in report.changed:
        _console.print(f"  {line}")
    if report.regraded_only:
        _console.print(f"[dim]{len(report.regraded_only)} trial(s) re-graded "
                       f"under changed grading rules (prompt and tools "
                       f"unchanged)[/]")
    if report.stale_judge and not judge_changed:
        _console.print(f"[yellow]{len(report.stale_judge)} trial(s) dropped a "
                       f"stale judge score:[/] their deterministic grade moved, "
                       f"and the stored verdict was anchored on the old one. "
                       f"They are unjudged until you re-run with --judge.")
    for label, trials in (("stimulus changed — needs a re-run",
                           report.skipped_drift),
                          ("the run recorded no response", report.skipped_error),
                          ("needs --sandbox", report.skipped_sandbox),
                          ("task no longer in the bank", report.skipped_unknown)):
        if trials:
            _console.print(f"[yellow]skipped ({label}):[/] {len(trials)} — "
                           f"{', '.join(sorted(set(trials))[:6])}"
                           f"{' …' if len(set(trials)) > 6 else ''}")
    _console.print(f"Re-scored results saved to [bold]{out_path}[/]")


@app.command()
def judge(
    results: Path = typer.Option(..., help="Saved results JSON file."),
    judge_endpoint: Optional[str] = typer.Option(None, help="Judge endpoint URL."),
    judge_model: Optional[str] = typer.Option(None, help="Judge model name."),
    output: Optional[Path] = typer.Option(None, help="Output file for judged results."),
    allow_partial: bool = typer.Option(False, "--allow-partial",
                                       help="Exit 0 even if some modules ended "
                                            "unjudged (default: exit 2)."),
) -> None:
    """Run the LLM judge over saved results and write an annotated copy.

    Exits 2 when any module ended fully or partly unjudged — a judged file with
    coverage gaps is not comparable to a complete one, because unjudged modules
    fall back to their deterministic score. The output file is still written.
    """
    settings = JudgeSettings()
    if judge_endpoint:
        settings.endpoint = judge_endpoint
    if judge_model:
        settings.model = judge_model

    out_path = output or default_judge_output(results)
    if out_path.resolve() == results.resolve():
        raise typer.BadParameter("output must differ from the input results file")

    bench = load_results(results)
    if any(r.llm_score is not None for r in bench.results):
        raise typer.BadParameter(
            "input already contains judge scores — pass the raw results "
            "file, not a judged one"
        )
    judged = asyncio.run(judge_results(bench, settings))
    save_results(judged, out_path)
    print_report(judged)
    _console.print(f"\nJudged results saved to [bold]{out_path}[/]")

    cov = judge_coverage(judged.results)
    gaps = cov["unjudged_modules"] + cov["partial_modules"]
    if gaps and not allow_partial:
        _console.print(f"[red]incomplete judge coverage "
                       f"({cov['judged']}/{cov['total']} trials): "
                       f"{', '.join(gaps)} — re-run judge on this raw file "
                       f"before comparing it, or pass --allow-partial[/]")
        raise typer.Exit(code=2)


@app.command()
def compare(
    files: list[Path] = typer.Argument(..., help="Two or more results files."),
    weights: str = typer.Option("balanced", help="Weight preset: "
                                + "|".join(MODULE_WEIGHT_PRESETS)),
    scheme: str = typer.Option("module", help="Headline scheme: module|band|legacy."),
) -> None:
    """Compare multiple saved results files side by side."""
    if len(files) < 2:
        raise typer.BadParameter("provide at least two results files")
    if weights not in MODULE_WEIGHT_PRESETS:
        raise typer.BadParameter(f"unknown preset: {weights}")
    if scheme not in _HEADLINE_SCHEMES:
        raise typer.BadParameter(
            "scheme must be one of " + ", ".join(_HEADLINE_SCHEMES))
    print_comparison([load_results(f) for f in files], weights_name=weights, scheme=scheme)


def _models_in(files: list[Path]) -> list[str]:
    """Model names present in a set of result files."""
    names = []
    for f in files:
        try:
            names.append(load_results(f).meta.model)
        except Exception:      # a corrupt file is the loader's problem, not ours
            continue
    return names


# Class names are long and the table is width-bound; `soft_spread` moved to
# --json to make room for the band reading, which is the column the panel-wide
# `disc` structurally cannot provide. Full class names stay in the JSON output.
_CLASS_ABBR = {
    "discriminating": "disc", "floor_only": "floor", "dead_easy": "dead-e",
    "dead_hard": "dead-h", "restraint": "restr", "anchor": "anchor",
    "flaky": "flaky", "weak": "weak", "inverted_leg": "inv-lg",
}


def _registry_fleet(files: list[Path]) -> list[str]:
    """Models present in `files`, ordered small-to-large by models.yaml.

    Serving variants (`variant_of`) are dropped: they are the same weights, so
    a difference between them is a configuration finding, not a size one, and
    including them makes every variant pair look like an inversion.
    """
    registry = load_model_registry()
    present = []
    for path in files:
        try:
            name = load_results(path).meta.model
        except Exception:                    # a corrupt file is not our problem
            continue
        entry = registry.get(name)
        if entry is None or entry.get("variant_of"):
            continue
        size = entry.get("active_b", entry.get("params_b"))
        if size is not None:
            present.append((size, name))
    return [name for _, name in sorted(present)]


@app.command()
def items(
    results_dir: Path = typer.Option(Path("results"), "--results-dir",
                                     help="Directory of saved results files."),
    judged: bool = typer.Option(False, "--judged",
                                help="Use judged copies instead of raw results."),
    module: Optional[str] = typer.Option(None, "--module",
                                         help="Only show tasks from this module."),
    min_models: int = typer.Option(1, "--min-models",
                                   help="Warn if a task is seen in fewer models."),
    json_output: Optional[Path] = typer.Option(None, "--json",
                                               help="Also write full stats as JSON."),
    cohort: Optional[str] = typer.Option(None, "--cohort",
                                         help="Comma-separated models in "
                                              "weak-to-strong order for "
                                              "band_discrimination. Defaults to "
                                              "PROBE_MODELS."),
) -> None:
    """Per-task item analysis (pass rate, discrimination, flakiness) across
    every saved result file — identifies saturated and discriminating tasks."""
    files = find_result_files(results_dir, judged=judged)
    if not files:
        raise typer.BadParameter(f"no result files found in {results_dir}")
    band = ([m.strip() for m in cohort.split(",") if m.strip()] if cohort
            else ProbeSettings().model_list())
    rows = collect_item_stats(files, cohort=band)
    if module:
        rows = [r for r in rows if r["module"] == module]

    table = Table(title=f"Item analysis ({len(files)} model(s), "
                        f"{'judged' if judged else 'raw'})")
    # Headers are abbreviated because the band column pushed an already-wide
    # table past 80 columns, and rich answers that by truncating every header to
    # an ellipsis — which loses the reader more than short names do.
    table.add_column("task", no_wrap=True)
    table.add_column("module", max_width=5)
    # class and band carry the whole point of this table — the band reading is
    # what the panel-wide disc column structurally cannot show — so they get
    # fixed widths rather than being the first thing rich abbreviates away.
    table.add_column("class", no_wrap=True, min_width=6)
    table.add_column("pass", justify="right")
    table.add_column("disc", justify="right")
    table.add_column("band", justify="right", no_wrap=True, min_width=5)
    # The band column is strong-minus-weak, so an inverted middle rung is
    # invisible in it — pf_01 reads +0.33 there while its 12B->27B leg runs
    # backwards. `legs` is the same cohort rung by rung.
    table.add_column("legs", justify="right", no_wrap=True, min_width=9)
    table.add_column("flky", justify="right")
    table.add_column("n", justify="right")
    table.add_column("sec", justify="right")
    table.add_column("tok", justify="right")
    for r in rows:
        style = "dim" if r["n_models"] < min_models else None
        table.add_row(
            r["task_id"], r["module"], _CLASS_ABBR.get(r.get("class", ""),
                                                        r.get("class", "—")),
            f"{r['pass_rate']:.2f}",
            f"{r['discrimination']:+.2f}",
            ("—" if r.get("band_discrimination") is None
             else f"{r['band_discrimination']:+.2f}"),
            # Leading zero dropped and slash-joined: this table was already at
            # the width where rich truncates headers to an ellipsis, and losing
            # `pass`/`disc` to buy `legs` would be a bad trade.
            ("/".join("—" if x is None else f"{x:+.2f}".replace("0.", ".")
                      for x in r.get("band_legs") or []) or "—"),
            str(r["flaky_models"]), str(r["n_models"]),
            f"{r['mean_duration']:.1f}" if r["mean_duration"] is not None else "—",
            f"{r['mean_completion_tokens']:.0f}" if r["mean_completion_tokens"] is not None else "—",
            style=style,
        )
    _console.print(table)

    # Name the cohort. `disc` and `band` answer different questions and a reader
    # who assumes they answer the same one will read `disc` as a band verdict —
    # which is how the v0.15 cut kept bottom-band items believing it was
    # preserving ranking power.
    present = [m for m in band if any(m == r_model for r_model in _models_in(files))]
    _console.print(
        f"[dim]band = strong minus weak over the declared cohort "
        f"{' -> '.join(band)} (weak to strong, never re-sorted). "
        f"disc = top third minus bottom third of all {len(files)} model(s), "
        f"which on this panel is dominated by the sub-3B tail.[/]")
    missing = [m for m in band if m not in present]
    if missing:
        _console.print(f"[yellow]cohort model(s) absent from {results_dir}: "
                       + ", ".join(missing) + " — band readings are partial.[/]")

    floor_only = [r["task_id"] for r in rows if r.get("class") == "floor_only"]
    if floor_only:
        _console.print(
            f"[yellow]{len(floor_only)}/{len(rows)} task(s) carry NO band "
            f"signal — the whole cohort passes them, however well they separate "
            f"below it. Not dead weight (they are why the lower board is "
            f"separable), but they cannot move the band:[/] "
            + ", ".join(floor_only))

    inverted = [r for r in rows if r.get("band_monotonic") is False]
    if inverted:
        # The gap a task reports is not always the gap it measures. Spelled out
        # per rung because the summary number cannot show which leg moved.
        _console.print(
            f"\n[bold yellow]{len(inverted)}/{len(rows)} task(s) rank the "
            f"declared cohort BACKWARDS on at least one rung[/] — the band "
            f"column averages this away:")
        for r in inverted:
            legs = " -> ".join("?" if f is None else f"{f:.2f}"
                               for f in r.get("band_fracs") or [])
            _console.print(f"  {r['task_id']:<9} {legs}   "
                           f"band {r['band_discrimination']:+.2f}")

    dead = [r["task_id"] for r in rows
            if r.get("class") in ("dead_easy", "dead_hard")]
    if dead:
        _console.print(f"[yellow]{len(dead)}/{len(rows)} task(s) saturated "
                       f"across all {len(files)} model(s) — no ranking signal:[/] "
                       + ", ".join(dead))

    anchors = [r for r in rows if r.get("class") == "anchor"]
    if anchors:
        # Saturation is the point of an anchor, so it is reported as a health check
        # rather than as dead weight. See analysis.classify_task.
        _console.print(
            f"[dim]{len(anchors)} anchor task(s) passed by every model, as "
            f"intended — floor sentinels, 0.10 of the band-weighted headline: "
            + ", ".join(r["task_id"] for r in anchors) + "[/]")

    restraint = [r for r in rows if r.get("class") == "restraint"]
    if restraint:
        # Reported apart from the ranking, and read the opposite way: these
        # grade a restraint, so a low pass rate among strong models is a
        # finding about the fleet rather than a defect in the task.
        _console.print(
            f"\n[bold]{len(restraint)} restraint task(s)[/] — graded on what the "
            f"model declines to do, so the discrimination column above does not "
            f"apply to them (see analysis._RESTRAINT_TASKS):")
        for r in sorted(restraint, key=lambda x: x["pass_rate"]):
            failed = round((1 - r["pass_rate"]) * r["n_models"])
            _console.print(
                f"  [bold]{r['task_id']}[/] ({r['module']}) pass={r['pass_rate']:.2f}"
                f" — {failed}/{r['n_models']} model(s) fail it")

    model_stats = collect_model_stats(files)
    ci_table = Table(title="Per-model headline with 95% CI (unit = task)")
    ci_table.add_column("model")
    ci_table.add_column("tasks", justify="right")
    # Tasks with fewer than k scorable trials are dropped from every number in
    # this table, so the count is printed rather than absorbed: a model scored
    # on 37 tasks is not comparable to one scored on 39, and the usual cause
    # is truncation, which favours the model that ran out of budget.
    ci_table.add_column("excl", justify="right")
    ci_table.add_column("det pass", justify="right")
    ci_table.add_column("pass", justify="right")
    # `pass` is module-weighted; `task rate` is the unweighted share of tasks
    # passed. The CI brackets the SECOND — Wilson needs an unweighted count —
    # so both are shown rather than pairing an interval with a number it does
    # not describe. Printed as an interval, not +/-: Wilson is asymmetric near
    # 1.0, which is where most of this fleet sits.
    ci_table.add_column("task rate", justify="right")
    ci_table.add_column("95% CI", justify="right")
    # Mean soft score beside the pass rate: the binary gate scores a 0.96
    # near-miss and a 0.30 collapse identically, which hides WHY a task failed.
    # It does not recover mid-vs-large signal (measured: -0.0066 soft vs
    # -0.0090 binary on the stored run) — that is a difficulty problem.
    ci_table.add_column("soft", justify="right")
    for s in model_stats:
        excluded = s.get("tasks_excluded", 0)
        ci_table.add_row(s["model"], str(s["n_tasks"]),
                         f"[yellow]{excluded}[/]" if excluded else "0",
                         f"{s['det_headline']:.3f}", f"{s['headline']:.3f}",
                         f"{s['pass_rate']:.3f}",
                         f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]",
                         "—" if s.get("mean_soft") is None
                         else f"{s['mean_soft']:.3f}")
    _console.print(ci_table)
    short = [s for s in model_stats if s.get("tasks_excluded")]
    if short:
        _console.print(
            "[yellow]excl[/] = tasks with fewer than k scorable trials, dropped "
            "from every column above: "
            + ", ".join(f"{s['model']} ({s['tasks_excluded']})" for s in short))

    # Declared order, in preference order: PROBE_FLEET_ORDER if the user
    # maintains one, else models.yaml sorted by ACTIVE parameters. The
    # registry is the better source — it cannot go stale against the fleet the
    # way a hand-kept env var does — but an explicit order still wins, because
    # a user comparing two models of equal size may have a reason to declare
    # which they expect to be stronger.
    fleet = ProbeSettings().fleet_list() or _registry_fleet(files)
    if fleet:
        # At k=3 a single flaky trial moves a module by 0.33/n_tasks, which on
        # a ~5-task module is ~0.07. Filtering below 0.10 keeps the report to
        # inversions that survive one bad trial; unfiltered, all-pairs across
        # 12 models buries the real ones (multi_turn_if runs backwards by 0.67)
        # under dozens of noise-sized ones.
        inversions = fleet_monotonicity(files, fleet, min_delta=0.10)
        if inversions:
            _console.print(
                f"\n[bold yellow]{len(inversions)} module/pair inversion(s)[/] — "
                f"a model declared LARGER scoring lower (declared order: "
                f"{' -> '.join(fleet)}):")
            for row in inversions[:12]:
                _console.print(
                    f"  {row['module']:<14} {row['larger']} "
                    f"{row['larger_score']:.3f} < {row['smaller']} "
                    f"{row['smaller_score']:.3f}  ({row['delta']:+.3f})")
            if len(inversions) > 12:
                _console.print(f"  [dim]... and {len(inversions) - 12} more[/]")
            _console.print(
                "[dim]An inversion is not automatically a bank defect, but the "
                "headline is not measuring size on that module. At k=3 one "
                "flaky trial moves a module by 0.33/n_tasks — read small "
                "deltas accordingly.[/]")
        else:
            _console.print("[dim]No module ranks the declared fleet order "
                           "backwards.[/]")
    else:
        _console.print("[dim]PROBE_FLEET_ORDER unset — skipping the "
                       "non-monotonicity check (discovery-run-brief §3.2).[/]")
    if model_stats:
        n = model_stats[0]["n_tasks"]
        p = sum(s["headline"] for s in model_stats) / len(model_stats)
        _console.print(
            f"[dim]at n={n} tasks and p≈{p:.2f}, separating two models needs "
            f"~{sample_size_independent(p, 0.10)} tasks for a 10pt gap and "
            f"~{sample_size_independent(p, 0.15)} for 15pt (independent banks); "
            f"gaps inside the CI above are ties, not ranks.[/]")

    pairs = pairwise_separability(files)
    if pairs:
        pair_table = Table(title="Pairwise separability (paired sign test, "
                                 "Holm-corrected across all pairs)")
        pair_table.add_column("higher on the board", max_width=24)
        pair_table.add_column("lower", max_width=24)
        pair_table.add_column("W-L", justify="right", no_wrap=True)
        pair_table.add_column("p adj", justify="right", no_wrap=True)
        pair_table.add_column("verdict", no_wrap=True)
        pair_table.add_column("need", justify="right", no_wrap=True)
        # Adjacent rows first: those are the gaps a reader of the sorted board
        # is most likely to mistake for a ranking.
        for pair in sorted(pairs, key=lambda x: (not x["adjacent"],
                                                 x["p_adjusted"])):
            needed = pair["tasks_needed"]
            pair_table.add_row(
                pair["better"] + (" *" if pair["adjacent"] else ""),
                pair["worse"],
                f"{pair['wins']}-{pair['losses']}",
                f"{pair['p_adjusted']:.3f}",
                "[green]separable[/]" if pair["separable"] else "[yellow]tie[/]",
                "—" if pair["separable"] or not needed
                else (">10k" if needed > 10000 else str(needed)),
                style=None if pair["separable"] else "dim")
        _console.print(pair_table)
        separable = [x for x in pairs if x["separable"]]
        _console.print(
            f"[dim]* adjacent on the sorted board. {len(separable)}/{len(pairs)} "
            f"pair(s) separable; every other gap is a tie this bank cannot "
            f"resolve, however the rows are ordered. 'need' is the paired "
            f"(McNemar) estimate for the observed gap. Rows here are ordered "
            f"by the UNWEIGHTED per-task pass rate, which is the unit the "
            f"sign test runs on — not the module-weighted headline the "
            f"leaderboard sorts by, so the two orders can differ.[/]")

    low_coverage = [r["task_id"] for r in rows if r["n_models"] < min_models]
    if low_coverage:
        _console.print(f"[yellow]{len(low_coverage)} task(s) seen in fewer than "
                       f"{min_models} model(s):[/] " + ", ".join(low_coverage))

    if json_output:
        json_output.write_text(_json.dumps(rows, indent=2))
        _console.print(f"\nFull stats written to [bold]{json_output}[/]")


@app.command()
def leaderboard(
    results_dir: Path = typer.Option(Path("results"), "--results-dir",
                                     help="Directory of saved results files."),
    output: Path = typer.Option(Path("results/leaderboard.html"), "--output",
                                help="HTML file to write (a sibling .json is "
                                     "also written with the same data)."),
    weights: str = typer.Option("balanced", help="Weight preset: "
                                + "|".join(MODULE_WEIGHT_PRESETS)),
    scheme: str = typer.Option("module", help="Headline scheme: module|band|legacy."),
    open_browser: bool = typer.Option(False, "--open",
                                      help="Open the leaderboard in your "
                                           "default browser after writing it."),
) -> None:
    """Build a static, sortable HTML leaderboard across every saved model
    result — one row per model, one column per module. Judged copies are
    preferred over raw ones when both exist for a model. Re-running overwrites
    the same file with fresh data; the template itself never changes."""
    if weights not in MODULE_WEIGHT_PRESETS:
        raise typer.BadParameter(f"unknown preset: {weights}")
    if scheme not in _HEADLINE_SCHEMES:
        raise typer.BadParameter(
            "scheme must be one of " + ", ".join(_HEADLINE_SCHEMES))
    data = build_leaderboard(results_dir, weights_name=weights, scheme=scheme)
    if not data["rows"]:
        raise typer.BadParameter(f"no result files found in {results_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_leaderboard_html(data))
    json_path = output.with_suffix(".json")
    json_path.write_text(_json.dumps(data, indent=2))

    _console.print(f"Leaderboard written to [bold]{output}[/] "
                  f"({len(data['rows'])} model(s))")
    _console.print(f"Data written to [bold]{json_path}[/]")
    if open_browser:
        webbrowser.open(output.resolve().as_uri())


@app.command()
def probe(
    module: str = typer.Option(..., "--module",
                               help="Module the candidate belongs to; its yaml "
                                    "is read from the scratch bank."),
    task: str = typer.Option(..., "--task", help="Candidate task id."),
    reprobe: bool = typer.Option(False, "--reprobe",
                                 help="Top an existing 3-trial probe up to 5 "
                                      "without re-running the first three, to "
                                      "break a one-trial-wide tie."),
    override: bool = typer.Option(False, "--override",
                                  help="Run past the per-construct cycle cap. "
                                       "Stamped permanently in the probe log."),
    top_up: bool = typer.Option(False, "--top-up",
                                help="Reuse the last cycle's trials and re-run "
                                     "only the models that came back short of "
                                     "--trials, e.g. after a trial truncated "
                                     "incomplete and was dropped as "
                                     "unscorable."),
) -> None:
    """Screen one candidate task against the declared model trio.

    Author the candidate in .scratch/probe/tasks/<module>.yaml, then probe it.
    A verdict in minutes replaces a full 61-task × 10-model sweep that, in
    v0.13, told us 6 of 16 new tasks separated nobody.

    This is a high-recall SCREEN, not a test: at 3-vs-3 trials only a perfect
    1.0/0.0 split is nominally significant. The fleet sweep is the evidence.
    """
    settings = ProbeSettings()
    if module in JUDGE_DECIDES and not JudgeSettings().api_key:
        _console.print("[yellow]No JUDGE_API_KEY: probing "
                       f"'{module}' on deterministic scores only. The fleet "
                       "sweep judges it, so a candidate can flip.[/]")
    try:
        verdict, meta = asyncio.run(run_probe(module, task, probe=settings,
                                              reprobe=reprobe,
                                              override=override,
                                              top_up=top_up))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"probe {task} ({module}) — cycle {meta['cycle']}")
    table.add_column("slot"); table.add_column("model")
    table.add_column("passes", justify="right")
    if meta["judged"]:
        table.add_column("det", justify="right")
    for slot, model in zip(("weak", "mid", "strong"), meta["models"]):
        passes, n = meta["counts"][model]
        row = [slot, model, f"{passes}/{n}"]
        if meta["judged"]:
            det_passes, det_n = meta["det_counts"][model]
            row.append(f"{det_passes}/{det_n}")
        table.add_row(*row)
    _console.print(table)

    colour = {"ACCEPT": "green", "REPROBE": "yellow"}.get(verdict.verdict, "red")
    gap = "—" if verdict.gap is None else f"{verdict.gap:+.2f}"
    _console.print(f"[bold {colour}]{verdict.verdict}[/] — {verdict.reason}"
                  + (f": {verdict.detail}" if verdict.detail else ""))
    _console.print(f"gap {gap}   splits: "
                   + (", ".join(verdict.splits) or "none")
                   + f"   {meta['elapsed_s']}s")
    if "m|s" in verdict.splits:
        _console.print("[dim]splits mid|strong — the coverage the bank most "
                       "lacks[/]")
    if meta.get("judge_diverged"):
        # The judge only decides in tools/multi_turn_if, and there it can
        # rescue a trial the deterministic grader failed — which on a
        # structural task means overruling the only grader that can see the
        # damage. Surface it; do not pick a side silently.
        _console.print(f"[bold magenta]judge divergence[/]: without the judge "
                       f"this is {meta['det_verdict']} — {meta['det_reason']}. "
                       "Check whether the judge is rescuing trials that broke "
                       "structure the deterministic grader caught.")
    if verdict.accepted:
        _console.print("[dim]Screen only: at 3-vs-3 trials just a 1.00/0.00 "
                       "split reaches p=0.05. Confirm on the full fleet.[/]")
    remaining = settings.max_cycles - meta["cycle"]
    if verdict.verdict != "INVALID" and remaining <= 1 and not verdict.accepted:
        _console.print(f"[yellow]{max(remaining, 0)} cycle(s) left for "
                       f"{task}.[/]")


def _slug(name: str) -> str:
    """Make a filesystem-safe slug from a model name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


if __name__ == "__main__":
    app()
