"""Candidate-task probe loop: screen one task against a declared model trio.

Task authoring used to be open-loop — write N tasks, sweep 61 tasks × 10 models
× 3 trials, then find out nothing discriminated. v0.13 is the proof: of 16 new
tasks, 6 separated nobody and none separated the 4B from the 12B.

This closes the loop. Author one candidate in a scratch bank, run it against
three models × three trials, get a verdict in minutes, iterate or abandon. Only
survivors are promoted into ``tasks/`` and swept across the fleet.

The probe is a HIGH-RECALL SCREEN, not a test. At 3-vs-3 trials only a perfect
1.0/0.0 split reaches nominal p=0.05 on a one-sided Fisher test; every other
accept is a prior update. The 10-model sweep remains the evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .analysis import ProbeVerdict, probe_verdict
from .config import BenchSettings, JudgeSettings, ProbeSettings
from .judge import (JUDGE_DISPLAY_ONLY_MODULES, JUDGE_SKIP_MODULES,
                    judge_results)
from .models import BenchResult
from .reporter import aggregate_by_task, load_results, save_results
from .runner import run_bench

# Modules where a judge verdict can actually move ``success``. Everything else
# is either never judged (JUDGE_SKIP_MODULES) or judged for display only
# (JUDGE_DISPLAY_ONLY_MODULES), so skipping the judge there costs nothing and
# saves a remote round trip per cycle — most of the loop's speed.
JUDGE_DECIDES = frozenset(
    {"tools", "multi_turn_if"}) - JUDGE_SKIP_MODULES - JUDGE_DISPLAY_ONLY_MODULES


def slug(name: str) -> str:
    """Filesystem-safe model name (mirrors cli._slug)."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def probe_paths(probe: ProbeSettings, task_id: str) -> tuple[Path, Path, Path]:
    """Return (tasks_dir, run_dir, log_path) for a probe of ``task_id``."""
    root = probe.dir
    return root / "tasks", root / "runs" / task_id, root / "log.jsonl"


def assert_not_results_dir(path: Path, bench: BenchSettings) -> None:
    """Refuse any probe output path that could land in the real results dir.

    A stray ``*_raw_results*.json`` under ``results/`` does not just overwrite a
    sweep — ``analysis.find_result_files`` globs that directory, so it would
    also enter ``items`` and ``leaderboard`` as a one-task "model".
    """
    resolved = path.resolve()
    guarded = bench.output_dir.resolve()
    if resolved == guarded or guarded in resolved.parents:
        raise ValueError(
            f"probe output {resolved} is inside the results dir {guarded}; "
            "refusing to write where a full-bank sweep lives")


def cycles_for(log_path: Path, task_id: str) -> int:
    """Count scoring cycles already logged for a task. INVALID does not count.

    Harness problems — infra errors, a missing sandbox backend, malformed yaml
    — say nothing about the task, so they must not spend the budget.
    """
    if not log_path.exists():
        return 0
    n = 0
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("task_id") == task_id and entry.get("verdict") != "INVALID":
            n += 1
    return n


def candidate_fingerprint(tasks_dir: Path, module: str) -> str:
    """Hash the scratch yaml so the log shows when a candidate was edited."""
    path = tasks_dir / f"{module}.yaml"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def read_expectation(tasks_dir: Path, module: str, task_id: str) -> dict | None:
    """Read the candidate's pre-registered ``probe_expect`` block, if any.

    Declaring the expected ordering BEFORE the first cycle is the only bias
    control here that produces evidence rather than a vibe: a prediction
    rewritten between cycles shows up as a changed fingerprint next to an
    unchanged claim.
    """
    import yaml
    path = tasks_dir / f"{module}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    for raw in data.get("tasks", []):
        if raw.get("id") == task_id:
            return raw.get("probe_expect")
    return None


async def _run_one_model(model: str, *, module: str, task_id: str,
                         tasks_dir: Path, out_path: Path,
                         probe: ProbeSettings, reprobe: bool,
                         add: int | None = None) -> BenchResult:
    """Run the candidate against one model.

    ``add`` tops the existing file up by that many trials instead of running a
    fresh set — used to replace trials that came back unscorable. Re-running
    all three models to recover one dropped trial wastes two models' time for
    nothing.
    """
    settings = BenchSettings()
    settings.model = model
    # The dead-endpoint guard reads "the first N trials all came back empty" as
    # a broken endpoint, which is right for a 45-task sweep and wrong here: a
    # probe is one task at three trials, and the weak model returning nothing on
    # all three is the verdict we are screening for, not an infra failure.
    settings.sanity_check_after = 0
    assert_not_results_dir(out_path, settings)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topping_up = reprobe or add is not None
    previous = load_results(out_path) if topping_up and out_path.exists() else None
    if reprobe and previous is None:
        raise ValueError(f"--reprobe needs an existing probe run at {out_path}")
    if add is not None and previous is None:
        add = None  # nothing to top up; fall back to a normal run
    if previous is not None:
        # Inherit endpoint/temperature/thinking so add_trials' config check
        # passes, the same way --reuse-params does for `run`.
        settings.endpoint = previous.meta.endpoint
        settings.temperature = previous.meta.temperature
        if previous.meta.thinking is not None:
            settings.thinking = previous.meta.thinking

    bench = await run_bench(
        settings,
        profile="full",          # a candidate without `fast:` would vanish
        task_filter=task_id,
        modules=[module],
        trials=None if topping_up else probe.trials,
        add_trials=(2 if reprobe else add),
        previous=previous,
        tasks_dir=tasks_dir,
    )
    save_results(bench, out_path)
    return bench


async def run_probe(module: str, task_id: str, *,
                    probe: ProbeSettings | None = None,
                    judge: JudgeSettings | None = None,
                    reprobe: bool = False,
                    override: bool = False,
                    top_up: bool = False) -> tuple[ProbeVerdict, dict]:
    """Probe one candidate against the declared trio and return (verdict, meta).

    Models run sequentially: they now share one endpoint, and two concurrent
    ``run_bench`` calls would fight over the rich Live display and stdin.
    """
    probe = probe or ProbeSettings()
    models = probe.model_list()
    if len(models) != 3:
        raise ValueError(f"PROBE_MODELS must name exactly 3 models, got {models}")

    tasks_dir, run_dir, log_path = probe_paths(probe, task_id)
    if not (tasks_dir / f"{module}.yaml").exists():
        raise ValueError(f"no scratch bank at {tasks_dir / f'{module}.yaml'}")

    spent = cycles_for(log_path, task_id)
    if spent >= probe.max_cycles and not override:
        raise ValueError(
            f"{task_id} has already used {spent}/{probe.max_cycles} scoring "
            "cycles. Past that you are fitting the trio, not finding a "
            "construct — abandon it, or pass --override (logged permanently).")

    started = time.monotonic()
    counts: dict[str, tuple[int, int]] = {}
    det_counts: dict[str, tuple[int, int]] = {}
    infra_error = False
    async def measure(model: str, add: int | None = None) -> None:
        nonlocal infra_error
        out_path = run_dir / f"{slug(model)}_raw_results.json"
        bench = await _run_one_model(model, module=module, task_id=task_id,
                                     tasks_dir=tasks_dir, out_path=out_path,
                                     probe=probe, reprobe=reprobe, add=add)
        if module in JUDGE_DECIDES:
            bench = await judge_results(bench, judge or JudgeSettings(),
                                        only={(module, task_id)})
            save_results(bench, out_path)
        infra_error = infra_error or any(r.infra_error for r in bench.results)
        entry = aggregate_by_task(bench.results).get(task_id)
        counts[model] = (entry["passes"], entry["n"]) if entry else (0, 0)
        det_counts[model] = ((entry["det_passes"], entry["n"]) if entry
                             else (0, 0))

    for model in models:
        if top_up:
            # Reuse whatever the last cycle already recorded for this model and
            # only run the ones that came back short. Recovering one dropped
            # trial should not cost a fresh run of the other two models.
            out_path = run_dir / f"{slug(model)}_raw_results.json"
            if out_path.exists():
                entry = aggregate_by_task(
                    load_results(out_path).results).get(task_id)
                have = entry["n"] if entry else 0
                if have >= probe.trials:
                    counts[model] = (entry["passes"], have)
                    det_counts[model] = (entry["det_passes"], have)
                    continue
                if have:
                    await measure(model, add=probe.trials - have)
                    continue
        await measure(model)
        if time.monotonic() - started > probe.cycle_timeout:
            infra_error = True
            break

    # counted() drops trials that truncated incomplete or had their sandbox
    # skipped, so a model can come back short of `trials` through no fault of
    # the task. Top up only the short models — re-running the whole trio to
    # recover one trial spends two models' time for nothing. One attempt: if a
    # model truncates every time, that is a finding, not something to retry at.
    want = probe.trials
    for model in models:
        have = counts.get(model, (0, 0))[1]
        if 0 < have < want and not infra_error:
            await measure(model, add=want - have)

    trials = 5 if reprobe else probe.trials
    verdict = probe_verdict(*(counts.get(m, (0, 0)) for m in models),
                            trials=trials, infra_error=infra_error)
    # Grade the same trials without the judge as well. The judge only decides
    # in `tools` and `multi_turn_if`, and there it can rescue a trial whose
    # deterministic score was low — which on a structural task (an
    # indentation-exact in-place edit, say) means overruling the one grader
    # that can actually see the damage. When the two verdicts disagree, that
    # disagreement is the finding and the probe must not hide it behind
    # whichever number it happened to pick.
    det_verdict = probe_verdict(*(det_counts.get(m, (0, 0)) for m in models),
                                trials=trials, infra_error=infra_error)

    meta = {
        "task_id": task_id,
        "module": module,
        "cycle": spent + 1,
        "models": models,
        "counts": {m: list(counts.get(m, (0, 0))) for m in models},
        "det_counts": {m: list(det_counts.get(m, (0, 0))) for m in models},
        "det_verdict": det_verdict.verdict,
        "det_reason": det_verdict.reason,
        "judge_diverged": det_verdict.verdict != verdict.verdict,
        "trials": trials,
        "fingerprint": candidate_fingerprint(tasks_dir, module),
        "expected": read_expectation(tasks_dir, module, task_id),
        "judged": module in JUDGE_DECIDES,
        "elapsed_s": round(time.monotonic() - started, 1),
        "overridden": bool(override and spent >= probe.max_cycles),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    append_log(log_path, verdict, meta)
    return verdict, meta


def append_log(log_path: Path, verdict: ProbeVerdict, meta: dict) -> None:
    """Append one cycle to the probe log. Append-only, on purpose.

    "This task passed on cycle 7" has to stay visible; a task tuned to the trio
    leaves a trail of near-misses, and that trail is the evidence.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {**meta, **asdict(verdict)}
    with log_path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def run_probe_sync(*args, **kwargs) -> tuple[ProbeVerdict, dict]:
    return asyncio.run(run_probe(*args, **kwargs))
