"""Optional LLM judge pass: one batched call per module over saved results."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from .config import JudgeSettings
from .models import BenchResult, TaskResult
from .runner import ChatClient

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
# Flat judge entry object (no nested braces) — used to salvage individual
# verdicts when the array as a whole won't parse (truncation, or one bad
# reasoning string), so a single malformed entry doesn't null the whole module.
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# Judge score at/above this counts as a binary pass (drives pass^k). An LLM
# judge rarely emits exactly 1.0, so a perfect-only bar would fail genuinely
# correct answers; 0.85 matches the data_extract "mean match" success bar.
JUDGE_PASS_THRESHOLD = 0.85

# A RESCUE (flipping a deterministic failure to a pass) needs more than clearing
# JUDGE_PASS_THRESHOLD. The judge prompt tells the model to treat the
# deterministic score as the default verdict and keep it when it agrees
# (templates/judge_prompt.j2), so an echoed score is agreement, not a verdict.
# Deterministic success meanwhile requires ~0.999 (scorer.py), so every det
# near-miss in [0.85, 0.999) used to be rescued by an echo that never disagreed
# with anything — worth up to +11 points of headline on models whose scorers
# land just under the bar. A rescue now requires the judge to have actually
# RAISED the score, and to have raised it to near-correct.
JUDGE_RESCUE_THRESHOLD = 0.95
JUDGE_RESCUE_EPSILON = 1e-6

# Mirror of the rescue guard on the demote side. Same premise: an echoed score
# is agreement, not a verdict. The trap is that JUDGE_PASS_THRESHOLD is a
# judge-scale bar while several deterministic PASSES score below it (a correct
# but inefficient tool_loop episode lands at 0.8333, a redundant-call
# tool_state episode at 0.8462), so echoing the det score used to fail a trial
# the judge never disagreed with — 3 of 12 demotes across 6 judged runs, each
# with reasoning that affirmed success. The epsilon is coarse because judges
# emit 2-decimal scores: 0.83 against a det 0.8333 is an echo. Every genuine
# demote in that data was a large drop (1.0 -> 0.4, 1.0 -> 0.2).
JUDGE_ECHO_EPSILON = 0.02

# Modules the judge is not asked about at all. Across 13 models x 52 tasks x 3
# trials the judge changed ZERO verdicts on these — det and judge agreed on all
# 585 trials — so the calls were pure token cost. The deterministic checks here
# are mechanical (JSON validity, word and bullet counts, exact numeric or letter
# answers, a single tool call's name and arguments) and an LLM grading them adds
# noise, not information. Their det verdict stands and llm_score stays null.
# (tool_simple was here too until v0.10 merged it into `tools`, which IS judged:
# skipping one axis of a module would need a second grouping key in the judge for
# one task's worth of calls.)
# `tool_arg_typing` was in this set until v0.13 dissolved it into `format`,
# which is already here. The de_* extraction tasks came the other way and are
# now skipped rather than display-only — no change to any score, since the
# judge was already forbidden from moving their verdicts.
JUDGE_SKIP_MODULES = frozenset({"format", "knowledge"})

# Modules where the judge's verdict must NOT touch success in either direction —
# the det result is authoritative and the judge score/reasoning are kept for
# display only. (This replaced the old demote-only OBJECTIVE_MODULES tier: every
# module that was in it is now either skipped or display-only, and a half-open
# door the judge could still push through was not worth keeping.)
#   adversarial: prompt-injection resistance. The deterministic constraints
#     encode the correct (resist-the-injection) behavior, but the judge is
#     handed the injection prompt itself and reads it as the user's goal — so it
#     penalises a model for correctly refusing a jailbreak.
#   code: correctness is decided by executing the FAIL_TO_PASS / PASS_TO_PASS
#     unit tests in the sandbox — actual ground truth. The judge can only see
#     the source and tends to demote a fully-passing solution on formatting
#     grounds (e.g. "didn't put only the function in one code block"), zeroing a
#     functionally-correct answer and making success flaky across trials.
#   long_context: recall graded by string matching. (data_extract was here too
#     until v0.13 folded it into the skipped `format` module.)
#     Over 2028 trials the judge moved 3 verdicts here and TWO
#     of them were plainly wrong — it claimed an lc_08 reply "ends with a
#     complete sentence" when the reply ends `**7284**.`, and called a fenced
#     JSON object invalid while 14 identically-fenced trials passed unremarked.
#     The one real catch (a transposed access code) is now covered by
#     `expected.exact` in the numeric scorer.
JUDGE_DISPLAY_ONLY_MODULES = frozenset(
    {"adversarial", "code", "long_context"}
)

_console = Console(stderr=True)


def _trial_uids(results: list[TaskResult]) -> list[str]:
    """Unique judge-prompt id per trial (task_id#trial_index_within_task).

    Indexed per task_id rather than by position in ``results``, so the id in a
    warning or retry always names the Nth trial of that specific task instead
    of an arbitrary module-wide offset.
    """
    seen: dict[str, int] = {}
    uids = []
    for r in results:
        i = seen.get(r.task_id, 0)
        seen[r.task_id] = i + 1
        uids.append(f"{r.task_id}#{i}")
    return uids


def render_judge_prompt(module: str, results: list[TaskResult],
                        uids: list[str] | None = None) -> str:
    """Render the batched Jinja2 judge prompt for one module's results.

    ``uids`` lets a retry pass reuse the original trial ids for a subset of
    results, so a follow-up call's replies still merge into the same keys.
    """
    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))
    template = env.get_template("judge_prompt.j2")
    uids = uids or _trial_uids(results)
    tasks = [
        {
            "task_id": uid,
            "prompt": r.prompt,
            "expected": json.dumps(r.expected),
            "turns": (json.dumps([t.model_dump() for t in r.turns], indent=2)
                      if r.turns else None),
            "response": r.response_raw,
            "det_score": r.det_score,
            # Surfaced as its own flag, not left buried in the expected blob:
            # the judge reads `goal_tool` there and reliably demotes a correct
            # prose reply for "not calling the expected tool".
            "text_answer_ok": bool(r.expected.get("text_answer_ok")),
            # Criteria the task DELIBERATELY does not grade. Without this the
            # judge re-derives requirements from the prompt and reimposes ones
            # a task dropped on purpose: it demoted a pf_01 trial for a stale
            # frontmatter date, which is exactly the check that task removed
            # after it was measured to track a per-model habit rather than
            # capability. A narrowed rubric has to be visible to the judge or
            # it is not actually narrowed.
            "not_graded": r.expected.get("not_graded") or [],
        }
        for uid, r in zip(uids, results)
    ]
    return template.render(module=module, tasks=tasks)


def parse_judge_batch(text: str) -> dict[str, tuple[float, str]]:
    """Parse the judge's JSON array; return {task_id: (score, reasoning)}.

    Falls back to salvaging individual ``{...}`` entries when the array as a
    whole won't parse (e.g. truncation or one malformed reasoning string), so a
    single bad entry doesn't discard every verdict in the batch.
    """
    entries: list[Any] = []
    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            loaded = json.loads(match.group(0))
            if isinstance(loaded, list):
                entries = loaded
        except json.JSONDecodeError:
            entries = []
    if not entries:
        # Salvage: parse each flat object independently; skip the unparseable.
        for obj in _JSON_OBJ_RE.finditer(text):
            try:
                entries.append(json.loads(obj.group(0)))
            except json.JSONDecodeError:
                continue
    scores: dict[str, tuple[float, str]] = {}
    for entry in entries:
        parsed = _parse_entry(entry)
        if parsed:
            task_id = parsed[0]
            if task_id in scores:
                # A repeated task_id silently overwrites the earlier verdict,
                # which is how a missing trial id and a duplicate one show up
                # together: the judge conflated two trials into one entry.
                _console.print(f"[yellow]warning:[/] judge repeated task_id "
                               f"{task_id!r} in its reply; keeping the last "
                               f"verdict")
            scores[task_id] = (parsed[1], parsed[2])
    return scores


def default_judge_output(input_path: Path) -> Path:
    """Derive the judged output path: <stem>_judged.json next to the input."""
    return input_path.with_name(f"{input_path.stem}_judged.json")


async def judge_results(bench: BenchResult, settings: JudgeSettings,
                        only: set[tuple[str, str]] | None = None) -> BenchResult:
    """Judge each module with one batched call; return an annotated copy.

    ``only`` restricts judging to the given ``(module, task_id)`` pairs; every
    other trial keeps whatever verdict it already carries. That is what makes a
    re-judge after a scorer change affordable — the judge prompt shows the
    deterministic score as its default verdict, so only trials whose
    deterministic score actually moved need a fresh opinion.
    """
    judged = bench.model_copy(deep=True)
    client = ChatClient(
        endpoint=settings.endpoint, model=settings.model,
        timeout=settings.timeout, api_key=settings.api_key,
        max_tokens=settings.max_tokens, temperature=settings.temperature,
        max_attempts=settings.max_attempts,
        retry_backoff=settings.retry_backoff,
    )
    semaphore = asyncio.Semaphore(settings.concurrency)
    by_module: dict[str, list[TaskResult]] = {}
    for result in judged.results:
        if only is not None and (result.module, result.task_id) not in only:
            continue
        if result.module in JUDGE_SKIP_MODULES:
            continue
        by_module.setdefault(result.module, []).append(result)
    if not by_module:
        await client.close()
        return judged

    async def judge_module(module: str, results: list[TaskResult]) -> None:
        async with semaphore:
            await _judge_module(client, module, results)

    await asyncio.gather(*(judge_module(m, r) for m, r in by_module.items()))

    # A module whose single batched call failed transport-side (a burst of 503s
    # outlasting the client's own attempts) ends up with every trial unjudged,
    # which then silently drops that module from the judged-score denominator.
    # One serialized second pass over just those modules recovers it.
    stranded = {m: r for m, r in by_module.items()
                if not any(x.llm_score is not None for x in r)}
    if stranded:
        _console.print(f"[yellow]retrying {len(stranded)} fully-unjudged "
                       f"module(s):[/] {', '.join(sorted(stranded))}")
        await asyncio.sleep(settings.retry_backoff)
        for module, results in sorted(stranded.items()):
            await _judge_module(client, module, results)

    await client.close()
    return judged


async def _call_judge(client: ChatClient, module: str, prompt: str) -> str | None:
    """One judge chat call; returns the reply text, or None on transport failure."""
    try:
        response = await client.chat([{"role": "user", "content": prompt}])
        return response["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        _console.print(f"[yellow]warning:[/] judge call failed for module "
                       f"{module}: {exc}")
        return None


def _is_rescue(result: TaskResult) -> bool:
    """True if the judge genuinely raised a det failure to near-correct.

    Both halves matter: strictly above the deterministic score (it disagreed at
    all) and at/above JUDGE_RESCUE_THRESHOLD (it disagreed enough to call the
    response correct). A flat 1.0 rescues either way — that is the judge
    asserting the response is fully correct, not echoing anything, and no
    scorer produces det_score 1.0 on a failing trial. ``llm_score`` is
    non-None at every call site.
    """
    assert result.llm_score is not None
    if result.llm_score < JUDGE_RESCUE_THRESHOLD:
        return False
    return (result.llm_score > result.det_score + JUDGE_RESCUE_EPSILON
            or result.llm_score >= 1.0)


def _is_demote(result: TaskResult) -> bool:
    """True if the judge genuinely lowered the deterministic score.

    Only consulted for trials that deterministically passed, so it can never
    manufacture a pass — the rescue path is untouched.
    """
    assert result.llm_score is not None
    return result.llm_score < result.det_score - JUDGE_ECHO_EPSILON


# Deterministic verdicts a judge may never overturn. These are not opinions
# about quality — they are facts about what the episode did: it called a tool
# it was told not to call, it spent more turns than the budget allows, it
# looped, or it was cut off mid-generation. An LLM reading a transcript is not
# better placed to decide these than the harness that recorded them, and on
# `tools` (0.30 of the headline) the judge can move `success` in both
# directions, so without this a future task's forbidden-tool gate would be
# overridable at llm_score >= 0.95.
_HARD_GATES = ("used_forbidden_tool", "within_budget", "no_loop_detected",
               "no_premature_stop")


def hard_gate_failure(result: TaskResult) -> str | None:
    """The name of the deterministic gate this trial failed, if any."""
    breakdown = result.det_breakdown or {}
    if breakdown.get("used_forbidden_tool"):
        return "used_forbidden_tool"
    for gate in ("within_budget", "no_loop_detected", "no_premature_stop"):
        value = breakdown.get(gate)
        if value is not None and not value:
            return gate
    if result.truncation_class == "degenerate":
        return "degenerate_truncation"
    return None


def judge_anchor_is_stale(result: TaskResult) -> bool:
    """True if `llm_score` was formed against a different deterministic score.

    Trials judged before `judge_anchor_det` existed carry None and cannot be
    checked here; `rescore` stamps the anchor for those the moment it moves one,
    which is the only point where the old score is still known.
    """
    if result.judge_anchor_det is None:
        return False
    return abs(result.judge_anchor_det - result.det_score) > 1e-9


def apply_judge_verdict(result: TaskResult, module: str) -> None:
    """Resolve ``result.success`` from the deterministic and judge verdicts.

    Split out of the judging loop so ``sllmb rescore`` re-derives the final
    verdict from a stored ``llm_score`` through exactly this code path, rather
    than a second copy of the rules that can drift from it.

    A verdict whose anchor has moved (``judge_anchor_is_stale``) stays on
    display but may no longer lift a deterministic failure: the gap it shows is
    an artefact of the re-score, not a disagreement. It regains that power when
    the judge re-scores it against the number that now exists.

    Expects ``result.llm_score`` set and ``result.success`` still holding the
    deterministic verdict (which is also frozen into ``det_success`` here, so
    pre-field raw files get backfilled). Calling it twice on the same trial
    without resetting ``success`` back to the deterministic verdict would
    freeze the judge's own answer in as ``det_success`` — callers that re-judge
    after re-scoring must clear the stale verdict first.
    """
    assert result.llm_score is not None
    result.det_success = result.success
    # The judge overrides the deterministic success flag, but on
    # objectively-graded modules it may only DEMOTE (det is ground truth
    # there) — never rescue a det failure. Elsewhere it overrides both ways.
    judged_pass = result.llm_score >= JUDGE_PASS_THRESHOLD
    gate = hard_gate_failure(result)
    if judged_pass and not result.det_success and gate is not None:
        # A rescue over a hard gate is not an appeal, it is overruling the
        # record. Refused regardless of how convinced the judge is.
        judged_pass = False
        result.judge_blocked_by = gate
    if judged_pass and not result.det_success and judge_anchor_is_stale(result):
        judged_pass = False
        result.judge_blocked_by = (
            f"stale judge anchor (judged against det "
            f"{result.judge_anchor_det:.4f}, now {result.det_score:.4f})")
    if judged_pass and not result.det_success and not _is_rescue(result):
        # The judge kept (or barely moved) the deterministic score, so it
        # never disagreed with the failure — no rescue on an echo.
        judged_pass = False
    if not judged_pass and result.det_success and not _is_demote(result):
        # The judge kept the deterministic score on a trial that
        # deterministically passed: agreement, not a demotion. The displayed
        # llm_score is left exactly as the judge emitted it — a ~zero delta is
        # the honest signal here.
        judged_pass = True
    if module in JUDGE_DISPLAY_ONLY_MODULES or module in JUDGE_SKIP_MODULES:
        # det is authoritative; never let the judge flip success. Skipped
        # modules land here too: files judged before they were skipped still
        # carry an llm_score, and a module we deliberately stopped asking about
        # must not keep grading through stored data.
        return
    result.success = judged_pass


async def _judge_module(client: ChatClient, module: str,
                        results: list[TaskResult]) -> None:
    """Judge one module's results in place; null scores on any failure."""
    uids = _trial_uids(results)
    content = await _call_judge(client, module, render_judge_prompt(module, results, uids))
    if content is None:
        return
    scores = parse_judge_batch(content)
    if not scores:
        _console.print(f"[yellow]warning:[/] malformed judge JSON for module "
                       f"{module}; scores set to null")
        return

    # The judge model occasionally drops a trial from its reply array (seen
    # with near-duplicate responses within a task, likely deduped or merged
    # on the judge's side). One retry scoped to just the missing trials
    # recovers most of these without re-spending tokens on the whole module.
    missing = [i for i, uid in enumerate(uids) if uid not in scores]
    if missing:
        retry_results = [results[i] for i in missing]
        retry_uids = [uids[i] for i in missing]
        retry_content = await _call_judge(
            client, module, render_judge_prompt(module, retry_results, retry_uids))
        if retry_content is not None:
            scores.update(parse_judge_batch(retry_content))

    for i, result in enumerate(results):
        uid = uids[i]
        if uid in scores:
            result.llm_score, result.llm_reasoning = scores[uid]
            # The score the judge was just shown, so a later re-score can tell
            # whether this verdict still stands against the current one.
            result.judge_anchor_det = result.det_score
            apply_judge_verdict(result, module)
        else:
            _console.print(f"[yellow]warning:[/] judge omitted {uid}; "
                           f"score set to null")


def _parse_entry(entry: Any) -> tuple[str, float, str] | None:
    """Validate one judge array entry; None if malformed."""
    if not isinstance(entry, dict):
        return None
    try:
        task_id = str(entry["task_id"])
        score = float(entry["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return task_id, min(1.0, max(0.0, score)), str(entry.get("reasoning", ""))
