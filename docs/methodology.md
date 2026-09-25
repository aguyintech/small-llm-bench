# Methodology

How a trial becomes a number, what the LLM judge is allowed to change, and
how to read a scorecard without over-reading it. The [CLI reference](cli.md)
covers commands and options; the [changelog](../CHANGELOG.md) covers why the
rules are what they are.

- [The headline](#the-headline)
- [Det score and pass^k](#det-score-and-passk)
- [What gets scored](#what-gets-scored)
- [The deterministic scorers](#the-deterministic-scorers)
- [Recovery](#recovery)
- [The LLM judge](#the-llm-judge)
- [Reproducibility and comparability](#reproducibility-and-comparability)
- [Reading a scorecard](#reading-a-scorecard)
- [Interpretation checklist](#interpretation-checklist)
- [Task calibration](#task-calibration)
- [Sandboxing](#sandboxing)

---

## The headline

The leaderboard ranks by **module-weighted pass^3**:

1. Each trial is a pass or a fail.
2. Each task gets a pass^k score from its trials (below).
3. Tasks are averaged within their module.
4. Modules are combined with a weight preset (`balanced` by default).

Every row on a board uses the same `k` (`headline_k`), so one file with a
missing trial cannot change the exponent for anyone else.

Three per-module numbers exist in the code. They answer different questions,
so it is worth knowing which one you are looking at:

| Number | Field | What it is |
|---|---|---|
| **pass^k, module-weighted** | `headline` | What the leaderboard ranks by |
| mean det score | `overall_det` | Partial credit; useful when authoring tasks |
| mean per-task pass fraction | `items` pairwise table | The unit the paired sign test runs on |

### Weight presets

| Module | `balanced` | `agentic` | `coding` | Why it carries that weight (balanced) |
|---|---|---|---|---|
| `tools` | 0.30 | 0.38 | 0.28 | Tool use is why a local model gets deployed at all |
| `format` | 0.15 | 0.12 | 0.15 | Structured output is how a model gets wired into anything |
| `code` | 0.15 | 0.08 | 0.21 | Executed against hidden tests: the most valid signal in the bank |
| `multi_turn_if` | 0.13 | 0.16 | 0.11 | Instruction retention over turns, a known small-model failure |
| `long_context` | 0.12 | 0.12 | 0.12 | Retrieval over long documents is a main reason to run a small model |
| `adversarial` | 0.08 | 0.08 | 0.07 | Robustness matters, but is partly restraint, which does not track size |
| `knowledge` | 0.07 | 0.06 | 0.06 | A small model is not a knowledge store |

The weights are a stated claim about what a locally-run model is for, not a fit
to the board. One rule bounds them, enforced in
`tests/test_v13_bank_and_weights.py`: **no module may carry more than 2× or
less than 0.5× its share of the task bank.** That is what stops a preset from
resting a third of the headline on five tasks.

### Other schemes

`--scheme band` (v0.4–v0.12) weighted tasks by calibration band, and
`--scheme legacy` (v0.3) by a baseline/hard tier. Both remain so old files can
be re-scored the way they were published. Difficulty-weighting was dropped
because it is circular (it weights by the thing being measured) and unstable
(an honest re-banding left four tasks carrying 40% of the score). Bands are now
a reporting axis: the scorecard's *Capability bands* table shows pass^k per
band, and its overall row is band-weighted, so it will not match the module
headline exactly.

---

## Det score and pass^k

**Det score** is a 0–1 grade with partial credit, averaged over trials: two
fields of three right is 0.67; the right end state reached with extra calls
might be 0.9. It says how close the output was.

**pass^k** throws partial credit away. A trial passes only at det score
≥ 0.999 (or when the judge's verdict, under the rules below, says so). For a
task with `n` trials of which `c` passed:

```
pass^k = C(c, k) / C(n, k)
```

the probability that `k` trials drawn at random all pass. pass^1 is the plain
success rate; pass^3 credits a capability only if the model shows it every
time. A task passed 2 of 3 times contributes 0.67 to pass^1 and 0 to pass^3,
so the gap between the two is a flakiness gauge.

`code` sets its own pass flag: a solution that passes after execution feedback
is a pass, even though its det score is 0.85 or 0.70.

A task holding fewer than `k` scorable trials is dropped from that model's
pass^k and counted in the row's `tasks_excluded`. **A row scored on 36 tasks is
not comparable to one scored on 39.**

---

## What gets scored

- **Infra errors are excluded, not zeroed.** A transport failure, timeout or
  5xx is absent from the scores. A 500 whose body says the server could not
  parse the model's tool call is the model's fault and scores as a failure.
- **A truncated trial fails.** A completion that hit its token cap is
  classified by what it produced:
  - `degenerate` — it looped (repeated word n-grams, or a character cycle
    with no whitespace). Always a failure.
  - `incomplete` — it did not loop, it just did not finish. Also a failure:
    at the current caps no model at or above 9B dense hits them, so what
    remains is a model that could not finish.

  In `knowledge`, `long_context` and `format` a truncated reply scores 0,
  because a cut-off answer never reached the user. Other modules grade what
  was produced, so a truncated trial that already passed (working code
  followed by rambling, say) stays a pass. A truncated turn in `multi_turn_if`
  scores 0 for that turn only.
- **Reasoning is not the answer.** `<think>` blocks, including unbalanced
  ones, are stripped before grading. The reasoning channel is recorded per turn
  so a cut turn can be diagnosed.
- **Silence scores 0.** An empty reply does not vacuously satisfy negative
  constraints like "under 80 words".
- **A deployment too small for the bank is refused**, not scored. `run` checks
  the served context window before it starts.

The scorecard prints a coverage line when any of this applied, including what
each cut turn produced (reasoning characters, content characters, tool calls).

---

## The deterministic scorers

**`tools`** — one module, graded by task shape. Each task declares an `axis`
that the scorecard prints as a sub-row.

- *Single call* (`call`, `parallel`, `signature`): tool called, right tool,
  required arguments present, argument values matched. `parallel` tasks need
  several independent calls in one turn; `signature` tasks need argument types
  that match the schema (an integer is not `"3"`).
- *Stateful* (`state`): graded on the **final state** of the simulated
  backend (key-value store, orders, filesystem), not the call sequence.
  `goal_match·0.60 + no_side_effects·0.20 + efficiency·0.10 + no_loop·0.10`.
  `goal_match` credits the share of the gap between the seeded and the
  expected state that the model closed, so doing nothing scores 0. Touching a
  protected key is a policy violation.
  Document edits can be graded with `file_checks`, structural invariants
  instead of one byte-exact file: `frontmatter`, `table_wellformed`,
  `table_rows_preserved`, `table_row_position`, `section_contains`,
  `bullets_wellformed`, `lines_preserved`, each optionally scoped to a
  Markdown section. A broad instruction has many correct renderings; this
  grades what must be preserved and where new material must land.
- *Agentic loop* (`loop`):
  `goal_reached·0.40 + efficiency·0.30 + no_loop·0.20 + no_premature_stop·0.10`
  (content checks, where present, take a share). Repeating the same call with
  the same arguments zeroes `no_loop`, unless the previous call returned an
  error.
- *Hard gates*, on every shape: calling a `forbidden_tools` tool, and
  spending more than `max(2 × optimal_turns, optimal_turns + 3)` calls, fail
  the trial outright.
- `final_text_checks` grade the model's own closing words, so an episode that
  did the work and then claimed something false fails.

**`code`** — the model's code runs in a [sandbox](#sandboxing) against hidden
test cases with a 5-second timeout. A static parse check runs first. See
[Recovery](#recovery) for the repair loop.

**`format`** — a list of verifiable constraints per task: valid JSON, exact
keys, nested types, values, Markdown structure, word limits, required and
banned content. Score is the fraction met; a JSON task scores 0 if the output
is not JSON. Extraction tasks (`de_*`) match field by field, with accepted
variants per field. A constraint that names nothing to look for **fails
closed**, and a bank-wide test rejects unknown constraint keys, so a mis-keyed
check cannot pass silently.

**`knowledge`** — numeric answers are extracted and matched within ±1%
(exactly, for identifiers); factual answers by substring or fuzzy match. Tasks
that say where the answer goes ("end your reply with the number") grade that
too.

**`long_context`** — the needle has to be the value retrieved. Near-miss and
stale-value distractors score as failures. Documents are generated per task
(4k to 32k tokens of log filler, or a long meeting transcript) with a unique
prefix, so
a prefix-caching server cannot skip the prefill of one task because it saw
another.

**`multi_turn_if`** — constraints accumulate across turns and the last turn is
checked against every rule still standing. A turn can revoke an earlier rule,
and an earlier input can be corrected so a derived value must be recomputed.

**`adversarial`** — scored as utility under attack: the model has to do the
real task and ignore the injected one. Refusing the whole task is not credited
as safety. One task (`adv_21`) is graded on restraint instead and is labelled
as such in `items`.

---

## Recovery

These models run in a loop with errors fed back, so the question that matters
is whether a model reliably arrives at the right outcome, not whether its first
token was right.

- **`code` runs an execute-and-fix loop.** Between attempts the model sees the
  real traceback or the inputs that came back wrong, never the expected value.
  Passing at any attempt is a pass; the det score decays 1.00 / 0.85 / 0.70 so
  one-shotting still ranks highest, and code that never passes is capped at
  0.5. Two tasks (`cd_23`, `cd_27`) allow one attempt only, so first-try skill
  shows up in the headline too. The leaderboard's **1st-try code** column
  shows the rest.
- **Tools fail on purpose.** A task can make a tool fail once
  (`first_call_behavior`), reject a wrong argument on every call
  (`require_args`), or never work at all (`always_fail`). They are paired so
  the fixes differ: a transient lock wants the same call retried, a rejected
  argument wants it changed, and a permanent failure wants the model to say it
  did not work instead of claiming success.
- **Efficiency is priced, thrashing fails.** An extra exploratory call costs a
  fraction of one dimension; more than the budget above fails the trial.
- **Recovery never buys a pass for the wrong outcome.** Final state, executed
  tests and the retrieved needle are graded as found.

---

## The LLM judge

The judge is an optional pass over a saved file (`sllmb judge`). It never
touches the model under test and never modifies the raw file. It is told to
treat the deterministic score as the default verdict and change it only for a
concrete reason: raise it when the output works and the check penalised a
harmless difference, lower it when the output looks right but would fail in
practice. It is an appeals court on the deterministic scorer, not a second
scorer.

**Which modules it may move:**

| Modules | Judge role |
|---|---|
| `tools`, `multi_turn_if` | May move pass/fail, both directions, under the rules below |
| `adversarial`, `code`, `long_context` | Shown, never moves a verdict |
| `format`, `knowledge` | Not sent at all |

`adversarial` is display-only because the judge reads the injected instruction
as the user's goal; `code` because the executed tests are ground truth;
`long_context` because its few verdict changes were judge errors. `format` and
`knowledge` are mechanical checks, and across 585 judged trials the judge
changed none of them.

**Rules for moving a verdict:**

- A **rescue** (deterministic fail → pass) needs a judge score ≥ 0.95 that is
  strictly above the deterministic one. Echoing the deterministic score is
  agreement, not a verdict.
- A **demotion** (deterministic pass → fail) needs a judge score below 0.85
  that actually lowered the deterministic one by more than 0.02.
- The judge can never overturn a **hard gate**: a forbidden tool call, a blown
  call budget, a detected loop, or a premature stop. A refused rescue is
  recorded in `judge_blocked_by`.
- A verdict is tied to the deterministic score the judge was shown
  (`judge_anchor_det`). If a later `rescore` moves that score, the old verdict
  loses the power to rescue until the trial is re-judged.

**Coverage.** A module whose batched call fails loses every verdict in it, so
calls retry with long backoff and fully unjudged modules get a second pass.
Whatever is still missing is reported: `judge` exits 2, and the scorecard and
leaderboard flag the row. A partially judged row uses the deterministic score
for the missing modules, which makes it a blend. Report coverage alongside a
judged number, or use the raw file.

**How much it moves.** On the 27-model reference fleet the judge changed 7
verdicts out of 2,430 judged trials, all in `tools`. Treat it as a court that
rarely sits.

---

## Reproducibility and comparability

pass^k is a statement about a sampler, so the sampler is recorded.

- `--seed` (default 0) sends a seed offset by trial index. Servers that honour
  it make a run reproducible; the k trials of a task stay k different samples.
- An unset temperature is recorded as "server default" in words.
- The results file records bench version, task bank hash, per-trial task and
  tool-world hashes, token budget, thinking, sandbox backend, interpreter,
  concurrency and served context window.

`leaderboard` compares each row's settings against the most common ones and
lists differences in `comparability_mismatch`. `--only-new` and `--add-trials`
refuse to pool trials recorded under different settings, or against a changed
task or tool world.

**Timeouts scale with the budget.** The read timeout per request is
`max_tokens / 20 tok/s + 120 s` (floor 60 s), so a 12k-token request is
allowed about twelve minutes on a slow model and a dead endpoint still fails in
ten seconds at connect. A read timeout is never retried: the same request
would do the same work and time out again.

**Interrupted runs resume.** Every finished trial is appended to
`<output>.partial.jsonl`; re-running the same command picks them up.

---

## Reading a scorecard

`run` and `score` print one table per file:

| Column | Meaning |
|---|---|
| `Module` | Module, with the `tools` axes as sub-rows |
| `Tasks` | Distinct tasks (not trials) |
| `Weight` | The module's weight in the active preset |
| `Det score` | Mean det score, with partial credit |
| `Pass` | Deterministic pass^k at the file's `k` |
| `tok/s (wall)` | Completion tokens ÷ wall time. Display only |
| `pp tok/s` | Prompt processing: evaluated prompt tokens ÷ server-reported prefill time, excluding prefix-cache hits |
| `tg tok/s` | Generation: completion tokens ÷ server-reported decode time |
| `LLM judge` | Mean judge score (judged files only) |
| `Delta` | `LLM judge − Det score` |
| `Pass (LLM)` | Pass^k after the judge's rescues and demotions |
| `Pass Δ` | `Pass (LLM) − Pass` |

Below it: pass^k per **capability band**, det score per **difficulty**, the
`code` solved-vs-one-shot split, and any warnings (skipped code tasks, judge
coverage, excluded trials, truncation with the cut-turn evidence).

The speed split comes from the server's own timing report and never affects
the score. llama.cpp server reports it on every response. oMLX reports it
only on streaming responses, which the bench does not use, so both columns
show `—`. Other servers get token counts only. Decode speed depends on
batching, so compare `tg tok/s` only between runs at the same `--concurrency`.

---

## Interpretation checklist

Before writing anything up:

1. **Raw or judged file?** Say which a number came from.
2. **Judge coverage 100%?** If not, the judged overall is a blend.
3. **Trials.** pass^3 needs at least three trials, and trials only vary if
   the server samples (pass `--temperature` if it defaults to greedy).
4. **Tasks scored.** A row with exclusions is scored on a smaller bank.
5. **Intervals.** Run `sllmb items` and check the 95% interval and the
   pairwise table. Gaps inside the interval are ties.
6. **Sandbox.** If code tasks were skipped, `code` and the overall are not
   comparable.
7. **Preset.** `balanced`, `agentic` and `coding` can reorder models. Name
   the preset with the number.
8. **Version.** Numbers do not compare across bench versions; the scorecard
   title carries the version.

---

## Task calibration

Every task has a difficulty (`easy`/`medium`/`hard`) and a calibration band
assigned from measured pass rates across the reference fleet:

| Band | Fleet pass rate | Role |
|---|---|---|
| `anchor` | ≥ 95% | Harness health: if a model fails these, suspect the setup |
| `mid` | 60–95% | |
| `hard` | 30–60% | |
| `frontier` | < 30% | Currently empty |

Tasks without a measured band fall back to one derived from difficulty.

A new task goes through `sllmb probe` against a weak/mid/strong trio before it
enters the bank, and stays only if a full-fleet sweep confirms it. `sllmb
items` is what retires tasks: a task nobody fails, or that ranks models
backwards, costs runtime and adds noise to the sign test. Retired definitions
are kept in `tests/fixtures/retired_bank/`, where tests still use them.

---

## Sandboxing

Only `code` executes model output. Every other module simulates its tools
with a deterministic mock registry and does no real I/O.

The sandbox backend is auto-detected, best first:

1. **docker** — macOS and Linux
2. **podman** — rootless drop-in for Docker
3. **bwrap** (bubblewrap) — Linux, no daemon
4. **sandbox-exec** — built into macOS (deprecated by Apple, still shipped)
5. **rlimit** — resource limits only, **no filesystem or network isolation**

Backends 1–4 confine network and filesystem; the container backends run with
`--network none`, a read-only root, dropped capabilities, a non-root user and
memory/PID/CPU caps. Docker and Podman pull `python:3.12-slim` once.

**Fail-closed:** if only `rlimit` is available, code tasks are skipped (scored
0, with a notice) unless you pass `--allow-unsandboxed`. macOS does not
enforce `RLIMIT_AS`, so even the memory cap is a no-op there.

```bash
sllmb run -k code                   # docker if present, else sandbox-exec on macOS
sllmb run --sandbox docker          # force a backend
sllmb run --allow-unsandboxed       # permit the rlimit-only fallback
```
