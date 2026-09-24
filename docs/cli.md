# CLI reference

Every command, every option, with examples. The package installs two names for
the same program, `small-llm-bench` and the shorter `sllmb`; this page uses
`sllmb`.

Defaults come from a `.env` file in the current directory (copy
`.env.example` to start) and from the environment. A command-line option always
wins over both. See [Configuration](#configuration) for every variable.

Run the CLI from the repository root: the task bank in `tasks/` is found
relative to the working directory, or to the checkout the package was
installed from in editable mode.

| Command | What it does |
|---|---|
| [`run`](#run) | Run the bank against an endpoint and save a raw results file |
| [`judge`](#judge) | Optional LLM-judge pass over a saved file; writes a `*_judged.json` copy |
| [`score`](#score) | Re-print the scorecard for a saved file |
| [`compare`](#compare) | Two or more saved files side by side |
| [`leaderboard`](#leaderboard) | Static HTML leaderboard over a directory of results |
| [`items`](#items) | Per-task item analysis, confidence intervals, pairwise separability |
| [`rescore`](#rescore) | Re-grade stored trials with the current scorer, no model calls |
| [`migrate`](#migrate) | Bring stored files in line with the current task bank |
| [`probe`](#probe) | Screen one candidate task against a three-model trio |

---

## `run`

Runs the bank against an OpenAI-compatible endpoint, scores every trial
deterministically as it finishes, prints the scorecard, and writes the results
file.

```
sllmb run [OPTIONS]
```

**Target and selection**

| Option | Default | Description |
|---|---|---|
| `--model` | `BENCH_MODEL` | Model name as the endpoint knows it (e.g. `qwen3:8b`). |
| `--endpoint` | `BENCH_ENDPOINT` | Endpoint URL, including the `/v1` suffix. |
| `--profile` | `full` | `full` (39 tasks) or `fast` (17 tasks, a smoke test). |
| `--fast` | off | Deprecated alias for `--profile fast`. |
| `--filter`, `-k` | — | Only tasks whose id or module name contains this substring. |
| `--modules` | — | Comma-separated exact module names, e.g. `tools,code`. Combined with `--filter` by AND. |
| `--tasks-dir` | discovered `tasks/` | Load tasks from another directory. Only the modules named by `--modules` are read. Requires `--output`, so a scratch run cannot overwrite a full-bank file. |
| `--output` | `BENCH_OUTPUT_DIR/<model>_raw_results.json` | Results file. The model name is slugged (`qwen3:8b` → `qwen3_8b`). |

**Sampling and budget**

| Option | Default | Description |
|---|---|---|
| `--trials` | `BENCH_TRIALS` (1) | Trials per task. Use 3 for pass^3, the published headline. |
| `--temperature` | server default | Sampling temperature. Omitted means the server's own default applies, and the file records that. |
| `--seed` | `BENCH_SEED` (0) | Base seed, offset by trial index so k trials are k different samples. `-1` sends no seed, for servers that reject the field. |
| `--thinking` | server default | Sends `chat_template_kwargs={"enable_thinking": true}`. llama.cpp needs `--jinja`. |
| `--max-tokens` | per-module caps | Completion cap per request. Overrides every per-module and per-task cap in both directions, prints which it moved, and marks the file `max_tokens_override` so it never pools with a default-cap run. |
| `--concurrency` | `BENCH_CONCURRENCY` (2) | Concurrent requests. Use 1 for speed numbers that describe the model rather than your batching. |

**Resuming and reusing trials**

| Option | Default | Description |
|---|---|---|
| `--resume` / `--no-resume` | on | Recover trials from `<output>.partial.jsonl`, the checkpoint an interrupted run leaves behind. Only trials from an identically configured run are recovered. |
| `--only-new` | off | Reuse trials already in the output file whose task content and run config are unchanged; run only what is missing. |
| `--reuse-params` | off | Adopt temperature, thinking and any recorded `--max-tokens` override from the existing file, for options not given on the command line. The endpoint is never adopted; it comes from `--endpoint` or `BENCH_ENDPOINT`. |
| `--ignore-task-hash` | off | Match old trials by `(module, task_id)` only, so only truncated, infra-errored or missing trials re-run. Pair with `--add-trials` after a cap change. |
| `--ignore-world-hash` | off | Reuse trials recorded against a different version of the simulated tool world. Only when you know the change cannot affect those tasks. |
| `--add-trials N` | — | Add N more trials per task on top of the file (3 + 2 = 5). Errors on a config mismatch instead of running fresh. |

The endpoint is not part of a run's identity. If your server's address
changes, resuming, `--only-new` and `--add-trials` keep reusing the trials it
already produced; the new address is simply recorded on the next run.

**Code sandbox and deployment checks**

| Option | Default | Description |
|---|---|---|
| `--sandbox` | `BENCH_SANDBOX_BACKEND` (`auto`) | `auto`, `docker`, `podman`, `bwrap`, `sandbox-exec` or `rlimit`. See [Sandboxing](methodology.md#sandboxing). |
| `--allow-unsandboxed` | off | Allow the rlimit-only fallback (no filesystem or network isolation). Without it, code tasks are skipped when no real sandbox exists. |
| `--sandbox-memory` | `BENCH_SANDBOX_MEMORY_MB` (256) | Sandbox memory cap in MB. |
| `--allow-undersized-context` | off | Run even though the served context window cannot hold every task. Those tasks score 0. |
| `--save-responses` | off | Keep raw API responses in the file (much larger; for debugging). |
| `--verbose`, `-v` | off | Per-task outcome, score breakdown and a snippet of each failure. |

Before any work, `run` reads the served context window from llama.cpp's
`/props` (or llama-swap's `/upstream/<model>/props`) and refuses to start if a
task cannot fit, naming the tasks. An endpoint that cannot report its window is
not an error. It also aborts early if the first `BENCH_SANITY_CHECK_AFTER`
trials all come back empty, which is what a wrong model id or a server without
a tool API looks like.

Press `p` during a run to pause after the in-flight requests finish. Paused
time is not counted in the recorded duration.

### Examples

```bash
# Full bank, three trials, the settings the reference fleet used
sllmb run --model qwen3:8b --trials 3 --thinking --concurrency 1

# 17-task smoke test of a new endpoint
sllmb run --endpoint http://gpu-box:8000/v1 --model my-model --profile fast

# Only the tools and code modules
sllmb run --model qwen3:8b --modules tools,code

# A run was interrupted: re-issue the same command and it resumes
sllmb run --model qwen3:8b --trials 3 --thinking --concurrency 1

# Firm up a borderline model with two more trials per task
sllmb run --model qwen3:8b --add-trials 2 --reuse-params
```

---

## `judge`

Runs the optional LLM judge over a saved file and writes an annotated copy.
The original is never modified. One batched call per judged module, run in
parallel up to `JUDGE_CONCURRENCY`. `format` and `knowledge` are never sent,
so a full-bank file costs five calls.

```
sllmb judge --results FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--results` | required | Saved results file. |
| `--judge-endpoint` | `JUDGE_ENDPOINT` | OpenAI-compatible judge endpoint. |
| `--judge-model` | `JUDGE_MODEL` | Judge model name. |
| `--output` | `<results>_judged.json` | Where to write the annotated copy. Must differ from `--results`. |
| `--allow-partial` | off | Exit 0 even when some module ended unjudged. |

- The API key comes only from `JUDGE_API_KEY`, never a flag, so it stays out
  of shell history.
- Exit code `2` means at least one module is unjudged after all retries. The
  file is written anyway. A judged number from such a file mixes judged and
  deterministic modules; re-judge before comparing it.
- A whole module goes in one prompt, so use a judge with a large context
  window.
- What the judge may and may not change is in
  [the methodology](methodology.md#the-llm-judge).

```bash
sllmb judge --results results/qwen3_8b_raw_results.json

# A local judge instead of the default cloud one
sllmb judge --results results/qwen3_8b_raw_results.json \
  --judge-endpoint http://localhost:11434/v1 --judge-model qwen3:32b
```

---

## `score`

Re-prints the scorecard for a saved file. No model calls.

```
sllmb score --results FILE [--weights PRESET] [--scheme SCHEME]
```

| Option | Default | Description |
|---|---|---|
| `--results` | required | Saved results file, raw or judged. |
| `--weights` | `balanced` | Module weight preset: `balanced`, `agentic` or `coding`. |
| `--scheme` | `module` | Headline scheme. `module` is the current one. `band` and `legacy` reproduce the headlines of v0.4–v0.12 and v0.3 files. |

### Weight presets

Exact values, from `MODULE_WEIGHT_PRESETS` in `small_llm_bench/scorer.py`:

| Module | `balanced` | `agentic` | `coding` |
|---|---|---|---|
| `tools` | 0.30 | 0.38 | 0.28 |
| `format` | 0.15 | 0.12 | 0.15 |
| `code` | 0.15 | 0.08 | 0.21 |
| `multi_turn_if` | 0.13 | 0.16 | 0.11 |
| `long_context` | 0.12 | 0.12 | 0.12 |
| `adversarial` | 0.08 | 0.08 | 0.07 |
| `knowledge` | 0.07 | 0.06 | 0.06 |

Weights renormalise over the modules present in a file. Every preset obeys
one rule, enforced by `tests/test_v13_bank_and_weights.py`: no module may carry
more than 2× or less than 0.5× its share of the task bank. A preset can
emphasise a module; it cannot rest the headline on too few tasks to measure it.

```bash
sllmb score --results results/qwen3_8b_raw_results_judged.json
sllmb score --results results/qwen3_8b_raw_results_judged.json --weights agentic
```

---

## `compare`

Two or more saved files side by side: one column per model, one row per
module, and the weighted overall.

```
sllmb compare FILE1 FILE2 [FILE3 ...] [--weights PRESET] [--scheme SCHEME]
```

```bash
sllmb compare results/qwen3_8b_raw_results.json results/llama3.1_8b_raw_results.json
```

---

## `leaderboard`

Builds one self-contained HTML page over every saved result in a directory,
plus a sibling `.json` with the same data. Where a model has both a raw and a
judged file, the judged one is used; a judged file on its own is a full row.
No server, no CDN, no build step: open the file.

```
sllmb leaderboard [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--results-dir` | `results` | Directory of result files. |
| `--output` | `results/leaderboard.html` | HTML to write; `leaderboard.json` is written beside it. |
| `--weights` | `balanced` | Weight preset. |
| `--scheme` | `module` | Headline scheme. |
| `--open` | off | Open the page in your browser afterwards. |

The page has four tabs over the same data:

- **Leaderboard** — every model as a bar on a fixed 0–100 axis, grouped by
  score tier: S ≥ 90, A ≥ 80, B ≥ 70, C ≥ 60, D ≥ 45, E ≥ 30, F below.
- **By tier** — one chart per tier, all on the same axis.
- **By params** — models bucketed by total parameters (24B+, 7–23B, 3–6B,
  under 3B) and split into dense and sparse. Sizes come from
  [`models.yaml`](../models.yaml); a model missing from it shows `?`.
- **Detailed** — the full sortable table: size, tasks scored, run minutes,
  det score, pass, judged columns, one column per module, speed columns.

Clicking a model opens a card with a radar of its per-module pass^k, tasks
excluded, run time, throughput, first-try code rate and judge coverage.

Rows are ranked on the judged headline, falling back to the deterministic one
for a model with no judge run. Every row is computed at one fleet-wide `k`.
Things the page flags rather than hides:

- **`=` badge** — the paired sign test cannot separate this row from the one
  above it. The header chip shows how many model pairs are separable at all.
- **⚠ and `*`** — the judge did not cover every trial; the affected module
  cells fall back to the deterministic score.
- **`comparability_mismatch`** (in the JSON and the card) — this row's run
  settings differ from the most common ones: task bank, bench version,
  trials, thinking, token budget, profile, sampler or sandbox. A
  served context window too small for the bank is flagged too; a different
  window that still fits is not.

```bash
sllmb leaderboard --open
sllmb leaderboard --results-dir results/reference --open     # the shipped fleet
```

---

## `items`

Item analysis over every result file in a directory. This is what task
pruning and calibration are based on, and what to check before quoting a
ranking.

```
sllmb items [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--results-dir` | `results` | Directory of result files. |
| `--judged` | off | Read the `*_judged.json` copies instead of the raw files. |
| `--module` | — | Only this module's tasks. |
| `--min-models` | 1 | Dim tasks seen in fewer models than this. |
| `--json` | — | Also write the full per-task stats to this file. |
| `--cohort` | `PROBE_MODELS` | Comma-separated models in weak-to-strong order, for the band columns. |

It prints, in order:

1. **A per-task table**: pass rate, `disc` (top third minus bottom third of
   the loaded models), `band` (strongest minus weakest model of the declared
   cohort), the cohort read rung by rung, flakiness, and cost.
2. **A class per task**:

   | Class | Meaning |
   |---|---|
   | `discriminating` | Separates the top third of models from the bottom third by at least 0.35. |
   | `floor_only` | The whole cohort passes it; it separates only models below the cohort. |
   | `inverted_leg` | Clears the discrimination floor but ranks some cohort rung backwards. |
   | `flaky` | Half or more of the models pass it on some trials and fail on others. |
   | `anchor` | Every model passes, and the task is a declared floor sentinel. |
   | `dead_easy` / `dead_hard` | Every model passes / fails on every trial. No ranking signal. |
   | `restraint` | Graded on what the model declines to do; discrimination does not apply. |
   | `weak` | None of the above. |

3. **A per-model headline with a 95% Wilson interval.** The unit is the task,
   not the trial. Overlapping intervals are ties.
4. **Size inversions** — a model declared larger in `models.yaml` scoring lower
   on a module. MoEs are ordered by active parameters.
5. **Pairwise separability** — for every pair of models, wins–losses over the
   tasks both ran, an exact sign-test p-value Holm-corrected across all pairs,
   a `separable`/`tie` verdict, and for ties the number of tasks the gap would
   need. Adjacent pairs on the board come first, marked `*`. This table sorts
   by unweighted per-task pass rate, so its order can differ from the
   leaderboard's.

```bash
sllmb items --judged
sllmb items --module tools --json item_stats.json
```

---

## `rescore`

Re-grades stored trials with the current deterministic scorer. Everything the
scorer reads is stored per trial, so a scorer fix reaches old runs without
calling a model.

```
sllmb rescore --results FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--results` | required | Saved results file, raw or judged. |
| `--output` | `<input>_rescored.json` | Where to write the re-graded copy. |
| `--in-place` | off | Overwrite the input instead. |
| `--tasks-dir` | discovered `tasks/` | Task bank to grade against. |
| `--allow-task-drift` | off | Also re-grade trials whose stimulus changed. The result is not comparable: the stored response answers the old task. |
| `--sandbox` | none | Needed to re-grade `code`, which re-executes the candidates. |
| `--allow-unsandboxed` | off | Permit code re-grading without a real sandbox. |
| `--judge` | off | Re-judge only the trials whose deterministic grade moved. |
| `--judge-endpoint` / `--judge-model` | `JUDGE_*` | Judge to use with `--judge`. |

- **Grading-only changes are applied.** A new or corrected check changes
  nothing the model saw.
- **Stimulus changes are skipped** and listed as needing a re-run: a changed
  prompt, system prompt, tool list, or injected tool behaviour.
- **Trials with no response are left alone.** A trial whose request failed has
  nothing to grade.
- **A stored judge score cannot rescue a re-graded trial.** The judge was shown
  the old deterministic score; once that score moves, the judge verdict loses
  its power to overturn a failure until `--judge` asks again. Re-scoring twice
  gives the same result as once.

```bash
sllmb rescore --results results/m_raw_results_judged.json --in-place --judge
sllmb rescore --results results/m_raw_results.json --sandbox docker
```

---

## `migrate`

Brings stored files in line with the current bank: drops trials for retired
tasks, carries trials across a module rename, and re-stamps the bank hash.
Nothing is re-graded; use `rescore` for that.

```
sllmb migrate (--results FILE | --results-dir DIR) [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--results` | — | One file. |
| `--results-dir` | — | Every raw and judged file at the top level of this directory. |
| `--tasks-dir` | discovered `tasks/` | Bank to migrate against. |
| `--in-place` | off | Overwrite the inputs (default writes `*_migrated.json`). |
| `--dry-run` | off | Report what would change; write nothing. |

```bash
sllmb migrate --results-dir results --dry-run
sllmb migrate --results-dir results --in-place
```

---

## `probe`

Screens **one candidate task** against a declared weak/mid/strong trio before
it enters the bank. The candidate lives in a scratch bank at
`PROBE_DIR/tasks/<module>.yaml` (default `.scratch/probe/`, gitignored), so
`tasks/` is untouched until you promote a survivor by hand.

```
sllmb probe --module MODULE --task TASK_ID [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--module` | required | Module the candidate belongs to. |
| `--task` | required | Candidate task id. |
| `--reprobe` | off | Top a 3-trial probe up to 5 without re-running the first three, to break a one-trial tie. |
| `--top-up` | off | Reuse the last cycle and re-run only models that came back short of `PROBE_TRIALS`. |
| `--override` | off | Run past the per-construct cycle cap. Recorded permanently in the log. |

| Verdict | Rule (`w`, `m`, `s` = pass fraction of weak, mid, strong) |
|---|---|
| `INVALID` | Too few scorable trials or an infra error. The numbers are unreadable, not bad. |
| `REJECT: inverted` | Any rung goes backwards. |
| `REJECT: broken` | `s < 2/3`: the strongest model cannot pass it either. |
| `REJECT: saturated` | All three pass everything. |
| `REJECT: no_signal` | Monotone with no gap. |
| `REPROBE` | Monotone, `s ≥ 2/3`, one trial of separation. Use `--reprobe`. |
| `ACCEPT` | Monotone, `s ≥ 2/3`, `w ≤ 1/3`, `s − w ≥ 2/3`. |

Two design points:

- **The order is declared, never derived.** A model order computed from the
  same run cannot detect a task that ranks models backwards, which is the
  failure worth catching.
- **It prints the deterministic and the judged verdict side by side.** They
  can disagree, and the probe shows the disagreement rather than picking one.

This is a high-recall screen, not a significance test: at three trials per
model only a perfect 1.00/0.00 split reaches p = 0.05. A cap of
`PROBE_MAX_CYCLES` scoring cycles per construct keeps it from becoming a way to
fit tasks to three particular models, and every cycle is appended to
`PROBE_DIR/log.jsonl` with a hash of the candidate.

---

## Configuration

All variables can be set in `.env` or the environment. CLI options override
them.

**Model under test (`run`)**

| Variable | Default | Description |
|---|---|---|
| `BENCH_ENDPOINT` | `http://localhost:11434/v1` | Endpoint to benchmark. |
| `BENCH_MODEL` | `qwen3:8b` | Model name. |
| `BENCH_API_KEY` | — | Bearer token for the endpoint, if it needs one. |
| `BENCH_CONCURRENCY` | `2` | Concurrent requests. |
| `BENCH_TRIALS` | `1` | Trials per task. |
| `BENCH_TEMPERATURE` | unset | Only sent when set. |
| `BENCH_THINKING` | unset | Only sent when set. |
| `BENCH_SEED` | `0` | Base seed; `-1` sends none. |
| `BENCH_MAX_TOKENS` | `8192` | Cap for any task without a module or task cap. Every module in the current bank has one, so this only matters together with `--max-tokens`. |
| `BENCH_OUTPUT_DIR` | `./results` | Where result files go. |

Per-module completion caps (`_MODULE_MAX_TOKENS` in `modules/base.py`):
`tools` 4096, `adversarial`/`code`/`multi_turn_if` 6144, `format`/`knowledge`
8192, `long_context` 12288. A few tasks set a larger cap of their own. The cap
is per request, so a multi-turn episode can use it on every turn.

**Timeouts and retries (`run`)**

| Variable | Default | Description |
|---|---|---|
| `BENCH_TIMEOUT` | `60` | Floor for the per-request read timeout, in seconds. The actual timeout is derived per request (below). |
| `BENCH_MIN_GENERATION_TOK_S` | `20` | Throughput floor used to derive the read timeout from `max_tokens`. |
| `BENCH_PREFILL_ALLOWANCE` | `120` | Seconds added on top for prompt processing. |
| `BENCH_CONNECT_TIMEOUT` | `10` | Connect timeout, kept short so a dead endpoint fails fast. |
| `BENCH_MAX_ATTEMPTS` | `3` | Attempts per request. Read timeouts are never retried. |
| `BENCH_RETRY_BACKOFF` | `1.0` | Base backoff between attempts, in seconds. |
| `BENCH_SANITY_CHECK_AFTER` | `3` | Abort when the first N trials all return nothing. `0` disables. |

The read timeout for a request is `max(BENCH_TIMEOUT, max_tokens /
BENCH_MIN_GENERATION_TOK_S + BENCH_PREFILL_ALLOWANCE)`.

**Code sandbox (`run`, `rescore`)**

| Variable | Default | Description |
|---|---|---|
| `BENCH_SANDBOX_BACKEND` | `auto` | `auto`, `docker`, `podman`, `bwrap`, `sandbox-exec` or `rlimit`. |
| `BENCH_SANDBOX_MEMORY_MB` | `256` | Memory cap in MB. |
| `BENCH_ALLOW_UNSANDBOXED` | `false` | Allow the rlimit-only fallback. |
| `BENCH_CODE_TIMEOUT` | `5.0` | Per-execution timeout in seconds. |

**Judge (`judge`, `rescore --judge`)**

| Variable | Default | Description |
|---|---|---|
| `JUDGE_ENDPOINT` | `https://generativelanguage.googleapis.com/v1beta/openai` | Judge endpoint (OpenAI-compatible). |
| `JUDGE_MODEL` | `gemini-2.5-flash` | Judge model. |
| `JUDGE_API_KEY` | — | Sent as `Authorization: Bearer`. |
| `JUDGE_CONCURRENCY` | `4` | Concurrent judge calls (one per module). |
| `JUDGE_MAX_TOKENS` | `32768` | Completion cap for one module's verdicts. |
| `JUDGE_TIMEOUT` | `120` | Per-call timeout in seconds. |
| `JUDGE_MAX_ATTEMPTS` | `6` | Attempts per call. A lost call loses a whole module. |
| `JUDGE_RETRY_BACKOFF` | `4.0` | Base backoff, about two minutes of total tolerance. |
| `JUDGE_TEMPERATURE` | `0.0` | Judge sampling temperature. |

**Probe loop (`probe`, `items --cohort`)**

| Variable | Default | Description |
|---|---|---|
| `PROBE_MODELS` | `qwen3.5-4b,gemma-4-12b,qwen3.6-27b` | The trio, weak to strong. Never re-sorted. |
| `PROBE_DIR` | `.scratch/probe` | Scratch bank, per-model runs and the cycle log. |
| `PROBE_TRIALS` | `3` | Trials per model per cycle. |
| `PROBE_MAX_CYCLES` | `3` | Scoring cycles per construct. `INVALID` cycles do not count. |
| `PROBE_CYCLE_TIMEOUT` | `600` | Abort a cycle after this many seconds. Raise it for 5-trial cycles on slow models. |
| `PROBE_FLEET_ORDER` | — | Optional full-fleet order for the size-inversion check. When unset, the order comes from `models.yaml`. |
