# Changelog

How small-llm-bench got to where it is. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), condensed to the net
effect of each version. The full engineering log, with the measurements behind
every decision and the ones that were later reversed, is kept in
[docs/history/engineering-log.md](docs/history/engineering-log.md).

**Scores do not compare across versions.** Nearly every release below changed
the task bank or a scoring rule. Use `sllmb migrate` and `sllmb rescore` to
bring an old results file forward where the change allows it, and re-run where
it does not.

## [1.0.0] — 2026-09-24

First public release: 39 tasks over 7 modules, a 23-model reference fleet, and
a pre-publication audit of the framework, the bank and every stored result.

### Scoring

- **Running out of tokens is a failure.** A truncated trial that did not loop
  (`incomplete`) used to be excluded, which by this point was lifting only the
  weakest models by deleting tasks they could not finish from their
  denominator. Caps were raised first (`tools` to 4096, several per-task
  caps), so what still truncates is a model that could not finish.
- **One `k` for the whole board.** Every row is scored at the same pass^k
  exponent, and a file's `k` is the most common trial count, so one dropped
  request can no longer relabel a whole file as pass^2.
- **Stateful tasks grade the change, not the end state.** `goal_match` credits
  the share of the gap between the seeded and expected state that the model
  closed, so doing nothing scores 0 rather than most of the credit.
- **Checks that could pass without testing anything now fail closed:** an
  empty reply against negative constraints, `section_contains` with no
  target, `table_rows_preserved` on a short table, and error detection that
  matched the word "error" anywhere in a tool result.
- **What gets graded is the answer.** `<think>` blocks are stripped in every
  shape, including unbalanced ones; only a terminal turn counts as the
  model's closing words; a truncated `multi_turn_if` turn scores 0 for that
  turn instead of being graded on its reasoning dump.
- **Loops are caught in any form**, including a character cycle with no
  whitespace (`20s20s20s…`), and the reasoning channel is recorded per turn so
  a cut turn can be diagnosed.
- **A tool call the server cannot parse is the model's failure**, not an infra
  error. The server's error body is kept in the record.
- **The code repair loop no longer reveals expected values**, which on a
  five-case task let a model pass by copying the report.
- **Three tasks stopped grading conversational habit.** `tst_35` is graded on
  structure instead of an exact summary string, and `mt_11`/`mt_21` no longer
  penalise a model for acknowledging the rules instead of re-pasting notes.
  Six prompts stopped asking for output nothing grades.

### Judge

- It can never overturn a hard gate (forbidden tool, blown call budget,
  detected loop, premature stop); a refused rescue is recorded in
  `judge_blocked_by`.
- A verdict is tied to the deterministic score the judge was shown. If
  `rescore` moves that score, the verdict cannot rescue the trial until it is
  re-judged. Re-scoring is idempotent.

### Harness

- **Reproducible runs.** `--seed` (default 0, offset per trial). Results record
  the sampler, sandbox backend, interpreter and served context window, and the
  leaderboard flags rows whose settings differ.
- **Context preflight.** `run` reads the served window from llama.cpp or
  llama-swap and refuses to start if a task cannot fit, instead of scoring
  rejected prompts as failures.
- **Crash recovery.** Every finished trial is checkpointed; re-running the same
  command resumes (`--resume`, on by default).
- **Timeouts derive from the token cap** and a read timeout is never retried.
  Paused time no longer counts toward run duration, and rows report summed
  trial time, which stays honest for runs assembled with `--only-new`.
- `--add-trials` no longer drops tasks from the headline, the `items` interval
  is computed over the same tasks as the headline, and one scorer exception no
  longer loses a whole run.
- `sllmb` is installed as a short alias.

### Leaderboard

- Four tabs: **Leaderboard** (bars by score tier), **By tier**, **By params**
  (dense vs sparse per size bucket) and **Detailed** (the full table). Score
  tiers S to F are fixed cuts on the headline.
- A card per model with a radar of per-module pass^3, tasks excluded, run
  time, throughput, first-try code rate and judge coverage.
- Rows rank on the number they display: the judged headline, falling back to
  the deterministic one.
- **`models.yaml`**, a registry of total and active parameters taken from model
  cards, drives the size column and the size-inversion check. MoEs are
  compared by active parameters.

### Bank

- `lc_67` added: who owns a work item after a later turn in a meeting
  transcript reverses it. The only long-context task whose answer is not
  literally in the document. `lc_31` retired to pay for it.

### Docs

- README rewritten as a landing page with screenshots; the CLI reference and
  the methodology moved to `docs/`.

## [0.17.0] — 2026-09-07

- **`tst_63`**, the first task that separates mid-size models (9–12B) from
  large ones (27B+): a queue of timestamped stock adjustments where a later
  request corrects an earlier figure, and the correction decides whether a
  conditional rule fires. Both mid models fail it and both large models pass.
  It needs a 20-turn and 16k-token budget, or the budget becomes the
  measurement.

## [0.16.0] — 2026-08-31

- **`items` can see the band it is cut for.** `band_discrimination` measures a
  task over a declared weak-to-strong cohort instead of the top and bottom
  thirds of whatever is loaded, and `band_legs` reads that cohort rung by
  rung. New classes: `floor_only` (separates only below the cohort) and
  `inverted_leg` (ranks some rung backwards). A fleet-wide check reports any
  module where a larger model scores lower.
- What it showed: the bank had no task a 12B fails that a 27B passes. That set
  the target for 0.17.
- A loop that starts late in a long response is now caught.
- `fm_81` added: a Portuguese summary with rules about what not to translate.

## [0.15.0] — 2026-08-30

- **The bank was cut from 59 to 37 tasks** and a run from 64 to 38 minutes.
  Every task that ranked models backwards or carried no signal was retired,
  and the number of statistically separable model pairs went up. Minimum
  tasks per module dropped to 4; the weight corridor is what keeps a small
  module from dominating.
- Removed with the cut: the `discovery` axis and the restraint tasks, which
  came in pairs and tracked post-training rather than size. Retired
  definitions live on in `tests/fixtures/retired_bank/`, where tests still
  use them.
- `rescore` treats a changed system prompt as a changed stimulus, and
  refreshes band and difficulty metadata from the bank.
- `expected.not_graded` tells the judge which criteria a task deliberately
  does not grade.
- `--reuse-params` adopts a recorded `--max-tokens` override.
- Finding: MoEs rank by active parameters. `gemma-4-26b-a4b` lands below
  `gemma-4-12b`.

## [0.14.0] — 2026-08-28

- **`sllmb probe`** screens a candidate task against a declared three-model
  trio in minutes, before it costs a full-fleet sweep. Verdicts are ACCEPT,
  REPROBE, REJECT (saturated, broken, inverted, no signal) or INVALID, capped
  at three cycles per construct with an append-only log.
- **`file_checks`**: document edits graded on structural invariants (tables
  well formed, rows preserved, a row at the right position, bullets well
  nested) instead of one byte-exact file.
- `run --tasks-dir` for scratch banks.
- `final_text_checks` now apply to stateful tasks too.
- Long-context documents get a unique prefix per task, so prefix caching
  cannot hide their prefill cost, and the generator is part of each trial's
  identity.
- `BENCH_SANITY_CHECK_AFTER` aborts a run whose first trials all come back
  empty.
- Three stateful tools tasks added (`tst_51`, `tst_55`, `tst_57`) and six
  saturated tasks retired.

## [0.13.0] — 2026-08-27

- **The headline became module-weighted pass^k.** It had been weighted by
  difficulty band, which is circular and, once the bands were measured
  honestly, left four tasks carrying 40% of the score. Bands are now a
  reporting axis; `--scheme band` and `--scheme legacy` remain for old files.
- **Weights are bounded by item count:** no module may carry more than 2× or
  less than 0.5× its share of the bank.
- The bank was rebuilt around what discriminated: ten tasks retired, sixteen
  added, and `data_extract` and `tool_arg_typing` folded into `format`
  (9 → 7 modules).
- A `multi_turn_if` turn can revoke an earlier rule.
- Code extraction reads every fenced block, not just the first.
- Per-module token caps were refit against measured usage across ten models.

## [0.12.0] — 2026-08-25

- **Scalar argument types are compared by value by default.** A strict type
  check was splitting models by vendor (chat-template serialisation), not by
  size. Strict typing remains where a schema declares it.
- **Truncation is split into `degenerate` (looping) and `incomplete`.**
- An explicit `--max-tokens` overrides every per-module cap and is recorded,
  so a run with a different budget never pools with a default one.
- The project got git history.

## [0.11.0] — 2026-08

- **More independent ways to fail:** a tool that never works and must be
  reported honestly (`always_fail`), a complex tool signature, integer
  arguments pinned to integers, and a call budget that fails thrashing
  (`max(2 × optimal, optimal + 3)` calls).
- Two code tasks allow no repair, so first-try skill reaches the headline, and
  the leaderboard gained a **1st-try code** column.
- **`items` pairwise separability**: wins–losses per model pair, an exact
  sign test, Holm-corrected, and an `=` badge on the board for rows it cannot
  separate.
- **The simulated tool world is part of a trial's identity** (`world_hash`),
  so `--only-new` cannot reuse trials recorded against a different world.
- `contains`/`not_contains` accept `value`, `any` and `all`, and an empty
  target fails closed.

## [0.10.0] — 2026-08

- **A stateful mock filesystem**, so the bench can grade an edit. With it,
  `unchanged_paths` (protect neighbouring files) and the first document-edit
  task.
- The four `tool_*` modules merged into **`tools`**, with an `axis` per task
  reported as sub-rows.
- New tasks from published benchmarks with measured spread in this size band:
  a needle that shares no words with its question (NoLiMa) and predicting a
  function's output (CRUXEval).
- **`sllmb migrate`** brings stored files in line with a changed bank.
- The judge stopped being asked about `format` and `knowledge`, where it had
  changed no verdicts.
- Ten tasks that measured nothing across a 13-model sweep retired.

## [0.9.0] — 2026-08

- **`sllmb rescore`** re-grades stored trials with the current scorer, without
  calling a model, and re-judges only what moved.
- Calling a forbidden tool is a hard zero on every path.
- A tool can reject a wrong argument on every call (`require_args`), instead
  of failing only the first call whatever its arguments.
- The judge can no longer demote a pass by echoing its score.
- Prefill and decode speed reported separately, from the server's own timings.

## [0.8.0] — 2026-08-21

Recovery-aware scoring.

- **`code` runs an execute-and-fix loop**: the model sees the real traceback
  or failing inputs and tries again. A pass at any attempt is a pass; the det
  score decays 1.00 / 0.85 / 0.70.
- **Asking a clarifying question in prose counts** the same as asking through
  `ask_user`. Which one the model used is reported, not scored.
- A prompt rejected for exceeding the served context is reported as
  **exceeded context**, not as a wrong answer.

## [0.7.0] — 2026-08-21

Judge integrity.

- **A judge can rescue a failure only by disagreeing with it**: its score must
  exceed the deterministic one and reach 0.95. Echoing the deterministic
  score had been turning near-misses into passes.
- Judge coverage is reported everywhere, `judge` exits 2 when a module ends
  unjudged, and a missing module falls back to its deterministic score
  instead of leaving the average.
- `items` labels each task and prints a per-model 95% interval.
- Result files record the task bank hash; the leaderboard flags rows scored
  on a different bank or config.

## [0.6.0] — 2026-08-20

- 26 tasks that every strong model passed and that did not track model
  strength were cut (66 → 45), and five hard tasks added.
- `text_answer_ok`: a task can accept a plain-text reply as reaching the goal.

## [0.5.0] — 2026-07-08

- Bank cut from 96 to 66 tasks, with ten new tasks aimed at separating models
  under 35B from each other. `full` became the default profile.

## [0.4.0] — 2026-07-07

- **`items`**: cross-model item analysis (pass rate, discrimination,
  flakiness, cost), the basis for every later pruning decision.
- Calibration bands (`anchor`/`mid`/`hard`/`frontier`) and `--profile`.
- Per-module token caps set from observed usage.

## [0.3.0] — 2026-07-06

- `--only-new` reuses recorded trials whose task and config are unchanged.
- Temperature and thinking are sent only when set, so the server's defaults
  apply otherwise.

## [0.2.0] — 2026-06-21

- Seven new modules (tool discovery, stateful tools, long context,
  adversarial, multi-turn instruction following, format, data extraction).
- pass^k with a reliability curve, and a judge-adjusted pass column beside the
  deterministic one.

## [0.1.0] — 2026-06-07

- Initial release: tool calling, agentic loops, code and knowledge modules; a
  simulated tool registry with injected failures; deterministic scorers; an
  optional batched LLM judge; `run`, `score`, `judge` and `compare`.

[1.0.0]: https://github.com/aguyintech/small-llm-bench/releases/tag/v1.0.0
