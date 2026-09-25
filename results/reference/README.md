# Reference panel

The published fleet: 23 models run on bench v1.0.0, 39 tasks × 3 trials each,
judged. It is here so you can see the board without running anything, check
the claims in the main README against the data, and place a new model among
these without re-running them. The board is published from this directory at
https://aguyintech.github.io/small-llm-bench/ on every push to `main`.

```sh
sllmb leaderboard --results-dir results/reference --open   # the board
sllmb items --results-dir results/reference --judged       # item analysis, intervals, pairwise tests
```

## What is in it

One `<model>_raw_results_judged.json` per model, about 1 MB each and 26 MB in
all. A judged file holds everything the raw run recorded (every trial's
prompt, conversation turns, tool calls, final state, score breakdown, token
counts and timings) plus the judge's score and reasoning, so no raw files are
shipped. Raw API responses were not saved for these runs.

The server address is replaced with `redacted`, in the run metadata and in
the text of a few server error messages. Nothing else was edited.

| | Models |
|---|---|
| 24B and up | qwen3.8-27b, gemma-4-31b, qwen3.6-27b-thinking-cap, qwen3.6-27b, muse-glimmer-30b, qwen3.6-35b-a3b (MoE, 3B active), ornith-1.5-35b (MoE, 3B active), gemma-4-26b-a4b (MoE, 3.8B active) |
| 7B – 23B | gemma-4-12b, qwen3.5-9b, neohorse-1-9b, gemma-4-e4b-it (MoE, 4B active), granite-4.2-8b, LFM2.5-8B-A1B (MoE, 1B active), Ling-3.0-Tiny (MoE, 1.3B active) |
| 3B – 6B | neohorse-1-4b, qwen3.5-4b, gemma-4-e2b-it (PLE, 2.3B active), spark-x2.5-4b, granite-4.2-3b |
| under 3B | minicpm5-2b, LFM2.5-2.6B, qwen3.5-0.8b |

Sizes come from [`models.yaml`](../../models.yaml), taken from model cards.
Fine-tunes are on the board next to their base model on purpose:
`qwen3.6-27b-thinking-cap` is a fine-tune of `qwen3.6-27b`, and the
`neohorse-1` models are post-trained from Qwen3.5. Whether a fine-tune beats
its base is one of the questions the board is for. Read the pairwise table in
`sllmb items` before answering it: at this bank size, most such gaps are ties.

## Adding your own model

Run it into this directory with the **same settings every row here used**,
then rebuild the board:

```sh
sllmb run --model <name> --trials 3 --thinking --concurrency 1 \
    --output results/reference/<name>_raw_results.json
sllmb judge --results results/reference/<name>_raw_results.json
sllmb leaderboard --results-dir results/reference --open
```

`leaderboard` flags any row whose run settings differ from the most common
ones (`comparability_mismatch`). A flagged row is still shown and still
ranked, so read the flag before reading the rank. If your model is not in
`models.yaml` it scores normally and shows `?` for size; adding an entry is a
one-line change.

## What "same settings" means

Every row in this panel was produced with:

- the full profile at 3 trials,
- the bank's own per-module and per-task token caps (no `--max-tokens`),
- `--thinking`,
- the server's default sampler with the default seed,
- concurrency 1,
- bench v1.0.0 and the same task bank hash.

The leaderboard checks all of these except concurrency, which affects only the
speed columns. It does not check the endpoint: where your server lives says
nothing about how the model was run. The rule is strict because pass^k is a
statement about a sampler: an earlier fleet was run at a mix of 8192 and 16384
token caps with the top-ranked model on the lower one.

## Keeping it current

The test suite reads this directory. It checks that every model here has a
registry entry, that every stored trial matches the current task bank, and
that re-scoring the panel with the current scorer moves no verdict. A change
that would silently alter the published numbers therefore fails CI. When a
change is meant to move them, re-run or re-score the affected models, replace
their files here, and say so in the CHANGELOG.
