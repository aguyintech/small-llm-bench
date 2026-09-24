# Reference panel

The published fleet, so a new model can be placed on the board without
re-running every other model.

```sh
sllmb items --results-dir results/reference --judged
sllmb leaderboard --results-dir results/reference
```

To add your own model to the comparison, run it into this directory with the
**same settings every row here used** and regenerate the board:

```sh
sllmb run --model <name> --trials 3 --thinking --concurrency 1 \
    --output results/reference/<name>_raw_results.json
sllmb judge --results results/reference/<name>_raw_results.json
sllmb leaderboard --results-dir results/reference
```

`leaderboard` flags any row whose run settings differ from the most common
ones (`comparability_mismatch`). A flagged row is still shown and still
ranked, so read the flag before reading the rank.

## What "same settings" means

Every row in this panel was produced with:

- the full profile at 3 trials,
- the bank's own per-module and per-task token caps (no `--max-tokens`),
- `--thinking`,
- the server's default sampler with the default seed,
- concurrency 1,
- the same bench version and task bank hash.

The leaderboard checks all of these except concurrency, which affects speed
columns only. The rule is strict because pass^k is a statement about a
sampler: an earlier fleet was run at a mix of 8192 and 16384 token caps with
the top-ranked model on the lower one.

Model sizes come from `models.yaml` in the repository root. A model absent
from it scores normally and shows `?` in the params column; adding an entry is
a one-line change.
