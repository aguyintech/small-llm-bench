"""Settings loaded from environment variables and the .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchSettings(BaseSettings):
    """Configuration for the benchmark runner (BENCH_* env vars)."""

    model_config = SettingsConfigDict(
        env_prefix="BENCH_", env_file=".env", extra="ignore"
    )

    endpoint: str = "http://localhost:11434/v1"
    model: str = "qwen3:8b"
    concurrency: int = 2
    # FLOOR for the per-request read timeout, not the whole budget. The real
    # value is derived from the request's own max_tokens (see
    # ChatClient.read_timeout_for): a bank with a 16k cap and a fleet spanning
    # 21-264 tok/s cannot be served by one number. Measured 2026-09-08: the
    # long_context 12288 cap needs 182s on gemma-4-12b and this used to be set
    # to 180, so roughly one trial in five died two seconds short.
    timeout: float = 60.0
    # Connect separately, and briefly. A dead endpoint should fail in seconds
    # even when a legitimate generation is allowed ten minutes; one global
    # timeout has to choose between those and gets one of them wrong. With
    # this split, `sanity_check_after` can still abort a misconfigured run
    # quickly instead of waiting out three full generation budgets.
    connect_timeout: float = 10.0
    # Throughput floor used to derive the read timeout. The slowest sustained
    # rate measured across the reference fleet is gemma-4-31b at 20.8 tok/s
    # (p5); 20 is that rounded down, so the derived timeout covers the whole
    # panel rather than the median of it.
    min_generation_tok_s: float = 20.0
    # Added on top of the derived generation time. Peak prefill measured on the
    # fleet is 94.5s (gemma-4-31b on a 32k haystack).
    prefill_allowance: float = 120.0
    max_attempts: int = 3
    retry_backoff: float = 1.0
    output_dir: Path = Path("./results")
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float | None = None
    thinking: bool | None = None
    trials: int = 1
    # Base seed sent with every request as `seed`, offset per trial so the k
    # trials of a task are k different samples rather than k copies of one.
    # Servers that ignore the field are unaffected; those that honour it make
    # a run reproducible, which pass^k otherwise cannot be — the metric
    # measures a sampler, and until v1.0 nothing recorded which sampler.
    # Set to -1 to send no seed at all.
    seed: int = 0
    sandbox_backend: str = "auto"   # auto|docker|podman|bwrap|sandbox-exec|rlimit
    sandbox_memory_mb: int = 256
    allow_unsandboxed: bool = False
    code_timeout: float = 5.0
    # Abort the run when the first N trials to finish ALL return nothing (a
    # transport/parse error, or an empty completion with no tool calls). That
    # is the signature of a misconfigured endpoint — wrong model id, server
    # down, no tool API — and without this the whole ~30-minute budget is spent
    # measuring it. Deliberately gated on empty output rather than on score, so
    # a model that simply fails everything still runs to completion. 0 disables.
    sanity_check_after: int = 3


class ProbeSettings(BaseSettings):
    """Configuration for the candidate-task probe loop (PROBE_* env vars).

    ``models`` is a comma-separated list in DECLARED weak-to-strong order. The
    verdict never re-sorts it — see ``analysis.probe_verdict`` for why a
    data-derived order cannot detect an inverted task.
    """

    model_config = SettingsConfigDict(
        env_prefix="PROBE_", env_file=".env", extra="ignore"
    )

    # v0.14: the 2.6B floor slot was dropped. Three of the four ACCEPTs the
    # previous trio produced split only 2.6B-vs-4B (mid and strong both 1.0), so
    # the screen was certifying tasks that carry no information about the gap the
    # bank actually lacks — tst_51/tst_55/tst_57 were banked that way.
    models: str = "qwen3.5-4b,gemma-4-12b,qwen3.6-27b"
    # Scratch root. Everything the loop writes lives here, and `.scratch/` is
    # gitignored — a probe run must never write into results/, which items and
    # leaderboard glob for *_raw_results*.json.
    dir: Path = Path(".scratch/probe")
    trials: int = 3
    # Cap per construct. Cycle 1 is the idea, 2 fixes a real defect, 3 is the
    # last shot; past that you are fitting the trio rather than finding a
    # construct. INVALID cycles do not count.
    max_cycles: int = 3
    # Abort a cycle that outruns this. The per-request timeout does not cover
    # a model that generates 20k characters on every trial.
    cycle_timeout: float = 600.0
    # The full fleet in DECLARED weak-to-strong order, for the
    # non-monotonicity check (discovery-run-brief.md §3.2). Declared for the
    # same reason `models` is: an order derived from observed scores makes
    # every inversion vanish by construction, which is precisely the finding
    # being looked for. MoEs are placed by ACTIVE parameters, not total — the
    # bank already ranks them that way (gemma-4-26b-a4b lands below
    # gemma-4-12b on its 4B active, CHANGELOG v0.15).
    #
    # Empty by default: an unmaintained order is worse than none, because a
    # stale rung reports inversions that are really just a mislabelled ladder.
    fleet_order: str = ""

    def model_list(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]

    def fleet_list(self) -> list[str]:
        return [m.strip() for m in self.fleet_order.split(",") if m.strip()]


class JudgeSettings(BaseSettings):
    """Configuration for the LLM judge (JUDGE_* env vars)."""

    model_config = SettingsConfigDict(
        env_prefix="JUDGE_", env_file=".env", extra="ignore"
    )

    endpoint: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    concurrency: int = 4
    timeout: float = 120.0
    # Batched judging emits one verdict (score + 2-3 sentence reasoning) per
    # trial; a whole module's trials share one reply. 8192 truncated the last
    # entries on verbose modules (format/adversarial); 16384 still truncated
    # the tail of the knowledge module on qwythos-9b — so give the reply
    # ample headroom. Override with JUDGE_MAX_TOKENS.
    max_tokens: int = 32768
    temperature: float = 0.0
    # Gemini answers 503 in bursts when overloaded. A module whose call
    # exhausts its attempts loses every verdict in that batch, so judging
    # retries longer and slower than the bench runner does (runner defaults:
    # 3 attempts / 1.0s). 6 attempts at 4s base backoff spans ~2min.
    max_attempts: int = 6
    retry_backoff: float = 4.0
