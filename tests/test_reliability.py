"""Tests for v0.3: tier weighting, pass^k reliability, binary success, reasoning."""

from __future__ import annotations

import pytest

from small_llm_bench.models import Task, TaskResult
from small_llm_bench.modules.base import (infer_band, infer_tier, load_tasks,
                                          message_text)
from small_llm_bench.reporter import (aggregate_by_task, capability_tier_scores,
                                      headline_overall)
from small_llm_bench.scorer import (TIER_BASELINE_WEIGHT, pass_hat_k, score_state,
                                    score_task, score_tool_simple, tier_weighted)
from pathlib import Path

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"


# --- pass^k math --------------------------------------------------------------

class TestPassHatK:
    def test_flaky_one_of_three_decays(self):
        assert pass_hat_k([(1, 3)], 1) == pytest.approx(1 / 3)
        assert pass_hat_k([(1, 3)], 2) == 0.0
        assert pass_hat_k([(1, 3)], 3) == 0.0

    def test_two_of_three(self):
        assert pass_hat_k([(2, 3)], 1) == pytest.approx(2 / 3)
        assert pass_hat_k([(2, 3)], 2) == pytest.approx(1 / 3)
        assert pass_hat_k([(2, 3)], 3) == 0.0

    def test_reliable_stays_flat(self):
        for k in (1, 2, 3):
            assert pass_hat_k([(3, 3)], k) == 1.0

    def test_averages_over_tasks(self):
        assert pass_hat_k([(3, 3), (0, 3)], 1) == 0.5


# --- tier weighting -----------------------------------------------------------

class TestTierWeighting:
    def test_incapable_model_floors_at_baseline_weight(self):
        # passes all baseline, zero hard
        assert tier_weighted(1.0, 0.0) == pytest.approx(TIER_BASELINE_WEIGHT)

    def test_capable_model(self):
        assert tier_weighted(1.0, 1.0) == 1.0

    def test_missing_tier_falls_back(self):
        assert tier_weighted(None, 0.5) == 0.5
        assert tier_weighted(0.8, None) == 0.8


# --- tier inference -----------------------------------------------------------

class TestTierInference:
    def test_baseline_modules(self):
        assert infer_tier("tools", {"axis": "call"}) == "baseline"
        assert infer_tier("knowledge", {}) == "baseline"
        assert infer_tier("format", {}) == "baseline"

    def test_hard_modules(self):
        assert infer_tier("tools", {"axis": "loop"}) == "hard"
        assert infer_tier("tools", {"axis": "state"}) == "hard"
        assert infer_tier("long_context", {}) == "hard"

    def test_code_scratch_vs_repair(self):
        assert infer_tier("code", {}) == "baseline"
        assert infer_tier("code", {"buggy_code": "def f(): pass"}) == "hard"

    def test_explicit_override(self):
        assert infer_tier("knowledge", {"tier": "hard"}) == "hard"

    def test_real_tasks_get_tiers(self, tasks_dir):
        code = {t.id: t.tier for t in load_tasks("code", tasks_dir=tasks_dir)}
        assert code["cd_21"] == "hard"          # repair (buggy_code)
        # The from-scratch side of the inference is pinned on cd_02, retired
        # from the bank in v0.15 as code's only `easy` task. Every code task
        # left carries buggy_code, so without this the "baseline" branch of
        # infer_tier has no real-task coverage at all.
        retired = {t.id: t.tier for t in load_tasks("code", profile="full",
                                                    tasks_dir=RETIRED_BANK)}
        assert retired["cd_02"] == "baseline"   # from scratch
        # tools used to span both tiers: a single `call` was one-shot baseline
        # capability while loop/state/discovery episodes compounded. ts_16 was
        # the last `call` task and was retired in v0.13 at 0.97 pass and +0.11
        # discrimination, so every tools task now infers "hard".
        tools = {t.axis: t.tier for t in load_tasks("tools", tasks_dir=tasks_dir)}
        assert "call" not in tools
        assert set(tools.values()) == {"hard"}
        # The rule itself is unchanged, and would still apply to a new call task.
        assert infer_tier("tools", {"axis": "call"}) == "baseline"


class TestBandInference:
    def test_falls_back_to_difficulty_not_tier(self):
        # format defaults to baseline tier for every task regardless of how
        # hard an individual task is; band must not inherit that mistake.
        assert infer_band("hard", {}) == "hard"
        assert infer_band("medium", {}) == "mid"
        assert infer_band("easy", {}) == "anchor"

    def test_explicit_override(self):
        assert infer_band("hard", {"band": "frontier"}) == "frontier"

    def test_real_tasks_get_bands(self, tasks_dir):
        fmt = {t.id: (t.difficulty, t.band)
              for t in load_tasks("format", tasks_dir=tasks_dir)}
        # a hard-difficulty format task without an explicit band must land in
        # the "hard" band, not "anchor" (which infer_tier's baseline default
        # would have produced).
        hard_untagged = [d for d, b in fmt.values() if d == "hard" and b == "hard"]
        assert hard_untagged, "expected at least one hard/untagged format task"


class TestProfileFiltering:
    def test_full_profile_returns_everything(self, tasks_dir):
        full = load_tasks("tools", profile="full", tasks_dir=tasks_dir)
        every = load_tasks("tools", tasks_dir=tasks_dir)
        assert len(full) == len(every)

    def test_fast_bool_still_works_as_legacy_alias(self, tasks_dir):
        by_fast_kwarg = load_tasks("tools", fast=True, tasks_dir=tasks_dir)
        by_profile = load_tasks("tools", profile="fast", tasks_dir=tasks_dir)
        assert {t.id for t in by_fast_kwarg} == {t.id for t in by_profile}


class TestPerTaskMaxTokens:
    def test_module_default_applied_when_unset(self, tasks_dir):
        from small_llm_bench.modules.base import _MODULE_MAX_TOKENS
        tasks = load_tasks("tools", tasks_dir=tasks_dir)
        default = _MODULE_MAX_TOKENS["tools"]
        # pf_01 overrides: its tool call carries a whole ~70-line document back
        # through write_file, which does not fit a cap fitted on short calls.
        # A truncated trial is DROPPED as unscorable rather than failed, so the
        # task would return an INVALID verdict that says nothing about it.
        # tst_63 overrides for the same reason, measured: at the 8192 module
        # cap, two of gemma-4-12b's five trials came back
        # `truncation_class: incomplete` and were EXCLUDED from the pass rate,
        # leaving too few scorable trials to read a verdict at all. The 27B
        # finishes it in 9 calls on ~3k tokens; the 12B needs 20 calls and
        # 8.8k+ and still misses, so the cap was hiding the result rather than
        # measuring it.
        # tat_03 joined in v1.0: LFM2.5-8B-A1B lost two of three trials to the
        # 2048 cap, and an `incomplete` trial is EXCLUDED rather than failed, so
        # the task was being read off a single trial.
        overrides = {"pf_01", "tst_63", "tat_03"}
        assert all(t.max_tokens == default for t in tasks if t.id not in overrides)
        assert all(t.max_tokens > default for t in tasks if t.id in overrides)

    def test_explicit_yaml_value_overrides_module_default(self, tmp_path):
        from small_llm_bench.modules.base import _MODULE_MAX_TOKENS
        (tmp_path / "tools.yaml").write_text(
            "tasks:\n"
            "  - id: capped\n"
            "    prompt: hi\n"
            "    max_tokens: 99\n"
            "  - id: uncapped\n"
            "    prompt: hi\n"
        )
        by_id = {t.id: t for t in load_tasks("tools", tasks_dir=tmp_path)}
        assert by_id["capped"].max_tokens == 99
        assert by_id["uncapped"].max_tokens == _MODULE_MAX_TOKENS["tools"]

    def test_an_unlisted_module_falls_back_to_the_global_default(self, tmp_path):
        """`adversarial` used to be the fixture for this, on the reading that a
        module absent from the table runs uncapped. It does not — it inherits
        BENCH_MAX_TOKENS at request time, a value nobody chose for it. v0.13
        gave every real module an explicit cap after measuring that the
        inherited 8192 was 15% too small for long_context and 25% too large for
        adversarial, so the case now needs a synthetic module to test at all.
        """
        (tmp_path / "unlisted.yaml").write_text(
            "tasks:\n  - id: u_01\n    prompt: hi\n")
        tasks = load_tasks("unlisted", tasks_dir=tmp_path)
        assert all(t.max_tokens is None for t in tasks)


# --- binary success -----------------------------------------------------------

class TestBinarySuccess:
    def test_tool_simple_perfect(self, make_call):
        r = score_tool_simple({"tool_name": "get_weather",
                               "required_args": {"city": "Tokyo"}},
                              [make_call("get_weather", city="Tokyo")])
        assert r.success is True

    def test_tool_simple_missing_arg(self, make_call):
        r = score_tool_simple({"tool_name": "get_weather",
                               "required_args": {"city": "Tokyo"}},
                              [make_call("get_weather")])
        assert r.success is False

    def test_state_policy_refusal_success(self):
        state = {"balance": 500, "refunds": []}
        r = score_state({"unchanged": ["balance"],
                         "expected_state": {"balance": 500, "refunds": []}},
                        state, state, [])
        assert r.success is True

    def test_knowledge_success_via_score_task(self):
        task = Task(id="k", module="knowledge", prompt="2+2?",
                    answer_type="numeric", expected={"answer": 4})
        result = TaskResult(task_id="k", module="knowledge", prompt="x",
                            response_raw="The answer is 4")
        scored = score_task(task, result)
        assert scored.success is True
        bad = TaskResult(task_id="k", module="knowledge", prompt="x",
                         response_raw="It is 7")
        assert score_task(task, bad).success is False


# --- reasoning-model handling -------------------------------------------------

class TestMessageText:
    def test_plain_content(self):
        assert message_text({"content": "hello"}) == "hello"

    def test_reasoning_fallback_when_content_empty(self):
        msg = {"content": "", "reasoning_content": "the code is 4827"}
        assert message_text(msg) == "the code is 4827"

    def test_strips_think_block(self):
        msg = {"content": "<think>hmm let me see</think>final answer"}
        assert message_text(msg) == "final answer"


# --- reporter aggregation -----------------------------------------------------

def _res(task_id, tier, success, module="tools"):
    return TaskResult(task_id=task_id, module=module, prompt="p", tier=tier,
                      success=success, det_score=1.0 if success else 0.0)


class TestReporterAggregation:
    def test_groups_trials_by_task(self):
        results = [_res("a", "hard", True), _res("a", "hard", False),
                   _res("a", "hard", True)]
        by_task = aggregate_by_task(results)
        assert by_task["a"]["passes"] == 2
        assert by_task["a"]["n"] == 3

    def test_capability_tiers_and_headline(self):
        results = [_res("b1", "baseline", True, "knowledge"),
                   _res("b2", "baseline", True, "knowledge"),
                   _res("h1", "hard", False), _res("h2", "hard", False)]
        tiers = capability_tier_scores(results)
        assert tiers["baseline"]["pass1"] == 1.0
        assert tiers["hard"]["pass1"] == 0.0
        # all baseline pass, all hard fail -> floor at baseline weight
        assert headline_overall(results, scheme="legacy") == pytest.approx(TIER_BASELINE_WEIGHT)

    def test_headline_pass_k_penalises_flaky_tasks(self):
        # one hard task passes 2 of 3 trials: solid at pass^1, zero at pass^3.
        results = [_res("b1", "baseline", True, "knowledge")] * 3 + [
            _res("h1", "hard", True), _res("h1", "hard", True),
            _res("h1", "hard", False)]
        # pass^1: hard tier = 2/3; pass^3: hard tier = 0 (never all-3-pass).
        k1 = headline_overall(results, 1, scheme="legacy")
        k3 = headline_overall(results, 3, scheme="legacy")
        assert k3 < k1
        assert k3 == pytest.approx(TIER_BASELINE_WEIGHT)  # baseline 1.0, hard 0.0
