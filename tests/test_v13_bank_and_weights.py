"""v0.13: the bank earns its weights, and the weights come from a stated claim.

Two measurements drove this version and these tests pin both.

The leaderboard used to sort by `band_weighted(pass^k)`, so MODULE_WEIGHT_PRESETS
never touched the ranking — it only fed a display column. And the bands it did
depend on were wrong: judged against the project's own published thresholds, 43
of 55 tasks sat in the wrong band, 22 of them because they never set `band` and
inherited "hard" from a difficulty fallback.

Retagging the bands honestly turned out to be worse than leaving them: it left
four tasks carrying 0.40 of the score, where one task flip moved the headline
ten points. That is the failure mode these tests exist to prevent, generalised
into a rule — weight must stay coupled to item count.
"""

from __future__ import annotations

import pytest

from small_llm_bench.migrate import migrate_bench
from small_llm_bench.models import BenchMeta, BenchResult, TaskResult
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.runner import all_modules
from small_llm_bench.scorer import (MODULE_WEIGHT_PRESETS, extract_code,
                                    score_tool_simple)
from pathlib import Path

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"

# Below roughly this, a module score at pass^3 moves in visible steps and its
# share of the headline is mostly noise. `data_extract` and `tool_arg_typing`
# had three tasks each and were dissolved into `format` for exactly that.
#
# Lowered 6 -> 4 in v0.15 as a runtime trade, not a change of view — see the
# same constant in test_tasks_schema.py for the reasoning. The corridor below
# is what actually keeps a thin module from dominating: at 4 tasks in a 40-task
# bank a module's share is 0.10, so its weight may not exceed 0.20.
MIN_TASKS_PER_MODULE = 4

# How far a weight may sit from the module's share of the bank. The band scheme
# had no such rule and reached 0.40 on four tasks.
MIN_RATIO, MAX_RATIO = 0.5, 2.0


@pytest.fixture
def bank(tasks_dir):
    return {m.name: load_tasks(m.name, tasks_dir=tasks_dir, profile="full")
            for m in all_modules()}


class TestBankShape:
    def test_every_module_clears_the_reliability_floor(self, bank):
        thin = {name: len(ts) for name, ts in bank.items()
                if len(ts) < MIN_TASKS_PER_MODULE}
        assert not thin, f"modules below the floor: {thin}"

    def test_the_bank_keeps_a_spread_of_anchors(self, bank):
        """An anchor is the harness-health canary: near-universal pass, so a
        broken run reads as a broken run rather than as a capability result.

        Deliberately NOT one per module. Bands come from measurement, and
        long_context, multi_turn_if and adversarial have no task measured at
        or above the 95% anchor threshold — their easiest are 0.90, 0.80 and
        0.93. Labelling one anyway would put a claim in the file that the data
        does not support, which is the habit v0.13 exists to break. The gap is
        real and the fix is a genuinely easy task in those modules, not a
        relabelling.
        """
        anchors = {name: [t.id for t in tasks if t.band == "anchor"]
                   for name, tasks in bank.items()}
        with_anchor = {n: ids for n, ids in anchors.items() if ids}
        assert len(with_anchor) >= 3, anchors
        assert sum(len(ids) for ids in with_anchor.values()) >= 5, anchors
        # And they stay canaries: a bank that is mostly anchors measures nothing.
        total = sum(len(ts) for ts in bank.values())
        assert sum(len(ids) for ids in with_anchor.values()) <= total // 4

    def test_the_dissolved_modules_are_gone_everywhere(self, bank):
        assert set(bank) == set(MODULE_WEIGHT_PRESETS["balanced"])
        assert "data_extract" not in bank
        assert "tool_arg_typing" not in bank

    def test_format_holds_nothing_it_cannot_run(self, bank):
        """FormatModule sends one message with no tools and no state. tat_03
        could not follow the rest of tool_arg_typing here for that reason, and
        nothing else may be filed here by mistake either."""
        for task in bank["format"]:
            assert not task.tools, task.id
            assert not task.initial_state, task.id
            assert not task.conversation, task.id


class TestWeights:
    @pytest.mark.parametrize("preset", sorted(MODULE_WEIGHT_PRESETS))
    def test_presets_sum_to_one_over_exactly_the_modules_present(self, preset, bank):
        weights = MODULE_WEIGHT_PRESETS[preset]
        assert set(weights) == set(bank)
        assert sum(weights.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("preset", sorted(MODULE_WEIGHT_PRESETS))
    def test_no_weight_strays_far_from_its_share_of_the_bank(self, preset, bank):
        """The rule that keeps the weights arguable.

        They are set from a claim about what this benchmark measures, not from
        the board — but a claim that puts a third of the headline on six tasks
        is measuring noise however well argued it is.
        """
        total = sum(len(ts) for ts in bank.values())
        for name, weight in MODULE_WEIGHT_PRESETS[preset].items():
            share = len(bank[name]) / total
            ratio = weight / share
            assert MIN_RATIO <= ratio <= MAX_RATIO, (
                f"{preset}/{name}: weight {weight} is {ratio:.2f}x its "
                f"{share:.3f} share of the bank")


class TestTokenCaps:
    """The caps were refit in v0.13 against 10 models x 3 trials at 16384.

    They had never been measured against this fleet: every reference run to
    date passed an explicit --max-tokens, so the defaults were dead
    configuration. The first run that used them lost three tasks to truncation,
    with the model reasoning correctly and being cut off before it could answer.
    """

    def test_every_module_names_its_own_cap(self):
        """A module missing from the table does not run uncapped — it inherits
        BENCH_MAX_TOKENS, which is a fallback nobody chose for it. That is how
        long_context ended up 15% too small and adversarial 25% too large."""
        from small_llm_bench.modules.base import _MODULE_MAX_TOKENS
        missing = [m.name for m in all_modules() if m.name not in _MODULE_MAX_TOKENS]
        assert not missing, f"modules falling back to the global default: {missing}"

    def test_the_prose_modules_clear_the_measured_p99(self):
        """format and knowledge truncated one single-request reply in seven at
        4096. They are where a model without a separate thinking channel spends
        its budget reasoning before it reaches the answer."""
        from small_llm_bench.modules.base import _MODULE_MAX_TOKENS
        # p99 of the v0.13 calibration corpus, per module.
        measured_p99 = {"format": 6116, "knowledge": 6227, "long_context": 9760,
                        "adversarial": 5360}
        for module, p99 in measured_p99.items():
            assert _MODULE_MAX_TOKENS[module] >= p99, module


class TestMultiTurnTurnsAreGradable:
    """A turn is graded against every constraint accumulated so far, so a turn
    that requests nothing cannot be graded on the ones it inherits.

    mt_25's fourth turn originally just announced a word limit. gemma-4-12b
    answered "Understood, I will keep replies under 70 words" — the right answer
    to what was asked — and failed the inherited bullet and content checks, 0/3
    on a model near the top of the board. The same run showed the mirror
    problem two turns later: the model pruned a resolved risk from the list,
    which is correct standup behaviour, and was scored as having forgotten the
    rule. Both are fixed by asking for content on the rule turn and revoking
    the checks that stop being true.
    """

    _REPLIES = [
        "STANDUP: The API team has shipped caching, and mobile is waiting on review.",
        "STANDUP: API shipped caching. Mobile is waiting on review. Risk: review may run long.",
        "STANDUP:\n* The API team has shipped caching.\n* Mobile is waiting on review.\n"
        "* Risk: delay if the review runs long.",
        "STANDUP:\n* The API team has shipped caching.\n* Mobile is waiting on review.\n"
        "* Risk: delay if the review runs long.",
        "STANDUP:\n* The API team has shipped caching.\n* Mobile is waiting on review.\n"
        "* Design is waiting on infra for the migration.\n* Risk: delay if the review runs long.",
        # The risk resolved with the review, so the list legitimately shrinks.
        "STANDUP:\n* The API team has shipped caching.\n* Mobile review is complete.\n"
        "* Design is waiting on infra for the migration.",
        "STANDUP:\n* The API team has shipped caching.\n* Mobile review is complete.\n"
        "* Design is waiting on infra for the migration.\n\nFiled.",
    ]

    def test_a_correct_dialogue_passes_every_turn(self, bank):
        from small_llm_bench.models import TurnRecord
        from small_llm_bench.scorer import score_multi_turn_if
        task = next(t for t in bank["multi_turn_if"] if t.id == "mt_25")
        res = score_multi_turn_if(
            task.conversation,
            [TurnRecord(role="assistant", content=c) for c in self._REPLIES])
        assert res.breakdown["per_turn"] == [1.0] * 7, res.breakdown["per_turn"]
        assert res.success is True

    def test_the_turn_1_rules_are_still_what_it_measures(self, bank):
        """Turn 5 puts the banned word in the user's own message, four turns
        after the ban. That bait is the task; the bullet scaffolding is not."""
        from small_llm_bench.models import TurnRecord
        from small_llm_bench.scorer import score_multi_turn_if
        task = next(t for t in bank["multi_turn_if"] if t.id == "mt_25")
        assert "blocker" in task.conversation[4]["prompt"]
        baited = list(self._REPLIES)
        baited[4] = baited[4].replace("waiting on infra", "a blocker on infra")
        res = score_multi_turn_if(
            task.conversation,
            [TurnRecord(role="assistant", content=c) for c in baited])
        assert res.success is False
        assert res.breakdown["per_turn"][4] < 1.0


class TestScorerPathsHaveTasks:
    def test_the_parallel_path_is_reached_by_a_real_task(self, bank):
        """`Task.parallel` and scorer._score_parallel shipped in v0.6 and no
        task reached either until v0.13. A fully-implemented grading path with
        no tasks behind it is untested surface that reads as coverage."""
        parallel = [t for t in bank["tools"] if t.parallel]
        assert parallel, "no task exercises _score_parallel"
        for task in parallel:
            assert task.axis == "parallel", task.id

    def test_parallel_scoring_rejects_a_spurious_extra_call(self, make_call):
        """What stops tsp_01 from being passable by calling everything.

        Pinned on tsp_02, which was retired from the bank in v0.15 but kept as
        a fixture: the guard it exercises still protects tsp_01, which is very
        much live (and is now an anchor).
        """
        task = next(t for t in load_tasks("tools", profile="full",
                                          tasks_dir=RETIRED_BANK)
                    if t.id == "tsp_02")
        exact = [make_call(s["tool_name"], **s["required_args"])
                 for s in task.parallel]
        assert score_tool_simple(task.expected, exact,
                                 parallel=task.parallel).success is True
        extra = exact + [make_call("get_stock_price", symbol="EURUSD")]
        assert score_tool_simple(task.expected, extra,
                                 parallel=task.parallel).success is False

    def test_the_support_code_path_is_reached_by_a_real_task(self, bank):
        """Declared in v0.6, unused until cd_34 — and unreached again since.

        cd_34 was the only task that ever set `support_code`, and the second
        v0.15 cut retired it at +0.33 on a 12-model panel. The field is now
        declared and unreachable from the bank, which is registered in
        test_tasks_schema._UNCOVERED_SCORER_PATHS. The definition lives on in
        tests/fixtures/retired_bank/code.yaml, so the scorer path itself is
        still exercised — just not by anything that ships.
        """
        from tests.test_tasks_schema import _UNCOVERED_SCORER_PATHS
        have = any(t.support_code for t in bank["code"])
        if "support_code" in _UNCOVERED_SCORER_PATHS:
            assert not have, ("a bank task sets support_code again — drop it "
                              "from _UNCOVERED_SCORER_PATHS")
        else:
            assert have


class TestExtractCodeAcrossBlocks:
    def test_imports_and_function_in_separate_blocks_are_joined(self):
        """The cd_27 failure generalised: 9 of 9 trials that emitted the import
        passed and 0 of 21 that did not, and a model splitting the two across
        blocks lost for a formatting reason."""
        code = extract_code(
            "First the import:\n```python\nfrom shipping.rates import BASE_FEE\n```\n"
            "Then the function:\n```python\ndef f():\n    return BASE_FEE\n```\n")
        assert "from shipping.rates import BASE_FEE" in code
        assert "def f():" in code

    def test_a_non_python_block_beside_the_answer_is_dropped(self):
        code = extract_code(
            "```json\n{\"a\": 1}\n```\n```python\ndef f():\n    return 2\n```\n")
        assert code == "def f():\n    return 2"


class TestMigrateAcrossTheMerge:
    def _bench(self, module, task_id):
        meta = BenchMeta(model="m", endpoint="e", timestamp="t",
                         duration_seconds=1.0, bench_version="0.13")
        return BenchResult(meta=meta, results=[
            TaskResult(task_id=task_id, module=module, prompt="p")])

    @pytest.mark.parametrize("stored,task_id,now", [
        ("data_extract", "de_07", "format"),
        ("tool_arg_typing", "tat_03", "tools"),
    ])
    def test_a_moved_task_is_relabelled_not_dropped(self, stored, task_id, now,
                                                    tasks_dir):
        """Both old modules dissolved into `format` — except tat_03, which went
        to `tools`. migrate must route each stored trial by where its task
        actually lives now, or 1,650 saved trials are stranded."""
        bench = self._bench(stored, task_id)
        report = migrate_bench(bench, tasks_dir=tasks_dir)
        assert report.dropped == []
        assert report.relabelled == 1
        assert bench.results[0].module == now

    def test_a_retired_task_leaves_no_trace(self, tasks_dir):
        bench = self._bench("tool_arg_typing", "tat_01")
        report = migrate_bench(bench, tasks_dir=tasks_dir)
        assert report.dropped == ["tool_arg_typing:tat_01"]
        assert bench.results == []
