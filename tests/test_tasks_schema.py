"""Validates that all YAML task files parse and match the expected schema."""

from __future__ import annotations

import pytest

from small_llm_bench.models import Task
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.modules.mock_registry import ALL_TOOL_SCHEMAS, TOOL_SCHEMAS

EXPECTED_COUNTS = {
    # 12 -> 13 in v0.17: tst_63 banked (a later request corrects an earlier
    # figure, which changes whether a conditional rebalancing rule fires).
    # First task in the bank to separate mid from large with BOTH mid models
    # agreeing: mid pooled 2/10 vs strong pooled 10/10, Fisher p=0.00071.
    "tools": (13, 4),
    "code": (5, 3),
    "knowledge": (4, 3),
    # 4 -> 5 in v0.16: fm_81 banked (glossary/proper-noun/enum translation plus
    # automation-actor attribution), probed twice at gap +0.40 then +0.67.
    "format": (5, 1),
    "multi_turn_if": (4, 1),
    # fast 3 -> 2 in v1.0: lc_31 retired (it duplicated lc_08's multi_key
    # construct at 1.5x the cost) and lc_67 banked in its place. lc_67 is
    # deliberately NOT in the fast profile — that profile is a smoke test and
    # this is the most expensive task in the module at 8.6 min/model on a 12B.
    "long_context": (4, 2),
    "adversarial": (4, 3),
}

# The floor every module has to clear to keep its weight. Below roughly this,
# a module score at pass^3 moves in visible steps and its share of the headline
# is mostly noise — which is what dissolved `data_extract` and
# `tool_arg_typing` (three tasks each) into `format` in v0.13.
#
# Lowered 6 -> 4 in v0.15, as a deliberate trade rather than a revised belief:
# the bank cost 64 min per model against a 30 min target, and 28 of its 59
# tasks carried no ranking signal. Dropping them left adversarial, format and
# knowledge with only 2 discriminating items each, so the floor had to come
# down or those modules would have kept dead weight purely to satisfy it. The
# noise concern is unchanged and now applies: a 4-task module moves in 25%
# steps, so read those module scores as indicative and the headline as the
# measurement.
MIN_TASKS_PER_MODULE = 4

# Modules whose sole fast representative is medium-difficulty by label but a
# proven strong discriminator empirically (real pass-rate gap data from the
# reference model fleet) — the difficulty label undersells them, so the
# "fast subset must span to hard" invariant doesn't apply.
_FAST_HARD_INVARIANT_EXEMPT: set[str] = set()

# Modules that are intentionally all-hard-tier. tool_discovery and
# long_context say so in their YAML headers (their medium tasks ds_01/lc_04 were
# pruned in v0.5). multi_turn_if joined in v0.6: mt_03 and mt_05 were its only
# medium tasks and both were saturated (95%+ of models scored 3/3, negative
# discrimination), so the module is hard-only by design now.
# format joined in v0.10: fm_13 was its only medium task and carried +0.17
# discrimination across 13 models, so the module is hard-only by design now.
# tools spans four axes with their own difficulty spread, and its `call` axis is
# a single anchor — the invariant is meaningless at module level there.
# code joined in v0.15: cd_02 (binary search, band anchor) was its only `easy`
# task and went in the runtime cut, so the module is hard-only now.
_DIFFICULTY_SPAN_EXEMPT = {"long_context", "multi_turn_if", "format", "tools",
                           "code"}

# Scorer paths that NO task in the shipped bank exercises any more. Every entry
# is a live code path with zero coverage, which is how `final_text_checks` sat
# silently dead on stateful tasks until v0.15 — so the gap is listed, not
# assumed away, and the tests below read this instead of quietly dropping their
# assertions. Delete an entry the moment a task covers it again.
_UNCOVERED_SCORER_PATHS = {
    # kn_09 was the bank's only `answer_type: calibration` task. It graded
    # whether a model says "unknowable" to an unanswerable question, and it was
    # retired in the v0.15 runtime cut (+0.22 discrimination, 19.6 s/trial).
    # `_score_calibration` is now unreachable from the bank.
    "calibration",
    # fm_17 was the bank's only task carrying a `system_prompt`, so the
    # system-adherence construct — rules stated once in the system prompt and
    # checked on the reply — is no longer measured at all.
    "system_adherence",
    # kn_17 was the bank's only 10-option MMLU-Pro item, so `_score_multiple_
    # choice` is unreachable too and `test_multiple_choice_tasks_well_formed`
    # now passes vacuously. knowledge is four numeric tasks.
    "multiple_choice",
    # cd_34 was the only task ever to use `Task.support_code` — a driver that
    # makes a stateful object's answer depend on the calls before it. Retired
    # in the second v0.15 cut once a 12-model panel put it at +0.33, which
    # leaves the field declared and unreached.
    "support_code",
    # tst_35 was the only task with `accept_state`, and in v1.0 it stopped
    # grading its summary line byte-exact, so it no longer needs a spelling
    # variant to be passable. The mechanism is still exercised directly in
    # tests/test_v10_new_tasks.py::TestAcceptState — what is gone is any bank
    # task reaching it. A new task that needs per-key alternatives should
    # first ask whether it is grading prose that ought to be structural.
    "accept_state",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@pytest.fixture(params=sorted(EXPECTED_COUNTS))
def module_tasks(request, tasks_dir):
    """All tasks for one module, parametrized over every module."""
    return request.param, load_tasks(request.param, tasks_dir=tasks_dir)


def test_tasks_parse_and_validate(module_tasks):
    name, tasks = module_tasks
    assert all(isinstance(t, Task) for t in tasks)
    assert all(t.module == name for t in tasks)
    assert all(t.prompt for t in tasks)


def test_task_counts(module_tasks):
    name, tasks = module_tasks
    total, fast = EXPECTED_COUNTS[name]
    assert len(tasks) == total
    assert sum(t.fast for t in tasks) == fast


def test_every_task_has_valid_difficulty(module_tasks):
    _, tasks = module_tasks
    assert all(t.difficulty in VALID_DIFFICULTIES for t in tasks)


def test_every_module_spans_difficulty_tiers(module_tasks):
    """Each module should include more than one tier so the report discriminates."""
    name, tasks = module_tasks
    if name in _DIFFICULTY_SPAN_EXEMPT:
        return
    assert len({t.difficulty for t in tasks}) >= 2


def test_task_ids_unique(module_tasks):
    _, tasks = module_tasks
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))


def test_tool_tasks_reference_known_tools(tasks_dir):
    """Stateful tasks may also reach for the state-backed tools, so they are
    checked against the merged schema set."""
    for task in load_tasks("tools", tasks_dir=tasks_dir):
        assert task.tools, f"{task.id} has no tools"
        assert set(task.tools) <= set(ALL_TOOL_SCHEMAS)
        # The state-backed tools (list_files/read_file/write_file over
        # `initial_state`) are only wired up for tasks that seed a state; a task
        # without one must stick to the stateless mocks or it would call into an
        # empty world.
        if not task.initial_state:
            assert set(task.tools) <= set(TOOL_SCHEMAS), task.id
        has_target = (task.expected.get("tool_name")
                      or task.expected.get("goal_tool")
                      or task.expected.get("no_call")
                      # A task can be graded purely on the prose it returns
                      # plus the tools it must NOT reach for (tl_35: draft it,
                      # do not send it) — there is no goal call to name.
                      or task.expected.get("text_answer_ok")
                      or task.expected.get("expected_state")
                      # Or purely on structural invariants over the files it
                      # edited, for tasks whose instruction is broad enough that
                      # several outputs are correct (pf_01).
                      or task.expected.get("file_checks")
                      or task.parallel)
        assert has_target, f"{task.id} has no scoring target"


def test_every_tools_task_declares_its_axis(tasks_dir):
    """The axis drives the reported sub-rows; an unlabelled task vanishes from
    the breakdown while still counting toward the module."""
    # `signature` and `parallel` joined in v0.13: signature came in with
    # tat_03 when the tool_arg_typing module was dissolved, parallel with the
    # first tasks to reach scorer._score_parallel.
    valid = {"call", "loop", "state", "discovery", "signature", "parallel"}
    for task in load_tasks("tools", tasks_dir=tasks_dir):
        assert task.axis in valid, f"{task.id} axis={task.axis!r}"


def test_parallel_specs_reference_known_tools(tasks_dir):
    for task in load_tasks("tools", tasks_dir=tasks_dir):
        for spec in task.parallel:
            assert spec.get("tool_name") in TOOL_SCHEMAS


def test_code_tasks_have_function_and_cases(tasks_dir):
    for task in load_tasks("code", tasks_dir=tasks_dir):
        assert task.function_name
        assert 3 <= len(task.test_cases) <= 6, task.id
        # The prompt has to name what the model is asked to write. Usually that
        # is the graded entry point itself; where the task ships a driver in
        # support_code (cd_34), the entry point is the driver's name and the
        # prompt names the class it builds instead.
        if task.support_code:
            assert task.function_name in task.support_code, task.id
        else:
            assert task.function_name in task.prompt, task.id


def test_knowledge_tasks_have_valid_answer_type(tasks_dir):
    tasks = load_tasks("knowledge", tasks_dir=tasks_dir)
    for task in tasks:
        assert task.answer_type in ("numeric", "factual", "calibration",
                                    "multiple_choice")
        assert "answer" in task.expected
    have = sum(t.answer_type == "calibration" for t in tasks)
    if "calibration" in _UNCOVERED_SCORER_PATHS:
        assert have == 0, ("knowledge now has a calibration task again — drop "
                           "'calibration' from _UNCOVERED_SCORER_PATHS")
    else:
        assert have >= 1
    have_mc = sum(t.answer_type == "multiple_choice" for t in tasks)
    if "multiple_choice" in _UNCOVERED_SCORER_PATHS:
        assert have_mc == 0, ("knowledge has a multiple-choice task again — "
                              "drop it from _UNCOVERED_SCORER_PATHS")
    else:
        assert have_mc >= 1


def test_multiple_choice_tasks_well_formed(tasks_dir):
    for task in load_tasks("knowledge", tasks_dir=tasks_dir):
        if task.answer_type != "multiple_choice":
            continue
        assert len(task.choices) == 10, f"{task.id} should have 10 choices"
        assert task.expected["answer"] in list("ABCDEFGHIJ")


def test_state_axis_tasks_grade_the_final_state(tasks_dir):
    for task in load_tasks("tools", tasks_dir=tasks_dir):
        if task.axis != "state":
            continue
        assert task.tools, f"{task.id} has no tools"
        assert set(task.tools) <= set(ALL_TOOL_SCHEMAS)
        assert (task.expected.get("expected_state")
                or task.expected.get("file_checks")), (
            f"{task.id} grades neither the final state nor its file invariants")


def test_long_context_tasks(tasks_dir):
    tasks = load_tasks("long_context", tasks_dir=tasks_dir)
    assert len(tasks) >= 1
    assert all(t.tier == "hard" for t in tasks)
    for t in tasks:
        assert t.haystack.get("filler_tokens", 0) >= 1000
        assert "answer" in t.expected
        assert t.answer_type in ("numeric", "factual")


def test_every_module_fast_subset_has_a_hard_task(tasks_dir):
    """Fast scores stay reliable only if the fast subset still spans to hard."""
    for module in EXPECTED_COUNTS:
        if module in _FAST_HARD_INVARIANT_EXEMPT:
            continue
        fast_tasks = load_tasks(module, fast=True, tasks_dir=tasks_dir)
        assert any(t.difficulty == "hard" for t in fast_tasks), \
            f"{module} fast subset has no hard task"


def test_format_tasks_have_constraints_and_answer_type(tasks_dir):
    tasks = load_tasks("format", tasks_dir=tasks_dir)
    valid = {"json", "markdown", "constraint", "system_adherence"}
    for task in tasks:
        assert task.answer_type in valid, f"{task.id} bad answer_type"
        # Extraction tasks (de_*, merged in from data_extract in v0.13) are
        # graded per-field against expected.extracted rather than by the
        # constraint engine.
        assert task.constraints or task.expected.get("extracted"), \
            f"{task.id} has neither constraints nor expected.extracted"
    have = any(t.system_prompt for t in tasks)
    if "system_adherence" in _UNCOVERED_SCORER_PATHS:
        assert not have, ("format carries a system_prompt task again — drop "
                          "'system_adherence' from _UNCOVERED_SCORER_PATHS")
    else:
        assert have


def test_fast_filter(tasks_dir):
    fast_tasks = load_tasks("tools", fast=True, tasks_dir=tasks_dir)
    assert all(t.fast for t in fast_tasks)
    assert len(fast_tasks) == 4
    # every axis keeps a fast representative, or the fast profile silently
    # stops measuring one of them. `call` left the bank in v0.13 with ts_16;
    # `discovery` left in v0.15 with ds_11/ds_14/ds_15 — see the axis note in
    # tasks/tools.yaml for why the whole axis went rather than part of it.
    assert {t.axis for t in fast_tasks} == {"loop", "state", "signature",
                                            "parallel"}


def test_json_constraints_reference_keys_the_prompt_asks_for(tasks_dir):
    """A constraint on a key the prompt never mentions makes a task unpassable.

    Caught in v0.12: fm_22 gained `port` constraints while the matching prompt
    edit silently failed to apply (the YAML escapes its quotes, so the
    search-and-replace missed). Every model then failed a key it was never
    asked for. This is the same class of defect as a prompt demanding something
    nothing grades, and it is worth a cheap guard in both directions.
    """
    for module in ("format", "knowledge", "long_context"):
        for task in load_tasks(module, tasks_dir=tasks_dir):
            prompt = (task.prompt or "") + (task.system_prompt or "")
            for check in task.constraints or []:
                path = check.get("path") or ""
                root = str(path).split(".")[0]
                if not root or root.isdigit():
                    continue
                assert root in prompt, \
                    f"{task.id}: constrains '{root}' but the prompt never mentions it"
            for key in (c for check in task.constraints or []
                        for c in check.get("keys", []) or []):
                assert str(key) in prompt, \
                    f"{task.id}: requires key '{key}' the prompt never mentions"
