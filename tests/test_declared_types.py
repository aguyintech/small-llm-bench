"""Declared-type conformance: graded where a schema says so, and nowhere else.

Pins the v0.11 audit's remedy across the v0.13 move. The strict scalar-type
rule used to be the silent default for every state value in `tools`, where it
governed five tasks inside the module carrying 48% of the weight and split
models by tool-call serialization convention rather than capability (gemma
0.93, LFM 0.67, qwen 0.04, ornith 0.00, with no relationship to size). Removing
those five tasks changed the #1 model.

v0.12 quarantined the property in a `tool_arg_typing` module. v0.13 dissolved
that module: honouring a declared type is structured-output discipline, which
is what `format` already measures, so `fm_35` carries the construct there.
tat_01 and tat_02 were retired at 0.97 with no discrimination, and tat_03 —
a 12-turn tool episode the format module has no path to run — moved to `tools`
as `axis: signature`.

The founding rule survives both moves: grade a type ONLY where a schema
declares it, so that getting it wrong is something a real API answers with a
400. Never invent a type requirement in prompt text alone.
"""

from __future__ import annotations

import json

import pytest

from small_llm_bench.models import TaskResult, ToolCall, TurnRecord
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.runner import all_modules
from small_llm_bench.scorer import MODULE_WEIGHT_PRESETS, score_format, score_task
from pathlib import Path

RETIRED_BANK = Path(__file__).resolve().parent / "fixtures" / "retired_bank"

# The only task in the bank that opts into strict scalar typing.
_STRICT_TASK = "tat_03"


def _result(task, calls, final_state=None):
    return TaskResult(
        task_id=task.id, module=task.module, prompt=task.prompt,
        turns=[TurnRecord(role="assistant",
                          tool_calls=[ToolCall(name=n, arguments=a)
                                      for n, a in calls])],
        final_state=final_state or {})


def _bank(tasks_dir):
    return {t.id: t for m in all_modules()
            for t in load_tasks(m.name, tasks_dir=tasks_dir)}


class TestModuleWiring:
    def test_the_quarantine_module_is_gone(self):
        assert "tool_arg_typing" not in {m.name for m in all_modules()}
        assert "tool_arg_typing" not in MODULE_WEIGHT_PRESETS["balanced"]

    @pytest.mark.parametrize("preset", sorted(MODULE_WEIGHT_PRESETS))
    def test_presets_still_sum_to_one(self, preset):
        """overall_score renormalizes anyway, but the printed Weight column is
        only readable if the convention holds."""
        assert sum(MODULE_WEIGHT_PRESETS[preset].values()) == pytest.approx(1.0)

    def test_exactly_one_task_opts_into_strict_types(self, tasks_dir):
        """Strictness is opt-in and stays that way. If it leaks back into the
        rest of the bank as a default, the vendor artefact returns with it."""
        strict = [t.id for t in _bank(tasks_dir).values()
                  if t.expected.get("strict_types")]
        assert strict == [_STRICT_TASK]


class TestToolSchemaTypes:
    def test_a_typed_arg_copied_straight_from_a_string_fails(self, tasks_dir):
        """tat_03: the values arrive as quoted strings in kv, but update_order
        declares order_id: integer and create_order declares total: number, so
        the model must CONVERT rather than copy. Passing them straight through
        is what a real API answers with a 400.

        This replaced a first version of tat_03 that demanded a quoted number
        through kv_set, whose schema declares `value` as "any type". That
        version reproduced the very artefact this construct exists to contain,
        splitting gemma/LFM 3/3 against qwen/ornith 0/3 — the same vendor line
        as the tst_* tasks it was meant to replace.
        """
        task = _bank(tasks_dir)[_STRICT_TASK]
        kept = {"target_order": "601", "reorder_total": "250"}
        others = {"id": 602, "customer": "Globex", "total": 80,
                  "status": "open"}
        cancelled = {"id": 601, "customer": "Acme", "total": 150,
                     "status": "cancelled"}
        calls = [("update_order", {"order_id": 601, "status": "cancelled"}),
                 ("create_order", {"customer": "Acme", "total": 250})]
        good = _result(task, calls,
                       {"orders": [cancelled, others,
                                   {"id": 603, "customer": "Acme",
                                    "total": 250, "status": "open"}],
                        "kv": kept})
        bad = _result(task, calls,
                      {"orders": [cancelled, others,
                                  {"id": 603, "customer": "Acme",
                                   "total": "250", "status": "open"}],
                       "kv": kept})
        assert score_task(task, good).success is True
        assert score_task(task, bad).success is False


class TestPromptSchemaTypes:
    """fm_35 grades the same property one turn at a time, against a schema the
    prompt states rather than one a tool declares."""

    _REFERENCE = {"version": 3, "build": "0042", "enabled": True,
                  "tags": ["beta"], "timeout_seconds": 2.5, "retries": 0}

    def _score(self, tasks_dir, payload):
        task = _bank(tasks_dir)["fm_35"]
        res = score_format(task.constraints, task.answer_type,
                           json.dumps(payload))
        return res.score >= 0.999

    def test_the_reference_object_passes(self, tasks_dir):
        assert self._score(tasks_dir, self._REFERENCE) is True

    @pytest.mark.parametrize("field,value,why", [
        ("build", 42, "a zero-padded build number coerced to an integer"),
        ("tags", "beta", "a single tag emitted bare instead of in an array"),
        ("version", "3", "an integer quoted"),
        ("enabled", "true", "a boolean quoted"),
        ("retries", False, "false where an integer zero was declared"),
    ])
    def test_no_uniform_encoding_passes(self, tasks_dir, field, value, why):
        """The values are chosen to pull in different directions: preferring
        native JSON types loses `build`, quoting everything loses the rest."""
        assert self._score(tasks_dir, {**self._REFERENCE, field: value}) is False, why


@pytest.mark.parametrize("stored", [302, "302"])
def test_either_encoding_passes_in_the_tools_module(tasks_dir, stored):
    """The control for the whole change.

    kv_set declares `value` as "any type", so on that field 302 and "302" are
    both valid and the choice between them is the chat template's, not the
    model's reasoning. Measured across the fleet the split is total — gemma and
    LFM store a string 27/27, qwen and ornith an integer ~21/27 — while every
    model, down to a 0.8B, emits a correct integer when a schema actually
    declares one. So neither encoding may be failed here.
    """
    # tst_21 was retired from the bank in v0.15 (0.93 pass, +0.22). The RULE it
    # pins is not retired: it is the module header's standing rule that a type
    # is graded only where a schema declares one.
    task = {t.id: t for t in load_tasks("tools", profile="full",
                                        tasks_dir=RETIRED_BANK)}["tst_21"]
    state = {
        "orders": [
            {"id": 301, "customer": "Wayne", "total": 95, "status": "cancelled"},
            {"id": 302, "customer": "Wayne", "total": 180, "status": "open"},
        ],
        "kv": {"pending_customer": "Wayne", "pending_total": "180",
               "new_order_id": stored},
    }
    calls = [("create_order", {"customer": "Wayne", "total": 180}),
             ("cancel_order", {"order_id": 301}),
             ("kv_set", {"key": "new_order_id", "value": stored})]
    assert score_task(task, _result(task, calls, state)).success is True
