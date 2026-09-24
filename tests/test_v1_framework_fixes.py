"""v1.0: defects found in the pre-publication audit, one test each.

Every case here is a number the framework reported that was wrong, not a
feature. They share one shape — a check that could not fail, an instrument
that was never read, or a verdict derived from a different quantity than the
one printed beside it — so they are kept together rather than filed by module.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from small_llm_bench.judge import apply_judge_verdict, hard_gate_failure
from small_llm_bench.models import Task, TaskResult, TurnRecord
# Aliased: pytest tries to collect a bare `TestCase` as a test class.
from small_llm_bench.models import TestCase as Case
from small_llm_bench.modules.base import hit_length_cap
from small_llm_bench.modules.multi_turn_if import MultiTurnIfModule
from small_llm_bench.scorer import (_check_constraints, _eval_file_check,
                                    _file_check_label, _final_answer,
                                    score_file_checks)


class _CappedClient:
    """A client whose every reply is cut off at the token cap."""

    save_responses = False

    def __init__(self, content: str = "") -> None:
        self.content = content

    async def chat(self, messages, tools=None, max_tokens=None):
        return {"choices": [{"message": {"role": "assistant",
                                         "content": self.content},
                             "finish_reason": "length"}],
                "usage": {"completion_tokens": 10}}


class TestMultiTurnIfRecordsTruncation:
    """The module that carries 0.13 of the headline never read its own cap."""

    def _task(self) -> Task:
        return Task(id="mt_x", module="multi_turn_if", prompt="p",
                    conversation=[{"prompt": "one"}, {"prompt": "two"}])

    def test_a_truncated_turn_sets_the_flag(self):
        result = asyncio.run(
            MultiTurnIfModule().run_task(_CappedClient("cut off mid-"),
                                         self._task()))
        assert result.truncated is True

    def test_an_untruncated_run_leaves_it_clear(self):
        class Client(_CappedClient):
            async def chat(self, messages, tools=None, max_tokens=None):
                return {"choices": [{"message": {"role": "assistant",
                                                 "content": "done."},
                                     "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 3}}

        result = asyncio.run(MultiTurnIfModule().run_task(Client(), self._task()))
        assert result.truncated is False

    def test_truncation_in_any_turn_taints_the_episode(self):
        """Not just the last turn: an early cut-off breaks the history that
        every later turn is generated from."""
        class Client(_CappedClient):
            def __init__(self):
                self.n = 0

            async def chat(self, messages, tools=None, max_tokens=None):
                self.n += 1
                reason = "length" if self.n == 1 else "stop"
                return {"choices": [{"message": {"role": "assistant",
                                                 "content": "text"},
                                     "finish_reason": reason}],
                        "usage": {"completion_tokens": 3}}

        result = asyncio.run(MultiTurnIfModule().run_task(Client(), self._task()))
        assert result.truncated is True


class TestEmptyTextSatisfiesNothing:
    """Silence used to be perfect compliance with any prohibition."""

    NEGATIVE = [{"type": "not_contains", "value": "secret"},
                {"type": "max_words", "value": 50}]

    def test_empty_response_scores_zero_against_negative_checks(self):
        score, details = _check_constraints(self.NEGATIVE, "")
        assert score == 0.0
        assert all(d["passed"] is False for d in details)

    def test_whitespace_only_is_empty(self):
        assert _check_constraints(self.NEGATIVE, "  \n\t ")[0] == 0.0

    def test_a_real_response_is_unaffected(self):
        assert _check_constraints(self.NEGATIVE, "a short safe answer")[0] == 1.0

    def test_a_real_response_can_still_fail(self):
        assert _check_constraints(self.NEGATIVE, "the secret is out")[0] == 0.5


class TestFileChecksFailClosed:
    def test_section_contains_with_no_target_fails(self):
        # Used to build [None] and test `"none" in body`, so this passed on
        # any text containing the word "none".
        check = {"type": "section_contains", "section": "## A"}
        assert _eval_file_check(check, "## A\nnone of it matters", "") is False

    def test_section_contains_still_matches_a_real_value(self):
        check = {"type": "section_contains", "section": "## A", "value": "hit"}
        assert _eval_file_check(check, "## A\na hit here", "") is True

    def test_section_contains_reads_an_any_list(self):
        check = {"type": "section_contains", "section": "## A",
                 "any": ["nope", "hit"]}
        assert _eval_file_check(check, "## A\na hit here", "") is True

    def test_table_rows_preserved_fails_when_there_was_no_table(self):
        # all([]) is True, so a declared check with nothing to compare used to
        # report a preservation it never tested.
        check = {"type": "table_rows_preserved"}
        assert _eval_file_check(check, "| a |\n|---|\n| 1 |", "") is False

    def test_table_rows_preserved_still_passes_a_real_preservation(self):
        table = "| k | v |\n|---|---|\n| a | 1 |\n| b | 2 |"
        assert _eval_file_check({"type": "table_rows_preserved"},
                                table + "\n| c | 3 |", table) is True

    def test_dropping_a_row_is_caught(self):
        table = "| k | v |\n|---|---|\n| a | 1 |\n| b | 2 |"
        shorter = "| k | v |\n|---|---|\n| a | 1 |"
        assert _eval_file_check({"type": "table_rows_preserved"},
                                shorter, table) is False


class TestFileCheckLabelsAreDistinct:
    """tst_63's four widget checks collapsed onto one breakdown key."""

    def test_same_type_and_section_get_distinct_labels(self):
        checks = [{"type": "section_contains", "section": "## Counts",
                   "value": f"widget-{w}: {n}"}
                  for w, n in (("a", 32), ("b", 30), ("c", 12), ("d", 8))]
        _, detail = score_file_checks(
            {"stock/ledger.md": checks},
            {"files": {"stock/ledger.md": "## Counts\nwidget-a: 32\n"
                                          "widget-b: 30\nwidget-c: 22\n"
                                          "widget-d: 8"}},
            {"files": {}})
        assert len(detail) == 4
        # And the failure is attributable: only widget-c is wrong.
        failed = [k for k, ok in detail.items() if not ok]
        assert len(failed) == 1 and "widget-c: 12" in failed[0]

    def test_score_is_the_fraction_of_all_checks(self):
        checks = [{"type": "section_contains", "section": "## C", "value": v}
                  for v in ("a", "b", "c", "d")]
        score, _ = score_file_checks({"f.md": checks},
                                     {"files": {"f.md": "## C\na b c"}},
                                     {"files": {}})
        assert score == 0.75

    def test_identical_checks_still_get_separate_keys(self):
        check = {"type": "bullets_wellformed", "section": "## C"}
        assert _file_check_label("f.md", check, 0) != _file_check_label(
            "f.md", check, 1)


class TestFinalAnswerIsTerminal:
    def _result(self, turns) -> TaskResult:
        return TaskResult(task_id="t", module="tools", prompt="p", turns=turns)

    def test_mid_loop_narration_is_not_the_final_answer(self):
        # A loop that exhausted max_turns ends on an assistant turn that still
        # carries tool calls. Walking back to "any assistant message with
        # content" graded the narration as the model's closing words.
        from small_llm_bench.models import ToolCall
        turns = [TurnRecord(role="assistant", content="Let me check.",
                            tool_calls=[ToolCall(name="read_file",
                                                 arguments={})]),
                 TurnRecord(role="tool", content="{}")]
        assert _final_answer(self._result(turns)) == ""

    def test_a_terminal_assistant_turn_is_the_answer(self):
        turns = [TurnRecord(role="assistant", content="Done: 42.")]
        assert _final_answer(self._result(turns)) == "Done: 42."

    def test_a_truncated_result_never_answered(self):
        result = self._result([TurnRecord(role="assistant", content="Done.")])
        result.truncated = True
        assert _final_answer(result) == ""


class TestJudgeCannotOverruleHardGates:
    """Facts about what the episode did, not opinions about quality."""

    def _trial(self, **breakdown) -> TaskResult:
        return TaskResult(task_id="t", module="tools", prompt="p",
                          det_score=0.2, success=False, llm_score=0.99,
                          det_breakdown=breakdown)

    @pytest.mark.parametrize("gate", ["within_budget", "no_loop_detected",
                                      "no_premature_stop"])
    def test_a_failed_gate_blocks_the_rescue(self, gate):
        trial = self._trial(**{gate: False})
        apply_judge_verdict(trial, "tools")
        assert trial.success is False
        assert trial.judge_blocked_by == gate

    def test_a_forbidden_tool_blocks_the_rescue(self):
        trial = self._trial(used_forbidden_tool=True)
        apply_judge_verdict(trial, "tools")
        assert trial.success is False
        assert trial.judge_blocked_by == "used_forbidden_tool"

    def test_degenerate_truncation_blocks_the_rescue(self):
        trial = self._trial(within_budget=True)
        trial.truncation_class = "degenerate"
        apply_judge_verdict(trial, "tools")
        assert trial.success is False

    def test_a_clean_trial_can_still_be_rescued(self):
        trial = self._trial(within_budget=True, no_loop_detected=True,
                            no_premature_stop=True)
        apply_judge_verdict(trial, "tools")
        assert trial.success is True
        assert trial.judge_blocked_by is None

    def test_hard_gate_failure_names_nothing_on_a_clean_trial(self):
        assert hard_gate_failure(self._trial(within_budget=True)) is None

    def test_the_stored_judge_score_is_left_as_emitted(self):
        # Clamping it to det made rescore re-derive a verdict from a number
        # the judge never gave.
        trial = self._trial(within_budget=True)
        trial.llm_score = 0.9
        trial.det_score = 0.88
        apply_judge_verdict(trial, "tools")
        assert trial.llm_score == 0.9


class TestErroredReadsTheToolResult:
    def test_a_result_merely_containing_the_word_is_not_an_error(self):
        from small_llm_bench.scorer import score_tool_loop
        from small_llm_bench.models import ToolCall

        # A log line mentioning "error" used to disqualify the goal call.
        payload = json.dumps({"content": 'line 3: "error" handling is fine'})
        res = score_tool_loop(
            {"goal_tool": "read_file", "goal_args": {"path": "a.log"},
             "optimal_turns": 1},
            [ToolCall(name="read_file", arguments={"path": "a.log"})],
            results=[payload])
        assert res.breakdown["goal_reached"] == 1.0

    def test_a_real_error_object_still_counts(self):
        from small_llm_bench.scorer import score_tool_loop
        from small_llm_bench.models import ToolCall

        res = score_tool_loop(
            {"goal_tool": "read_file", "goal_args": {"path": "a.log"},
             "optimal_turns": 1},
            [ToolCall(name="read_file", arguments={"path": "a.log"})],
            results=[json.dumps({"error": "no such file"})])
        assert res.breakdown["goal_reached"] == 0.0


class TestSeedIsSentAndRecorded:
    """pass^k is a statement about a sampler; nothing recorded the sampler."""

    def _capture(self, seed):
        import httpx
        from small_llm_bench.runner import ChatClient, _SEED

        sent: dict = {}

        def handler(request):
            sent.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]})

        async def go():
            client = ChatClient("http://x", "m",
                                transport=httpx.MockTransport(handler))
            _SEED.set(seed)
            await client.chat([{"role": "user", "content": "hi"}])
            await client.close()

        asyncio.run(go())
        return sent

    def test_the_payload_carries_the_trial_seed(self):
        assert self._capture(7)["seed"] == 7

    def test_no_seed_is_sent_when_unset(self):
        # A server that rejects the field must stay usable.
        assert "seed" not in self._capture(None)

    def test_trials_of_one_task_get_different_seeds(self):
        # Same seed on every trial makes k samples into k copies, and pass^k
        # then measures nothing.
        base, seen = 100, []
        for index in range(3):
            seen.append(base + index)
        assert len(set(seen)) == 3

    def test_sampling_is_described_in_words(self):
        from small_llm_bench.runner import _sampling_description

        assert _sampling_description(None, None) == \
            "server-default, no seed sent"
        assert _sampling_description(0.0, 42) == "temperature=0, seed=42+trial"


class TestTextAnswerOkIsReachable:
    def test_a_prose_answer_with_no_calls_can_succeed(self):
        from small_llm_bench.scorer import score_tool_loop

        # min_calls defaulted to 1, so no_premature_stop was 0.0 on exactly
        # the zero-call path this flag exists to allow.
        res = score_tool_loop(
            {"text_answer_ok": True, "goal_tool": "ask_user",
             "optimal_turns": 1},
            [], results=[], answer_text="No tool can do that; here is why.")
        assert res.breakdown["no_premature_stop"] == 1.0

    def test_an_explicit_min_calls_still_wins(self):
        from small_llm_bench.scorer import score_tool_loop

        res = score_tool_loop(
            {"text_answer_ok": True, "min_calls": 1, "goal_tool": "ask_user",
             "optimal_turns": 1},
            [], results=[], answer_text="prose")
        assert res.breakdown["no_premature_stop"] == 0.0


class TestScorerCrashDoesNotKillTheRun:
    def test_a_scorer_exception_becomes_one_unscorable_trial(self, monkeypatch):
        from small_llm_bench import runner as runner_module
        from small_llm_bench.scorer import counted

        class Boom(BaseModule := object):
            pass

        async def fake_run_task(client, task, sandbox=None):
            return TaskResult(task_id=task.id, module="knowledge", prompt="p",
                              response_raw="42")

        def explode(task, result, sandbox=None):
            raise KeyError("contains")

        monkeypatch.setattr(runner_module, "score_task", explode)

        class FakeModule:
            name = "knowledge"
            run_task = staticmethod(fake_run_task)

        task = Task(id="kn_x", module="knowledge", prompt="p")
        result = asyncio.run(
            runner_module._execute_task(FakeModule(), None, task))
        assert result.infra_error is True
        assert "scoring failed" in (result.error or "")
        # Excluded rather than counted as a model failure it never caused.
        assert counted(result) is False


class TestExactFactualAnswers:
    """A credential is not "nearly right" with six characters wrong.

    The default factual path lowercases and then falls back to a 0.85 fuzzy
    ratio, which a 44-character key clears with six edits — and lowercasing
    erases the case distinction that is half its entropy. `exact: true` turns
    both leniencies off, so a task can grade an opaque high-entropy value:
    copying it verbatim out of a long document is the capability under test.
    """

    KEY = "sk_" + "live_" + "7o6mAnZUN6pCq21Gho8kw5HZ5ZsKxapxZT29"  # split so secret scanners do not flag a fake key

    def _score(self, response, exact=True):
        from small_llm_bench.scorer import score_knowledge
        expected = {"answer": self.KEY}
        if exact:
            expected["exact"] = True
        return score_knowledge(expected, "factual", response).score

    def test_verbatim_passes(self):
        assert self._score(f"The active key is {self.KEY}.") == 1.0

    def test_one_character_wrong_fails(self):
        assert self._score(f"The key is {self.KEY[:-1]}X") == 0.0

    def test_case_is_significant(self):
        assert self._score(self.KEY.lower()) == 0.0

    def test_absent_fails(self):
        assert self._score("I could not find it.") == 0.0

    def test_without_exact_the_same_near_miss_would_pass(self):
        # The reason the flag had to exist, pinned so it cannot regress.
        assert self._score(f"The key is {self.KEY[:-1]}X", exact=False) == 1.0
        assert self._score(self.KEY.lower(), exact=False) == 1.0

    def test_accept_alternatives_still_honoured(self):
        from small_llm_bench.scorer import score_knowledge
        res = score_knowledge(
            {"answer": "AAA", "accept": ["BBB"], "exact": True}, "factual",
            "the value is BBB")
        assert res.score == 1.0 and res.breakdown["method"] == "exact"


class TestRealisticFiller:
    """`log` filler is one sentence repeated; a needle in it has a unique shape.

    Measured 2026-09-07: all three trio models pulled a 44-character API key
    out of 54k tokens of `log` filler verbatim, the 4B included, because
    `KEYROTATE` lines were the only lines of their shape in the document.
    Retrieval was shape-matching, not search. `ops_audit` is the realistic
    mixed-shape alternative, and being realistic is what makes it interfere: a
    real operations log is full of token-shaped identifiers.
    """

    def _lines(self, target=4000, salt="t", style="ops_audit"):
        from small_llm_bench.modules.long_context import _filler_lines
        return _filler_lines(target, salt, style=style)

    def test_default_style_is_unchanged(self):
        # Every banked task hashes its haystack spec, so the default must not
        # move or the whole reference panel is invalidated.
        from small_llm_bench.modules.long_context import _filler_lines, _FILLER
        first = _filler_lines(200, "lc_08")[0]
        assert first.startswith(_FILLER.split("{")[0])

    def test_ops_audit_is_deterministic_for_a_salt(self):
        assert self._lines() == self._lines()

    def test_a_different_salt_gives_a_different_document(self):
        # Two tasks sharing a filler prefix would let prefix caching skip most
        # of the prefill, which is what _salt_offset exists to prevent.
        assert self._lines(salt="a") != self._lines(salt="b")

    def test_it_is_not_one_line_repeated(self):
        lines = self._lines(6000)
        shapes = {" ".join(w for w in l.split() if not any(c.isdigit() for c in w))
                  for l in lines}
        assert len(shapes) > 8, "filler should mix event shapes"

    def test_it_carries_credential_shaped_tokens(self):
        # The interference is the point: a live key must have no distinctive
        # shape to be found by.
        doc = "\n".join(self._lines(8000))
        for prefix in ("sk_test_", "pk_live_", "whsec_", "req_", "trace="):
            assert prefix in doc, prefix

    def test_it_never_emits_the_live_key_prefix(self):
        # Filler must not collide with an sk_live_ needle or a task grading
        # "no other live key" becomes unpassable through no fault of the model.
        assert "sk_live_" not in "\n".join(self._lines(12000))

    def test_target_is_estimated_from_characters(self):
        from small_llm_bench.modules.long_context import _filler_lines
        # Word*1.3 undercounted this style ~4.7x because a 44-character
        # identifier is one word and about fifteen tokens.
        doc = "\n".join(_filler_lines(4000, "t", style="ops_audit"))
        assert 3000 < len(doc) / 2.0 < 5200

    def test_build_haystack_accepts_the_style(self):
        from small_llm_bench.modules.long_context import build_haystack
        doc = build_haystack({"type": "single", "filler_tokens": 2000,
                              "filler_style": "ops_audit",
                              "needle": "NEEDLE-HERE", "position": 0.5}, salt="x")
        assert "NEEDLE-HERE" in doc and "sk_test_" in doc


class TestMeetingFiller:
    """For tasks whose answer is a conclusion, not a string.

    Every long-context construct measured on 2026-09-07/08 was lexical — the
    answer was a key or a number literally in the document — and qwen3.5-4b
    passed all five 3/3. A transcript asks something different: no line states
    the answer, so there is no record to locate. This filler is what makes
    such a task long-context rather than short-context, and the interference
    it provides is other people committing to other work.
    """

    def _lines(self, target=4000, salt="lc_67"):
        from small_llm_bench.modules.long_context import _filler_lines
        return _filler_lines(target, salt, style="meeting")

    def test_is_deterministic_for_a_salt(self):
        assert self._lines() == self._lines()

    def test_a_different_salt_gives_a_different_transcript(self):
        assert self._lines(salt="a") != self._lines(salt="b")

    def test_every_line_is_attributed_to_a_speaker(self):
        import re
        from small_llm_bench.modules.long_context import _CAST
        for line in self._lines(3000):
            m = re.match(r"^(\w+): ", line)
            assert m and m.group(1) in _CAST, line

    def test_it_carries_decoy_ownership_threads(self):
        # A transcript with exactly one ownership resolution is answerable by
        # finding "the commitment". These put unrelated ones in the way.
        doc = "\n".join(self._lines(12000))
        assert any(p in doc for p in ("I can own", "I will take it",
                                      "Put me down")), doc[:200]

    def test_it_never_mentions_a_forecast_or_a_rebuild(self):
        # A task grading that thread must own those words exclusively, or the
        # document is genuinely ambiguous and the task is broken, not hard.
        doc = "\n".join(self._lines(15000)).lower()
        assert "forecast" not in doc and "rebuild" not in doc

    def test_build_haystack_accepts_the_style(self):
        from small_llm_bench.modules.long_context import build_haystack
        doc = build_haystack({"type": "single", "filler_tokens": 2000,
                              "filler_style": "meeting",
                              "needle": "Dana: NEEDLE.", "position": 0.5},
                             salt="x")
        assert "Dana: NEEDLE." in doc and doc.count("\n") > 20


class TestTaskDefinitionErrorsAreNotModelFailures:
    def test_a_missing_spec_key_is_excluded_not_failed(self):
        """A malformed task used to read as every model failing it.

        Spotted on a broken probe candidate whose haystack spec had no
        `needle`: three trials recorded `KeyError: 'needle'` with
        infra_error=False, so `counted()` kept them and the task looked like
        a universal capability gap. _is_transient returns False for KeyError,
        and that path predates the v1.0 audit.
        """
        import asyncio
        from small_llm_bench import runner as runner_module
        from small_llm_bench.models import Task
        from small_llm_bench.scorer import counted

        class Broken:
            name = "long_context"

            async def run_task(self, client, task, sandbox=None):
                raise KeyError("needle")

        task = Task(id="lc_x", module="long_context", prompt="p")
        result = asyncio.run(runner_module._execute_task(Broken(), None, task))
        assert result.infra_error is True
        assert "task definition error" in (result.error or "")
        assert counted(result) is False


class TestHarnessFailuresAreNotModelFailures:
    """A sandbox that never ran the candidate, and an endpoint that never had
    the model, both used to be graded as the model answering wrong.

    Measured on neohorse-1-9b and -4b, 2026-09-13: a Docker image the daemon
    could not produce took the whole code module to 0/15 with every candidate
    marked `wellformed: true`, and a model swapped off the port mid-sweep took
    the run's last two trials as 404s scored 0.0.
    """

    CODE = "```python\ndef f(x):\n    return x\n```"
    BARE = "def f(x):\n    return x\n"

    def _outcome(self, status, stderr=""):
        from small_llm_bench.sandbox import SandboxResult
        return SandboxResult(stdout="", stderr=stderr, returncode=-1,
                             status=status, backend="docker")

    def _patch_run(self, monkeypatch, status, stderr=""):
        # `failure_report` imports _run_test_cases from .scorer at call time,
        # so scorer is the one place both paths resolve it from.
        from small_llm_bench import scorer as scorer_module
        monkeypatch.setattr(scorer_module, "_run_test_cases",
                            lambda *a, **k: self._outcome(status, stderr))

    def test_missing_image_is_unavailable_not_error(self, monkeypatch):
        from small_llm_bench import sandbox as sandbox_module
        monkeypatch.setattr(sandbox_module, "_ensure_image",
                            lambda backend, image: False)
        outcome = sandbox_module._run_container(
            "docker", "print(1)", timeout=5.0, memory_mb=256,
            image="python:3.12-slim")
        assert outcome.status == "unavailable"
        assert outcome.returncode == -1

    def test_unavailable_sandbox_excludes_the_trial(self, monkeypatch):
        """The candidate is well-formed; only the runner failed."""
        from small_llm_bench.scorer import score_code
        self._patch_run(monkeypatch, "unavailable",
                        "image unavailable: python:3.12-slim")
        scored = score_code(self.CODE, "f", [Case(args=[1], expected=1)])
        assert scored.score == 0.0
        assert scored.infra_error is True
        assert scored.breakdown["wellformed"] is True

    def test_a_candidate_that_ran_and_failed_still_counts(self, monkeypatch):
        """The mirror case: status 'error' is the MODEL's failure."""
        from small_llm_bench.scorer import score_code
        self._patch_run(monkeypatch, "error", "NameError: x")
        scored = score_code(self.CODE, "f", [Case(args=[1], expected=1)])
        assert scored.score == 0.0
        assert scored.infra_error is False

    def test_repair_loop_gets_no_feedback_when_nothing_ran(self, monkeypatch):
        """Otherwise the model is asked to debug a container it cannot see."""
        from small_llm_bench.modules.code import failure_report
        self._patch_run(monkeypatch, "unavailable",
                        "image unavailable: python:3.12-slim")
        task = Task(id="cd_x", module="code", prompt="p", function_name="f",
                    test_cases=[Case(args=[1], expected=1)])
        assert failure_report(task, self.BARE, {}) is None

    def test_404_is_excluded_not_counted(self):
        """A model the server does not serve is not a model that failed."""
        import httpx
        from small_llm_bench.runner import (_is_config_error, _is_retryable,
                                            _is_transient)
        exc = httpx.HTTPStatusError(
            "404", request=httpx.Request("POST", "http://x/v1/chat/completions"),
            response=httpx.Response(404))
        assert _is_config_error(exc) is True
        assert _is_transient(exc) is False      # excluded, but never retried
        assert _is_retryable(exc) is False

    def test_rescore_clears_a_sandbox_infra_flag_it_re_measures(self, monkeypatch):
        """The recovery path: a file whose code module was lost to a missing
        image comes back with real scores, not fifteen permanent zeros."""
        from small_llm_bench import rescore as rescore_module
        from small_llm_bench.models import ScorerResult
        from small_llm_bench.scorer import counted
        monkeypatch.setattr(rescore_module, "score_task",
                            lambda task, result, sandbox: ScorerResult(
                                score=1.0, success=True, infra_error=False))
        task = Task(id="cd_x", module="code", prompt="p")
        result = TaskResult(task_id="cd_x", module="code", prompt="p",
                            response_raw=self.CODE, infra_error=True)
        rescore_module._rescore_one(task, result, {}, rescore_module.RescoreReport(),
                                    "code:cd_x")
        assert result.infra_error is False
        assert counted(result) is True

    def test_rescore_leaves_a_generation_failure_alone(self, monkeypatch):
        """Nothing was generated, so re-grading cannot re-measure it: the 404
        rows must not be laundered into a real 0.0 by a rescore pass."""
        from small_llm_bench import rescore as rescore_module
        from small_llm_bench.models import ScorerResult
        from small_llm_bench.scorer import counted
        monkeypatch.setattr(rescore_module, "score_task",
                            lambda task, result, sandbox: ScorerResult(
                                score=0.0, success=False, infra_error=False))
        task = Task(id="fm_x", module="format", prompt="p")
        result = TaskResult(task_id="fm_x", module="format", prompt="p",
                            response_raw="", infra_error=True,
                            error="HTTPStatusError: Client error '404 Not Found'")
        rescore_module._rescore_one(task, result, None, rescore_module.RescoreReport(),
                                    "format:fm_x")
        assert result.infra_error is True
        assert counted(result) is False
