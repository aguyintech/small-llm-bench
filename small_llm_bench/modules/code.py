"""Code generation module: model writes a Python function, scored by execution.

Tasks may grant more than one attempt (``repair_attempts``). Extra attempts run
the model's code in the same sandbox the scorer uses and hand back the concrete
failure — traceback, syntax error, or which inputs came back wrong (never the
expected value: the report is a failing-test signal, not the answer key).
That mirrors how these models are actually used: nobody ships the first draft
unexecuted. One-shotting still scores highest (see ``_REPAIR_DECAY`` in
``scorer``), but forgetting an import is no longer indistinguishable from not
knowing the answer.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import BaseModule, completion_tokens, hit_length_cap, message_text

if TYPE_CHECKING:
    from ..runner import ChatClient

_RETRY_HEADER = ("Your code did not pass. Fix it and reply with a single "
                 "corrected Python code block.\n\n")
# Enough for the model to see the pattern, few enough to stay inside the code
# module's 4096-token completion cap on the next attempt.
# How many failing cases a repair report names. The report says which INPUTS
# are wrong and what the function returned for them — never what it should
# have returned. Printing `expected` handed over the grading key: cd_15 has
# five test cases total and three repair attempts, so a model could pass by
# transcribing the reported pairs instead of fixing the function, and the
# _REPAIR_DECAY discount prices attempts rather than leakage.
_MAX_REPORTED_CASES = 5
_MAX_STDERR_LINES = 10


class CodeModule(BaseModule):
    """Tests Python function generation against executable test cases."""

    name = "code"

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Ask the model for a function, then let it fix what it got wrong.

        Repair tasks ship a ``buggy_code`` implementation that is shown to the
        model to fix; plain tasks are written from scratch. Long-context tasks
        prepend several ``context_files`` (a simulated codebase) where the fix
        depends on a detail living in another file.

        The loop stops early on a pass, on a repeat of the previous code (more
        turns won't help a model that isn't changing anything), or when no
        sandbox is available to produce feedback with.
        """
        history: list[dict[str, Any]] = [
            {"role": "user", "content": _compose_prompt(task)}]
        turns: list[TurnRecord] = []
        responses: list[dict[str, Any]] = []
        max_attempts = max(1, task.repair_attempts)
        tokens = 0
        content = ""
        truncated = False
        previous_code: str | None = None
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            response = await client.chat(history, max_tokens=task.max_tokens)
            message = response["choices"][0]["message"]
            content = message_text(message)
            turn_tokens = completion_tokens(response)
            tokens += turn_tokens
            truncated = hit_length_cap(response)
            responses.append(response)
            turns.append(TurnRecord(role="assistant", content=content,
                                    completion_tokens=turn_tokens,
                                    truncated=truncated))

            if attempt >= max_attempts or sandbox is None:
                break
            code = extract_code_for_repair(content)
            report = failure_report(task, code, sandbox)
            if report is None:  # passed, or nothing diagnosable to report
                break
            if code and code == previous_code:
                break
            previous_code = code
            history.append({"role": "assistant", "content": content})
            history.append({"role": "user", "content": report})
            turns.append(TurnRecord(role="user", content=report))

        return TaskResult(
            task_id=task.id,
            module=self.name,
            prompt=task.prompt,
            expected={"function_name": task.function_name},
            response_raw=content,
            turns=turns,
            completion_tokens=tokens,
            truncated=truncated,
            attempts_used=attempt,
            raw_api_responses=responses if client.save_responses else None,
        )


def _compose_prompt(task: Task) -> str:
    """Assemble the first-attempt prompt: codebase, task, buggy implementation."""
    parts: list[str] = []
    if task.context_files:
        files = "\n\n".join(
            f"# ===== FILE: {name} =====\n{content}"
            for name, content in task.context_files.items()
        )
        parts.append("You are working in an existing codebase. Relevant "
                     f"files:\n\n{files}")
    parts.append(task.prompt)
    if task.buggy_code:
        parts.append("Here is the current implementation, which contains a "
                     f"bug:\n```python\n{task.buggy_code}\n```")
    return "\n\n".join(parts)


def extract_code_for_repair(content: str) -> str:
    """The scorer's extractor, re-exported so the loop grades what it will grade."""
    from ..scorer import extract_code

    return extract_code(content)


def failure_report(task: Task, code: str,
                   sandbox: dict[str, Any]) -> str | None:
    """Run ``code`` and describe what went wrong, or None if there's nothing
    useful to say — it passed, or the sandbox couldn't tell us anything.

    Returning None on an undiagnosable outcome is deliberate: spending an
    attempt on "it failed somehow" teaches the model nothing.
    """
    from ..scorer import _parse_outcomes, _run_test_cases

    if not code:
        return (_RETRY_HEADER + "No Python code block was found in your reply. "
                "Reply with the complete function inside one ```python block.")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"{_RETRY_HEADER}Syntax error: {exc}"

    cases = list(task.test_cases) + list(task.regression_cases)
    # Mirror score_code's defaults so the loop runs the code under the same
    # limits it will later be graded under, whatever subset the caller passed.
    settings = {"timeout": 5.0, "memory_mb": 256, "backend": "auto",
                "allow_unsandboxed": False, **sandbox}
    outcome = _run_test_cases(code, task.function_name or "", cases,
                              support_code=task.support_code or "",
                              context_files=task.context_files, **settings)
    if outcome.status in ("skipped_no_sandbox", "unavailable"):
        # No feedback: the candidate never ran, so there is nothing to tell the
        # model. Handing it the harness's own failure ("Import/setup error:
        # image unavailable: python:3.12-slim") spends repair attempts asking
        # it to fix a container it cannot see.
        return None
    if outcome.status == "timeout":
        return (_RETRY_HEADER + "Execution timed out — the function did not "
                "finish. Check for an infinite loop or unbounded recursion.")
    if outcome.status == "oom":
        return _RETRY_HEADER + "Execution ran out of memory."
    if outcome.status == "error":
        tail = "\n".join(outcome.stderr.strip().splitlines()[-_MAX_STDERR_LINES:])
        return f"{_RETRY_HEADER}Import/setup error:\n{tail}"

    details = _parse_outcomes(outcome.stdout)
    if details is None:
        return None
    failures = [(case, d) for case, d in zip(cases, details) if not d["ok"]]
    if not failures:
        return None

    fn = task.function_name or "the function"
    lines = []
    for case, detail in failures[:_MAX_REPORTED_CASES]:
        args = ", ".join(repr(a) for a in case.args)
        if detail["error"]:
            lines.append(f"  {fn}({args}) -> raised {detail['error']}")
        else:
            lines.append(f"  {fn}({args}) -> returned {detail['got']}, "
                         f"which is wrong")
    header = f"Failing cases ({len(failures)} of {len(cases)}):"
    return f"{_RETRY_HEADER}{header}\n" + "\n".join(lines)
