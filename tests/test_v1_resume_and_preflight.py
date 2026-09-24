"""v1.0: crash recovery, context preflight, and one exponent for the board.

Three defects the first clean sweep exposed, all of which cost a real
measurement:

* a run wrote its results once, at the end, so an interrupt at minute 78 of 79
  produced nothing;
* a model served a quarter of the context window scored 0 on the tasks that
  did not fit, and nothing in the file said so;
* one dead request relabelled a whole file's k and moved a model two board
  positions.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from small_llm_bench.config import BenchSettings
from small_llm_bench.models import (BenchMeta, BenchResult, Task, TaskResult,
                                    task_content_hash, task_world_hash)
from small_llm_bench.runner import (TrialCheckpoint, _checkpoint_fingerprint,
                                    _observed_trials, checkpoint_path_for,
                                    discover_served_context, load_checkpoint,
                                    oversized_tasks)


def _result(task_id: str = "t", **kw) -> TaskResult:
    return TaskResult(task_id=task_id, module="tools", prompt="p", **kw)


def _fingerprint(**over):
    base = {"model": "m", "profile": "full",
            "thinking": True, "temperature": None, "seed": 0,
            "max_tokens": 8192, "max_tokens_override": False,
            "bench_version": "1.0.0"}
    base.update(over)
    return base


class TestCheckpointSidecar:
    def test_sidecar_sits_beside_the_results_file(self):
        path = checkpoint_path_for(Path("results/m_raw_results.json"))
        assert path == Path("results/m_raw_results.json.partial.jsonl")

    def test_each_trial_is_readable_before_the_run_ends(self, tmp_path):
        """The point of the sidecar: a killed run leaves everything it did."""
        path = tmp_path / "out.json.partial.jsonl"
        fp = _fingerprint()
        with TrialCheckpoint(path, fp) as cp:
            cp.append(_result("a", det_score=1.0))
            # Read from a SEPARATE handle mid-run — this is what recovery does.
            assert len(load_checkpoint(path, fp)) == 1
            cp.append(_result("b"))
            assert len(load_checkpoint(path, fp)) == 2

    def test_recovered_trials_keep_their_scores(self, tmp_path):
        path = tmp_path / "out.partial.jsonl"
        fp = _fingerprint()
        with TrialCheckpoint(path, fp) as cp:
            cp.append(_result("a", det_score=0.75, success=True,
                              completion_tokens=42))
        (recovered,) = load_checkpoint(path, fp)
        assert (recovered.task_id, recovered.det_score, recovered.success,
                recovered.completion_tokens) == ("a", 0.75, True, 42)

    def test_a_torn_final_line_costs_only_that_line(self, tmp_path):
        """The expected shape of an interrupted write: keep what came before."""
        path = tmp_path / "out.partial.jsonl"
        fp = _fingerprint()
        with TrialCheckpoint(path, fp) as cp:
            cp.append(_result("a"))
            cp.append(_result("b"))
        with path.open("a") as handle:
            handle.write('{"task_id": "c", "modu')
        assert [r.task_id for r in load_checkpoint(path, fp)] == ["a", "b"]

    def test_a_different_config_recovers_nothing(self, tmp_path):
        """A recovered trial may only rejoin a run that would have made it."""
        path = tmp_path / "out.partial.jsonl"
        with TrialCheckpoint(path, _fingerprint()) as cp:
            cp.append(_result("a"))
        assert load_checkpoint(path, _fingerprint(thinking=False)) == []
        assert load_checkpoint(path, _fingerprint(max_tokens=16384)) == []
        assert load_checkpoint(path, _fingerprint(model="other")) == []

    def test_a_moved_server_still_recovers_its_trials(self, tmp_path):
        """The endpoint is where the server was, not how the model ran. A
        server whose address changed mid-sweep keeps what it already served."""
        path = tmp_path / "out.partial.jsonl"
        before = _checkpoint_fingerprint(
            BenchSettings(model="m", endpoint="http://10.0.0.5/v1"),
            "full", None, 0, False)
        after = _checkpoint_fingerprint(
            BenchSettings(model="m", endpoint="http://10.0.0.9/v1"),
            "full", None, 0, False)
        with TrialCheckpoint(path, before) as cp:
            cp.append(_result("a"))
        assert len(load_checkpoint(path, after)) == 1

    def test_a_sidecar_written_by_1_0_0_still_recovers(self, tmp_path):
        """1.0.0 sidecars fingerprinted the endpoint. Dropping it from the
        fingerprint must not orphan a run that was interrupted before."""
        path = tmp_path / "out.partial.jsonl"
        with TrialCheckpoint(path, _fingerprint(endpoint="http://old/v1")) as cp:
            cp.append(_result("a"))
        assert len(load_checkpoint(path, _fingerprint())) == 1

    def test_missing_empty_and_headerless_files_recover_nothing(self, tmp_path):
        fp = _fingerprint()
        assert load_checkpoint(tmp_path / "nope.jsonl", fp) == []
        (tmp_path / "empty.jsonl").write_text("")
        assert load_checkpoint(tmp_path / "empty.jsonl", fp) == []
        (tmp_path / "junk.jsonl").write_text("not json\n")
        assert load_checkpoint(tmp_path / "junk.jsonl", fp) == []

    def test_reopening_appends_rather_than_truncating(self, tmp_path):
        """A resumed run must not throw away what the first session recovered."""
        path = tmp_path / "out.partial.jsonl"
        fp = _fingerprint()
        with TrialCheckpoint(path, fp) as cp:
            cp.append(_result("a"))
        with TrialCheckpoint(path, fp) as cp:
            cp.append(_result("b"))
        assert [r.task_id for r in load_checkpoint(path, fp)] == ["a", "b"]
        # And only one header line, or the second would parse as a trial.
        first = json.loads(path.read_text().splitlines()[0])
        assert "__checkpoint__" in first
        assert sum("__checkpoint__" in line for line in
                   path.read_text().splitlines()) == 1

    def test_discard_removes_the_sidecar(self, tmp_path):
        path = tmp_path / "out.partial.jsonl"
        cp = TrialCheckpoint(path, _fingerprint()).open()
        cp.append(_result("a"))
        cp.discard()
        assert not path.exists()

    def test_fingerprint_covers_every_field_that_makes_trials_blendable(self):
        settings = BenchSettings(model="m", endpoint="http://x/v1")
        fp = _checkpoint_fingerprint(settings, "full", 0.7, 5, True)
        assert set(fp) == {"model", "profile", "thinking",
                           "temperature", "seed", "max_tokens",
                           "max_tokens_override", "bench_version"}
        assert (fp["temperature"], fp["seed"], fp["max_tokens_override"]) == (
            0.7, 5, True)


class TestServedContextPreflight:
    def _task(self, tid, filler=0, cap=None):
        return Task(id=tid, module="long_context", prompt="p",
                    max_tokens=cap,
                    haystack={"filler_tokens": filler} if filler else {})

    def test_the_measured_lc_08_case_is_caught(self):
        """qwen3.5-4b: --parallel 4 on --ctx-size 65536 served 16384, and
        lc_08's 16000-token haystack plus a 16384 cap cannot fit it."""
        selected = [(None, self._task("lc_08", filler=16000, cap=16384))]
        assert oversized_tasks(selected, 16384) == [("lc_08", 32384)]
        # The same task on the same server without the slot split is fine.
        assert oversized_tasks(selected, 65536) == []

    def test_small_prompt_tasks_are_judged_on_their_cap_alone(self):
        selected = [(None, self._task("tst_01", cap=4096))]
        assert oversized_tasks(selected, 2048) == [("tst_01", 4096)]
        assert oversized_tasks(selected, 8192) == []

    def test_results_are_worst_first(self):
        selected = [(None, self._task("small", filler=1000, cap=1000)),
                    (None, self._task("huge", filler=60000, cap=8192))]
        assert [tid for tid, _ in oversized_tasks(selected, 512)] == ["huge",
                                                                     "small"]

    @pytest.mark.asyncio
    async def test_discovery_reads_the_per_sequence_window(self):
        """llama.cpp reports n_ctx PER SLOT, which is what one request may use."""
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/props"
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": 16384},
                "total_slots": 4})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            got = await discover_served_context(client, "http://x/v1", "m")
        assert got == 16384

    @pytest.mark.asyncio
    async def test_discovery_falls_back_to_the_llama_swap_upstream_path(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            if request.url.path == "/props":
                return httpx.Response(404, json={"error": "no model id"})
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": 65536}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            got = await discover_served_context(client, "http://x/v1", "qwen")
        assert got == 65536
        assert seen == ["/props", "/upstream/qwen/props"]

    @pytest.mark.asyncio
    async def test_an_endpoint_that_cannot_introspect_is_not_an_error(self):
        """Refusing to run against a server that merely declines to answer
        would be worse than the problem this guard exists for."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            assert await discover_served_context(client, "http://x/v1", "m") is None

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_not_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            assert await discover_served_context(client, "http://x/v1", "m") is None

    @pytest.mark.asyncio
    async def test_garbage_json_is_not_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>nope</html>")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            assert await discover_served_context(client, "http://x/v1", "m") is None


class TestOneExponentForTheBoard:
    def test_one_dead_request_no_longer_relabels_the_file(self):
        """The measured case: 39 tasks at 3 trials, one ReadTimeout on de_07."""
        results = []
        for i in range(38):
            results += [_result(f"t{i}") for _ in range(3)]
        results += [_result("de_07"), _result("de_07"),
                    _result("de_07", infra_error=True)]
        assert _observed_trials(results, requested=3) == 3

    def test_the_board_scores_every_row_at_one_k(self, tmp_path, monkeypatch):
        """pass^2 > pass^3 always, so ranking rows at different k hands the
        short row a free lift over every task it holds."""
        from small_llm_bench import leaderboard as lb

        def write(name: str, trials: int, passes: int) -> None:
            results = [
                TaskResult(task_id=f"t{i}", module="tools", prompt="p",
                           det_score=1.0 if j < passes else 0.0,
                           success=j < passes, det_success=j < passes,
                           task_hash="h")
                for i in range(4) for j in range(3)]
            meta = BenchMeta(model=name, endpoint="e", timestamp="t",
                             duration_seconds=1.0, bench_version="1.0.0",
                             trials=trials, task_set_hash="same")
            (tmp_path / f"{name}_raw_results.json").write_text(
                BenchResult(meta=meta, results=results).model_dump_json())

        # Identical trials; only the recorded k differs.
        write("claims_k2", trials=2, passes=2)
        write("honest_k3", trials=3, passes=2)
        data = lb.build_leaderboard(tmp_path, scheme="module")
        assert data["headline_k"] == 3
        by_model = {r["model"]: r for r in data["rows"]}
        assert (by_model["claims_k2"]["overall_pass"]
                == by_model["honest_k3"]["overall_pass"])


# --- end to end through run_bench -------------------------------------------

def _run_fingerprint(settings: BenchSettings):
    """The fingerprint run_bench will compute for these settings.

    Mirrors its resolution: seed < 0 means "unset", anything else is the seed.
    """
    base_seed = None if settings.seed < 0 else settings.seed
    return _checkpoint_fingerprint(settings, "full", settings.temperature,
                                   base_seed, False)


@pytest.fixture
def scoring_execute(monkeypatch):
    """Every task answers, counting attempts so reuse is observable."""
    calls: list[str] = []

    async def fake(module, client, task, sandbox=None, seed=None):
        calls.append(task.id)
        # Real trials carry these; without them _plan_work cannot recognise a
        # recovered trial as reusable and would re-run everything.
        return TaskResult(task_id=task.id, module=module.name,
                          prompt=task.prompt, response_raw="ok",
                          det_score=1.0, success=True, det_success=True,
                          task_hash=task_content_hash(task),
                          world_hash=task_world_hash(task))

    monkeypatch.setattr("small_llm_bench.runner._execute_task", fake)

    # No endpoint to introspect in a unit test; the preflight's documented
    # None case. Stubbed rather than left to a connection refusal so these
    # tests never touch the network.
    async def no_props(client, base_url, model):
        return None

    monkeypatch.setattr("small_llm_bench.runner.discover_served_context",
                        no_props)
    return calls


async def test_run_bench_writes_every_trial_as_it_goes(
        scoring_execute, tasks_dir, tmp_path):
    """The sidecar must be complete BEFORE run_bench returns — that is the
    whole point. Read it back with a fresh loader, as recovery would."""
    from small_llm_bench.runner import run_bench

    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=0)
    sidecar = tmp_path / "out.json.partial.jsonl"
    bench = await run_bench(settings, modules=["knowledge"], trials=1,
                            tasks_dir=tasks_dir, checkpoint_path=sidecar)
    fp = _run_fingerprint(settings)
    recovered = load_checkpoint(sidecar, fp)
    assert len(recovered) == len(bench.results) == len(scoring_execute)
    assert {r.task_id for r in recovered} == {r.task_id for r in bench.results}


async def test_an_interrupted_run_resumes_instead_of_repeating_itself(
        scoring_execute, tasks_dir, tmp_path):
    """The measured motivation: 79 minutes of work must survive a kill."""
    from small_llm_bench.runner import run_bench

    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=0)
    sidecar = tmp_path / "out.json.partial.jsonl"

    # Session one dies after its first trial.
    first = await run_bench(settings, modules=["knowledge"], trials=1,
                            tasks_dir=tasks_dir, checkpoint_path=sidecar)
    done = len(first.results)
    fp = _run_fingerprint(settings)
    keep = load_checkpoint(sidecar, fp)[:1]
    sidecar.write_text(json.dumps({"__checkpoint__": fp}) + "\n"
                       + keep[0].model_dump_json() + "\n")

    scoring_execute.clear()
    second = await run_bench(settings, modules=["knowledge"], trials=1,
                             tasks_dir=tasks_dir, checkpoint_path=sidecar)
    assert len(second.results) == done
    # The recovered trial was not run again.
    assert len(scoring_execute) == done - 1
    assert keep[0].task_id not in scoring_execute


async def test_no_resume_ignores_the_sidecar(
        scoring_execute, tasks_dir, tmp_path):
    from small_llm_bench.runner import run_bench

    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=0)
    sidecar = tmp_path / "out.json.partial.jsonl"
    first = await run_bench(settings, modules=["knowledge"], trials=1,
                            tasks_dir=tasks_dir, checkpoint_path=sidecar)
    scoring_execute.clear()
    await run_bench(settings, modules=["knowledge"], trials=1,
                    tasks_dir=tasks_dir, checkpoint_path=sidecar, resume=False)
    assert len(scoring_execute) == len(first.results)


async def test_run_bench_refuses_a_window_too_small_for_the_bank(
        scoring_execute, tasks_dir, tmp_path, monkeypatch):
    from small_llm_bench import runner as runner_mod
    from small_llm_bench.runner import UndersizedContextError, run_bench

    async def tiny(client, base_url, model):
        return 512

    monkeypatch.setattr(runner_mod, "discover_served_context", tiny)

    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=0)
    with pytest.raises(UndersizedContextError, match="512 tokens per request"):
        await run_bench(settings, modules=["long_context"], trials=1,
                        tasks_dir=tasks_dir)
    assert scoring_execute == []      # refused BEFORE spending the budget


async def test_the_escape_hatch_runs_anyway(
        scoring_execute, tasks_dir, monkeypatch):
    from small_llm_bench import runner as runner_mod
    from small_llm_bench.runner import run_bench

    async def tiny(client, base_url, model):
        return 512

    monkeypatch.setattr(runner_mod, "discover_served_context", tiny)

    settings = BenchSettings(model="m", endpoint="http://localhost:1/v1",
                             concurrency=1, sanity_check_after=0)
    bench = await run_bench(settings, modules=["long_context"], trials=1,
                            tasks_dir=tasks_dir,
                            allow_undersized_context=True)
    assert scoring_execute and bench.meta.served_context == 512


# --- a 500 the model earned ---------------------------------------------------

class TestMalformedToolCallIsAModelFailure:
    """LFM2.5-8B-A1B, 2026-09-09: three tasks vanished from its bank across two
    runs because llama.cpp answers 500 when a tool call's arguments are not
    valid JSON, and a 500 was unconditionally infra."""

    def _http_error(self, status: int, body: str) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "http://x/v1/chat/completions")
        response = httpx.Response(status, text=body, request=request)
        return httpx.HTTPStatusError("err", request=request, response=response)

    _REAL_BODY = (
        '{"error":{"code":500,"message":"Failed to parse tool call arguments '
        'as JSON: [json.exception.parse_error.101] parse error at line 2, '
        'column 0: syntax error while parsing value - invalid string: control '
        'character U+000A (LF) must be escaped to \\\\u000A or \\\\n; last '
        'read: \'\\"# 2026-08-24<U+000A>\'","type":"server_error"}}')

    def test_the_measured_body_is_recognised(self):
        from small_llm_bench.runner import _is_malformed_tool_call
        assert _is_malformed_tool_call(self._http_error(500, self._REAL_BODY))

    def test_it_is_neither_transient_nor_retryable(self):
        """Excluding it flattered the model that could not do the task, and
        retrying it just re-emits the same bytes three times."""
        from small_llm_bench.runner import _is_retryable, _is_transient
        exc = self._http_error(500, self._REAL_BODY)
        assert _is_transient(exc) is False
        assert _is_retryable(exc) is False

    def test_a_bare_500_stays_infra(self):
        """Deliberately narrow: only a body naming a parse failure counts."""
        from small_llm_bench.runner import _is_malformed_tool_call, _is_transient
        exc = self._http_error(500, "internal server error")
        assert _is_malformed_tool_call(exc) is False
        assert _is_transient(exc) is True

    def test_other_statuses_are_untouched(self):
        from small_llm_bench.runner import (_is_context_overflow,
                                            _is_malformed_tool_call)
        overflow = self._http_error(400, "the request exceeds the available "
                                         "context size")
        assert _is_malformed_tool_call(overflow) is False
        assert _is_context_overflow(overflow) is True
        assert _is_malformed_tool_call(self._http_error(503, "unavailable")) is False

    def test_a_non_http_error_is_not_one(self):
        from small_llm_bench.runner import _is_malformed_tool_call
        assert _is_malformed_tool_call(httpx.ConnectError("refused")) is False

    def test_the_server_body_reaches_the_result_file(self):
        """Without it the record read 'Server error 500' plus a link to MDN,
        and the cause could only be found in llama.cpp's log on another host."""
        from small_llm_bench.runner import _error_text
        text = _error_text(self._http_error(500, self._REAL_BODY))
        assert text.startswith("HTTPStatusError: ")
        assert "server said:" in text
        assert "Failed to parse tool call arguments" in text

    def test_a_long_body_is_truncated(self):
        from small_llm_bench.runner import _error_text
        text = _error_text(self._http_error(500, "x" * 5000))
        assert len(text) < 1000

    def test_a_transport_error_carries_no_body(self):
        from small_llm_bench.runner import _error_text
        assert _error_text(httpx.ConnectError("refused")) == (
            "ConnectError: refused")
