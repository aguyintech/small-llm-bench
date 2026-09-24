"""Tests for pure CLI logic (not the Typer command wiring itself)."""

from __future__ import annotations

from small_llm_bench.cli import apply_reuse_params
from small_llm_bench.config import BenchSettings
from small_llm_bench.models import BenchMeta, BenchResult


def _previous(**meta_kwargs) -> BenchResult:
    meta = BenchMeta(model="m", endpoint="http://old/v1", timestamp="",
                     duration_seconds=0.0, bench_version="0",
                     temperature=0.3, thinking=True, **meta_kwargs)
    return BenchResult(meta=meta, results=[])


def test_reuse_params_none_leaves_settings_untouched():
    settings = BenchSettings(endpoint="http://default/v1")
    apply_reuse_params(settings, None, endpoint=None, temperature=None, thinking=False)
    assert settings.endpoint == "http://default/v1"
    assert settings.temperature is None
    assert settings.thinking is None


def test_reuse_params_adopts_previous_config_when_not_passed():
    settings = BenchSettings(endpoint="http://default/v1")
    apply_reuse_params(settings, _previous(), endpoint=None, temperature=None, thinking=False)
    assert settings.endpoint == "http://old/v1"
    assert settings.temperature == 0.3
    assert settings.thinking is True


def test_reuse_params_explicit_flags_win_over_previous():
    settings = BenchSettings(endpoint="http://default/v1")
    apply_reuse_params(settings, _previous(), endpoint="http://explicit/v1",
                       temperature=0.9, thinking=False)
    assert settings.endpoint == "http://explicit/v1"
    assert settings.temperature == 0.9
    # thinking's CLI flag has no "explicit off" state, so a previous
    # thinking=True is still adopted when --thinking wasn't passed.
    assert settings.thinking is True


def test_reuse_params_keeps_todays_budget_when_previous_did_not_override():
    """A non-overriding previous run must not drag today's budget back down.

    This is the raise-and-retry path: `_config_matches` accepts a previous run
    whose cap was smaller, so keeping the bigger budget lets --only-new retry
    old truncated trials without discarding every good trial in the file.
    """
    settings = BenchSettings(max_tokens=8192)
    reused = apply_reuse_params(settings, _previous(max_tokens=4096),
                                endpoint=None, temperature=None, thinking=False)
    assert settings.max_tokens == 8192
    assert reused is False


def test_reuse_params_adopts_a_recorded_max_tokens_override():
    """The whole point of the flag: reproduce the config so trials are reusable.

    `_config_matches` returns False when the stored run has
    max_tokens_override and this run does not, so without adopting it a
    3-trial top-up silently swept the entire 177-trial bank instead.
    """
    settings = BenchSettings(max_tokens=8192)
    reused = apply_reuse_params(settings,
                                _previous(max_tokens=16384,
                                          max_tokens_override=True),
                                endpoint=None, temperature=None, thinking=False)
    assert settings.max_tokens == 16384
    assert reused is True


def test_reuse_params_override_is_not_adopted_without_a_previous_run():
    settings = BenchSettings(max_tokens=8192)
    reused = apply_reuse_params(settings, None, endpoint=None,
                                temperature=None, thinking=False)
    assert settings.max_tokens == 8192
    assert reused is False


def test_adopted_override_makes_the_previous_config_match():
    """End to end on the actual defect: adopting the override is what makes
    `_config_matches` accept the file, which is what makes --only-new reuse."""
    from small_llm_bench.runner import _config_matches

    previous = _previous(max_tokens=16384, max_tokens_override=True)
    settings = BenchSettings(max_tokens=8192)
    settings.model, settings.endpoint = "m", "http://old/v1"

    # Before: the stored run overrides, this one does not — refused.
    assert _config_matches(previous.meta, settings,
                           max_tokens_explicit=False) is False

    reused = apply_reuse_params(settings, previous, endpoint=None,
                                temperature=None, thinking=False)
    assert _config_matches(previous.meta, settings,
                           max_tokens_explicit=reused) is True


def test_judge_command_exits_2_on_incomplete_coverage(tmp_path, monkeypatch):
    """A judged file with coverage gaps is not comparable to a complete one, so
    the command must fail loudly — while still writing the file."""
    import json

    from typer.testing import CliRunner

    import small_llm_bench.cli as cli_module
    from small_llm_bench.models import TaskResult

    def _bench(judged_modules: set[str]) -> BenchResult:
        meta = BenchMeta(model="m", endpoint="http://x/v1", timestamp="t",
                         duration_seconds=1.0, bench_version="0.6.0")
        results = [
            TaskResult(task_id="cd_01", module="code", prompt="p", det_score=1.0),
            TaskResult(task_id="tl_01", module="tools", prompt="p", det_score=0.5),
        ]
        for r in results:
            if r.module in judged_modules:
                r.llm_score = 1.0
        return BenchResult(meta=meta, results=results)

    raw = tmp_path / "m_raw_results.json"
    raw.write_text(json.dumps(_bench(set()).model_dump()))
    out = tmp_path / "m_raw_results_judged.json"

    # tool_loop stays unjudged, as if its batched call had 503'd
    monkeypatch.setattr(cli_module, "judge_results",
                        lambda bench, settings: _bench({"code"}))
    monkeypatch.setattr(cli_module.asyncio, "run", lambda coro: coro)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["judge", "--results", str(raw)])
    assert result.exit_code == 2
    assert out.exists()  # work is never thrown away

    allowed = runner.invoke(cli_module.app,
                            ["judge", "--results", str(raw), "--allow-partial"])
    assert allowed.exit_code == 0
