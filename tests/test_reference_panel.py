"""The shipped reference panel must stay a faithful record of the current bench.

`results/reference/` is what a user compares their model against and what the
README's numbers are computed from. These tests fail when the code moves away
from it: a task edited without re-running the panel, a scorer change that
would silently alter published verdicts, a model the registry cannot size.
When such a change is intended, regenerate the affected files and record it in
the CHANGELOG; do not relax the test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from small_llm_bench.leaderboard import build_leaderboard, load_model_registry
from small_llm_bench.models import task_content_hash
from small_llm_bench.modules.base import load_tasks
from small_llm_bench.reporter import load_results
from small_llm_bench.rescore import rescore_bench
from small_llm_bench.runner import all_modules

_REPO_ROOT = Path(__file__).parent.parent
_PANEL = _REPO_ROOT / "results" / "reference"
_FILES = sorted(_PANEL.glob("*_raw_results_judged.json"))

# Private address ranges, home-directory paths and mDNS host names: anything a
# stored error message or config field could carry off the machine it ran on.
_PRIVATE = re.compile(r"\b(?:10\.\d+|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"
                      r"|/Users/|/home/[a-z]|\b[\w-]+\.local\b")


@pytest.fixture(scope="module")
def panel():
    return [load_results(f) for f in _FILES]


def test_the_panel_ships():
    """The README tells users to compare against this directory."""
    assert _FILES, "results/reference/ holds no judged result files"


def test_every_model_on_the_panel_resolves_to_a_size():
    """Nine of 21 rows rendered `?` until the registry was filled from model
    cards. This is the guard against that returning."""
    registry = load_model_registry(_REPO_ROOT)
    data = build_leaderboard(_PANEL, scheme="module")
    assert len(data["rows"]) == len(_FILES)
    missing = [r["model"] for r in data["rows"] if r["model"] not in registry]
    assert missing == []
    assert all(r["size"]["bucket"] is not None for r in data["rows"])


def test_no_row_is_flagged_against_the_rest():
    """The panel is one configuration. A flag here means a file was produced
    differently from the others and the comparison it anchors is not clean."""
    rows = build_leaderboard(_PANEL, scheme="module")["rows"]
    assert {r["model"]: r["comparability_mismatch"] for r in rows
            if r["comparability_mismatch"]} == {}


def test_every_stored_trial_matches_the_current_task_bank(panel):
    """A task edited after the panel was run leaves trials that answer a
    question the bank no longer asks."""
    current = {(m.name, t.id): task_content_hash(t)
               for m in all_modules() for t in load_tasks(m.name, profile="full")}
    stale = sorted({f"{b.meta.model}:{r.task_id}" for b in panel
                    for r in b.results
                    if current.get((r.module, r.task_id)) != r.task_hash})
    assert stale == []


def test_the_current_scorer_reproduces_every_published_grade(panel):
    """Re-grading the stored trials must change no verdict and no det score.
    `code` is left out without a sandbox, as `rescore` itself does."""
    moved = []
    for bench in panel:
        before = [(r.success, r.det_score) for r in bench.results]
        rescore_bench(bench)
        for (success, det), r in zip(before, bench.results):
            if r.success != success or abs((r.det_score or 0) - (det or 0)) > 1e-9:
                moved.append(f"{bench.meta.model}:{r.task_id}")
    assert moved == []


def test_nothing_private_left_the_machine_it_ran_on():
    for path in _FILES:
        text = path.read_text(encoding="utf-8")
        found = sorted(set(_PRIVATE.findall(text)))
        assert found == [], f"{path.name}: {found}"
