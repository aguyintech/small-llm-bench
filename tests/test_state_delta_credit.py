"""goal_state_match grades the change, not the end string.

Regression cover for the defect the granite-4.2-8b run surfaced: on tst_57 a
model that read every file and wrote nothing scored goal_state_match 0.964 —
the seeded ledger already differs from the target in only two numbers — for a
det_score of 0.978 on a task it had not touched. Across the 22 stored runs, 52
trials ended with the world byte-identical to its seed, at a mean det of 0.752
and a best of 0.996.
"""

from __future__ import annotations

from small_llm_bench.scorer import score_state

# The real tst_57 shapes: a ledger seeded with three counts, two of which the
# task requires changing.
_LEDGER_INITIAL = ("# Stock ledger\n\n## Counts\n"
                   "- widget-a: 40\n- widget-b: 25\n- widget-c: 12\n")
_LEDGER_EXPECTED = ("# Stock ledger\n\n## Counts\n"
                    "- widget-a: 40\n- widget-b: 40\n- widget-c: 2\n")
_LEDGER_HALF = ("# Stock ledger\n\n## Counts\n"
                "- widget-a: 40\n- widget-b: 40\n- widget-c: 12\n")

_INITIAL = {"files": {"stock/ledger.md": _LEDGER_INITIAL}}
_EXPECTED = {"optimal_turns": 6,
             "expected_state": {"files": {"stock/ledger.md": _LEDGER_EXPECTED}}}


class TestDeltaCredit:
    def test_untouched_state_scores_zero_not_almost_one(self):
        """The trial that started this: five reads, no writes."""
        r = score_state(_EXPECTED, _INITIAL, _INITIAL, [])
        assert r.breakdown["goal_state_match"] == 0.0
        assert r.success is False
        # The absolute similarity is preserved for audit, and is the number
        # that used to be graded.
        assert r.breakdown["per_key_absolute"]["files"] > 0.95

    def test_correct_end_state_still_scores_one(self):
        final = {"files": {"stock/ledger.md": _LEDGER_EXPECTED}}
        r = score_state(_EXPECTED, final, _INITIAL, [])
        assert r.breakdown["goal_state_match"] == 1.0
        assert r.success is True

    def test_partial_change_sorts_between_nothing_and_everything(self):
        """widget-b fixed, widget-c left alone.

        Not "about 0.5": the credit is still a Levenshtein share, so fixing
        widget-b (25 -> 40, two chars) closes more of the gap than widget-c
        (12 -> 2, one char) and this lands at 0.80. What the rule guarantees is
        the ordering — untouched scores strictly less than partial, which
        scores strictly less than done — and that is what decides a board.
        """
        def match(text):
            return score_state(_EXPECTED, {"files": {"stock/ledger.md": text}},
                               _INITIAL, []).breakdown["goal_state_match"]

        assert match(_LEDGER_INITIAL) < match(_LEDGER_HALF) < match(_LEDGER_EXPECTED)
        assert 0.0 < match(_LEDGER_HALF) < 1.0

    def test_state_made_worse_clamps_at_zero(self):
        final = {"files": {"stock/ledger.md": "wiped\n"}}
        r = score_state(_EXPECTED, final, _INITIAL, [])
        assert r.breakdown["goal_state_match"] == 0.0

    def test_creating_a_key_from_nothing_is_unaffected(self):
        """Nothing seeded means no baseline to subtract."""
        expected = {"expected_state": {"kv": {"a": "1"}}}
        r = score_state(expected, {"kv": {"a": "1"}}, {}, [])
        assert r.breakdown["goal_state_match"] == 1.0
        assert "per_key_absolute" not in r.breakdown

    def test_refusal_task_keeps_full_credit_for_changing_nothing(self):
        """When the expectation IS the seed, leaving it alone is the goal."""
        expected = {"unchanged": ["balance"],
                    "expected_state": {"balance": 500, "refunds": []}}
        state = {"balance": 500, "refunds": []}
        r = score_state(expected, state, state, [])
        assert r.breakdown["goal_state_match"] == 1.0
        assert r.score == 1.0

    def test_refusal_task_still_penalises_touching_the_state(self):
        expected = {"expected_state": {"balance": 500}}
        r = score_state(expected, {"balance": 400}, {"balance": 500}, [])
        assert r.breakdown["goal_state_match"] == 0.0

    def test_accept_state_credits_the_gap_of_the_matching_candidate(self):
        """Delta is per candidate: the best raw match and the closest seed can
        be different alternatives, and crossing them scores a gap nobody asked
        the model to close."""
        expected = {"expected_state": {"kv": {"status": "shipped"}},
                    "accept_state": {"kv": [{"status": "dispatched"}]}}
        initial = {"kv": {"status": "dispatched"}}
        # Reached the canonical target from a seed that already satisfied the
        # alternative: full credit for the canonical gap it did close.
        r = score_state(expected, {"kv": {"status": "shipped"}}, initial, [])
        assert r.breakdown["goal_state_match"] == 1.0
        # Untouched: still satisfies the alternative, so it is correct and
        # scores through the no-gap branch rather than being zeroed.
        r = score_state(expected, initial, initial, [])
        assert r.breakdown["goal_state_match"] == 1.0
