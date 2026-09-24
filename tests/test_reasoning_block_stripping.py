"""Inline reasoning blocks, and the turns that get cut inside one.

Two gaps found while auditing the granite-4.2-8b run:

* `<think>.*?</think>` matches only the balanced form, and both unbalanced ones
  occur in the stored fleet — a stray `</think>` (the server forwarded the
  closer after eating the opener) reaches the graded answer in minicpm5-2b
  `tst_63` and qwen3.6-27b `kn_21`, and a stray `<think>` is what a turn cut at
  the token cap ends with.
* `multi_turn_if` is not truncation-gated, so a cut turn was graded on whatever
  text it had produced. Constraint checks are exactly what a reasoning dump
  satisfies by accident: the model recites the rules it is tracking.
"""

from __future__ import annotations

from small_llm_bench.models import TurnRecord, strip_reasoning
from small_llm_bench.modules.base import message_text
from small_llm_bench.scorer import score_multi_turn_if


class TestStripReasoning:
    def test_balanced_block(self):
        assert strip_reasoning("<think>weighing it up</think>The answer is 4") \
            == "The answer is 4"

    def test_stray_closer_drops_what_came_before_it(self):
        """The opener was stripped upstream; everything up to the closer is
        thinking."""
        assert strip_reasoning("so the final answer is the ledger\n</think>\n"
                               "Done — widget-a is now 32.") \
            == "Done — widget-a is now 32."

    def test_stray_opener_drops_what_comes_after_it(self):
        """A turn cut at the cap ends mid-thought with no closer."""
        assert strip_reasoning("Here goes.<think>wait, let me recount the") \
            == "Here goes."

    def test_a_cut_thought_with_no_answer_yields_nothing(self):
        assert strip_reasoning("<think>let me work through this care") == ""

    def test_plain_text_is_untouched(self):
        assert strip_reasoning("  OWNER: Marco  ") == "OWNER: Marco"

    def test_message_text_strips_the_unbalanced_form(self):
        assert message_text({"content": "reasoning</think>OWNER: Tomas"}) \
            == "OWNER: Tomas"

    def test_message_text_still_falls_back_to_the_reasoning_channel(self):
        assert message_text({"content": "",
                             "reasoning_content": "<think>x</think>42"}) == "42"


def _turn(content: str, *, truncated: bool = False) -> TurnRecord:
    return TurnRecord(role="assistant", content=content, truncated=truncated)


_CONVERSATION = [
    {"constraints": [{"type": "contains", "value": "Known issues:"}]},
    {"constraints": [{"type": "ends_with", "value": "Shipped."}]},
]


class TestTruncatedTurnIsNotGraded:
    def test_a_cut_turn_scores_zero_however_good_its_dump_looks(self):
        """The mt_11 shape: the cut turn's reasoning quotes the constraint."""
        turns = [_turn("Known issues: none. Shipped."),
                 _turn("...the rule says Known issues: must appear, and it must "
                       "end with Shipped. so my reply will", truncated=True)]
        r = score_multi_turn_if(_CONVERSATION, turns)
        assert r.breakdown["per_turn"] == [1.0, 0.0]
        assert r.success is False
        assert r.breakdown["detail"][1]["checks"][0]["type"] == "delivered"

    def test_the_turns_that_did_answer_keep_their_score(self):
        """Why this is per turn and not the whole-trial _TRUNCATION_GATED."""
        turns = [_turn("Known issues: none. Shipped."),
                 _turn("cut", truncated=True)]
        r = score_multi_turn_if(_CONVERSATION, turns)
        assert r.breakdown["per_turn"][0] == 1.0

    def test_an_untruncated_dialogue_is_unaffected(self):
        turns = [_turn("Known issues: none. Shipped."),
                 _turn("Known issues: one. Shipped.")]
        r = score_multi_turn_if(_CONVERSATION, turns)
        assert r.breakdown["per_turn"] == [1.0, 1.0]
        assert r.success is True
