"""Tests for task-context prompting (Stage 14).

Variants add the task context the original respondents actually had — a faithful
task brief and/or sibling items from the same instrument — on top of the
calibrated_commitment distributional ask, isolating the *context* effect.
"""

from scrye.data import SimBenchRecord
from scrye.persona import (
    CalibratedCommitmentStrategy,
    TaskContextStrategy,
    build_item_corpus,
)


def _rec(stem, ds="OSPsychMACH", options=("A", "B", "C", "D", "E")):
    return SimBenchRecord(
        dataset_name=ds,
        split="pop",
        input_template=stem,
        options=tuple(options),
        human_answer={o: 1.0 / len(options) for o in options},
        group_prompt="You are a user of openpsychometrics.org.",
        segment={},
        num_grouping_vars=0,
        group_size=100,
    )


ITEMS = [
    "Indicate agreement: Honesty is the best policy.\nOptions: (A): Disagree ...",
    "Indicate agreement: Anyone who trusts anyone is asking for trouble.\nOptions: (A): ...",
    "Indicate agreement: It is hard to get ahead without cutting corners.\nOptions: (A): ...",
    "Indicate agreement: Most people are basically good.\nOptions: (A): ...",
]


def test_corpus_collects_unique_stems_per_dataset():
    recs = [_rec(s) for s in ITEMS] + [_rec(ITEMS[0])]  # one duplicate
    corpus = build_item_corpus(recs)
    assert set(corpus) == {"OSPsychMACH"}
    assert len(corpus["OSPsychMACH"]) == len(ITEMS)  # dedup


def test_items_mode_includes_siblings_and_excludes_target():
    corpus = build_item_corpus([_rec(s) for s in ITEMS])
    strat = TaskContextStrategy(corpus, mode="items", k=3)
    target = _rec(ITEMS[1])
    user = strat.build_messages(target)[-1]["content"]
    assert "Honesty is the best policy" in user            # a sibling shown
    # the target's own statement appears once (as the question), not duplicated
    assert user.count("Anyone who trusts anyone is asking for trouble") == 1


def test_brief_mode_adds_task_description_no_siblings():
    corpus = build_item_corpus([_rec(s) for s in ITEMS])
    briefs = {"OSPsychMACH": "the MACH-IV Machiavellianism scale"}
    strat = TaskContextStrategy(corpus, mode="brief", briefs=briefs)
    user = strat.build_messages(_rec(ITEMS[1]))[-1]["content"]
    assert "MACH-IV Machiavellianism scale" in user
    assert "Honesty is the best policy" not in user         # no siblings in brief mode


def test_both_mode_has_brief_and_siblings():
    corpus = build_item_corpus([_rec(s) for s in ITEMS])
    briefs = {"OSPsychMACH": "the MACH-IV scale"}
    strat = TaskContextStrategy(corpus, mode="both", briefs=briefs, k=2)
    user = strat.build_messages(_rec(ITEMS[1]))[-1]["content"]
    assert "MACH-IV scale" in user
    assert "Honesty is the best policy" in user


def test_keeps_calibrated_commitment_system_and_ask():
    corpus = build_item_corpus([_rec(s) for s in ITEMS])
    strat = TaskContextStrategy(corpus, mode="items")
    msgs = strat.build_messages(_rec(ITEMS[0]))
    assert msgs[0]["content"] == CalibratedCommitmentStrategy.SYSTEM
    assert "answer distribution" in msgs[-1]["content"].lower()


def test_deterministic_sibling_selection():
    corpus = build_item_corpus([_rec(s) for s in ITEMS])
    strat = TaskContextStrategy(corpus, mode="items", k=2)
    a = strat.build_messages(_rec(ITEMS[2]))[-1]["content"]
    b = strat.build_messages(_rec(ITEMS[2]))[-1]["content"]
    assert a == b


def test_no_siblings_available_degrades_gracefully():
    corpus = build_item_corpus([_rec(ITEMS[0])])  # only the target's own stem
    strat = TaskContextStrategy(corpus, mode="items")
    user = strat.build_messages(_rec(ITEMS[0]))[-1]["content"]
    # still a valid prompt with the question and the distributional ask
    assert "answer distribution" in user.lower()
