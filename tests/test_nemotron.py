"""Offline tests for the Nemotron persona bank and grounded-voting predictor.

No network / LLM: the persona bank is built from a tiny synthetic DataFrame, and
the predictor is exercised with a fake client returning canned votes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from simbench_exp.data import SimBenchRecord
from simbench_exp.nemotron import CENSUS_REGION, PersonaBank
from simbench_exp.predict import (
    EnrichedGroundedPredictor,
    EnsemblePredictor,
    GroundedAveragingPredictor,
    GroundedVotingPredictor,
)
from simbench_exp.worldview import calibration_weights


def _bank() -> PersonaBank:
    rows = []
    # 40 adults spread across axes + a few minors that must be excluded.
    specs = [
        ("Female", 25, "never_married", "high_school", "CA"),      # West, 18-29
        ("Male", 40, "married_present", "bachelors", "NY"),        # Northeast, 30-49
        ("Female", 55, "divorced", "graduate", "TX"),             # South, 50-64
        ("Male", 70, "widowed", "less_than_9th", "IL"),           # Midwest, 65+
    ]
    for i in range(10):
        for sex, age, marital, edu, state in specs:
            rows.append(dict(uuid=f"{sex}{age}{i}", sex=sex, age=age,
                             marital_status=marital, education_level=edu,
                             state=state, occupation="x",
                             persona=f"{sex} {age} from {state}, persona #{i}."))
    for i in range(5):  # minors — must be dropped
        rows.append(dict(uuid=f"kid{i}", sex="Female", age=12, marital_status="never_married",
                         education_level="high_school", state="CA", occupation="student",
                         persona="a child"))
    return PersonaBank(pd.DataFrame(rows))


def _rec(segment: dict[str, str]) -> SimBenchRecord:
    return SimBenchRecord(
        dataset_name="OpinionQA", split="grouped",
        input_template="Q? Options: A, B", options=("A", "B"),
        human_answer={"A": 0.5, "B": 0.5}, group_prompt="", segment=segment,
    )


def test_minors_excluded():
    bank = _bank()
    assert (bank.adults["age"] >= 18).all()
    assert len(bank.adults) == 40


@pytest.mark.parametrize("seg, ok", [
    ({}, True),
    ({"AGE": "30-49"}, True),
    ({"SEX": "female"}, True),
    ({"CREGION": "West"}, True),
    ({"EDUCATION": "postgraduate"}, True),
    ({"MARITAL": "married"}, True),
    ({"MARITAL": "living with a partner"}, False),   # no Nemotron equivalent
    ({"RACE": "white"}, False),
    ({"INCOME": "$100,000 or more"}, False),
    ({"POLIDEOLOGY": "liberal"}, False),
    ({"AGE": "not-a-bin"}, False),
])
def test_matchability(seg, ok):
    assert _bank().is_matchable(seg) is ok


def test_axis_filters_select_right_personas():
    bank = _bank()
    # West region == CA in the fixture; everyone selected must be from a West state.
    panel = bank.panel_for(_rec({"CREGION": "West"}), k=5, base_seed=0)
    assert panel is not None and len(panel) == 5
    assert all("from CA" in p for p in panel)
    # 65+ bin -> only the age-70 personas (from IL).
    old = bank.panel_for(_rec({"AGE": "65+"}), k=5, base_seed=0)
    assert all(" 70 " in p for p in old)


def test_unmatchable_segment_returns_none():
    assert _bank().panel_for(_rec({"POLPARTY": "Democrat"}), k=5) is None


def test_fixed_panel_is_deterministic_and_reused():
    bank = _bank()
    a = bank.panel_for(_rec({"SEX": "female"}), k=6, base_seed=0)
    b = bank.panel_for(_rec({"SEX": "female"}), k=6, base_seed=0)
    assert a == b  # same segment -> identical electorate (cached, seeded)


def test_pop_panel_uses_whole_corpus():
    bank = _bank()
    panel = bank.panel_for(_rec({}), k=12, base_seed=0)
    assert panel is not None and len(panel) == 12


def test_census_region_map_is_sane():
    assert CENSUS_REGION["CA"] == "West"
    assert CENSUS_REGION["NY"] == "Northeast"
    assert CENSUS_REGION["TX"] == "South"
    assert CENSUS_REGION["IL"] == "Midwest"
    assert "PR" not in CENSUS_REGION  # Puerto Rico has no OpinionQA region


# --- predictor: tally + fallback, with a fake client ---------------------- #

class _FakeClient:
    """Returns a vote for option A for the first `a_votes` calls, else B."""

    def __init__(self, a_votes: int):
        self.a_votes = a_votes
        self.calls = 0

    def complete(self, messages, **kw):
        i = self.calls
        self.calls += 1
        choice = "A" if i < self.a_votes else "B"

        class _R:
            text = f'{{"choice": "{choice}"}}'
        return _R()


def test_grounded_voting_tally():
    bank = _bank()
    client = _FakeClient(a_votes=8)  # 8 A, 2 B out of k=10
    pred = GroundedVotingPredictor(client, bank=bank, k=10, alpha=0.5, max_workers=4)
    out = pred.predict(_rec({"SEX": "female"}))
    # (8+0.5)/(10+1) = 0.7727 ; (2+0.5)/11 = 0.2273
    assert out["A"] == pytest.approx(8.5 / 11)
    assert out["B"] == pytest.approx(2.5 / 11)
    assert pred.n_grounded == 1 and pred.n_fallback == 0


class _DistClient:
    """Alternates two per-persona distributions so the average is checkable."""

    def __init__(self, dists):
        self.dists = dists
        self.calls = 0

    def complete(self, messages, **kw):
        import json
        d = self.dists[self.calls % len(self.dists)]
        self.calls += 1

        class _R:
            text = json.dumps(d)
        return _R()


def test_grounded_averaging_means_per_persona_distributions():
    bank = _bank()
    # half the panel says {A:0.8,B:0.2}, half says {A:0.2,B:0.8} -> mean {A:0.5,B:0.5}
    client = _DistClient([{"A": 0.8, "B": 0.2}, {"A": 0.2, "B": 0.8}])
    pred = GroundedAveragingPredictor(client, bank=bank, k=10, max_workers=1)
    out = pred.predict(_rec({"SEX": "female"}))
    assert out["A"] == pytest.approx(0.5)
    assert out["B"] == pytest.approx(0.5)
    assert pred.n_grounded == 1


def test_grounded_averaging_preserves_spread_unlike_argmax():
    # Every persona is genuinely torn 55/45; averaging must keep ~0.55/0.45,
    # whereas an argmax tally would collapse to 1.0/0.0.
    bank = _bank()
    client = _DistClient([{"A": 0.55, "B": 0.45}])
    pred = GroundedAveragingPredictor(client, bank=bank, k=8, max_workers=1)
    out = pred.predict(_rec({"SEX": "female"}))
    assert out["A"] == pytest.approx(0.55)
    assert out["B"] == pytest.approx(0.45)


class _ConstPredictor:
    def __init__(self, dist):
        self.dist = dist
        self.name = "const"
    def predict(self, record):
        return dict(self.dist)


def test_ensemble_blends_by_weight():
    a = _ConstPredictor({"A": 1.0, "B": 0.0})
    b = _ConstPredictor({"A": 0.0, "B": 1.0})
    rec = _rec({})
    eq = EnsemblePredictor([a, b]).predict(rec)
    assert eq["A"] == pytest.approx(0.5) and eq["B"] == pytest.approx(0.5)
    w = EnsemblePredictor([a, b], weights=[0.75, 0.25]).predict(rec)
    assert w["A"] == pytest.approx(0.75) and w["B"] == pytest.approx(0.25)


def test_calibration_weights_match_target_marginal():
    # panel is 75% moderate, 25% conservative; target is 50/50 -> reweight to 50/50.
    ideologies = ["moderate", "moderate", "moderate", "conservative"]
    target = {"moderate": 0.5, "conservative": 0.5}
    w = calibration_weights(ideologies, target)
    assert sum(w) == pytest.approx(1.0)
    # weighted ideology mass should be 50/50
    mod = sum(wi for wi, ide in zip(w, ideologies) if ide == "moderate")
    con = sum(wi for wi, ide in zip(w, ideologies) if ide == "conservative")
    assert mod == pytest.approx(0.5) and con == pytest.approx(0.5)
    # each conservative (rarer) outweighs each moderate
    assert w[3] > w[0]


def test_calibration_weights_degenerate_falls_back_to_equal():
    w = calibration_weights(["moderate", "moderate"], {"liberal": 1.0})  # no overlap
    assert w == pytest.approx([0.5, 0.5])


def test_enriched_predictor_weighted_average():
    rec = _rec({})
    # persona 0 says A, persona 1 says B; weights 0.25/0.75 -> A:0.25, B:0.75
    client = _DistClient([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
    pred = EnrichedGroundedPredictor(client, ["p0", "p1"], weights=[0.25, 0.75], max_workers=1)
    out = pred.predict(rec)
    assert out["A"] == pytest.approx(0.25) and out["B"] == pytest.approx(0.75)


def test_grounded_voting_falls_back_on_unmatchable():
    bank = _bank()

    class _Fallback:
        name = "fb"
        def predict(self, record):
            return {"A": 0.9, "B": 0.1}

    pred = GroundedVotingPredictor(_FakeClient(0), bank=bank, k=10, fallback=_Fallback())
    out = pred.predict(_rec({"POLIDEOLOGY": "liberal"}))
    assert out == {"A": 0.9, "B": 0.1}
    assert pred.n_fallback == 1 and pred.n_grounded == 0
