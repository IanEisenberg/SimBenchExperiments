"""Tests for post-stratification weights, distribution tables, and the combiner.

Synthetic records exercise the math exactly; an optional real-data test pins the
verified property that group_size-weighted subgroup truths reproduce the
marginal truth.
"""

import pytest

from simbench_exp.data import SimBenchRecord, load_split
from simbench_exp.distributions import (
    DistributionRow,
    SegmentWeights,
    build_segment_weights,
    distribution_table,
)
from simbench_exp.predict import PostStratificationPredictor, Predictor


def _rec(segment, truth, grouping_keys, group_size, q="Q?", dataset="ESS"):
    return SimBenchRecord(
        dataset_name=dataset,
        split="grouped",
        input_template=q,
        options=tuple(truth.keys()),
        human_answer=dict(truth),
        group_prompt="",
        segment=dict(segment),
        grouping_keys=tuple(sorted(grouping_keys)),
        num_grouping_vars=len(segment),
        group_size=group_size,
    )


def _country_question():
    """A country marginal plus its gender decomposition for one question."""
    marginal = _rec({"country": "Kenya"}, {"A": 0.6, "B": 0.4}, ("country",), 100)
    male = _rec({"country": "Kenya", "gender": "male"}, {"A": 0.8, "B": 0.2}, ("country", "gender"), 60)
    female = _rec({"country": "Kenya", "gender": "female"}, {"A": 0.3, "B": 0.7}, ("country", "gender"), 40)
    return marginal, male, female


# -- SegmentWeights --------------------------------------------------------
def test_children_weights_sum_to_one_and_match_group_size():
    marginal, male, female = _country_question()
    weights = SegmentWeights([marginal, male, female])
    children = weights.children(marginal, over="gender")
    ws = {child.segment["gender"]: w for w, child in children}
    assert abs(sum(ws.values()) - 1.0) < 1e-12
    assert ws["male"] == pytest.approx(0.6)
    assert ws["female"] == pytest.approx(0.4)


def test_decompose_variables_lists_gender():
    marginal, male, female = _country_question()
    weights = SegmentWeights([marginal, male, female])
    assert weights.decompose_variables(marginal) == ["gender"]


def test_children_empty_when_no_decomposition():
    marginal, male, female = _country_question()
    weights = SegmentWeights([marginal, male, female])
    # the leaf cells cannot be decomposed further
    assert weights.children(male) == []


def test_auto_pick_decomposition_variable():
    marginal, male, female = _country_question()
    weights = build_segment_weights([marginal, male, female])
    children = weights.children(marginal)  # over=None -> auto
    assert {c.segment["gender"] for _, c in children} == {"male", "female"}


def test_recombined_truth_reproduces_marginal_synthetic():
    marginal, male, female = _country_question()
    weights = SegmentWeights([marginal, male, female])
    children = weights.children(marginal, over="gender")
    combined = {o: 0.0 for o in marginal.options}
    for w, child in children:
        for o in marginal.options:
            combined[o] += w * child.human_answer[o]
    # 0.6*0.8 + 0.4*0.3 = 0.60 ; 0.6*0.2 + 0.4*0.7 = 0.40
    assert combined["A"] == pytest.approx(marginal.human_answer["A"])
    assert combined["B"] == pytest.approx(marginal.human_answer["B"])


# -- distribution_table ----------------------------------------------------
def test_distribution_table_weights_sum_to_one_per_subset():
    marginal, male, female = _country_question()
    rows = distribution_table([marginal, male, female])
    assert all(isinstance(r, DistributionRow) for r in rows)
    by_group: dict[tuple, float] = {}
    for r in rows:
        by_group.setdefault(r.grouping_keys, 0.0)
        by_group[r.grouping_keys] += r.weight
    for total in by_group.values():
        assert total == pytest.approx(1.0)


# -- PostStratificationPredictor -------------------------------------------
class _CellPredictor(Predictor):
    """Predicts each child's own truth; lets us check recombination exactly."""

    name = "cell"

    def predict(self, record):
        return dict(record.human_answer)


def test_post_strat_recombines_to_marginal():
    marginal, male, female = _country_question()
    weights = SegmentWeights([marginal, male, female])
    predictor = PostStratificationPredictor(_CellPredictor(), weights, over="gender")
    out = predictor.predict(marginal)
    assert out["A"] == pytest.approx(0.6)
    assert out["B"] == pytest.approx(0.4)
    assert predictor.n_decomposed == 1
    assert predictor.n_fallback == 0


def test_post_strat_falls_back_without_children():
    marginal, male, female = _country_question()
    weights = SegmentWeights([marginal, male, female])
    predictor = PostStratificationPredictor(_CellPredictor(), weights)
    out = predictor.predict(male)  # a leaf cell -> fall back to base on itself
    assert out == pytest.approx(male.human_answer)
    assert predictor.n_fallback == 1


# -- real-data property (skipped if SimBench not downloaded) ---------------
def test_recombination_reproduces_marginal_on_real_data():
    try:
        grouped = load_split("grouped", download=False)
    except FileNotFoundError:
        pytest.skip("SimBench grouped split not downloaded")

    weights = SegmentWeights(grouped)
    predictor = PostStratificationPredictor(_CellPredictor(), weights, over="gender")

    checked = 0
    for rec in grouped:
        if rec.grouping_keys != ("country",):
            continue
        children = weights.children(rec, over="gender")
        if len(children) < 2:
            continue
        out = predictor.predict(rec)
        tvd = 0.5 * sum(abs(out[o] - rec.human_answer.get(o, 0.0)) for o in rec.options)
        assert tvd < 0.02, f"{rec.dataset_name} {rec.segment}: TVD={tvd}"
        checked += 1
        if checked >= 50:
            break
    assert checked > 0, "no country marginals with a gender decomposition found"
