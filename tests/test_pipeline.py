"""Tests for the swappable pipeline, scoring helpers, and counterfactual metric.

A fake predictor lets us exercise the full chain (Predictor -> Calibrator ->
Pipeline -> evaluate -> counterfactual_sensitivity) with no network access.
"""

import numpy as np
import pandas as pd

from simbench_exp.calibrate import Calibrator, IdentityCalibrator
from simbench_exp.data import SimBenchRecord
from simbench_exp.evaluate import (
    build_normalizers,
    counterfactual_sensitivity,
    evaluate,
    required_question_records,
    stratified_sample,
    summarize,
)
from simbench_exp.pipeline import Pipeline
from simbench_exp.predict import Predictor, UniformPredictor
from simbench_exp.scoring import delta_alignment, response_entropy


def _rec(dataset, options, truth, segment=None, group_size=100, q="Q?"):
    return SimBenchRecord(
        dataset_name=dataset,
        split="grouped",
        input_template=q,
        options=tuple(options),
        human_answer=dict(truth),
        group_prompt="",
        segment=segment or {},
        num_grouping_vars=len(segment or {}),
        group_size=group_size,
    )


class FixedPredictor(Predictor):
    """Returns a preset distribution (keyed by record id via a dict)."""

    name = "fixed"

    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, record):
        return dict(self.mapping[id(record)])


# -- entropy ---------------------------------------------------------------
def test_entropy_uniform_is_one():
    assert abs(response_entropy({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) - 1.0) < 1e-9


def test_entropy_onehot_is_zero():
    assert response_entropy({"A": 1.0, "B": 0.0}) == 0.0


# -- delta alignment -------------------------------------------------------
def test_delta_alignment_same_direction():
    assert abs(delta_alignment(np.array([0.1, -0.1]), np.array([0.2, -0.2])) - 1.0) < 1e-9


def test_delta_alignment_opposite():
    assert abs(delta_alignment(np.array([0.1, -0.1]), np.array([-0.2, 0.2])) + 1.0) < 1e-9


def test_delta_alignment_zero_shift_is_nan():
    assert np.isnan(delta_alignment(np.array([0.0, 0.0]), np.array([0.1, -0.1])))


# -- pipeline --------------------------------------------------------------
def test_pipeline_identity_passthrough():
    rec = _rec("ESS", ["A", "B"], {"A": 0.7, "B": 0.3})
    pipe = Pipeline(FixedPredictor({id(rec): {"A": 0.6, "B": 0.4}}))
    assert pipe.name == "fixed+identity"
    assert pipe.predict(rec) == {"A": 0.6, "B": 0.4}


def test_pipeline_calibrator_applied():
    rec = _rec("ESS", ["A", "B"], {"A": 0.7, "B": 0.3})

    class FlipCalibrator(Calibrator):
        name = "flip"

        def transform(self, record, pred):
            return {"A": pred["B"], "B": pred["A"]}

    pipe = Pipeline(FixedPredictor({id(rec): {"A": 0.6, "B": 0.4}}), FlipCalibrator())
    assert pipe.predict(rec) == {"A": 0.4, "B": 0.6}


def test_evaluate_produces_scores():
    recs = [
        _rec("ESS", ["A", "B"], {"A": 0.7, "B": 0.3}),
        _rec("ISSP", ["A", "B", "C"], {"A": 0.5, "B": 0.3, "C": 0.2}),
    ]
    pipe = Pipeline(UniformPredictor())
    df = evaluate(pipe, recs, progress=False)
    assert len(df) == 2
    assert {"score", "truth_entropy", "dataset"} <= set(df.columns)
    # Uniform predictor scores ~0 by construction.
    assert abs(df["score"].mean()) < 1e-6


def test_uniform_scores_zero_aggregate_eq2():
    # SimBench baseline-by-construction: under Eq. 2 (dataset-level normalizer),
    # the uniform predictor must average to 0 over a dataset, even though
    # individual instances are nonzero. This reproduces the paper's defining
    # property of the score.
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(40):
        a = rng.random()
        recs.append(_rec("ESS", ["A", "B"], {"A": float(a), "B": float(1 - a)}))
    normalizers = build_normalizers(recs)
    df = evaluate(Pipeline(UniformPredictor()), recs, normalizers=normalizers, progress=False)
    # Individual scores are NOT all zero...
    assert df["score"].abs().max() > 1.0
    # ...but the dataset mean is 0 by construction.
    assert abs(df["score"].mean()) < 1e-9


def test_build_normalizers_keys_by_split_dataset():
    recs = [
        _rec("ESS", ["A", "B"], {"A": 0.9, "B": 0.1}),
        _rec("ISSP", ["A", "B"], {"A": 0.5, "B": 0.5}),
    ]
    norms = build_normalizers(recs)
    assert ("grouped", "ESS") in norms
    assert ("grouped", "ISSP") in norms
    # ISSP truth is uniform -> TVD-to-uniform 0; ESS is skewed -> positive.
    assert norms[("grouped", "ISSP")] == 0.0
    assert norms[("grouped", "ESS")] > 0.0


def test_summarize_keys():
    recs = [_rec("ESS", ["A", "B"], {"A": 0.7, "B": 0.3})]
    df = evaluate(Pipeline(UniformPredictor()), recs, progress=False)
    s = summarize(df)
    assert {"mean_score", "ci_low", "ci_high", "n", "frac_below_uniform"} <= set(s)


# -- counterfactual sensitivity --------------------------------------------
def test_counterfactual_perfect_direction():
    # Two segments whose truths diverge from the pooled mean; a predictor that
    # nails the direction should score alignment ~1.
    q = "Trust?"
    r1 = _rec("ESS", ["A", "B"], {"A": 0.8, "B": 0.2}, segment={"g": "x"}, q=q)
    r2 = _rec("ESS", ["A", "B"], {"A": 0.2, "B": 0.8}, segment={"g": "y"}, q=q)
    preds = {id(r1): {"A": 0.7, "B": 0.3}, id(r2): {"A": 0.3, "B": 0.7}}
    df = evaluate(Pipeline(FixedPredictor(preds)), [r1, r2], progress=False)
    mean_align, table = counterfactual_sensitivity(df)
    assert mean_align > 0.99
    assert len(table) == 2


def test_counterfactual_wrong_direction_negative():
    q = "Trust?"
    r1 = _rec("ESS", ["A", "B"], {"A": 0.8, "B": 0.2}, segment={"g": "x"}, q=q)
    r2 = _rec("ESS", ["A", "B"], {"A": 0.2, "B": 0.8}, segment={"g": "y"}, q=q)
    # Predictions shift the OPPOSITE way.
    preds = {id(r1): {"A": 0.3, "B": 0.7}, id(r2): {"A": 0.7, "B": 0.3}}
    df = evaluate(Pipeline(FixedPredictor(preds)), [r1, r2], progress=False)
    mean_align, _ = counterfactual_sensitivity(df)
    assert mean_align < -0.99


# -- sampling helpers ------------------------------------------------------
def test_stratified_sample_is_seeded_and_spread():
    recs = [_rec("ESS", ["A", "B"], {"A": 0.6, "B": 0.4}) for _ in range(20)]
    recs += [_rec("ISSP", ["A", "B"], {"A": 0.5, "B": 0.5}) for _ in range(20)]
    s1 = stratified_sample(recs, 10, seed=1)
    s2 = stratified_sample(recs, 10, seed=1)
    assert [id(r) for r in s1] == [id(r) for r in s2]  # deterministic
    assert {r.dataset_name for r in s1} == {"ESS", "ISSP"}  # both datasets present


def test_required_question_matching():
    recs = [
        _rec("LatinoBarometro", ["A"], {"A": 1.0}, q="...trust in the president?"),
        _rec("ESS", ["A"], {"A": 1.0}, q="...free to live their own life..."),
        _rec("ESS", ["A"], {"A": 1.0}, q="...use the internet on devices..."),
        _rec("Other", ["A"], {"A": 1.0}, q="unrelated question"),
    ]
    groups = required_question_records(recs)
    assert len(groups["trust_president"]) == 1
    assert len(groups["gay_rights"]) == 1
    assert len(groups["internet_use"]) == 1
