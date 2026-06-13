"""Unit tests for the SimBench scoring module.

These pin the two anchor properties of the score (perfect match -> 100,
uniform prediction -> 0) plus TVD math and the bootstrap CI, so any future
refactor of the metric is caught immediately.
"""

import numpy as np

from scrye.scoring import (
    aggregate_score,
    bootstrap_ci,
    simbench_score,
    total_variation_distance,
    uniform_like,
)


def test_tvd_identical_is_zero():
    p = np.array([0.5, 0.3, 0.2])
    assert total_variation_distance(p, p) == 0.0


def test_tvd_disjoint_is_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert total_variation_distance(p, q) == 1.0


def test_tvd_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.4, 0.5])
    assert total_variation_distance(p, q) == total_variation_distance(q, p)


def test_perfect_prediction_scores_100():
    truth = {"A": 0.6, "B": 0.3, "C": 0.1}
    assert simbench_score(truth, truth) == 100.0


def test_uniform_prediction_scores_0():
    truth = {"A": 0.6, "B": 0.3, "C": 0.1}
    pred = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert abs(simbench_score(pred, truth)) < 1e-9


def test_worse_than_uniform_is_negative():
    # Prediction further from truth (in TVD) than uniform is -> negative score.
    truth = {"A": 0.6, "B": 0.3, "C": 0.1}
    pred = {"A": 0.0, "B": 0.0, "C": 1.0}
    assert simbench_score(pred, truth) < 0.0


def test_uniform_truth_returns_baseline_zero():
    # When ground truth IS uniform, TVD(P,U)=0 and S is defined as 0.
    truth = {"A": 0.5, "B": 0.5}
    pred = {"A": 0.9, "B": 0.1}
    assert simbench_score(pred, truth) == 0.0


def test_missing_labels_treated_as_zero_mass():
    truth = {"A": 0.5, "B": 0.3, "C": 0.2}
    pred = {"A": 0.5, "B": 0.3}  # C missing -> 0
    # Should not raise and should be a finite score below 100.
    s = simbench_score(pred, truth)
    assert -200.0 < s < 100.0


def test_options_order_does_not_change_score():
    truth = {"A": 0.6, "B": 0.3, "C": 0.1}
    pred = {"A": 0.4, "B": 0.4, "C": 0.2}
    s1 = simbench_score(pred, truth, options=["A", "B", "C"])
    s2 = simbench_score(pred, truth, options=["C", "B", "A"])
    assert abs(s1 - s2) < 1e-9


def test_prediction_renormalized():
    # Unnormalized prediction (sums to 2) should be treated as a distribution.
    truth = {"A": 0.6, "B": 0.4}
    pred_norm = {"A": 0.6, "B": 0.4}
    pred_unnorm = {"A": 1.2, "B": 0.8}
    assert abs(simbench_score(pred_norm, truth) - simbench_score(pred_unnorm, truth)) < 1e-9


def test_aggregate_is_mean():
    assert aggregate_score([10.0, 20.0, 30.0]) == 20.0


def test_bootstrap_ci_brackets_mean():
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    mean, lo, hi = bootstrap_ci(scores, n_boot=500, seed=1)
    assert mean == 30.0
    assert lo <= mean <= hi


def test_uniform_like():
    p = np.array([0.1, 0.2, 0.7])
    u = uniform_like(p)
    assert np.allclose(u, np.array([1 / 3, 1 / 3, 1 / 3]))
