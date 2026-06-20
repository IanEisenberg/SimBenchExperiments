"""Tests for the Ladder gate (Blum & Hardt 2015) and bootstrap threshold."""

from scrye.ladder import LadderGate, bootstrap_eta


def test_ladder_accepts_only_significant_improvement():
    gate = LadderGate(eta=1.0)
    assert gate.consider(10.0) is True   # first beats -inf
    assert gate.best == 10.0
    assert gate.consider(10.5) is False  # within eta -> reject, best unchanged
    assert gate.best == 10.0
    assert gate.consider(11.5) is True   # beats 10.0 + 1.0
    assert gate.best == 11.5


def test_ladder_counts_every_query():
    gate = LadderGate(eta=0.5)
    for s in [1.0, 2.0, 0.5, 3.0]:
        gate.consider(s)
    assert gate.k == 4  # K counts queries, not accepts


def test_bootstrap_eta_zero_for_constant_scores():
    assert bootstrap_eta([5.0] * 50, mult=2.0, seed=1) == 0.0


def test_bootstrap_eta_positive_and_scales_with_mult():
    scores = [float(i) for i in range(50)]
    e1 = bootstrap_eta(scores, mult=1.0, seed=1)
    e2 = bootstrap_eta(scores, mult=2.0, seed=1)
    assert e1 > 0.0
    assert abs(e2 - 2.0 * e1) < 1e-9
