"""Tests for the recalibration calibrators the levers compose."""

import numpy as np

import pytest

from scrye.calibrate import (
    AbstainCalibrator,
    ChainCalibrator,
    DirichletCalibrator,
    EntropyTargetCalibrator,
    EntropyTempScaling,
    FeatureEntropyTargetCalibrator,
    IdentityCalibrator,
    TempScaling,
    make_calibrator,
    temper_to_entropy,
)
from scrye.data import SimBenchRecord
from scrye.scoring import response_entropy


def _rec(options, truth, dataset="ESS"):
    return SimBenchRecord(
        dataset_name=dataset, split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def test_temp_identity_at_one():
    rec = _rec(["A", "B"], {"A": 0.6, "B": 0.4})
    out = TempScaling(T=1.0).transform(rec, {"A": 0.7, "B": 0.3})
    assert abs(out["A"] - 0.7) < 1e-9 and abs(out["B"] - 0.3) < 1e-9


def test_temp_high_flattens_toward_uniform():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    out = TempScaling(T=50.0).transform(rec, {"A": 0.9, "B": 0.1})
    assert abs(out["A"] - 0.5) < 0.05  # nearly uniform


def test_temp_low_sharpens():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    out = TempScaling(T=0.2).transform(rec, {"A": 0.7, "B": 0.3})
    assert out["A"] > 0.7  # the larger mass grows


def test_dirichlet_smooths_toward_uniform():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    out = DirichletCalibrator(alpha=1.0).transform(rec, {"A": 1.0, "B": 0.0})
    assert 0.0 < out["B"] < 0.5  # zero mass pulled up


def test_entropy_temp_flattens_high_entropy_more():
    # A near-uniform prediction (high entropy) should be flattened more than a
    # peaked one under the same positive slope.
    rec = _rec(["A", "B", "C", "D"], {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    cal = EntropyTempScaling(slope=2.0)
    peaked = cal.transform(rec, {"A": 0.97, "B": 0.01, "C": 0.01, "D": 0.01})
    flat = cal.transform(rec, {"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.20})
    # The peaked input keeps more of its peak than the flat input keeps its spread shape.
    assert peaked["A"] > flat["A"]


def test_chain_applies_in_order():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    chain = ChainCalibrator([TempScaling(T=2.0), DirichletCalibrator(alpha=0.0)])
    out = chain.transform(rec, {"A": 0.9, "B": 0.1})
    assert abs(sum(out.values()) - 1.0) < 1e-9 and out["A"] < 0.9  # flattened


def test_temp_fit_recovers_flattening_temperature():
    # Raw predictions are systematically over-sharp vs truth; fit should pick T>1.
    recs, raws = [], []
    for _ in range(30):
        recs.append(_rec(["A", "B"], {"A": 0.55, "B": 0.45}))
        raws.append({"A": 0.95, "B": 0.05})
    cal = TempScaling().fit(recs, raws)
    assert cal.T > 1.0


def test_abstain_calibrator_falls_back_to_uniform_on_bad_datasets():
    truth = {"A": 0.7, "B": 0.2, "C": 0.1}
    # GOOD dataset: predictions close to truth (better than uniform)
    good = [(_rec(["A", "B", "C"], truth, dataset="GOOD"),
             {"A": 0.65, "B": 0.22, "C": 0.13}) for _ in range(6)]
    # BAD dataset: predictions on the wrong option (worse than uniform)
    bad = [(_rec(["A", "B", "C"], truth, dataset="BAD"),
            {"A": 0.05, "B": 0.05, "C": 0.90}) for _ in range(6)]
    recs = [r for r, _ in good + bad]
    raws = [p for _, p in good + bad]

    cal = AbstainCalibrator().fit(recs, raws)
    assert "BAD" in cal.datasets_ and "GOOD" not in cal.datasets_
    # a BAD-dataset record is replaced with uniform
    out_bad = cal.transform(*bad[0])
    assert out_bad == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    # a GOOD-dataset record is passed through unchanged
    assert cal.transform(*good[0]) == pytest.approx(good[0][1])


def test_abstain_calibrator_unfit_is_passthrough():
    cal = AbstainCalibrator()
    pred = {"A": 0.6, "B": 0.4}
    assert cal.transform(_rec(["A", "B"], {"A": 0.5, "B": 0.5}), pred) == pytest.approx(pred)


def test_make_calibrator_dispatch():
    assert isinstance(make_calibrator("temp", T=1.5), TempScaling)
    assert isinstance(make_calibrator("entropy_temp", slope=1.0), EntropyTempScaling)
    assert isinstance(make_calibrator("dirichlet", alpha=0.5), DirichletCalibrator)
    assert isinstance(make_calibrator("identity"), IdentityCalibrator)
    assert isinstance(make_calibrator("entropy_target", gain=1.0), EntropyTargetCalibrator)
    assert isinstance(make_calibrator("abstain"), AbstainCalibrator)


# --- temper_to_entropy ----------------------------------------------------
def test_temper_to_entropy_hits_target_both_directions():
    # raise the entropy of a sharp distribution
    sharp = {"A": 0.9, "B": 0.07, "C": 0.03}
    up = temper_to_entropy(sharp, 0.8)
    assert response_entropy(up) == __import__("pytest").approx(0.8, abs=0.02)
    # lower the entropy of a near-uniform distribution
    flat = {"A": 0.34, "B": 0.33, "C": 0.33}
    down = temper_to_entropy(flat, 0.3)
    assert response_entropy(down) == __import__("pytest").approx(0.3, abs=0.02)
    # ranking is preserved (tempering is monotone)
    assert list(up) == ["A", "B", "C"] and up["A"] > up["B"] > up["C"]


# --- EntropyTargetCalibrator ----------------------------------------------
def _compressed_pairs(n=200):
    """Synthetic data where the prediction entropy is compressed toward ~0.7:
    truth ranges over consensus..contested, pred is a flattened version."""
    rng = np.random.default_rng(0)
    recs, raws = [], []
    for _ in range(n):
        # truth: a 4-option dist with a controllable peak
        peak = rng.uniform(0.3, 0.95)
        rest = (1 - peak) / 3
        truth = {"A": peak, "B": rest, "C": rest, "D": rest}
        rng_order = ["A", "B", "C", "D"]
        recs.append(_rec(rng_order, truth))
        # prediction: compressed toward uniform (always too flat on sharp truths)
        ppeak = 0.4 + 0.3 * peak           # slope 0.3 -> heavy compression
        prest = (1 - ppeak) / 3
        raws.append({"A": ppeak, "B": prest, "C": prest, "D": prest})
    return recs, raws


def test_entropy_target_fit_learns_decompression_and_raises_score():
    recs, raws = _compressed_pairs()
    cal = EntropyTargetCalibrator().fit(recs, raws)
    # the regression recovers a positive truth~pred relationship and a de-compressing
    # (>1) slope, since predicted entropy is compressed toward the mean
    assert cal.slope_ > 1.0
    assert cal.gain > 0                          # must de-compress, not sit at identity
    # de-compression EXPANDS the spread of predicted entropies (undoes compression)
    h_before = np.array([response_entropy(r) for r in raws])
    h_after = np.array([response_entropy(cal.transform(rec, r))
                        for rec, r in zip(recs, raws)])
    assert h_after.std() > h_before.std()
    # and mean SimBench score improves vs identity
    from scrye.scoring import simbench_score
    base = np.mean([simbench_score(r, rec.human_answer) for rec, r in zip(recs, raws)])
    cald = np.mean([simbench_score(cal.transform(rec, r), rec.human_answer)
                    for rec, r in zip(recs, raws)])
    assert cald > base


def _dataset_keyed_pairs(n=240):
    """Two datasets with opposite true spreads; the prediction is compressed
    (medium entropy) for both, so the dataset id — not the prediction — carries
    the contestedness signal a feature model can exploit."""
    rng = np.random.default_rng(1)
    recs, raws = [], []
    for i in range(n):
        sharp = (i % 2 == 0)
        ds = "SHARP" if sharp else "FLAT"
        if sharp:
            peak = rng.uniform(0.8, 0.95)        # low-entropy truth
        else:
            peak = rng.uniform(0.28, 0.4)        # high-entropy truth
        rest = (1 - peak) / 3
        truth = {"A": peak, "B": rest, "C": rest, "D": rest}
        rec = SimBenchRecord(dataset_name=ds, split="grouped", input_template="Q?",
                             options=("A", "B", "C", "D"), human_answer=truth,
                             group_prompt="")
        recs.append(rec)
        # compressed prediction: medium entropy regardless of dataset
        ppeak = rng.uniform(0.5, 0.6)
        prest = (1 - ppeak) / 3
        raws.append({"A": ppeak, "B": prest, "C": prest, "D": prest})
    return recs, raws


def test_feature_entropy_target_uses_features_to_beat_own_entropy():
    recs, raws = _dataset_keyed_pairs()
    cal = FeatureEntropyTargetCalibrator().fit(recs, raws)
    assert cal.gain > 0 and cal.w_ is not None
    # tempering to the feature-predicted target expands the entropy spread and
    # improves score, where own-entropy de-compression (no dataset signal) cannot
    from scrye.scoring import simbench_score
    base = np.mean([simbench_score(r, rec.human_answer) for rec, r in zip(recs, raws)])
    cald = np.mean([simbench_score(cal.transform(rec, r), rec.human_answer)
                    for rec, r in zip(recs, raws)])
    assert cald > base + 1.0
    # SHARP-dataset items get sharpened relative to FLAT-dataset items
    sharp = [cal.transform(rec, r) for rec, r in zip(recs, raws) if rec.dataset_name == "SHARP"]
    flat = [cal.transform(rec, r) for rec, r in zip(recs, raws) if rec.dataset_name == "FLAT"]
    assert np.mean([response_entropy(d) for d in sharp]) < \
           np.mean([response_entropy(d) for d in flat])


def test_feature_entropy_target_unknown_dataset_does_not_crash():
    recs, raws = _dataset_keyed_pairs(40)
    cal = FeatureEntropyTargetCalibrator().fit(recs, raws)
    other = _rec(["A", "B", "C"], {"A": 0.5, "B": 0.3, "C": 0.2})  # dataset 'ESS' unseen
    out = cal.transform(other, {"A": 0.5, "B": 0.3, "C": 0.2})
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_entropy_target_gain_zero_is_identity():
    recs, raws = _compressed_pairs(50)
    cal = EntropyTargetCalibrator(gain=0.0)
    cal.slope_, cal.intercept_ = 1.0, 0.0   # bypass fit
    pred = {"A": 0.5, "B": 0.3, "C": 0.2}
    out = cal.transform(_rec(["A", "B", "C"], {"A": 0.5, "B": 0.3, "C": 0.2}), pred)
    assert out == __import__("pytest").approx(pred, abs=1e-9)
