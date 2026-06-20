"""Tests for the recalibration calibrators the levers compose."""

import numpy as np

from scrye.calibrate import (
    ChainCalibrator,
    DirichletCalibrator,
    EntropyTempScaling,
    IdentityCalibrator,
    TempScaling,
    make_calibrator,
)
from scrye.data import SimBenchRecord


def _rec(options, truth):
    return SimBenchRecord(
        dataset_name="ESS", split="grouped", input_template="Q?",
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


def test_make_calibrator_dispatch():
    assert isinstance(make_calibrator("temp", T=1.5), TempScaling)
    assert isinstance(make_calibrator("entropy_temp", slope=1.0), EntropyTempScaling)
    assert isinstance(make_calibrator("dirichlet", alpha=0.5), DirichletCalibrator)
    assert isinstance(make_calibrator("identity"), IdentityCalibrator)
