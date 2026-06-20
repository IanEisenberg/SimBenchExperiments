"""Tests for reporting: K-corrected band and the production score_fn shape."""

import numpy as np

from scrye.evaluate import build_normalizers
from scrye.ledger import ExperimentNode, Ledger
from scrye.report import final_report, k_corrected_band, make_score_fn
from scrye.data import SimBenchRecord
from scrye.spec import PipelineSpec


def _rec(options, truth, ds="ESS"):
    return SimBenchRecord(
        dataset_name=ds, split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def test_k_corrected_band_widens_with_k():
    lo1, hi1 = k_corrected_band(50.0, n=200, k=1)
    lo100, hi100 = k_corrected_band(50.0, n=200, k=100)
    assert (hi100 - lo100) > (hi1 - lo1)  # more queries -> wider honest band
    assert lo1 <= 50.0 <= hi1


def test_score_fn_returns_mean_and_breakdown():
    recs = [_rec(["A", "B"], {"A": 0.7, "B": 0.3}), _rec(["A", "B"], {"A": 0.4, "B": 0.6})]
    norms = build_normalizers(recs)
    score_fn = make_score_fn(norms)
    spec = PipelineSpec(predictor="uniform")  # no network
    mean, breakdown = score_fn(spec, recs)
    assert isinstance(mean, float)
    assert "by_entropy" in breakdown
    assert breakdown["cost_usd"] == 0.0  # uniform predictor makes no API calls


def test_final_report_scores_uniform_to_zero(tmp_path):
    # Uniform predictor averages to 0 under Eq.2 normalizers (paper property).
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(40):
        a = float(rng.random())
        recs.append(_rec(["A", "B"], {"A": a, "B": 1 - a}))
    norms = build_normalizers(recs)
    led = Ledger(tmp_path / "r.jsonl")
    rep = final_report(PipelineSpec(predictor="uniform"), recs, norms, led)
    assert abs(rep["mean_score"]) < 1e-9
    assert rep["global_k"] == 0
    assert "k_band_low" in rep and "k_band_high" in rep
    assert rep["total_cost_usd"] == 0.0  # empty ledger -> no spend
