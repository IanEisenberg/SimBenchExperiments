"""Reporting — the production scorer and the honest, test-once headline.

`make_score_fn` is the real `score_fn` the search loop consumes: it builds a
pipeline from a spec, runs the evaluation harness, and returns the mean S plus
an entropy-binned breakdown. `final_report` scores the frozen winning spec on
the TEST split exactly once and reports the raw mean, its bootstrap CI, and a
band widened for the realized global K (the Ladder generalization factor), so
the headline number discloses how much search produced it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from .evaluate import evaluate, summarize
from .ledger import Ledger
from .spec import PipelineSpec, build_from_spec


def _pipe_cost(pipe) -> float:
    """Best-effort cumulative $ spend behind a pipeline's predictor (0 if none).

    Sees through the post-stratification wrapper to its base predictor's client,
    mirroring scrye.experiment.pipeline_model."""
    pred = pipe.predictor
    client = getattr(pred, "client", None) or getattr(getattr(pred, "base", None), "client", None)
    usage = getattr(client, "usage", None)
    return float(getattr(usage, "cost_usd", 0.0)) if usage is not None else 0.0


def make_score_fn(normalizers, *, client=None, max_workers: int = 8):
    """Build the production score_fn: (spec, records) -> (mean_score, breakdown)."""

    def score_fn(spec: PipelineSpec, records: Sequence) -> tuple[float, dict]:
        pipe = build_from_spec(spec, client=client)
        before = _pipe_cost(pipe)
        df = evaluate(pipe, records, normalizers=normalizers,
                      max_workers=max_workers, progress=False)
        cost = _pipe_cost(pipe) - before  # new OpenRouter spend for this scoring call
        mean = float(df["score"].mean()) if len(df) else float("nan")
        # Entropy-binned breakdown (consensus vs diverse), the SimBench axis.
        bins = pd.cut(df["truth_entropy"], [0, 0.33, 0.66, 1.0], include_lowest=True)
        by_entropy = {str(k): float(v) for k, v in df.groupby(bins, observed=True)["score"].mean().items()}
        return mean, {"by_entropy": by_entropy, "n": len(df), "cost_usd": cost}

    return score_fn


def k_corrected_band(mean: float, n: int, k: int, *, mult: float = 1.0) -> tuple[float, float]:
    """Widen a band around `mean` by the Ladder factor (log(k·n)/n)^(1/3).

    This reflects that a number selected from k adaptive val queries generalizes
    only up to ~(log(k·n)/n)^(1/3); a single query (k=1) gives the tightest band.
    """
    n = max(1, n)
    k = max(1, k)
    half = mult * (math.log(k * n) / n) ** (1.0 / 3.0) * 100.0  # on the 0-100 S scale
    return (mean - half, mean + half)


def final_report(spec: PipelineSpec, test: Sequence, normalizers, ledger: Ledger,
                 *, client=None) -> dict:
    """Score the frozen spec ONCE on test; report raw mean, CI, and K-band."""
    pipe = build_from_spec(spec, client=client)
    _cost_before = _pipe_cost(pipe)
    df = evaluate(pipe, test, normalizers=normalizers, progress=False)
    test_eval_cost = _pipe_cost(pipe) - _cost_before
    summ = summarize(df)
    k = ledger.global_k()
    lo, hi = k_corrected_band(summ["mean_score"], n=len(df), k=k)
    return {
        "pipeline": pipe.name,
        "lever_path": list(spec.lever_path),
        "n": len(df),
        "mean_score": summ["mean_score"],
        "ci_low": summ["ci_low"],
        "ci_high": summ["ci_high"],
        "global_k": k,
        "k_band_low": lo,
        "k_band_high": hi,
        "total_cost_usd": ledger.total_cost(),  # search-phase spend only
        "test_eval_cost_usd": test_eval_cost,
        "grand_total_cost_usd": ledger.total_cost() + test_eval_cost,
    }
