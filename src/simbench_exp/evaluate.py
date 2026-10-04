"""Evaluation harness — runs a pipeline over records and produces analysis.

The harness is agnostic to what is inside the pipeline: swap the predictor or
calibrator and every summary, breakdown, and figure below keeps working. The
central object is a tidy results :class:`pandas.DataFrame` (one row per record)
that the viz layer consumes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import REQUIRED_QUESTIONS
from .data import SimBenchRecord
from .pipeline import Pipeline
from .scoring import (
    bootstrap_ci,
    delta_alignment,
    response_entropy,
    simbench_score,
    tvd_to_uniform,
)


# -- normalizers (SimBench Eq. 2) ------------------------------------------
def build_normalizers(records: Sequence[SimBenchRecord]) -> dict[tuple[str, str], float]:
    """Per-dataset normalizers for the SimBench score (Eq. 2).

    Keyed by (split, dataset_name) -> mean TVD(truth, uniform) over that
    dataset. The score is normalized by a *dataset-level* scalar, so build this
    from the FULL split (all loaded records), not a subsample, to (a) match the
    paper's denominator and (b) make scores comparable across subsamples. The
    normalizer is model-independent (it only uses ground truth).
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for r in records:
        groups.setdefault((r.split, r.dataset_name), []).append(tvd_to_uniform(r.human_answer))
    return {k: float(np.mean(v)) for k, v in groups.items()}


# -- sampling helpers ------------------------------------------------------
def stratified_sample(
    records: Sequence[SimBenchRecord],
    n: int,
    seed: int = 0,
) -> list[SimBenchRecord]:
    """Sample ~n records spread evenly across source datasets (seeded).

    Even coverage keeps any one large survey from dominating the headline mean
    and gives every dataset enough rows for a per-dataset breakdown.
    """
    by_ds: dict[str, list[SimBenchRecord]] = {}
    for r in records:
        by_ds.setdefault(r.dataset_name, []).append(r)
    rng = random.Random(seed)
    datasets = sorted(by_ds)
    per = max(1, n // len(datasets))
    out: list[SimBenchRecord] = []
    for ds in datasets:
        pool = by_ds[ds]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out[:n]


def required_question_records(
    records: Sequence[SimBenchRecord],
) -> dict[str, list[SimBenchRecord]]:
    """Group records matching each assignment-required question."""
    out: dict[str, list[SimBenchRecord]] = {k: [] for k in REQUIRED_QUESTIONS}
    for r in records:
        text = r.input_template.lower()
        for key, needle in REQUIRED_QUESTIONS.items():
            if needle in text:
                out[key].append(r)
    return out


# -- core evaluation -------------------------------------------------------
def evaluate(
    pipeline: Pipeline,
    records: Sequence[SimBenchRecord],
    normalizers: dict[tuple[str, str], float] | None = None,
    max_workers: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the pipeline over records; return one tidy row per record.

    `normalizers` are the per-dataset SimBench Eq. 2 scalars from
    :func:`build_normalizers`. Pass one built from the FULL split for
    paper-comparable scores; if omitted, normalizers are built from `records`
    themselves (fine for a quick look, but subsample-dependent).

    Columns: dataset, split, n_options, is_population, segment, group_size,
    truth_entropy (normalized entropy of the human distribution), score (the
    per-instance SimBench S under Eq. 2), plus pred/truth/options/input_template
    for drill-downs and figures.
    """
    records = list(records)
    if normalizers is None:
        normalizers = build_normalizers(records)
    preds = pipeline.predict_batch(records, max_workers=max_workers, progress=progress)
    rows = []
    for rec, pred in zip(records, preds):
        norm = normalizers.get((rec.split, rec.dataset_name))
        rows.append(
            {
                "dataset": rec.dataset_name,
                "split": rec.split,
                "n_options": rec.num_options,
                "is_population": rec.is_population,
                "segment": rec.segment,
                "group_size": rec.group_size,
                "truth_entropy": response_entropy(rec.human_answer),
                "score": simbench_score(
                    pred, rec.human_answer, options=list(rec.options), normalizer=norm
                ),
                "input_template": rec.input_template,
                "options": list(rec.options),
                "pred": pred,
                "truth": rec.human_answer,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["pipeline"] = pipeline.name
    return df


# -- summaries -------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline mean SimBench score with a bootstrap 95% CI."""
    mean, lo, hi = bootstrap_ci(df["score"].tolist())
    return {
        "pipeline": df.attrs.get("pipeline", "?"),
        "n": len(df),
        "mean_score": mean,
        "ci_low": lo,
        "ci_high": hi,
        "frac_below_uniform": float((df["score"] < 0).mean()),
    }


def score_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean score (+ bootstrap CI and n) grouped by a column."""
    rows = []
    for key, grp in df.groupby(column):
        mean, lo, hi = bootstrap_ci(grp["score"].tolist())
        rows.append({column: key, "n": len(grp), "mean_score": mean,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)


# -- counterfactual sensitivity -------------------------------------------
def _aligned(dist: dict, options: list[str]) -> np.ndarray:
    v = np.array([float(dist.get(o, 0.0)) for o in options], dtype=float)
    s = v.sum()
    return v / s if s > 0 else np.full(len(options), 1.0 / max(1, len(options)))


def counterfactual_sensitivity(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Does conditioning shift predictions in the right *direction*?

    For each question (same dataset + input_template) with multiple segment
    rows, we form a group-size-weighted population reference from both the
    predictions and the truths, then measure the cosine alignment between each
    segment's predicted shift and its true shift (segment minus population).
    This isolates conditioning *direction* from unconditioned-prior accuracy,
    as the assignment asks.

    Returns (mean alignment over segments, per-segment table). Segments whose
    true shift is ~zero contribute NaN and are dropped from the mean.
    """
    rows = []
    for (ds, q), grp in df.groupby(["dataset", "input_template"]):
        if len(grp) < 2:
            continue
        options = grp.iloc[0]["options"]
        w = np.array([gs if gs and gs > 0 else 1.0 for gs in grp["group_size"]], dtype=float)
        w = w / w.sum()
        preds = np.vstack([_aligned(p, options) for p in grp["pred"]])
        truths = np.vstack([_aligned(t, options) for t in grp["truth"]])
        pred_pop = w @ preds
        truth_pop = w @ truths
        for i, (_, r) in enumerate(grp.iterrows()):
            align = delta_alignment(preds[i] - pred_pop, truths[i] - truth_pop)
            rows.append({
                "dataset": ds,
                "segment": r["segment"],
                "alignment": align,
                "true_shift_mag": float(np.linalg.norm(truths[i] - truth_pop)),
            })
    table = pd.DataFrame(rows)
    valid = table["alignment"].dropna() if not table.empty else pd.Series(dtype=float)
    mean_align = float(valid.mean()) if len(valid) else float("nan")
    return mean_align, table
