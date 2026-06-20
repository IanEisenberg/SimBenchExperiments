"""Visualization suite — all functions take the results DataFrame from
:func:`scrye.evaluate.evaluate` (or a slice of it) and return a matplotlib
Figure, so they work unchanged for any pipeline.

The figures are chosen to *diagnose* a predictor, not just score it:
  * score histogram          -> overall quality and the worse-than-uniform tail
  * score by dataset         -> where it works and where it breaks
  * score vs truth entropy   -> the mode-seeking hypothesis (does it fail on
                                high-entropy / divided questions?)
  * calibration scatter      -> systematic over/under-confidence -> motivates
                                the calibration component
  * example distributions    -> qualitative spot checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .scoring import bootstrap_ci


def _baseline_line(ax):
    ax.axhline(0, color="0.6", lw=1, ls="--", zorder=0)


def plot_score_histogram(df: pd.DataFrame) -> Figure:
    """Distribution of per-instance SimBench scores, with mean and uniform (0)."""
    fig = Figure(figsize=(6.5, 3.8))
    ax = fig.subplots()
    scores = df["score"].clip(lower=-100)  # clip extreme negative tail for readability
    ax.hist(scores, bins=40, color="#4C72B0", alpha=0.85)
    mean = df["score"].mean()
    ax.axvline(0, color="0.5", ls="--", lw=1, label="uniform baseline (S=0)")
    ax.axvline(mean, color="#C44E52", lw=2, label=f"mean S = {mean:.1f}")
    ax.set_xlabel("SimBench score S  (per question; clipped at -100)")
    ax.set_ylabel("count")
    ax.set_title(f"Score distribution — {df.attrs.get('pipeline', '')}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_by_dataset(df: pd.DataFrame) -> Figure:
    """Mean score per source dataset with bootstrap 95% CIs."""
    rows = []
    for ds, grp in df.groupby("dataset"):
        mean, lo, hi = bootstrap_ci(grp["score"].tolist())
        rows.append((ds, mean, lo, hi, len(grp)))
    rows.sort(key=lambda r: r[1])
    labels = [f"{r[0]} (n={r[4]})" for r in rows]
    means = [r[1] for r in rows]
    lo = [r[1] - r[2] for r in rows]
    hi = [r[3] - r[1] for r in rows]

    fig = Figure(figsize=(7, max(3.5, 0.35 * len(rows) + 1)))
    ax = fig.subplots()
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=[lo, hi], color="#55A868", alpha=0.85, capsize=3)
    ax.axvline(0, color="0.5", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean SimBench score S")
    ax.set_title(f"Score by dataset — {df.attrs.get('pipeline', '')}")
    fig.tight_layout()
    return fig


def plot_score_vs_entropy(df: pd.DataFrame) -> Figure:
    """Per-question score vs. normalized truth entropy (mode-seeking test).

    The SimBench paper finds instruction-tuned models fail on high-entropy
    (divided) questions. A downward trend here is that failure mode showing up;
    the binned line makes it legible through the scatter.
    """
    fig = Figure(figsize=(6.5, 4))
    ax = fig.subplots()
    ax.scatter(df["truth_entropy"], df["score"].clip(lower=-100),
               s=12, alpha=0.35, color="#4C72B0")
    # Binned means to show the trend.
    bins = np.linspace(0, 1, 9)
    idx = np.digitize(df["truth_entropy"], bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        m = df["score"][idx == b]
        if len(m):
            xs.append((bins[b - 1] + bins[b]) / 2)
            ys.append(m.mean())
    ax.plot(xs, ys, "-o", color="#C44E52", lw=2, label="binned mean")
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("normalized entropy of human distribution (0=consensus, 1=divided)")
    ax.set_ylabel("SimBench score S (clipped at -100)")
    ax.set_title(f"Score vs. response entropy — {df.attrs.get('pipeline', '')}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_by_n_options(df: pd.DataFrame) -> Figure:
    """Mean score by number of answer options."""
    rows = []
    for k, grp in df.groupby("n_options"):
        mean, lo, hi = bootstrap_ci(grp["score"].tolist())
        rows.append((k, mean, lo, hi, len(grp)))
    rows.sort()
    fig = Figure(figsize=(6.5, 3.8))
    ax = fig.subplots()
    x = [r[0] for r in rows]
    means = [r[1] for r in rows]
    err = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
    ax.errorbar(x, means, yerr=err, fmt="-o", color="#8172B3", capsize=3)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("number of answer options")
    ax.set_ylabel("mean SimBench score S")
    ax.set_title(f"Score by option count — {df.attrs.get('pipeline', '')}")
    fig.tight_layout()
    return fig


def plot_calibration(df: pd.DataFrame, n_bins: int = 10) -> Figure:
    """Reliability scatter: predicted probability vs. empirical probability.

    Pools every (option) cell across all questions: x = predicted prob, y =
    true prob, binned. Points below the diagonal mean the model put too much
    mass there (over-confidence). Systematic deviation from y=x is precisely
    what a calibration component would correct — so this plot is the visual
    argument for adding one.
    """
    pred_p, true_p = [], []
    for _, r in df.iterrows():
        for o in r["options"]:
            pred_p.append(float(r["pred"].get(o, 0.0)))
            true_p.append(float(r["truth"].get(o, 0.0)))
    pred_p = np.array(pred_p)
    true_p = np.array(true_p)

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.scatter(pred_p, true_p, s=8, alpha=0.15, color="#4C72B0")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(pred_p, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        mask = idx == b
        if mask.sum():
            bx.append(pred_p[mask].mean())
            by.append(true_p[mask].mean())
    ax.plot(bx, by, "-o", color="#C44E52", lw=2, label="binned mean")
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical probability")
    ax.set_title(f"Calibration — {df.attrs.get('pipeline', '')}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_example_distributions(df: pd.DataFrame, n: int = 6, seed: int = 0) -> Figure:
    """Small multiples of predicted vs. empirical for a few sampled questions."""
    sample = df.sample(min(n, len(df)), random_state=seed)
    ncols = 3
    nrows = int(np.ceil(len(sample) / ncols))
    fig = Figure(figsize=(4 * ncols, 2.8 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, sample.iterrows()):
        ax.axis("on")
        opts = r["options"]
        x = np.arange(len(opts))
        ax.bar(x - 0.2, [r["truth"].get(o, 0) for o in opts], 0.4, label="human", color="#4C72B0")
        ax.bar(x + 0.2, [r["pred"].get(o, 0) for o in opts], 0.4, label="pred", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(opts, fontsize=7)
        ax.set_title(f"{r['dataset']} | S={r['score']:.0f}", fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Example predictions — {df.attrs.get('pipeline', '')}")
    fig.tight_layout()
    return fig


def plot_system_comparison(
    table: pd.DataFrame,
    score_col: str = "mean_score",
    label_col: str = "system",
) -> Figure:
    """Horizontal bar chart of a topline comparison table with 95% CIs.

    Consumes the DataFrame returned by :func:`scrye.experiment.compare` (or
    :func:`scrye.experiment.model_sweep`): one row per system, with
    ``mean_score`` and ``ci_low``/``ci_high``. Systems are ordered by score so
    the headline ranking reads top-to-bottom; the uniform baseline (S=0) is
    marked for reference.
    """
    t = table.sort_values(score_col, ascending=True).reset_index(drop=True)
    labels = t[label_col].astype(str).tolist()
    y = np.arange(len(t))
    fig = Figure(figsize=(7, 0.6 * len(t) + 1.6))
    ax = fig.subplots()
    means = t[score_col].to_numpy()
    if {"ci_low", "ci_high"}.issubset(t.columns):
        err = np.vstack([means - t["ci_low"].to_numpy(), t["ci_high"].to_numpy() - means])
    else:
        err = None
    ax.barh(y, means, color="#4C72B0", alpha=0.85,
            xerr=err, error_kw={"ecolor": "0.3", "capsize": 3})
    ax.axvline(0, color="0.5", ls="--", lw=1, label="uniform baseline (S=0)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean SimBench score S  (95% CI)")
    ax.set_title("System comparison")
    for yi, m in zip(y, means):
        ax.annotate(f"{m:.1f}", (m, yi), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
