"""SimBench scoring.

Primary metric (Hu et al., arXiv 2510.17516, Eq. 1):

    S(P, Q) = 100 * (1 - TVD(P, Q) / TVD(P, U))

where P is the empirical human distribution, Q the prediction, U the uniform
distribution over the same K options, and TVD is total variation distance,

    TVD(P, Q) = 0.5 * sum_i |p_i - q_i|.

Interpretation:
  * S = 100  -> perfect match (TVD = 0)
  * S = 0    -> no better than predicting uniform (the naive baseline)
  * S < 0    -> worse than uniform

The aggregate SimBench score is the mean of per-instance S over a set of test
cases. Predictions and ground truth are aligned on a shared, ordered option set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _aligned_vectors(
    pred: Mapping[str, float],
    truth: Mapping[str, float],
    options: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align two label->prob maps onto a shared ordered option set.

    Missing labels are treated as zero mass. If `options` is not given, the
    union of keys (truth first, to keep a stable order) is used.
    """
    if options is None:
        options = list(truth.keys())
        options += [k for k in pred.keys() if k not in truth]
    p = np.array([float(truth.get(o, 0.0)) for o in options], dtype=float)
    q = np.array([float(pred.get(o, 0.0)) for o in options], dtype=float)
    return p, q


def _renormalize(v: np.ndarray) -> np.ndarray:
    total = v.sum()
    if total <= 0:
        # Uniform fallback for an all-zero vector.
        return np.full_like(v, 1.0 / len(v)) if len(v) else v
    return v / total


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TVD between two probability vectors (each renormalized first)."""
    p = _renormalize(np.asarray(p, dtype=float))
    q = _renormalize(np.asarray(q, dtype=float))
    return 0.5 * float(np.abs(p - q).sum())


def uniform_like(p: np.ndarray) -> np.ndarray:
    k = len(p)
    return np.full(k, 1.0 / k) if k else p


def simbench_score(
    pred: Mapping[str, float],
    truth: Mapping[str, float],
    options: Sequence[str] | None = None,
) -> float:
    """Per-instance SimBench score S in [.., 100]. See module docstring.

    The normalizer TVD(P, U) is the TVD of the *ground truth* to uniform; when
    the ground truth is itself uniform this is 0 and S is undefined — we return
    0.0 in that degenerate case (the prediction can do no better than uniform by
    construction, so it earns the baseline score).
    """
    p, q = _aligned_vectors(pred, truth, options)
    p = _renormalize(p)
    q = _renormalize(q)
    denom = total_variation_distance(p, uniform_like(p))
    if denom <= 0:
        return 0.0
    return 100.0 * (1.0 - total_variation_distance(p, q) / denom)


def aggregate_score(scores: Sequence[float]) -> float:
    """Mean SimBench score over test cases."""
    arr = np.asarray(list(scores), dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def bootstrap_ci(
    scores: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and a percentile bootstrap CI for an aggregate score.

    Returns (mean, lo, hi). Used to put confidence intervals on every headline
    comparison rather than reporting bare point estimates.
    """
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)
