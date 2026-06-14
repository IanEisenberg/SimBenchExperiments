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


def tvd_to_uniform(truth: Mapping[str, float]) -> float:
    """TVD between a ground-truth distribution and uniform over its options.

    Depends only on the truth, so the per-dataset mean of this quantity is the
    model-independent normalizer in the benchmark's Eq. 2 (see `simbench_score`).
    """
    p = _renormalize(np.array(list(truth.values()), dtype=float))
    return total_variation_distance(p, uniform_like(p))


def simbench_score(
    pred: Mapping[str, float],
    truth: Mapping[str, float],
    options: Sequence[str] | None = None,
    normalizer: float | None = None,
) -> float:
    """SimBench score S (Hu et al., arXiv 2510.17516).

    Two normalization modes:

    * **Eq. 2 (paper default, use this for reported numbers):** pass
      `normalizer` = the mean TVD(P, U) across the question's *dataset*
      (a single scalar). This is numerically stable and makes the uniform
      predictor average to 0 over a dataset. Build normalizers with
      :func:`scrye.evaluate.build_normalizers`.
    * **Eq. 1 (per-instance, the conceptual form):** leave `normalizer=None`
      and the denominator is this instance's own TVD(P, U). Unstable on
      near-uniform truths (denominator → 0); fine for unit math and singletons,
      not for aggregate reporting.

    Returns 100 for a perfect match; ~0 for uniform-level prediction; negative
    when worse than uniform.
    """
    p, q = _aligned_vectors(pred, truth, options)
    p = _renormalize(p)
    q = _renormalize(q)
    denom = normalizer if normalizer is not None else total_variation_distance(p, uniform_like(p))
    if denom is None or denom <= 0:
        return 0.0
    return 100.0 * (1.0 - total_variation_distance(p, q) / denom)


def response_entropy(dist: Mapping[str, float], normalized: bool = True) -> float:
    """Shannon entropy of a response distribution.

    With `normalized=True`, divides by log(K) so the value is in [0, 1] and
    comparable across questions with different option counts (0 = consensus /
    one-hot, 1 = uniform). This is the axis the SimBench paper found predicts
    where instruction-tuned models fail (mode-seeking on high-entropy items).
    """
    p = _renormalize(np.array(list(dist.values()), dtype=float))
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p)).sum())
    return h / np.log(len(dist)) if normalized else h


def delta_alignment(
    pred_delta: np.ndarray,
    truth_delta: np.ndarray,
) -> float:
    """Cosine similarity between a predicted shift and the true shift.

    Used by the counterfactual sensitivity metric: each vector is the change in
    a distribution when conditioning on a segment (segment minus population).
    +1 = shift in exactly the right direction, -1 = exactly wrong, 0 =
    orthogonal. Returns NaN when either shift is ~zero (no signal to align).
    """
    a = np.asarray(pred_delta, dtype=float)
    b = np.asarray(truth_delta, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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
