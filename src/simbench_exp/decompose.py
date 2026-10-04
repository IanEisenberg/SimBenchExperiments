"""Error decomposition — *why* a system's SimBench score is what it is.

SimBench reports one number per item: ``S = 100·(1 − TVD(P,Q)/Z)``. That tells
us *how much* a prediction is wrong, not *how*. This module splits the
underlying error ``TVD(P,Q)`` into two non-negative, interpretable parts::

    TVD(P, Q) = concentration_err + location_err

* **concentration_err** = ½·Σ|sort(P) − sort(Q)| — error in the distribution
  *profile* (how peaked vs. spread the mass is), independent of *which* option
  carries it. It is the minimum TVD achievable by relabeling Q's options, so it
  isolates over-/under-dispersion: *are we representing the group's true
  diversity vs. concentration?*
* **location_err** = TVD(P,Q) − concentration_err ≥ 0 — the extra error from
  putting otherwise well-shaped mass on the *wrong* options: *is the weight in
  the right place?*

Both terms are ≥ 0 by the rearrangement inequality (sorting two vectors the same
way minimizes their L1 distance), and they sum exactly to TVD. Because the
SimBench normalizer ``Z`` depends only on the truth, a score *difference* between
two systems on the same items splits the same way — letting us attribute a
score *gain* to "fixed the spread" vs. "moved mass to the right options".

Two signed facets say which *direction* each error runs:

* **entropy_gap** = H_norm(Q) − H_norm(P): <0 = over-sharpened (too concentrated),
  >0 = too diffuse. The magnitude is a signed cousin of concentration_err.
* **mode_match** / **mode_mass_err**: did we pick the truth's dominant option, and
  do we put the right amount of mass on it? Interpretable location facets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .scoring import _aligned_vectors, _renormalize


def _norm_entropy(v: np.ndarray) -> float:
    """Shannon entropy of a probability vector, normalized to [0, 1] by log K."""
    p = v[v > 0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p)).sum())
    return h / np.log(len(v))


def decompose_error(
    pred: Mapping[str, float],
    truth: Mapping[str, float],
    options: Sequence[str] | None = None,
) -> dict:
    """Split TVD(P, Q) into concentration vs. location error, plus signed facets.

    Returns a dict with: ``tvd``, ``concentration_err``, ``location_err``
    (concentration + location == tvd, both ≥ 0); ``pred_entropy``,
    ``truth_entropy`` (normalized, in [0, 1]); ``entropy_gap`` =
    pred_entropy − truth_entropy; ``mode_match`` (bool: argmax(Q) == argmax(P));
    and ``mode_mass_err`` = Q[argmax P] − P[argmax P] (signed).
    """
    p, q = _aligned_vectors(pred, truth, options)
    p = _renormalize(p)
    q = _renormalize(q)

    tvd = 0.5 * float(np.abs(p - q).sum())
    # Sort both descending: aligning the profiles is the minimum TVD over
    # relabelings, so what's left is pure shape (concentration) mismatch.
    ps = np.sort(p)[::-1]
    qs = np.sort(q)[::-1]
    concentration = 0.5 * float(np.abs(ps - qs).sum())
    location = max(0.0, tvd - concentration)  # clip float noise; ≥ 0 in theory

    true_mode = int(np.argmax(p))
    pred_mode = int(np.argmax(q))

    return {
        "tvd": tvd,
        "concentration_err": concentration,
        "location_err": location,
        "pred_entropy": _norm_entropy(q),
        "truth_entropy": _norm_entropy(p),
        "entropy_gap": _norm_entropy(q) - _norm_entropy(p),
        "mode_match": bool(true_mode == pred_mode),
        "mode_mass_err": float(q[true_mode] - p[true_mode]),
    }


def decompose_records(
    records: Sequence[Mapping],
    normalizers: Mapping[tuple[str, str], float] | None = None,
    passthrough: Sequence[str] = (
        "dataset", "split", "n_options", "is_population", "group_size", "score",
    ),
) -> pd.DataFrame:
    """Decompose a list of result records (each with ``pred`` + ``truth``).

    `records` are the per-item dicts saved in ``outputs/runs/<run>.results.json``
    (keys include ``pred``, ``truth``, ``dataset``, ``split``, ...). Returns one
    row per record with the decomposition columns plus any present `passthrough`
    fields.

    If `normalizers` is given (the SimBench Eq. 2 map from
    :func:`simbench_exp.evaluate.build_normalizers`, keyed ``(split, dataset)``), three
    score-scale columns are added so the score *loss* (distance below 100) splits
    the same way as TVD:

      * ``score_loss`` = 100·TVD / Z       (== 100 − recorded score)
      * ``conc_loss``  = 100·concentration / Z
      * ``loc_loss``   = 100·location / Z   (conc_loss + loc_loss == score_loss)
    """
    rows: list[dict] = []
    for rec in records:
        d = decompose_error(rec["pred"], rec["truth"], rec.get("options"))
        row = {k: rec[k] for k in passthrough if k in rec}
        row.update(d)
        if normalizers is not None:
            z = normalizers.get((rec.get("split"), rec.get("dataset")))
            if z and z > 0:
                row["score_loss"] = 100.0 * d["tvd"] / z
                row["conc_loss"] = 100.0 * d["concentration_err"] / z
                row["loc_loss"] = 100.0 * d["location_err"] / z
        rows.append(row)
    return pd.DataFrame(rows)


def system_summary(df: pd.DataFrame) -> pd.Series:
    """Aggregate one system's decomposed frame into headline error facets.

    Reports mean TVD and its concentration/location split (absolute and as a
    share of TVD), the signed mean entropy gap (direction of mis-dispersion),
    mode accuracy, and — if score-scale columns are present — the mean score loss
    and how it divides between concentration and location.
    """
    n = len(df)
    tvd = df["tvd"].mean()
    conc = df["concentration_err"].mean()
    loc = df["location_err"].mean()
    out = {
        "n": n,
        "mean_tvd": tvd,
        "conc_err": conc,
        "loc_err": loc,
        "conc_share": conc / tvd if tvd > 0 else float("nan"),
        "loc_share": loc / tvd if tvd > 0 else float("nan"),
        "entropy_gap": df["entropy_gap"].mean(),     # <0 = over-sharpened on average
        "mode_acc": df["mode_match"].mean(),
        "mode_mass_err": df["mode_mass_err"].mean(),
    }
    if "score_loss" in df.columns:
        out["score_loss"] = df["score_loss"].mean()
        out["conc_loss"] = df["conc_loss"].mean()
        out["loc_loss"] = df["loc_loss"].mean()
    return pd.Series(out)
