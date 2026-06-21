"""Calibrators — the post-hoc *correction* component of the pipeline.

A calibrator transforms a predicted distribution into a corrected one. It may
first be `fit` on a held-out set of (record, raw_prediction, truth) triples;
the default :class:`IdentityCalibrator` is a no-op so the pipeline runs with no
calibration out of the box. Future calibrators (temperature scaling,
entropy-aware recalibration targeting mode-seeking, Dirichlet mapping) implement
this same interface and drop into the pipeline unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np

from .data import SimBenchRecord
from .scoring import response_entropy, simbench_score


class Calibrator(ABC):
    """Transforms a raw predicted distribution into a corrected one.

    Implementations set a short `name` and implement :meth:`transform`. If a
    calibrator needs fitting, override :meth:`fit`; it is called once on a
    held-out set before evaluation. The base `fit` is a no-op.
    """

    name: str = "calibrator"

    def fit(
        self,
        records: Sequence[SimBenchRecord],
        raw_preds: Sequence[Mapping[str, float]],
    ) -> "Calibrator":
        """Fit on held-out data. Default: no-op. Returns self for chaining."""
        return self

    @abstractmethod
    def transform(
        self, record: SimBenchRecord, pred: Mapping[str, float]
    ) -> dict[str, float]:
        ...


class IdentityCalibrator(Calibrator):
    """No-op calibrator — passes predictions through unchanged (the default)."""

    name = "identity"

    def transform(
        self, record: SimBenchRecord, pred: Mapping[str, float]
    ) -> dict[str, float]:
        return dict(pred)


# --- recalibration calibrators (append to src/scrye/calibrate.py) ---------


def _as_vec(pred: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    keys = list(pred)
    v = np.array([max(0.0, float(pred[k])) for k in keys], dtype=float)
    return keys, v


def _temper_vec(v: np.ndarray, T: float) -> np.ndarray:
    s = v.sum()
    if s <= 0 or T <= 0:
        return np.full_like(v, 1.0 / len(v)) if len(v) else v
    p = v / s
    p = np.power(p, 1.0 / T)
    tot = p.sum()
    return p / tot if tot > 0 else np.full_like(p, 1.0 / len(p))


class TempScaling(Calibrator):
    """Temperature scaling: q_i^(1/T) renormalized. T>1 flattens (treats the
    mode-seeking over-sharpness SimBench diagnoses); T<1 sharpens; T=1 is a no-op.
    `fit` grid-searches T to maximize mean SimBench score on held-out data."""

    name = "temp"

    def __init__(self, T: float = 1.0) -> None:
        self.T = T

    def fit(self, records, raw_preds):
        grid = [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        best_T, best_score = 1.0, float("-inf")
        for T in grid:
            scores = []
            for rec, raw in zip(records, raw_preds):
                q = self.__class__(T=T).transform(rec, raw)
                scores.append(simbench_score(q, rec.human_answer, options=list(rec.options)))
            mean = float(np.mean(scores)) if scores else float("-inf")
            if mean > best_score:
                best_T, best_score = T, mean
        self.T = best_T
        return self

    def transform(self, record, pred):
        keys, v = _as_vec(pred)
        p = _temper_vec(v, self.T)
        return {k: float(x) for k, x in zip(keys, p)}


class EntropyTempScaling(Calibrator):
    """Entropy-conditioned temperature: T(pred) = 1 + slope * H_norm(pred), so
    flatten more where the prediction is already diffuse — the regime SimBench
    shows instruct models over-sharpen (r=-0.942). `fit` grid-searches slope."""

    name = "entropy_temp"

    def __init__(self, slope: float = 0.0) -> None:
        self.slope = slope

    def _temp_for(self, pred) -> float:
        return 1.0 + self.slope * response_entropy(pred, normalized=True)

    def fit(self, records, raw_preds):
        grid = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
        best_s, best_score = 0.0, float("-inf")
        for slope in grid:
            cal = self.__class__(slope=slope)
            scores = [
                simbench_score(cal.transform(rec, raw), rec.human_answer, options=list(rec.options))
                for rec, raw in zip(records, raw_preds)
            ]
            mean = float(np.mean(scores)) if scores else float("-inf")
            if mean > best_score:
                best_s, best_score = slope, mean
        self.slope = best_s
        return self

    def transform(self, record, pred):
        keys, v = _as_vec(pred)
        p = _temper_vec(v, self._temp_for(pred))
        return {k: float(x) for k, x in zip(keys, p)}


def _norm_entropy_vec(v: np.ndarray) -> float:
    """Normalized Shannon entropy of a probability vector (divided by log K)."""
    p = v[v > 0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p)).sum())
    return h / np.log(len(v))


def temper_to_entropy(
    pred: Mapping[str, float], h_target: float, iters: int = 50, eps: float = 1e-9
) -> dict[str, float]:
    """Temperature-scale `pred` so its normalized entropy equals `h_target`.

    Bisects the temperature `T` (q ∝ p^(1/T): T>1 flattens, T<1 sharpens) until the
    normalized entropy matches the target. Ranking is preserved (tempering is
    monotone); mass is clipped to `eps` first so the entropy can be *raised* even
    from a near-one-hot prediction. `h_target` is clamped to [0, 1].
    """
    keys = list(pred)
    v = np.array([max(0.0, float(pred[k])) for k in keys], dtype=float)
    if len(v) <= 1 or v.sum() <= 0:
        return {k: float(x) for k, x in zip(keys, v)} if len(v) else {}
    p = np.clip(v / v.sum(), eps, None)
    p = p / p.sum()
    target = min(max(float(h_target), 0.0), 1.0)
    lo, hi = 1e-3, 1e3
    for _ in range(iters):
        T = (lo + hi) / 2
        q = np.power(p, 1.0 / T)
        q = q / q.sum()
        if _norm_entropy_vec(q) < target:   # need more entropy -> larger T
            lo = T
        else:
            hi = T
    q = np.power(p, 1.0 / ((lo + hi) / 2))
    q = q / q.sum()
    return {k: float(x) for k, x in zip(keys, q)}


class EntropyTargetCalibrator(Calibrator):
    """Entropy **de-compression**: fix the regression-to-mean on the spread axis.

    Diagnostic (notebook 04): predicted entropy is a *compressed* version of the
    truth's (``pred_H ≈ a + b·truth_H`` with b ≈ 0.5), so the model over-spreads
    consensus questions and is roughly right on contested ones — a bidirectional
    error a global temperature cannot fix.

    `fit` regresses ``truth_H`` on ``pred_H`` (minimum-MSE target given the
    prediction's own entropy), then grid-searches a `gain` g: each item is
    tempered to ``H_target = H_pred + g·(Ĥ − H_pred)``. ``g=0`` is identity;
    ``g=1`` applies the full fitted correction; ``g>1`` over-corrects. The chosen
    `gain` maximizes mean SimBench score on the fit set. Unlike
    :class:`EntropyTempScaling` (one-sided: only flattens high-entropy items),
    this is two-sided — it *sharpens* over-diffuse predictions and flattens
    over-sharp ones, as the compression demands.
    """

    name = "entropy_target"

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain
        self.slope_ = 1.0       # truth_H ≈ intercept_ + slope_ * pred_H
        self.intercept_ = 0.0

    def _target_h(self, pred: Mapping[str, float]) -> float:
        hp = response_entropy(pred, normalized=True)
        reg = min(max(self.intercept_ + self.slope_ * hp, 0.0), 1.0)
        return min(max(hp + self.gain * (reg - hp), 0.0), 1.0)

    def fit(self, records, raw_preds):
        hp = np.array([response_entropy(p, normalized=True) for p in raw_preds])
        ht = np.array([response_entropy(r.human_answer, normalized=True) for r in records])
        if len(hp) >= 2 and np.std(hp) > 1e-9:
            self.slope_, self.intercept_ = np.polyfit(hp, ht, 1)
        grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        best_g, best_score = 0.0, float("-inf")
        for g in grid:
            self.gain = g
            scores = [
                simbench_score(self.transform(rec, raw), rec.human_answer,
                               options=list(rec.options))
                for rec, raw in zip(records, raw_preds)
            ]
            mean = float(np.mean(scores)) if scores else float("-inf")
            if mean > best_score:
                best_g, best_score = g, mean
        self.gain = best_g
        return self

    def transform(self, record, pred):
        hp = response_entropy(pred, normalized=True)
        target = self._target_h(pred)
        if self.gain == 0.0 or abs(target - hp) < 1e-6:
            return dict(pred)                       # exact no-op
        return temper_to_entropy(pred, target)


class FeatureEntropyTargetCalibrator(Calibrator):
    """Entropy de-compression with the target spread predicted from **question
    features**, not the prediction's own (compressed) entropy.

    Stage 08 showed the spread headroom (+~9 grouped oracle) is *not* recoverable
    from the prediction's own entropy — it is nearly constant, so it cannot say
    which questions are contested. Question features can: dataset identity, option
    count, conditioning, and the prediction's *shape* (peak height, top-2 gap,
    effective option count) predict the truth's entropy at out-of-sample R² ≈ 0.58.

    `fit` ridge-regresses ``truth_H`` on those features (continuous features
    z-scored on the fit set), then grid-searches a `gain` g blending the prediction
    toward the feature-predicted target ``H_pred + g·(Ĥ_feat − H_pred)``. `transform`
    tempers each prediction to that target. Feature groups are toggleable
    (`use_dataset`, `use_pred_shape`) so ablations share one class.
    """

    name = "feat_entropy_target"

    def __init__(self, gain: float = 1.0, ridge: float = 1.0,
                 use_dataset: bool = True, use_pred_shape: bool = True) -> None:
        self.gain = gain
        self.ridge = ridge
        self.use_dataset = use_dataset
        self.use_pred_shape = use_pred_shape
        self.datasets_: list[str] = []
        self.w_ = None              # weight vector (incl. trailing intercept)
        self.mean_ = None           # fit-set feature means (for z-scoring)
        self.std_ = None

    def _raw_features(self, record: SimBenchRecord, pred: Mapping[str, float]) -> list[float]:
        feats: list[float] = []
        if self.use_dataset:
            feats += [1.0 if record.dataset_name == d else 0.0 for d in self.datasets_]
        feats.append(float(record.num_options))
        feats.append(1.0 if record.split == "pop" else 0.0)
        feats.append(float(record.num_grouping_vars or 0))
        if self.use_pred_shape:
            _, v = _as_vec(pred)
            s = v.sum()
            p = np.sort(v / s)[::-1] if s > 0 else np.full_like(v, 1.0 / max(len(v), 1))
            top2gap = float(p[0] - p[1]) if len(p) > 1 else 1.0
            neff = float(1.0 / np.sum(p ** 2)) if np.sum(p ** 2) > 0 else float(len(p))
            feats += [float(p[0]), top2gap, response_entropy(pred, normalized=True), neff]
        return feats

    def _predict_target(self, record, pred) -> float:
        x = np.array(self._raw_features(record, pred), dtype=float)
        z = (x - self.mean_) / self.std_
        return float(np.dot(np.append(z, 1.0), self.w_))

    def fit(self, records, raw_preds):
        self.datasets_ = sorted({r.dataset_name for r in records})
        X = np.array([self._raw_features(r, p) for r, p in zip(records, raw_preds)], dtype=float)
        y = np.array([response_entropy(r.human_answer, normalized=True) for r in records])
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Z = np.hstack([(X - self.mean_) / self.std_, np.ones((len(X), 1))])
        # ridge solve; do not penalize the intercept (last column)
        pen = self.ridge * np.eye(Z.shape[1]); pen[-1, -1] = 0.0
        self.w_ = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
        # choose the blend gain that maximizes fit-set score
        grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        best_g, best_score = 0.0, float("-inf")
        for g in grid:
            self.gain = g
            scores = [simbench_score(self.transform(rec, raw), rec.human_answer,
                                     options=list(rec.options))
                      for rec, raw in zip(records, raw_preds)]
            mean = float(np.mean(scores)) if scores else float("-inf")
            if mean > best_score:
                best_g, best_score = g, mean
        self.gain = best_g
        return self

    def transform(self, record, pred):
        if self.w_ is None or self.gain == 0.0:
            return dict(pred)
        hp = response_entropy(pred, normalized=True)
        reg = min(max(self._predict_target(record, pred), 0.0), 1.0)
        target = min(max(hp + self.gain * (reg - hp), 0.0), 1.0)
        if abs(target - hp) < 1e-6:
            return dict(pred)
        return temper_to_entropy(pred, target)


class DirichletCalibrator(Calibrator):
    """Additive (Dirichlet) smoothing toward uniform: q_i ∝ q_i + alpha. The
    alpha=0 case is identity; large alpha approaches uniform. `fit` grid-searches
    alpha. Degenerate shrinkage-to-uniform falls out for free."""

    name = "dirichlet"

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(self, records, raw_preds):
        grid = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
        best_a, best_score = 0.0, float("-inf")
        for alpha in grid:
            cal = self.__class__(alpha=alpha)
            scores = [
                simbench_score(cal.transform(rec, raw), rec.human_answer, options=list(rec.options))
                for rec, raw in zip(records, raw_preds)
            ]
            mean = float(np.mean(scores)) if scores else float("-inf")
            if mean > best_score:
                best_a, best_score = alpha, mean
        self.alpha = best_a
        return self

    def transform(self, record, pred):
        keys, v = _as_vec(pred)
        s = v.sum()
        p = (v / s if s > 0 else np.full_like(v, 1.0 / len(v))) + self.alpha
        p = p / p.sum()
        return {k: float(x) for k, x in zip(keys, p)}


class ChainCalibrator(Calibrator):
    """Apply calibrators in sequence. `fit` threads predictions through each
    stage so later stages fit on earlier stages' output."""

    def __init__(self, stages: Sequence[Calibrator]) -> None:
        self.stages = list(stages)

    @property
    def name(self) -> str:  # type: ignore[override]
        return "+".join(s.name for s in self.stages) or "chain"

    def fit(self, records, raw_preds):
        cur = list(raw_preds)
        for stage in self.stages:
            stage.fit(records, cur)
            cur = [stage.transform(rec, p) for rec, p in zip(records, cur)]
        return self

    def transform(self, record, pred):
        out = dict(pred)
        for stage in self.stages:
            out = stage.transform(record, out)
        return out


def make_calibrator(name: str, **kwargs) -> Calibrator:
    """Factory: calibrator name -> instance."""
    table = {
        "identity": IdentityCalibrator,
        "temp": TempScaling,
        "entropy_temp": EntropyTempScaling,
        "entropy_target": EntropyTargetCalibrator,
        "feat_entropy_target": FeatureEntropyTargetCalibrator,
        "dirichlet": DirichletCalibrator,
    }
    if name == "chain":
        return ChainCalibrator([make_calibrator(s["name"], **s.get("kwargs", {}))
                                for s in kwargs.get("stages", [])])
    if name not in table:
        raise KeyError(f"Unknown calibrator {name!r}; have {sorted(table) + ['chain']}.")
    return table[name](**kwargs)
