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

from .data import SimBenchRecord


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
import numpy as np

from .scoring import response_entropy, simbench_score


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
    `fit` grid-searches T to minimize mean SimBench TVD on held-out data."""

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
        "dirichlet": DirichletCalibrator,
    }
    if name == "chain":
        return ChainCalibrator([make_calibrator(s["name"], **s.get("kwargs", {}))
                                for s in kwargs.get("stages", [])])
    if name not in table:
        raise KeyError(f"Unknown calibrator {name!r}; have {sorted(table) + ['chain']}.")
    return table[name](**kwargs)
