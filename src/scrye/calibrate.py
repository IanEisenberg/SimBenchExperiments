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


def make_calibrator(name: str, **kwargs) -> Calibrator:
    """Factory: calibrator name -> instance. Expanded in Task 2."""
    if name == "identity":
        return IdentityCalibrator()
    raise KeyError(f"Unknown calibrator {name!r}.")
