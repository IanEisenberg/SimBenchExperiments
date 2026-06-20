"""PipelineSpec — a serializable, content-addressed description of a pipeline.

A spec names a model key, a predictor registry name, and an ordered list of
calibrator stages. Its `config_hash` is a stable SHA-256 over the *resolved*
configuration (model + predictor + calibrators) — NOT over `lever_path`, which
is provenance only. Two different lever orders that resolve to the same
pipeline are therefore the same experiment (same hash, same cache entry).

`build_from_spec` turns a spec into a concrete `Pipeline` using the existing
experiment registry, so a spec is the unit the search loop and ledger pass
around instead of a live pipeline object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .calibrate import make_calibrator
from .experiment import build_pipeline
from .pipeline import Pipeline


@dataclass
class PipelineSpec:
    model: str = "gemini-flash-lite"
    predictor: str = "zero_shot"
    calibrators: list[dict] = field(default_factory=list)  # [{"name": str, "kwargs": {...}}]
    lever_path: list[str] = field(default_factory=list)     # provenance, NOT hashed

    def resolved(self) -> dict:
        """The hash-bearing configuration (excludes lever_path)."""
        return {
            "model": self.model,
            "predictor": self.predictor,
            "calibrators": self.calibrators,
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.resolved(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_from_spec(spec: PipelineSpec, *, client=None) -> Pipeline:
    """Assemble a concrete Pipeline from a spec via the experiment registry."""
    cals = [make_calibrator(c["name"], **c.get("kwargs", {})) for c in spec.calibrators]
    if not cals:
        calibrator = None
    elif len(cals) == 1:
        calibrator = cals[0]
    else:
        from .calibrate import ChainCalibrator  # Task 2

        calibrator = ChainCalibrator(cals)
    return build_pipeline(
        model=spec.model, predictor=spec.predictor, calibrator=calibrator, client=client
    )
