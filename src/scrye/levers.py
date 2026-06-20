"""Lever registry — the pre-registered set of theory-motivated interventions.

A lever is a pure `(PipelineSpec, params) -> PipelineSpec` transform. Pre-
registering this set before any val contact freezes the multiple-comparisons
surface: the search loop may only explore these, and proposing anything off-
registry is a gated, human-approved act (see scrye.search). Each lever carries
its hypothesis, mechanism, and expected direction so the loop's rationale is
grounded in the SimBench findings, not vibes.

v1 scope: three recalibration levers (wrap a calibrator), plus model and
predictor swaps that, by parameter, cover the spec's `model.base_vs_instruct`
and `condition.delta` cases. The spec's `elicit.verbalized` is already the
default predictor; `ensemble.paraphrase` needs a new networked predictor and is
deferred to a follow-up.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from .spec import PipelineSpec

REGISTRY_VERSION = "v1"


@dataclass(frozen=True)
class Lever:
    id: str
    hypothesis: str
    mechanism: str
    expected_direction: str
    apply: Callable[[PipelineSpec, dict], PipelineSpec]


def _with(spec: PipelineSpec, **changes) -> PipelineSpec:
    """Return a deep copy of `spec` with fields replaced (keeps levers pure)."""
    new = PipelineSpec(
        model=changes.get("model", spec.model),
        predictor=changes.get("predictor", spec.predictor),
        calibrators=copy.deepcopy(changes.get("calibrators", spec.calibrators)),
        lever_path=list(spec.lever_path),
    )
    return new


def _append_calibrator(name: str):
    def _apply(spec: PipelineSpec, params: dict) -> PipelineSpec:
        cals = copy.deepcopy(spec.calibrators) + [{"name": name, "kwargs": dict(params)}]
        return _with(spec, calibrators=cals)
    return _apply


def _set_model(spec: PipelineSpec, params: dict) -> PipelineSpec:
    return _with(spec, model=params["model"])


def _set_predictor(spec: PipelineSpec, params: dict) -> PipelineSpec:
    return _with(spec, predictor=params["predictor"])


LEVER_REGISTRY: dict[str, Lever] = {
    "recalib.global_temp": Lever(
        "recalib.global_temp",
        "One global temperature corrects mode-seeking over-sharpness.",
        "RLHF minimizes mode-seeking KL -> over-confident distributions.",
        "↑ overall S, especially mid-entropy items.",
        _append_calibrator("temp"),
    ),
    "recalib.entropy_temp": Lever(
        "recalib.entropy_temp",
        "Flatten more where the model is diffuse — correct high-entropy items most.",
        "SimBench: instruct hurts high-entropy, r=-0.942 with response entropy.",
        "↑ high-entropy S without regressing consensus.",
        _append_calibrator("entropy_temp"),
    ),
    "recalib.dirichlet": Lever(
        "recalib.dirichlet",
        "Additive smoothing toward uniform lifts under-predicted tails.",
        "Mode collapse zeroes plausible options; smoothing restores dispersion.",
        "↑ S on multi-modal items.",
        _append_calibrator("dirichlet"),
    ),
    "model.swap": Lever(
        "model.swap",
        "A base/less-aligned model is a better high-entropy simulator.",
        "Alignment–simulation tradeoff: mass-covering > mode-seeking on diverse items.",
        "↑ high-entropy S; may need less recalibration.",
        _set_model,
    ),
    "predictor.swap": Lever(
        "predictor.swap",
        "Predict the segment's shift (delta) rather than 'be this group'.",
        "SimBench: demographic conditioning degrades all models; delta dodges it.",
        "↑ counterfactual alignment without prior degradation.",
        _set_predictor,
    ),
}


def apply_lever(spec: PipelineSpec, lever_id: str, params: dict) -> PipelineSpec:
    """Apply a registered lever, returning a new spec with extended lever_path.

    Raises KeyError if `lever_id` is not in the frozen registry — the search
    loop relies on this to enforce the off-registry gate.
    """
    if lever_id not in LEVER_REGISTRY:
        raise KeyError(f"Off-registry lever {lever_id!r}; registered: {sorted(LEVER_REGISTRY)}.")
    out = LEVER_REGISTRY[lever_id].apply(spec, params or {})
    out.lever_path = list(spec.lever_path) + [lever_id]
    return out
