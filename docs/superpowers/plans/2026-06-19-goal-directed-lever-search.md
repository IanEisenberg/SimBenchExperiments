# Goal-Directed Lever Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a semi-autonomous, Ladder-guarded loop that explores theory-motivated SimBench interventions ("levers") on a dev split, adjudicates each on a metered val split, and records the whole search path in a resumable experiment-tree ledger — without overfitting the holdout.

**Architecture:** Six new modules layered on the existing harness, plus a cost-capture extension to `LLMClient`. A `PipelineSpec` is a content-addressed description of a pipeline; a `Lever` is a pure `spec -> spec` transform; a `Ladder` gate accepts a candidate on val only if it beats the running best by a data-derived threshold η; a `Ledger` records every step as a DAG node keyed by config hash, carrying its OpenRouter spend; a `search` loop drives propose→score-dev→gate→ladder→log under a `SearchContract` with a dollar cap; a `report` layer produces the K-corrected, test-once headline and a cost summary. Scoring is injected (`score_fn`) so the whole loop is unit-testable with no network; the production `score_fn` measures real per-experiment dollar cost from the client.

**Tech Stack:** Python 3.12, numpy, pandas, pytest. No new dependencies. Builds on `simbench_exp.pipeline`, `simbench_exp.calibrate`, `simbench_exp.predict`, `simbench_exp.evaluate`, `simbench_exp.scoring`, `simbench_exp.experiment`, `simbench_exp.splits`.

## Global Constraints

- **Python 3.12**, `from __future__ import annotations` at the top of every new module (matches existing files).
- **No new third-party dependencies.** numpy/pandas/pytest only.
- **Unit tests never hit the network.** Inject fakes (a `score_fn` or a `FixedPredictor`), exactly as `tests/test_pipeline.py` does. Follow its `_rec(...)` fixture pattern for building `SimBenchRecord`s.
- **Style:** frozen dataclasses for value types; short `name`/`id` attributes; module-level docstring on every file; one clear responsibility per module.
- **SimBench score** is `simbench_exp.scoring.simbench_score`; aggregate via `simbench_exp.scoring.aggregate_score`; CIs via `simbench_exp.scoring.bootstrap_ci`. Always pass dataset-level `normalizers` from `simbench_exp.evaluate.build_normalizers` for reported numbers.
- **Determinism:** any timestamp or id generation is injected (`now_fn`) or derived from counts, so tests are reproducible. Never call `datetime.now()` inside a tested function without an injectable override.
- **Cost:** OpenRouter is the only paid layer (inside `score_fn`). Every experiment's new spend is captured (cache hits = $0), logged on its ledger node, and bounded by `SearchContract.cost_cap_usd` (default **$1000**). Cost rides in the `score_fn` breakdown dict under `"cost_usd"` — no signature change. See spec §9.
- **Commit after every task** with the message shown in the task's final step.

---

### Task 1: `PipelineSpec` — content-addressed pipeline description

**Files:**
- Create: `src/simbench_exp/spec.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Consumes: `simbench_exp.experiment.build_pipeline`, `simbench_exp.calibrate.ChainCalibrator` (Task 2 — for >1 calibrator; Task 1 only needs the 0/1 calibrator path, so guard the multi path behind a local import to avoid an ordering dependency).
- Produces:
  - `PipelineSpec(model: str, predictor: str, calibrators: list[dict], lever_path: list[str])` with `.resolved() -> dict` and `.config_hash() -> str`.
  - `build_from_spec(spec: PipelineSpec, *, client=None) -> Pipeline`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spec.py
"""Tests for PipelineSpec: stable content-addressing and pipeline assembly."""

from simbench_exp.spec import PipelineSpec, build_from_spec


def test_config_hash_is_stable_and_short():
    s = PipelineSpec(model="gemini-flash-lite", predictor="zero_shot")
    h = s.config_hash()
    assert isinstance(h, str) and len(h) == 16
    assert s.config_hash() == h  # deterministic across calls


def test_config_hash_ignores_lever_path_but_tracks_resolved_config():
    # lever_path is provenance only — two specs with the same resolved config
    # but different provenance are the SAME experiment (same hash).
    a = PipelineSpec(predictor="zero_shot", lever_path=["x"])
    b = PipelineSpec(predictor="zero_shot", lever_path=["y", "z"])
    assert a.config_hash() == b.config_hash()


def test_config_hash_changes_with_calibrator():
    a = PipelineSpec(predictor="zero_shot")
    b = PipelineSpec(predictor="zero_shot", calibrators=[{"name": "temp", "kwargs": {"T": 1.5}}])
    assert a.config_hash() != b.config_hash()


def test_build_from_spec_uniform_needs_no_client():
    # The "uniform" predictor builds with no client/network (registry thunk).
    pipe = build_from_spec(PipelineSpec(predictor="uniform"))
    assert pipe.name.startswith("uniform")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.spec'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/spec.py
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
```

This imports `make_calibrator` from `calibrate`, which Task 2 adds. To keep Task 1 green on its own, add a minimal stub now and replace it in Task 2:

```python
# Append to src/simbench_exp/calibrate.py (temporary stub, expanded in Task 2)
def make_calibrator(name: str, **kwargs) -> Calibrator:
    """Factory: calibrator name -> instance. Expanded in Task 2."""
    if name == "identity":
        return IdentityCalibrator()
    raise KeyError(f"Unknown calibrator {name!r}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/spec.py src/simbench_exp/calibrate.py tests/test_spec.py
git commit -m "feat: add content-addressed PipelineSpec + build_from_spec"
```

---

### Task 2: Calibrators the levers wrap — temperature, entropy-temperature, Dirichlet, chain

**Files:**
- Modify: `src/simbench_exp/calibrate.py` (replace the `make_calibrator` stub from Task 1; add four calibrators)
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `simbench_exp.scoring.response_entropy`, `simbench_exp.scoring.total_variation_distance`, `simbench_exp.scoring.simbench_score`.
- Produces:
  - `TempScaling(T: float = 1.0)` — flattens (T>1) or sharpens (T<1) a distribution.
  - `EntropyTempScaling(slope: float = 0.0)` — temperature `1 + slope * H(pred)`.
  - `DirichletCalibrator(alpha: float = 0.0)` — additive smoothing toward uniform.
  - `ChainCalibrator(stages: list[Calibrator])` — apply stages in order; `fit` threads predictions through.
  - `make_calibrator(name, **kwargs) -> Calibrator` mapping `"temp"|"entropy_temp"|"dirichlet"|"identity"|"chain"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
"""Tests for the recalibration calibrators the levers compose."""

import numpy as np

from simbench_exp.calibrate import (
    ChainCalibrator,
    DirichletCalibrator,
    EntropyTempScaling,
    IdentityCalibrator,
    TempScaling,
    make_calibrator,
)
from simbench_exp.data import SimBenchRecord


def _rec(options, truth):
    return SimBenchRecord(
        dataset_name="ESS", split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def test_temp_identity_at_one():
    rec = _rec(["A", "B"], {"A": 0.6, "B": 0.4})
    out = TempScaling(T=1.0).transform(rec, {"A": 0.7, "B": 0.3})
    assert abs(out["A"] - 0.7) < 1e-9 and abs(out["B"] - 0.3) < 1e-9


def test_temp_high_flattens_toward_uniform():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    out = TempScaling(T=50.0).transform(rec, {"A": 0.9, "B": 0.1})
    assert abs(out["A"] - 0.5) < 0.05  # nearly uniform


def test_temp_low_sharpens():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    out = TempScaling(T=0.2).transform(rec, {"A": 0.7, "B": 0.3})
    assert out["A"] > 0.7  # the larger mass grows


def test_dirichlet_smooths_toward_uniform():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    out = DirichletCalibrator(alpha=1.0).transform(rec, {"A": 1.0, "B": 0.0})
    assert 0.0 < out["B"] < 0.5  # zero mass pulled up


def test_entropy_temp_flattens_high_entropy_more():
    # A near-uniform prediction (high entropy) should be flattened more than a
    # peaked one under the same positive slope.
    rec = _rec(["A", "B", "C", "D"], {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    cal = EntropyTempScaling(slope=2.0)
    peaked = cal.transform(rec, {"A": 0.97, "B": 0.01, "C": 0.01, "D": 0.01})
    flat = cal.transform(rec, {"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.20})
    # The peaked input keeps more of its peak than the flat input keeps its spread shape.
    assert peaked["A"] > flat["A"]


def test_chain_applies_in_order():
    rec = _rec(["A", "B"], {"A": 0.5, "B": 0.5})
    chain = ChainCalibrator([TempScaling(T=2.0), DirichletCalibrator(alpha=0.0)])
    out = chain.transform(rec, {"A": 0.9, "B": 0.1})
    assert sum(out.values()) - 1.0 < 1e-9 and out["A"] < 0.9  # flattened


def test_temp_fit_recovers_flattening_temperature():
    # Raw predictions are systematically over-sharp vs truth; fit should pick T>1.
    recs, raws = [], []
    for _ in range(30):
        recs.append(_rec(["A", "B"], {"A": 0.55, "B": 0.45}))
        raws.append({"A": 0.95, "B": 0.05})
    cal = TempScaling().fit(recs, raws)
    assert cal.T > 1.0


def test_make_calibrator_dispatch():
    assert isinstance(make_calibrator("temp", T=1.5), TempScaling)
    assert isinstance(make_calibrator("entropy_temp", slope=1.0), EntropyTempScaling)
    assert isinstance(make_calibrator("dirichlet", alpha=0.5), DirichletCalibrator)
    assert isinstance(make_calibrator("identity"), IdentityCalibrator)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibrate.py -v`
Expected: FAIL with `ImportError: cannot import name 'TempScaling'`.

- [ ] **Step 3: Write minimal implementation**

Replace the temporary `make_calibrator` stub at the end of `src/simbench_exp/calibrate.py` with the following (keep the existing `Calibrator`/`IdentityCalibrator` above unchanged):

```python
# --- recalibration calibrators (append to src/simbench_exp/calibrate.py) ---------
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

    name = "chain"

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
```

Note: `ChainCalibrator.name` is defined as a property, which shadows the class attribute — remove the `name = "chain"` line if your linter objects; the property is authoritative.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibrate.py tests/test_spec.py -v`
Expected: PASS (all). Re-run Task 1's `test_build_from_spec_uniform_needs_no_client` to confirm the stub replacement didn't break it.

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/calibrate.py tests/test_calibrate.py
git commit -m "feat: add temperature/entropy-temp/Dirichlet/chain calibrators"
```

---

### Task 3: Lever registry — pure `spec -> spec` transforms

**Files:**
- Create: `src/simbench_exp/levers.py`
- Test: `tests/test_levers.py`

**Interfaces:**
- Consumes: `simbench_exp.spec.PipelineSpec`.
- Produces:
  - `Lever(id, hypothesis, mechanism, expected_direction, apply)` where `apply: Callable[[PipelineSpec, dict], PipelineSpec]`.
  - `LEVER_REGISTRY: dict[str, Lever]` with v1 ids: `recalib.global_temp`, `recalib.entropy_temp`, `recalib.dirichlet`, `model.swap`, `predictor.swap`.
  - `apply_lever(spec: PipelineSpec, lever_id: str, params: dict) -> PipelineSpec` — pure; appends to `lever_path`; raises `KeyError` off-registry.
  - `REGISTRY_VERSION: str`.

**Note on v1 scope:** the spec's seeded list also names `elicit.verbalized` and `ensemble.paraphrase`. `elicit.verbalized` is already the default predictor, and `ensemble.paraphrase` needs a new networked predictor; both are deferred. v1 implements the cleanly-composable subset (three recalibration levers + model swap + predictor/strategy swap, which covers `condition.delta` and `model.base_vs_instruct` by parameter). Document this in the module docstring.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_levers.py
"""Tests for the pre-registered lever set: purity, composition, provenance."""

import pytest

from simbench_exp.levers import LEVER_REGISTRY, apply_lever
from simbench_exp.spec import PipelineSpec


def test_global_temp_appends_calibrator_and_is_pure():
    base = PipelineSpec(predictor="zero_shot")
    out = apply_lever(base, "recalib.global_temp", {"T": 1.5})
    assert out.calibrators == [{"name": "temp", "kwargs": {"T": 1.5}}]
    assert base.calibrators == []  # original untouched (pure)
    assert out.lever_path == ["recalib.global_temp"]
    assert out.config_hash() != base.config_hash()


def test_model_swap_changes_model_only():
    base = PipelineSpec(model="gemini-flash-lite", predictor="zero_shot")
    out = apply_lever(base, "model.swap", {"model": "qwen-72b"})
    assert out.model == "qwen-72b" and out.predictor == "zero_shot"


def test_predictor_swap_changes_predictor():
    base = PipelineSpec(predictor="zero_shot")
    out = apply_lever(base, "predictor.swap", {"predictor": "uniform"})
    assert out.predictor == "uniform"


def test_levers_compose_and_accumulate_path():
    s = PipelineSpec(predictor="zero_shot")
    s = apply_lever(s, "recalib.global_temp", {"T": 2.0})
    s = apply_lever(s, "recalib.dirichlet", {"alpha": 0.05})
    assert [c["name"] for c in s.calibrators] == ["temp", "dirichlet"]
    assert s.lever_path == ["recalib.global_temp", "recalib.dirichlet"]


def test_off_registry_raises():
    with pytest.raises(KeyError):
        apply_lever(PipelineSpec(), "not.a.lever", {})


def test_registry_levers_have_required_metadata():
    for lever in LEVER_REGISTRY.values():
        assert lever.hypothesis and lever.mechanism and lever.expected_direction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_levers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.levers'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/levers.py
"""Lever registry — the pre-registered set of theory-motivated interventions.

A lever is a pure `(PipelineSpec, params) -> PipelineSpec` transform. Pre-
registering this set before any val contact freezes the multiple-comparisons
surface: the search loop may only explore these, and proposing anything off-
registry is a gated, human-approved act (see simbench_exp.search). Each lever carries
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_levers.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/levers.py tests/test_levers.py
git commit -m "feat: add pre-registered lever registry (recalib/model/predictor)"
```

---

### Task 4: The Ladder gate + bootstrap η

**Files:**
- Create: `src/simbench_exp/ladder.py`
- Test: `tests/test_ladder.py`

**Interfaces:**
- Consumes: numpy only.
- Produces:
  - `LadderGate(eta: float, best: float = -inf, k: int = 0)` with `.consider(score: float) -> bool` (increments `k`, accepts and updates `best` iff `score > best + eta`).
  - `bootstrap_eta(scores: Sequence[float], *, mult: float = 1.0, n_boot: int = 2000, seed: int = 0) -> float` — `mult ×` the std of bootstrapped means of a reference score vector (the val-S noise scale).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ladder.py
"""Tests for the Ladder gate (Blum & Hardt 2015) and bootstrap threshold."""

from simbench_exp.ladder import LadderGate, bootstrap_eta


def test_ladder_accepts_only_significant_improvement():
    gate = LadderGate(eta=1.0)
    assert gate.consider(10.0) is True   # first beats -inf
    assert gate.best == 10.0
    assert gate.consider(10.5) is False  # within eta -> reject, best unchanged
    assert gate.best == 10.0
    assert gate.consider(11.5) is True   # beats 10.0 + 1.0
    assert gate.best == 11.5


def test_ladder_counts_every_query():
    gate = LadderGate(eta=0.5)
    for s in [1.0, 2.0, 0.5, 3.0]:
        gate.consider(s)
    assert gate.k == 4  # K counts queries, not accepts


def test_bootstrap_eta_zero_for_constant_scores():
    assert bootstrap_eta([5.0] * 50, mult=2.0, seed=1) == 0.0


def test_bootstrap_eta_positive_and_scales_with_mult():
    scores = [float(i) for i in range(50)]
    e1 = bootstrap_eta(scores, mult=1.0, seed=1)
    e2 = bootstrap_eta(scores, mult=2.0, seed=1)
    assert e1 > 0.0
    assert abs(e2 - 2.0 * e1) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ladder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.ladder'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/ladder.py
"""The Ladder gate — the only door to the val set.

Blum & Hardt (2015, arXiv:1607.00091): release a new best only when it beats
the running best by more than a threshold eta. Under this rule the held-out
estimate's generalization error is bounded ~ (log(k·n)/n)^(1/3) regardless of
how many candidates k are tried — so an adaptive search cannot climb a noise
ladder. We choose this over Thresholdout because Thresholdout's differential-
privacy constants are impractical at our sample sizes.

eta is data-derived, not guessed: `bootstrap_eta` estimates the sampling noise
scale of the val mean-S, so the gate only fires on improvements larger than the
holdout's own finite-sample wobble.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class LadderGate:
    eta: float
    best: float = float("-inf")
    k: int = 0  # total queries spent (the global K when this gate is the run's gate)

    def consider(self, score: float) -> bool:
        """Spend one query; accept (and raise the bar) iff score > best + eta."""
        self.k += 1
        if score > self.best + self.eta:
            self.best = score
            return True
        return False


def bootstrap_eta(
    scores: Sequence[float],
    *,
    mult: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """`mult` × the std of bootstrapped means of a reference val score vector.

    This is the noise scale of the val mean-S: the smallest difference between
    two pipelines that is distinguishable from finite-sample wobble. Use a
    reference pipeline's per-record val scores as `scores`.
    """
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(mult * means.std())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ladder.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/ladder.py tests/test_ladder.py
git commit -m "feat: add Ladder gate + bootstrap eta for guarded val queries"
```

---

### Task 5: The experiment-tree ledger

**Files:**
- Create: `src/simbench_exp/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `dataclasses`, `json`.
- Produces:
  - `ExperimentNode` (frozen dataclass) with the fields from spec §6.1, including `cost_usd: float = 0.0` and `cum_cost_usd: float = 0.0` (trailing defaults so existing construction sites are unaffected).
  - `Ledger(path: str | Path)` with: `append(node)`, classmethod `load(path) -> Ledger`, `nodes: list[ExperimentNode]`, `by_id(node_id)`, `by_config_hash(h) -> ExperimentNode | None`, `val_nodes() -> list`, `global_k() -> int` (distinct queried config hashes), `best_val() -> float`, `total_cost() -> float` (sum of node `cost_usd`), `children(node_id) -> list`, `path_to_root(node_id) -> list`, `next_node_id() -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ledger.py
"""Tests for the append-only experiment-tree ledger."""

from simbench_exp.ledger import ExperimentNode, Ledger


def _node(led, config_hash, parent=None, lever="recalib.global_temp",
          dev=10.0, val=None, accepted=None, k=None, cost=0.0, cum=0.0):
    return ExperimentNode(
        node_id=led.next_node_id(), config_hash=config_hash, parent_id=parent,
        lever_id=lever, rationale="because", dev_score=dev, dev_breakdown={},
        val_score=val, eta=(0.5 if val is not None else None), accepted=accepted,
        global_k_at_query=k, timestamp="2026-06-19T00:00:00",
        cost_usd=cost, cum_cost_usd=cum,
    )


def test_append_and_load_roundtrip(tmp_path):
    p = tmp_path / "run.jsonl"
    led = Ledger(p)
    n0 = _node(led, "hashA")
    led.append(n0)
    reloaded = Ledger.load(p)
    assert len(reloaded.nodes) == 1
    assert reloaded.nodes[0].config_hash == "hashA"
    assert reloaded.by_id(n0.node_id).lever_id == "recalib.global_temp"


def test_by_config_hash_is_cache_lookup(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1))
    assert led.by_config_hash("hashA").val_score == 12.0
    assert led.by_config_hash("missing") is None


def test_global_k_counts_distinct_queried_hashes(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1))
    led.append(_node(led, "hashB", val=11.0, accepted=False, k=2))
    led.append(_node(led, "hashA", val=12.0, accepted=False, k=2))  # revisit, same hash
    led.append(_node(led, "hashC", val=None))  # dev-only, no val query
    assert led.global_k() == 2  # hashA, hashB; revisit and dev-only don't add


def test_best_val_among_queried(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1))
    led.append(_node(led, "hashB", val=9.0, accepted=False, k=2))
    assert led.best_val() == 12.0


def test_total_cost_sums_new_spend(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1, cost=3.0, cum=3.0))
    led.append(_node(led, "hashB", val=9.0, accepted=False, k=2, cost=2.0, cum=5.0))
    led.append(_node(led, "hashC", cost=0.0, cum=5.0))  # cache hit / dev-only, no new spend
    assert abs(led.total_cost() - 5.0) < 1e-9


def test_path_to_root_and_children(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    n0 = _node(led, "h0"); led.append(n0)
    n1 = _node(led, "h1", parent=n0.node_id); led.append(n1)
    n2 = _node(led, "h2", parent=n1.node_id); led.append(n2)
    assert [n.node_id for n in led.path_to_root(n2.node_id)] == [n0.node_id, n1.node_id, n2.node_id]
    assert [n.node_id for n in led.children(n0.node_id)] == [n1.node_id]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.ledger'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/ledger.py
"""The experiment-tree ledger — an append-only DAG of search states.

Each node is one experiment: a config-hashed pipeline state, the lever that
produced it, its dev score, and (only if a val query was spent) its val score,
the Ladder threshold, the verdict, and the cumulative global K at query time.
Because state is the ordered composition of levers, a node's full config is
reconstructable from its path to root, and any node is a resumption point.

The ledger is the source of truth for global K (distinct val-queried configs
across the WHOLE tree — branching never resets it) and for the cache: revisiting
a config_hash already queried returns its stored score and costs no new K.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentNode:
    node_id: str
    config_hash: str
    parent_id: str | None
    lever_id: str
    rationale: str
    dev_score: float
    dev_breakdown: dict
    val_score: float | None
    eta: float | None
    accepted: bool | None
    global_k_at_query: int | None
    timestamp: str
    cost_usd: float = 0.0       # NEW OpenRouter spend for this node (cache hits = 0.0)
    cum_cost_usd: float = 0.0   # running tree total at this node


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.nodes: list[ExperimentNode] = []

    # -- persistence -------------------------------------------------------
    def append(self, node: ExperimentNode) -> None:
        self.nodes.append(node)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(node)) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        led = cls(path)
        p = Path(path)
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        led.nodes.append(ExperimentNode(**json.loads(line)))
        return led

    # -- ids ---------------------------------------------------------------
    def next_node_id(self) -> str:
        return f"n{len(self.nodes):04d}"

    # -- lookups -----------------------------------------------------------
    def by_id(self, node_id: str) -> ExperimentNode | None:
        return next((n for n in self.nodes if n.node_id == node_id), None)

    def by_config_hash(self, config_hash: str) -> ExperimentNode | None:
        """First node with a val score for this config (the cache hit), else
        any node with this config, else None."""
        queried = [n for n in self.nodes if n.config_hash == config_hash and n.val_score is not None]
        if queried:
            return queried[0]
        return next((n for n in self.nodes if n.config_hash == config_hash), None)

    def val_nodes(self) -> list[ExperimentNode]:
        return [n for n in self.nodes if n.val_score is not None]

    def global_k(self) -> int:
        return len({n.config_hash for n in self.val_nodes()})

    def best_val(self) -> float:
        vals = [n.val_score for n in self.val_nodes()]
        return max(vals) if vals else float("-inf")

    def total_cost(self) -> float:
        """Total NEW OpenRouter spend across the whole tree (cache hits add 0)."""
        return float(sum(n.cost_usd for n in self.nodes))

    # -- tree --------------------------------------------------------------
    def children(self, node_id: str) -> list[ExperimentNode]:
        return [n for n in self.nodes if n.parent_id == node_id]

    def path_to_root(self, node_id: str) -> list[ExperimentNode]:
        chain: list[ExperimentNode] = []
        cur = self.by_id(node_id)
        while cur is not None:
            chain.append(cur)
            cur = self.by_id(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ledger.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/ledger.py tests/test_ledger.py
git commit -m "feat: add append-only experiment-tree ledger with global-K"
```

---

### Task 6: The search loop — contract, gate policy, propose→score→gate→ladder→log

**Files:**
- Create: `src/simbench_exp/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `simbench_exp.spec.PipelineSpec`, `simbench_exp.levers.apply_lever` / `LEVER_REGISTRY`, `simbench_exp.ladder.LadderGate`, `simbench_exp.ledger.Ledger`/`ExperimentNode`.
- Produces:
  - `GatePolicy(gate_off_registry_proposal=True, gate_on_val_query=False, gate_at_val_budget_frac=1.0, gate_on_surprise=True, gate_after_dev_steps=None, gate_at_cost_frac=0.8)`.
  - `SearchContract(goal, allowed_levers, autonomy_budget, val_query_budget, gate_policy, eta=0.0, cost_cap_usd=1000.0)`.
  - `Proposal(lever_id, params, rationale)`; a `propose_fn: Callable[[SearchState], Proposal | None]`.
  - `SearchState(spec, dev_best, val_best, k_spent, history)`.
  - `SearchOutcome(stop_reason, final_spec, steps)`.
  - `run_search(contract, ledger, *, dev, val, score_fn, propose_fn, now_fn, start_spec=None, client=None) -> SearchOutcome` where `score_fn: Callable[[PipelineSpec, Sequence], tuple[float, dict]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
"""Tests for the guarded search loop — offline, via injected score_fn/propose_fn."""

from simbench_exp.ledger import Ledger
from simbench_exp.search import GatePolicy, Proposal, SearchContract, run_search
from simbench_exp.spec import PipelineSpec


def _now():
    return "2026-06-19T00:00:00"


def _contract(**kw):
    base = dict(
        goal="improve S", allowed_levers=["recalib.global_temp", "recalib.dirichlet"],
        autonomy_budget=5, val_query_budget=3, gate_policy=GatePolicy(), eta=0.5,
    )
    base.update(kw)
    return SearchContract(**base)


def test_off_registry_proposal_halts(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    proposals = iter([Proposal("not.a.lever", {}, "rogue")])
    out = run_search(
        _contract(), led, dev=[1], val=[1],
        score_fn=lambda spec, recs: (10.0, {}),
        propose_fn=lambda state: next(proposals, None), now_fn=_now,
    )
    assert out.stop_reason == "off_registry_proposal"
    assert len(led.nodes) == 0  # nothing logged for a rejected rogue proposal


def test_accepts_improving_lever_and_spends_one_k(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # dev scores improve so the lever is promoted; val score clears eta.
    scores = {"dev": 20.0, "val": 30.0}
    proposals = iter([Proposal("recalib.global_temp", {"T": 1.5}, "flatten")])

    def score_fn(spec, recs):
        return (scores["dev"] if recs == ["dev"] else scores["val"]), {"entropy_temp": 0.0}

    out = run_search(
        _contract(autonomy_budget=1), led, dev=["dev"], val=["val"],
        score_fn=score_fn, propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    assert led.global_k() == 1
    assert led.nodes[0].accepted is True
    assert led.nodes[0].val_score == 30.0
    assert out.final_spec.lever_path == ["recalib.global_temp"]


def test_cached_config_revisit_spends_no_new_k(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # Two proposals that resolve to the SAME config (same lever+params).
    proposals = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "first"),
        Proposal("recalib.global_temp", {"T": 1.5}, "again"),
    ])

    def score_fn(spec, recs):
        return (20.0 if recs == ["dev"] else 30.0), {}

    out = run_search(
        _contract(autonomy_budget=2, gate_policy=GatePolicy(gate_on_surprise=False)),
        led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    # Second step revisits the same config_hash off the SAME parent -> cache hit.
    assert led.global_k() == 1


def test_val_budget_cap_halts(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # Each distinct lever improves dev (promoted) and clears val eta -> spends K.
    seq = [
        Proposal("recalib.global_temp", {"T": 1.5}, "a"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "b"),
        Proposal("recalib.dirichlet", {"alpha": 0.1}, "c"),
    ]
    proposals = iter(seq)
    n = {"i": 0}

    def score_fn(spec, recs):
        # Monotonically increasing val so each clears eta; dev always promotes.
        if recs == ["dev"]:
            return 20.0 + n["i"], {}
        n["i"] += 1
        return 30.0 + 5 * n["i"], {}

    out = run_search(
        _contract(autonomy_budget=10, val_query_budget=2,
                  gate_policy=GatePolicy(gate_on_surprise=False)),
        led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    assert out.stop_reason == "val_budget_reached"
    assert led.global_k() == 2


def test_propose_none_stops_cleanly(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    out = run_search(
        _contract(), led, dev=["dev"], val=["val"],
        score_fn=lambda spec, recs: (10.0, {}),
        propose_fn=lambda state: None, now_fn=_now,
    )
    assert out.stop_reason == "proposer_done"


def test_cost_cap_halts_before_exceeding(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # Each step's dev score carries $10 of new spend (in the breakdown); cap = $25,
    # so the loop logs two $10 steps and blocks the third before it can spend.
    proposals = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "a"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "b"),
        Proposal("recalib.dirichlet", {"alpha": 0.1}, "c"),
    ])

    def score_fn(spec, recs):
        # Dev call carries the cost; val call is free here. Dev always promotes.
        if recs == ["dev"]:
            return 20.0 + len(spec.lever_path), {"cost_usd": 10.0}
        return 30.0 + len(spec.lever_path), {"cost_usd": 0.0}

    out = run_search(
        # gate_at_cost_frac=1.0 disables the soft pause so we test the hard cap alone.
        _contract(autonomy_budget=10, val_query_budget=10, cost_cap_usd=25.0,
                  gate_policy=GatePolicy(gate_on_surprise=False, gate_at_cost_frac=1.0)),
        led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    assert out.stop_reason == "cost_cap_reached"
    assert abs(led.total_cost() - 20.0) < 1e-9       # two $10 steps logged; third blocked
    assert led.nodes[-1].cum_cost_usd == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.search'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/search.py
"""The guarded search loop — bounded autonomy over a frozen lever set.

The operator hands the loop a SearchContract (goal + allowed levers + budgets +
gate policy). Each step: a proposer picks a registered lever; the resulting spec
is scored freely on DEV; if it improves dev it is promoted to a single Ladder-
guarded VAL query (skipped — and no K spent — if that config was already
queried). Every step is logged to the ledger DAG. The loop halts at a budget, on
an off-registry proposal, or when a gate fires, returning control to the human.

Scoring is injected (`score_fn`) so the loop is fully unit-testable offline; the
production `score_fn` wraps `simbench_exp.evaluate.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .ladder import LadderGate
from .ledger import ExperimentNode, Ledger
from .levers import LEVER_REGISTRY, apply_lever
from .spec import PipelineSpec


@dataclass(frozen=True)
class GatePolicy:
    gate_off_registry_proposal: bool = True   # forced True (see run_search)
    gate_on_val_query: bool = False
    gate_at_val_budget_frac: float = 1.0
    gate_on_surprise: bool = True
    gate_after_dev_steps: int | None = None
    gate_at_cost_frac: float = 0.8            # pause when spend hits this fraction of the cap


@dataclass(frozen=True)
class SearchContract:
    goal: str
    allowed_levers: list[str]
    autonomy_budget: int
    val_query_budget: int
    gate_policy: GatePolicy
    eta: float = 0.0
    cost_cap_usd: float = 1000.0   # hard ceiling on cumulative NEW OpenRouter spend


@dataclass(frozen=True)
class Proposal:
    lever_id: str
    params: dict
    rationale: str


@dataclass
class SearchState:
    spec: PipelineSpec
    dev_best: float
    val_best: float
    k_spent: int
    cum_cost: float = 0.0        # cumulative NEW spend so far
    last_step_cost: float = 0.0  # cost of the previous step (projection for the pre-step guard)
    history: list = field(default_factory=list)


@dataclass
class SearchOutcome:
    stop_reason: str
    final_spec: PipelineSpec
    steps: int


ScoreFn = Callable[[PipelineSpec, Sequence], tuple[float, dict]]
ProposeFn = Callable[[SearchState], "Proposal | None"]
NowFn = Callable[[], str]


def run_search(
    contract: SearchContract,
    ledger: Ledger,
    *,
    dev: Sequence,
    val: Sequence,
    score_fn: ScoreFn,
    propose_fn: ProposeFn,
    now_fn: NowFn,
    start_spec: PipelineSpec | None = None,
    client=None,
) -> SearchOutcome:
    spec = start_spec or PipelineSpec()
    # Reconstruct global state from the ledger so resuming never resets K/best.
    gate = LadderGate(eta=contract.eta, best=ledger.best_val(), k=ledger.global_k())
    state = SearchState(spec=spec, dev_best=float("-inf"),
                        val_best=ledger.best_val(), k_spent=ledger.global_k(),
                        cum_cost=ledger.total_cost())

    steps = 0
    dev_steps = 0
    for _ in range(contract.autonomy_budget):
        proposal = propose_fn(state)
        if proposal is None:
            return SearchOutcome("proposer_done", state.spec, steps)

        # The one non-disableable gate: off-registry proposals always halt.
        if proposal.lever_id not in LEVER_REGISTRY or proposal.lever_id not in contract.allowed_levers:
            return SearchOutcome("off_registry_proposal", state.spec, steps)

        # Pre-step dollar guard: if the projected next spend (≈ last step's spend)
        # would cross the cap, halt before spending rather than overshoot it.
        if state.last_step_cost and state.cum_cost + state.last_step_cost > contract.cost_cap_usd:
            return SearchOutcome("cost_cap_reached", state.spec, steps)

        new_spec = apply_lever(state.spec, proposal.lever_id, proposal.params)
        chash = new_spec.config_hash()

        dev_score, breakdown = score_fn(new_spec, dev)
        dev_steps += 1
        step_cost = float(breakdown.get("cost_usd", 0.0))  # new spend for the dev score
        promote = dev_score > state.dev_best

        val_score = eta_used = accepted = k_at = None
        if promote:
            # Pre-query gates.
            gp = contract.gate_policy
            if gp.gate_on_val_query:
                return SearchOutcome("gate_on_val_query", state.spec, steps)
            if state.k_spent >= contract.val_query_budget:
                return SearchOutcome("val_budget_reached", state.spec, steps)

            cached = ledger.by_config_hash(chash)
            if cached is not None:
                # Cache hit: reuse the prior val score, spend NO new K.
                val_score = cached.val_score
                accepted = val_score is not None and val_score > gate.best + gate.eta
                eta_used = gate.eta
                k_at = state.k_spent
            else:
                val_score, val_breakdown = score_fn(new_spec, val)
                step_cost += float(val_breakdown.get("cost_usd", 0.0))  # val score spend
                accepted = gate.consider(val_score)  # spends one K, may raise best
                eta_used = gate.eta
                state.k_spent = gate.k
                k_at = gate.k

            if accepted:
                state.spec = new_spec
                state.dev_best = dev_score
                state.val_best = gate.best

        # Account the step's new spend before logging, so the node carries it.
        state.cum_cost += step_cost
        state.last_step_cost = step_cost

        node = ExperimentNode(
            node_id=ledger.next_node_id(), config_hash=chash,
            parent_id=_last_node_id(ledger), lever_id=proposal.lever_id,
            rationale=proposal.rationale, dev_score=dev_score, dev_breakdown=breakdown,
            val_score=val_score, eta=eta_used, accepted=accepted,
            global_k_at_query=k_at, timestamp=now_fn(),
            cost_usd=step_cost, cum_cost_usd=state.cum_cost,
        )
        ledger.append(node)
        state.history.append(node.node_id)
        steps += 1

        # Post-step gates.
        gp = contract.gate_policy
        if promote and state.k_spent >= contract.val_query_budget:
            return SearchOutcome("val_budget_reached", state.spec, steps)
        if (contract.val_query_budget
                and state.k_spent >= gp.gate_at_val_budget_frac * contract.val_query_budget
                and gp.gate_at_val_budget_frac < 1.0):
            return SearchOutcome("val_budget_frac_gate", state.spec, steps)
        if (contract.cost_cap_usd
                and state.cum_cost >= gp.gate_at_cost_frac * contract.cost_cap_usd
                and gp.gate_at_cost_frac < 1.0):
            return SearchOutcome("cost_frac_gate", state.spec, steps)
        if gp.gate_after_dev_steps and dev_steps >= gp.gate_after_dev_steps:
            return SearchOutcome("dev_steps_gate", state.spec, steps)
        if gp.gate_on_surprise and promote and accepted is False:
            # dev predicted improvement but val rejected -> surprising.
            return SearchOutcome("surprise_gate", state.spec, steps)

    return SearchOutcome("autonomy_budget_reached", state.spec, steps)


def _last_node_id(ledger: Ledger) -> str | None:
    return ledger.nodes[-1].node_id if ledger.nodes else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search.py -v`
Expected: PASS (6 passed). If `test_val_budget_cap_halts` stops with `val_budget_reached` at `global_k()==2`, the budget accounting is correct; `test_cost_cap_halts_before_exceeding` confirms the dollar cap blocks the third step.

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/search.py tests/test_search.py
git commit -m "feat: add guarded search loop with bounded autonomy + gates"
```

---

### Task 7: Per-call dollar cost capture in `LLMClient`

**Files:**
- Modify: `src/simbench_exp/config.py` (add `MODEL_PRICES` + `price_for`)
- Modify: `src/simbench_exp/llm.py` (add `Usage.cost_usd`, native-cost capture + token fallback, store cost on the cache record)
- Test: `tests/test_llm_cost.py`

**Interfaces:**
- Consumes: existing `Usage`/`LLMClient` in `simbench_exp.llm`; `simbench_exp.config`.
- Produces:
  - `MODEL_PRICES: dict[str, tuple[float, float]]` — model id → (prompt $/Mtok, completion $/Mtok).
  - `price_for(model, prices=None) -> tuple[float, float]` — table lookup; unknown model → `(0.0, 0.0)`.
  - `estimate_cost(prompt_tokens, completion_tokens, model, prices=None) -> float`.
  - `cost_of_record(record, prices=None) -> float` — prefer native `record["cost"]`, else token estimate.
  - `Usage.cost_usd: float` accumulating **new** spend only (cache hits add 0), included in `as_dict()`.

**Why:** `LLMClient.Usage` tracks tokens but not dollars (see spec §9.2). The production `score_fn` (Task 8) measures per-experiment cost as the delta of `client.usage.cost_usd` across a scoring call, so this primitive must land first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_cost.py
"""Cost capture: native-cost preference, token-price fallback, cache-hit = $0."""

from simbench_exp.config import price_for
from simbench_exp.llm import Usage, cost_of_record, estimate_cost


def test_estimate_cost_uses_price_table():
    prices = {"m": (1.0, 2.0)}  # $1/Mtok in, $2/Mtok out
    # 1e6 prompt tokens -> $1; 1e6 completion -> $2; total $3.
    assert abs(estimate_cost(1_000_000, 1_000_000, "m", prices) - 3.0) < 1e-9


def test_unknown_model_estimates_zero():
    assert price_for("nope", {"m": (1.0, 2.0)}) == (0.0, 0.0)
    assert estimate_cost(1_000_000, 1_000_000, "nope", {"m": (1.0, 2.0)}) == 0.0


def test_cost_of_record_prefers_native_cost():
    rec = {"cost": 0.42, "prompt_tokens": 9, "completion_tokens": 9, "model": "m"}
    assert cost_of_record(rec, {"m": (1000.0, 1000.0)}) == 0.42  # native wins over estimate


def test_cost_of_record_falls_back_to_estimate():
    rec = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "model": "m"}
    assert abs(cost_of_record(rec, {"m": (1.0, 2.0)}) - 1.0) < 1e-9


def test_usage_dict_includes_cost():
    u = Usage()
    u.cost_usd = 1.25
    assert u.as_dict()["cost_usd"] == 1.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_cost.py -v`
Expected: FAIL with `ImportError: cannot import name 'cost_of_record'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/simbench_exp/config.py`:

```python
# Per-model OpenRouter prices, USD per 1M tokens (prompt, completion). Used ONLY
# as a fallback when a response omits native usage.cost. Seed the models the
# search may swap in; an unknown model estimates to $0 (native cost still wins).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "google/gemini-2.0-flash-lite-001": (0.075, 0.30),
    "qwen/qwen-2.5-72b-instruct": (0.35, 0.40),
    "deepseek/deepseek-chat": (0.38, 0.89),
}


def price_for(model: str, prices: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    """(prompt, completion) $/Mtok for a model; (0.0, 0.0) if unknown."""
    table = MODEL_PRICES if prices is None else prices
    return table.get(model, (0.0, 0.0))
```

In `src/simbench_exp/llm.py`, extend the import and `Usage`, add the two module functions, and capture cost in `complete`:

```python
# change the existing config import line to also bring in price_for:
from .config import CACHE_DIR, OpenRouterConfig, price_for
```

```python
# add cost_usd to Usage (new spend only; cache hits add 0):
@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }
```

```python
# module-level helpers (place after _cache_key):
def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str,
                  prices: dict | None = None) -> float:
    """USD cost from token counts and a per-Mtok price table."""
    p_in, p_out = price_for(model, prices)
    return prompt_tokens / 1e6 * p_in + completion_tokens / 1e6 * p_out


def cost_of_record(record: dict, prices: dict | None = None) -> float:
    """Prefer OpenRouter's native `cost`; else estimate from tokens."""
    native = record.get("cost")
    if native is not None:
        return float(native)
    return estimate_cost(record.get("prompt_tokens", 0), record.get("completion_tokens", 0),
                         record.get("model", ""), prices)
```

In `complete`, request native cost and accumulate new spend (changes shown in context):

```python
        completion = self._client.chat.completions.create(
            **payload, extra_body={"usage": {"include": True}}
        )
        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        native_cost = getattr(usage, "cost", None)
        if native_cost is None and usage is not None:
            native_cost = (getattr(usage, "model_extra", None) or {}).get("cost")
        record = {
            "text": text,
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cost": native_cost,  # may be None -> estimated below
            "params": {k: v for k, v in payload.items() if k != "messages"},
        }

        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += record["prompt_tokens"]
            self.usage.completion_tokens += record["completion_tokens"]
            self.usage.cost_usd += cost_of_record(record)  # new spend only
```

Note: a cache hit returns before this block, so `cost_usd` accumulates **new** spend only — matching the K-accounting rule that re-walking explored territory is free.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_cost.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/config.py src/simbench_exp/llm.py tests/test_llm_cost.py
git commit -m "feat: capture per-call OpenRouter cost in LLMClient (native + fallback)"
```

---

### Task 8: Reporting — production `score_fn`, K-corrected band, test-once headline

**Files:**
- Create: `src/simbench_exp/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `simbench_exp.spec.build_from_spec`, `simbench_exp.evaluate.evaluate`/`summarize`, `simbench_exp.scoring.aggregate_score`, `simbench_exp.ledger.Ledger`.
- Produces:
  - `make_score_fn(normalizers, *, client=None, max_workers=8) -> ScoreFn` — the production scorer (`spec, records -> (mean_score, breakdown)`), `breakdown` carrying entropy-binned means and `"cost_usd"` (the new OpenRouter spend for this scoring call, measured as the client's `usage.cost_usd` delta).
  - `k_corrected_band(mean: float, n: int, k: int, *, mult: float = 1.0) -> tuple[float, float]` — widen a band by the Ladder factor `(log(max(k,1)·n)/n)^(1/3)`.
  - `final_report(spec, test, normalizers, ledger, *, client=None) -> dict` — score the frozen spec ONCE on test; return raw mean, CI, K-corrected band, `global_k`, and `total_cost_usd` (the whole search's spend, from the ledger).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
"""Tests for reporting: K-corrected band and the production score_fn shape."""

import numpy as np

from simbench_exp.evaluate import build_normalizers
from simbench_exp.ledger import ExperimentNode, Ledger
from simbench_exp.report import final_report, k_corrected_band, make_score_fn
from simbench_exp.data import SimBenchRecord
from simbench_exp.spec import PipelineSpec


def _rec(options, truth, ds="ESS"):
    return SimBenchRecord(
        dataset_name=ds, split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def test_k_corrected_band_widens_with_k():
    lo1, hi1 = k_corrected_band(50.0, n=200, k=1)
    lo100, hi100 = k_corrected_band(50.0, n=200, k=100)
    assert (hi100 - lo100) > (hi1 - lo1)  # more queries -> wider honest band
    assert lo1 <= 50.0 <= hi1


def test_score_fn_returns_mean_and_breakdown():
    recs = [_rec(["A", "B"], {"A": 0.7, "B": 0.3}), _rec(["A", "B"], {"A": 0.4, "B": 0.6})]
    norms = build_normalizers(recs)
    score_fn = make_score_fn(norms)
    spec = PipelineSpec(predictor="uniform")  # no network
    mean, breakdown = score_fn(spec, recs)
    assert isinstance(mean, float)
    assert "by_entropy" in breakdown
    assert breakdown["cost_usd"] == 0.0  # uniform predictor makes no API calls


def test_final_report_scores_uniform_to_zero(tmp_path):
    # Uniform predictor averages to 0 under Eq.2 normalizers (paper property).
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(40):
        a = float(rng.random())
        recs.append(_rec(["A", "B"], {"A": a, "B": 1 - a}))
    norms = build_normalizers(recs)
    led = Ledger(tmp_path / "r.jsonl")
    rep = final_report(PipelineSpec(predictor="uniform"), recs, norms, led)
    assert abs(rep["mean_score"]) < 1e-9
    assert rep["global_k"] == 0
    assert "k_band_low" in rep and "k_band_high" in rep
    assert rep["total_cost_usd"] == 0.0  # empty ledger -> no spend
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.report'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/report.py
"""Reporting — the production scorer and the honest, test-once headline.

`make_score_fn` is the real `score_fn` the search loop consumes: it builds a
pipeline from a spec, runs the evaluation harness, and returns the mean S plus
an entropy-binned breakdown. `final_report` scores the frozen winning spec on
the TEST split exactly once and reports the raw mean, its bootstrap CI, and a
band widened for the realized global K (the Ladder generalization factor), so
the headline number discloses how much search produced it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .evaluate import evaluate, summarize
from .ledger import Ledger
from .scoring import response_entropy
from .spec import PipelineSpec, build_from_spec


def _pipe_cost(pipe) -> float:
    """Best-effort cumulative $ spend behind a pipeline's predictor (0 if none).

    Sees through the post-stratification wrapper to its base predictor's client,
    mirroring simbench_exp.experiment.pipeline_model."""
    pred = pipe.predictor
    client = getattr(pred, "client", None) or getattr(getattr(pred, "base", None), "client", None)
    usage = getattr(client, "usage", None)
    return float(getattr(usage, "cost_usd", 0.0)) if usage is not None else 0.0


def make_score_fn(normalizers, *, client=None, max_workers: int = 8):
    """Build the production score_fn: (spec, records) -> (mean_score, breakdown)."""

    def score_fn(spec: PipelineSpec, records: Sequence) -> tuple[float, dict]:
        pipe = build_from_spec(spec, client=client)
        before = _pipe_cost(pipe)
        df = evaluate(pipe, records, normalizers=normalizers,
                      max_workers=max_workers, progress=False)
        cost = _pipe_cost(pipe) - before  # new OpenRouter spend for this scoring call
        mean = float(df["score"].mean()) if len(df) else float("nan")
        # Entropy-binned breakdown (consensus vs diverse), the SimBench axis.
        bins = pd.cut(df["truth_entropy"], [0, 0.33, 0.66, 1.0], include_lowest=True)
        by_entropy = {str(k): float(v) for k, v in df.groupby(bins, observed=True)["score"].mean().items()}
        return mean, {"by_entropy": by_entropy, "n": len(df), "cost_usd": cost}

    return score_fn


def k_corrected_band(mean: float, n: int, k: int, *, mult: float = 1.0) -> tuple[float, float]:
    """Widen a band around `mean` by the Ladder factor (log(k·n)/n)^(1/3).

    This reflects that a number selected from k adaptive val queries generalizes
    only up to ~(log(k·n)/n)^(1/3); a single query (k=1) gives the tightest band.
    """
    n = max(1, n)
    k = max(1, k)
    half = mult * (math.log(k * n) / n) ** (1.0 / 3.0) * 100.0  # on the 0-100 S scale
    return (mean - half, mean + half)


def final_report(spec: PipelineSpec, test: Sequence, normalizers, ledger: Ledger,
                 *, client=None) -> dict:
    """Score the frozen spec ONCE on test; report raw mean, CI, and K-band."""
    pipe = build_from_spec(spec, client=client)
    df = evaluate(pipe, test, normalizers=normalizers, progress=False)
    summ = summarize(df)
    k = ledger.global_k()
    lo, hi = k_corrected_band(summ["mean_score"], n=len(df), k=k)
    return {
        "pipeline": pipe.name,
        "lever_path": list(spec.lever_path),
        "n": len(df),
        "mean_score": summ["mean_score"],
        "ci_low": summ["ci_low"],
        "ci_high": summ["ci_high"],
        "global_k": k,
        "k_band_low": lo,
        "k_band_high": hi,
        "total_cost_usd": ledger.total_cost(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/report.py tests/test_report.py
git commit -m "feat: add K-corrected reporting and production score_fn"
```

---

### Task 9: Tree renderer + full-loop integration test

**Files:**
- Create: `src/simbench_exp/tree_view.py`
- Test: `tests/test_tree_view.py`
- Test: `tests/test_search_integration.py`

**Interfaces:**
- Consumes: `simbench_exp.ledger.Ledger`.
- Produces: `tree_lines(ledger: Ledger) -> list[str]` — an ASCII tree, one line per node, marking accept (`✓`) / reject (`✗`) / dev-only (`·`) and showing lever id + scores. Plus an integration test exercising the full propose→ledger→report chain offline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_view.py
"""Tests for the ASCII experiment-tree renderer."""

from simbench_exp.ledger import ExperimentNode, Ledger
from simbench_exp.tree_view import tree_lines


def _n(led, h, parent, lever, val, accepted):
    return ExperimentNode(
        node_id=led.next_node_id(), config_hash=h, parent_id=parent, lever_id=lever,
        rationale="r", dev_score=10.0, dev_breakdown={}, val_score=val,
        eta=0.5, accepted=accepted, global_k_at_query=(1 if val else None),
        timestamp="2026-06-19T00:00:00",
    )


def test_tree_lines_marks_status_and_nests(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    n0 = _n(led, "h0", None, "recalib.global_temp", 30.0, True); led.append(n0)
    n1 = _n(led, "h1", n0.node_id, "recalib.dirichlet", 29.0, False); led.append(n1)
    n2 = _n(led, "h2", n0.node_id, "model.swap", None, None); led.append(n2)
    lines = tree_lines(led)
    assert len(lines) == 3
    assert any("✓" in ln and "recalib.global_temp" in ln for ln in lines)
    assert any("✗" in ln and "recalib.dirichlet" in ln for ln in lines)
    assert any("·" in ln and "model.swap" in ln for ln in lines)
    # children are indented deeper than their parent
    parent_line = next(ln for ln in lines if "recalib.global_temp" in ln)
    child_line = next(ln for ln in lines if "recalib.dirichlet" in ln)
    assert (len(child_line) - len(child_line.lstrip())) > (len(parent_line) - len(parent_line.lstrip()))
```

```python
# tests/test_search_integration.py
"""End-to-end (offline) integration: a short search produces a coherent ledger
and a test-once report, with global K honestly counted across the tree."""

from simbench_exp.ledger import Ledger
from simbench_exp.report import final_report
from simbench_exp.search import GatePolicy, Proposal, SearchContract, run_search
from simbench_exp.evaluate import build_normalizers
from simbench_exp.data import SimBenchRecord
from simbench_exp.spec import PipelineSpec


def _rec(options, truth):
    return SimBenchRecord(
        dataset_name="ESS", split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def test_full_loop_offline(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    contract = SearchContract(
        goal="improve S", allowed_levers=["recalib.global_temp", "recalib.dirichlet"],
        autonomy_budget=4, val_query_budget=3,
        gate_policy=GatePolicy(gate_on_surprise=False), eta=0.5,
    )
    seq = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "flatten over-sharp output"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "lift zeroed tails"),
    ])

    def score_fn(spec, recs):
        # dev and val both improve as levers stack, so both are promoted+accepted.
        # Each scoring call reports $1 of new spend, so each step costs $2 (dev+val).
        depth = len(spec.lever_path)
        return (20.0 + 5 * depth, {"by_entropy": {}, "cost_usd": 1.0})

    out = run_search(
        contract, led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(seq, None), now_fn=lambda: "2026-06-19T00:00:00",
    )
    assert out.stop_reason == "proposer_done"
    assert led.global_k() == 2
    assert out.final_spec.lever_path == ["recalib.global_temp", "recalib.dirichlet"]
    assert abs(led.total_cost() - 4.0) < 1e-9  # two steps × (dev $1 + val $1)

    # Report on a held-out "test" set with the uniform predictor (no network).
    recs = [_rec(["A", "B"], {"A": 0.7, "B": 0.3})]
    norms = build_normalizers(recs)
    rep = final_report(PipelineSpec(predictor="uniform"), recs, norms, led)
    assert rep["global_k"] == 2  # report reads K from the ledger, not the run
    assert abs(rep["total_cost_usd"] - 4.0) < 1e-9  # spend carried through to the report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tree_view.py tests/test_search_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simbench_exp.tree_view'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simbench_exp/tree_view.py
"""ASCII renderer for the experiment-tree ledger.

Turns the search DAG into an indented, status-marked tree for the notebook and
the presentation: ✓ accepted on val, ✗ rejected by the Ladder, · dev-only (no
val query spent). This rendered tree plus the query ledger is the rigor artifact
that shows the search did not launder noise.
"""

from __future__ import annotations

from .ledger import ExperimentNode, Ledger


def _mark(node: ExperimentNode) -> str:
    if node.val_score is None:
        return "·"
    return "✓" if node.accepted else "✗"


def _fmt(node: ExperimentNode) -> str:
    val = "" if node.val_score is None else f" val={node.val_score:.2f}"
    cost = f" ${node.cost_usd:.2f}" if node.cost_usd else ""
    return f"{_mark(node)} {node.lever_id} (dev={node.dev_score:.2f}{val}){cost}"


def tree_lines(ledger: Ledger, depth: int = 0, parent_id: str | None = None) -> list[str]:
    """Render the tree as indented lines, depth-first from the roots."""
    lines: list[str] = []
    for node in ledger.children(parent_id) if parent_id else _roots(ledger):
        lines.append("  " * depth + _fmt(node))
        lines.extend(tree_lines(ledger, depth + 1, node.node_id))
    return lines


def _roots(ledger: Ledger) -> list[ExperimentNode]:
    return [n for n in ledger.nodes if n.parent_id is None]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tree_view.py tests/test_search_integration.py -v`
Expected: PASS (2 passed). Then run the full suite: `pytest -q` — all green.

- [ ] **Step 5: Commit**

```bash
git add src/simbench_exp/tree_view.py tests/test_tree_view.py tests/test_search_integration.py
git commit -m "feat: add experiment-tree renderer + offline integration test"
```

---

## Notes for the implementer

- **The notebook wiring** (a cell that builds a 3-way split via `make_split(fractions={"dev":..,"val":..,"test":..})`, a `make_score_fn(normalizers)`, an interactive `propose_fn` driven by you/Claude, and `tree_lines(ledger)` to view progress) is assembled in `notebooks/01_experiments.ipynb` after these modules land. It is not a coded task here because it is exploratory and network-bound; the modules above are everything it needs.
- **η in production:** call `bootstrap_eta(reference_val_scores, mult=...)` once on a baseline pipeline's per-record val scores, then pass the result as `SearchContract.eta`. Start `mult` conservative (favor false rejects).
- **Resuming a branch:** `Ledger.load(path)`, pick a `node_id`, rebuild its spec by replaying `path_to_root(node_id)`'s lever ids from a base `PipelineSpec` via `apply_lever`, and pass it as `start_spec`. Because `run_search` seeds the Ladder from `ledger.best_val()`/`ledger.global_k()` **and `cum_cost` from `ledger.total_cost()`**, both global K and cumulative spend survive the resume — the $1000 cap is enforced across resumed branches, not per-run.
- **Cost in production:** pass a single shared `client` into `make_score_fn(normalizers, client=client)` so the `usage.cost_usd` delta isolates each experiment's spend; with no shared client a fresh one is built per call and the delta still equals that call's spend. Seed `MODEL_PRICES` for any model `model.swap` may select, so the fallback estimate is meaningful when a response lacks native `usage.cost`. Watch the live total with `ledger.total_cost()`.

---

## Self-Review

**Spec coverage:**
- §2 three-way split discipline → notebook wiring note + `make_split` (existing); split *roles* enforced by the loop only touching `val` through the gate. ✓
- §3 lever registry (pre-registered, frozen, metadata) → Task 3. ✓ (v1 subset documented; `elicit.verbalized`/`ensemble.paraphrase` explicitly deferred.)
- §4 goal/scope contract + gate policy + non-disableable off-registry gate → Task 6. ✓
- §5 Ladder gate + data-derived η + global-K accounting → Task 4 (gate/η) + Task 5 (`global_k`) + Task 6 (wiring). ✓
- §6 experiment-tree ledger (node schema incl. cost, resume, free dev replay, persistence, viz) → Task 5 + Task 9 renderer + resume note. ✓
- §7 reporting (test-once, K-corrected band, ceiling normalization, cost summary) → Task 8. Ceiling-normalization reuses the existing reliability-ceiling helper from the harness; `final_report` exposes raw mean + CI + K-band + `total_cost_usd`, and ceiling normalization is applied at the notebook layer where the ceiling normalizers live. ✓
- §9 compute & cost model (who-pays boundary, per-experiment cost, $1000 cap) → Task 7 (capture) + Task 5 (node cost fields) + Task 6 (cap enforcement + frac gate) + Task 8 (score_fn cost + report total). ✓
- §10 open parameters → surfaced as `mult`, `eta`, budgets, `cost_cap_usd`, price table — all injectable. ✓

**Placeholder scan:** no TBD/TODO; the prior `if False else` leftover in `final_report` is now written in final form. All test and impl steps carry complete code.

**Type consistency:** `score_fn` signature `(PipelineSpec, Sequence) -> (float, dict)` is identical in Task 6 (consumer), Task 8 (`make_score_fn`), and both Task 9 tests; the breakdown dict's `"cost_usd"` key is written by Task 8 and read by Task 6 via `.get("cost_usd", 0.0)`. `apply_lever(spec, lever_id, params)` matches across Tasks 3/6. `ExperimentNode` field set (incl. trailing `cost_usd`/`cum_cost_usd` defaults) is identical in Tasks 5/6/9. `config_hash() -> str` (Task 1) is the cache key in Tasks 5/6. `LadderGate.consider` returns the accept bool used in Task 6. `cost_of_record`/`estimate_cost`/`price_for` (Task 7) are consumed only inside `LLMClient`. Consistent.
