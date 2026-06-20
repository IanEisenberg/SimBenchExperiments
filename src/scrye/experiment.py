"""Experiment harness — compare simulation systems and models on topline numbers.

This is the seam the experiments notebook drives. The two things you iterate on
are made first-class *variables*:

  * **the simulation system** — which :class:`~scrye.predict.Predictor` (and
    optional :class:`~scrye.calibrate.Calibrator`) is inside the pipeline, and
  * **the model** — which OpenRouter model the predictor calls.

Both are swapped without touching the evaluation/scoring code. The headline
entry point is :func:`compare`, which runs any number of named pipelines over
the *same* records and returns one tidy topline row per system (mean SimBench
score + bootstrap CI, plus counterfactual-sensitivity alignment). Per-system
result frames are stashed in ``df.attrs["results"]`` for drill-down and figures.

Normalizers (SimBench Eq. 2) must be built once from the FULL split and passed
in, so scores stay comparable across systems and subsamples — see
:func:`scrye.evaluate.build_normalizers`.

Example::

    from scrye.experiment import build_pipeline, compare, model_sweep

    systems = {
        "zero-shot": build_pipeline(model="gemini-flash-lite"),
        "uniform":   build_pipeline(predictor="uniform"),
    }
    compare(systems, dev_eval, normalizers=norm)

    # or sweep one system across models:
    model_sweep(["gemini-flash-lite", "qwen-72b"], dev_eval, normalizers=norm)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .calibrate import Calibrator
from .config import DEFAULT_MODEL, OPENROUTER_MODELS
from .data import SimBenchRecord
from .evaluate import counterfactual_sensitivity, evaluate, summarize
from .llm import LLMClient
from .pipeline import Pipeline
from .predict import Predictor, UniformPredictor, ZeroShotPredictor


# -- model selection -------------------------------------------------------
def resolve_model(model: str) -> str:
    """Accept a short key from :data:`~scrye.config.OPENROUTER_MODELS` or a raw
    model id. ``"qwen-72b"`` -> ``"qwen/qwen-2.5-72b-instruct"``; an unknown
    string is assumed to already be a full model id and passed through."""
    return OPENROUTER_MODELS.get(model, model)


def make_client(model: str = DEFAULT_MODEL, **client_kwargs) -> LLMClient:
    """Build a cached :class:`LLMClient` for a model key or id."""
    return LLMClient(resolve_model(model), **client_kwargs)


# -- simulation-system registry -------------------------------------------
# name -> factory(get_client) -> Predictor. Register new simulation systems here
# so the notebook (and `build_pipeline`) can select them by a short string.
# `get_client` is a *thunk*: call it only if the system needs an LLM, so
# building a non-LLM system (e.g. "uniform") never constructs a client or
# requires an API key.
ClientThunk = Callable[[], LLMClient]
PREDICTOR_REGISTRY: dict[str, Callable[[ClientThunk], Predictor]] = {
    "zero_shot": lambda get_client: ZeroShotPredictor(get_client()),
    "uniform": lambda get_client: UniformPredictor(),
}


def build_pipeline(
    *,
    model: str = DEFAULT_MODEL,
    predictor: str | Predictor = "zero_shot",
    calibrator: Calibrator | None = None,
    client: LLMClient | None = None,
    **client_kwargs,
) -> Pipeline:
    """Assemble a pipeline from a model + simulation-system + optional calibrator.

    Args:
        model: model key or id (ignored if a ready ``client`` or a ready
            ``predictor`` instance is supplied).
        predictor: a registry name (e.g. ``"zero_shot"``) or a ready
            :class:`Predictor` instance.
        calibrator: optional correction stage (defaults to identity).
        client: reuse an existing client instead of building one (lets several
            systems share one cache/usage counter).
        **client_kwargs: forwarded to :func:`make_client` (e.g. ``temperature``).
    """
    if isinstance(predictor, Predictor):
        pred = predictor
    else:
        if predictor not in PREDICTOR_REGISTRY:
            raise KeyError(
                f"Unknown predictor {predictor!r}; "
                f"registered: {sorted(PREDICTOR_REGISTRY)}."
            )
        # Lazily build (and memoize) the client only if the system asks for it.
        _cache: dict[str, LLMClient] = {}

        def get_client() -> LLMClient:
            if "c" not in _cache:
                _cache["c"] = client or make_client(model, **client_kwargs)
            return _cache["c"]

        pred = PREDICTOR_REGISTRY[predictor](get_client)
    return Pipeline(pred, calibrator)


# -- introspection helpers -------------------------------------------------
def pipeline_model(pipeline: Pipeline) -> str:
    """Best-effort model id behind a pipeline's predictor ("-" if none)."""
    client = getattr(pipeline.predictor, "client", None)
    return getattr(client, "model", "-")


# -- the headline comparison ----------------------------------------------
def compare(
    systems: Mapping[str, Pipeline],
    records: Sequence[SimBenchRecord],
    *,
    normalizers: dict[tuple[str, str], float] | None = None,
    max_workers: int = 8,
    with_counterfactual: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate several named pipelines on the same records; one row each.

    Args:
        systems: label -> :class:`Pipeline`. Labels become the table index.
        records: the evaluation set (e.g. a dev subsample). All systems see the
            identical records, so scores are directly comparable.
        normalizers: SimBench Eq. 2 per-dataset scalars from the FULL split
            (:func:`scrye.evaluate.build_normalizers`). Strongly recommended;
            if omitted, each evaluate() builds its own from ``records`` and the
            scores are no longer cross-comparable to other runs.
        with_counterfactual: also compute mean counterfactual-sensitivity
            alignment (conditioning *direction* correctness) per system.

    Returns:
        A topline DataFrame sorted by ``mean_score`` (desc), with columns
        ``system, model, n, mean_score, ci_low, ci_high, frac_below_uniform``
        (+ ``cf_alignment``). The per-system result frames are available at
        ``df.attrs["results"][label]`` for drill-downs and figures.
    """
    rows = []
    per_system: dict[str, pd.DataFrame] = {}
    for label, pipe in systems.items():
        res = evaluate(
            pipe, records, normalizers=normalizers,
            max_workers=max_workers, progress=progress,
        )
        per_system[label] = res
        summ = summarize(res)
        row = {
            "system": label,
            "model": pipeline_model(pipe),
            "n": summ["n"],
            "mean_score": summ["mean_score"],
            "ci_low": summ["ci_low"],
            "ci_high": summ["ci_high"],
            "frac_below_uniform": summ["frac_below_uniform"],
        }
        if with_counterfactual:
            align, _ = counterfactual_sensitivity(res)
            row["cf_alignment"] = align
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)
    out.attrs["results"] = per_system
    return out


def model_sweep(
    models: Sequence[str],
    records: Sequence[SimBenchRecord],
    *,
    predictor: str = "zero_shot",
    calibrator: Calibrator | None = None,
    normalizers: dict[tuple[str, str], float] | None = None,
    max_workers: int = 8,
    with_counterfactual: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Run one simulation system across several models; compare topline scores.

    A thin wrapper over :func:`compare` that builds one pipeline per model. This
    is the cross-model portability ablation: hold the method fixed, vary only
    the model. Labels are the model keys.
    """
    systems = {
        m: build_pipeline(model=m, predictor=predictor, calibrator=calibrator)
        for m in models
    }
    return compare(
        systems, records, normalizers=normalizers, max_workers=max_workers,
        with_counterfactual=with_counterfactual, progress=progress,
    )
