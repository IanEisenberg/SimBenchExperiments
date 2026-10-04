"""`ask` — predict a population's answer distribution for a freeform question.

The deployable core behind the ``simbench-ask`` CLI. It runs the project's best
method for a *cold* question: the ``calibrated_commitment`` prompt strategy (the
Stage-10 winner) on ``gemini-3.1-flash-lite`` (the dominant model lever). See
``docs/experiments/stage-17-final-test.md``.

A typed-in question has no SimBench ``dataset_name`` and no ground truth, so the
dataset-level :class:`~simbench_exp.calibrate.AbstainCalibrator` (and the task-kind
router) cannot apply — the method reduces cleanly to one calibrated_commitment
call, parsed into a distribution over the options. There is no abstention.

Free-text ``audience`` conditioning is woven into the prompt's population phrase
via :class:`~simbench_exp.persona.CalibratedCommitmentStrategy`'s audience override; with
no audience the prompt is byte-identical to the validated strategy.
"""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .data import SimBenchRecord
from .experiment import make_client
from .llm import LLMClient
from .persona import CalibratedCommitmentStrategy
from .predict import ZeroShotPredictor

#: 4-point Likert option texts used when the caller supplies no options.
DEFAULT_LIKERT: tuple[str, ...] = (
    "Agree",
    "Somewhat agree",
    "Somewhat disagree",
    "Disagree",
)

#: The best deployable model for a freeform question (Stage 17).
DEFAULT_ASK_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class AskResult:
    """The prediction for one asked question.

    Attributes:
        question: the question text as asked.
        audience: the free-text population conditioning, or ``None`` for the
            unconditioned general population.
        model: the resolved model id that produced the prediction.
        options: ``{label: option_text}`` (labels ``A, B, C, …``).
        distribution: ``{label: probability}``, normalized to sum 1.
    """

    question: str
    audience: str | None
    model: str
    options: dict[str, str]
    distribution: dict[str, float]


def _labels(n: int) -> list[str]:
    """Letter labels A, B, C, … for ``n`` options (max 26)."""
    if n > len(string.ascii_uppercase):
        raise ValueError(f"At most {len(string.ascii_uppercase)} options supported, got {n}.")
    return list(string.ascii_uppercase[:n])


def _option_texts(options: Sequence[str] | None) -> list[str]:
    texts = [str(o).strip() for o in options] if options else list(DEFAULT_LIKERT)
    texts = [t for t in texts if t]
    if len(texts) < 2:
        raise ValueError("Need at least two answer options.")
    return texts


def build_ask_record(question: str, options: Sequence[str] | None = None) -> SimBenchRecord:
    """Construct a :class:`SimBenchRecord` for a freeform question.

    Faithful to the dataset shape: letter labels with an ``Options:`` block in the
    ``input_template``. ``human_answer`` is a uniform placeholder — predictors
    never read it (only scoring does), and a cold question has no ground truth.
    """
    texts = _option_texts(options)
    labels = _labels(len(texts))
    opts_block = "\n".join(f"({label}): {text}" for label, text in zip(labels, texts))
    input_template = f"{question.strip()}\n\nOptions:\n{opts_block}"
    return SimBenchRecord(
        dataset_name="__ask__",
        split="pop",
        input_template=input_template,
        options=tuple(labels),
        human_answer={label: 1.0 / len(labels) for label in labels},
        group_prompt="",
        answer_options=dict(zip(labels, texts)),
        segment={},
        num_grouping_vars=0,
    )


def _normalize(dist: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    vals = {label: max(0.0, float(dist.get(label, 0.0))) for label in labels}
    total = sum(vals.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: value / total for label, value in vals.items()}


def ask(
    question: str,
    options: Sequence[str] | None = None,
    audience: str | None = None,
    *,
    model: str = DEFAULT_ASK_MODEL,
    client: LLMClient | None = None,
) -> AskResult:
    """Predict how a population answers ``question`` over its answer options.

    Args:
        question: the question text.
        options: answer-option texts; ``None``/empty ⇒ :data:`DEFAULT_LIKERT`.
        audience: free-text population to condition on (e.g. "women in Finland");
            ``None`` ⇒ unconditioned general population.
        model: model key or id (ignored if ``client`` is supplied).
        client: reuse an existing :class:`LLMClient` (tests inject a fake one).

    Returns:
        An :class:`AskResult` with the option texts and the normalized
        distribution over the option labels.
    """
    record = build_ask_record(question, options)
    labels = list(record.options)
    client = client or make_client(model)
    predictor = ZeroShotPredictor(
        client,
        name="calibrated_commitment",
        strategy=CalibratedCommitmentStrategy(audience=audience or None),
    )
    raw = predictor.predict(record)
    return AskResult(
        question=question,
        audience=audience or None,
        model=getattr(client, "model", model),
        options=dict(record.answer_options),
        distribution=_normalize(raw, labels),
    )
