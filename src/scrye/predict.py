"""Predictors — the *simulation* component of the pipeline.

A predictor maps a :class:`~scrye.data.SimBenchRecord` to a predicted answer
distribution `{option_label: probability}`. This is the swappable seam for the
modeling approach: the zero-shot verbalized baseline lives here, and future
approaches (persona Monte Carlo, retrieval-anchored, ...) implement the same
`Predictor` interface and drop into the pipeline unchanged.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .data import SimBenchRecord
from .llm import LLMClient


class Predictor(ABC):
    """Maps a record to a predicted distribution over its options.

    Implementations set a short `name` (used in result tables and figure
    labels) and implement :meth:`predict`. The returned dict must cover exactly
    `record.options`; the pipeline renormalizes defensively, but a predictor
    should aim to return a proper distribution.
    """

    name: str = "predictor"

    @abstractmethod
    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        ...


def _uniform(record: SimBenchRecord) -> dict[str, float]:
    k = record.num_options or 1
    return {o: 1.0 / k for o in record.options}


def _extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a model response, tolerantly."""
    if not text:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        out = json.loads(candidate)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


class ZeroShotPredictor(Predictor):
    """Verbalized-distribution zero-shot baseline (the project default).

    Prompts the model to state a probability for each answer option and parses
    the JSON. Robust by construction: unparseable or empty responses fall back
    to the uniform distribution (and increment `n_parse_failures`), so a single
    bad generation never crashes a batch. The SimBench paper establishes that
    verbalized distributions beat first-token logprobs for instruct models,
    which is why this is the baseline elicitation.
    """

    def __init__(self, client: LLMClient, name: str = "zero_shot") -> None:
        self.client = client
        self.name = name
        self.n_parse_failures = 0

    def _build_prompt(self, record: SimBenchRecord) -> str:
        opts = ", ".join(record.options)
        persona = record.group_prompt.strip()
        persona_block = f"{persona}\n\n" if persona else ""
        return (
            f"{persona_block}{record.input_template.strip()}\n\n"
            "Estimate how a large, representative sample of such people would "
            "answer. Give the probability (between 0 and 1) that a randomly "
            "sampled person chooses each option.\n"
            f"Respond with ONLY a JSON object mapping each option label "
            f"[{opts}] to its probability. The probabilities must sum to 1. "
            "No other text."
        )

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        text = self.client.prompt(self._build_prompt(record))
        parsed = _extract_json_object(text)
        if not parsed:
            self.n_parse_failures += 1
            return _uniform(record)
        out: dict[str, float] = {}
        for opt in record.options:
            try:
                out[opt] = max(0.0, float(parsed.get(opt, 0.0)))
            except (TypeError, ValueError):
                out[opt] = 0.0
        if sum(out.values()) <= 0:
            self.n_parse_failures += 1
            return _uniform(record)
        return out


class UniformPredictor(Predictor):
    """The naive baseline: always predict uniform. Scores ~0 by construction.

    Useful as a sanity floor in comparisons and ablations.
    """

    name = "uniform"

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        return _uniform(record)
