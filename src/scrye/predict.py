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
from .distributions import SegmentWeights
from .llm import LLMClient
from .persona import DEFAULT_STRATEGY, PromptStrategy, get_strategy


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


def _parse_vote(text: str, options) -> str | None:
    """Parse a single discrete choice from a voter response.

    Prefers an explicit ``{"choice": "<label>"}`` JSON object; otherwise falls
    back to a bare label mention, but only when *exactly one* option label
    appears as a standalone token (an ambiguous "A or B" returns None so the
    draw is skipped rather than guessed).
    """
    opts = [str(o) for o in options]
    obj = _extract_json_object(text)
    if obj is not None and "choice" in obj:
        choice = str(obj["choice"]).strip()
        return choice if choice in opts else None
    if not text:
        return None
    hits = [o for o in opts if re.search(rf"\b{re.escape(o)}\b", text)]
    return hits[0] if len(hits) == 1 else None


class VotingEnsemblePredictor(Predictor):
    """Simulate a population by polling individuals who each cast ONE vote.

    Draw ``n_individuals`` synthetic people (via
    :func:`~scrye.persona.sample_disposition`, varied along task-agnostic
    dispositional axes), ask each to commit to a single option
    (:func:`~scrye.persona.voter_messages`), and tally the votes into a
    distribution. Unlike :class:`MonteCarloPredictor`, no per-individual
    distribution is averaged, so the group's spread is the genuine vote split:
    sharp when sampled people agree, diffuse when they disagree.

    ``alpha`` adds Laplace smoothing to the tally (default 0.5), so an option
    that drew zero votes still carries a little mass — a finite sample of
    voters should not assert probability exactly 0. Unparseable votes are
    skipped (counted in ``n_parse_failures``); if every vote fails, falls back
    to uniform. Variation is injected by the seeded sampler, so results are
    reproducible at ``temperature=0`` and cache cleanly.
    """

    def __init__(
        self,
        client: LLMClient,
        n_individuals: int = 24,
        temperature: float = 0.0,
        base_seed: int = 0,
        alpha: float = 0.5,
        name: str = "voting_ensemble",
        sampler=None,
    ) -> None:
        from .persona import sample_disposition

        self.client = client
        self.n_individuals = n_individuals
        self.temperature = temperature
        self.base_seed = base_seed
        self.alpha = alpha
        self.name = name
        self.sampler = sampler or sample_disposition
        self.n_parse_failures = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        from .persona import voter_messages

        tally = {opt: 0 for opt in record.options}
        valid = 0
        for i in range(self.n_individuals):
            disposition = self.sampler(record, self.base_seed * 10_000 + i)
            text = self.client.complete(
                voter_messages(record, disposition),
                temperature=self.temperature,
                seed=self.base_seed + i,
            ).text
            choice = _parse_vote(text, record.options)
            if choice is None:
                self.n_parse_failures += 1
                continue
            tally[choice] += 1
            valid += 1
        if valid == 0:
            return _uniform(record)
        smoothed = {opt: tally[opt] + self.alpha for opt in record.options}
        grand = sum(smoothed.values())
        return {opt: value / grand for opt, value in smoothed.items()}


class ZeroShotPredictor(Predictor):
    """Verbalized-distribution zero-shot predictor (the project default).

    Prompts the model to state a probability for each answer option and parses
    the JSON. Robust by construction: unparseable or empty responses fall back
    to the uniform distribution (and increment `n_parse_failures`), so a single
    bad generation never crashes a batch. The SimBench paper establishes that
    verbalized distributions beat first-token logprobs for instruct models,
    which is why this is the baseline elicitation.

    The conditioning prompt is supplied by a swappable
    :class:`~scrye.persona.PromptStrategy` (default ``simbench_faithful``, which
    reproduces the original baseline prompt byte-for-byte). Pass a different
    strategy name to ablate demographic-conditioning styles.
    """

    def __init__(
        self,
        client: LLMClient,
        name: str = "zero_shot",
        strategy: str | PromptStrategy = DEFAULT_STRATEGY,
    ) -> None:
        self.client = client
        self.name = name
        self.strategy = (
            strategy if isinstance(strategy, PromptStrategy) else get_strategy(strategy)
        )
        self.n_parse_failures = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        text = self.client.complete(self.strategy.build_messages(record)).text
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


class MonteCarloPredictor(Predictor):
    """Simulate a population by sampling individuals and averaging their answers.

    Instead of one call describing the whole group, draw ``n_individuals`` from
    the segment — each a *synthetic within-group person* (a worldview lean plus
    the demographic attributes the segment leaves open, via
    :func:`~scrye.persona.sample_persona`) — and average their predicted
    distributions. The group's spread emerges from cross-draw variation rather
    than from a single call self-reporting its diversity. This tests whether
    *external* aggregation of sampled individuals beats *internal* group framing.

    The variation is injected by a **seeded RNG**, not the model's sampler: the
    model is near-deterministic across API seeds/temperature, so identical
    prompts give identical draws. Each draw's persona descriptor differs, so the
    aggregate is varied yet reproducible (``temperature=0`` + cache).
    Each individual's distribution is normalized before averaging so
    every draw contributes equally. Unparseable draws are skipped (counted in
    ``n_parse_failures``); if every draw fails, falls back to uniform.
    """

    def __init__(
        self,
        client: LLMClient,
        n_individuals: int = 20,
        temperature: float = 0.0,
        base_seed: int = 0,
        name: str = "monte_carlo",
        sampler=None,
    ) -> None:
        from .persona import sample_persona

        self.client = client
        self.n_individuals = n_individuals
        self.temperature = temperature
        self.base_seed = base_seed
        self.name = name
        self.sampler = sampler or sample_persona
        self.n_parse_failures = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        from .persona import individual_messages

        agg = {opt: 0.0 for opt in record.options}
        valid = 0
        for i in range(self.n_individuals):
            persona = self.sampler(record, self.base_seed * 10_000 + i)
            messages = individual_messages(record, persona)
            text = self.client.complete(
                messages, temperature=self.temperature, seed=self.base_seed + i
            ).text
            parsed = _extract_json_object(text)
            if not parsed:
                self.n_parse_failures += 1
                continue
            draw: dict[str, float] = {}
            for opt in record.options:
                try:
                    draw[opt] = max(0.0, float(parsed.get(opt, 0.0)))
                except (TypeError, ValueError):
                    draw[opt] = 0.0
            total = sum(draw.values())
            if total <= 0:
                self.n_parse_failures += 1
                continue
            for opt in record.options:  # normalize each individual, then average
                agg[opt] += draw[opt] / total
            valid += 1
        if valid == 0:
            return _uniform(record)
        grand = sum(agg.values())
        return {opt: value / grand for opt, value in agg.items()}


class PostStratificationPredictor(Predictor):
    """Combine subgroup predictions into a coarser estimate by reweighting.

    Instead of predicting a coarse target (e.g. a country marginal) directly,
    decompose it into finer demographic cells via :class:`SegmentWeights`,
    predict each cell with a `base` predictor, and average the cell predictions
    weighted by each cell's population share (``group_size``).

    Motivation: SimBench shows direct demographic conditioning often *hurts*, so
    decompose-then-recombine is a candidate that exploits real population
    structure instead. When a record has no available decomposition (no child
    cells in the data), it falls back to the base predictor on the record
    itself, so this drops into any pipeline unchanged.

    `over` optionally pins the decomposition variable (e.g. ``"gender"``);
    otherwise the variable with the most respondent coverage is chosen.
    `min_children` is the smallest decomposition worth taking. The counters
    `n_decomposed` / `n_fallback` record how often each path was used.
    """

    def __init__(
        self,
        base: Predictor,
        weights: SegmentWeights,
        over: str | None = None,
        name: str = "post_strat",
        min_children: int = 2,
    ) -> None:
        self.base = base
        self.weights = weights
        self.over = over
        self.name = name
        self.min_children = min_children
        self.n_decomposed = 0
        self.n_fallback = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        children = self.weights.children(record, over=self.over)
        if len(children) < self.min_children:
            self.n_fallback += 1
            return self.base.predict(record)

        self.n_decomposed += 1
        agg = {opt: 0.0 for opt in record.options}
        for weight, child in children:
            pred = self.base.predict(child)
            for opt in record.options:
                agg[opt] += weight * float(pred.get(opt, 0.0))

        total = sum(agg.values())
        if total <= 0:
            return _uniform(record)
        return {opt: value / total for opt, value in agg.items()}
