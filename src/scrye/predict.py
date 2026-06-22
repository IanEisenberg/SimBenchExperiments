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


class GroundedVotingPredictor(Predictor):
    """Simulate a population by polling a grounded Nemotron persona electorate.

    The Stage-18 system. Like :class:`VotingEnsemblePredictor`, draws K individuals
    and tallies one discrete vote each — but the individuals are **real
    census-grounded personas** (their narratives) sampled from
    :class:`~scrye.nemotron.PersonaBank`, not synthetic disposition clauses. For an
    unconditioned (pop) record the panel is a representative sample of US adults; for
    a segment it is filtered to that segment on the axes Nemotron carries.

    When the segment is unmatchable (pinned on race/income/religion/politics, which
    Nemotron lacks) or the matched pool is too small, the record is routed to
    ``fallback`` (a prompt-based predictor) — mirroring
    :class:`PostStratificationPredictor`'s no-decomposition fallback. ``n_grounded``
    / ``n_fallback`` record the realized split for the coverage log.

    The K persona calls run concurrently (``max_workers``); the panel is fixed and
    seeded so every record sharing a segment polls the same people, which keeps the
    on-disk LLM cache warm and the electorate auditable. ``alpha`` Laplace-smooths
    the tally so an option with zero votes still carries a little mass.
    """

    def __init__(
        self,
        client: LLMClient,
        bank=None,
        k: int = 50,
        temperature: float = 0.0,
        base_seed: int = 0,
        alpha: float = 0.5,
        min_pool: int = 10,
        fallback: "Predictor | None" = None,
        max_workers: int = 16,
        name: str = "grounded_voting",
    ) -> None:
        if bank is None:
            from .nemotron import default_bank

            bank = default_bank()
        self.client = client
        self.bank = bank
        self.k = k
        self.temperature = temperature
        self.base_seed = base_seed
        self.alpha = alpha
        self.min_pool = min_pool
        self.fallback = fallback
        self.max_workers = max_workers
        self.name = name
        self.n_parse_failures = 0
        self.n_grounded = 0
        self.n_fallback = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        from concurrent.futures import ThreadPoolExecutor

        from .persona import persona_voter_messages

        panel = self.bank.panel_for(
            record, self.k, base_seed=self.base_seed, min_pool=self.min_pool
        )
        if not panel:
            self.n_fallback += 1
            return self.fallback.predict(record) if self.fallback else _uniform(record)

        self.n_grounded += 1

        def _vote(i_text):
            i, text = i_text
            resp = self.client.complete(
                persona_voter_messages(record, text),
                temperature=self.temperature,
                seed=self.base_seed + i,
            ).text
            return _parse_vote(resp, record.options)

        workers = max(1, min(self.max_workers, len(panel)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            choices = list(ex.map(_vote, enumerate(panel)))

        tally = {opt: 0 for opt in record.options}
        valid = 0
        for choice in choices:
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


class EnrichedGroundedPredictor(Predictor):
    """Weighted average over a worldview-enriched, ideology-calibrated panel.

    The Stage-18 final system. Polls a *fixed* panel of personas each carrying an
    inferred worldview (see :mod:`scrye.worldview`), and averages their per-persona
    answer distributions weighted by ``weights`` — post-stratification weights that
    align the panel's ideology mix to OpinionQA's real marginal. With equal weights
    it is the un-calibrated ablation. Pop-focused: it uses the fixed panel for every
    record (the unconditioned US electorate) and ignores ``record.segment``.
    """

    def __init__(
        self,
        client: LLMClient,
        panel_texts: list[str],
        weights: list[float] | None = None,
        temperature: float = 0.0,
        base_seed: int = 0,
        max_workers: int = 5,
        name: str = "enriched_grounded",
    ) -> None:
        self.client = client
        self.panel = list(panel_texts)
        n = len(self.panel)
        self.weights = list(weights) if weights is not None else [1.0 / n] * n
        self.temperature = temperature
        self.base_seed = base_seed
        self.max_workers = max_workers
        self.name = name
        self.n_parse_failures = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        from concurrent.futures import ThreadPoolExecutor

        from .persona import persona_dist_messages

        def _draw(i_text):
            i, text = i_text
            resp = self.client.complete(
                persona_dist_messages(record, text),
                temperature=self.temperature, seed=self.base_seed + i,
            ).text
            parsed = _extract_json_object(resp)
            if not parsed:
                return None
            d = {o: max(0.0, float(parsed.get(o, 0.0) or 0.0)) for o in record.options}
            t = sum(d.values())
            return {o: d[o] / t for o in record.options} if t > 0 else None

        workers = max(1, min(self.max_workers, len(self.panel)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            draws = list(ex.map(_draw, enumerate(self.panel)))

        agg = {o: 0.0 for o in record.options}
        wsum = 0.0
        for w, draw in zip(self.weights, draws):
            if draw is None:
                self.n_parse_failures += 1
                continue
            for o in record.options:
                agg[o] += w * draw[o]
            wsum += w
        if wsum <= 0:
            return _uniform(record)
        grand = sum(agg.values())
        return {o: v / grand for o, v in agg.items()}


class GroundedAveragingPredictor(Predictor):
    """Grounded persona electorate that AVERAGES per-persona distributions.

    The Stage-18 Round-2 system, and the principled counterpart to
    :class:`GroundedVotingPredictor`. Each persona is asked for *their own*
    probability over the options (:func:`~scrye.persona.persona_dist_messages`),
    and the group prediction is the mean of those distributions across the panel.

    Why average rather than tally discrete votes: the population's choice-fraction
    *is* the average over people of each person's choice-probability, so averaging
    is the unbiased estimator — sampling/argmax-then-tally only adds variance or
    discards within-person uncertainty (which over-concentrates the group, as
    Round 1 showed). The risk it trades into is over-dispersion if the model hedges
    each persona toward uniform (the Stage-03 failure); the calibrated
    ``PERSONA_DIST_SYSTEM`` prompt and grounded narratives are what guard against
    that. Shares the bank, fixed panel, fallback, and concurrency of the voting
    predictor; ``n_grounded`` / ``n_fallback`` log the coverage split.
    """

    def __init__(
        self,
        client: LLMClient,
        bank=None,
        k: int = 50,
        temperature: float = 0.0,
        base_seed: int = 0,
        min_pool: int = 10,
        fallback: "Predictor | None" = None,
        max_workers: int = 16,
        name: str = "grounded_averaging",
    ) -> None:
        if bank is None:
            from .nemotron import default_bank

            bank = default_bank()
        self.client = client
        self.bank = bank
        self.k = k
        self.temperature = temperature
        self.base_seed = base_seed
        self.min_pool = min_pool
        self.fallback = fallback
        self.max_workers = max_workers
        self.name = name
        self.n_parse_failures = 0
        self.n_grounded = 0
        self.n_fallback = 0

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        from concurrent.futures import ThreadPoolExecutor

        from .persona import persona_dist_messages

        panel = self.bank.panel_for(
            record, self.k, base_seed=self.base_seed, min_pool=self.min_pool
        )
        if not panel:
            self.n_fallback += 1
            return self.fallback.predict(record) if self.fallback else _uniform(record)

        self.n_grounded += 1

        def _dist(i_text):
            i, text = i_text
            resp = self.client.complete(
                persona_dist_messages(record, text),
                temperature=self.temperature,
                seed=self.base_seed + i,
            ).text
            parsed = _extract_json_object(resp)
            if not parsed:
                return None
            draw = {}
            for opt in record.options:
                try:
                    draw[opt] = max(0.0, float(parsed.get(opt, 0.0)))
                except (TypeError, ValueError):
                    draw[opt] = 0.0
            total = sum(draw.values())
            if total <= 0:
                return None
            return {opt: draw[opt] / total for opt in record.options}  # per-person norm

        workers = max(1, min(self.max_workers, len(panel)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            draws = list(ex.map(_dist, enumerate(panel)))

        agg = {opt: 0.0 for opt in record.options}
        valid = 0
        for draw in draws:
            if draw is None:
                self.n_parse_failures += 1
                continue
            for opt in record.options:
                agg[opt] += draw[opt]
            valid += 1
        if valid == 0:
            return _uniform(record)
        grand = sum(agg.values())
        return {opt: value / grand for opt, value in agg.items()}


class EnsemblePredictor(Predictor):
    """Blend several predictors' distributions by a fixed weighted average.

    ``predictors`` are run on the same record and their (renormalized) outputs are
    mixed by ``weights`` (defaults to equal; normalized to sum 1). Useful for
    combining a bottom-up persona system (good spread / conditioning direction)
    with a top-down single-call (good location). When the sub-predictors are
    already cached for a record set, the blend is scored at zero marginal LLM cost,
    and the mixing weight can be swept on dev (then val-gated).
    """

    def __init__(self, predictors, weights=None, name: str = "ensemble") -> None:
        if not predictors:
            raise ValueError("EnsemblePredictor needs at least one predictor.")
        self.predictors = list(predictors)
        if weights is None:
            weights = [1.0] * len(self.predictors)
        if len(weights) != len(self.predictors):
            raise ValueError("weights must match predictors in length.")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        self.weights = [w / total for w in weights]
        self.name = name

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        agg = {opt: 0.0 for opt in record.options}
        for w, pred in zip(self.weights, self.predictors):
            dist = pred.predict(record)
            sub_total = sum(max(0.0, float(dist.get(opt, 0.0))) for opt in record.options)
            if sub_total <= 0:
                continue
            for opt in record.options:
                agg[opt] += w * max(0.0, float(dist.get(opt, 0.0))) / sub_total
        grand = sum(agg.values())
        if grand <= 0:
            return _uniform(record)
        return {opt: value / grand for opt, value in agg.items()}


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


class RoutingPredictor(Predictor):
    """Dispatch each record to a sub-predictor chosen by an upfront task classifier.

    The system from Stage 15: a `classifier` labels the record's *task kind*
    (see :mod:`scrye.taskkind`), and `routes` maps each kind to the Predictor
    that works best for that kind (task-context, voting, uniform-abstain, or the
    base prompt). Records whose kind is absent from `routes` fall back to
    `default`. ``kind_counts`` records the realized routing for inspection.

    Because the route is keyed on a property of the *question* rather than the
    dataset, a previously unseen dataset of a known kind is handled correctly
    without any per-dataset fitting.
    """

    def __init__(self, classifier, routes: dict[str, Predictor],
                 default: Predictor, name: str = "router") -> None:
        from collections import Counter

        self.classifier = classifier
        self.routes = routes
        self.default = default
        self.name = name
        self.kind_counts: Counter = Counter()

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        kind = self.classifier(record)
        self.kind_counts[kind] += 1
        return self.routes.get(kind, self.default).predict(record)


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
