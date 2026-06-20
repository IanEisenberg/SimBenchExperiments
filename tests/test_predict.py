"""Tests for MonteCarloPredictor — sampling individuals and averaging.

No network: a sequence-returning fake client stands in for the LLM.
"""

from scrye.data import SimBenchRecord
from scrye.llm import LLMResponse
from scrye.predict import MonteCarloPredictor


def _rec(options=("A", "B")):
    return SimBenchRecord(
        dataset_name="ESS",
        split="grouped",
        input_template="Q?\nA) a\nB) b",
        options=tuple(options),
        human_answer={o: 1.0 / len(options) for o in options},
        group_prompt="You are from Finland.",
        segment={"cntry": "Finland"},
        num_grouping_vars=1,
        group_size=100,
    )


class _SeqClient:
    """Returns canned texts in order; records the per-call overrides it saw."""

    model = "vendor/fake"

    def __init__(self, texts):
        self.texts = texts
        self.seeds: list = []
        self.temps: list = []

    def complete(self, messages, **overrides):
        self.seeds.append(overrides.get("seed"))
        self.temps.append(overrides.get("temperature"))
        text = self.texts[len(self.seeds) - 1]
        return LLMResponse(text=text, model=self.model, cached=False)


def test_monte_carlo_averages_normalized_individual_distributions():
    # two decided-but-opposite individuals -> 50/50 group estimate
    c = _SeqClient(['{"A": 1, "B": 0}', '{"A": 0, "B": 1}'])
    p = MonteCarloPredictor(c, n_individuals=2, base_seed=0)
    out = p.predict(_rec())
    assert abs(out["A"] - 0.5) < 1e-9
    assert abs(out["B"] - 0.5) < 1e-9


def test_monte_carlo_normalizes_each_individual_before_averaging():
    # individual 1 is unnormalized (sums to 4) but should still weight equally
    c = _SeqClient(['{"A": 4, "B": 0}', '{"A": 0, "B": 1}'])
    p = MonteCarloPredictor(c, n_individuals=2)
    out = p.predict(_rec())
    assert abs(out["A"] - 0.5) < 1e-9  # not 4/5: each draw normalized first


def test_monte_carlo_uses_distinct_seeds_and_temperature():
    c = _SeqClient(['{"A": 1, "B": 0}'] * 3)
    p = MonteCarloPredictor(c, n_individuals=3, base_seed=10, temperature=0.7)
    p.predict(_rec())
    assert c.seeds == [10, 11, 12]
    assert all(t == 0.7 for t in c.temps)


def test_monte_carlo_skips_unparseable_draws():
    # one good, one garbage -> result equals the single good draw
    c = _SeqClient(['{"A": 1, "B": 0}', 'sorry, no json'])
    p = MonteCarloPredictor(c, n_individuals=2)
    out = p.predict(_rec())
    assert out["A"] == 1.0 and out["B"] == 0.0
    assert p.n_parse_failures == 1


def test_monte_carlo_falls_back_to_uniform_when_all_fail():
    c = _SeqClient(['nope', 'still nope'])
    p = MonteCarloPredictor(c, n_individuals=2)
    out = p.predict(_rec())
    assert abs(out["A"] - 0.5) < 1e-9 and abs(out["B"] - 0.5) < 1e-9
