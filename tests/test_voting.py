"""Tests for VotingEnsemblePredictor — simulate individuals who each cast ONE
discrete vote; the distribution emerges from tallying votes.

This is the simulation-faithful alternative to MonteCarloPredictor (which
averages per-individual *distributions* and tends to over-disperse). No
network: a sequence-returning fake client stands in for the LLM.
"""

from scrye.data import SimBenchRecord
from scrye.llm import LLMResponse
from scrye.persona import sample_disposition, voter_messages
from scrye.predict import VotingEnsemblePredictor, _parse_vote


def _rec(options=("A", "B")):
    return SimBenchRecord(
        dataset_name="Choices13k",
        split="pop",
        input_template="You face two gambles.\nA) gamble a\nB) gamble b",
        options=tuple(options),
        human_answer={o: 1.0 / len(options) for o in options},
        group_prompt="",
        segment={},
        num_grouping_vars=0,
        group_size=100,
    )


class _SeqClient:
    model = "vendor/fake"

    def __init__(self, texts):
        self.texts = texts
        self.seeds: list = []

    def complete(self, messages, **overrides):
        self.seeds.append(overrides.get("seed"))
        return LLMResponse(text=self.texts[len(self.seeds) - 1], model=self.model, cached=False)


# --------------------------- vote parsing --------------------------- #

def test_parse_vote_reads_json_choice():
    assert _parse_vote('{"choice": "B"}', ["A", "B"]) == "B"


def test_parse_vote_reads_bare_label():
    assert _parse_vote("I would pick A.", ["A", "B"]) == "A"


def test_parse_vote_ambiguous_returns_none():
    # mentions both labels and no explicit choice key -> unparseable
    assert _parse_vote("could be A or B honestly", ["A", "B"]) is None


def test_parse_vote_unknown_returns_none():
    assert _parse_vote("Z", ["A", "B"]) is None


# --------------------------- tallying ------------------------------- #

def test_votes_are_tallied_into_a_distribution():
    c = _SeqClient(['{"choice": "A"}', '{"choice": "A"}',
                    '{"choice": "A"}', '{"choice": "B"}'])
    p = VotingEnsemblePredictor(c, n_individuals=4, alpha=0.0)
    out = p.predict(_rec())
    assert abs(out["A"] - 0.75) < 1e-9
    assert abs(out["B"] - 0.25) < 1e-9


def test_alpha_smoothing_gives_zero_vote_options_mass():
    # all 3 vote A; with add-1 smoothing B should still get nonzero mass
    c = _SeqClient(['{"choice": "A"}'] * 3)
    p = VotingEnsemblePredictor(c, n_individuals=3, alpha=1.0)
    out = p.predict(_rec())
    # counts A=3,B=0 -> smoothed (3+1)/(3+2)=0.8 , (0+1)/5=0.2
    assert abs(out["A"] - 0.8) < 1e-9
    assert abs(out["B"] - 0.2) < 1e-9


def test_unparseable_votes_are_skipped():
    c = _SeqClient(['{"choice": "A"}', 'no idea'])
    p = VotingEnsemblePredictor(c, n_individuals=2, alpha=0.0)
    out = p.predict(_rec())
    assert out["A"] == 1.0 and out["B"] == 0.0
    assert p.n_parse_failures == 1


def test_all_fail_falls_back_to_uniform():
    c = _SeqClient(['nope', 'nope'])
    p = VotingEnsemblePredictor(c, n_individuals=2)
    out = p.predict(_rec())
    assert abs(out["A"] - 0.5) < 1e-9 and abs(out["B"] - 0.5) < 1e-9


def test_distinct_seeds_per_draw():
    c = _SeqClient(['{"choice": "A"}'] * 3)
    VotingEnsemblePredictor(c, n_individuals=3, base_seed=5).predict(_rec())
    assert c.seeds == [5, 6, 7]


# --------------------------- dispositions --------------------------- #

def test_sample_disposition_deterministic_and_varied():
    r = _rec()
    assert sample_disposition(r, 0) == sample_disposition(r, 0)
    seen = {sample_disposition(r, s) for s in range(12)}
    assert len(seen) > 1  # variation across seeds


def test_voter_messages_request_single_choice():
    r = _rec()
    msgs = voter_messages(r, "is a bold risk-taker")
    user = msgs[-1]["content"]
    assert "is a bold risk-taker" in user
    # asks for one choice, not a distribution
    assert "choice" in user.lower()
