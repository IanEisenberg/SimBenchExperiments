"""Tests for task-kind classification + RoutingPredictor (Stage 15)."""

from simbench_exp.data import SimBenchRecord
from simbench_exp.llm import LLMResponse
from simbench_exp.predict import RoutingPredictor
from simbench_exp.taskkind import (
    KIND_ROUTES,
    KINDS,
    LLMTaskClassifier,
    heuristic_task_kind,
)


def _rec(stem, ds="X", options=("A", "B"), group=""):
    return SimBenchRecord(
        dataset_name=ds, split="pop", input_template=stem, options=tuple(options),
        human_answer={o: 1.0 / len(options) for o in options}, group_prompt=group,
        segment={}, num_grouping_vars=0, group_size=100)


# ----------------------------- heuristic ----------------------------- #

def test_heuristic_detects_risky_choice():
    r = _rec("There are two gambling machines. Machine A: $10 with 1.0% chance ...")
    assert heuristic_task_kind(r) == "risky_choice"


def test_heuristic_detects_moral_dilemma():
    r = _rec("A self-driving car with sudden brake failure must swerve or stay ...")
    assert heuristic_task_kind(r) == "moral_dilemma"


def test_heuristic_detects_personality_scale():
    r = _rec("Indicate your level of agreement with the following statement: ...",
             group="You are a user of openpsychometrics.org.")
    assert heuristic_task_kind(r) == "personality_scale"


def test_heuristic_detects_opinion_survey_by_ordinal_options():
    r = _rec("Has the government done a good or bad job?",
             options=("Very good", "Somewhat good", "Somewhat bad", "Very bad"))
    assert heuristic_task_kind(r) == "opinion_survey"


def test_heuristic_default_is_other():
    assert heuristic_task_kind(_rec("Rate this joke from 1 to 5.")) == "other"


# --------------------------- LLM classifier -------------------------- #

class _StubClient:
    """Returns a canned label; counts calls to prove per-stem caching."""
    model = "vendor/fake"

    def __init__(self, label):
        self.label = label
        self.calls = 0

    def complete(self, messages, **overrides):
        self.calls += 1
        return LLMResponse(text=self.label, model=self.model, cached=False)


def test_llm_classifier_parses_label_and_memoizes_per_stem():
    c = _StubClient("risky_choice")
    clf = LLMTaskClassifier(c)
    r = _rec("two gambling machines ...")
    assert clf(r) == "risky_choice"
    assert clf(r) == "risky_choice"          # second call same stem
    assert c.calls == 1                       # memoized, not re-queried


def test_llm_classifier_falls_back_when_offtaxonomy():
    c = _StubClient("banana")                 # not a valid kind
    clf = LLMTaskClassifier(c)
    r = _rec("There are two gambling machines. Machine A: $10 with 1.0% chance ...")
    assert clf(r) == "risky_choice"           # heuristic fallback kicks in


def test_routes_cover_every_kind():
    assert set(KIND_ROUTES) == set(KINDS)


# --------------------------- RoutingPredictor ------------------------ #

class _TagPredictor:
    def __init__(self, tag):
        self.tag = tag

    def predict(self, record):
        return {"tag": self.tag}


def test_router_dispatches_by_kind():
    routes = {"risky_choice": _TagPredictor("voting"),
              "opinion_survey": _TagPredictor("context")}
    router = RoutingPredictor(
        classifier=lambda r: "risky_choice", routes=routes,
        default=_TagPredictor("base"))
    assert router.predict(_rec("q"))["tag"] == "voting"
    assert router.kind_counts["risky_choice"] == 1


def test_router_falls_back_to_default_for_unmapped_kind():
    router = RoutingPredictor(
        classifier=lambda r: "other", routes={"risky_choice": _TagPredictor("voting")},
        default=_TagPredictor("base"))
    assert router.predict(_rec("q"))["tag"] == "base"
