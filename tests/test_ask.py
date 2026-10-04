"""Tests for the `ask` core — predict a population's answer distribution for a
freeform multiple-choice question.

No network: a capturing fake client stands in for the LLM.
"""

import json

import pytest

from simbench_exp.ask import (
    DEFAULT_LIKERT,
    AskResult,
    ask,
    build_ask_record,
)
from simbench_exp.llm import LLMResponse
from simbench_exp.persona import CalibratedCommitmentStrategy


class _CaptureClient:
    """Returns one canned text; records the messages it last saw."""

    model = "google/gemini-3.1-flash-lite"

    def __init__(self, text='{"A": 0.5, "B": 0.5}'):
        self.text = text
        self.messages = None

    def complete(self, messages, **overrides):
        self.messages = messages
        return LLMResponse(text=self.text, model=self.model, cached=False)


# -- option handling -------------------------------------------------------


def test_defaults_to_four_point_likert_when_options_blank():
    res = ask("Will remote work keep growing?", options=None, client=_CaptureClient())
    assert list(res.options.values()) == list(DEFAULT_LIKERT)
    assert list(res.options) == ["A", "B", "C", "D"]


def test_assigns_letter_labels_to_provided_options():
    res = ask("Q?", options=["Yes", "No", "Maybe"], client=_CaptureClient('{"A":1,"B":0,"C":0}'))
    assert res.options == {"A": "Yes", "B": "No", "C": "Maybe"}


def test_options_block_rendered_into_input_template():
    rec = build_ask_record("Do you agree?", ["Yes", "No"])
    assert "Options:\n(A): Yes\n(B): No" in rec.input_template
    assert rec.input_template.startswith("Do you agree?")
    assert rec.options == ("A", "B")


def test_requires_at_least_two_options():
    with pytest.raises(ValueError):
        ask("Q?", options=["only one"], client=_CaptureClient())


# -- distribution ----------------------------------------------------------


def test_distribution_is_normalized_to_sum_one():
    # client returns an un-normalized pair summing to 4
    res = ask("Q?", options=["a", "b"], client=_CaptureClient('{"A": 2, "B": 2}'))
    assert abs(sum(res.distribution.values()) - 1.0) < 1e-9
    assert abs(res.distribution["A"] - 0.5) < 1e-9


def test_distribution_covers_exactly_the_option_labels():
    res = ask("Q?", options=["a", "b", "c"], client=_CaptureClient('{"A":0.6,"B":0.3,"C":0.1}'))
    assert set(res.distribution) == {"A", "B", "C"}


# -- prompt faithfulness ---------------------------------------------------


def test_unconditioned_prompt_is_byte_identical_to_calibrated_commitment():
    # With no audience, the messages `ask` sends must equal the validated
    # CalibratedCommitmentStrategy output for the same record.
    rec = build_ask_record("Do you agree?", ["Yes", "No"])
    expected = CalibratedCommitmentStrategy().build_messages(rec)

    client = _CaptureClient('{"A":0.5,"B":0.5}')
    ask("Do you agree?", options=["Yes", "No"], audience=None, client=client)
    assert client.messages == expected


def test_audience_appears_in_population_phrase():
    client = _CaptureClient('{"A":0.5,"B":0.5}')
    ask("Q?", options=["a", "b"], audience="US tech workers in 2025", client=client)
    user = client.messages[-1]["content"]
    assert "Consider a large, representative sample of US tech workers in 2025." in user
    # system prompt is unchanged by conditioning
    assert client.messages[0]["content"] == CalibratedCommitmentStrategy.SYSTEM


def test_result_carries_resolved_model_id():
    res = ask("Q?", options=["a", "b"], client=_CaptureClient())
    assert res.model == "google/gemini-3.1-flash-lite"


def test_askresult_is_serializable_shape():
    res = ask("Q?", options=["a", "b"], client=_CaptureClient())
    assert isinstance(res, AskResult)
    assert res.question == "Q?"
