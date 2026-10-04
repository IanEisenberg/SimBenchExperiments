"""Tests for the `simbench-ask` CLI surface — parsing, rendering, and a
network-free run() with an injected fake client.
"""

import json

from simbench_exp.ask import AskResult
from simbench_exp.cli_ask import parse_options, render_json, render_text, run
from simbench_exp.llm import LLMResponse


class _CaptureClient:
    model = "google/gemini-3.1-flash-lite"

    def __init__(self, text='{"A": 0.6, "B": 0.4}'):
        self.text = text

    def complete(self, messages, **overrides):
        return LLMResponse(text=self.text, model=self.model, cached=False)


def _result():
    return AskResult(
        question="Will remote work keep growing?",
        audience="US tech workers in 2025",
        model="google/gemini-3.1-flash-lite",
        options={"A": "Agree", "B": "Somewhat agree", "C": "Somewhat disagree", "D": "Disagree"},
        distribution={"A": 0.44, "B": 0.31, "C": 0.16, "D": 0.09},
    )


# -- option parsing --------------------------------------------------------


def test_parse_options_splits_on_semicolons_and_strips():
    assert parse_options("Agree; Somewhat agree ;Disagree") == ["Agree", "Somewhat agree", "Disagree"]


def test_parse_options_blank_returns_none():
    assert parse_options("") is None
    assert parse_options(None) is None
    assert parse_options("   ") is None


def test_parse_options_keeps_text_with_commas():
    assert parse_options("Yes, definitely; No") == ["Yes, definitely", "No"]


# -- rendering -------------------------------------------------------------


def test_render_text_shows_each_option_label_text_and_prob():
    out = render_text(_result())
    assert "A  Agree" in out
    assert "0.44" in out
    assert "audience: US tech workers in 2025" in out
    assert "Will remote work keep growing?" in out


def test_render_text_unconditioned_says_general_population():
    res = AskResult("Q?", None, "m", {"A": "a", "B": "b"}, {"A": 0.5, "B": 0.5})
    assert "general population" in render_text(res)


def test_render_json_has_expected_shape():
    payload = json.loads(render_json(_result()))
    assert set(payload) == {"question", "audience", "model", "options", "distribution"}
    assert payload["distribution"]["A"] == 0.44
    assert payload["options"]["A"] == "Agree"


# -- run (network-free) ----------------------------------------------------


def test_run_with_injected_client_returns_text_block():
    out = run(
        question="Q?",
        options=["Yes", "No"],
        audience=None,
        model="gemini-3.1-flash-lite",
        as_json=False,
        client=_CaptureClient('{"A": 0.6, "B": 0.4}'),
    )
    assert "A  Yes" in out
    assert "0.6" in out


def test_run_json_mode_returns_parseable_json():
    out = run(
        question="Q?",
        options=["Yes", "No"],
        audience="cat owners",
        model="gemini-3.1-flash-lite",
        as_json=True,
        client=_CaptureClient('{"A": 0.6, "B": 0.4}'),
    )
    payload = json.loads(out)
    assert payload["audience"] == "cat owners"
    assert abs(sum(payload["distribution"].values()) - 1.0) < 1e-9
