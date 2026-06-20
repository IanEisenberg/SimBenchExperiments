"""Tests for the persona factory: verbalization, country catalog, strategies.

No network access — strategies only build message dicts.
"""

import pytest

from scrye.data import SimBenchRecord
from scrye.persona import (
    ALL_COUNTRIES,
    COUNTRIES,
    DEFAULT_STRATEGY,
    STRATEGIES,
    extract_locale,
    extract_year,
    faithful_prompt,
    get_strategy,
    verbalize_segment,
)


def _rec(segment, group_prompt, options=("A", "B"), q="Do you agree?\nA) yes\nB) no"):
    return SimBenchRecord(
        dataset_name="ESS",
        split="grouped",
        input_template=q,
        options=tuple(options),
        human_answer={o: 1.0 / len(options) for o in options},
        group_prompt=group_prompt,
        segment=dict(segment),
        num_grouping_vars=len(segment),
        group_size=100,
    )


# -- country catalog -------------------------------------------------------
def test_country_catalog_covers_all_grouped_datasets():
    assert set(COUNTRIES) == {
        "OpinionQA", "ESS", "Afrobarometer", "ISSP", "LatinoBarometro"
    }
    assert len(COUNTRIES["ESS"]) == 28
    assert len(COUNTRIES["Afrobarometer"]) == 39


def test_all_countries_sorted_and_deduped():
    assert ALL_COUNTRIES == tuple(sorted(set(ALL_COUNTRIES)))
    assert "Finland" in ALL_COUNTRIES  # appears in ESS and ISSP, listed once


# -- verbalize_segment -----------------------------------------------------
def test_verbalize_country_and_age_ordering():
    desc = verbalize_segment({"age_group": "30-49", "cntry": "Finland"})
    # country leads, then age, regardless of dict order
    assert desc == "from Finland, in the 30-49 age group"


def test_verbalize_numeric_scale_strips_trailing_zero():
    desc = verbalize_segment({"lrscale": "8.0"})
    assert "at 8 on a 0-10 left-right scale" in desc
    assert "8.0" not in desc


def test_verbalize_unknown_key_uses_fallback():
    desc = verbalize_segment({"some_new_var": "weird value"})
    assert desc == "whose some new var is weird value"


def test_verbalize_empty_segment_is_empty():
    assert verbalize_segment({}) == ""


def test_verbalize_locale_added_when_no_country():
    desc = verbalize_segment({"AGE": "18-29"}, locale="the United States")
    assert desc == "from the United States, in the 18-29 age group"


def test_verbalize_locale_ignored_when_country_present():
    desc = verbalize_segment({"country": "Kenya"}, locale="the United States")
    assert desc == "from Kenya"


# -- context extraction ----------------------------------------------------
def test_extract_year():
    assert extract_year("The year is 2016. You are from Israel.") == "2016"
    assert extract_year("no year here") is None


def test_extract_locale_from_segment_then_prompt():
    assert extract_locale(_rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")) == "Finland"
    # OpinionQA-style: country implicit in the prompt
    assert extract_locale(_rec({}, "You are from the United States. The year is 2017.")) == "the United States"


# -- strategies ------------------------------------------------------------
def test_faithful_prompt_contains_persona_and_question_and_json():
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    prompt = faithful_prompt(rec)
    assert "You are from Finland" in prompt
    assert "Do you agree?" in prompt
    assert "JSON object" in prompt
    assert "[A, B]" in prompt


def test_faithful_strategy_matches_faithful_prompt():
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    messages = get_strategy("simbench_faithful").build_messages(rec)
    assert messages == [{"role": "user", "content": faithful_prompt(rec)}]


def test_default_strategy_is_faithful():
    assert DEFAULT_STRATEGY == "simbench_faithful"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
@pytest.mark.parametrize(
    "segment,prompt",
    [
        ({}, "The year is 2017. You are from the United States."),
        ({"cntry": "Finland"}, "The year is 2016. You are from Finland."),
        ({"cntry": "Finland", "age_group": "30-49"},
         "The year is 2016. You are from Finland. You are in the 30-49 age group."),
    ],
)
def test_every_strategy_builds_valid_messages(name, segment, prompt):
    rec = _rec(segment, prompt)
    messages = get_strategy(name).build_messages(rec)
    assert isinstance(messages, list) and messages
    assert all(m["role"] in {"system", "user"} for m in messages)
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)
    # the question and the JSON ask must always reach the model
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    assert "Do you agree?" in user
    assert "[A, B]" in user


def test_third_person_strategies_mention_the_group():
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    for name in ("representative_sample", "anti_flattening", "contextualized"):
        user = "\n".join(
            m["content"] for m in get_strategy(name).build_messages(rec)
            if m["role"] == "user"
        )
        assert "from Finland" in user
        assert "2016" in user


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("does_not_exist")
