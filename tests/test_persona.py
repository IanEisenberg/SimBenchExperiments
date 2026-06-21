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


def test_calibrated_commitment_registered_direct_and_mode_first():
    assert "calibrated_commitment" in STRATEGIES
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    messages = get_strategy("calibrated_commitment").build_messages(rec)
    assert len(messages) == 2 and messages[0]["role"] == "system"
    text = " ".join(m["content"] for m in messages)
    # direct-answer format: must NOT ask for reasoning-first
    assert "reasoning first" not in text.lower() and "LAST" not in text
    # carries the four design elements + the mode-first lever + the group/year
    sysmsg = messages[0]["content"].lower()
    assert "leading answer" in sysmsg or "most common" in sysmsg   # mode-first
    assert "stereotype" in sysmsg and "minority" in sysmsg          # anti-flatten/diversity
    assert "divided" in sysmsg or "concentrated" in sysmsg          # consensus awareness
    user = messages[1]["content"]
    assert "from Finland" in user and "[A, B]" in user


def test_diversity_elicitation_registered_and_reasons_first():
    assert "diversity_elicitation" in STRATEGIES
    rec = _rec({"cntry": "Finland"}, "You are from Finland.")
    messages = STRATEGIES["diversity_elicitation"].build_messages(rec)
    assert len(messages) == 2 and messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "from Finland" in user
    assert "RANGE of views" in user
    assert "[A, B]" in user and "LAST" in user


def test_individual_strategy_frames_one_person_and_is_not_a_group_system():
    from scrye.persona import IndividualStrategy

    # deliberately NOT registered: a single individual is a poor group estimate
    assert "individual" not in STRATEGIES
    rec = _rec({"cntry": "Finland"}, "You are from Finland.")
    user = "\n".join(
        m["content"] for m in IndividualStrategy().build_messages(rec)
        if m["role"] == "user"
    )
    assert "ONE specific person" in user
    assert "from Finland" in user


def test_sample_persona_varies_by_seed_and_is_reproducible():
    from scrye.persona import sample_persona

    rec = _rec({"cntry": "Finland"}, "You are from Finland.")
    assert sample_persona(rec, 3) == sample_persona(rec, 3)  # seeded -> stable
    assert len({sample_persona(rec, s) for s in range(8)}) > 1  # draws differ


def test_sample_persona_skips_attributes_the_segment_already_pins():
    from scrye.persona import sample_persona

    age_clauses = ("are a young adult", "are middle-aged", "are older or retired")
    rec = _rec({"cntry": "Finland", "age": "30-49"}, "You are from Finland.")
    for s in range(12):
        descriptor = sample_persona(rec, s)
        assert not any(a in descriptor for a in age_clauses)


# -- Stage 07: superforecaster strategies ----------------------------------
def _user(name, rec):
    return "\n".join(
        m["content"] for m in get_strategy(name).build_messages(rec) if m["role"] == "user"
    )


def test_outside_view_registered_and_anchors_base_rate_first():
    assert "outside_view" in STRATEGIES
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    messages = get_strategy("outside_view").build_messages(rec)
    assert len(messages) == 2 and messages[0]["role"] == "system"
    assert "base rate" in messages[0]["content"].lower()
    user = messages[1]["content"]
    # base-rate-first, then a modest adjustment for this group
    assert "OUTSIDE VIEW" in user and "base rate" in user.lower()
    assert "ADJUST" in user
    # outside-view step precedes the adjust step
    assert user.index("OUTSIDE VIEW") < user.index("ADJUST")
    assert "from Finland" in user
    assert "[A, B]" in user and "LAST" in user


def test_entropy_first_registered_and_commits_spread_before_distribution():
    assert "entropy_first" in STRATEGIES
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    user = _user("entropy_first", rec)
    # an explicit 1-5 spread rating that the distribution must match
    assert "1-5" in user or "1–5" in user
    assert "divided" in user.lower()
    assert "SPREAD" in user and "MATCH" in user.upper()
    # spread is rated before the distribution is produced
    assert user.upper().index("SPREAD") < user.upper().index("DISTRIBUTION")
    assert "from Finland" in user
    assert "[A, B]" in user and "LAST" in user


def test_superforecaster_runs_full_pipeline_in_order():
    assert "superforecaster" in STRATEGIES
    rec = _rec({"cntry": "Finland"}, "The year is 2016. You are from Finland.")
    user = _user("superforecaster", rec)
    # all four moves present
    for marker in ("OUTSIDE VIEW", "SPREAD", "ADJUST", "PREMORTEM"):
        assert marker in user, marker
    # and in pipeline order: outside view -> spread -> adjust -> premortem
    order = [user.index(m) for m in ("OUTSIDE VIEW", "SPREAD", "ADJUST", "PREMORTEM")]
    assert order == sorted(order)
    assert "base rate" in user.lower()
    assert "from Finland" in user
    assert "[A, B]" in user and "LAST" in user


def test_new_forecasting_strategies_register_as_predictors():
    from scrye.experiment import PREDICTOR_REGISTRY

    for name in ("outside_view", "entropy_first", "superforecaster"):
        assert name in PREDICTOR_REGISTRY
