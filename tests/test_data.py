"""Tests for SimBench data parsing.

The pure parsing helpers are unit-tested without network access. The full-load
tests are skipped automatically if the dataset hasn't been downloaded yet, so
the suite stays green on a fresh checkout.
"""

import pytest

from simbench_exp.data import (
    SimBenchRecord,
    _normalize_distribution,
    _parse_keyset,
    _parse_mapping,
    _row_to_record,
)
from simbench_exp.config import RAW_DIR, SIMBENCH_FILES


# -- pure helpers ----------------------------------------------------------
def test_parse_mapping_handles_python_literal():
    assert _parse_mapping("{'cntry': 'Finland', 'age_group': '30-49'}") == {
        "cntry": "Finland",
        "age_group": "30-49",
    }


def test_parse_mapping_empty_cases():
    assert _parse_mapping("{}") == {}
    assert _parse_mapping("nan") == {}
    assert _parse_mapping(None) == {}
    assert _parse_mapping("") == {}


def test_parse_keyset_set_literal():
    assert _parse_keyset("{'age_group', 'cntry'}") == ("age_group", "cntry")


def test_parse_keyset_empty_set():
    assert _parse_keyset("set()") == ()
    assert _parse_keyset(None) == ()


def test_normalize_distribution_renormalizes_percentages():
    out = _normalize_distribution({"A": 60.0, "B": 30.0, "C": 10.0})
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert abs(out["A"] - 0.6) < 1e-9


def test_normalize_distribution_handles_imperfect_sum():
    out = _normalize_distribution({"A": 33.0, "B": 33.0, "C": 33.0})
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_row_to_record_population():
    row = {
        "dataset_name": "ESS",
        "input_template": "Q?\nOptions:\n(A): yes\n(B): no",
        "human_answer": "{'A': 70.0, 'B': 30.0}",
        "group_prompt_template": "You are from Finland.",
        "group_prompt_variable_map": "{}",
        "grouping_keys": "set()",
        "num_grouping_vars": 0,
        "group_size": 100,
    }
    rec = _row_to_record(row, "grouped")
    assert isinstance(rec, SimBenchRecord)
    assert rec.options == ("A", "B")
    assert abs(rec.human_answer["A"] - 0.7) < 1e-9
    assert rec.is_population


def test_row_to_record_conditioned():
    row = {
        "dataset_name": "ESS",
        "input_template": "Q?",
        "human_answer": "{'A': 50, 'B': 50}",
        "group_prompt_template": "You are from Finland. You are in the 30-49 age group.",
        "group_prompt_variable_map": "{'cntry': 'Finland', 'age_group': '30-49'}",
        "grouping_keys": "{'age_group', 'cntry'}",
        "num_grouping_vars": 2,
        "group_size": 50,
    }
    rec = _row_to_record(row, "grouped")
    assert rec.segment == {"cntry": "Finland", "age_group": "30-49"}
    assert rec.grouping_keys == ("age_group", "cntry")
    assert not rec.is_population


def test_row_to_record_skips_unusable():
    row = {"dataset_name": "X", "human_answer": "{}", "input_template": "Q"}
    assert _row_to_record(row, "pop") is None


# -- full-load smoke test (skipped if data not downloaded) -----------------
def _data_present() -> bool:
    return all((RAW_DIR / f).exists() for f in SIMBENCH_FILES.values())


@pytest.mark.skipif(not _data_present(), reason="SimBench CSVs not downloaded")
def test_load_split_counts():
    from simbench_exp.data import load_split

    pop = load_split("pop", download=False)
    grouped = load_split("grouped", download=False)
    # Counts reported in the SimBench paper.
    assert len(pop) == 7167
    assert len(grouped) == 6343
    # Every record has a normalized distribution.
    for r in pop[:100]:
        assert abs(sum(r.human_answer.values()) - 1.0) < 1e-6
