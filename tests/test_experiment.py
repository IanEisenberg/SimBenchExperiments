"""Tests for the experiment harness.

Uses only the no-network ``uniform`` predictor so the suite stays offline.
Verifies model resolution, pipeline assembly, and that compare() produces a
well-formed topline table with per-system result frames attached.
"""

from __future__ import annotations

import pandas as pd

from scrye.data import SimBenchRecord
from scrye.experiment import (
    PREDICTOR_REGISTRY,
    build_pipeline,
    compare,
    resolve_model,
)


def _rec(ds: str, template: str, segment: dict | None = None) -> SimBenchRecord:
    return SimBenchRecord(
        dataset_name=ds,
        split="grouped",
        input_template=template,
        options=("A", "B", "C"),
        human_answer={"A": 0.5, "B": 0.3, "C": 0.2},
        group_prompt="",
        segment=segment or {},
        group_size=100,
        num_grouping_vars=len(segment or {}),
    )


def test_resolve_model_key_and_passthrough():
    assert resolve_model("qwen-72b") == "qwen/qwen-2.5-72b-instruct"
    # An unknown string is assumed to be a full id already.
    assert resolve_model("vendor/some-model") == "vendor/some-model"


def test_registry_has_baselines():
    assert {"zero_shot", "uniform"} <= set(PREDICTOR_REGISTRY)


def test_build_pipeline_uniform_needs_no_client():
    pipe = build_pipeline(predictor="uniform")
    assert pipe.predictor.name == "uniform"


def test_build_pipeline_unknown_predictor_raises():
    import pytest

    with pytest.raises(KeyError):
        build_pipeline(predictor="does_not_exist")


def test_compare_produces_topline_table():
    records = [
        _rec("DS", "Q1?"),
        _rec("DS", "Q1?", {"cntry": "Finland"}),
        _rec("DS", "Q2?"),
        _rec("DS", "Q2?", {"cntry": "Chile"}),
    ]
    systems = {"uniform": build_pipeline(predictor="uniform")}
    table = compare(systems, records, progress=False)
    assert isinstance(table, pd.DataFrame)
    for col in ("system", "model", "n", "mean_score", "ci_low", "ci_high", "cf_alignment"):
        assert col in table.columns
    assert table.loc[0, "system"] == "uniform"
    assert table.loc[0, "n"] == len(records)
    # Per-system result frames are attached for drill-down.
    assert "uniform" in table.attrs["results"]
    assert len(table.attrs["results"]["uniform"]) == len(records)
