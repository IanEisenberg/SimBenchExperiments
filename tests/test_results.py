"""Tests for scrye.results — query and reproduction helpers."""

import pytest

from scrye.ledger import ExperimentNode, Ledger
from scrye.results import (
    best_progression,
    format_node_report,
    node_report,
    nodes_frame,
    replay_proposals,
    spec_from_node,
)
from scrye.spec import PipelineSpec


def _node(led, config_hash, parent=None, lever="recalib.global_temp",
          dev=10.0, val=None, accepted=None, k=None,
          cost=0.0, cum=0.0, params=None, spec_config=None, rationale="because"):
    return ExperimentNode(
        node_id=led.next_node_id(), config_hash=config_hash, parent_id=parent,
        lever_id=lever, rationale=rationale, dev_score=dev, dev_breakdown={},
        val_score=val, eta=(0.5 if val is not None else None), accepted=accepted,
        global_k_at_query=k, timestamp="2026-06-20T00:00:00",
        cost_usd=cost, cum_cost_usd=cum,
        params=params or {},
        spec_config=spec_config or {},
    )


def _make_ledger(tmp_path):
    """Build a small synthetic ledger with three nodes (two val-queried)."""
    p = tmp_path / "run.jsonl"
    led = Ledger(p)

    spec0 = PipelineSpec()
    spec1 = PipelineSpec(calibrators=[{"name": "temp", "kwargs": {"T": 1.5}}])
    spec2 = PipelineSpec(calibrators=[
        {"name": "temp", "kwargs": {"T": 1.5}},
        {"name": "dirichlet", "kwargs": {"alpha": 0.05}},
    ])

    n0 = _node(led, "hash0", lever="recalib.global_temp",
               dev=10.0, val=20.0, accepted=True, k=1,
               cost=1.0, cum=1.0,
               params={"T": 1.5}, spec_config=spec1.resolved(),
               rationale="try temp")
    led.append(n0)

    n1 = _node(led, "hash1", parent=n0.node_id, lever="recalib.dirichlet",
               dev=15.0, val=25.0, accepted=True, k=2,
               cost=1.5, cum=2.5,
               params={"alpha": 0.05}, spec_config=spec2.resolved(),
               rationale="try dirichlet")
    led.append(n1)

    # Dev-only node (no val query)
    n2 = _node(led, "hash2", parent=n1.node_id, lever="recalib.global_temp",
               dev=5.0, val=None, accepted=None, k=None,
               cost=0.0, cum=2.5,
               params={"T": 2.0}, spec_config=spec0.resolved(),
               rationale="lower score dev only")
    led.append(n2)

    return led, (n0, n1, n2)


# -- nodes_frame -----------------------------------------------------------

def test_nodes_frame_shape(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    df = nodes_frame(led)
    assert len(df) == 3
    assert list(df.columns) == [
        "node_id", "parent_id", "lever_id", "depth",
        "dev_score", "val_score", "accepted",
        "global_k_at_query", "cost_usd", "cum_cost_usd", "config_hash",
    ]


def test_nodes_frame_depth(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    df = nodes_frame(led).set_index("node_id")
    assert df.loc[n0.node_id, "depth"] == 0
    assert df.loc[n1.node_id, "depth"] == 1
    assert df.loc[n2.node_id, "depth"] == 2


def test_nodes_frame_scores(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    df = nodes_frame(led).set_index("node_id")
    assert df.loc[n0.node_id, "val_score"] == 20.0
    assert df.loc[n2.node_id, "val_score"] is None or str(df.loc[n2.node_id, "val_score"]) in ("nan", "None")


# -- node_report -----------------------------------------------------------

def test_node_report_fields(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    r = node_report(led, n1.node_id)
    assert r["lever_path"] == ["recalib.global_temp", "recalib.dirichlet"]
    assert r["val_score"] == 25.0
    assert r["params"] == {"alpha": 0.05}
    assert r["spec_config"]["calibrators"][1]["name"] == "dirichlet"
    assert r["rationale"] == "try dirichlet"


def test_node_report_missing_raises(tmp_path):
    led, _ = _make_ledger(tmp_path)
    with pytest.raises(KeyError):
        node_report(led, "nonexistent_id")


# -- format_node_report ----------------------------------------------------

def test_format_node_report_returns_string(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    text = format_node_report(led, n0.node_id)
    assert isinstance(text, str)
    assert "recalib.global_temp" in text
    assert "20.0" in text  # val score
    assert "T" in text  # params key


# -- best_progression ------------------------------------------------------

def test_best_progression_columns(tmp_path):
    led, _ = _make_ledger(tmp_path)
    bp = best_progression(led)
    assert list(bp.columns) == ["step", "node_id", "lever_id", "val_score", "best_so_far", "is_new_best"]


def test_best_progression_logic(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    bp = best_progression(led)
    # n0: val=20, first -> is_new_best; n1: val=25 > 20 -> is_new_best
    assert len(bp) == 2  # only two val-queried nodes
    assert bp.iloc[0]["val_score"] == 20.0
    assert bool(bp.iloc[0]["is_new_best"]) is True
    assert bp.iloc[1]["val_score"] == 25.0
    assert bool(bp.iloc[1]["is_new_best"]) is True
    assert bp.iloc[1]["best_so_far"] == 25.0


def test_best_progression_lower_score_not_new_best(tmp_path):
    """A lower val_score after a higher one must have is_new_best=False."""
    p = tmp_path / "run.jsonl"
    led = Ledger(p)
    spec0 = PipelineSpec()
    n0 = _node(led, "h0", lever="recalib.global_temp",
               dev=20.0, val=30.0, accepted=True, k=1,
               spec_config=spec0.resolved())
    led.append(n0)
    n1 = _node(led, "h1", parent=n0.node_id, lever="recalib.dirichlet",
               dev=22.0, val=15.0, accepted=False, k=2,
               spec_config=spec0.resolved())
    led.append(n1)
    bp = best_progression(led)
    assert len(bp) == 2
    assert bool(bp.iloc[0]["is_new_best"]) is True
    assert bp.iloc[1]["val_score"] == 15.0
    assert bool(bp.iloc[1]["is_new_best"]) is False
    assert bp.iloc[1]["best_so_far"] == 30.0  # max unchanged


# -- spec_from_node --------------------------------------------------------

def test_spec_from_node(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    spec = spec_from_node(led, n1.node_id)
    assert isinstance(spec, PipelineSpec)
    assert len(spec.calibrators) == 2


def test_spec_from_node_missing_raises(tmp_path):
    led, _ = _make_ledger(tmp_path)
    with pytest.raises(KeyError):
        spec_from_node(led, "bad_id")


def test_spec_from_node_empty_spec_config_raises(tmp_path):
    p = tmp_path / "run.jsonl"
    led = Ledger(p)
    n = _node(led, "h0", spec_config={})  # no spec_config
    led.append(n)
    with pytest.raises(ValueError):
        spec_from_node(led, n.node_id)


# -- replay_proposals ------------------------------------------------------

def test_replay_proposals(tmp_path):
    led, (n0, n1, n2) = _make_ledger(tmp_path)
    proposals = replay_proposals(led)
    assert len(proposals) == 3
    assert proposals[0].lever_id == "recalib.global_temp"
    assert proposals[0].params == {"T": 1.5}
    assert proposals[1].lever_id == "recalib.dirichlet"
    assert proposals[1].params == {"alpha": 0.05}
    assert proposals[2].lever_id == "recalib.global_temp"
