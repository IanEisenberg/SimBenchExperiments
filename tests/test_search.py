# tests/test_search.py
"""Tests for the guarded search loop — offline, via injected score_fn/propose_fn."""

from scrye.ledger import Ledger
from scrye.search import GatePolicy, Proposal, SearchContract, run_search
from scrye.spec import PipelineSpec


def _now():
    return "2026-06-19T00:00:00"


def _contract(**kw):
    base = dict(
        goal="improve S", allowed_levers=["recalib.global_temp", "recalib.dirichlet"],
        autonomy_budget=5, val_query_budget=3, gate_policy=GatePolicy(), eta=0.5,
    )
    base.update(kw)
    return SearchContract(**base)


def test_off_registry_proposal_halts(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    proposals = iter([Proposal("not.a.lever", {}, "rogue")])
    out = run_search(
        _contract(), led, dev=[1], val=[1],
        score_fn=lambda spec, recs: (10.0, {}),
        propose_fn=lambda state: next(proposals, None), now_fn=_now,
    )
    assert out.stop_reason == "off_registry_proposal"
    assert len(led.nodes) == 0  # nothing logged for a rejected rogue proposal


def test_accepts_improving_lever_and_spends_one_k(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # dev scores improve so the lever is promoted; val score clears eta.
    scores = {"dev": 20.0, "val": 30.0}
    proposals = iter([Proposal("recalib.global_temp", {"T": 1.5}, "flatten")])

    def score_fn(spec, recs):
        return (scores["dev"] if recs == ["dev"] else scores["val"]), {"entropy_temp": 0.0}

    out = run_search(
        _contract(autonomy_budget=1), led, dev=["dev"], val=["val"],
        score_fn=score_fn, propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    assert led.global_k() == 1
    assert led.nodes[0].accepted is True
    assert led.nodes[0].val_score == 30.0
    assert out.final_spec.lever_path == ["recalib.global_temp"]


def test_cached_config_revisit_spends_no_new_k(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # Two proposals that resolve to the SAME config (same lever+params).
    proposals = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "first"),
        Proposal("recalib.global_temp", {"T": 1.5}, "again"),
    ])

    def score_fn(spec, recs):
        return (20.0 if recs == ["dev"] else 30.0), {}

    out = run_search(
        _contract(autonomy_budget=2, gate_policy=GatePolicy(gate_on_surprise=False)),
        led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    # Second step revisits the same config_hash off the SAME parent -> cache hit.
    assert led.global_k() == 1


def test_val_budget_cap_halts(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # Each distinct lever improves dev (promoted) and clears val eta -> spends K.
    seq = [
        Proposal("recalib.global_temp", {"T": 1.5}, "a"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "b"),
        Proposal("recalib.dirichlet", {"alpha": 0.1}, "c"),
    ]
    proposals = iter(seq)
    n = {"i": 0}

    def score_fn(spec, recs):
        # Monotonically increasing val so each clears eta; dev always promotes.
        if recs == ["dev"]:
            return 20.0 + n["i"], {}
        n["i"] += 1
        return 30.0 + 5 * n["i"], {}

    out = run_search(
        _contract(autonomy_budget=10, val_query_budget=2,
                  gate_policy=GatePolicy(gate_on_surprise=False)),
        led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    assert out.stop_reason == "val_budget_reached"
    assert led.global_k() == 2


def test_propose_none_stops_cleanly(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    out = run_search(
        _contract(), led, dev=["dev"], val=["val"],
        score_fn=lambda spec, recs: (10.0, {}),
        propose_fn=lambda state: None, now_fn=_now,
    )
    assert out.stop_reason == "proposer_done"


def test_cost_cap_halts_before_exceeding(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # Each step's dev score carries $10 of new spend (in the breakdown); cap = $25,
    # so the loop logs two $10 steps and blocks the third before it can spend.
    proposals = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "a"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "b"),
        Proposal("recalib.dirichlet", {"alpha": 0.1}, "c"),
    ])

    def score_fn(spec, recs):
        # Dev call carries the cost; val call is free here. Dev always promotes.
        if recs == ["dev"]:
            return 20.0 + len(spec.lever_path), {"cost_usd": 10.0}
        return 30.0 + len(spec.lever_path), {"cost_usd": 0.0}

    out = run_search(
        # gate_at_cost_frac=1.0 disables the soft pause so we test the hard cap alone.
        _contract(autonomy_budget=10, val_query_budget=10, cost_cap_usd=25.0,
                  gate_policy=GatePolicy(gate_on_surprise=False, gate_at_cost_frac=1.0)),
        led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(proposals, None), now_fn=_now,
    )
    assert out.stop_reason == "cost_cap_reached"
    assert abs(led.total_cost() - 20.0) < 1e-9       # two $10 steps logged; third blocked
    assert led.nodes[-1].cum_cost_usd == 20.0
