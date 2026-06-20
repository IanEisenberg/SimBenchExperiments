"""Tests for the append-only experiment-tree ledger."""

from scrye.ledger import ExperimentNode, Ledger


def _node(led, config_hash, parent=None, lever="recalib.global_temp",
          dev=10.0, val=None, accepted=None, k=None, cost=0.0, cum=0.0):
    return ExperimentNode(
        node_id=led.next_node_id(), config_hash=config_hash, parent_id=parent,
        lever_id=lever, rationale="because", dev_score=dev, dev_breakdown={},
        val_score=val, eta=(0.5 if val is not None else None), accepted=accepted,
        global_k_at_query=k, timestamp="2026-06-19T00:00:00",
        cost_usd=cost, cum_cost_usd=cum,
    )


def test_append_and_load_roundtrip(tmp_path):
    p = tmp_path / "run.jsonl"
    led = Ledger(p)
    n0 = _node(led, "hashA")
    led.append(n0)
    reloaded = Ledger.load(p)
    assert len(reloaded.nodes) == 1
    assert reloaded.nodes[0].config_hash == "hashA"
    assert reloaded.by_id(n0.node_id).lever_id == "recalib.global_temp"


def test_by_config_hash_is_cache_lookup(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1))
    assert led.by_config_hash("hashA").val_score == 12.0
    assert led.by_config_hash("missing") is None


def test_global_k_counts_distinct_queried_hashes(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1))
    led.append(_node(led, "hashB", val=11.0, accepted=False, k=2))
    led.append(_node(led, "hashA", val=12.0, accepted=False, k=2))  # revisit, same hash
    led.append(_node(led, "hashC", val=None))  # dev-only, no val query
    assert led.global_k() == 2  # hashA, hashB; revisit and dev-only don't add


def test_best_val_among_queried(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1))
    led.append(_node(led, "hashB", val=9.0, accepted=False, k=2))
    assert led.best_val() == 12.0


def test_total_cost_sums_new_spend(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.append(_node(led, "hashA", val=12.0, accepted=True, k=1, cost=3.0, cum=3.0))
    led.append(_node(led, "hashB", val=9.0, accepted=False, k=2, cost=2.0, cum=5.0))
    led.append(_node(led, "hashC", cost=0.0, cum=5.0))  # cache hit / dev-only, no new spend
    assert abs(led.total_cost() - 5.0) < 1e-9


def test_path_to_root_and_children(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    n0 = _node(led, "h0"); led.append(n0)
    n1 = _node(led, "h1", parent=n0.node_id); led.append(n1)
    n2 = _node(led, "h2", parent=n1.node_id); led.append(n2)
    assert [n.node_id for n in led.path_to_root(n2.node_id)] == [n0.node_id, n1.node_id, n2.node_id]
    assert [n.node_id for n in led.children(n0.node_id)] == [n1.node_id]
