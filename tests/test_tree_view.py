"""Tests for the ASCII experiment-tree renderer."""

from simbench_exp.ledger import ExperimentNode, Ledger
from simbench_exp.tree_view import tree_lines


def _n(led, h, parent, lever, val, accepted):
    return ExperimentNode(
        node_id=led.next_node_id(), config_hash=h, parent_id=parent, lever_id=lever,
        rationale="r", dev_score=10.0, dev_breakdown={}, val_score=val,
        eta=0.5, accepted=accepted, global_k_at_query=(1 if val else None),
        timestamp="2026-06-19T00:00:00",
    )


def test_tree_lines_marks_status_and_nests(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    n0 = _n(led, "h0", None, "recalib.global_temp", 30.0, True); led.append(n0)
    n1 = _n(led, "h1", n0.node_id, "recalib.dirichlet", 29.0, False); led.append(n1)
    n2 = _n(led, "h2", n0.node_id, "model.swap", None, None); led.append(n2)
    lines = tree_lines(led)
    assert len(lines) == 3
    assert any("✓" in ln and "recalib.global_temp" in ln for ln in lines)
    assert any("✗" in ln and "recalib.dirichlet" in ln for ln in lines)
    assert any("·" in ln and "model.swap" in ln for ln in lines)
    # children are indented deeper than their parent
    parent_line = next(ln for ln in lines if "recalib.global_temp" in ln)
    child_line = next(ln for ln in lines if "recalib.dirichlet" in ln)
    assert (len(child_line) - len(child_line.lstrip())) > (len(parent_line) - len(parent_line.lstrip()))
