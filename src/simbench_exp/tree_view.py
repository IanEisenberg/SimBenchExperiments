"""ASCII renderer for the experiment-tree ledger.

Turns the search DAG into an indented, status-marked tree for the notebook and
the presentation: ✓ accepted on val, ✗ rejected by the Ladder, · dev-only (no
val query spent). This rendered tree plus the query ledger is the rigor artifact
that shows the search did not launder noise.
"""

from __future__ import annotations

from .ledger import ExperimentNode, Ledger


def _mark(node: ExperimentNode) -> str:
    if node.val_score is None:
        return "·"
    return "✓" if node.accepted else "✗"


def _fmt(node: ExperimentNode) -> str:
    val = "" if node.val_score is None else f" val={node.val_score:.2f}"
    cost = f" ${node.cost_usd:.2f}" if node.cost_usd else ""
    return f"{_mark(node)} {node.lever_id} (dev={node.dev_score:.2f}{val}){cost}"


def tree_lines(ledger: Ledger, depth: int = 0, parent_id: str | None = None) -> list[str]:
    """Render the tree as indented lines, depth-first from the roots."""
    lines: list[str] = []
    for node in ledger.children(parent_id) if parent_id else _roots(ledger):
        lines.append("  " * depth + _fmt(node))
        lines.extend(tree_lines(ledger, depth + 1, node.node_id))
    return lines


def _roots(ledger: Ledger) -> list[ExperimentNode]:
    return [n for n in ledger.nodes if n.parent_id is None]
