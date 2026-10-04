"""The experiment-tree ledger — an append-only DAG of search states.

Each node is one experiment: a config-hashed pipeline state, the lever that
produced it, its dev score, and (only if a val query was spent) its val score,
the Ladder threshold, the verdict, and the cumulative global K at query time.
Because state is the ordered composition of levers, a node's full config is
reconstructable from its path to root, and any node is a resumption point.

The ledger is the source of truth for global K (distinct val-queried configs
across the WHOLE tree — branching never resets it) and for the cache: revisiting
a config_hash already queried returns its stored score and costs no new K.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentNode:
    node_id: str
    config_hash: str
    parent_id: str | None
    lever_id: str
    rationale: str
    dev_score: float
    dev_breakdown: dict
    val_score: float | None
    eta: float | None
    accepted: bool | None
    global_k_at_query: int | None
    timestamp: str
    cost_usd: float = 0.0       # NEW OpenRouter spend for this node (cache hits = 0.0)
    cum_cost_usd: float = 0.0   # running tree total at this node
    params: dict = field(default_factory=dict)        # the lever's params (replayable proposals)
    spec_config: dict = field(default_factory=dict)   # resolved pipeline config -> node's exact pipeline is rebuildable


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.nodes: list[ExperimentNode] = []

    # -- persistence -------------------------------------------------------
    def append(self, node: ExperimentNode) -> None:
        self.nodes.append(node)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(node)) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        led = cls(path)
        p = Path(path)
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        led.nodes.append(ExperimentNode(**json.loads(line)))
        return led

    # -- ids ---------------------------------------------------------------
    def next_node_id(self) -> str:
        return f"n{len(self.nodes):04d}"

    # -- lookups -----------------------------------------------------------
    def by_id(self, node_id: str) -> ExperimentNode | None:
        return next((n for n in self.nodes if n.node_id == node_id), None)

    def by_config_hash(self, config_hash: str) -> ExperimentNode | None:
        """First node with a val score for this config (the cache hit), else
        any node with this config, else None."""
        queried = [n for n in self.nodes if n.config_hash == config_hash and n.val_score is not None]
        if queried:
            return queried[0]
        return next((n for n in self.nodes if n.config_hash == config_hash), None)

    def val_nodes(self) -> list[ExperimentNode]:
        return [n for n in self.nodes if n.val_score is not None]

    def global_k(self) -> int:
        return len({n.config_hash for n in self.val_nodes()})

    def best_val(self) -> float:
        vals = [n.val_score for n in self.val_nodes()]
        return max(vals) if vals else float("-inf")

    def total_cost(self) -> float:
        """Total NEW OpenRouter spend across the whole tree (cache hits add 0)."""
        return float(sum(n.cost_usd for n in self.nodes))

    # -- tree --------------------------------------------------------------
    def children(self, node_id: str) -> list[ExperimentNode]:
        return [n for n in self.nodes if n.parent_id == node_id]

    def path_to_root(self, node_id: str) -> list[ExperimentNode]:
        chain: list[ExperimentNode] = []
        cur = self.by_id(node_id)
        while cur is not None:
            chain.append(cur)
            cur = self.by_id(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))
