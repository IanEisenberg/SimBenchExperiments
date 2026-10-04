"""Experiment-results query and reproduction helpers.

All read from a loaded Ledger. Requires pandas.

Functions:
    nodes_frame(ledger)        — DataFrame of all nodes
    node_report(ledger, id)    — full dict readout of one node
    format_node_report(...)    — human-readable string for the notebook
    best_progression(ledger)   — the "Ladder climb" as a DataFrame
    spec_from_node(ledger, id) — rebuild a PipelineSpec from a node's spec_config
    replay_proposals(ledger)   — reconstruct the recorded proposal sequence
"""

from __future__ import annotations

import pandas as pd

from .ledger import Ledger
from .search import Proposal
from .spec import PipelineSpec


def nodes_frame(ledger: Ledger) -> pd.DataFrame:
    """One row per node.

    Columns: node_id, parent_id, lever_id, depth, dev_score, val_score,
    accepted, global_k_at_query, cost_usd, cum_cost_usd, config_hash.
    """
    rows = []
    for node in ledger.nodes:
        depth = len(ledger.path_to_root(node.node_id)) - 1
        rows.append({
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "lever_id": node.lever_id,
            "depth": depth,
            "dev_score": node.dev_score,
            "val_score": node.val_score,
            "accepted": node.accepted,
            "global_k_at_query": node.global_k_at_query,
            "cost_usd": node.cost_usd,
            "cum_cost_usd": node.cum_cost_usd,
            "config_hash": node.config_hash,
        })
    return pd.DataFrame(rows)


def node_report(ledger: Ledger, node_id: str) -> dict:
    """Full readout of one node.

    Returns dict with: lever_path, rationale, dev_score, dev_breakdown,
    val_score, eta, accepted, global_k_at_query, cost_usd, cum_cost_usd,
    params, spec_config.

    Raises KeyError if node_id is missing.
    """
    node = ledger.by_id(node_id)
    if node is None:
        raise KeyError(f"Node {node_id!r} not found in ledger.")
    path = ledger.path_to_root(node_id)
    lever_path = [n.lever_id for n in path]
    return {
        "lever_path": lever_path,
        "rationale": node.rationale,
        "dev_score": node.dev_score,
        "dev_breakdown": node.dev_breakdown,
        "val_score": node.val_score,
        "eta": node.eta,
        "accepted": node.accepted,
        "global_k_at_query": node.global_k_at_query,
        "cost_usd": node.cost_usd,
        "cum_cost_usd": node.cum_cost_usd,
        "params": node.params,
        "spec_config": node.spec_config,
    }


def format_node_report(ledger: Ledger, node_id: str) -> str:
    """A readable multi-line string of a node report (for notebook printing)."""
    r = node_report(ledger, node_id)
    lines = [
        f"Node: {node_id}",
        f"  Lever path : {' -> '.join(r['lever_path']) if r['lever_path'] else '(root)'}",
        f"  Rationale  : {r['rationale']}",
        f"  Dev score  : {r['dev_score']:.4f}",
        f"  Val score  : {r['val_score']:.4f}" if r["val_score"] is not None else "  Val score  : (not queried)",
        f"  Eta        : {r['eta']}",
        f"  Accepted   : {r['accepted']}",
        f"  Global K   : {r['global_k_at_query']}",
        f"  Cost USD   : ${r['cost_usd']:.4f}  (cum: ${r['cum_cost_usd']:.4f})",
        f"  Params     : {r['params']}",
        f"  Spec config: {r['spec_config']}",
    ]
    if r["dev_breakdown"]:
        lines.append(f"  Dev breakdown:")
        for k, v in r["dev_breakdown"].items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def best_progression(ledger: Ledger) -> pd.DataFrame:
    """The Ladder climb: val_nodes in append order, with running max and is_new_best.

    Columns: step (1-based), node_id, lever_id, val_score, best_so_far, is_new_best.
    """
    rows = []
    running_max = float("-inf")
    for step, node in enumerate(ledger.val_nodes(), start=1):
        val = node.val_score  # guaranteed non-None for val_nodes
        is_new_best = val > running_max
        running_max = max(running_max, val)
        rows.append({
            "step": step,
            "node_id": node.node_id,
            "lever_id": node.lever_id,
            "val_score": val,
            "best_so_far": running_max,
            "is_new_best": is_new_best,
        })
    return pd.DataFrame(rows)


def spec_from_node(ledger: Ledger, node_id: str) -> PipelineSpec:
    """Rebuild a PipelineSpec from a node's recorded spec_config.

    Raises KeyError if node_id missing; raises ValueError if spec_config is empty.
    """
    node = ledger.by_id(node_id)
    if node is None:
        raise KeyError(f"Node {node_id!r} not found in ledger.")
    if not node.spec_config:
        raise ValueError(
            f"Node {node_id!r} has no spec_config (recorded before v2 reproducibility "
            "plumbing). Cannot reconstruct PipelineSpec."
        )
    return PipelineSpec(**node.spec_config)


def replay_proposals(ledger: Ledger) -> list[Proposal]:
    """Reconstruct the recorded proposal sequence in node order.

    Returns a list of Proposal(lever_id, params, rationale) for every node in the
    ledger, in append order. Pass this list to a proposer to enable deterministic
    re-runs.
    """
    return [
        Proposal(lever_id=node.lever_id, params=node.params, rationale=node.rationale)
        for node in ledger.nodes
    ]


# ---------------------------------------------------------------------------
# Canonical run registry — single source of truth for analysis notebooks
# ---------------------------------------------------------------------------

from .config import OUTPUTS_DIR as _OUTPUTS_DIR

#: Registry of named analysis runs: name -> {path, description, best_system}.
#: "final" is the authoritative run for error decomposition; others are kept
#: for historical comparison.  path is relative to OUTPUTS_DIR / "runs".
#:
#: All analysis runs are on **dev** by design. Diagnostic notebooks load these
#: repeatedly, so they must not touch `val` (sparingly used, gated) or `test`
#: (sealed, one-shot). The one-shot test numbers live in the Stage 17 doc and
#: notebook 03, not here.
ANALYSIS_RUNS: dict[str, dict] = {
    "final": {
        "path": "2026-06-21-decomp-final.results.json",
        "description": "Faithful vs final system (capped dev, ~715 recs)",
        "best_system": "final_router",
        "note": (
            "Generated by scripts/run_decomp_data.py. The final system is the "
            "val-confirmed task-kind router, but on held-out TEST it is "
            "statistically tied with the simpler calibrated_commitment+abstain "
            "(Stage 16/17) — the router's val edge did not transfer."
        ),
    },
    "model-sweep": {
        "path": "2026-06-20-model-sweep.results.json",
        "description": "3 models x 3 strategies (capped dev, ~440 recs)",
        "best_system": "gemini-3.1-flash-lite::anti_flattening",
    },
    "fulldev": {
        "path": "2026-06-20-fulldev-confirm.results.json",
        "description": "Faithful / anti_flattening / contextualized on full dev (6151 recs)",
        "best_system": "anti_flattening",
    },
}


def best_analysis_run(name: str = "final") -> "Path":
    """Return the absolute path to a canonical analysis run results file.

    ``name`` must be a key of :data:`ANALYSIS_RUNS`.  Defaults to ``"final"``
    (the faithful-vs-router run).  Falls back to ``"fulldev"`` if the requested
    file does not exist yet (e.g. before :mod:`scripts.run_decomp_data` has run).

    Usage in a notebook::

        from simbench_exp.results import best_analysis_run
        import json
        per_system = json.load(open(best_analysis_run()))

    Raises :class:`FileNotFoundError` only when *all* fallbacks are missing.
    """
    from pathlib import Path as _Path

    order = [name] + [k for k in ANALYSIS_RUNS if k != name]
    for key in order:
        entry = ANALYSIS_RUNS.get(key)
        if entry is None:
            continue
        p = _OUTPUTS_DIR / "runs" / entry["path"]
        if p.exists():
            if key != name:
                import warnings
                warnings.warn(
                    f"best_analysis_run({name!r}) not found; falling back to {key!r}.",
                    stacklevel=2,
                )
            return p
    raise FileNotFoundError(
        f"No analysis run found for {name!r}. "
        f"Run `uv run python scripts/run_decomp_data.py` to generate it."
    )
