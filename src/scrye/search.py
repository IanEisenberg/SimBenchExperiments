# src/scrye/search.py
"""The guarded search loop — bounded autonomy over a frozen lever set.

The operator hands the loop a SearchContract (goal + allowed levers + budgets +
gate policy). Each step: a proposer picks a registered lever; the resulting spec
is scored freely on DEV; if it improves dev it is promoted to a single Ladder-
guarded VAL query (skipped — and no K spent — if that config was already
queried). Every step is logged to the ledger DAG. The loop halts at a budget, on
an off-registry proposal, or when a gate fires, returning control to the human.

Scoring is injected (`score_fn`) so the loop is fully unit-testable offline; the
production `score_fn` wraps `scrye.evaluate.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .ladder import LadderGate
from .ledger import ExperimentNode, Ledger
from .levers import LEVER_REGISTRY, apply_lever
from .spec import PipelineSpec


@dataclass(frozen=True)
class GatePolicy:
    gate_off_registry_proposal: bool = True   # forced True (see run_search)
    gate_on_val_query: bool = False
    gate_at_val_budget_frac: float = 1.0
    gate_on_surprise: bool = True
    gate_after_dev_steps: int | None = None
    gate_at_cost_frac: float = 0.8            # pause when spend hits this fraction of the cap


@dataclass(frozen=True)
class SearchContract:
    goal: str
    allowed_levers: list[str]
    autonomy_budget: int
    val_query_budget: int
    gate_policy: GatePolicy
    eta: float = 0.0
    cost_cap_usd: float = 1000.0   # hard ceiling on cumulative NEW OpenRouter spend


@dataclass(frozen=True)
class Proposal:
    lever_id: str
    params: dict
    rationale: str


@dataclass
class SearchState:
    spec: PipelineSpec
    dev_best: float
    val_best: float
    k_spent: int
    cum_cost: float = 0.0        # cumulative NEW spend so far
    last_step_cost: float = 0.0  # cost of the previous step (projection for the pre-step guard)
    history: list = field(default_factory=list)


@dataclass
class SearchOutcome:
    stop_reason: str
    final_spec: PipelineSpec
    steps: int


ScoreFn = Callable[[PipelineSpec, Sequence], tuple[float, dict]]
ProposeFn = Callable[[SearchState], "Proposal | None"]
NowFn = Callable[[], str]


def run_search(
    contract: SearchContract,
    ledger: Ledger,
    *,
    dev: Sequence,
    val: Sequence,
    score_fn: ScoreFn,
    propose_fn: ProposeFn,
    now_fn: NowFn,
    start_spec: PipelineSpec | None = None,
    client=None,
) -> SearchOutcome:
    spec = start_spec or PipelineSpec()
    # Reconstruct global state from the ledger so resuming never resets K/best.
    gate = LadderGate(eta=contract.eta, best=ledger.best_val(), k=ledger.global_k())
    state = SearchState(spec=spec, dev_best=float("-inf"),
                        val_best=ledger.best_val(), k_spent=ledger.global_k(),
                        cum_cost=ledger.total_cost())

    steps = 0
    dev_steps = 0
    for _ in range(contract.autonomy_budget):
        proposal = propose_fn(state)
        if proposal is None:
            return SearchOutcome("proposer_done", state.spec, steps)

        # The one non-disableable gate: off-registry proposals always halt.
        if proposal.lever_id not in LEVER_REGISTRY or proposal.lever_id not in contract.allowed_levers:
            return SearchOutcome("off_registry_proposal", state.spec, steps)

        # Pre-step dollar guard: if the projected next spend (≈ last step's spend)
        # would cross the cap, halt before spending rather than overshoot it.
        if state.last_step_cost and state.cum_cost + state.last_step_cost > contract.cost_cap_usd:
            return SearchOutcome("cost_cap_reached", state.spec, steps)

        new_spec = apply_lever(state.spec, proposal.lever_id, proposal.params)
        chash = new_spec.config_hash()

        dev_score, breakdown = score_fn(new_spec, dev)
        dev_steps += 1
        step_cost = float(breakdown.get("cost_usd", 0.0))  # new spend for the dev score
        promote = dev_score > state.dev_best

        val_score = eta_used = accepted = k_at = None
        if promote:
            # Pre-query gates.
            gp = contract.gate_policy
            if gp.gate_on_val_query:
                return SearchOutcome("gate_on_val_query", state.spec, steps)
            if state.k_spent >= contract.val_query_budget:
                return SearchOutcome("val_budget_reached", state.spec, steps)

            cached = ledger.by_config_hash(chash)
            if cached is not None and cached.val_score is not None:
                # Cache hit: reuse the prior val score, spend NO new K.
                val_score = cached.val_score
                accepted = val_score is not None and val_score > gate.best + gate.eta
                eta_used = gate.eta
                k_at = state.k_spent
            else:
                val_score, val_breakdown = score_fn(new_spec, val)
                step_cost += float(val_breakdown.get("cost_usd", 0.0))  # val score spend
                accepted = gate.consider(val_score)  # spends one K, may raise best
                eta_used = gate.eta
                state.k_spent = gate.k
                k_at = gate.k

            if accepted:
                state.spec = new_spec
                state.dev_best = dev_score
                state.val_best = gate.best

        # Account the step's new spend before logging, so the node carries it.
        state.cum_cost += step_cost
        state.last_step_cost = step_cost

        node = ExperimentNode(
            node_id=ledger.next_node_id(), config_hash=chash,
            parent_id=_last_node_id(ledger), lever_id=proposal.lever_id,
            rationale=proposal.rationale, dev_score=dev_score, dev_breakdown=breakdown,
            val_score=val_score, eta=eta_used, accepted=accepted,
            global_k_at_query=k_at, timestamp=now_fn(),
            cost_usd=step_cost, cum_cost_usd=state.cum_cost,
        )
        ledger.append(node)
        state.history.append(node.node_id)
        steps += 1

        # Post-step gates.
        gp = contract.gate_policy
        if promote and state.k_spent >= contract.val_query_budget:
            return SearchOutcome("val_budget_reached", state.spec, steps)
        if (contract.val_query_budget
                and state.k_spent >= gp.gate_at_val_budget_frac * contract.val_query_budget
                and gp.gate_at_val_budget_frac < 1.0):
            return SearchOutcome("val_budget_frac_gate", state.spec, steps)
        if (contract.cost_cap_usd
                and state.cum_cost >= gp.gate_at_cost_frac * contract.cost_cap_usd
                and gp.gate_at_cost_frac < 1.0):
            return SearchOutcome("cost_frac_gate", state.spec, steps)
        if gp.gate_after_dev_steps and dev_steps >= gp.gate_after_dev_steps:
            return SearchOutcome("dev_steps_gate", state.spec, steps)
        if gp.gate_on_surprise and promote and accepted is False:
            # dev predicted improvement but val rejected -> surprising.
            return SearchOutcome("surprise_gate", state.spec, steps)

    return SearchOutcome("autonomy_budget_reached", state.spec, steps)


def _last_node_id(ledger: Ledger) -> str | None:
    return ledger.nodes[-1].node_id if ledger.nodes else None
