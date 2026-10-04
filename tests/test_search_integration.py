"""End-to-end (offline) integration: a short search produces a coherent ledger
and a test-once report, with global K honestly counted across the tree."""

from simbench_exp.ledger import Ledger
from simbench_exp.report import final_report
from simbench_exp.search import GatePolicy, Proposal, SearchContract, run_search
from simbench_exp.evaluate import build_normalizers
from simbench_exp.data import SimBenchRecord
from simbench_exp.spec import PipelineSpec


def _rec(options, truth):
    return SimBenchRecord(
        dataset_name="ESS", split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def test_full_loop_offline(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    contract = SearchContract(
        goal="improve S", allowed_levers=["recalib.global_temp", "recalib.dirichlet"],
        autonomy_budget=4, val_query_budget=3,
        gate_policy=GatePolicy(gate_on_surprise=False), eta=0.5,
    )
    seq = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "flatten over-sharp output"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "lift zeroed tails"),
    ])

    def score_fn(spec, recs):
        # dev and val both improve as levers stack, so both are promoted+accepted.
        # Each scoring call reports $1 of new spend, so each step costs $2 (dev+val).
        depth = len(spec.lever_path)
        return (20.0 + 5 * depth, {"by_entropy": {}, "cost_usd": 1.0})

    out = run_search(
        contract, led, dev=["dev"], val=["val"], score_fn=score_fn,
        propose_fn=lambda s: next(seq, None), now_fn=lambda: "2026-06-19T00:00:00",
    )
    assert out.stop_reason == "proposer_done"
    assert led.global_k() == 2
    assert out.final_spec.lever_path == ["recalib.global_temp", "recalib.dirichlet"]
    assert abs(led.total_cost() - 4.0) < 1e-9  # two steps × (dev $1 + val $1)

    # Cost-consistency invariant: the running total in the last node must equal
    # the independent sum total_cost() produces.
    assert abs(led.total_cost() - led.nodes[-1].cum_cost_usd) < 1e-9

    # Report on a held-out "test" set with the uniform predictor (no network).
    recs = [_rec(["A", "B"], {"A": 0.7, "B": 0.3})]
    norms = build_normalizers(recs)
    rep = final_report(PipelineSpec(predictor="uniform"), recs, norms, led)
    assert rep["global_k"] == 2  # report reads K from the ledger, not the run
    assert abs(rep["total_cost_usd"] - 4.0) < 1e-9  # spend carried through to the report
