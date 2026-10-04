"""Offline demo of the goal-directed lever search — NO API calls, no spend.

Runs the guarded loop with a deterministic *stub* scorer so you can see the
whole machinery work end to end: propose a lever -> score on dev (free) ->
promote -> Ladder-gated val query (spends global K) -> log to the experiment
tree -> render the tree -> produce a test-once, K-corrected report.

Run it:

    python examples/demo_lever_search.py

This is the skeleton for a *real* run: swap the stub `score_fn` for
`simbench_exp.report.make_score_fn(normalizers, client=...)` (which calls OpenRouter)
and feed real dev/val/test splits from `simbench_exp.splits.make_split`. See the
"REAL RUN" note at the bottom.
"""

import pathlib
import sys
import tempfile

# Make the src-layout package importable without an editable install.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from simbench_exp.data import SimBenchRecord
from simbench_exp.evaluate import build_normalizers
from simbench_exp.ledger import Ledger
from simbench_exp.report import final_report
from simbench_exp.search import GatePolicy, Proposal, SearchContract, run_search
from simbench_exp.spec import PipelineSpec
from simbench_exp.tree_view import tree_lines


def _rec(options, truth):
    return SimBenchRecord(
        dataset_name="ESS", split="grouped", input_template="Q?",
        options=tuple(options), human_answer=dict(truth), group_prompt="",
    )


def main() -> None:
    # A pre-registered exploration contract. eta=0.5 is the Ladder noise buffer
    # (in a real run, set it from simbench_exp.ladder.bootstrap_eta on a baseline's
    # val scores, computed BEFORE any val query).
    contract = SearchContract(
        goal="improve high-entropy S without regressing consensus",
        allowed_levers=["recalib.global_temp", "recalib.dirichlet"],
        autonomy_budget=4,
        val_query_budget=3,
        gate_policy=GatePolicy(gate_on_surprise=False),
        eta=0.5,
    )

    # The "proposer" — in production this is Claude reading the trail and choosing
    # the next theory-motivated lever. Here it is a fixed script of two proposals.
    proposals = iter([
        Proposal("recalib.global_temp", {"T": 1.5}, "flatten mode-seeking over-sharp output"),
        Proposal("recalib.dirichlet", {"alpha": 0.05}, "lift under-predicted tails"),
    ])

    # The stub scorer: deterministic, offline. Dev and val both improve as levers
    # stack, so each is promoted on dev and accepted by the Ladder on val. Each
    # scoring call reports $1 of (pretend) spend so the cost ledger is non-trivial.
    def score_fn(spec, records):
        depth = len(spec.lever_path)
        return 20.0 + 5.0 * depth, {"by_entropy": {}, "cost_usd": 1.0}

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(pathlib.Path(tmp) / "demo_run.jsonl")
        outcome = run_search(
            contract, ledger,
            dev=["dev"], val=["val"],
            score_fn=score_fn,
            propose_fn=lambda state: next(proposals, None),
            now_fn=lambda: "2026-06-20T00:00:00",
        )

        print("=" * 64)
        print("GUARDED LEVER SEARCH — offline demo (stub scorer, no API calls)")
        print("=" * 64)
        print(f"stop reason     : {outcome.stop_reason}")
        print(f"final levers    : {outcome.final_spec.lever_path}")
        print(f"global K (val)  : {ledger.global_k()}   <- distinct val-queried configs")
        print(f"total cost (USD): {ledger.total_cost():.2f}   <- cache hits cost $0")
        print()
        print("Experiment tree (✓ accepted on val / ✗ rejected / · dev-only):")
        for line in tree_lines(ledger):
            print("  " + line)
        print()

        # Test-once headline on a held-out set. Uniform predictor => no network,
        # and it averages to ~0 under SimBench Eq.2 normalizers (the paper's
        # baseline property). The K-corrected band widens with global K.
        test_records = [
            _rec(["A", "B"], {"A": 0.7, "B": 0.3}),
            _rec(["A", "B", "C"], {"A": 0.5, "B": 0.3, "C": 0.2}),
        ]
        normalizers = build_normalizers(test_records)
        report = final_report(PipelineSpec(predictor="uniform"), test_records, normalizers, ledger)

        print("Test-once report (frozen pipeline scored on TEST exactly once):")
        for key in ("pipeline", "n", "mean_score", "ci_low", "ci_high",
                    "global_k", "k_band_low", "k_band_high",
                    "total_cost_usd", "test_eval_cost_usd", "grand_total_cost_usd"):
            print(f"  {key:22s}: {report[key]}")
        print()
        print("Note: 'selected from global_k distinct val queries' — the K-band is the")
        print("generalization-honest interval (Blum-Hardt Ladder bound), wider than the")
        print("raw bootstrap CI. That widening is the overfitting-honesty artifact.")

    # ----------------------------------------------------------------------
    # REAL RUN (live, costs money): replace the stub score_fn with
    #   from simbench_exp.report import make_score_fn
    #   score_fn = make_score_fn(normalizers, client=make_client("gemini-flash-lite"))
    # build dev/val/test via simbench_exp.splits.make_split(load_all()...), set
    # contract.eta from simbench_exp.ladder.bootstrap_eta(baseline_val_scores), and let
    # Claude (or your loop) supply real Proposals. Requires OPENROUTER_API_KEY.


if __name__ == "__main__":
    main()
