"""Stage 13 Round 1 — discrete-vote simulation on the weak pop tasks (dev).

Compare on the three abstain datasets (Choices13k, MoralMachine, OSPsychMACH):
  - uniform            (the current abstain fallback; the bar, ~0)
  - calibrated_commitment single-call (what abstain replaced)
  - voting_ensemble    (NEW: K individuals each cast one vote; tally = dist)

Usage:
    uv run python scripts/run_voting_sim.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers
from simbench_exp.experiment import make_client
from simbench_exp.persona import get_strategy
from simbench_exp.predict import VotingEnsemblePredictor, ZeroShotPredictor
from simbench_exp.scoring import bootstrap_ci, simbench_score
from simbench_exp.splits import make_split

RUN_NAME = "2026-06-21-voting-sim"
MODEL = "gemini-3.1-flash-lite"
DATASETS = {"Choices13k": 60, "MoralMachine": 60, "OSPsychMACH": 9999}
N_VOTES = 16
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def uniform(rec):
    k = rec.num_options or len(rec.options) or 1
    return {o: 1.0 / k for o in rec.options}


def score(rec, pred, norm):
    return simbench_score(pred, rec.human_answer, options=list(rec.options),
                          normalizer=norm.get((rec.split, rec.dataset_name)))


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)

    # deterministic per-dataset subsample of dev
    recs = []
    for ds, cap in DATASETS.items():
        pool = [r for r in split.dev if r.dataset_name == ds]
        pool = sorted(pool, key=lambda r: r.input_template)[:cap]
        recs.extend(pool)
        print(f"[voting] {ds}: {len(pool)} dev recs")
    print(f"[voting] total {len(recs)} recs; {N_VOTES} votes each "
          f"≈ {len(recs) * N_VOTES} calls")

    client = make_client(MODEL, max_retries=10, timeout=90)
    cc = ZeroShotPredictor(client, name="calibrated_commitment",
                           strategy=get_strategy("calibrated_commitment"))
    ve = VotingEnsemblePredictor(client, n_individuals=N_VOTES, alpha=0.5)

    def cc_pred(r): return cc.predict(r)
    def ve_pred(r): return ve.predict(r)

    with ThreadPoolExecutor(max_workers=8) as ex:
        cc_preds = list(ex.map(cc_pred, recs))
    with ThreadPoolExecutor(max_workers=8) as ex:
        ve_preds = list(ex.map(ve_pred, recs))

    systems = {
        "uniform": [uniform(r) for r in recs],
        "calibrated_commitment": cc_preds,
        "voting_ensemble": ve_preds,
    }
    sc = {name: [score(r, p, norm) for r, p in zip(recs, preds)]
          for name, preds in systems.items()}

    # per-dataset table
    by_ds = {}
    for i, r in enumerate(recs):
        by_ds.setdefault(r.dataset_name, []).append(i)

    print(f"\n{'dataset':16s} {'n':>4s}  " +
          "  ".join(f"{s:>22s}" for s in systems))
    rows = []
    for ds, idx in by_ds.items():
        row = {"dataset": ds, "n": len(idx)}
        cells = []
        for s in systems:
            vals = [sc[s][i] for i in idx]
            _, lo, hi = bootstrap_ci(vals)
            row[s] = round(float(np.mean(vals)), 2)
            row[f"{s}_ci"] = [round(lo, 1), round(hi, 1)]
            cells.append(f"{np.mean(vals):6.2f} [{lo:5.1f},{hi:5.1f}]")
        rows.append(row)
        print(f"{ds:16s} {len(idx):4d}  " + "  ".join(cells))

    # pooled over the 3 datasets
    allidx = list(range(len(recs)))
    print(f"\n{'POOLED':16s} {len(recs):4d}  " + "  ".join(
        f"{np.mean([sc[s][i] for i in allidx]):6.2f}" for s in systems))

    pooled = {s: round(float(np.mean(sc[s])), 2) for s in systems}
    out = {"run": RUN_NAME, "model": MODEL, "n_votes": N_VOTES,
           "per_dataset": rows, "pooled": pooled,
           "n_parse_failures": ve.n_parse_failures,
           "secs": round(time.time() - t0, 1)}
    (OUTPUTS / f"{RUN_NAME}.results.json").write_text(json.dumps(out, indent=2))
    print(f"\nvote parse failures: {ve.n_parse_failures} / {len(recs) * N_VOTES}")
    print(f"saved -> {OUTPUTS / f'{RUN_NAME}.results.json'}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
