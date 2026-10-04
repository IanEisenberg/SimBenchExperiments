"""Stage 11 — abstention / uniform-fallback on hard tasks (dev, pop).

(1) Per-pop-task breakdown for anti_flattening vs calibrated_commitment.
(2) Abstention: learn on a disjoint dev-fit half which datasets the model scores
    below uniform on, and predict uniform there on the dev-eval half. Oracle
    ceiling = per-item max(score, 0).

Usage:
    uv run python scripts/run_abstention.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers
from simbench_exp.experiment import make_client
from simbench_exp.persona import get_strategy
from simbench_exp.predict import ZeroShotPredictor
from simbench_exp.scoring import bootstrap_ci, simbench_score
from simbench_exp.splits import make_split

RUN_NAME = "2026-06-21-abstention"
MODEL = "gemini-3.1-flash-lite"
STRATEGIES = ["anti_flattening", "calibrated_commitment"]
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def preds_for(strategy_name, records, workers=8):
    pred = ZeroShotPredictor(make_client(MODEL, max_retries=10, timeout=90),
                             name=strategy_name, strategy=get_strategy(strategy_name))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(pred.predict, records))


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
    pop = [r for r in split.dev if r.split == "pop"]
    print(f"[abstain] dev pop n={len(pop)}")

    # predictions + per-item model score, per strategy
    model_pred = {s: preds_for(s, pop) for s in STRATEGIES}
    model_score = {s: [score(r, p, norm) for r, p in zip(pop, model_pred[s])]
                   for s in STRATEGIES}
    uni_score = [score(r, uniform(r), norm) for r in pop]   # what abstaining yields

    # ---- (1) per-pop-task breakdown ----
    by_ds = {}
    for i, r in enumerate(pop):
        by_ds.setdefault(r.dataset_name, []).append(i)
    rows = []
    for ds, idx in by_ds.items():
        row = {"dataset": ds, "n": len(idx),
               "uniform": round(float(np.mean([uni_score[i] for i in idx])), 1)}
        for s in STRATEGIES:
            row[s] = round(float(np.mean([model_score[s][i] for i in idx])), 1)
        row["cc_vs_anti"] = round(row["calibrated_commitment"] - row["anti_flattening"], 1)
        rows.append(row)
    tbl = pd.DataFrame(rows).sort_values("calibrated_commitment")
    pd.set_option("display.width", 200)
    print("\n=== per-pop-task (mean SimBench) ===")
    print(tbl.to_string(index=False))

    # ---- (2) abstention, learned per-dataset on disjoint dev-fit half ----
    # split each dataset 50/50 (first half fit, second half eval), deterministic
    fit_idx, eval_idx = [], []
    for ds, idx in by_ds.items():
        h = len(idx) // 2
        fit_idx += idx[:h]; eval_idx += idx[h:]
    print(f"\n[abstain] fit n={len(fit_idx)} eval n={len(eval_idx)} (per-dataset 50/50)")

    print("\n=== abstention on dev-eval pop ===")
    summary = []
    for s in STRATEGIES:
        ms = model_score[s]
        # learn abstain set on FIT: datasets whose fit-mean model score < 0
        abstain = set()
        for ds, idx in by_ds.items():
            f = [ms[i] for i in idx if i in set(fit_idx)]
            if f and np.mean(f) < 0:
                abstain.add(ds)
        # apply on EVAL
        base = [ms[i] for i in eval_idx]
        absd = [uni_score[i] if pop[i].dataset_name in abstain else ms[i] for i in eval_idx]
        oracle = [max(ms[i], uni_score[i]) for i in eval_idx]   # per-item ceiling
        bm, blo, bhi = bootstrap_ci(base)
        am, alo, ahi = bootstrap_ci(absd)
        om = float(np.mean(oracle))
        cap = (am - bm) / (om - bm) if om > bm else float("nan")
        print(f"  {s:22s} model={bm:5.1f}[{blo:.1f},{bhi:.1f}]  "
              f"abstain={am:5.1f}[{alo:.1f},{ahi:.1f}]  (+{am-bm:.1f})  "
              f"oracle={om:5.1f}  captured {cap:.0%}")
        print(f"       abstains on: {sorted(abstain)}")
        summary.append({"strategy": s, "model": round(bm, 2), "abstain": round(am, 2),
                        "delta": round(am - bm, 2), "oracle": round(om, 2),
                        "abstain_datasets": sorted(abstain)})

    tbl.to_csv(OUTPUTS / f"{RUN_NAME}.pertask.csv", index=False)
    (OUTPUTS / f"{RUN_NAME}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[abstain] {time.time()-t0:.0f}s; saved outputs/runs/{RUN_NAME}.*")


if __name__ == "__main__":
    main()
