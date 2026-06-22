"""Stage 14 Round 2 — grouped-survey confirmation of task_context_items (dev).

Confirm the Round-1 control-group signal: does task_context_items beat
calibrated_commitment on the GROUPED split across all five grouped datasets?

Usage:
    uv run python scripts/run_task_context_grouped.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scrye.data import load_all
from scrye.evaluate import build_normalizers
from scrye.experiment import make_client
from scrye.persona import TaskContextStrategy, build_item_corpus, get_strategy
from scrye.predict import ZeroShotPredictor
from scrye.scoring import bootstrap_ci, simbench_score
from scrye.splits import make_split

RUN_NAME = "2026-06-21-task-context-grouped"
MODEL = "gemini-3.1-flash-lite"
GROUPED = ["ESS", "OpinionQA", "Afrobarometer", "ISSP", "LatinoBarometro"]
CAP = 120
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def score(rec, pred, norm):
    return simbench_score(pred, rec.human_answer, options=list(rec.options),
                          normalizer=norm.get((rec.split, rec.dataset_name)))


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)
    corpus = build_item_corpus(full)

    recs = []
    for ds in GROUPED:
        pool = sorted([r for r in split.dev
                       if r.dataset_name == ds and r.split == "grouped"],
                      key=lambda r: (r.input_template, str(r.segment)))[:CAP]
        recs.extend(pool)
        print(f"[ctx-grouped] {ds}: {len(pool)} dev recs")
    print(f"[ctx-grouped] total {len(recs)}")

    client = make_client(MODEL, max_retries=10, timeout=90)
    systems = {
        "calibrated_commitment": ZeroShotPredictor(
            client, strategy=get_strategy("calibrated_commitment")),
        "task_context_items": ZeroShotPredictor(
            client, strategy=TaskContextStrategy(corpus, mode="items", k=6)),
    }
    sc = {}
    for name, pred in systems.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            preds = list(ex.map(pred.predict, recs))
        sc[name] = [score(r, p, norm) for r, p in zip(recs, preds)]
        print(f"  scored {name}")

    by_ds = {}
    for i, r in enumerate(recs):
        by_ds.setdefault(r.dataset_name, []).append(i)

    print(f"\n{'dataset':16s} {'n':>4s}  {'cc':>8s}  {'items':>8s}  {'Δ':>7s}")
    rows = []
    wins = 0
    for ds in GROUPED:
        idx = by_ds[ds]
        cc = float(np.mean([sc["calibrated_commitment"][i] for i in idx]))
        it = float(np.mean([sc["task_context_items"][i] for i in idx]))
        wins += it > cc
        rows.append({"dataset": ds, "n": len(idx), "cc": round(cc, 2),
                     "items": round(it, 2), "delta": round(it - cc, 2)})
        print(f"{ds:16s} {len(idx):4d}  {cc:8.2f}  {it:8.2f}  {it-cc:+7.2f}")

    cc_all = sc["calibrated_commitment"]
    it_all = sc["task_context_items"]
    cc_m, cc_lo, cc_hi = bootstrap_ci(cc_all)
    it_m, it_lo, it_hi = bootstrap_ci(it_all)
    # paired bootstrap on the delta
    d = np.array(it_all) - np.array(cc_all)
    _, dlo, dhi = bootstrap_ci(d.tolist())
    print(f"\nPOOLED grouped   cc={cc_m:.2f} [{cc_lo:.1f},{cc_hi:.1f}]   "
          f"items={it_m:.2f} [{it_lo:.1f},{it_hi:.1f}]")
    print(f"paired Δ = {it_m-cc_m:+.2f}   95% CI [{dlo:+.2f}, {dhi:+.2f}]   "
          f"per-dataset wins: {wins}/5")

    out = {"run": RUN_NAME, "model": MODEL, "cap": CAP, "per_dataset": rows,
           "pooled": {"cc": round(cc_m, 2), "items": round(it_m, 2),
                      "delta": round(it_m - cc_m, 2),
                      "delta_ci": [round(dlo, 2), round(dhi, 2)],
                      "wins": f"{wins}/5"},
           "secs": round(time.time() - t0, 1)}
    (OUTPUTS / f"{RUN_NAME}.results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUTPUTS / f'{RUN_NAME}.results.json'}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
