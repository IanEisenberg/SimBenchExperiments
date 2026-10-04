"""Stage 14 Round 3 — leakage-safe per-dataset routing of task_context (dev).

Round 2 showed task_context_items helps ESS/OpinionQA but hurts the Global-South
barometers. Learn per dataset on a dev-FIT half whether items beats cc; apply to
the disjoint dev-EVAL half. 2-fold. All predictions are cached from Round 2, so
this is fast.

Usage:
    uv run python scripts/run_task_context_select.py
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
from simbench_exp.persona import TaskContextStrategy, build_item_corpus, get_strategy
from simbench_exp.predict import ZeroShotPredictor
from simbench_exp.scoring import bootstrap_ci, simbench_score
from simbench_exp.splits import make_split

RUN_NAME = "2026-06-21-task-context-select"
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

    per_ds_recs = {}
    for ds in GROUPED:
        per_ds_recs[ds] = sorted(
            [r for r in split.dev if r.dataset_name == ds and r.split == "grouped"],
            key=lambda r: (r.input_template, str(r.segment)))[:CAP]

    client = make_client(MODEL, max_retries=10, timeout=90)
    cc = ZeroShotPredictor(client, strategy=get_strategy("calibrated_commitment"))
    it = ZeroShotPredictor(client, strategy=TaskContextStrategy(corpus, mode="items", k=6))

    cc_sc, it_sc = {}, {}
    for ds, recs in per_ds_recs.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            cpred = list(ex.map(cc.predict, recs))
            ipred = list(ex.map(it.predict, recs))
        cc_sc[ds] = np.array([score(r, p, norm) for r, p in zip(recs, cpred)])
        it_sc[ds] = np.array([score(r, p, norm) for r, p in zip(recs, ipred)])

    # 2-fold leakage-safe per-dataset selector
    def fold(fit_is_even):
        sel, base_cc, base_it = [], [], []
        chosen = {}
        for ds in GROUPED:
            c, i = cc_sc[ds], it_sc[ds]
            n = len(c); m = np.arange(n) % 2 == 0
            fit = m if fit_is_even else ~m
            ev = ~fit
            use_items = i[fit].mean() > c[fit].mean()
            chosen[ds] = "items" if use_items else "cc"
            sel.extend((i if use_items else c)[ev].tolist())
            base_cc.extend(c[ev].tolist()); base_it.extend(i[ev].tolist())
        return np.array(sel), np.array(base_cc), np.array(base_it), chosen

    s1, c1, i1, ch1 = fold(True)
    s2, c2, i2, ch2 = fold(False)
    sel = np.concatenate([s1, s2]); base_cc = np.concatenate([c1, c2]); base_it = np.concatenate([i1, i2])

    print("per-dataset (full-dev):")
    for ds in GROUPED:
        print(f"  {ds:16s} cc={cc_sc[ds].mean():7.2f}  items={it_sc[ds].mean():7.2f}  "
              f"Δ={it_sc[ds].mean()-cc_sc[ds].mean():+6.2f}")
    print(f"\nfold1 chose: {ch1}\nfold2 chose: {ch2}")
    print(f"\neval-half pooled:  cc={base_cc.mean():.2f}  items={base_it.mean():.2f}  "
          f"selective={sel.mean():.2f}")
    d_sel = sel - base_cc
    _, lo, hi = bootstrap_ci(d_sel.tolist())
    print(f"selective Δ vs cc = {sel.mean()-base_cc.mean():+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]")

    out = {"run": RUN_NAME,
           "per_dataset": {ds: {"cc": round(float(cc_sc[ds].mean()), 2),
                                "items": round(float(it_sc[ds].mean()), 2)}
                           for ds in GROUPED},
           "fold1": ch1, "fold2": ch2,
           "eval_cc": round(float(base_cc.mean()), 2),
           "eval_items": round(float(base_it.mean()), 2),
           "eval_selective": round(float(sel.mean()), 2),
           "selective_delta": round(float(sel.mean() - base_cc.mean()), 2),
           "selective_delta_ci": [round(lo, 2), round(hi, 2)],
           "secs": round(time.time() - t0, 1)}
    (OUTPUTS / f"{RUN_NAME}.results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUTPUTS / f'{RUN_NAME}.results.json'}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
