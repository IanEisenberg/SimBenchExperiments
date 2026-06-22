"""Stage 14 follow-up — task_context_items vs calibrated_commitment on the POP split.

Answers: does the sibling-items context that won on grouped also help pop?
Dev-only, leakage-safe. Per-dataset + pooled + leakage-safe selective routing.
"""
from __future__ import annotations
import json, time
from collections import Counter
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

MODEL = "gemini-3.1-flash-lite"
CAP = 40
OUT = Path("/Users/ian/.claude/jobs/2080a501/tmp/ctx_pop.results.json")


def score(rec, pred, norm):
    return simbench_score(pred, rec.human_answer, options=list(rec.options),
                          normalizer=norm.get((rec.split, rec.dataset_name)))


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)
    corpus = build_item_corpus(full)

    devpop = [r for r in split.dev if r.split == "pop"]
    datasets = [d for d, _ in Counter(r.dataset_name for r in devpop).most_common()]

    per_ds = {}
    for ds in datasets:
        pool = sorted([r for r in devpop if r.dataset_name == ds],
                      key=lambda r: r.input_template)[:CAP]
        if pool:
            per_ds[ds] = pool

    client = make_client(MODEL, max_retries=10, timeout=90)
    cc = ZeroShotPredictor(client, strategy=get_strategy("calibrated_commitment"))
    it = ZeroShotPredictor(client, strategy=TaskContextStrategy(corpus, mode="items", k=6))

    cc_sc, it_sc = {}, {}
    for ds, recs in per_ds.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            cp = list(ex.map(cc.predict, recs))
            ip = list(ex.map(it.predict, recs))
        cc_sc[ds] = np.array([score(r, p, norm) for r, p in zip(recs, cp)])
        it_sc[ds] = np.array([score(r, p, norm) for r, p in zip(recs, ip)])
        print(f"{ds:18s} n={len(recs):3d}  cc={cc_sc[ds].mean():7.2f}  "
              f"items={it_sc[ds].mean():7.2f}  Δ={it_sc[ds].mean()-cc_sc[ds].mean():+6.2f}")

    cc_all = np.concatenate([cc_sc[d] for d in per_ds])
    it_all = np.concatenate([it_sc[d] for d in per_ds])
    d = it_all - cc_all
    _, dlo, dhi = bootstrap_ci(d.tolist())
    print(f"\nPOOLED pop  cc={cc_all.mean():.2f}  items={it_all.mean():.2f}  "
          f"Δ={it_all.mean()-cc_all.mean():+.2f}  95% CI [{dlo:+.2f},{dhi:+.2f}]")

    # leakage-safe 2-fold per-dataset selector
    def fold(even):
        sel, base = [], []
        ch = {}
        for ds in per_ds:
            c, i = cc_sc[ds], it_sc[ds]
            m = np.arange(len(c)) % 2 == 0
            fit = m if even else ~m
            use = i[fit].mean() > c[fit].mean()
            ch[ds] = "items" if use else "cc"
            sel.extend((i if use else c)[~fit].tolist()); base.extend(c[~fit].tolist())
        return np.array(sel), np.array(base), ch
    s1, b1, c1 = fold(True); s2, b2, c2 = fold(False)
    sel = np.concatenate([s1, s2]); base = np.concatenate([b1, b2])
    _, slo, shi = bootstrap_ci((sel - base).tolist())
    print(f"selective pop  cc={base.mean():.2f}  selective={sel.mean():.2f}  "
          f"Δ={sel.mean()-base.mean():+.2f}  95% CI [{slo:+.2f},{shi:+.2f}]")
    helped = sorted([d for d in per_ds if it_sc[d].mean() > cc_sc[d].mean() + 1])
    print("datasets where items helps (>+1):", helped)

    OUT.write_text(json.dumps({
        "pooled": {"cc": round(float(cc_all.mean()), 2), "items": round(float(it_all.mean()), 2),
                   "delta": round(float(it_all.mean()-cc_all.mean()), 2), "ci": [round(dlo,2), round(dhi,2)]},
        "selective": {"cc": round(float(base.mean()), 2), "selective": round(float(sel.mean()), 2),
                      "delta": round(float(sel.mean()-base.mean()), 2), "ci": [round(slo,2), round(shi,2)]},
        "per_dataset": {d: {"cc": round(float(cc_sc[d].mean()), 2), "items": round(float(it_sc[d].mean()), 2)}
                        for d in per_ds},
        "secs": round(time.time()-t0, 1)}, indent=2))
    print(f"\nsaved -> {OUT}  ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
