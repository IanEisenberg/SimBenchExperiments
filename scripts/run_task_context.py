"""Stage 14 Round 1 — task-context prompting on dev.

Compare calibrated_commitment (atomized control) vs three context variants
(brief / items / both) on weak+psychometric datasets, with two strong controls.

Usage:
    uv run python scripts/run_task_context.py
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
from simbench_exp.persona import (
    TaskContextStrategy,
    build_item_corpus,
    get_strategy,
)
from simbench_exp.predict import ZeroShotPredictor
from simbench_exp.scoring import bootstrap_ci, simbench_score
from simbench_exp.splits import make_split

RUN_NAME = "2026-06-21-task-context"
MODEL = "gemini-3.1-flash-lite"
WEAK = ["OSPsychMACH", "Choices13k", "MoralMachine", "OSPsychBig5", "OSPsychRWAS"]
CONTROL = ["OpinionQA", "ESS"]
CAP = 40
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def score(rec, pred, norm):
    return simbench_score(pred, rec.human_answer, options=list(rec.options),
                          normalizer=norm.get((rec.split, rec.dataset_name)))


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)
    corpus = build_item_corpus(full)   # instrument items (questions only)

    # subsample dev per dataset (prefer pop; ESS uses grouped)
    recs = []
    for ds in WEAK + CONTROL:
        pool = sorted([r for r in split.dev if r.dataset_name == ds],
                      key=lambda r: r.input_template)[:CAP]
        recs.extend(pool)
        print(f"[ctx] {ds}: {len(pool)} dev recs")

    client = make_client(MODEL, max_retries=10, timeout=90)
    systems = {
        "calibrated_commitment": ZeroShotPredictor(
            client, strategy=get_strategy("calibrated_commitment")),
        "task_context_brief": ZeroShotPredictor(
            client, strategy=TaskContextStrategy(corpus, mode="brief")),
        "task_context_items": ZeroShotPredictor(
            client, strategy=TaskContextStrategy(corpus, mode="items", k=6)),
        "task_context_both": ZeroShotPredictor(
            client, strategy=TaskContextStrategy(corpus, mode="both", k=6)),
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

    names = list(systems)
    print(f"\n{'dataset':14s} {'n':>3s}  " + "  ".join(f"{n[:18]:>18s}" for n in names))
    rows = []
    for ds in WEAK + CONTROL:
        idx = by_ds.get(ds, [])
        if not idx:
            continue
        row = {"dataset": ds, "n": len(idx), "group": "weak" if ds in WEAK else "control"}
        cells = []
        for n in names:
            m = float(np.mean([sc[n][i] for i in idx]))
            row[n] = round(m, 2)
            cells.append(f"{m:18.2f}")
        rows.append(row)
        print(f"{ds:14s} {len(idx):3d}  " + "  ".join(cells))

    def pooled(group, n):
        idx = [i for i, r in enumerate(recs)
               if (r.dataset_name in WEAK) == (group == "weak")]
        vals = [sc[n][i] for i in idx]
        _, lo, hi = bootstrap_ci(vals)
        return float(np.mean(vals)), lo, hi

    print(f"\n{'POOLED weak':14s}      " + "  ".join(f"{pooled('weak',n)[0]:18.2f}" for n in names))
    print(f"{'POOLED control':14s}      " + "  ".join(f"{pooled('control',n)[0]:18.2f}" for n in names))
    print("\nΔ vs calibrated_commitment (weak pooled, with 95% CI on the variant):")
    base, _, _ = pooled("weak", "calibrated_commitment")
    for n in names[1:]:
        m, lo, hi = pooled("weak", n)
        print(f"  {n:22s} {m:7.2f}  Δ={m-base:+.2f}   CI[{lo:.1f},{hi:.1f}]")

    out = {"run": RUN_NAME, "model": MODEL, "cap": CAP, "per_dataset": rows,
           "pooled": {g: {n: round(pooled(g, n)[0], 2) for n in names}
                      for g in ("weak", "control")},
           "secs": round(time.time() - t0, 1)}
    (OUTPUTS / f"{RUN_NAME}.results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUTPUTS / f'{RUN_NAME}.results.json'}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
