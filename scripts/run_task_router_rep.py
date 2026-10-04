"""Stage 15 — task-kind routing evaluation (dev).

base cc  vs  routed (kind map)  vs  routed+abstain-floor  vs  per-dataset oracle.

Usage:
    uv run python scripts/run_task_router.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers
from simbench_exp.experiment import make_client
from simbench_exp.persona import TaskContextStrategy, build_item_corpus, get_strategy
from simbench_exp.predict import (
    RoutingPredictor,
    UniformPredictor,
    VotingEnsemblePredictor,
    ZeroShotPredictor,
)
from simbench_exp.scoring import bootstrap_ci, simbench_score
from simbench_exp.splits import make_split
from simbench_exp.taskkind import KIND_ROUTES, LLMTaskClassifier

MODEL = "gemini-3.1-flash-lite"
CAP = 40
OUT = Path("outputs/runs/2026-06-21-task-router-rep.results.json")


def uniform(rec):
    k = rec.num_options or len(rec.options) or 1
    return {o: 1.0 / k for o in rec.options}


def score(rec, pred, norm):
    return simbench_score(pred, rec.human_answer, options=list(rec.options),
                          normalizer=norm.get((rec.split, rec.dataset_name)))


def pooled(vals):
    a = np.array(vals)
    _, lo, hi = bootstrap_ci(a.tolist())
    return float(a.mean()), lo, hi


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)
    corpus = build_item_corpus(full)

    import random as _rnd
    recs = []
    for ds in sorted(set(r.dataset_name for r in split.dev)):
        pool = [r for r in split.dev if r.dataset_name == ds]
        # prefer distinct templates: one random rec per template, then fill
        bytmpl = {}
        for r in pool:
            bytmpl.setdefault(r.input_template, []).append(r)
        rng = _rnd.Random(0)
        reps = [rng.choice(v) for v in bytmpl.values()]
        rng.shuffle(reps)
        if len(reps) < CAP:  # top up from the rest to reach CAP
            rest = [r for r in pool if r not in reps]; rng.shuffle(rest)
            reps = reps + rest
        recs.extend(reps[:CAP])
    print(f"[router] {len(recs)} dev recs across "
          f"{len(set(r.dataset_name for r in recs))} datasets")

    client = make_client(MODEL, max_retries=10, timeout=90)
    base = ZeroShotPredictor(client, strategy=get_strategy("calibrated_commitment"))
    task_context = ZeroShotPredictor(client, strategy=TaskContextStrategy(corpus, mode="items", k=6))
    voting = VotingEnsemblePredictor(client, n_individuals=16, alpha=0.5)
    routes = {
        "task_context": task_context,
        "voting": voting,
        "abstain": UniformPredictor(),
        "base": base,
    }
    # map KIND -> intervention name -> predictor
    kind_to_pred = {kind: routes[name] for kind, name in KIND_ROUTES.items()}
    router = RoutingPredictor(LLMTaskClassifier(client), kind_to_pred, default=base)

    # predictions
    def run(pred, rs):
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(pred.predict, rs))

    base_pred = run(base, recs)
    ctx_pred = run(task_context, recs)        # for the per-dataset oracle
    routed_pred = run(router, recs)
    print("kind routing realized:", dict(router.kind_counts))

    base_sc = np.array([score(r, p, norm) for r, p in zip(recs, base_pred)])
    ctx_sc = np.array([score(r, p, norm) for r, p in zip(recs, ctx_pred)])
    routed_sc = np.array([score(r, p, norm) for r, p in zip(recs, routed_pred)])
    uni_sc = np.array([score(r, uniform(r), norm) for r in recs])

    # abstain-floor: datasets where routed < uniform on dev -> predict uniform
    ds_arr = np.array([r.dataset_name for r in recs])
    floor_sc = routed_sc.copy()
    abstained = []
    for ds in set(ds_arr):
        m = ds_arr == ds
        if routed_sc[m].mean() < uni_sc[m].mean():
            floor_sc[m] = uni_sc[m]
            abstained.append(ds)
    # per-dataset oracle: best of cc/ctx per dataset (uses dataset identity)
    oracle_sc = base_sc.copy()
    for ds in set(ds_arr):
        m = ds_arr == ds
        if ctx_sc[m].mean() > base_sc[m].mean():
            oracle_sc[m] = ctx_sc[m]

    splits = np.array([r.split for r in recs])
    def show(name, sc):
        ov = pooled(sc.tolist())
        gp = pooled(sc[splits == "grouped"].tolist())
        pp = pooled(sc[splits == "pop"].tolist())
        print(f"  {name:26s} overall {ov[0]:6.2f} [{ov[1]:.1f},{ov[2]:.1f}]   "
              f"grouped {gp[0]:6.2f}   pop {pp[0]:6.2f}")
        return {"overall": round(ov[0], 2), "overall_ci": [round(ov[1], 1), round(ov[2], 1)],
                "grouped": round(gp[0], 2), "pop": round(pp[0], 2)}

    print("\n=== systems (dev) ===")
    res = {"base": show("base (cc)", base_sc),
           "routed": show("routed", routed_sc),
           "routed_floor": show("routed + abstain-floor", floor_sc),
           "oracle": show("per-dataset oracle", oracle_sc)}
    # paired delta routed_floor vs base
    d = floor_sc - base_sc
    dm, dlo, dhi = pooled(d.tolist())
    print(f"\nrouted+floor − base: paired Δ={dm:+.2f}  95% CI [{dlo:+.2f},{dhi:+.2f}]")
    print(f"abstain-floor datasets: {sorted(abstained)}")

    out = {"systems": res,
           "routed_floor_vs_base_delta": round(dm, 2),
           "routed_floor_vs_base_ci": [round(dlo, 2), round(dhi, 2)],
           "kind_counts": dict(router.kind_counts),
           "abstained": sorted(abstained),
           "secs": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
