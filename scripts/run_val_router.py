"""Stage 16 — val confirmation of the task-kind router + mechanism analysis.

Champion (Stage 12: cc + abstain) vs Challenger (Stage 15: routed + floor) on
FULL val. Reports grouped/pop/pooled with paired CIs, the decision rule, and the
entropy / mode / decomposition / per-kind mechanism analysis.

Usage:
    uv run python scripts/run_val_router.py
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scrye.data import load_all
from scrye.decompose import decompose_error
from scrye.evaluate import build_normalizers
from scrye.experiment import make_client
from scrye.persona import TaskContextStrategy, build_item_corpus, get_strategy
from scrye.predict import (
    RoutingPredictor,
    UniformPredictor,
    VotingEnsemblePredictor,
    ZeroShotPredictor,
)
from scrye.scoring import bootstrap_ci, simbench_score
from scrye.splits import make_split
from scrye.taskkind import KIND_ROUTES, LLMTaskClassifier

MODEL = "gemini-3.1-flash-lite"
CHAMP_ABSTAIN = {"Choices13k", "MoralMachine", "OSPsychMACH"}  # Stage 12
FLOOR = {"OSPsychMACH"}                                        # Stage 15 dev-fit
OUT = Path("outputs/runs/2026-06-21-val-router.results.json")


def uniform(rec):
    k = rec.num_options or len(rec.options) or 1
    return {o: 1.0 / k for o in rec.options}


def score(rec, pred, norm):
    return simbench_score(pred, rec.human_answer, options=list(rec.options),
                          normalizer=norm.get((rec.split, rec.dataset_name)))


def pooled(vals):
    a = np.array(vals); _, lo, hi = bootstrap_ci(a.tolist())
    return float(a.mean()), lo, hi


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)
    corpus = build_item_corpus(full)
    val = split.val
    print(f"[val] {len(val)} recs (grouped {sum(r.split=='grouped' for r in val)}, "
          f"pop {sum(r.split=='pop' for r in val)})")

    client = make_client(MODEL, max_retries=10, timeout=90)
    clf = LLMTaskClassifier(client)
    cc = ZeroShotPredictor(client, strategy=get_strategy("calibrated_commitment"))
    task_context = ZeroShotPredictor(client, strategy=TaskContextStrategy(corpus, mode="items", k=6))
    voting = VotingEnsemblePredictor(client, n_individuals=16, alpha=0.5)
    routes = {"task_context": task_context, "voting": voting,
              "abstain": UniformPredictor(), "base": cc}
    router = RoutingPredictor(clf, {k: routes[v] for k, v in KIND_ROUTES.items()}, default=cc)

    def champion_pred(r):
        return uniform(r) if r.dataset_name in CHAMP_ABSTAIN else cc.predict(r)

    def challenger_pred(r):
        return uniform(r) if r.dataset_name in FLOOR else router.predict(r)

    with ThreadPoolExecutor(max_workers=8) as ex:
        champ = list(ex.map(champion_pred, val))
    with ThreadPoolExecutor(max_workers=8) as ex:
        chal = list(ex.map(challenger_pred, val))
    kinds = [clf(r) for r in val]   # cached; for the per-kind breakdown
    print("kind counts:", dict(Counter(kinds)))

    champ_sc = np.array([score(r, p, norm) for r, p in zip(val, champ)])
    chal_sc = np.array([score(r, p, norm) for r, p in zip(val, chal)])
    splits = np.array([r.split for r in val])

    def block(mask, label):
        c, h = champ_sc[mask], chal_sc[mask]
        cm, clo, chi = pooled(c.tolist()); hm, hlo, hhi = pooled(h.tolist())
        _, dlo, dhi = pooled((h - c).tolist())
        print(f"  {label:9s} champ {cm:6.2f} [{clo:.1f},{chi:.1f}]  "
              f"chal {hm:6.2f} [{hlo:.1f},{hhi:.1f}]  Δ={hm-cm:+.2f} [{dlo:+.2f},{dhi:+.2f}]")
        return {"champ": round(cm, 2), "chal": round(hm, 2),
                "delta": round(hm - cm, 2), "delta_ci": [round(dlo, 2), round(dhi, 2)]}

    print("\n=== VAL scores ===")
    res = {"overall": block(np.ones(len(val), bool), "overall"),
           "grouped": block(splits == "grouped", "grouped"),
           "pop": block(splits == "pop", "pop")}

    # decision rule
    ov, gp, pp = res["overall"], res["grouped"], res["pop"]
    rule1 = ov["delta_ci"][0] > 0           # pooled gain significant (CI lower > 0)
    rule2 = gp["delta_ci"][1] >= 0          # grouped not significantly negative (upper CI >= 0)
    rule3 = pp["delta_ci"][1] >= 0          # pop not significantly negative
    verdict = "CONFIRMED" if (rule1 and rule2 and rule3) else "NOT CONFIRMED"
    print(f"\nRule1 pooled CI>0: {rule1} | Rule2 grouped no-regress: {rule2} | "
          f"Rule3 pop no-regress: {rule3}  -> {verdict}")

    # ---- mechanism analysis ----
    dc = [decompose_error(p, r.human_answer, list(r.options)) for r, p in zip(val, champ)]
    dh = [decompose_error(p, r.human_answer, list(r.options)) for r, p in zip(val, chal)]

    def mean(dl, k): return float(np.mean([d[k] for d in dl]))
    print("\n=== MECHANISM (val, champion -> challenger) ===")
    mech = {}
    for k in ("pred_entropy", "truth_entropy", "concentration_err", "location_err",
              "mode_mass_err"):
        mech[k] = {"champ": round(mean(dc, k), 4), "chal": round(mean(dh, k), 4)}
    mech["abs_entropy_gap"] = {
        "champ": round(float(np.mean([abs(d["entropy_gap"]) for d in dc])), 4),
        "chal": round(float(np.mean([abs(d["entropy_gap"]) for d in dh])), 4)}
    mech["mode_match_rate"] = {
        "champ": round(float(np.mean([d["mode_match"] for d in dc])), 4),
        "chal": round(float(np.mean([d["mode_match"] for d in dh])), 4)}
    print(f"  truth entropy (target):        {mech['truth_entropy']['champ']:.3f}")
    print(f"  pred entropy   champ {mech['pred_entropy']['champ']:.3f} -> chal {mech['pred_entropy']['chal']:.3f}")
    print(f"  |entropy gap|  champ {mech['abs_entropy_gap']['champ']:.3f} -> chal {mech['abs_entropy_gap']['chal']:.3f}")
    print(f"  mode-match     champ {mech['mode_match_rate']['champ']:.3f} -> chal {mech['mode_match_rate']['chal']:.3f}")
    print(f"  concentration_err champ {mech['concentration_err']['champ']:.3f} -> chal {mech['concentration_err']['chal']:.3f}")
    print(f"  location_err      champ {mech['location_err']['champ']:.3f} -> chal {mech['location_err']['chal']:.3f}")

    # per-kind score
    print("\n=== per-kind (val) ===")
    by_kind = defaultdict(list)
    for i, kd in enumerate(kinds):
        by_kind[kd].append(i)
    perkind = {}
    for kd, idx in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        c = float(np.mean([champ_sc[i] for i in idx]))
        h = float(np.mean([chal_sc[i] for i in idx]))
        perkind[kd] = {"n": len(idx), "champ": round(c, 2), "chal": round(h, 2),
                       "delta": round(h - c, 2), "route": KIND_ROUTES[kd]}
        print(f"  {kd:18s} n={len(idx):4d}  champ={c:6.2f} -> chal={h:6.2f}  "
              f"Δ={h-c:+6.2f}  [{KIND_ROUTES[kd]}]")

    out = {"verdict": verdict, "scores": res,
           "rules": {"pooled_sig": rule1, "grouped_no_regress": rule2, "pop_no_regress": rule3},
           "mechanism": mech, "per_kind": perkind,
           "kind_counts": dict(Counter(kinds)), "secs": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n{verdict}  saved -> {OUT}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
