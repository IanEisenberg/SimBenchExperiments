"""Stage 13 Round 2 — is the Choices13k voting win real out-of-sample?

Run voting_ensemble on FULL dev Choices13k (tighter CI) and validate a
leakage-safe per-dataset selector: on a dev-FIT half decide per dataset whether
voting beats uniform; apply that choice to the disjoint dev-EVAL half. Report
eval-half mean for the selector vs pure uniform-abstain. 2-fold (swap halves)
for robustness. MoralMachine/OSPsych included to confirm the selector routes
them to uniform.

Usage:
    uv run python scripts/run_voting_r2.py
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
from scrye.predict import VotingEnsemblePredictor
from scrye.scoring import bootstrap_ci, simbench_score
from scrye.splits import make_split

RUN_NAME = "2026-06-21-voting-r2"
MODEL = "gemini-3.1-flash-lite"
DATASETS = ["Choices13k", "MoralMachine", "OSPsychMACH"]
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

    recs = []
    for ds in DATASETS:
        pool = sorted([r for r in split.dev if r.dataset_name == ds],
                      key=lambda r: r.input_template)
        recs.extend(pool)
        print(f"[r2] {ds}: {len(pool)} dev recs")
    print(f"[r2] total {len(recs)} recs; {N_VOTES} votes each "
          f"≈ {len(recs) * N_VOTES} calls")

    client = make_client(MODEL, max_retries=10, timeout=90)
    ve = VotingEnsemblePredictor(client, n_individuals=N_VOTES, alpha=0.5)
    with ThreadPoolExecutor(max_workers=8) as ex:
        ve_preds = list(ex.map(ve.predict, recs))

    uni_sc = {ds: [] for ds in DATASETS}
    vote_sc = {ds: [] for ds in DATASETS}
    for r, vp in zip(recs, ve_preds):
        uni_sc[r.dataset_name].append(score(r, uniform(r), norm))
        vote_sc[r.dataset_name].append(score(r, vp, norm))

    # full-dev per-dataset table
    print("\n== full-dev per-dataset ==")
    rows = []
    for ds in DATASETS:
        u, v = uni_sc[ds], vote_sc[ds]
        _, ulo, uhi = bootstrap_ci(u)
        _, vlo, vhi = bootstrap_ci(v)
        rows.append({"dataset": ds, "n": len(u),
                     "uniform": round(float(np.mean(u)), 2),
                     "voting": round(float(np.mean(v)), 2),
                     "voting_ci": [round(vlo, 1), round(vhi, 1)]})
        print(f"{ds:14s} n={len(u):4d}  uniform={np.mean(u):7.2f}  "
              f"voting={np.mean(v):7.2f} [{vlo:6.1f},{vhi:6.1f}]")

    # 2-fold leakage-safe per-dataset selector (fit picks voting-vs-uniform; eval scores)
    def fold_eval(fit_mask):
        """Return eval-half score for selector and for pure-uniform, pooled over datasets."""
        sel, uni = [], []
        chosen = {}
        for ds in DATASETS:
            u, v = np.array(uni_sc[ds]), np.array(vote_sc[ds])
            idx = np.arange(len(u))
            fit = idx[fit_mask(len(u))]
            ev = idx[~fit_mask(len(u))]
            use_vote = v[fit].mean() > u[fit].mean()   # decision learned on FIT only
            chosen[ds] = use_vote
            sel.extend((v if use_vote else u)[ev].tolist())
            uni.extend(u[ev].tolist())
        return np.mean(sel), np.mean(uni), chosen

    even = lambda n: (np.arange(n) % 2 == 0)
    odd = lambda n: (np.arange(n) % 2 == 1)
    s1, u1, c1 = fold_eval(even)
    s2, u2, c2 = fold_eval(odd)
    sel_mean = (s1 + s2) / 2
    uni_mean = (u1 + u2) / 2
    print("\n== 2-fold selector (eval halves, pooled over 3 datasets) ==")
    print(f"  fold1 chose: {c1}")
    print(f"  fold2 chose: {c2}")
    print(f"  pure-uniform-abstain : {uni_mean:7.2f}")
    print(f"  selective (vote|uni) : {sel_mean:7.2f}   delta={sel_mean-uni_mean:+.2f}")

    out = {"run": RUN_NAME, "model": MODEL, "n_votes": N_VOTES,
           "per_dataset": rows,
           "selector": {"fold1": {k: bool(v) for k, v in c1.items()},
                        "fold2": {k: bool(v) for k, v in c2.items()},
                        "uniform_mean": round(float(uni_mean), 2),
                        "selective_mean": round(float(sel_mean), 2),
                        "delta": round(float(sel_mean - uni_mean), 2)},
           "parse_failures": ve.n_parse_failures,
           "secs": round(time.time() - t0, 1)}
    (OUTPUTS / f"{RUN_NAME}.results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUTPUTS / f'{RUN_NAME}.results.json'}  ({out['secs']}s)")


if __name__ == "__main__":
    main()
