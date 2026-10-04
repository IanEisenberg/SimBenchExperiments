"""Stage 18 Round 3 — ensemble grounded_averaging x calibrated_commitment.

Both sub-systems are already cached on OpinionQA pop dev, so blending
``w*averaging + (1-w)*calibrated_commitment`` and scoring it costs no new LLM
calls. Sweeps w on dev and prints score + predicted entropy per weight.

    uv run python scripts/ensemble_sweep.py
"""

from __future__ import annotations

import math

import pandas as pd

from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers, evaluate, summarize
from simbench_exp.experiment import make_client
from simbench_exp.nemotron import default_bank
from simbench_exp.pipeline import Pipeline
from simbench_exp.predict import (
    EnsemblePredictor,
    GroundedAveragingPredictor,
    ZeroShotPredictor,
)
from simbench_exp.splits import make_split

MODEL = "gemini-3.1-flash-lite"
K = 50


def _norm_entropy(dist: dict) -> float:
    ps = [p for p in dist.values() if p > 0]
    k = len(dist)
    if k <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in ps) / math.log(k)


def main() -> None:
    recs = load_all()
    allrecs = recs["pop"] + recs["grouped"]
    norm = build_normalizers(allrecs)
    split = make_split(allrecs, seed=0, unit="question")
    target = [r for r in split.dev if r.dataset_name == "OpinionQA" and r.split == "pop"]

    client = make_client(MODEL, max_retries=10, timeout=90)
    bank = default_bank()
    avg = GroundedAveragingPredictor(client, bank=bank, k=K, max_workers=5)
    cc = ZeroShotPredictor(client, name="calibrated_commitment",
                           strategy="calibrated_commitment")

    truth_H = None
    rows = []
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # w on averaging, (1-w) on calibrated_commitment.
        ens = EnsemblePredictor([avg, cc], weights=[w, 1 - w])
        res = evaluate(Pipeline(ens), target, normalizers=norm, max_workers=2, progress=False)
        s = summarize(res)
        if truth_H is None:
            truth_H = round(float(res["truth_entropy"].mean()), 3)
        rows.append({"w_avg": w, "mean_score": round(s["mean_score"], 2),
                     "ci_low": round(s["ci_low"], 2), "ci_high": round(s["ci_high"], 2),
                     "pred_H": round(float(res["pred"].map(_norm_entropy).mean()), 3)})
    table = pd.DataFrame(rows)
    print(f"\nOpinionQA pop dev, n={len(target)}, truth_H={truth_H}")
    print("w_avg=0 -> pure calibrated_commitment;  w_avg=1 -> pure grounded_averaging\n")
    print(table.to_string(index=False))
    best = table.loc[table["mean_score"].idxmax()]
    print(f"\nbest blend: w_avg={best['w_avg']}  score={best['mean_score']} "
          f"[{best['ci_low']}, {best['ci_high']}]  (CC alone = "
          f"{table.loc[table['w_avg']==0.0,'mean_score'].iloc[0]})")


if __name__ == "__main__":
    main()
