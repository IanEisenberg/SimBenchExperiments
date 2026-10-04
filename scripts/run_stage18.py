"""Stage 18 dev run — grounded Nemotron persona electorate vs naive LLM.

Round 1 (headline): whole-corpus panel on OpinionQA pop dev.
Round 2 (extension): demographically-filtered panel on OpinionQA grouped dev
(matchable axes grounded; unmatchable segments fall back to calibrated_commitment).

    uv run python scripts/run_stage18.py --round 1 --limit 4   # smoke
    uv run python scripts/run_stage18.py --round 1             # full headline
    uv run python scripts/run_stage18.py --round 2

Dev only. Results saved under outputs/runs/.
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from simbench_exp.config import OUTPUTS_DIR
from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers, counterfactual_sensitivity, evaluate, summarize
from simbench_exp.experiment import build_pipeline, make_client
from simbench_exp.nemotron import default_bank
from simbench_exp.pipeline import Pipeline
from simbench_exp.predict import GroundedAveragingPredictor, GroundedVotingPredictor, ZeroShotPredictor
from simbench_exp.splits import make_split


def _norm_entropy(dist: dict) -> float:
    """Shannon entropy of a distribution, normalized to [0,1] by log(k)."""
    ps = [p for p in dist.values() if p > 0]
    k = len(dist)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(k)

MODEL = "gemini-3.1-flash-lite"
K = 50


def _oq(records, split):
    return [r for r in records if r.dataset_name == "OpinionQA" and r.split == split]


def _row(label, pipe, recs, norm, max_workers):
    res = evaluate(pipe, recs, normalizers=norm, max_workers=max_workers, progress=True)
    s = summarize(res)
    align, _ = counterfactual_sensitivity(res)
    pred_H = float(res["pred"].map(_norm_entropy).mean())
    truth_H = float(res["truth_entropy"].mean())
    extra = {}
    pred = pipe.predictor
    if isinstance(pred, (GroundedVotingPredictor, GroundedAveragingPredictor)):
        extra = {"grounded": pred.n_grounded, "fallback": pred.n_fallback,
                 "parse_fail": pred.n_parse_failures}
    return res, {"system": label, "n": s["n"], "mean_score": round(s["mean_score"], 2),
                 "ci_low": round(s["ci_low"], 2), "ci_high": round(s["ci_high"], 2),
                 "frac_below_uniform": round(s["frac_below_uniform"], 3),
                 "pred_H": round(pred_H, 3), "truth_H": round(truth_H, 3),
                 "cf_align": round(align, 3), **extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2], default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap records for a smoke run")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    recs = load_all()
    allrecs = recs["pop"] + recs["grouped"]
    norm = build_normalizers(allrecs)
    split = make_split(allrecs, seed=0, unit="question")
    bank = default_bank()
    # OpenRouter caps this model at 300 rpm; keep total concurrency well under it
    # and lean on retries. The on-disk cache makes the (slow) first fill one-time.
    client = make_client(MODEL, max_retries=10, timeout=90)

    cc = lambda: ZeroShotPredictor(client, name="calibrated_commitment",
                                   strategy="calibrated_commitment")

    if args.round == 1:
        target = _oq(split.dev, "pop")
        if args.limit:
            target = target[: args.limit]
        voting = Pipeline(GroundedVotingPredictor(client, bank=bank, k=args.k, max_workers=5))
        averaging = Pipeline(GroundedAveragingPredictor(client, bank=bank, k=args.k, max_workers=5))
        systems = [
            ("uniform", build_pipeline(predictor="uniform")),
            ("simbench_faithful", build_pipeline(model=MODEL, predictor="simbench_faithful")),
            ("calibrated_commitment", Pipeline(cc())),
            (f"grounded_voting(K={args.k})", voting),
            (f"grounded_averaging(K={args.k})", averaging),
        ]
    else:
        target = _oq(split.dev, "grouped")
        if args.limit:
            target = target[: args.limit]
        grounded = Pipeline(GroundedVotingPredictor(
            client, bank=bank, k=args.k, max_workers=5, fallback=cc()))
        systems = [
            ("calibrated_commitment", Pipeline(cc())),
            (f"grounded_voting(K={args.k})", grounded),
        ]

    print(f"\nStage 18 Round {args.round}: OpinionQA "
          f"{'pop' if args.round==1 else 'grouped'} dev, n={len(target)}\n")

    rows, frames = [], {}
    for label, pipe in systems:
        # Grounded fans out K persona-calls internally (max_workers=5); keep its
        # record-level workers at 1 so total concurrency stays under the 300-rpm cap.
        mw = 1 if "grounded" in label else 8
        res, row = _row(label, pipe, target, norm, mw)
        rows.append(row)
        frames[label] = res
    table = pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)
    print("\n" + table.to_string(index=False))

    out = OUTPUTS_DIR / "runs" / f"2026-06-22-stage18-r{args.round}.results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
