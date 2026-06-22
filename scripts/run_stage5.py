"""Stage 18 Round 5 (final) — worldview-enriched, ideology-calibrated electorate.

Pipeline: fixed rich panel -> infer a sampled worldview per persona -> calibrate
the panel's ideology mix to OpinionQA's real POLIDEOLOGY marginal -> answer +
weighted-average. Scores on OpinionQA pop dev vs the champion and the un-enriched
baseline, reports the representativeness check, the correlation with the champion,
and (if there is independent signal) an ensemble sweep.

    uv run python scripts/run_stage5.py --limit 20   # smoke
    uv run python scripts/run_stage5.py              # full (n=248)
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from scrye.config import DATA_DIR
from scrye.data import load_all
from scrye.evaluate import build_normalizers, evaluate, summarize
from scrye.experiment import make_client
from scrye.nemotron import PersonaBank, render_persona
from scrye.pipeline import Pipeline
from scrye.predict import EnrichedGroundedPredictor, EnsemblePredictor, GroundedAveragingPredictor, ZeroShotPredictor
from scrye.splits import make_split
from scrye.worldview import build_enriched_panel

MODEL, K = "gemini-3.1-flash-lite", 50


def _H(dist):
    ps = [p for p in dist.values() if p > 0]
    return -sum(p * math.log(p) for p in ps) / math.log(len(dist)) if len(dist) > 1 else 0.0


def _row(label, pipe, recs, norm, mw):
    res = evaluate(pipe, recs, normalizers=norm, max_workers=mw, progress=True)
    s = summarize(res)
    return res, {"system": label, "score": round(s["mean_score"], 2),
                 "ci_low": round(s["ci_low"], 2), "ci_high": round(s["ci_high"], 2),
                 "pred_H": round(float(res["pred"].map(_H).mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(DATA_DIR / "nemotron" / "personas_slim.parquet")
    recs = load_all()
    allr = recs["pop"] + recs["grouped"]
    norm = build_normalizers(allr)
    split = make_split(allr, seed=0, unit="question")
    target = [r for r in split.dev if r.dataset_name == "OpinionQA" and r.split == "pop"]
    if args.limit:
        target = target[: args.limit]
    client = make_client(MODEL, max_retries=10, timeout=90)

    # Fixed rich panel + worldview enrichment + ideology calibration.
    bank = PersonaBank(df, text_mode="rich")
    rich_texts = [render_persona(r, "rich") for r in bank.panel_rows(target[0], K, base_seed=0)]
    panel = build_enriched_panel(client, rich_texts, recs["grouped"], calibrate=True)

    print("=== representativeness check: panel ideology vs OpinionQA real marginal ===")
    for b in panel["target_marginal"]:
        print(f"  {b:18s} panel={panel['panel_marginal'][b]:.2f}  real={panel['target_marginal'][b]:.2f}")

    # Note: grounded_avg(summary)=46.3 is the cited baseline (Round 2); we skip
    # re-evaluating it here to avoid 12k redundant cached reads. The only heavy
    # (new-call) step is the enriched predictor; uncalib reuses its cached answers.
    cc = lambda: ZeroShotPredictor(client, name="calibrated_commitment", strategy="calibrated_commitment")
    systems = [
        ("calibrated_commitment", Pipeline(cc()), 8),
        ("enriched(calibrated)", Pipeline(EnrichedGroundedPredictor(client, panel["texts"], weights=panel["weights"], max_workers=4)), 1),
        ("enriched(uncalib)", Pipeline(EnrichedGroundedPredictor(client, panel["texts"], weights=None, max_workers=4)), 1),
    ]
    print(f"\n=== topline (OpinionQA pop dev, n={len(target)}, truth_H=0.692) ===")
    frames, rows = {}, []
    for label, pipe, mw in systems:
        res, row = _row(label, pipe, target, norm, mw)
        frames[label] = res
        rows.append(row)
    print(pd.DataFrame(rows).sort_values("score", ascending=False).to_string(index=False))

    # Independence from the champion + ensemble.
    cc_s = frames["calibrated_commitment"]["score"].to_numpy()
    en_s = frames["enriched(calibrated)"]["score"].to_numpy()
    print(f"\nper-record score correlation enriched(calibrated) vs champion: "
          f"r = {np.corrcoef(cc_s, en_s)[0, 1]:.3f}")

    enr = EnrichedGroundedPredictor(client, panel["texts"], weights=panel["weights"], max_workers=5)
    print("\n=== ensemble sweep  w*enriched + (1-w)*champion ===")
    best = (0.0, -1e9)
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ens = Pipeline(EnsemblePredictor([enr, cc()], weights=[w, 1 - w]))
        res = evaluate(ens, target, normalizers=norm, max_workers=2, progress=False)
        sc = summarize(res)["mean_score"]
        print(f"  w={w:.1f}  score={sc:6.2f}")
        if sc > best[1]:
            best = (w, sc)
    print(f"  best: w={best[0]:.1f} score={best[1]:.2f}  (champion alone = {rows and frames['calibrated_commitment']['score'].mean():.2f})")


if __name__ == "__main__":
    main()
