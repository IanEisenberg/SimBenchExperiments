"""Stage 19 — method portability to frontier Claude (equivalent-model paper comparison).

For each model, score three systems on a stratified DEV subsample (sealed test
untouched): faithful (paper baseline), calibrated_commitment, and cc+abstain
(AbstainCalibrator fit per-model on a disjoint stratified VAL subsample).

Asks: does the faithful->cc+abstain gain port to a frontier Claude model, and does
the resulting split-avg S clear the paper's best published single model
(Claude-3.7-Sonnet, 40.80)? Claude-3.7-Sonnet is retired on OpenRouter, so the
Sonnet-4 line (4.5, 4.6) stands in as the equivalent/stronger successor.

Usage:
    uv run python scripts/run_stage19_claude.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from simbench_exp.calibrate import AbstainCalibrator
from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers, stratified_sample
from simbench_exp.experiment import make_client
from simbench_exp.persona import get_strategy
from simbench_exp.predict import ZeroShotPredictor
from simbench_exp.scoring import bootstrap_ci, simbench_score
from simbench_exp.splits import make_split

RUN_NAME = "2026-06-25-stage19-claude"
MODELS = ["gemini-3.1-flash-lite", "claude-sonnet-4.5", "claude-sonnet-4.6"]
EVAL_N = 3000   # stratified dev subsample
FIT_N = 2000    # stratified val subsample (abstain fit; disjoint from eval)
PAPER_BEST = 40.80  # Claude-3.7-Sonnet, Hu et al. Table 1
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def preds_for(client, strategy_name, records, workers=8):
    """Raw per-record predictions for one strategy (None on a failed call)."""
    pred = ZeroShotPredictor(client, name=strategy_name, strategy=get_strategy(strategy_name))

    def safe(rec):
        try:
            return pred.predict(rec)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(safe, records))


def scores(records, preds, norm):
    """Per-record SimBench scores, split-tagged; skips failed/empty preds."""
    g, p = [], []
    for rec, pred in zip(records, preds):
        if not pred:
            continue
        z = norm.get((rec.split, rec.dataset_name))
        s = simbench_score(pred, rec.human_answer, options=list(rec.options), normalizer=z)
        (g if rec.split == "grouped" else p).append(s)
    return g, p


def summarize(records, preds, norm):
    g, p = scores(records, preds, norm)
    alls = g + p
    m, lo, hi = bootstrap_ci(alls)
    gm = float(np.mean(g)) if g else float("nan")
    pm = float(np.mean(p)) if p else float("nan")
    return {
        "overall": round(m, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
        "grouped": round(gm, 2), "pop": round(pm, 2),
        "split_avg": round((gm + pm) / 2, 2),
        "n": len(alls),
    }


def paired_delta(records, preds_a, preds_b, norm):
    """Bootstrap CI on per-record (b - a) where both preds exist (faithful->method)."""
    diffs = []
    for rec, pa, pb in zip(records, preds_a, preds_b):
        if not pa or not pb:
            continue
        z = norm.get((rec.split, rec.dataset_name))
        sa = simbench_score(pa, rec.human_answer, options=list(rec.options), normalizer=z)
        sb = simbench_score(pb, rec.human_answer, options=list(rec.options), normalizer=z)
        diffs.append(sb - sa)
    m, lo, hi = bootstrap_ci(diffs)
    return {"delta": round(m, 2), "d_lo": round(lo, 2), "d_hi": round(hi, 2), "n": len(diffs)}


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)

    eval_recs = stratified_sample(split.dev, EVAL_N, seed=0)
    fit_recs = stratified_sample(split.val, FIT_N, seed=0)
    print(f"[stage19] eval(dev) n={len(eval_recs)} "
          f"(grouped {sum(r.split=='grouped' for r in eval_recs)}, "
          f"pop {sum(r.split=='pop' for r in eval_recs)}) | "
          f"abstain-fit(val) n={len(fit_recs)} | test UNTOUCHED")

    rows, meta = [], {}
    for model in MODELS:
        print(f"\n[stage19] === {model} ===")
        client = make_client(model, max_retries=10, timeout=120)

        # 1) fit abstain-set on val (cc predictions), per model
        cc_fit = preds_for(client, "calibrated_commitment", fit_recs)
        fit_pairs = [(r, p) for r, p in zip(fit_recs, cc_fit) if p]
        abstain = AbstainCalibrator().fit([r for r, _ in fit_pairs], [p for _, p in fit_pairs])
        print(f"[stage19]   abstain-set: {sorted(abstain.datasets_)}")

        # 2) eval predictions
        faith = preds_for(client, "simbench_faithful", eval_recs)
        cc = preds_for(client, "calibrated_commitment", eval_recs)
        cc_abs = [abstain.transform(r, p) if p else None for r, p in zip(eval_recs, cc)]

        systems = {"faithful": faith, "calibrated_commitment": cc, "cc+abstain": cc_abs}
        for name, preds in systems.items():
            s = summarize(eval_recs, preds, norm); s["model"] = model; s["system"] = name
            rows.append(s)

        d = paired_delta(eval_recs, faith, cc_abs, norm)
        print(f"[stage19]   faithful -> cc+abstain delta: {d['delta']:+.2f} "
              f"[{d['d_lo']:+.2f}, {d['d_hi']:+.2f}]  (n={d['n']})")
        meta[model] = {"abstain_datasets": sorted(abstain.datasets_), "delta_faith_to_ccabs": d}
        try:
            print(f"[stage19]   usage: {client.usage}")
        except Exception:
            pass

    tbl = pd.DataFrame(rows)[
        ["model", "system", "overall", "ci_lo", "ci_hi", "grouped", "pop", "split_avg", "n"]
    ]
    print("\n=== Stage 19 topline (eval = stratified dev subsample) ===")
    print(tbl.to_string(index=False))
    print(f"\npaper best published single model (Claude-3.7-Sonnet): {PAPER_BEST}")

    tbl.to_csv(OUTPUTS / f"{RUN_NAME}.topline.csv", index=False)
    (OUTPUTS / f"{RUN_NAME}.meta.json").write_text(json.dumps({
        "models": MODELS, "eval_n": EVAL_N, "fit_n": FIT_N,
        "eval_basis": "stratified dev subsample (seed=0)",
        "fit_basis": "stratified val subsample (seed=0)",
        "paper_best_single_model": PAPER_BEST,
        "per_model": meta, "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\n[stage19] {time.time()-t0:.0f}s; saved outputs/runs/{RUN_NAME}.*")


if __name__ == "__main__":
    main()
