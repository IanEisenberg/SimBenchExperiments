"""Stage 08 — entropy de-compression recalibration on anti_flattening (dev).

Predictions are entropy-compressed (pred_H ~ 0.46 + 0.48*truth_H); the oracle
ceiling for fixing the spread is ~+8 grouped. This stage fits post-hoc
calibrators on a DISJOINT dev-fit sample and evaluates on the standard dev-eval
sample. anti_flattening predictions are cached/cheap; calibrators are CPU.

Usage:
    uv run python scripts/run_entropy_recalib.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from simbench_exp.calibrate import (
    EntropyTargetCalibrator,
    EntropyTempScaling,
    TempScaling,
    temper_to_entropy,
)
from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers, stratified_sample
from simbench_exp.experiment import make_client
from simbench_exp.predict import ZeroShotPredictor
from simbench_exp.scoring import bootstrap_ci, response_entropy, simbench_score
from simbench_exp.splits import make_split

RUN_NAME = "2026-06-21-entropy-recalib"
MODEL = "gemini-3.1-flash-lite"
STRATEGY = "anti_flattening"
EVAL_SEED, FIT_SEED = 42, 7
N_EVAL_G, N_EVAL_P = 700, 300
N_FIT_G, N_FIT_P = 500, 250
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def raw_preds(predictor, records, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(predictor.predict, records))


def scored_frame(records, preds, normalizers):
    rows = []
    for rec, pred in zip(records, preds):
        z = normalizers.get((rec.split, rec.dataset_name))
        rows.append({"split": rec.split, "dataset": rec.dataset_name,
                     "score": simbench_score(pred, rec.human_answer,
                                             options=list(rec.options), normalizer=z)})
    return pd.DataFrame(rows)


def grouped_stats(df):
    g = df.loc[df["split"] == "grouped", "score"].tolist()
    p = df.loc[df["split"] == "pop", "score"].tolist()
    gm, glo, ghi = bootstrap_ci(g)
    return {"grouped": round(gm, 2), "g_lo": round(glo, 2), "g_hi": round(ghi, 2),
            "g_hw": round((ghi - glo) / 2, 2),
            "pop": round(float(np.mean(p)), 2) if p else None,
            "pooled": round(float(np.mean(g + p)), 2)}


def main():
    t0 = time.time()
    allr = load_all()
    full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    normalizers = build_normalizers(split.dev + split.val + split.test)
    dev_g = [r for r in split.dev if r.split == "grouped"]
    dev_p = [r for r in split.dev if r.split == "pop"]

    # eval = standard sample; fit = DISJOINT dev sample (remove eval objects first)
    eval_recs = stratified_sample(dev_g, N_EVAL_G, seed=EVAL_SEED) + \
                stratified_sample(dev_p, N_EVAL_P, seed=EVAL_SEED)
    eval_ids = {id(r) for r in eval_recs}
    fit_g = [r for r in dev_g if id(r) not in eval_ids]
    fit_p = [r for r in dev_p if id(r) not in eval_ids]
    fit_recs = stratified_sample(fit_g, N_FIT_G, seed=FIT_SEED) + \
               stratified_sample(fit_p, N_FIT_P, seed=FIT_SEED)
    assert not (eval_ids & {id(r) for r in fit_recs}), "fit/eval overlap!"
    print(f"[recalib] eval n={len(eval_recs)} (g={N_EVAL_G},p={N_EVAL_P}); "
          f"fit n={len(fit_recs)} (disjoint)")

    predictor = ZeroShotPredictor(make_client(MODEL, max_retries=10, timeout=90),
                                  name=STRATEGY, strategy=STRATEGY)
    print("[recalib] getting raw anti_flattening predictions (cached/live) …")
    fit_raw = raw_preds(predictor, fit_recs)
    eval_raw = raw_preds(predictor, eval_recs)

    # --- fitted calibrators (fit on fit set) ---
    temp = TempScaling().fit(fit_recs, fit_raw)
    etemp = EntropyTempScaling().fit(fit_recs, fit_raw)
    etarget = EntropyTargetCalibrator().fit(fit_recs, fit_raw)
    print(f"[recalib] fitted: global_temp T={temp.T}; entropy_temp slope={etemp.slope}; "
          f"entropy_target gain={etarget.gain} (truth_H≈{etarget.intercept_:.2f}+"
          f"{etarget.slope_:.2f}·pred_H)")

    systems = {
        "identity": eval_raw,
        "global_temp": [temp.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        "entropy_temp": [etemp.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        "entropy_target": [etarget.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        # reference only — uses the truth entropy (not deployable)
        "entropy_oracle": [temper_to_entropy(p, response_entropy(r.human_answer))
                           for r, p in zip(eval_recs, eval_raw)],
    }

    rows = []
    for name, preds in systems.items():
        st = grouped_stats(scored_frame(eval_recs, preds, normalizers))
        st["system"] = name
        rows.append(st)
    tbl = pd.DataFrame(rows).set_index("system")[
        ["pooled", "grouped", "g_lo", "g_hi", "g_hw", "pop"]]
    print("\n" + tbl.to_string() + "\n")

    base = tbl.loc["identity", "grouped"]
    gt = tbl.loc["global_temp", "grouped"]
    oracle = tbl.loc["entropy_oracle", "grouped"]
    print(f"identity grouped = {base};  global_temp = {gt};  oracle ceiling = {oracle} "
          f"(+{oracle - base:.1f})")
    for name in ("global_temp", "entropy_temp", "entropy_target"):
        g = tbl.loc[name, "grouped"]
        floor = max(tbl.loc[name, "g_hw"], tbl.loc["identity", "g_hw"])
        captured = (g - base) / (oracle - base) if oracle > base else float("nan")
        beats_id = "BEATS id" if g - base > floor else "~id"
        beats_gt = "BEATS global" if g - gt > floor else "~global"
        print(f"  {name:16s} grouped={g:5.1f}  vs id {g-base:+.1f} ({beats_id})  "
              f"vs global {g-gt:+.1f} ({beats_gt})  captured {captured:.0%} of oracle")

    elapsed = time.time() - t0
    tbl.to_csv(OUTPUTS / f"{RUN_NAME}.topline.csv")
    meta = {"run_name": RUN_NAME, "model": MODEL, "strategy": STRATEGY,
            "n_eval": len(eval_recs), "n_fit": len(fit_recs),
            "eval_seed": EVAL_SEED, "fit_seed": FIT_SEED,
            "fitted": {"global_temp_T": temp.T, "entropy_temp_slope": etemp.slope,
                       "entropy_target_gain": etarget.gain,
                       "reg_slope": float(etarget.slope_), "reg_intercept": float(etarget.intercept_)},
            "elapsed_sec": round(elapsed, 1)}
    (OUTPUTS / f"{RUN_NAME}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[recalib] done in {elapsed/60:.1f} min; saved outputs/runs/{RUN_NAME}.*")


if __name__ == "__main__":
    main()
