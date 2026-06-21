"""Stage 09 — feature-predicted target entropy on anti_flattening (dev).

Stage 08 showed the spread headroom isn't recoverable from the prediction's own
entropy. Here the target entropy is predicted from QUESTION FEATURES (dataset,
option count, conditioning, prediction shape) — which predict truth entropy at
out-of-sample R² ~ 0.58 — then each prediction is tempered to it. Calibrators are
fit on the disjoint dev-fit sample and scored on dev-eval (same samples as Stage 08).

Usage:
    uv run python scripts/run_feature_entropy.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scrye.calibrate import (
    EntropyTargetCalibrator,
    FeatureEntropyTargetCalibrator,
    temper_to_entropy,
)
from scrye.data import load_all
from scrye.evaluate import build_normalizers, stratified_sample
from scrye.experiment import make_client
from scrye.predict import ZeroShotPredictor
from scrye.scoring import bootstrap_ci, response_entropy, simbench_score
from scrye.splits import make_split

RUN_NAME = "2026-06-21-feature-entropy"
MODEL, STRATEGY = "gemini-3.1-flash-lite", "anti_flattening"
EVAL_SEED, FIT_SEED = 42, 7
N_EVAL_G, N_EVAL_P, N_FIT_G, N_FIT_P = 700, 300, 500, 250
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def raw_preds(predictor, records, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(predictor.predict, records))


def grouped_stats(records, preds, normalizers):
    g, p = [], []
    for rec, pred in zip(records, preds):
        z = normalizers.get((rec.split, rec.dataset_name))
        s = simbench_score(pred, rec.human_answer, options=list(rec.options), normalizer=z)
        (g if rec.split == "grouped" else p).append(s)
    gm, glo, ghi = bootstrap_ci(g)
    return {"pooled": round(float(np.mean(g + p)), 2), "grouped": round(gm, 2),
            "g_lo": round(glo, 2), "g_hi": round(ghi, 2),
            "g_hw": round((ghi - glo) / 2, 2),
            "pop": round(float(np.mean(p)), 2) if p else None}


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    normalizers = build_normalizers(split.dev + split.val + split.test)
    dev_g = [r for r in split.dev if r.split == "grouped"]
    dev_p = [r for r in split.dev if r.split == "pop"]

    eval_recs = stratified_sample(dev_g, N_EVAL_G, seed=EVAL_SEED) + \
                stratified_sample(dev_p, N_EVAL_P, seed=EVAL_SEED)
    eval_ids = {id(r) for r in eval_recs}
    fit_recs = stratified_sample([r for r in dev_g if id(r) not in eval_ids], N_FIT_G, seed=FIT_SEED) + \
               stratified_sample([r for r in dev_p if id(r) not in eval_ids], N_FIT_P, seed=FIT_SEED)
    assert not (eval_ids & {id(r) for r in fit_recs})
    print(f"[feat] eval n={len(eval_recs)}; fit n={len(fit_recs)} (disjoint)")

    predictor = ZeroShotPredictor(make_client(MODEL, max_retries=10, timeout=90),
                                  name=STRATEGY, strategy=STRATEGY)
    fit_raw, eval_raw = raw_preds(predictor, fit_recs), raw_preds(predictor, eval_recs)

    own = EntropyTargetCalibrator().fit(fit_recs, fit_raw)
    feat_full = FeatureEntropyTargetCalibrator().fit(fit_recs, fit_raw)
    feat_meta = FeatureEntropyTargetCalibrator(use_pred_shape=False).fit(fit_recs, fit_raw)
    feat_shape = FeatureEntropyTargetCalibrator(use_dataset=False).fit(fit_recs, fit_raw)

    # feature-model out-of-sample R^2 on eval truths (full model)
    ytrue = np.array([response_entropy(r.human_answer, True) for r in eval_recs])
    ypred = np.array([feat_full._predict_target(r, p) for r, p in zip(eval_recs, eval_raw)])
    r2 = 1 - np.sum((ytrue - ypred) ** 2) / np.sum((ytrue - ytrue.mean()) ** 2)
    print(f"[feat] gains: full={feat_full.gain} meta={feat_meta.gain} shape={feat_shape.gain} "
          f"own={own.gain}; feature-model eval R2={r2:.3f}")

    systems = {
        "identity": eval_raw,
        "own_entropy": [own.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        "feat_full": [feat_full.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        "feat_meta_only": [feat_meta.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        "feat_shape_only": [feat_shape.transform(r, p) for r, p in zip(eval_recs, eval_raw)],
        "entropy_oracle": [temper_to_entropy(p, response_entropy(r.human_answer))
                           for r, p in zip(eval_recs, eval_raw)],
    }
    rows = []
    for name, preds in systems.items():
        st = grouped_stats(eval_recs, preds, normalizers); st["system"] = name
        rows.append(st)
    tbl = pd.DataFrame(rows).set_index("system")[["pooled", "grouped", "g_lo", "g_hi", "g_hw", "pop"]]
    print("\n" + tbl.to_string() + "\n")

    base = tbl.loc["identity", "grouped"]; oracle = tbl.loc["entropy_oracle", "grouped"]
    print(f"identity grouped={base}; oracle={oracle} (+{oracle-base:.1f})")
    for name in ("own_entropy", "feat_full", "feat_meta_only", "feat_shape_only"):
        g = tbl.loc[name, "grouped"]
        floor = max(tbl.loc[name, "g_hw"], tbl.loc["identity", "g_hw"])
        captured = (g - base) / (oracle - base) if oracle > base else float("nan")
        verdict = "BEATS id" if g - base > floor else "~id"
        print(f"  {name:16s} grouped={g:5.1f}  vs id {g-base:+.1f} ({verdict})  "
              f"captured {captured:.0%} of oracle")

    elapsed = time.time() - t0
    tbl.to_csv(OUTPUTS / f"{RUN_NAME}.topline.csv")
    meta = {"run_name": RUN_NAME, "model": MODEL, "strategy": STRATEGY,
            "n_eval": len(eval_recs), "n_fit": len(fit_recs),
            "feature_model_eval_r2": round(float(r2), 3),
            "gains": {"full": feat_full.gain, "meta_only": feat_meta.gain,
                      "shape_only": feat_shape.gain, "own": own.gain},
            "elapsed_sec": round(elapsed, 1)}
    (OUTPUTS / f"{RUN_NAME}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[feat] done in {elapsed/60:.1f} min; saved outputs/runs/{RUN_NAME}.*")


if __name__ == "__main__":
    main()
