"""Stage 12 — val confirmation of calibrated_commitment + abstention.

Champion anti_flattening (cached val) vs challenger calibrated_commitment +
AbstainCalibrator (abstain-set fit on DEV, applied to val). 2nd val touch since
Stage 05; test untouched.

Usage:
    uv run python scripts/run_val_confirm_cc.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scrye.calibrate import AbstainCalibrator
from scrye.data import load_all
from scrye.evaluate import build_normalizers
from scrye.experiment import make_client
from scrye.persona import get_strategy
from scrye.predict import ZeroShotPredictor
from scrye.scoring import bootstrap_ci, simbench_score
from scrye.splits import make_split

RUN_NAME = "2026-06-21-val-confirm-cc"
MODEL = "gemini-3.1-flash-lite"
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


def preds_for(strategy_name, records, workers=8):
    pred = ZeroShotPredictor(make_client(MODEL, max_retries=10, timeout=90),
                             name=strategy_name, strategy=get_strategy(strategy_name))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(pred.predict, records))


def stats(records, preds, norm):
    g, p = [], []
    for rec, pred in zip(records, preds):
        z = norm.get((rec.split, rec.dataset_name))
        s = simbench_score(pred, rec.human_answer, options=list(rec.options), normalizer=z)
        (g if rec.split == "grouped" else p).append(s)
    gm, glo, ghi = bootstrap_ci(g)
    return {"grouped": round(gm, 2), "g_lo": round(glo, 2), "g_hi": round(ghi, 2),
            "g_hw": round((ghi - glo) / 2, 2),
            "pop": round(float(np.mean(p)), 2),
            "pooled": round(float(np.mean(g + p)), 2)}


def main():
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    norm = build_normalizers(split.dev + split.val + split.test)
    dev, val = split.dev, split.val
    print(f"[val-confirm] dev n={len(dev)}  val n={len(val)} "
          f"(grouped {sum(r.split=='grouped' for r in val)}, pop {sum(r.split=='pop' for r in val)})")

    # 1) fit abstain-set on full-dev calibrated_commitment predictions
    print("[val-confirm] calibrated_commitment on DEV (to fit abstain-set) …")
    cc_dev = preds_for("calibrated_commitment", dev)
    abstain = AbstainCalibrator().fit(dev, cc_dev)
    print(f"[val-confirm] abstain-set (fit on dev): {sorted(abstain.datasets_)}")

    # 2) val predictions
    print("[val-confirm] predictions on VAL …")
    anti_val = preds_for("anti_flattening", val)               # cached from Stage 05
    cc_val = preds_for("calibrated_commitment", val)           # challenger prompt
    cc_abs_val = [abstain.transform(r, p) for r, p in zip(val, cc_val)]

    systems = {"anti_flattening": anti_val, "calibrated_commitment": cc_val,
               "cc+abstain": cc_abs_val}
    rows = []
    for name, preds in systems.items():
        st = stats(val, preds, norm); st["system"] = name; rows.append(st)
    tbl = pd.DataFrame(rows).set_index("system")[["grouped", "g_lo", "g_hi", "g_hw", "pop", "pooled"]]
    print("\n=== VAL ===\n" + tbl.to_string())

    champ = tbl.loc["anti_flattening"]
    chal = tbl.loc["cc+abstain"]
    floor = max(chal["g_hw"], champ["g_hw"])
    dG = chal["grouped"] - champ["grouped"]
    dPop = chal["pop"] - champ["pop"]
    dPool = chal["pooled"] - champ["pooled"]
    print(f"\nchallenger cc+abstain vs champion anti_flattening (val):")
    print(f"  grouped {dG:+.2f} (floor {floor:.2f}) | pop {dPop:+.2f} | pooled {dPool:+.2f}")
    rule1 = dG >= 0
    rule2 = dPool > 0
    rule3 = dPop >= 0
    confirmed = rule1 and rule2 and rule3
    strong = dG > floor
    print(f"  rule1 grouped>=champ: {rule1} | rule2 pooled>champ: {rule2} | rule3 pop>=champ: {rule3}")
    print(f"  => {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}{' (STRONG, clears noise)' if confirmed and strong else ''}")

    tbl.to_csv(OUTPUTS / f"{RUN_NAME}.topline.csv")
    (OUTPUTS / f"{RUN_NAME}.meta.json").write_text(json.dumps({
        "abstain_datasets": sorted(abstain.datasets_),
        "deltas": {"grouped": round(dG, 2), "pop": round(dPop, 2), "pooled": round(dPool, 2),
                   "noise_floor": round(float(floor), 2)},
        "confirmed": bool(confirmed), "strong": bool(confirmed and strong),
        "elapsed_sec": round(time.time() - t0, 1)}, indent=2))
    print(f"\n[val-confirm] {time.time()-t0:.0f}s; saved outputs/runs/{RUN_NAME}.*")


if __name__ == "__main__":
    main()
