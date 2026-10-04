"""Stage 06 — calibration sweep on anti_flattening @ gemini-3.1-flash-lite.

Fixed hyperparameter grid over TempScaling, EntropyTempScaling, DirichletCalibrator.
All LLM predictions are cached from Stage 04; this is pure CPU post-processing.

Usage:
    uv run python scripts/run_calibration.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import simbench_exp
from simbench_exp.calibrate import DirichletCalibrator, EntropyTempScaling, TempScaling
from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers, stratified_sample
from simbench_exp.experiment import build_pipeline, compare

RUN_NAME = "2026-06-21-calibration"
SEED = 42
N_GROUPED = 700
N_POP = 300
MODEL = "gemini-3.1-flash-lite"
STRATEGY = "anti_flattening"

OUTPUTS = Path("outputs/runs")
OUTPUTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    print(f"[calibration] loading data …")
    all_records = load_all()
    full = all_records["grouped"] + all_records["pop"]

    # Build normalizers from the full dev split (Eq. 2)
    # We use all records across all splits as in prior stages.
    # The splits module gives us the proper full-split view via make_split.
    from simbench_exp.splits import make_split
    split = make_split(full, seed=0, unit="question")
    full_split_records = split.dev + split.val + split.test
    normalizers = build_normalizers(full_split_records)
    print(f"[calibration] normalizers built from {len(full_split_records)} records")

    # Subsample: separate grouped + pop stratification
    dev_grouped = [r for r in split.dev if r.split == "grouped"]
    dev_pop = [r for r in split.dev if r.split == "pop"]
    sample_grouped = stratified_sample(dev_grouped, N_GROUPED, seed=SEED)
    sample_pop = stratified_sample(dev_pop, N_POP, seed=SEED)
    records = sample_grouped + sample_pop
    print(f"[calibration] subsample: {len(records)} total "
          f"(grouped={len(sample_grouped)}, pop={len(sample_pop)})")

    # Build pipelines: identity + all calibrator configs
    systems: dict[str, simbench_exp.pipeline.Pipeline] = {}

    def af(calibrator=None):
        return build_pipeline(model=MODEL, predictor=STRATEGY, calibrator=calibrator)

    # Identity (control)
    systems["identity"] = af()

    # TempScaling
    for T in [0.7, 1.25, 1.5, 2.0, 3.0]:
        systems[f"temp_T{T}"] = af(TempScaling(T=T))

    # EntropyTempScaling
    for slope in [0.5, 1.0, 2.0, 3.0]:
        systems[f"entropy_slope{slope}"] = af(EntropyTempScaling(slope=slope))

    # DirichletCalibrator
    for alpha in [0.01, 0.05, 0.1, 0.2]:
        systems[f"dirichlet_a{alpha}"] = af(DirichletCalibrator(alpha=alpha))

    print(f"[calibration] running {len(systems)} configurations …")
    df = compare(systems, records, normalizers=normalizers, max_workers=16, progress=True)
    elapsed = time.time() - t0
    print(f"\n[calibration] done in {elapsed/60:.1f} min\n")

    # --- report ---
    from simbench_exp.scoring import bootstrap_ci

    all_results = df.attrs.get("results", {})
    rows = []
    for _, row in df.iterrows():
        system_name = row["system"]
        res = all_results.get(system_name)
        grouped_score = grouped_ci_lo = grouped_ci_hi = pop_score = None
        if res is not None:
            g_scores = res.loc[res["split"] == "grouped", "score"].tolist()
            p_scores = res.loc[res["split"] == "pop", "score"].tolist()
            if g_scores:
                gm, glo, ghi = bootstrap_ci(g_scores)
                grouped_score, grouped_ci_lo, grouped_ci_hi = round(gm, 2), round(glo, 2), round(ghi, 2)
            if p_scores:
                pop_score = round(float(sum(p_scores) / len(p_scores)), 2)
        rows.append({
            "system": system_name,
            "pooled": round(row["mean_score"], 2),
            "grouped": grouped_score,
            "grouped_ci": f"[{grouped_ci_lo}, {grouped_ci_hi}]" if grouped_ci_lo is not None else None,
            "pop": pop_score,
            "frac_below_uniform": round(row.get("frac_below_uniform", float("nan")), 3),
        })

    topline = pd.DataFrame(rows).set_index("system")
    print(topline.to_string())
    print()

    # Save topline CSV
    topline.to_csv(OUTPUTS / f"{RUN_NAME}.topline.csv")

    # Save meta
    meta = {
        "run_name": RUN_NAME,
        "model": MODEL,
        "strategy": STRATEGY,
        "n_grouped": len(sample_grouped),
        "n_pop": len(sample_pop),
        "n_total": len(records),
        "seed": SEED,
        "elapsed_sec": round(elapsed, 1),
        "n_configs": len(systems),
    }
    (OUTPUTS / f"{RUN_NAME}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[calibration] saved to outputs/runs/{RUN_NAME}.*")


if __name__ == "__main__":
    main()
