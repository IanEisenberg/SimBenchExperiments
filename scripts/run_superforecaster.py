"""Stage 07 — superforecaster prompting strategies on dev (n=1000).

Compares the locked incumbent (anti_flattening, no CoT) and a generic-CoT
reference (diversity_elicitation) against three superforecasting strategies:
outside_view, entropy_first, superforecaster. All on gemini-3.1-flash-lite,
temperature=0. dev only — val/test untouched.

Same dev subsample as Stage 06 (700 grouped + 300 pop, seed=42), so the
anti_flattening control is directly comparable. Normalizers are Eq. 2 scalars
from the FULL split. The new CoT strategies make live LLM calls (cached after
the first run); anti_flattening re-runs live but is deterministic at T=0.

Usage:
    uv run python scripts/run_superforecaster.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import scrye  # noqa: F401  (ensures package import side effects)
from scrye.data import load_all
from scrye.evaluate import build_normalizers, stratified_sample
from scrye.experiment import build_pipeline, compare
from scrye.scoring import bootstrap_ci
from scrye.splits import make_split

RUN_NAME = "2026-06-21-superforecaster"
SEED = 42
N_GROUPED = 700
N_POP = 300
MODEL = "gemini-3.1-flash-lite"
MAX_WORKERS = 8          # throttled: gemini-3.1-flash-lite is rpm-capped; CoT calls are slow
MAX_RETRIES = 10
TIMEOUT = 90.0

# system label -> predictor/strategy registry name
SYSTEMS = {
    "anti_flattening": "anti_flattening",            # incumbent control (no CoT)
    "diversity_elicitation": "diversity_elicitation",  # generic-CoT reference
    "outside_view": "outside_view",                  # ablation: base-rate anchoring
    "entropy_first": "entropy_first",                # ablation: spread-first
    "superforecaster": "superforecaster",            # full pipeline
}

OUTPUTS = Path("outputs/runs")
OUTPUTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    print("[superforecaster] loading data …")
    all_records = load_all()
    full = all_records["grouped"] + all_records["pop"]

    split = make_split(full, seed=0, unit="question")
    full_split_records = split.dev + split.val + split.test
    normalizers = build_normalizers(full_split_records)
    print(f"[superforecaster] normalizers from {len(full_split_records)} records")

    dev_grouped = [r for r in split.dev if r.split == "grouped"]
    dev_pop = [r for r in split.dev if r.split == "pop"]
    sample_grouped = stratified_sample(dev_grouped, N_GROUPED, seed=SEED)
    sample_pop = stratified_sample(dev_pop, N_POP, seed=SEED)
    records = sample_grouped + sample_pop
    print(f"[superforecaster] subsample: {len(records)} "
          f"(grouped={len(sample_grouped)}, pop={len(sample_pop)})")

    systems = {
        label: build_pipeline(
            model=MODEL, predictor=name,
            max_retries=MAX_RETRIES, timeout=TIMEOUT,
        )
        for label, name in SYSTEMS.items()
    }

    print(f"[superforecaster] running {len(systems)} systems "
          f"(max_workers={MAX_WORKERS}) …")
    df = compare(systems, records, normalizers=normalizers,
                 max_workers=MAX_WORKERS, progress=True)
    elapsed = time.time() - t0
    print(f"\n[superforecaster] done in {elapsed/60:.1f} min\n")

    # --- topline: pooled / grouped (+CI) / pop, in preregistered order ---
    all_results = df.attrs.get("results", {})
    rows = []
    for label in SYSTEMS:  # stable, preregistered order
        row = df[df["system"] == label].iloc[0]
        res = all_results.get(label)
        grouped = grouped_lo = grouped_hi = pop = None
        if res is not None:
            g = res.loc[res["split"] == "grouped", "score"].tolist()
            p = res.loc[res["split"] == "pop", "score"].tolist()
            if g:
                gm, glo, ghi = bootstrap_ci(g)
                grouped, grouped_lo, grouped_hi = round(gm, 2), round(glo, 2), round(ghi, 2)
            if p:
                pop = round(float(sum(p) / len(p)), 2)
        rows.append({
            "system": label,
            "pooled": round(row["mean_score"], 2),
            "grouped": grouped,
            "grouped_ci": f"[{grouped_lo}, {grouped_hi}]" if grouped_lo is not None else None,
            "grouped_halfwidth": round((grouped_hi - grouped_lo) / 2, 2) if grouped_lo is not None else None,
            "pop": pop,
            "cf_alignment": round(row.get("cf_alignment", float("nan")), 3),
            "frac_below_uniform": round(row.get("frac_below_uniform", float("nan")), 3),
        })

    topline = pd.DataFrame(rows).set_index("system")
    print(topline.to_string())
    print()

    # vs incumbent: gap and whether it clears the noise floor (larger half-width)
    inc = topline.loc["anti_flattening"]
    print("vs anti_flattening (incumbent) on grouped:")
    for label in SYSTEMS:
        if label == "anti_flattening":
            continue
        r = topline.loc[label]
        if r["grouped"] is None or inc["grouped"] is None:
            continue
        gap = r["grouped"] - inc["grouped"]
        floor = max(r["grouped_halfwidth"] or 0, inc["grouped_halfwidth"] or 0)
        verdict = "BEATS" if gap > floor else ("ties" if gap > -floor else "loses")
        print(f"  {label:22s} gap={gap:+6.2f}  noise_floor={floor:.2f}  -> {verdict}")
    print()

    topline.to_csv(OUTPUTS / f"{RUN_NAME}.topline.csv")

    # per-system records for notebook plots (grouped_table reads this)
    results_payload = {
        label: json.loads(res.to_json(orient="records"))
        for label, res in all_results.items()
    }
    (OUTPUTS / f"{RUN_NAME}.results.json").write_text(json.dumps(results_payload))

    meta = {
        "run_name": RUN_NAME,
        "model": MODEL,
        "systems": list(SYSTEMS),
        "n_grouped": len(sample_grouped),
        "n_pop": len(sample_pop),
        "n_total": len(records),
        "seed": SEED,
        "max_workers": MAX_WORKERS,
        "elapsed_sec": round(elapsed, 1),
        "total_cost_usd": round(sum(
            getattr(getattr(p.predictor, "client", None), "usage", None).cost_usd
            for p in systems.values()
            if getattr(getattr(p.predictor, "client", None), "usage", None)
        ), 4),
    }
    (OUTPUTS / f"{RUN_NAME}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[superforecaster] saved outputs/runs/{RUN_NAME}.* "
          f"(cost ≈ ${meta['total_cost_usd']})")


if __name__ == "__main__":
    main()
