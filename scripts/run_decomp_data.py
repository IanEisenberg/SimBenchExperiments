"""Generate per-record predictions for faithful vs final router on capped dev.

Saves outputs/runs/2026-06-21-decomp-final.results.json in the standard
{sys_name: [per-record dicts]} format consumed by decompose_records().

The final system (Stage 17):
    RoutingPredictor(LLMTaskClassifier, KIND_ROUTES)
    + AbstainCalibrator({"OSPsychMACH"})   <- floor fit on dev in Stage 15
    @ gemini-3.1-flash-lite

LLM calls should be cache-warm from stages 10-15 (calibrated_commitment,
task_context, task classification).  Expect ~1-2 min from cache.

Usage:
    uv run python scripts/run_decomp_data.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scrye.calibrate import AbstainCalibrator
from scrye.data import load_all
from scrye.evaluate import build_normalizers, evaluate
from scrye.experiment import make_client
from scrye.persona import TaskContextStrategy, build_item_corpus, get_strategy
from scrye.pipeline import Pipeline
from scrye.predict import RoutingPredictor, UniformPredictor, ZeroShotPredictor
from scrye.splits import make_split
from scrye.taskkind import KIND_ROUTES, LLMTaskClassifier

MODEL = "gemini-3.1-flash-lite"
CAP = 40   # records per dataset; matches Stage 15 subset
OUT = Path("outputs/runs/2026-06-21-decomp-final.results.json")


def capped_dev(split, cap=CAP):
    """Same capped subset used in Stage 15 (sorted by input_template, first cap)."""
    recs = []
    for ds in sorted(set(r.dataset_name for r in split.dev)):
        pool = sorted(
            [r for r in split.dev if r.dataset_name == ds],
            key=lambda r: r.input_template,
        )[:cap]
        recs.extend(pool)
    return recs


def df_to_records(df) -> list[dict]:
    """Convert evaluate() DataFrame to the {key: value} list format."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "dataset": row["dataset"],
            "split": row["split"],
            "n_options": int(row["n_options"]),
            "is_population": bool(row["is_population"]),
            "segment": row["segment"],
            "group_size": row["group_size"],
            "truth_entropy": float(row["truth_entropy"]),
            "score": float(row["score"]),
            "input_template": row["input_template"],
            "options": list(row["options"]),
            "pred": dict(row["pred"]),
            "truth": dict(row["truth"]),
        })
    return rows


def main():
    t0 = time.time()
    allr = load_all()
    full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    normalizers = build_normalizers(split.dev + split.val + split.test)
    recs = capped_dev(split)
    print(f"[decomp-data] {len(recs)} dev records, "
          f"{len(set(r.dataset_name for r in recs))} datasets")

    client = make_client(MODEL, max_retries=10, timeout=90)
    corpus = build_item_corpus(full)

    # -- Faithful (SimBench baseline) -----------------------------------------
    print("[decomp-data] running faithful ...")
    faithful_pred = ZeroShotPredictor(client, strategy=get_strategy("simbench_faithful"))
    faithful_pipe = Pipeline(faithful_pred)
    faithful_df = evaluate(faithful_pipe, recs, normalizers=normalizers, progress=True)
    print(f"  faithful  grouped={faithful_df[faithful_df['split']=='grouped']['score'].mean():.2f}"
          f"  pop={faithful_df[faithful_df['split']=='pop']['score'].mean():.2f}")

    # -- Final router (Stage 15-17 champion) -----------------------------------
    print("[decomp-data] running final router ...")
    base = ZeroShotPredictor(
        client, strategy=get_strategy("calibrated_commitment"), name="cc_base"
    )
    task_ctx = ZeroShotPredictor(
        client,
        strategy=TaskContextStrategy(corpus, mode="items", k=6),
        name="task_context",
    )
    kind_to_pred = {
        kind: {"task_context": task_ctx, "abstain": UniformPredictor(), "base": base}[name]
        for kind, name in KIND_ROUTES.items()
    }
    router = RoutingPredictor(LLMTaskClassifier(client), kind_to_pred, default=base,
                              name="router")
    # OSPsychMACH was the single dataset where router scored below uniform on dev
    # (determined in Stage 15; hardcoded here to reproduce the final system exactly)
    abstain = AbstainCalibrator(["OSPsychMACH"])
    router_pipe = Pipeline(router, abstain)
    router_df = evaluate(router_pipe, recs, normalizers=normalizers, progress=True)
    print(f"  router    grouped={router_df[router_df['split']=='grouped']['score'].mean():.2f}"
          f"  pop={router_df[router_df['split']=='pop']['score'].mean():.2f}")
    print(f"  kind routing: {dict(router.kind_counts)}")

    # -- Save -----------------------------------------------------------------
    out = {
        "faithful": df_to_records(faithful_df),
        "final_router": df_to_records(router_df),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    elapsed = round(time.time() - t0, 1)
    print(f"\n[decomp-data] saved {OUT}  "
          f"({len(out['faithful'])} records x 2 systems, {elapsed}s)")


if __name__ == "__main__":
    main()
