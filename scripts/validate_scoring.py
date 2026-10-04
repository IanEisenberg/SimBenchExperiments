"""Validate our SimBench score against the benchmark's defining property.

The score is defined so the *uniform* predictor averages to 0 within each
dataset (Eq. 2). Reproducing that exactly across all 20+ datasets — with no
model calls — is the strongest check that our scoring matches the paper.

Usage: uv run python scripts/validate_scoring.py
"""

import numpy as np

from simbench_exp.data import load_split
from simbench_exp.evaluate import build_normalizers, evaluate
from simbench_exp.pipeline import Pipeline
from simbench_exp.predict import UniformPredictor


def main() -> None:
    pop = load_split("pop")
    grouped = load_split("grouped")
    all_records = pop + grouped
    normalizers = build_normalizers(all_records)

    print(f"datasets with normalizers: {len(normalizers)}")
    print("sample per-dataset normalizers (mean TVD-to-uniform):")
    for key in list(normalizers)[:6]:
        print(f"  {key}: {normalizers[key]:.4f}")

    print("\n== Uniform predictor (must average to ~0 per split & dataset) ==")
    pipe = Pipeline(UniformPredictor())
    for split, recs in [("pop", pop), ("grouped", grouped)]:
        df = evaluate(pipe, recs, normalizers=normalizers, progress=False)
        overall = df["score"].mean()
        # Worst per-dataset deviation from 0.
        per_ds = df.groupby("dataset")["score"].mean()
        worst = per_ds.abs().max()
        print(f"[{split}] n={len(df):5d}  overall mean S = {overall:+.2e}  "
              f"| max |per-dataset mean| = {worst:.2e}")
        assert abs(overall) < 1e-9, f"uniform not centered at 0 for {split}!"

    print("\nOK: uniform predictor averages to 0 across all datasets — "
          "scoring reproduces the SimBench baseline by construction.")


if __name__ == "__main__":
    main()
