"""Build a slim local cache of Nemotron-Personas-USA for survey simulation.

Downloads a few parquet shards of `nvidia/Nemotron-Personas-USA` (the full set is
2.7 GB / 11 shards; a couple of shards is a representative sample for a 50-persona
panel and demographic filtering), keeps only the columns we condition on, and
writes a seeded, shuffled slim parquet to ``data/nemotron/personas_slim.parquet``.

Also prints the categorical value vocabularies and basic marginals so the
axis-mapping tables in ``simbench_exp.nemotron`` are built from the real data, not guesses.

Run once:

    uv run python scripts/prepare_nemotron.py            # 2 shards (~180k personas)
    uv run python scripts/prepare_nemotron.py --shards 3 # more coverage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from simbench_exp.config import DATA_DIR

REPO_ID = "nvidia/Nemotron-Personas-USA"
SLIM_COLS = [
    "uuid", "sex", "age", "marital_status", "education_level",
    "state", "city", "occupation", "persona",
    # richer narrative fields for the "rich conditioning" variant (Round 4)
    "professional_persona", "cultural_background",
    "hobbies_and_interests", "career_goals_and_ambitions",
]
OUT_PATH = DATA_DIR / "nemotron" / "personas_slim.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2, help="how many of 11 shards to pull")
    ap.add_argument("--sample", type=int, default=80_000,
                    help="rows to keep in the slim cache (seeded shuffle)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = []
    for i in range(args.shards):
        fname = f"data/train-{i:05d}-of-00011.parquet"
        print(f"downloading {fname} ...", flush=True)
        local = hf_hub_download(REPO_ID, fname, repo_type="dataset")
        frames.append(pd.read_parquet(local, columns=SLIM_COLS))
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df):,} personas from {args.shards} shard(s)")

    # Seeded shuffle + subsample → a representative slim panel source.
    df = df.sample(n=min(args.sample, len(df)), random_state=args.seed).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df):,} rows -> {OUT_PATH}")

    # Vocabularies + marginals to drive the axis maps.
    print("\n=== categorical vocabularies ===")
    for col in ("sex", "marital_status", "education_level"):
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())
    print("\nage: min/median/max =",
          int(df["age"].min()), int(df["age"].median()), int(df["age"].max()))
    print("\nstate (top 12):")
    print(df["state"].value_counts().head(12).to_string())
    print("n distinct states:", df["state"].nunique())
    print("\nsample persona text (first row, truncated):")
    print(repr(df["persona"].iloc[0][:400]))


if __name__ == "__main__":
    main()
