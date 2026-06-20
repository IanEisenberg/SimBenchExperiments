"""Persona factory demo / smoke.

Offline (no API key needed):
  * prints the country catalog,
  * renders every conditioning strategy for a sample grouped record,
  * builds the per-subset demographic distribution table and prints a summary.

Online (if OPENROUTER_API_KEY is set): runs a small ablation comparing each
conditioning strategy and a post-stratification predictor against the SimBench
score on a stratified subsample.

Usage:
    PYTHONPATH=src python scripts/persona_demo.py [--n 60] [--model MODEL] [--live]
"""

from __future__ import annotations

import argparse
from collections import Counter

from scrye.config import DEFAULT_MODEL
from scrye.data import load_split
from scrye.distributions import SegmentWeights, distribution_table
from scrye.persona import ALL_COUNTRIES, COUNTRIES, STRATEGIES, get_strategy


def _print_catalog() -> None:
    print(f"\n=== Country catalog ({len(ALL_COUNTRIES)} distinct) ===")
    for dataset, countries in COUNTRIES.items():
        print(f"  {dataset:16s} {len(countries):3d} countries")


def _sample_record(grouped):
    """A two-variable record (country × attribute) makes the richest example."""
    for rec in grouped:
        if len(rec.segment) == 2 and any(k in rec.segment for k in ("cntry", "country")):
            return rec
    return grouped[0]


def _print_strategies(rec) -> None:
    print("\n=== Conditioning strategies for one record ===")
    print(f"  dataset={rec.dataset_name} segment={rec.segment}")
    print(f"  group_prompt={rec.group_prompt!r}\n")
    for name in sorted(STRATEGIES):
        print(f"--- {name} ---")
        for msg in get_strategy(name).build_messages(rec):
            print(f"  [{msg['role']}] {msg['content']}")
        print()


def _print_distribution_table(grouped) -> None:
    rows = distribution_table(grouped)
    print(f"\n=== Distribution table: {len(rows)} cells ===")
    by_grouping = Counter(r.grouping_keys for r in rows)
    print("  cells per grouping-keys combo (top 10):")
    for keys, n in by_grouping.most_common(10):
        print(f"    {keys}: {n}")


def _run_live(grouped, n: int, model: str) -> None:
    from scrye.evaluate import build_normalizers, evaluate, stratified_sample, summarize
    from scrye.llm import LLMClient
    from scrye.pipeline import Pipeline
    from scrye.predict import PostStratificationPredictor, ZeroShotPredictor

    sample = stratified_sample(grouped, n=n)
    normalizers = build_normalizers(grouped)
    weights = SegmentWeights(grouped)
    client = LLMClient(model=model)

    print(f"\n=== Live ablation on {len(sample)} grouped records (model={model}) ===")
    base = ZeroShotPredictor(client, name="zero_shot", strategy="simbench_faithful")
    for strategy_name in sorted(STRATEGIES):
        predictor = ZeroShotPredictor(client, name=strategy_name, strategy=strategy_name)
        df = evaluate(Pipeline(predictor), sample, normalizers=normalizers, progress=False)
        s = summarize(df)
        print(f"  {strategy_name:22s} S={s['mean_score']:6.2f}  "
              f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]")

    post = PostStratificationPredictor(base, weights, name="post_strat")
    df = evaluate(Pipeline(post), sample, normalizers=normalizers, progress=False)
    s = summarize(df)
    print(f"  {'post_strat(faithful)':22s} S={s['mean_score']:6.2f}  "
          f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]  "
          f"(decomposed={post.n_decomposed}, fallback={post.n_fallback})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--live", action="store_true", help="run the LLM ablation")
    args = parser.parse_args()

    grouped = load_split("grouped")
    _print_catalog()
    _print_strategies(_sample_record(grouped))
    _print_distribution_table(grouped)

    if args.live:
        _run_live(grouped, n=args.n, model=args.model)
    else:
        print("\n(Pass --live with OPENROUTER_API_KEY set to run the ablation.)")


if __name__ == "__main__":
    main()
