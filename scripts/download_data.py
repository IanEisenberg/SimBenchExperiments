"""Download the SimBench dataset and print a quick structural summary.

Usage:
    uv run python scripts/download_data.py
"""

from collections import Counter

from simbench_exp.data import download_simbench, load_split


def main() -> None:
    paths = download_simbench()
    print("Downloaded:")
    for split, path in paths.items():
        print(f"  {split:8s} -> {path}")

    for split in paths:
        records = load_split(split, download=False)
        by_dataset = Counter(r.dataset_name for r in records)
        opt_counts = Counter(r.num_options for r in records)
        n_pop = sum(1 for r in records if r.is_population)
        print(f"\n[{split}] {len(records)} records "
              f"({n_pop} population / {len(records) - n_pop} conditioned)")
        print(f"  source datasets ({len(by_dataset)}): "
              + ", ".join(f"{k}={v}" for k, v in by_dataset.most_common()))
        print(f"  option-count distribution: {dict(sorted(opt_counts.items()))}")
        # Show a conditioned example if one exists, else the first record.
        sample = next((r for r in records if not r.is_population), records[0])
        print("  example record:")
        print(f"    dataset       = {sample.dataset_name}")
        print(f"    group_prompt  = {sample.group_prompt[:90]!r}")
        print(f"    segment       = {sample.segment}")
        print(f"    grouping_keys = {sample.grouping_keys}")
        print(f"    options       = {sample.options}")
        print(f"    answer_options= {dict(list(sample.answer_options.items())[:3])}")
        print(f"    human_answer  = { {k: round(v, 3) for k, v in sample.human_answer.items()} }")
        print(f"    group_size    = {sample.group_size}")


if __name__ == "__main__":
    main()
