---
# Scrye_Project-m8kw
title: Dev/val/test split engine + experiments notebook
status: in-progress
type: feature
priority: normal
created_at: 2026-06-15T05:20:48Z
updated_at: 2026-06-15T05:30:48Z
---

Leave-family-out split engine (fit/dev/test, required questions pinned to test) + a second notebook that runs experiments comparing simulation systems and models, producing topline numbers.

## Progress
- [x] splits.py: leave-family-out engine (threshold-bucketed, add-stable, required-Qs pinned to test)
- [x] test_splits.py: 16 tests, family-disjointness/determinism/pinning verified
- [x] Verified on real SimBench (13,510 recs): dev 49.6% / val 25.8% / test 24.6% of families; 259 required-Q records all in test
- [x] experiment.py: model+system swappable compare()/model_sweep() harness (+6 tests)
- [x] 01_experiments.ipynb: topline comparison notebook (verified end-to-end)

## Summary of Changes
Leave-family-out split engine + experiment harness, both swappable-by-design.

- **splits.py** — `make_split()` partitions by question family `(dataset_name, input_template)` so every grouped segment variant of an item stays in one bucket (no leakage). Threshold-bucketed on a seeded SHA-256 hash: a family bucket is a pure function of its key+seed, so the split is deterministic and stable under add/remove. Required questions pinned to test. Default dev .50 / val .25 / test .25; `unit=dataset` for the stronger generalization split.
- **experiment.py** — `build_pipeline(model=, predictor=, calibrator=)`, `compare(systems, records)` -> topline table (mean S + bootstrap CI + counterfactual alignment; per-system frames in .attrs), `model_sweep(models, ...)` for the portability ablation. Model and simulation-system are both first-class swappable variables; new predictors register in PREDICTOR_REGISTRY.
- **viz.plot_system_comparison** — ranked bar chart with CIs.
- **notebooks/01_experiments.ipynb** — the experiments bench: build split, set MODEL + SYSTEMS, compare, sweep models, drill into any system with the existing viz suite. test stays sealed.

Verified on real SimBench (13,510 recs): split dev 6151 / val 3767 / test 3592 recs (259 required-Q records all in test). Notebook runs end-to-end from the shared cache: zero-shot @ gemini-flash-lite S=17.3 (CI 4.8-29.6) vs uniform floor; model sweep shows qwen-7b S=-1.7 (raw zero-shot tracks capability). Full suite 60 passed, 1 skipped (network).
