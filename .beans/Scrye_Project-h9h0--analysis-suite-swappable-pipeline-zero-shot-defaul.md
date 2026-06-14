---
# Scrye_Project-h9h0
title: Analysis suite + swappable pipeline (zero-shot default)
status: completed
type: feature
priority: high
created_at: 2026-06-14T16:29:49Z
updated_at: 2026-06-14T16:40:38Z
---

Build a composable prediction pipeline (Predictor -> Calibrator) with the zero-shot verbalized predictor as the default, plus an evaluation harness and visualization suite. Notebook must run top-to-bottom and show the default baseline.

- [x] Predictor interface + ZeroShotPredictor (robust verbalized-distribution baseline)
- [x] Calibrator interface + IdentityCalibrator (no-op default)
- [x] Pipeline composing predictor+calibrator (threaded batch, cached)
- [x] scoring: response entropy + delta-alignment + tvd_to_uniform; FIXED to Eq.2
- [x] evaluate.py: results DataFrame, summaries+CIs, sampling, counterfactual sensitivity, build_normalizers
- [x] viz.py: histogram, by-dataset, vs-entropy, by-n-options, calibration scatter, examples
- [x] config: REQUIRED_QUESTIONS matchers
- [x] Rewrite notebook to use the suite with clear component-swap seam
- [x] Tests (38 pass) + cache populated + notebook verified end-to-end

## Summary of Changes
Commit c4d6675. Built swappable Predictor->Calibrator->Pipeline chain; evaluation harness + 6-figure viz suite, both pipeline-agnostic.

KEY FIX: SimBench score was Eq.1 (per-instance /TVD(P,U)) -> exploded on high-entropy questions, gave bogus mean S=-32.6. Verified against arXiv 2510.17516 that the benchmark uses Eq.2: normalize by per-DATASET mean TVD-to-uniform. Reimplemented; validate_scoring.py proves uniform predictor averages to 0 across all 25 dataset groups (max dev 2e-14) = reproduces the benchmark default by construction. Zero-shot default now scores sane S=18.8 (95% CI 12-25) on 300-record stratified sample with gemini-2.5-flash-lite.

Notebook 00_starter.ipynb runs top-to-bottom from cache (0 new calls on rerun), single component-swap cell. Baseline figures committed to docs/figures/.

Follow-up option: full-split reproduction against a specific paper model (e.g. qwen-2.5-72b / deepseek) to compare to paper-reported S — caveat: our verbalized prompt differs from theirs, so expect ballpark not exact.
