---
# Scrye_Project-8cis
title: Stage 06 — Calibration sweep on anti_flattening (dev, n=1000)
status: completed
type: task
priority: normal
created_at: 2026-06-21T05:20:27Z
updated_at: 2026-06-21T05:28:58Z
---

Preregister and run the three registered calibrators (TempScaling, EntropyTempScaling, DirichletCalibrator) on anti_flattening@gemini-3.1-flash-lite. Fixed hyperparameter grid sweep on a 1000-sample stratified dev subsample. All predictions cached from Stage 04 — no new LLM calls.

## Summary of Changes

Preregistered and ran Stage 06. Fixed hyperparameter grid sweep across 14 configs (identity + 5 TempScaling + 4 EntropyTempScaling + 4 Dirichlet) on a 990-record stratified dev subsample. All predictions served from cache (cost $0). Clean negative result: no calibrator beats identity on grouped score. anti_flattening+identity (grouped 45.01) is the confirmed final system. Stage doc written at docs/experiments/stage-06-calibration.md.
