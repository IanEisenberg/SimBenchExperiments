---
# Scrye_Project-7b2g
title: Stage 07 — superforecaster prompting strategies (dev)
status: completed
type: feature
priority: normal
created_at: 2026-06-21T05:40:33Z
updated_at: 2026-06-21T06:12:53Z
---

Test superforecasting-inspired prompt strategies (outside-view base-rate anchoring, entropy/spread-first, full superforecaster pipeline) against the anti_flattening incumbent on dev. Preregister -> implement strategies -> run on dev -> record.

## Progress

- [x] Preregister stage-07-superforecaster.md (signed off)
- [x] Implement 3 strategies (outside_view, entropy_first, superforecaster) + tests (TDD, all pass)
- [x] Strategies auto-register as predictors; CoT parsing smoke clean (0 failures)
- [x] Write scripts/run_superforecaster.py
- [ ] Run on dev (n=1000) — in progress
- [x] Record results in stage doc + README, commit

## Summary of Changes

Added 3 superforecasting PromptStrategy subclasses (outside_view, entropy_first, superforecaster) + tests (TDD). Ran Stage 07 on dev n=990 vs anti_flattening (no-CoT incumbent) and diversity_elicitation (generic-CoT ref).

**Negative result (H0).** Grouped scores: anti_flattening 45.01 (best) > diversity_elicitation 42.93 ~ outside_view 42.77 ~ entropy_first 42.53 > superforecaster 41.55. All forecasting strategies lose to the no-CoT incumbent and none beats generic CoT. More structure made it slightly worse. 2nd clean negative on 'make this model reason about distributions' (Stage 03 was 1st). Incumbent unchanged; no val/test touch. Cost $2.53.
