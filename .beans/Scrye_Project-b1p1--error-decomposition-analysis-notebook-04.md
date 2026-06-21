---
# Scrye_Project-b1p1
title: Error decomposition analysis + notebook (04)
status: completed
type: feature
priority: normal
created_at: 2026-06-21T06:26:57Z
updated_at: 2026-06-21T06:33:45Z
---

Decompose SimBench TVD into concentration (profile/spread) vs location (right-options) error per record; library module + tests + notebook 04 comparing a run's predictions vs ground truth, with by-entropy-regime cuts and the system progression story.

## Summary of Changes
Added scrye.decompose (TVD = concentration_err + location_err split, both >=0; signed entropy_gap + mode facets; decompose_records/system_summary) with 7 TDD tests. New notebook 04_error_decomposition: takes a run's results.json, compares pred vs truth distributionally — per-system error profile, the two wins decomposed, pred-vs-truth entropy scatter, by-entropy-regime cut, example distributions, pop/grouped contrast. Headline figure to docs/figures/. CLAUDE.md updated (module + notebook table).

Key finding (model-sweep, grouped): most error is concentration (~65-89%); the model jump 2.5->3.1 fixed both (concentration-led); the anti_flattening strategy win is PURELY location (covers right options, doesn't change spread); predictions are too diffuse on grouped (entropy_gap>0), explaining why Stage 06 'flatten more' calibration failed.
