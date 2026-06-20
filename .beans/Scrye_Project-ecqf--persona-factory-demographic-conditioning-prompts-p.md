---
# Scrye_Project-ecqf
title: 'Persona factory: demographic conditioning prompts + post-stratification'
status: completed
type: feature
priority: normal
created_at: 2026-06-20T00:56:06Z
updated_at: 2026-06-20T01:06:51Z
---

Build a persona factory that derives demographic-conditioning prompts (strategy registry) for all SimBench countries, plus per-task demographic distribution tables and a PostStratificationPredictor that recombines subgroup inferences into population estimates.

## Design
Spec: docs/superpowers/specs/2026-06-19-persona-factory-design.md

## Todo
- [x] src/scrye/persona.py: VARIABLE_DICTIONARY, COUNTRIES catalog, verbalize_segment, PromptStrategy registry (simbench_faithful, representative_sample, persona_embodiment, anti_flattening, contextualized)
- [x] src/scrye/distributions.py: build_segment_weights, PopulationCell tree, WeightSource interface (SimBenchWeights)
- [x] Refactor predict.py ZeroShotPredictor to be strategy-driven
- [x] PostStratificationPredictor in predict.py (subgroups -> country marginal, validated vs ('country',) truth)
- [x] Tests: strategies emit valid messages; weights sum to 1; weighted subgroup truths reproduce marginal truth
- [x] Demo script (scripts/persona_demo.py) comparing styles + post-strat vs direct
- [x] Write spec doc + self-review

## Summary of Changes

- **src/scrye/persona.py**: COUNTRIES catalog (102 distinct, all 5 grouped datasets), VARIABLE_DICTIONARY (40 keys + generic fallback), verbalize_segment, year/locale extraction, and a registry of 5 ablatable PromptStrategies (simbench_faithful, representative_sample, persona_embodiment, anti_flattening, contextualized). faithful_prompt() reproduces the prior baseline byte-for-byte (cache-safe).
- **src/scrye/distributions.py**: SegmentWeights (group_size-based post-stratification, children()/decompose_variables()), WeightSource protocol, distribution_table() artifact (one row per cell with respondent share).
- **src/scrye/predict.py**: ZeroShotPredictor is now strategy-driven (default unchanged); new PostStratificationPredictor decomposes a coarse target into finer cells, predicts each, recombines by population weight, falls back when no decomposition exists.
- **Tests**: 39 new tests pass; full suite green. Includes a real-data property test confirming group_size-weighted subgroup truths reproduce the country marginal (TVD < 0.02).
- **scripts/persona_demo.py**: offline catalog/prompt/distribution demo; opt-in --live LLM ablation.

Verified: full pytest suite passes; offline demo renders all strategies; recombination identity holds (mean TVD 0.0003 over 200 country-questions).
