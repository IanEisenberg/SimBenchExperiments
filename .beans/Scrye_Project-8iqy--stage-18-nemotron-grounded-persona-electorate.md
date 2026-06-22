---
# Scrye_Project-8iqy
title: 'Stage 18: Nemotron grounded persona electorate'
status: completed
type: feature
priority: normal
created_at: 2026-06-22T17:44:08Z
updated_at: 2026-06-22T23:13:13Z
---

Grounded Nemotron persona voting electorate (K=50, discrete vote, gemini-3.1-flash-lite) vs naive LLM on OpinionQA. Standalone first; ensemble with champion only if promising. Code+tests done; Round 1 dev run in progress. Early diag: electorate over-concentrates (personas collapse to modal option). Spec: docs/experiments/stage-18-nemotron-personas.md

Round 2 (per-persona distribution + average) built: PERSONA_DIST_SYSTEM + persona_dist_messages (calibrated, anti-hedge), GroundedAveragingPredictor, registry entry grounded_averaging, tests (21 green), entropy diagnostic in run script. Decision to average not sample (averaging = unbiased estimator; sampling adds variance). Waiting for Round1 voting cache fill (~51 pct) before launching combined run.

## Summary of Changes
Built + evaluated the grounded Nemotron persona electorate on OpinionQA pop dev (n=248, gemini-3.1).
- Discrete voting: 12.7 (over-concentrates).
- Distribution+average (the user's direction): +33.6 -> 46.3, spread recovered to near-calibrated entropy (0.644 vs truth 0.692), no over-dispersion. Hypothesis confirmed.
- Standalone still -17 vs calibrated_commitment (63.2); ensemble sweep best +0.33 (noise). Residual gap = location, not spread.
Shipped: nemotron.PersonaBank, Grounded{Voting,Averaging}Predictor, EnsemblePredictor, persona prompts, 24 tests green. Committed 7352721. Stage doc + README updated. Champion unchanged.
Deferred (optional follow-up): grouped-segment extension (panel filtering / counterfactual sensitivity).

## Final: persona line closed
Across 5 rounds: voting 12.7 -> averaging 46.3 -> worldview-enriched 51.8, all < champion 63.2. Each round fixed the predicted failure; final system decorrelated (r=0.56) but ensemble adds only +0.47 (noise). Negative with full mechanism. See stage-18 doc.
