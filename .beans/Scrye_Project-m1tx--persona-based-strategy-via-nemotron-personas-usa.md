---
# Scrye_Project-m1tx
title: Persona-based strategy via Nemotron-Personas-USA
status: in-progress
type: feature
priority: normal
created_at: 2026-06-22T15:43:33Z
updated_at: 2026-06-22T17:03:16Z
---

Design a new persona-conditioning PromptStrategy that uses synthetic personas from the Nemotron-Personas-USA dataset, integrated into the Ladder-gated search setup. Research + propose approaches first.

## Context gathered (codebase)

Two pre-existing injection seams fit Nemotron with minimal new code:
- `predict.MonteCarloPredictor` / `VotingEnsemblePredictor` both accept `sampler=` — currently `persona.sample_persona` / `sample_disposition` (synthetic axis pools). A Nemotron-backed sampler drops in directly.
- `distributions.WeightSource` is a Protocol behind `PostStratificationPredictor`; docstring explicitly anticipates 'an external census source… US → states.'

Scientific framing from experiment log:
- Stage 03: synthetic persona Monte-Carlo LOST (-15.7) — generic + over-dispersed.
- Stage 13: dispositional voting ensemble won only on Choices13k.
- Nemotron = census-grounded, demographically-matched, rich free-text personas → direct test of whether grounding rescues the Stage-03 failure.

Constraint: Nemotron-Personas-USA is US-only → directly applies to OpinionQA (+US rows of ISSP); not ESS/Afrobarometer/LatinoBarometro.

## Plan
- [ ] Get exact Nemotron schema (fields, census grounding, geography) — research agent running
- [ ] Propose 2-3 approaches (grounded ensemble / census post-strat / rich embodiment)
- [ ] Ask scope question (US-only datasets vs general approach)
- [ ] Write design doc to docs/superpowers/specs/ + preregister stage-18
