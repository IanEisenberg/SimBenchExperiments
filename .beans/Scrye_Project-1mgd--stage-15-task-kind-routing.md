---
# Scrye_Project-1mgd
title: 'Stage 15: task-kind routing'
status: completed
type: feature
priority: normal
created_at: 2026-06-22T04:12:31Z
updated_at: 2026-06-22T04:28:30Z
---

Replace per-dataset intervention routing with an upfront task-kind classifier that dispatches to the strategy that works for that KIND (task-context / voting / abstain / cc). Validate generalization via leave-one-dataset-out on dev.

## Summary
Built taskkind.py (LLMTaskClassifier + heuristic + KIND_ROUTES) and RoutingPredictor. 11 TDD tests. Stage 15: routed+abstain-floor beats base cc by paired +8.75 (CI [5.94,11.67]) on representative dev, beats per-dataset oracle with zero per-dataset params for positive routes. New unified candidate; val-gated confirmation recommended.
