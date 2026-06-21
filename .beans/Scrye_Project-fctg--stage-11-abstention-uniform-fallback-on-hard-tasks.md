---
# Scrye_Project-fctg
title: Stage 11 — abstention / uniform-fallback on hard tasks (dev)
status: completed
type: feature
priority: normal
created_at: 2026-06-21T20:05:16Z
updated_at: 2026-06-21T20:20:20Z
---

Per-pop-task breakdown for calibrated_commitment; abstain (predict uniform) on datasets where the model scores below uniform, learned per-dataset on dev-fit, applied to dev-eval. Recover the negative-scoring pop tasks.

## Summary
Per-pop-task breakdown + abstention. calibrated_commitment helps most tasks but is MORE wrong on unknowable high-entropy tasks (OSPsychMACH -55, MoralMachine -29) — commitment is confidently wrong there. AbstainCalibrator (predict uniform on datasets the model scores below uniform, learned on dev-fit) lifts dev-eval pop +3.1 -> 39.1 on calibrated_commitment (best combo); captures ~21% of the per-item oracle (50.7). Per-item confidence signal needed for the rest. Registered AbstainCalibrator + tests.
