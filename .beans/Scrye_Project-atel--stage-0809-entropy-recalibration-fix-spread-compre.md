---
# Scrye_Project-atel
title: Stage 08/09 — entropy recalibration (fix spread compression, dev)
status: completed
type: feature
priority: normal
created_at: 2026-06-21T07:06:19Z
updated_at: 2026-06-21T07:18:31Z
---

Diagnostic: predictions are entropy-compressed (pred_H ~ 0.46+0.48*truth_H), +8.4 oracle headroom on grouped. Build entropy de-compression calibrators, test vs identity + best global-T on dev. Up to 2 rounds.

## Summary
Two dev rounds on the entropy/spread problem. Built EntropyTargetCalibrator (own-entropy de-compression) + FeatureEntropyTargetCalibrator (feature-predicted target entropy) + temper_to_entropy helper; tests pass.

Stage 08: oracle headroom +9.5 grouped is real but NOT recoverable from prediction's own entropy (gain fit to 0); global temps hurt.
Stage 09: features predict truth-entropy at R2=0.58, but tempering STILL can't beat identity. Mechanism: oracle gain is gated behind mode correctness (+16.8 where top option right, -2.0 on the 39% where wrong). Spread entangled with location; post-hoc calibration closed. Binding constraint = mode/location accuracy. Incumbent unchanged; val/test untouched.
