---
# Scrye_Project-udfd
title: 'Stage 18 Round 5: worldview-enriched ideology-calibrated electorate'
status: completed
type: task
priority: normal
created_at: 2026-06-22T22:16:25Z
updated_at: 2026-06-22T23:13:13Z
---

Final persona inquiry. Pipeline: rich Nemotron panel -> infer sampled worldview per persona -> calibrate panel ideology mix to OpinionQA real POLIDEOLOGY marginal -> answer + weighted average. Test standalone on OpinionQA pop dev; if signal and decorrelated from champion, ensemble. Built worldview.py + EnrichedGroundedPredictor + tests (27 green). Smoke running.

Smoke (n=12) promising: enrichment +8.7 over summary baseline (48.7->57.4); ensemble lifts champion 60.2->64.5 at w=0.5 despite r=0.81 correlation. Representativeness check: panel 74pct moderate vs real 37pct, 0pct extremes (model softens to moderate) -> validates calibration need, though empty tails limit it. Full n=248 run launched.

## Summary
Final result (n=248 OpinionQA pop dev): enriched(calibrated) 51.84 [47.8,55.6] vs champion 63.24. Standalone +5.5 over bare-persona baseline (best grounded system); decorrelated r=0.56; ensemble best +0.47 (noise). Calibration +0.56 (panel had empty ideological tails: model softens to 74pct moderate vs real 37pct). Verdict: bottom-up persona electorate does not beat/improve champion; location-limited. Persona line closed. Committed 523cf05.
