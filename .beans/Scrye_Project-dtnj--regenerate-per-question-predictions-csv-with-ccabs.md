---
# Scrye_Project-dtnj
title: Regenerate per-question predictions CSV with cc+abstain
status: completed
type: task
priority: normal
created_at: 2026-06-24T06:07:09Z
updated_at: 2026-06-24T06:10:08Z
---

deliverables/simbench_question_predictions.csv was graded on anti_flattening (a coverage-driven fallback), not the shipped cc+abstain. calibrated_commitment is now 100% cache-covered across all 13,510 Qs, so regenerate from cache with calibrated_commitment + AbstainCalibrator (abstain-set fit on full dev = {Choices13k, MoralMachine, OSPsychMACH}). Update outputs/build_question_predictions.py and the CSV.

## Summary of Changes

**Problem:** `deliverables/simbench_question_predictions.csv` (symlink → `outputs/`) had `our_system = anti_flattening` for all 13,510 rows — a coverage-driven fallback, not the shipped headline `cc+abstain`. The original build script chose anti_flattening because cc was only partially cached at the time.

**Fix:** `calibrated_commitment` is now 100% cache-covered across all buckets (probed: dev 6151 / val 3767 / test 3592, 0 misses), so `outputs/build_question_predictions.py` now grades the real headline system:
- predictor = `calibrated_commitment` ZeroShot
- calibrator = `AbstainCalibrator` fit on full DEV cc predictions → abstain-set {Choices13k, MoralMachine, OSPsychMACH} (exactly matches Stage 12's recorded set)
- `our_system` column = `calibrated_commitment+abstain`

**Verification (exact reproduction of Stage 17 sealed test):** TEST overall 40.93 / grouped 44.02 / pop 37.55 / required 55.29 — match the stage doc to the decimal. 0 cache misses (free, deterministic). All 1100 abstain-dataset rows emit uniform. Mean our_score across full split: 41.24 (anti_flattening) → 43.28 (cc+abstain).

**Also rebuilt:** `outputs/simbench_explorer.html` (built from the CSV) — refreshed, 0 stale anti_flattening references. anti_flattening mentions in `docs/presentation/part-ii-feedback-architecture.md` are legitimate strategy/lineage references, left as-is.

**Files changed:** `outputs/build_question_predictions.py`, `outputs/simbench_question_predictions.csv`, `outputs/simbench_explorer.html`. Not committed (awaiting user).
