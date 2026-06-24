---
# Scrye_Project-alne
title: Refresh CLAUDE.md to match grown codebase
status: completed
type: task
priority: normal
created_at: 2026-06-24T05:45:45Z
updated_at: 2026-06-24T05:48:35Z
---

The repo's CLAUDE.md key-modules list, architecture diagram, persona-strategy description, notebooks table, and commands are stale after Stages 10-18. Update to document new modules (ask/cli_ask, taskkind, worldview, nemotron, distributions, viz, report, tree_view), the scrye-ask CLI deployable, the current best system (calibrated_commitment @ gemini-3.1-flash-lite + AbstainCalibrator), and fix the notebooks table.

## Summary of Changes
Refreshed CLAUDE.md (no rewrite — targeted edits to the existing well-written file):
1. **Commands** — added the `scrye-ask` deployable CLI (real `[project.scripts]` entry point, previously undocumented).
2. **Architecture diagram** — broadened predictor list (3→11: Routing/Voting/Grounded family) and calibrator list (2→9: Abstain/EntropyTarget/Dirichlet/Chain).
3. **New 'Current best system' block** — calibrated_commitment @ gemini-3.1-flash-lite + AbstainCalibrator (Stage 17 sealed test, 35.21→40.73); model-is-dominant-lever finding; router ties cc+abstain.
4. **Persona bullet** — 5→10 strategies + TaskContextStrategy; winner is now calibrated_commitment (was anti_flattening).
5. **Key modules** — documented 9 previously-missing modules: predict, taskkind, nemotron, worldview, distributions, ask/cli_ask, viz/report/tree_view. Dropped stale 'drives notebook 04' from decompose.
6. **Notebooks table** — removed the deleted 04_error_decomposition.ipynb; added 00_dataexploration.ipynb.
7. **Caching/artifacts** — noted the tracked deliverables/ directory.

## Summary of Changes

Rewrote README.md as a front door for someone engaging fresh:
- New intro + honest headline test result (40.9 overall; model is the dominant lever; harness reproduces the paper, 26.83 vs 27.61).
- New 'Where to start reading' doc map: executive-overview.md (first), docs/experiments/ (the science), beans (the work), CLAUDE.md, specs, deliverables.
- Added the deployable scrye-ask demo to Quick start.
- Added the Record->Predictor->Calibrator->Pipeline->evaluate->score loop + full ~18-module table (was only 4 modules); refreshed notebooks list to actual files.
- Added leakage-discipline and preregistration-workflow sections.
- Verified every class/strategy name, the build_normalizers location (evaluate.py, not scoring.py), and all doc links against the repo.
