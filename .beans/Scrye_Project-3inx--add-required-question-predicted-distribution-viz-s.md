---
# Scrye_Project-3inx
title: Add required-question predicted-distribution viz + save headline plots
status: completed
type: task
priority: normal
created_at: 2026-06-23T01:36:18Z
updated_at: 2026-06-23T01:43:22Z
---

Part I deliverable gap: notebook 03 reports required-Q scores but never renders predicted (Q) vs human (P) answer distributions for trust_president/gay_rights/internet_use. Add a reusable viz helper + notebook cells at the end of 03_experiment_results.ipynb, and save the most valuable plots to outputs/visualizations/.

## Todo
- [x] Add plot_required_question_distributions() to src/scrye/viz.py
- [x] Append cells to 03_experiment_results.ipynb: run final system (cc @ gemini-3.1-flash-lite) on test required-Q records, render Q vs P
- [x] Save headline figures to outputs/visualizations/
- [x] Generate PNGs now + verify required-Q mean S matches reported 55.29

## Summary of Changes

- Added reusable viz.plot_required_question_distributions() (population/group-size weighted P vs Q per question) to src/scrye/viz.py.
- Appended 4 cells to the end of notebooks/03_experiment_results.ipynb: a markdown intro, a code cell that reproduces the sealed split, runs the final reported system (calibrated_commitment @ gemini-3.1-flash-lite; cc+abstain reduces to cc on these opinion items) over the test required-Q records (all cached -> free), and renders Q vs P; plus a save cell.
- Verification: pooled required-Q mean S = 55.29, exactly matching the Stage-17 test report. Per-question: trust_president 56.5, gay_rights 48.2, internet_use 61.4. Full notebook executes headless (exit 0); pytest green.
- Saved 4 deliverable PNGs to outputs/visualizations/ (gitignored): required_question_distributions, ablation_strategy, model_portability, test_lineage.
