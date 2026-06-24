---
# Scrye_Project-72rg
title: Interactive per-question results explorer (HTML)
status: completed
type: feature
priority: normal
created_at: 2026-06-23T22:59:37Z
updated_at: 2026-06-23T23:54:23Z
---

Self-contained offline HTML explorer for outputs/simbench_question_predictions.csv. Filter by task_kind/dataset/bucket, sort (best/lift), scrollable virtualized question list, detail pane per question with an overlaid Human/Ours/Faithful distribution bar chart + uniform reference, question text, metadata, and scores. Output to outputs/.

## Summary of Changes

Built outputs/simbench_explorer.html — self-contained offline HTML explorer (vanilla JS, no libs/server) over the 13,510-question CSV. Filters (task_kind/dataset/eval_bucket/required/search), sort (Ours best/worst, lift, entropy), virtualized scrollable list, Prev/Next + arrow keys, per-question detail with overlaid Human/Ours/Faithful distribution chart (+uniform line), question text, metadata, score cards. Generator: outputs/build_explorer.py. Verified via headless-Chrome render.

## Update — two-page app

Split into two tabs. Page 1 = the item-level Questions explorer. Page 2 = Task summaries: topline cards (split-avg=(pop+grouped)/2, pooled=record-weighted mean, pop, grouped — Ours vs Faithful + lift) with a bucket selector (default test); a mean-SimBench-score-by-task overview (grouped bars, negatives below zero); and a task-type selector driving three per-task plots — score breakdown (split-avg/pop/grouped/pooled), per-question score histogram (Ours vs Faithful), and dispersion (mean answer entropy: Human/Ours/Faithful). All computed client-side from the embedded data. Verified both pages via headless-Chrome render.
