---
# Scrye_Project-dyv8
title: Build consolidated per-question predictions CSV
status: completed
type: task
priority: normal
created_at: 2026-06-23T16:57:33Z
updated_at: 2026-06-23T17:07:02Z
---

Single CSV in outputs/ with one row per SimBench question: human distribution, our predictions, recomputed SimBench score, plus queryable metadata (dataset, split, task_kind, segment, is_population, n_options, group_size, required_question). Source: full-coverage per-item runs; scores recomputed with full-split Eq.2 normalizers.

## Summary of Changes

Produced three artifacts in outputs/ (gitignored):
- simbench_question_predictions.csv — 13,510 rows (one per SimBench question, all dev/val/test buckets, pop+grouped).
- simbench_question_predictions.README.md — data dictionary.
- build_question_predictions.py — reproducible generator (reads only the LLM cache; no API spend).

Columns: question_id, dataset, task_kind, eval_bucket, simbench_file, is_population, num_segment_vars, segment_vars, segment, n_options, group_size, required_question, question_text, options, human_dist, truth_entropy, tvd_to_uniform, normalizer_Z, uniform_score, our_system, our_pred, our_score, faithful_pred, faithful_score.

"Our system" = anti_flattening (best full-coverage system, mean S~=41; the headline task-kind router scores the same on TEST but is only partially cached so cannot be reproduced per-question for free). Scores recomputed with full-split Eq.2 normalizers. Validated: test faithful=35.21 (pop 30.88 / grouped 39.19) exactly matches the authoritative TEST-final run; uniform_score averages to 0.00.
