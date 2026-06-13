---
# Scrye_Project-x3hk
title: 'Core utilities: dataset loading, LLM client, scoring'
status: completed
type: feature
priority: high
created_at: 2026-06-13T20:09:49Z
updated_at: 2026-06-13T20:16:48Z
---

Approach-agnostic foundations for the Scrye SimBench mini-project. Needed regardless of which modeling spine we pick.

- [x] Project scaffolding (pyproject, package layout, .env loading)
- [x] SimBench dataset download + loader (HF pitehu/SimBench: Pop + Grouped CSVs, parse records into typed schema)
- [x] OpenRouter LLM client (provider-agnostic, on-disk response cache, retries, cost/usage tracking)
- [x] Scoring module (SimBench score S = 100*(1 - TVD/TVD_uniform); TVD helpers + bootstrap CI)
- [x] Smoke tests (23 unit tests pass; live OpenRouter call + cache verified)
- [x] README / usage notes

## Summary of Changes
Built approach-agnostic foundations (commit 998d130):
- src/scrye/config.py — paths, .env loading, validated OpenRouter model ids, DEFAULT_MODEL=google/gemini-2.5-flash-lite
- src/scrye/data.py — download_simbench() + load_split(); typed SimBenchRecord captures segment (group_prompt_variable_map), grouping_keys (set-literal parse), answer_options. Verified counts 7167/6343.
- src/scrye/scoring.py — simbench_score (S=100*(1-TVD/TVD_uniform)), TVD, aggregate, bootstrap_ci
- src/scrye/llm.py — OpenRouter client (openai SDK), sha256 on-disk response cache, retries, usage tracking
- scripts/download_data.py, scripts/smoke_llm.py; README; pyproject (uv); .env.example
- 23 tests pass; live call returns Paris + cache hit confirmed

Key learnings: naive baseline = uniform (S=0); segments live in group_prompt_variable_map; grouping_keys is a Python set() literal; Pop split also contains conditioned (per-country) records. Validated OpenRouter ids recorded in config (gemini-2.0-flash-001 does NOT exist; use gemini-2.5-flash-lite).
