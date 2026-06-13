---
# Scrye_Project-x3hk
title: 'Core utilities: dataset loading, LLM client, scoring'
status: in-progress
type: feature
priority: high
created_at: 2026-06-13T20:09:49Z
updated_at: 2026-06-13T20:09:49Z
---

Approach-agnostic foundations for the Scrye SimBench mini-project. Needed regardless of which modeling spine we pick.

- [ ] Project scaffolding (pyproject, package layout, .env loading)
- [ ] SimBench dataset download + loader (HF pitehu/SimBench: Pop + Grouped CSVs, parse records into typed schema)
- [ ] OpenRouter LLM client (provider-agnostic, on-disk response cache, retries, cost/usage tracking)
- [ ] Scoring module (SimBench score S = 100*(1 - TVD/TVD_uniform); TVD helpers)
- [ ] Smoke tests (scoring unit tests; data-load + 1 live LLM call sanity check)
- [ ] README / usage notes
