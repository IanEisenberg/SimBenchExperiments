---
# Scrye_Project-60fk
title: 'Agentic web-tool predictor: design exploration'
status: in-progress
type: feature
priority: normal
created_at: 2026-06-22T15:44:43Z
updated_at: 2026-06-22T17:41:15Z
---

Explore an agentic approach to predicting SimBench survey-response distributions where an LLM agent has tool use (web search/fetch) to gather information before answering the defined prompts. Aligns with project prescription allowing 'the use of tools, including ones that access the internet to gather information.' This bean tracks the brainstorming/design phase.

## Spec written (Stage 18)

Preregistration drafted at docs/experiments/stage-18-agentic-web-evidence.md (PLANNED — held until concurrent stage finishes). Committed on branch worktree-stage-18-agentic-web-spec.

Locked: serious contender system; guards = identity stripping + domain blocklist (+ query guard + audit trail). Design: WebAgentPredictor (Predictor ABC unchanged) = bounded tool-calling agent over guarded, disk-cached web_search/web_fetch. New modules: agent.py, webtools.py, brief.py + tool-calling path in llm.py. Decision rule: beat closed-book twin AND champion AND pass leakage audit; dev pop opinion/knowledge slice ~60 recs. Open infra decision: search backend (Tavily/Brave) — not OpenRouter :online.

## Decisions recorded in experiment log (2026-06-22) — NOT running

Added an explicit Decisions log to docs/experiments/stage-18-agentic-web-evidence.md:
- D1 serious contender (not oracle)
- D2 leakage guards = identity stripping + domain blocklist (+ query guard + audit trail)
- D3 custom guarded web_search/web_fetch on OpenRouter function-calling — NOT :online plugin nor the server tool (verified: server tool hides queries/results from us, so no query guard/audit/cache)
- D4 disk-cache every search/fetch for deterministic re-runs
- D5 closed-book twin control isolates web access from scaffold
- D6 scope: dev pop opinion/knowledge ~60 recs; required Qs untouched
- OPEN: search backend (Tavily/Brave/Exa)

Status: PLANNED, deferred until concurrent stage finishes. Committed on branch worktree-stage-18-agentic-web-spec.
