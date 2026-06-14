---
# Scrye_Project-oip1
title: 'Research overview: synthesize papers + SimBench into strategy'
status: completed
type: task
priority: high
created_at: 2026-06-13T20:39:38Z
updated_at: 2026-06-13T20:47:49Z
---

Read scrye_simulation_brief.md, the four papers in docs/papers/ (Park 1000-people, SimBench, Generative Agents, AgentSociety), do further web research on relevant simulation/twin papers, and synthesize into docs/research_overview.md. Key gap to fix: the simulation brief did not incorporate SimBench findings.

## Summary of Changes
Created docs/research_overview.md. Read SimBench in full (the flagged gap), plus parallel-agent extractions of Generative Agents 2023, AgentSociety, and a structured web sweep (~25 papers). Key analytic output: SimBench corrects the brief on three points — (1) 'model is commodity' is half-wrong (40-pt spread, MMLU-Pro r=0.94, alignment-simulation tradeoff is a moat lever); (2) demographic conditioning is net-negative (ΔS all negative), strengthening persona-as-probe; (3) the attitude-vs-behavior gap is now benchmark-confirmed across 20 datasets, not a Park-games artifact. Added a new unifying mechanism (mode-seeking RLHF → variance collapse = miscalibration). Doc includes 6 testable hypotheses tied to RedBird telemetry, a prioritized papers-to-acquire list, and architecture implications (governance layer upgraded from wedge to load-bearing).
