---
# Scrye_Project-jttu
title: 'Part II deck: calm Architecture arrows + add deployment decision tiles'
status: completed
type: task
priority: normal
created_at: 2026-06-24T05:04:16Z
updated_at: 2026-06-24T05:25:04Z
---

Two requests on docs/presentation/part-ii-presentation.html:

1. Architecture page (page 2, svg-detail) reads as busy — many long dashed control/feedback arrows criss-cross the L1-L4 / method-loop band, plus 17 bright moving particles. Calm it: de-emphasize secondary edges, lower opacity, soften particles.

2. Deployment page (page 3, svg-deploy): add TOP-LEVEL deployment decisions (not per-component AI-slop detail). User's framing: start API-based via OpenRouter; fine-tune+host via a managed service (Together AI candidate); move to self-hosted/rented GPUs later when throughput justifies overhead; Tinker = open question; build our OWN eval + experiment/results store (reuse Part I Ledger/manifest/LadderGate). Also make the deployment tiles clickable with detail panels, like page 2.

## Todos
- [x] Reduce edge busyness on Architecture page (two-tier edges, lower opacity)
- [x] Soften particle animation
- [x] Add top-level deployment decisions insight to Deployment page
- [x] Make Deployment tiles clickable with detail panels

## Summary of Changes

**Architecture (page 2):** base edge opacity 0.55->0.4; new .edge.ctrl tier (0.2) applied to the 9 long crossing control/feedback arrows so they recede behind the primary flow; particles dimmed (r 3.5->2.4, opacity 0.85->0.45) and the 3 trails on the longest crossings removed.

**Deployment (page 3):** added a Key Deployment Decisions band (OpenRouter buy-first; managed fine-tune+host via Together AI; rent GPUs later = defer; build our own eval+experiment store; Tinker as an open question). All 17 building-block tiles now clickable with build/buy/defer detail panels (reusing the page-2 panel). De-slopped GPU Fine-tune + Survey Bridge tile subtexts.

Delivered to working copy (uncommitted, on top of the existing restructure WIP). Verified via headless-Chrome screenshots of both pages + a tile click.
