---
# Scrye_Project-sn3m
title: 'Executive overview doc: problem space, AI-discovery meta-approach, levers'
status: completed
type: task
priority: normal
created_at: 2026-06-22T23:08:47Z
updated_at: 2026-06-23T00:32:36Z
---

Write an executive overview document in docs/ that talks through: (1) the problem space (SimBench — predicting empirical human survey-response distributions, beating the naive uniform baseline), (2) the approach including the META approach of AI-based scientific discovery (preregistered stages, leave-family-out splits, lever registry, guarded search loop, Ladder gate, ledger/manifest reproducibility), and (3) the specific levers pursued across stages 01-17 with outcomes. Audience: executive/non-deep-technical reader. Complements the existing interactive docs/overview.html.

## Summary of Changes

Wrote `docs/executive-overview.md` — a plain-language executive tour structured as:
1. **Problem space** — SimBench distributional prediction; S = 100·(1 − TVD(P,Q)/TVD(P,U)) scoring vs the uniform baseline (S=0); why it's hard (demographic conditioning backfires, value–action gap, spread/location entanglement).
2. **Approach** — the Record→Predictor→Calibrator pipeline, then the AI-driven scientific-discovery meta-method: preregistered stages, leave-family-out dev/val/test splits with required-Qs pinned to test, the Ladder gate (Blum & Hardt 2015) with bootstrap-derived η, and the reproducibility apparatus (content-addressed specs, ledger DAG, manifest sidecar).
3. **Specific levers** — a verdict table across stages 01–17 (model swap = dominant; distributional framings + calibrated_commitment + abstention won; calibration / reasoning-first / Monte-Carlo / voting lost or didn't transfer) plus cross-cutting findings.
4. **Headline result** — final test ~40.9/100 (faithful@3.1 35.21 → cc+abstain 40.93), model dominance (+15.9) vs method stack (+5.5), router val-win not transferring, paper reproduction (26.83 brackets 27.61).
5. **Part II pointer** — forward-looking commercial feedback architecture.

Complements (does not duplicate) the interactive `docs/overview.html`. Committed on worktree branch `worktree-executive-overview-doc` as `eed7879` (isolated background-job worktree; not yet merged to main).

## Follow-up (2026-06-22, after nemotron merge)

User asked to also cover directions pursued-but-rejected / deferred, and to refresh the docs HTML. Done on branch `worktree-executive-overview-doc` (rebased onto main @ f7ad906 to pick up Stage-18 nemotron), commit `fbec6c6`:
- **executive-overview.md**: §2.2 reframes the experiment-crafting machinery as a deliberate first-class deliverable (scalable meta-science tool); new **§5 'Directions considered but not adopted'** — Nemotron grounded-persona electorate (5 rounds, rejected: solved spread to entropy 0.644 vs 0.692 but ~11pts below the calibrated_commitment champion, location-limited, ensemble +0.5 noise → persona line closed), agentic web-retrieval (WebAgentPredictor specified + preregistered, deferred for time), and the meta-science tool as a deliberate score-vs-infrastructure trade.
- **overview.html** (was untracked in working checkout; now committed on branch): new 'What We Found — Part I Results' section (40.9 final, model dominant +15.9, method +5.5, paper repro 26.8≈27.6) + meta-science callout; replaced 'Open Questions' with 'Directions Considered But Not Adopted'. Left Part II forward-looking hypotheses untouched (NOT falsely marked confirmed).
- **architecture-diagram.html**: reviewed, unchanged — it's the Part II commercial design diagram (principle-driven), still accurate.

Caveat flagged to user: overview.html being committed will collide with their untracked copy on merge.
