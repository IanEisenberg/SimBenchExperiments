---
# Scrye_Project-02nt
title: 'Stage 13: principled simulation for weak pop tasks'
status: completed
type: feature
priority: normal
created_at: 2026-06-22T01:07:07Z
updated_at: 2026-06-22T01:44:47Z
---

Find a simulation-faithful method (not benchmark artifact) that beats uniform on the abstain-set datasets (Choices13k, MoralMachine, OSPsychMACH). Core idea: discrete-vote heterogeneous-agent Monte Carlo with task-agnostic dispositional variation. Dev-only iteration; val gated only if a dev winner emerges.

## Summary of Changes

Built VotingEnsemblePredictor (discrete-vote heterogeneous-agent simulation) + sample_disposition/voter_messages (task-agnostic dispositional axes) + _parse_vote. 12 TDD tests (tests/test_voting.py). Stage 13 preregistered + recorded (docs/experiments/stage-13-voting-simulation.md).

Result: global H1 REFUTED (voting beats uniform on 1/3 abstain datasets). Real artifact-free win on Choices13k (+12.6 vs uniform, CI excludes 0 at n=237) — heterogeneous risk panel matches gamble choices. Catastrophic on MoralMachine (-158, confident wrong-mode). Selective routing nets +4.4 dev / ~+0.8 pop. Default system unchanged; optional selective promotion of Choices13k->voting needs a val gate. Mode/knowledge remains the binding constraint.
