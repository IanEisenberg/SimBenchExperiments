---
# Scrye_Project-grx2
title: Fold research_overview findings into gameplan spec
status: completed
type: task
priority: normal
created_at: 2026-06-14T02:10:22Z
updated_at: 2026-06-14T02:12:44Z
---

Incorporate components of docs/research_overview.md into the 2026-06-10 gameplan design spec: elevate human-reliability denominator over SimBench's uniform baseline, add mode-seeking KL mechanism + base-vs-instruct ablation, correct model-as-commodity to model-portability, ground Part II revealed-vs-stated in the benchmark-confirmed value-action gap.

## Summary of Changes
Wove research_overview.md components into the 2026-06-10 gameplan spec (marked inline as [R]):
1. Header: added research-grounding pointer to research_overview.md.
2. §3: flagged SimBench's uniform denominator as the field's weaker normalizer; elevated human-reliability ceiling as the principled fix (user-endorsed). Added mode-seeking KL mechanism (mass-covering vs mode-seeking) + the miscalibration-unifies-failures framing; noted grouped-conditioning degradation as benchmark evidence for delta-modeling.
3. §4.4: corrected 'model-as-commodity' to 'method-portability' (SimBench 40-pt spread, MMLU-Pro r=0.94).
4. §6.F: elevated reliability-ceiling normalization as the signature metric (report ceiling-normalized S, not uniform-baseline S); added base-vs-instruct ablation tied to the alignment-simulation tradeoff; grounded the counterfactual metric's directional/magnitude split in the rank-holds/magnitude-inflates literature.
5. §6 Approach B: sharpened personalization signal to 'modest-to-negative'; pointed to persona-as-validity-probe (H4) for the roadmap.
6. §7 Part II: grounded revealed-vs-stated routing in the benchmark-confirmed value-action gap.
7. §8: aligned 'model is a commodity' -> 'swappable' for internal consistency.
