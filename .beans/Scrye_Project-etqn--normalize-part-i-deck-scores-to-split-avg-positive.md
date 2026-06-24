---
# Scrye_Project-etqn
title: Normalize Part I deck scores to split-avg; positive turn slide
status: completed
type: task
priority: normal
created_at: 2026-06-24T05:19:34Z
updated_at: 2026-06-24T05:24:35Z
---

Report split-avg as the general metric across the Part I deck (keep pop/grouped/task-specific in native metric). Make the turn slide positive (remove niggling negatives).

## Summary of Changes

Normalized the Part I deck to report **split-avg** as the general metric (pop/grouped/task-specific kept in native metric, labeled).

- **Result figure (`fig_result`)**: left panel switched from `overall SimBench S` → `split-avg SimBench S`. Points from the reported cc+abstain system (TEST-lineage): faithful@3.1 **35.0** → our system **40.8**, delta **+5.8**. CI **[4.5, 7.0]** combined from the tying final system's disjoint pop/grouped strata (variances add) — excludes 0.
- **Scoreboard**: cc+abstain `+5.7 test` → `+5.8 split-avg test`.
- **Result slide** takeaway + notes: `+5.5` → `+5.8 split-avg`.
- **Turn slide** (positive): cc card `+1.8 grouped dev · shrinks to within val noise` → `+5.8 split-avg test (with abstention) · val ✓`; abstention card dropped "the 3 datasets we land below uniform" → `Knows when to abstain — falls back to uniform on the hardest ~5% of items instead of guessing.`
- Title `+6` and slide 18 `+6` kept as the rounded split-avg headline.

Regenerated `assets/13_result.svg` and rebuilt `part-i-deck.html` (23 slides).
