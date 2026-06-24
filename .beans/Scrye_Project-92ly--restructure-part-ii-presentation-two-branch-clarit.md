---
# Scrye_Project-92ly
title: 'Restructure Part II presentation: two-branch clarity'
status: completed
type: task
priority: normal
created_at: 2026-06-23T23:47:00Z
updated_at: 2026-06-24T00:25:12Z
---

Rework docs/presentation/part-ii-presentation.html per user feedback: (1) make System Flow MUCH simpler and distinct from Architecture; (2) remove abstain/routing; (3) show experimental axis of task decomposition + contextualization as enrichment into prompting and/or Centaur; (4) make the prompt/method optimization loop a SEPARATE system from the foundation model. Core message: data normalization flows to (a) a better foundation model and (b) separately an iterating method-optimization process. Deployment page much simpler. All pages simpler/readable; Architecture can be most complex but elaborates from the overview.

## Summary of Changes
Rebuilt docs/presentation/part-ii-presentation.html around the two-branch thesis.
- Page 0 Overview: reframed to the core loop (normalize -> records -> two improving systems) + two-system cards + experimental enrichment card; stripped routing/abstain.
- Page 1 System Flow: rebuilt as a simple vertical fork (Foundation Model | Method Optimization Loop) with the enrichment axis between them and a feedback loop; thinned particles.
- Page 2 Architecture: elaborates the flow with shared memory L1-L4, two distinct branch regions, enrichment band, prediction spectrum + update gate; removed the Coverage Policy/abstain node; updated all clickable NODE_DATA (22 nodes, 1:1 with data-ids).
- Page 3 Deployment: collapsed 6 dense rows -> 4 clean rows, two-branch infra split.
- Capped/centered each SVG so diagrams fit the viewport with a legend gutter.
Verified all 4 pages via headless-Chrome screenshots. Work is in worktree branch worktree-part-ii-two-branch-deck (uncommitted).
