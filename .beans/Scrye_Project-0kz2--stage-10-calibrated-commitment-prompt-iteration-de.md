---
# Scrye_Project-0kz2
title: Stage 10 — calibrated-commitment prompt iteration (dev)
status: completed
type: feature
priority: normal
created_at: 2026-06-21T16:34:58Z
updated_at: 2026-06-21T17:10:35Z
---

Up to 4 rounds combining contextualized + anti_flattening + consensus/entropy awareness + licensed commitment. Beat anti_flattening on dev grouped; lower the entropy floor on consensus questions without losing the location win.

## Summary
4 dev rounds of prompt iteration. Winner: calibrated_commitment (commit_v4) — a single mode-first prompt carrying all 4 requested elements, improving both grouped (+1.8) and pop (+2.1) over anti_flattening via better location/mode (not entropy commitment). Within dev noise → registered strategy + val-confirmation candidate; system unchanged, no val/test spent. Learnings: lever is location not commitment; one prompt beats regime-branching; pop is a heterogeneous non-survey task mix; context/aggressive-commitment add-ons didn't help.
