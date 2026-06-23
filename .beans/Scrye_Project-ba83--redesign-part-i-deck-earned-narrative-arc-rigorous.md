---
# Scrye_Project-ba83
title: Redesign Part I deck — earned narrative arc + rigorous data figures
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T05:18:37Z
updated_at: 2026-06-23T14:55:14Z
---

Rebuild the Part I deck (docs/presentation/part-i-deck.html) from a bottom-line-up-front 10-slide cut into an 18-slide earned-narrative arc for a 45-min talk. Spine: disciplined empirical discovery; simplicity revealed as the conclusion, not the opener; triple-honesty turn (clever<simple; model>method; fit>size). Replace hand-drawn schematic charts with real, deck-styled, labeled+CI figures regenerated from run data via scrye.viz. Spec: docs/presentation/part-i-storyboard.md.

## Todos
- [x] Spec: 18-slide narrative storyboard + per-slide visual plan (docs/presentation/part-i-storyboard.md)
- [x] User review of spec (headline locked: split-avg 25->41, +10 model/+6 method)
- [x] Figure pipeline: 13 deck-styled SVGs (build_deck_figures.py)
- [x] Build the 18 slides on the vanilla-JS viewer (build_deck.py)
- [x] Embed real figures (11 figure slides, inlined SVG)
- [x] VERIFY: per-Q scores (trust 56.5/gay 48.2/internet 61.4); opener framing (25->41)
- [x] Verify with headless Chrome (all 18 render; 18-page PDF)

## Feedback round 2 (2026-06-23)
Deck → 22 slides. Apparatus spine-only (Predictor/Calibrator defined) + search loop → splits slide; ladder pooled-val clarifier; NEW 'winning prompts' + 'Nemotron bet' slides; NEW cross-task×group figure+slide; abstain rate ~5% on survivors; model-vs-method test/dev labels + bitter lesson; cf_alignment method panel; close contrast fix; TVD spelled out; page-total auto-computes. faithful@Qwen required-Q bars not built (calls uncached). Spec (part-i-storyboard.md) now behind the deck — resync pending. Commit 4f6e49f.
