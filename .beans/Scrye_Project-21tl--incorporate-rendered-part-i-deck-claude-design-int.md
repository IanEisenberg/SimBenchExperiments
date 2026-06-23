---
# Scrye_Project-21tl
title: Incorporate rendered Part I deck (Claude Design) into repo
status: completed
type: task
priority: normal
created_at: 2026-06-23T04:21:03Z
updated_at: 2026-06-23T04:26:16Z
---

Bring the Claude Design 'Scrye System Deck' (10-slide rendered Part I deck) into docs/presentation as a self-contained, portable HTML deck. The .dc.html depends on Claude Design's React runtime and won't open standalone; extract the 10 inline-styled <section> slides verbatim and wrap them in a minimal vanilla-JS viewer (1920x1080 stage, arrow-key nav, speaker-notes toggle, print-to-PDF). Faithful to the Claude Design rendering.

## Summary of Changes

Incorporated the Claude Design 'Scrye System Deck' (10-slide Part I render) into `docs/presentation/` on worktree branch `worktree-part-i-deck-html` (commit 447ce72, not merged to main).

- **part-i-deck.dc.html** — the Claude Design source of truth (project b49f92eb…); needs that runtime, does not render standalone.
- **part-i-deck.html** — self-contained portable render: the 10 inline-styled `<section>` slides extracted verbatim, wrapped in a vanilla-JS 1920×1080 stage viewer (arrow-key nav, slide jump, speaker-notes toggle, fullscreen, hash deep-linking, print-to-PDF → 10 pages). No React / Claude Design runtime.
- **README.md** — artifact map (storyboard → .dc.html source → .html render), controls, PDF export, regen steps.

The deck depends only on Google Fonts; the 5 PNGs in the Claude Design `assets/` are unused by the final 10-slide cut (zero `<img>`/`assets/` refs), so not copied. Verified with headless Chrome: all 10 slides render faithfully and the print path yields a 10-page PDF.
