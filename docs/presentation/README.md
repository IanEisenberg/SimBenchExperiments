# Presentation

Talk artifacts for the research-candidate session. **Part I** (**~45 min**, inside a 60-min session) — spine = *disciplined empirical discovery*, a method for finding methods that doesn't fool itself. **Part II** — the forward-looking design that takes the same discipline from benchmark to commercial feedback loop.

## Part I artifacts

| File | What it is |
|---|---|
| `part-i-storyboard.md` | The **spec / blueprint** — 18-slide narrative arc, per-slide content + visual plan + Source pointers, design system, figure→run-data map. Edit this first. |
| `part-i-deck.html` | The **generated deck** — self-contained, opens in any browser, no runtime. 18 slides on a vanilla-JS 1920×1080 viewer with the data figures inlined as SVG. |
| `assets/*.svg` | The **13 data figures** — deck-styled (IBM Plex, palette, labeled axes + bootstrap CIs), regenerated from run data; inlined into the deck. |
| `part-i-deck.dc.html` | The **original 10-slide Claude Design render** (superseded by the generated 18-slide deck; kept for reference). |

## Part II artifacts

| File | What it is |
|---|---|
| `part-ii-presentation.html` | The **Part II deck** — a self-contained 4-page presentation (cover → animated system flow → full architecture with clickable node detail → deployment building blocks). Tab bar + ←/→ navigation; opens in any browser, no runtime. |
| `part-ii-feedback-architecture.md` | The **design document** — how the methodology extends from static survey distributions to a multi-tenant commercial system predicting individual behavioral outcomes (stamped predictions, gated updates, abstention, compounding memory). Source of truth behind the deck. |

## Build pipeline

The deck is **generated**, not hand-edited:

```bash
# 1. Figures — run where the data lives (outputs/ + data/cache/ are in the main checkout).
#    Recomputes from the run files + LLM cache (deterministic, free); writes assets/*.svg.
DECK_RUNS=/abs/path/to/outputs/runs \
  uv run --directory /abs/path/to/repo python scripts/build_deck_figures.py

# 2. Deck — inlines the SVGs + authors the conceptual slides; writes part-i-deck.html.
python scripts/build_deck.py
```

- `scripts/build_deck_figures.py` — 13 figures via `scrye.viz` / `scrye.decompose` / `scrye.scoring`, styled to the deck palette. IBM Plex isn't a system font, so each SVG carries an injected `<style>` that the deck's web fonts satisfy once inlined; dense scatters are rasterized to keep size down.
- `scripts/build_deck.py` — design tokens + slide helpers; emits the 18 `<section>`s (figure slides inline `assets/*.svg`; conceptual slides — title, task/metric, apparatus, splits, turn, scoreboard, close — are native HTML/SVG) wrapped in the viewer shell.

## Viewing `part-i-deck.html`

Open it in a browser (`open docs/presentation/part-i-deck.html`).

- **← / →** (or Space / PageUp-Down) — previous / next slide
- **1–9, 0** — jump to slide · **Home / End** — first / last
- **N** — toggle speaker notes (every slide has presenter notes) · **F** — fullscreen
- URL hash tracks the slide (`…/part-i-deck.html#14`) for deep-linking.

The stage is a fixed 1920×1080 scaled to the window. Only external dependency is Google Fonts (Spectral / IBM Plex Sans / IBM Plex Mono); degrades to system fonts offline.

## Exporting a PDF

**Print → Save as PDF** in the browser emits all 18 slides at 1920×1080, one per page. Headless:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --no-pdf-header-footer --virtual-time-budget=4000 \
  --print-to-pdf=part-i-deck.pdf \
  "file://$(pwd)/docs/presentation/part-i-deck.html"
```
