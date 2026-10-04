# Presentation

Results deck for the SimBench experiments — spine = *disciplined empirical discovery*, a method for finding methods that doesn't fool itself.

## Artifacts

| File | What it is |
|---|---|
| `deck-storyboard.md` | The **spec / blueprint** — 18-slide narrative arc, per-slide content + visual plan + Source pointers, design system, figure→run-data map. Edit this first. |
| `results-deck.html` | The **generated deck** — self-contained, opens in any browser, no runtime. 18 slides on a vanilla-JS 1920×1080 viewer with the data figures inlined as SVG. A copy lives at `deliverables/results-deck.html`. |
| `assets/*.svg` | The **13 data figures** — deck-styled (IBM Plex, palette, labeled axes + bootstrap CIs), regenerated from run data; inlined into the deck. |

## Build pipeline

The deck is **generated**, not hand-edited:

```bash
# 1. Figures — run where the data lives (outputs/ + data/cache/ are in the main checkout).
#    Recomputes from the run files + LLM cache (deterministic, free); writes assets/*.svg.
DECK_RUNS=/abs/path/to/outputs/runs \
  uv run --directory /abs/path/to/repo python scripts/build_deck_figures.py

# 2. Deck — inlines the SVGs + authors the conceptual slides; writes results-deck.html.
python scripts/build_deck.py
```

- `scripts/build_deck_figures.py` — 13 figures via `simbench_exp.viz` / `simbench_exp.decompose` / `simbench_exp.scoring`, styled to the deck palette. IBM Plex isn't a system font, so each SVG carries an injected `<style>` that the deck's web fonts satisfy once inlined; dense scatters are rasterized to keep size down.
- `scripts/build_deck.py` — design tokens + slide helpers; emits the 18 `<section>`s (figure slides inline `assets/*.svg`; conceptual slides — title, task/metric, apparatus, splits, turn, scoreboard, close — are native HTML/SVG) wrapped in the viewer shell.

## Viewing `results-deck.html`

Open it in a browser (`open docs/presentation/results-deck.html`).

- **← / →** (or Space / PageUp-Down) — previous / next slide
- **1–9, 0** — jump to slide · **Home / End** — first / last
- **N** — toggle speaker notes (every slide has presenter notes) · **F** — fullscreen
- URL hash tracks the slide (`…/results-deck.html#14`) for deep-linking.

The stage is a fixed 1920×1080 scaled to the window. Only external dependency is Google Fonts (Spectral / IBM Plex Sans / IBM Plex Mono); degrades to system fonts offline.

## Exporting a PDF

**Print → Save as PDF** in the browser emits all 18 slides at 1920×1080, one per page. Headless:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --no-pdf-header-footer --virtual-time-budget=4000 \
  --print-to-pdf=results-deck.pdf \
  "file://$(pwd)/docs/presentation/results-deck.html"
```
