# Part I — Presentation Spec (18-slide narrative cut)

**Scrye: Predicting Human Survey-Response Distributions with LLMs**

A **45-minute** Part I talk (~18 main slides + a Backup section for Q&A), inside a 60-min session — **Part II ships as a separate HTML deck**. Rendered on the same self-contained viewer as `part-i-deck.html` (vanilla-JS 1920×1080 stage; `← →` nav, `N` notes, print-to-PDF).

> **Spine:** *disciplined empirical discovery* — we submit a method for finding methods that doesn't fool itself.
> **Narrative shape:** an **earned** story. Show the *result* up front (the shift from baseline), then make the talk about **how** — and **how we know it's real**. Simplicity is the **conclusion**, revealed at the turn (Act 4), not the opener.
> **The triple-honesty turn:** clever loses to simple → the *model* did most of the work → and even that is *"fit, not size"*, honestly bounded by a model search we didn't finish.
> **Headline (locked):** sealed-test split-avg **≈25 → ≈41** — published SimBench faithful baseline (Qwen2.5-72B) → our system. Anatomy, one change at a time: faithful@Qwen **25.0** → swap to gemini-3.1 **35.0** (**+10** model) → our method **40.8** (**+5.8**, ≈+6). All on **split-avg** (the paper's metric); ≈ record-pooled "overall" here. gemini-2.5 dropped — not a real comparison.

This spec supersedes the earlier 10-slide cut (kept in git history). Every data-bearing slide names a **real figure** (regenerated from run data via `scrye.viz`) or a **native diagram**, plus a **Source:** pointer. Numbers below are pulled from the committed stage docs + run inventory; a few per-question splits are tagged **VERIFY** (extract from the regenerated figure before rendering).

---

## Workplan coverage (the four required elements)

| Required element | Acts / slides |
|---|---|
| Motivation & high-level intuition | Act 1 — slides 2–3 |
| Key modeling decisions & **formal** justifications | Act 2 — slides 4–6 (Ladder gate = the formal bound); Act 4 — slide 11 |
| Experimental results & ablation findings | Act 3 (graveyard) 8–10; Act 4 (scoreboard) 12; Act 5 (results) 13–16 |
| Limitations, risks, extensions | Act 6 — slides 17–18 |

---

## Deck at a glance

| # | Act | Slide | Job | Visual |
|---|---|---|---|---|
| 1 | 0 Hook | Title + the shift | ≈25 → ≈41; +10 model / +6 method; promise "how" | native morph |
| 2 | 1 Motivation | Task & metric | Distributional task; score vs uniform | native diagram |
| 3 | 1 | Why it's hard | The intuitions backfire | **real** example items |
| 4 | 2 Apparatus | The automated scientist | Spine + guarded search loop | native diagram |
| 5 | 2 | Validation discipline | Leave-family-out · sealed test · prereg | native diagram |
| 6 | 2 | The Ladder gate *(formal)* | η-gated val → bounded generalization | **real** val staircase |
| 7 | 3 Journey | Start naive | "Just ask" baseline + its failure signature | **real** hist + entropy |
| 8 | 3 | Graveyard I | Simulating people loses | **real** bars + CIs |
| 9 | 3 | Graveyard II | Thinking harder loses | **real** voting catastrophe |
| 10 | 3 | Graveyard III | Can't recalibrate your way out | **real** calibration |
| 11 | 4 Turn | Simplicity, earned | The 3 survivors = fixes to failures | **real** survivor Δ |
| 12 | 4 | The scoreboard | Full ablation receipt | table (color-coded) |
| 13 | 5 Payoff | The result | Sealed test +5.5 · paper reproduced | **real** CI progression |
| 14 | 5 | Model vs method *(honest)* | +10 model / +6 method; model's gain is mostly grouped; fit > size | **real** decomp + sweep |
| 15 | 5 | Required questions | 3 mandated Qs, predicted vs truth | **real** distributions |
| 16 | 5 | Counterfactual sensitivity | Right-direction conditioning (weak, gated) | **real** cf bars |
| 17 | 6 Limits | Limitations & risks | The honest ceiling | **real** error decomp |
| 18 | 6 | Extensions + close | Each a callback → bridge to Part II | native register |

**Dark slides for rhythm:** 1 (title), 4 (apparatus), 13 (result), 18 (close). Pacing ≈ 2.5 min/slide. Acts 2–4 are the heart — give them air.

---

## Design system

- **Type:** Spectral (display serif — titles, big numbers) · IBM Plex Sans (body) · IBM Plex Mono (eyebrows, data, sources).
- **Palette:** paper `#F3EFE7` · ink `#1B1A17` · secondary `#8C8579` · dark `#16151A` / `#ECE7DC`.
- **Semantic color (consistent everywhere, incl. figures):** green `#2E8568` = won & transferred · amber `#B5821E` = won on dev, didn't transfer · grey `#A39D92` = negative / baseline. **Green peak = "signal."**
- **Motif:** flat (uniform) → peaked (human) bars — flat = baseline, peaked = signal. Resolved at the close.
- **No emoji** in slides; colored dots for verdicts. Footnote the **Source** on every data slide.
- **Reported metric:** **split-avg S** — unweighted mean of the population & demographic-segment split scores (the SimBench paper's Table-1 aggregation; paper-comparable). Coincides with the record-pooled "overall" on our final system (40.8 vs 40.9). Headline anatomy is measured one change at a time, each step holding everything else fixed.

---

## Visual production plan *(the data-science rigor)*

**Principle:** real results get **real figures** — labeled axes with units, appropriate scales, bootstrap CIs — generated from the actual run data, not hand-drawn. Conceptual visuals (pipeline, splits, motif) stay native SVG/CSS.

- **New build script: `scripts/build_deck_figures.py`** — imports `scrye.viz` / `scrye.decompose` / `scrye.results`, applies a **deck matplotlib style** (rcParams: IBM Plex Sans/Mono, the palette above, despined, sized to each slide's figure region), loads the relevant `outputs/runs/*.results.json`, and writes **SVG** (vector, crisp on any projector; 2× PNG fallback) into `docs/presentation/assets/`. Figures are **regenerable from data** — numbers are never typed in by hand.
- **Existing toolkit reused:** `plot_score_histogram`, `plot_score_vs_entropy`, `plot_calibration`, `plot_example_distributions`, `plot_system_comparison`, `plot_required_question_distributions` (all return matplotlib `Figure`); `decompose_records` / `system_summary`; `bootstrap_ci`.
- **Data dependency:** `outputs/runs/`, `outputs/ledger/`, and `data/cache/` (699 MB, 178k calls, temp=0 + seeded → byte-identical replay) live in the **main checkout**, not the worktree (gitignored). The figure script must run where that data is reachable; SVGs are then committed into the worktree. The existing `docs/figures/*.png` are **baseline (`zero_shot_identity`)** cuts — most slides need a **final-system** regeneration.
- **Embedding:** each slide keeps its eyebrow / headline / source line; the figure `<img src="assets/<name>.svg">` fills the slide's chart region.

**Figure → run-data map:**

| Figure | viz function | Run |
|---|---|---|
| baseline score hist + score-vs-entropy | `plot_score_histogram`, `plot_score_vs_entropy` | `2026-06-21-decomp-final` (faithful) |
| example failure items (slide 3) | `plot_example_distributions` + `decompose_records` | `2026-06-21-decomp-final` |
| graveyard bars w/ CIs | `plot_system_comparison` | `2026-06-20-persona-mechanisms`, `2026-06-21-voting-sim`, Stage 18 (nb02) |
| calibration scatter | `plot_calibration` | baseline / `2026-06-21-decomp-final` |
| survivor Δ | `plot_system_comparison` | Stage 05/10/11 runs |
| val staircase (Ladder) | `results.best_progression` + val-confirm runs | Stage 05/12/16 |
| test CI progression + paper repro | `plot_system_comparison` | `2026-06-21-TEST-lineage`, `2026-06-21-TEST-faithful-models` |
| model sweep 3.1 vs 3.5 | `plot_system_comparison` (grouped by model) | `2026-06-20-model-sweep` |
| required-Q distributions | `plot_required_question_distributions` | final test cc+abstain @ 3.1 |
| cf_alignment by system | bar from `*.topline.csv` `cf_alignment` | Stage 01/04/05 toplines |
| error decomposition stacked | `system_summary` / `decompose_records` | `2026-06-21-decomp-final` |

---

## Slide-by-slide

### Act 0 — Hook

**Slide 1 — Title + the shift** *(dark)*
- **Key:** We took the published SimBench baseline from **≈25 to ≈41** — and the talk is *how*, and how we know it's real.
- **Content:** "Scrye — Predicting the distribution of human survey responses." Sealed-test **split-avg S ≈ 25 → ≈ 41**: the faithful SimBench baseline (Qwen2.5-72B; reproduces the paper's 27.6) → our system. Honest topline anatomy: **+10 from a better-fit base model · +6 from our method** (scale: 0 = uniform, 100 = perfect). Presenter · date. *(The detailed anatomy — and what it cost us in failed ideas — is the talk.)*
- **Visual:** the morph motif — flat uniform (S=0) dissolving into a peaked human distribution — carrying a labeled shift **25 → 41**; uniform 0 marked on the scale. native SVG/CSS.
- **Source:** `docs/experiments/stage-17-final-test.md`.

### Act 1 — Motivation & intuition

**Slide 2 — Task & metric**
- **Key:** Predict the *whole distribution* of human answers, scored against "no information."
- **Content:** Input = question + discrete options + *optional* demographic segment (may be empty = full population; no ground-truth access; must generalize to unseen questions/segments). **S = 100·(1 − TVD(P,Q)/TVD(P,U))** — 0 = uniform, 100 = exact, negative = worse than guessing. TVD = ½·Σ|pᵢ−qᵢ| = "share of probability mass in the wrong place." Three required questions sealed into test.
- **Visual:** input → predicted-histogram (Q vs P) diagram + the formula on a number line (uniform 0 → perfect 100). native.
- **Source:** `src/scrye/scoring.py` (`simbench_score`); SimBench Eq. 2.

**Slide 3 — Why this is genuinely hard**
- **Key:** The intuitive moves *backfire* — which is why a careful method matters.
- **Content:** (1) Conditioning over-concentrates — first-person persona *embodiment* collapses toward the mode. (2) Value–action gap — stated ≠ revealed; gambles & moral dilemmas resist survey-style prompting. (3) Spread vs location are entangled — right shape / wrong place, or right place / wrong spread; fixing one perturbs the other.
- **Visual:** **real** — three actual items from the data, each a P-vs-Q bar pair illustrating one failure mode (over-concentrated · right-mode/wrong-spread · right-spread/wrong-mode), pulled via `plot_example_distributions` + `decompose_records`. Labeled options (x) and probability (y). *Fallback:* clean native triptych if the real items read noisily at slide scale.
- **Source:** `src/scrye/decompose.py`; run `2026-06-21-decomp-final`. Stages 03 / 08–09 / 13.

### Act 2 — The apparatus *(how to search without fooling yourself)*

**Slide 4 — The automated scientist** *(dark)*
- **Key:** With ~18 ideas tried on one dataset, the real adversary is fooling yourself — so the search is honest *by construction*.
- **Content:** Swappable spine **Record → Predictor → Calibrator → score** (a new idea = one subclass + registry entry). The guarded loop: **propose lever → score on dev → if dev-best, one Ladder-gated val query**; halts on budget / off-registry proposal / gate fire.
- **Visual:** native two-panel — the pipeline (left) + the gated funnel (right): many candidates → dev → η-gated val → **sealed test**. Adapt `docs/figures/architecture-diagram.html`.
- **Source:** `predict.py` / `calibrate.py` / `pipeline.py`; `search.py`; `experiment.py`; `docs/experiments/README.md`.

**Slide 5 — Validation discipline**
- **Key:** Don't let the test leak; fix the rule before you look.
- **Content:** **Leave-family-out** splits — by `(dataset, input_template)` family, never random rows; required Qs **pinned to test**; **test sealed** until the final run. Preregistered stages — decision rule fixed *before* data. Split: seed 0, unit = question, dev/val/test = 0.5/0.25/0.25.
- **Visual:** native — the split engine: question families → dev/val/test buckets, required-Q lock → test, sealed-test vault.
- **Source:** `splits.py`; `manifest.py`; `docs/experiments/README.md`.

**Slide 6 — The Ladder gate** *(the formal justification)*
- **Key:** A val win counts only if it clears a noise floor → a real bound on generalization error under adaptive reuse.
- **Content:** **Blum & Hardt 2015 (Ladder):** accept a val improvement only if it beats the running best by more than **η**, a bootstrap-derived noise floor → bounds generalization error under adaptive search. Reproducible by construction: ledger DAG + manifest + the cache (178k calls, temp 0, seeded → byte-identical re-runs, free).
- **Visual:** **real** — the validation staircase (pooled val score climbing through *passed gates*): faithful@3.1 **36.90** → anti-flattening **43.43** (Stage 05 ✓) → cc+abstain **45.38** (Stage 12 ✓) → router **47.01** (Stage 16 ✓); each step annotated as a gate. From `results.best_progression` + val-confirm runs.
- **Source:** `ladder.py`; `results.py`; Stages 05 / 12 / 16.

### Act 3 — The journey / the graveyard

**Slide 7 — Start naive**
- **Key:** Just ask the model — here's where the honest baseline sits, and how it fails.
- **Content:** Faithful zero-shot (`simbench_faithful`) at the deployed model. The failure signature: a worse-than-uniform tail and **mode-seeking** — the model over-commits, collapsing spread on divided questions.
- **Visual:** **real** — score histogram (`plot_score_histogram`) + score-vs-truth-entropy (`plot_score_vs_entropy`, binned-mean overlay) showing scores fall as questions get more divided. Labeled axes; entropy 0 (consensus) → 1 (divided).
- **Source:** `viz.py`; run `2026-06-21-decomp-final` (faithful system).

**Slide 8 — Graveyard I: simulating people loses to asking about people**
- **Key:** Making it more "human" backfires.
- **Content:** Persona *embodiment* over-concentrates (collapses to mode). **Monte-Carlo individuals −15.7 grouped** (29.28 [25.1, 33.2] vs ~45 base). **1M-persona Nemotron census electorate:** best 51.8 vs the simple champion **63.2 (−11.4)**; 78% mode-agreement, no ideology axis.
- **Visual:** **real** — grouped-score bars vs the simple base, with bootstrap CIs, negatives shown honestly (`plot_system_comparison`).
- **Source:** Stage 03 (`2026-06-20-persona-mechanisms`); Stage 18 (`stage-18-nemotron-personas.md`).

**Slide 9 — Graveyard II: thinking harder loses to the direct ask**
- **Key:** Elaborate reasoning and aggregation backfire.
- **Content:** Superforecaster / reasoning-first / diversity-elicitation CoT all lose to the direct ask (**−2.7 to −3.1 grouped**). Discrete voting is catastrophic where the mode is wrong: **MoralMachine −157.86** [−172.8, −143.4] vs uniform −2.73 (confident, wrong-mode); Choices13k's +15.6 **didn't transfer** (pooled hypothesis refuted).
- **Visual:** **real** — the voting catastrophe as the hero bar (MoralMachine vs uniform, CIs) + the CoT deltas.
- **Source:** Stages 07 / 13 (`2026-06-21-voting-sim`) / 03.

**Slide 10 — Graveyard III: you can't recalibrate your way out**
- **Key:** Fixing the numbers after the fact backfires.
- **Content:** Post-hoc calibration (temperature scaling, etc.) all lose to the **identity** calibrator. Entropy de-compression isn't recoverable from the model's own entropy (slope **0.48** — it can't predict its own error).
- **Visual:** **real** — calibration reliability scatter (`plot_calibration`, perfect-calibration diagonal) showing the miscalibration; small entropy-recompression slope inset.
- **Source:** Stages 06 / 08 / 09.

### Act 4 — The turn

**Slide 11 — Simplicity, earned**
- **Key:** After all that, what held were a few simple, principled moves — each a fix to a failure you just watched.
- **Content:** **Distributional framing (anti-flattening)** — preserves minority mass (fixes over-concentration): +2.81 dev / **+4.26 test** grouped vs faithful@3.1, val-confirmed (53.0 val grouped). **Calibrated-commitment prompt** — fixes *location* / mode accuracy (one call; cc+abstain **+5.72 on test**). **Abstention** — uniform where we reliably fail (honest about uncertainty): +3.1 pop dev, **+3.26 pop val**, val-confirmed (Stage 12).
- **Visual:** **real** — survivor Δ bars with CIs, each labeled with the failure it fixes.
- **Source:** Stages 05 / 10 / 11 / 12.

**Slide 12 — The scoreboard** *(ablation receipt)*
- **Key:** The full ledger of what won, lost, and didn't transfer — now the receipt, not the spoiler.
- **Content (table, color-coded):**

| Lever | Verdict |
|---|---|
| Distributional framing (anti-flattening) | ● won — +4.26 test grouped |
| Base-model choice (gemini-3.1-flash-lite) | ● won — dominant, +10 split-avg vs published baseline |
| Calibrated-commitment prompt (fixes *location*) | ● won — cc+abstain +5.72 test |
| Abstention (uniform where we reliably fail) | ● won — +3.26 pop val |
| Task-kind routing | ◐ won on val, **didn't transfer** (router = cc+abstain on test) |
| Monte-Carlo personas · census electorate · voting · CoT · post-hoc calibration | ○ lose |

- **Visual:** the hero table (green/amber/grey dots); optional cumulative-gain companion.
- **Source:** `docs/experiments/README.md` stage table (01–18).

### Act 5 — The triple-honesty payoff

**Slide 13 — The result** *(dark)*
- **Key:** On a test set untouched until the final run, the method lifts faithful by **+5.5** (all CIs exclude 0) — and the harness reproduces the paper.
- **Content:** Holding the model fixed at gemini-3.1, our method lifts faithful **35.0 → 40.8 split-avg** (+5.8; overall +5.7 [+4.3, +6.8], CI excludes 0). Router **ties** cc+abstain on test → we report the **simpler** cc+abstain. Paper reproduction: faithful @ Qwen2.5-72B = **26.8 [24.4, 29.4] split-avg** brackets the paper's **27.6** ✓ (clean stratified sample).
- **Visual:** **real** — test CI progression (baseline → final, error bars) + a number-line showing our CI band with 27.61 inside it.
- **Source:** Stage 17 (`2026-06-21-TEST-lineage`, `2026-06-21-TEST-faithful-models`).

**Slide 14 — The honest headline: model vs method** *(dark)*
- **Key:** The model did most of the work; the method earns the margin on top — and even the model story is *fit, not size*.
- **Content:** The gap from the **published baseline** decomposes cleanly, one change at a time (split-avg, sealed test): faithful @ **Qwen2.5-72B 25.0** → swap to **gemini-3.1-flash-lite 35.0** (**+10**, same prompt) → add **our method 40.8** (**+5.8**). So **~⅔ model, ~⅓ method**. Honest nuance in the splits: the model's gain is **almost all on grouped/demographic questions (+18) and barely on pop (+1.8)** — *our method carries pop (+6.7)*. And it's **fit, not size/recency**: gemini-3.1-flash-lite ≥ gemini-3.5-flash on every system (anti-flat 41.8 vs 40.2; contextualized 42.0 vs 35.8; faithful 38.1 vs 35.4), and cheaper. Caveat: we **did not rigorously sweep the model space**.
- **Visual:** **real** — the two-step decomposition across **grouped / pop / split-avg** with CIs (Qwen → gemini-3.1 → our system); companion grouped bar gemini-3.1 vs 3.5 across the three strategies.
- **Source:** Stages 02 (`2026-06-20-model-sweep`) / 17.

**Slide 15 — The required-question predictions**
- **Key:** For the three mandated questions — first seen at final evaluation — predicted distributions against human truth.
- **Content:** trust_president · gay_rights · internet_use; predicted Q vs human P, per-question S; pinned to test throughout. Pooled required-Q progression **−1.3 (faithful@Qwen2.5-72B) → +51.5 (faithful@3.1) → +55.3 (our system)**.
- **Visual:** **real** — `plot_required_question_distributions`: three predicted-vs-actual distributions, labeled options + probability axis, each with its S badge. *(Replaces the old schematic histograms.)*
- **Source:** Stage 17; notebook 03 (cell 32). **VERIFY:** extract the three per-question scores from the regenerated figure.

**Slide 16 — Counterfactual sensitivity** *(required metric)*
- **Key:** Does conditioning shift the prediction in the *right direction*? We measure it and gate on it — and we're honest that the signal is weak.
- **Content:** `cf_alignment` = alignment of the **predicted** shift (segment − population) with the **ground-truth** shift; +1 right · −1 wrong · 0 orthogonal. Used as a **non-regression gate** (`cf_alignment ≥ baseline`) so accuracy can't be bought by getting directions wrong. Honest finding: **weak and weakly differentiated** — ~0.22 (Stage 01 dev) → ~0.39–0.41 (full dev) → 0.42 (val); does **not** separate strategies; rises with model quality → also model-bound.
- **Visual:** **real** — cf_alignment by system/stage with overlapping CIs; native shift-arrow inset (truth vs predicted, angle = alignment).
- **Source:** `scoring.py` (shift-alignment); Stage 01/04/05 toplines.

### Act 6 — Limitations, risks, extensions

**Slide 17 — Limitations & risks**
- **Key:** The honest ceiling.
- **Content:** **Limits** — spread still ~0.06 too diffuse (concentration error dominates the loss); behavioral / value-action tasks stay hard; under-represented slices harder; single-model dependence (model space unswept); bottom-up simulation never beat the direct ask. **Risks** — over-fit to SimBench's templates (the apparatus guards but doesn't eliminate it); "better model" gains may erode the method margin as base models improve; leakage risk in any tool-using extension.
- **Visual:** **real** — error-decomposition stacked bars (concentration vs location loss, faithful vs final) showing concentration dominates; entropy-calibration scatter (over-diffuse) as companion.
- **Source:** `decompose.py`; notebook 04 (`2026-06-21-decomp-final`); Stages 09 / 16.

**Slide 18 — Extensions + close** *(dark)*
- **Key:** The live frontier — each item a response to a failure we showed; the discipline is the transferable asset.
- **Content:** **Extensions** — *Agentic tool-use* (`WebAgentPredictor`: leakage-guarded, must beat a closed-book twin **and** the champion) → answers *no real-world signal*. *Multi-agent critique / debate* (surface minority mass, catch over-concentration) → answers *the persistent spread failure*. *More expressive multi-step prompting* — honestly flagged: our naive CoT **lost**, so promising-but-unproven → callback to the graveyard. *Fuller, principled model sweep* → callback to model-vs-method. **Close:** beat the baseline on a sealed test, reproduced the paper, demoted the wins that didn't transfer — the apparatus generalizes to any LLM-eval problem. → *Part II takes it from benchmark to commercial feedback loop.*
- **Visual:** native — compact extensions register (each tagged with the failure it answers) + the morph motif resolved (prediction overlaid on truth) with the one-line thesis.
- **Source:** Stages 13 / 16 / 18; bean `Scrye_Project-60fk`; `docs/part-ii-feedback-architecture.md`.

---

## Backup slides *(Q&A only)*

One-liners; expand from the cited source if discussion goes there.

- **B1 — Anti-flattening, mechanically:** embodiment collapses to the mode; distributional framing preserves minority mass. Val 53.0 vs 47.0. *Stage 01/05.*
- **B2 — Location vs spread:** oracle headroom is gated behind mode correctness; entropy stayed ~0.06 too diffuse throughout. *Stages 09/16; `decompose.py`; nb 04.*
- **B3 — Abstention + routing internals:** upfront LLM task-classifier → fixed kind→intervention map, *zero* per-dataset params; routed+abstain beats base cc +8.75 dev (representative). *Stages 11/15.*
- **B4 — The graveyard in detail:** calibration (all lose to identity); superforecaster/CoT (lose ×2); voting (MoralMachine −158); census electorate (78% mode-agreement, no ideology axis). *Stages 06/07/08/13/18.*
- **B5 — Ladder / η derivation:** bootstrap spread → η → accept rule; val touched only at Stages 05/12/16. *`ladder.py`, manifests.*
- **B6 — Per-dataset & cost:** pop vs grouped by dataset; tokens/$ per stage; cache makes re-runs free. *nb 03; `score_by_dataset`; `llm.py`.*
- **B7 — Prompt-strategy gallery:** the five `PromptStrategy` styles verbatim. *`persona.py`.*
- **B8 — The difficulty landscape:** truth-entropy spectrum (pop vs grouped), tertile bins. *nb 04.*

---

## Open items / VERIFY before render

1. **Per-question required-Q scores** (slide 15) — extract trust_president / gay_rights / internet_use individually from the regenerated `plot_required_question_distributions` figure (the inventory pooled them).
2. **Opener framing — RESOLVED:** lead with split-avg **≈25 → ≈41** (published Qwen2.5-72B faithful → our system); topline anatomy **+10 model / +6 method**. gemini-2.5 baseline dropped (not a real comparison). Deck reports **split-avg** throughout.
3. **Figure regeneration runs against the main checkout** (where `outputs/` + `data/cache/` live); SVGs committed into the worktree.
