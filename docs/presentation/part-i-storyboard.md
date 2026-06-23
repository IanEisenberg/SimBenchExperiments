# Part I — Presentation Storyboard (10-slide cut)

**Scrye: Predicting Human Survey-Response Distributions with LLMs**

A **tight 10-slide main flow** for Part I, sized for ~25 min inside a 60-min session (Part I + Part II + infrastructure). Hand to **Claude Design** to render; refine visuals there. The deep-dive slides are demoted to **Backup** at the bottom — pull them only if the discussion goes there.

> **Scope:** Part I only. **Spine:** *disciplined empirical discovery* — we submit a method for finding methods that doesn't fool itself. **Detail:** medium + visual intent; every data-bearing slide carries a **Source:** pointer so each number is verifiable.
>
> **Per slide:** Key message · Content (bullets) · Visual · Source. **Legend:** ✅ won & transferred · ⚠️ won on dev, didn't transfer · ❌ negative · 🔒 sealed-test.

## Deck at a glance (10 slides ≈ 25 min)

| # | Slide | Job |
|---|---|---|
| 1 | Title + bottom line | Thesis + headline number up front |
| 2 | The problem & the metric | Distributional task; score vs uniform baseline |
| 3 | Why it's hard | The three traps (the intuition) |
| 4 | A method for finding methods | The discovery apparatus — *the differentiator* |
| 5 | The lever scoreboard | What won / lost / didn't transfer (ablations) |
| 6 | The honest headline | Model is the dominant lever |
| 7 | Counterfactual sensitivity | Right-direction conditioning (required metric) |
| 8 | The result | Sealed test + paper reproduction |
| 9 | Required-question predictions | The three mandated questions |
| 10 | Limits, next, close | Honest limits → frontier → bridge to Part II |

Pacing: ~2.5 min/slide. Slides 4–6 are the heart; give them air. Everything in **Backup** is for Q&A only.

---

## Slide 1 — Title + bottom line up front

- **Key message:** We beat the SimBench baseline on a *sealed* test set and validated against the published paper — and the real contribution is a search that knows which of its own wins are real.
- **Content:**
  - Scrye — Predicting Human Survey-Response Distributions · *Beating the uniform baseline without fooling ourselves* · presenter · date.
  - 🔒 Sealed test: faithful **35.21 → ~40.9**, every CI excludes 0.
  - Honest split: model swap **+15.9**, method stack **+5.5** (we report both).
  - Harness reproduces the paper (26.83 brackets 27.61).
- **Visual:** Title with three stat cards (40.9 / +5.5 method / +15.9 model). Faint backdrop motif: a flat/uniform distribution morphing into a peaked human one.
- **Source:** `docs/experiments/stage-17-final-test.md`.
- **VERIFY:** Pin headline number — stage-17 row says **40.73** (router full-lineage), exec-overview bean says **40.93** (cc+abstain); state which system is the headline.

## Slide 2 — The problem & the metric

- **Key message:** Predict the *whole distribution* of human answers (not the top answer), scored so 0 = the naive uniform baseline and 100 = perfect — every point is earned over "no information."
- **Content:**
  - Input: question + discrete choices + optional demographic segment (may be empty = full population). Must generalize to unseen questions/segments; no ground-truth access.
  - **S = 100 · (1 − TVD(P,Q) / TVD(P,U))** — 0 = uniform, 100 = exact, negative = worse than guessing.
  - TVD = ½·L1: "how much probability mass is in the wrong place."
  - Three required questions, sealed into test: trust in president · gay rights · internet-use.
- **Visual:** Left: input→distribution diagram. Right: the formula on a number line (−∞ ← 0 uniform → 100 perfect) with a small shaded P-vs-Q TVD inset.
- **Source:** Assignment brief; `src/scrye/scoring.py` (`simbench_score`); SimBench paper Eq. 2.

## Slide 3 — Why this is genuinely hard

- **Key message:** The intuitive moves backfire — that's why a careful method matters.
- **Content:**
  - **Conditioning backfires:** first-person persona *embodiment* over-concentrates and loses (Monte-Carlo individuals −15.7).
  - **Value–action gap:** stated preferences ≠ revealed behavior; gambles and moral dilemmas resist survey-style prompting.
  - **Spread vs location are entangled:** right shape / wrong place, or right place / wrong spread — fixing one perturbs the other.
- **Visual:** Triptych of small P-vs-Q mini-charts, one per trap (collapsed mass · right-mode-wrong-spread · right-spread-wrong-mode).
- **Source:** Stage 03, Stages 08–09, Stage 13. `docs/experiments/`.

## Slide 4 — A method for finding methods *(the differentiator)*

- **Key message:** With 18 ideas tried on one dataset, the real adversary is fooling yourself — so we made the search honest *by construction*.
- **Content:**
  - Swappable spine: **Record → Predictor → Calibrator → score**; a new idea = one subclass + registry entry.
  - **Leave-family-out splits:** split by question family, not random rows; required Qs pinned to test; **test sealed** until the end.
  - **Ladder gate (Blum & Hardt 2015):** a val result counts only if it beats the best by more than a bootstrap-derived noise floor η → bounds generalization error under adaptive search.
  - **Reproducible:** preregistered stages (decision rule fixed before data) + ledger DAG + manifest + full LLM cache → every number regenerates for free.
- **Visual:** Two-panel: (left) the Record→Predictor→Calibrator pipeline; (right) the gated funnel — many candidates → dev → η-gated val → sealed test. Use the existing architecture diagram as the left panel.
- **Source:** `splits.py`, `ladder.py`, `spec.py`/`ledger.py`/`manifest.py`; `docs/figures/architecture-diagram.html`; `docs/experiments/README.md`.

## Slide 5 — The lever scoreboard *(ablations)*

- **Key message:** A handful of real, transferable wins — and an equal number of instructive failures, each kept and reported.
- **Content (table):**

| Lever | Verdict |
|---|---|
| Distributional framing (anti-flattening) | ✅ +6.0 val grouped |
| Model swap (2.5 → gemini-3.1-flash-lite) | ✅ dominant, +15.9 |
| Calibrated-commitment prompt (fixes *location*) | ✅ confirmed val |
| Abstention (uniform where we reliably fail) | ✅ pop +3.26 val |
| Task-kind routing (classify → intervention) | ✅ grouped +1.89 val |
| Post-hoc calibration · reasoning-first · Monte-Carlo personas | ❌ lose |
| Discrete voting | ⚠️ Choices13k win didn't transfer |
| Grounded census electorate | ❌ −11 vs simple prompt |

- **Visual:** The hero table, color-coded ✅/⚠️/❌. Optional companion: cumulative score gain across stages.
- **Source:** `docs/experiments/README.md` stage table (01–18).

## Slide 6 — The honest headline: the model is the dominant lever

- **Key message:** Switching to a stronger base model added **+15.9**; our entire method stack added **+5.5** on top — a strong submission says this out loud.
- **Content:**
  - faithful, 2.5 → gemini-3.1-flash-lite: **+15.9 overall / +25.4 grouped**.
  - Full preregistered method stack: **+5.5** on sealed test.
  - Implication: most available signal is "use a better base model"; methods earn the margin — and we proved the margin survives a sealed test.
- **Visual:** Stacked bar — baseline → +model jump (big) → +method stack (smaller) → final, each segment labeled with its delta.
- **Source:** Stage 17. `docs/experiments/stage-17-final-test.md`.

## Slide 7 — Counterfactual sensitivity *(assignment-required)*

- **Key message:** Beyond accuracy, does conditioning shift the prediction in the *right direction*? We measure it and use it as a guardrail.
- **Content:**
  - `cf_alignment`: alignment of the **predicted** shift (segment − population) with the **ground-truth** shift. +1 right · −1 wrong · 0 orthogonal.
  - Used as a **non-regression gate** (`cf_alignment ≥ baseline`): accuracy can't be bought by getting directions wrong.
  - Honest finding: directional sensitivity is **modest and weakly differentiated** (~0.40–0.45 for the better model); it rises with model quality → also model-bound.
- **Visual:** Two shift arrows (truth vs predicted), angle = alignment; small side bars of cf_alignment by system with overlapping CIs.
- **Source:** `src/scrye/scoring.py` (shift-alignment, ~line 130); Stages 01/04/05/07.

## Slide 8 — The result: sealed test + paper reproduction

- **Key message:** On a test set untouched until the final run, the method lifts the baseline by ~+5.5 (all CIs exclude 0) — and our harness brackets the paper's published number.
- **Content:**
  - 🔒 faithful **35.21 → ~40.9** on sealed test; all CIs exclude 0.
  - Router's +1.1 val edge **did not transfer** → we report the simpler **cc + abstain**.
  - Credibility anchor: faithful @ Qwen2.5-72B **S = 26.83 [24.4, 29.4]** brackets the paper's **27.61** ✓.
- **Visual:** Split panel — (left) progression bar with CIs baseline → final; (right) a number line showing our CI band with the paper's 27.61 falling inside it.
- **Source:** Stage 17 (final test + paper-reproduction sections).
- **VERIFY:** reconcile 40.73 vs 40.93 and which system is reported.

## Slide 9 — The required-question predictions

- **Key message:** For the three mandated questions — first seen at final evaluation — here are our predicted distributions against the human ground truth.
- **Content:**
  - Trust in president · Gay rights · Internet-use frequency.
  - Predicted vs actual bars + per-question score; note these were **pinned to test**.
- **Visual:** Three small predicted-vs-actual bar-pairs, each with its S badge.
- **Source:** Notebook 03 / `best_analysis_run`; Stage 17. **VERIFY:** pull the three predicted distributions + scores from the final-test run before rendering.

## Slide 10 — Limitations, what's next, and the bridge

- **Key message:** We're honest about the ceiling, the live frontier is tool-use, and the discipline is the transferable asset — which is exactly what Part II builds on.
- **Content:**
  - **Limits:** spread still ~0.06 too diffuse; behavioral + Global-South slices remain hard; bottom-up sim < direct ask; single-model dependence.
  - **Next:** `WebAgentPredictor` — a leakage-guarded tool-using agent (identity stripping + domain blocklist + audit), must beat a closed-book twin *and* the champion.
  - **Close:** beat the baseline on sealed test, reproduced the paper, demoted non-transferring wins — the apparatus generalizes to any LLM-eval problem. → *Part II takes it from benchmark to commercial feedback loop.*
- **Visual:** Left: compact risk register (risk · evidence · next). Right: the Slide 1 distribution motif resolved (prediction overlaid on truth) with the one-line thesis.
- **Source:** Stages 13/16/18; `Scrye_Project-60fk` bean; `docs/part-ii-feedback-architecture.md`.

---

## Backup slides *(Q&A only — not in the main 10)*

One-liners; expand from the cited source if the discussion pulls there.

- **B1 — Why anti-flattening wins:** persona embodiment collapses to the mode; distributional framing preserves minority mass. Val 53.0 vs 47.0. *Stage 01/05.*
- **B2 — Why location, not spread:** oracle headroom is gated behind mode correctness (+16.8 where top option right, −2.0 on the 39% where wrong); entropy stayed ~0.06 too diffuse throughout. *Stages 09/16, `decompose.py`, notebook 04.*
- **B3 — Abstention + routing mechanics:** an upfront LLM task-classifier → fixed kind→intervention map with *zero* per-dataset params; routed+abstain beats base cc +8.75 dev [+5.94, +11.67]. *Stages 11/15, `taskkind.py`.*
- **B4 — The graveyard, in detail:** calibration (all lose to identity), superforecaster/CoT (lose to direct ask ×2), entropy de-compression (not recoverable from own entropy, slope 0.48), voting (MoralMachine −158), census electorate (78% mode-agreement, no ideology axis). *Stages 06/07/08/13/18.*
- **B5 — Ladder / η derivation:** bootstrap spread → η → accept rule; val touched only at stages 05/12/16. *`ladder.py`, manifests.*
- **B6 — Per-dataset breakdown & cost:** pop vs grouped by dataset; tokens/$ per stage; cache makes re-runs free. *Notebook 03, `docs/figures/score_by_dataset__*.png`, `llm.py`.*
- **B7 — Prompt-strategy gallery:** the five `PromptStrategy` styles verbatim. *`persona.py`.*

---

## Design direction for Claude Design *(optional)*

- **Tone:** research-credible, not salesy; one idea per slide; lots of whitespace.
- **Motif:** uniform→peaked distribution morph as recurring visual language (flat = baseline, peaked = signal).
- **Color coding:** one accent for ✅ wins, grey for ❌ negatives, amber for ⚠️ didn't-transfer — consistent on the scoreboard.
- **Footnote the Source** on every stat slide; reviewers will ask "where's that from."
- **Charts over bullets** on slides 3, 6, 8, 9.
