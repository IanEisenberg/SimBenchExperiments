# Part I — Presentation Storyboard

**Scrye: Predicting Human Survey-Response Distributions with LLMs**

This is a **slide-by-slide storyboard** for the Part I research-candidate talk. It is the *content + intent* layer — titles, key messages, bullets, the visual each slide wants, and a source pointer for every number. Hand it to **Claude Design** to turn into rendered slides; refine visuals there.

> **Scope:** Part I only (the simulation work). Part II (feedback architecture) and Infrastructure get their own storyboards later. The full deliverable is a 60–90 min discussion; Part I is the bulk of it.
>
> **Narrative spine:** *Disciplined empirical discovery.* Most candidates submit "a method." We submit **a method for finding methods that doesn't fool itself** — preregistration, leave-family-out splits, a Ladder gate, and a reproducible ledger. The results are evidence the process works.
>
> **Detail level:** Medium + visual intent. Every data-bearing slide carries a **Source:** pointer (stage file / figure / module) so each claim is verifiable.

## How to read this doc

Each slide has:
- **Title** — slide headline
- **Key message** — the one sentence the audience should leave with
- **Content** — 3–5 bullets (speaker fills the rest)
- **Visual** — what to draw/show and rough layout
- **Source** — where the number/claim lives (so Claude Design and reviewers can verify)
- **VERIFY** *(when present)* — a number to reconcile against the stage file before final

**Legend used on result slides:** ✅ won & transferred to held-out · ⚠️ won on dev but didn't transfer · ❌ negative result · 🔒 sealed-test number.

---

## Deck at a glance

| Act | Slides | Purpose |
|---|---|---|
| 0 · Frame | 1–2 | Title + bottom-line-up-front |
| 1 · The problem | 3–6 | Task, metric, why it's hard, the data |
| 2 · A method for finding methods | 7–11 | Pipeline + the discovery apparatus (the differentiator) |
| 3 · The lever story | 12–19 | What won, what lost, and the honest model-dominance finding |
| 4 · Conditioning sensitivity | 20 | The assignment-required counterfactual metric |
| 5 · The result | 21–23 | Sealed-test headline, paper reproduction, required-question predictions |
| 6 · Reflection | 24–26 | Limitations, what's next, closing thesis |
| Appendix | A1–A6 | Full tables, derivations, prompt gallery, cost |

Target: **~26 main slides + appendix**. Trim Act 3 to fit a shorter slot; the appendix absorbs the depth a discussion will pull on.

---

# Act 0 — Frame

## Slide 1 — Title

- **Key message:** A disciplined search for an LLM method that predicts *population opinion distributions* and knows which of its own wins are real.
- **Content:**
  - Scrye — Predicting Human Survey-Response Distributions with SimBench
  - Subtitle: *Beating the uniform baseline without fooling ourselves*
  - Presenter name · date · "Part I — Simulation Implementation"
- **Visual:** Clean title slide. Optional faint backdrop: a distribution morphing from flat/uniform into a peaked human response shape (the whole project in one motif).

## Slide 2 — Bottom line up front

- **Key message:** We beat the SimBench baseline on a *sealed* test set, validated the harness against the published paper, and — honestly — found the model swap did most of the work.
- **Content:**
  - 🔒 Sealed test: faithful baseline **35.21 → ~40.9** (final system), every CI excludes 0.
  - The method stack adds **+5.5**; switching to a better model added **+15.9**. We report both.
  - Harness sanity check: reproduces the paper (S = 26.83 brackets the paper's 27.61).
  - The real contribution is the *process* — preregistered, leakage-disciplined, reproducible.
- **Visual:** Three big stat cards (40.9 / +5.5 method / +15.9 model) over a thin timeline of 18 stages.
- **Source:** `docs/experiments/stage-17-final-test.md`; headline reconciled with `Scrye_Project-sn3m` bean.
- **VERIFY:** Final-system test score — stage-17 row says full-lineage **40.73**; exec-overview bean says cc+abstain **40.93**. Pin the exact number + which system (cc+abstain vs router) on this slide.

---

# Act 1 — The problem

## Slide 3 — The task

- **Key message:** Predict the *whole distribution* of human answers — not the most likely answer — for a question, its choices, and an optional demographic segment.
- **Content:**
  - Input: survey question + discrete answer choices + optional conditioning segment (age/gender/…); segment may be empty = full population.
  - Output: a probability distribution over the choices.
  - Hard constraint: must **generalize** to unseen questions and segments; no ground-truth access.
  - Three required questions (always evaluated): trust in president · gay rights · internet-use frequency.
- **Visual:** Input→output diagram: a question card → LLM box → bar-chart distribution. Show one required question as the worked example.
- **Source:** Assignment brief (`docs/Scrye - AI Research Mini Project.pdf`), Task Definition.

## Slide 4 — The metric: score against the naive baseline

- **Key message:** Score is normalized so 0 = the uniform baseline and 100 = perfect — we have to *earn* every point over "no information."
- **Content:**
  - **S = 100 · (1 − TVD(P,Q) / TVD(P,U))**, where P = truth, Q = prediction, U = uniform.
  - 0 = uniform baseline · 100 = exact match · **negative = worse than guessing uniform.**
  - TVD (total-variation distance) = ½·L1: intuitive "how much probability mass is in the wrong place."
  - Normalizer is computed on the **full split**, not the subsample — the denominator must be dataset-level (paper Eq. 2).
- **Visual:** The formula, large, with a number line: −∞ ← 0 (uniform) → 100 (perfect). Inset: two small bar charts P vs Q with the TVD area shaded.
- **Source:** `src/scrye/scoring.py` (`simbench_score`); `CLAUDE.md` → Scoring; SimBench paper Eq. 2.

## Slide 5 — Why this is genuinely hard (three traps)

- **Key message:** The intuitive moves backfire — naive persona role-play, conflating values with actions, and confusing "wrong spread" with "wrong options."
- **Content:**
  - **Trap 1 — conditioning backfires.** First-person persona *embodiment* loses to a calmer distributional framing (Monte-Carlo individuals scored −15.7).
  - **Trap 2 — value–action gap.** Stated preferences ≠ revealed behavior; behavioral datasets (gambles, moral dilemmas) resist survey-style prompting.
  - **Trap 3 — spread vs location are entangled.** You can be the right *shape* in the wrong *place*, or the right place with the wrong spread — and fixing one perturbs the other.
- **Visual:** Triptych. Each trap = a small P-vs-Q mini-chart showing the failure mode (collapsed mass / right mode wrong spread / right spread wrong mode).
- **Source:** Stage 03 (Monte-Carlo −15.7), Stage 08–09 (entanglement), Stage 13 (behavioral). `docs/experiments/`.

## Slide 6 — The data: SimBench at a glance

- **Key message:** A mix of opinion surveys and behavioral tasks, split two ways (full-population "pop" and demographic "grouped") — and the required questions are sealed into test.
- **Content:**
  - Families: opinion (OpinionQA, ESS, global barometers), knowledge, behavioral (Choices13k gambles, MoralMachine, OSPsych).
  - Two regimes: **pop** (full population) and **grouped** (conditioned on a segment).
  - Required questions **pinned to test** — never seen during development.
  - Why it matters: the behavioral + Global-South slices are where methods quietly break.
- **Visual:** A 2-column taxonomy (opinion | behavioral) with row badges for pop/grouped; a lock icon on the required-questions row.
- **Source:** `src/scrye/data.py` (`load_all`), `src/scrye/splits.py`; `CLAUDE.md` → Leakage discipline.

---

# Act 2 — A method for finding methods *(the differentiator)*

## Slide 7 — The pipeline

- **Key message:** One small, swappable spine — Record → Predictor → Calibrator → scored distribution — so every experiment is a one-line swap, not a rewrite.
- **Content:**
  - `Record` (typed survey row) → `Predictor` (the conditioning strategy) → `Calibrator` (post-hoc fix) → `evaluate` → score.
  - Predictors and calibrators are registries: a new idea = one subclass + one registry entry, instantly available to notebooks *and* the search loop.
  - Everything is content-addressed and cached, so re-running any experiment is free and deterministic.
- **Visual:** The architecture diagram (already built). Highlight the two swappable surfaces (Predictor, Calibrator) in an accent color.
- **Source:** `docs/figures/architecture-diagram.html`; modules `predict.py`, `calibrate.py`, `pipeline.py`, `experiment.py`.

## Slide 8 — The trap we built guardrails against

- **Key message:** If you try 18 ideas and keep the best one on the same data, you will fool yourself — adaptive overfitting is the real adversary here.
- **Content:**
  - Iterating against a held-out set *leaks* it: each peek spends statistical power.
  - Naive "pick the dev winner" inflates scores that won't survive a fresh test.
  - So the apparatus exists to make the search **honest by construction**, not by good intentions.
  - This is the slide that frames why everything that follows is worth the overhead.
- **Visual:** A simple "garden of forking paths" sketch — many candidate methods funneling into one held-out set, with a crack forming. Contrast with the gated funnel introduced next.
- **Source:** Motivation; formalized via `ladder.py` (Blum & Hardt 2015). `docs/superpowers/specs/2026-06-19-goal-directed-lever-search-design.md`.

## Slide 9 — Leave-family-out splits

- **Key message:** We split by *question family*, not random rows — and seal the test set — so "generalizes" means generalizes to unseen *kinds* of questions.
- **Content:**
  - Split unit = `(dataset_name, input_template)` family. Never a random row split (that leaks templates across splits).
  - **dev** = free iteration · **val** = gated selection only · **test** = sealed until the very end.
  - Required questions pinned to **test**.
  - Result: a dev win has to clear the noise floor *and* transfer to val before it's believed.
- **Visual:** Three buckets (dev / val / test) fed by family blocks, with the required-Qs block routed by a lock into test. Annotate "val touched only N times."
- **Source:** `src/scrye/splits.py`; `CLAUDE.md` → Leakage discipline.

## Slide 10 — The Ladder gate (Blum & Hardt, 2015)

- **Key message:** A val result only "counts" if it beats the current best by more than the noise floor η — this is what bounds generalization error under repeated, adaptive querying.
- **Content:**
  - η is derived from the **bootstrap** spread of the score, not hand-picked.
  - Improvements ≤ η are reported as the incumbent's value (no information spent).
  - We touched val only a handful of times across 18 stages — and each touch is logged.
  - This is the formal backbone of "we didn't overfit our own search."
- **Visual:** Step-function "ladder" chart: the reported val score only steps up when the true gain clears η; small bumps are absorbed.
- **Source:** `src/scrye/ladder.py` (`LadderGate`); Blum & Hardt 2015. Val touches logged in stages 05/12/16.

## Slide 11 — Reproducibility: preregistration → ledger → manifest

- **Key message:** Every stage is preregistered before we see data, and every run leaves a machine-readable receipt — so any number in this deck can be regenerated for free.
- **Content:**
  - **Preregister:** hypothesis + exact configs + decision rule written and approved *before* running (decision rule fixed before the data).
  - **Ledger:** a DAG of every search step (`outputs/ledger/<run>.jsonl`).
  - **Manifest:** frozen config sidecar (split seed/fractions, η, model, dataset fingerprint).
  - **Cache:** every LLM call hashed to disk → deterministic, zero-cost re-runs.
- **Visual:** The 4-step stage loop (Preregister → Run → Record → Next) as a cycle, with the three artifacts (stage .md, ledger, manifest) hanging off it.
- **Source:** `spec.py`, `ledger.py`, `manifest.py`, `llm.py`; `docs/experiments/README.md`.

---

# Act 3 — The lever story

## Slide 12 — How to read the scoreboard

- **Key message:** Each lever is a preregistered hypothesis with a fixed decision rule; a win must clear the noise floor *and* transfer to held-out val — and we kept the score honest with a directional gate.
- **Content:**
  - "Win" = clears bootstrap noise on dev → confirmed on val → only then into the system.
  - We gate on `cf_alignment ≥ baseline` so accuracy is never *bought* by getting demographic directions wrong.
  - Negatives are kept and reported — they tell us where the binding constraint is.
- **Visual:** A legend card (✅ / ⚠️ / ❌ / 🔒) and the gate condition, setting up the next slide's table.
- **Source:** Decision rules in each `docs/experiments/stage-*.md`.

## Slide 13 — The lever scoreboard *(anchor slide)*

- **Key message:** Eighteen preregistered stages; a handful of real, transferable wins — and an equal number of instructive failures.
- **Content (table):**

| Lever | Hypothesis | Verdict |
|---|---|---|
| Demographic *framing* (anti-flattening) | distributional framing > first-person persona | ✅ +6.0 val grouped |
| Model swap (2.5 → gemini-3.1-flash-lite) | a better model lifts everything | ✅ dominant, +15.9 |
| Calibrated-commitment prompt | mode-first prompt fixes *location* | ✅ confirmed val |
| Abstention (uniform fallback) | predict uniform where we reliably fail | ✅ pop +3.26 val |
| Task-kind routing | classify task → fixed intervention map | ✅ grouped +1.89 val |
| Post-hoc calibration | temp/Dirichlet fixes spread | ❌ all lose to identity |
| Reasoning-first / superforecaster | elicited reasoning helps | ❌ loses to direct ask |
| Monte-Carlo personas | simulate individuals, tally | ❌ −15.7 |
| Discrete voting | dispositional agents vote | ⚠️ Choices13k win didn't transfer |
| Grounded Nemotron electorate | census personas vote | ❌ location-limited, −11 vs cc |

- **Visual:** Render as the hero table, color-coded by verdict. Consider a small companion bar showing cumulative score gain across stages.
- **Source:** `docs/experiments/README.md` stage table (rows 01–18).

## Slide 14 — What won #1: condition the *distribution*, not a persona

- **Key message:** Asking the model to describe how a *population* would answer beats making it role-play a single demographic individual.
- **Content:**
  - `anti_flattening` strategy: keep minority mass, resist collapsing to the mode.
  - Held-out val: **53.0 vs 47.0** faithful (+6.0, clears noise, exceeds the dev gap).
  - Directionally honest too: `cf_alignment` 0.419 ≥ faithful's 0.397.
  - Mechanism: first-person embodiment over-concentrates; distributional framing preserves spread.
- **Visual:** Side-by-side P-vs-Q for one question: persona-embodiment (collapsed) vs anti-flattening (spread matches truth).
- **Source:** Stage 01 (+8 grouped dev), Stage 05 (val 53.0 vs 47.0). `docs/experiments/stage-05-val-confirmation.md`.

## Slide 15 — What won #2: a mode-first prompt fixes *location*

- **Key message:** The biggest remaining error was picking the wrong top option — a calibrated-commitment prompt improves *where* the mass sits, not how spread out it is.
- **Content:**
  - `calibrated_commitment`: commit to the most likely option first, then distribute the rest.
  - +1.8 grouped / +2.1 pop over anti-flattening on dev; confirmed on val.
  - Diagnostic insight: the oracle's headroom is **gated behind mode correctness** (+16.8 where the top option is right, −2.0 on the 39% where it's wrong).
  - One prompt beat the more complex "regime-branching" alternatives.
- **Visual:** Error-decomposition chart — TVD split into concentration (spread) vs location (options); show location shrinking after this lever.
- **Source:** Stage 09 (mode-gating), Stage 10 (cc). `docs/figures/error_decomposition_modelsweep.png`.

## Slide 16 — What won #3: abstain, and route by task kind

- **Key message:** Know when *not* to guess — predict uniform where the model reliably fails, and use an upfront classifier to send each task to the right intervention with *zero* per-dataset parameters.
- **Content:**
  - **Abstention:** a few high-entropy datasets score *below* uniform; `AbstainCalibrator` falls back to uniform there → pop +3.26 on val.
  - **Routing:** an LLM classifies the task (opinion / knowledge / risky-choice / moral / personality) → fixed kind→intervention map (`taskkind.py`).
  - Routed + abstain-floor beats base cc by **+8.75** paired on dev (95% CI [+5.94, +11.67]); val grouped **+1.89**.
  - Generalizes *by construction* — the positive routes use no per-dataset tuning.
- **Visual:** A router diagram: task → classifier → 4–5 branches (each labeled with its intervention) → score.
- **Source:** Stage 11 (abstain), Stage 15 (routing +8.75), Stage 16 (val +1.89). `src/scrye/taskkind.py`.

## Slide 17 — The honest headline: the model is the dominant lever

- **Key message:** Switching to a stronger model added **+15.9**; our entire method stack added **+5.5** on top. A strong submission says this out loud.
- **Content:**
  - faithful baseline, 2.5 → gemini-3.1-flash-lite: **+15.9 overall / +25.4 grouped**.
  - The full preregistered method stack: **+5.5** on top of that, on sealed test.
  - Implication: most of the available signal is "use a better base model" — methods earn the margin.
  - Why we lead with this: it's the finding a careful reviewer is looking to see whether you'll hide.
- **Visual:** A stacked bar: baseline → +model jump (big) → +method stack (smaller) → final. Label each segment with its delta.
- **Source:** Stage 17 final test. `docs/experiments/stage-17-final-test.md`.

## Slide 18 — What lost — and why losing is information

- **Key message:** Five well-motivated ideas failed, and each failure pointed at the same binding constraint: getting the *mode/location* right, not fixing the spread.
- **Content:**
  - ❌ **Post-hoc calibration** (temp/entropy/Dirichlet): all lose to identity — the winning prompt needs no correction.
  - ❌ **Reasoning-first / superforecaster:** elicited CoT loses to a direct distributional ask (2 independent confirmations).
  - ❌ **Entropy de-compression:** predictions are entropy-compressed (slope 0.48), but the gain isn't recoverable from the prediction's *own* entropy — needs an external contestedness signal.
  - ⚠️ **Discrete voting:** a real Choices13k win (+12.6 vs uniform) that **did not transfer** to val; catastrophic where the mode is wrong (MoralMachine −158).
  - ❌ **Grounded persona electorate:** census personas barely disagree (78% mode-agreement, no ideology axis); even after worldview enrichment it's −11 vs the simple prompt.
- **Visual:** A "graveyard" row of five cards, each with the idea, the result, and the one-line lesson. Through-line callout: *the binding constraint is location, not spread.*
- **Source:** Stages 06, 07, 08, 13, 18. `docs/experiments/`.

## Slide 19 — The diagnosis, in one chart

- **Key message:** Across every lever, spread stayed slightly too diffuse (~0.06 entropy) while the wins all came from moving mass to the right option — location is the lever that pays.
- **Content:**
  - Error decomposition: concentration error (spread) vs location error (options), each ≥ 0, summing to TVD.
  - Wins (framing, cc, routing) shrank **location**; entropy stayed ~0.06 too diffuse throughout.
  - This is why post-hoc calibration and entropy tricks failed: they target spread, which wasn't the bottleneck.
- **Visual:** Before/after stacked TVD bars (concentration vs location) across the lever progression; location bar shrinking, concentration roughly flat.
- **Source:** `src/scrye/decompose.py`; notebook 04; Stage 16 ("entropy untouched, gain is location/mode").

---

# Act 4 — Conditioning sensitivity *(assignment-required)*

## Slide 20 — Counterfactual sensitivity: does conditioning move the needle the right way?

- **Key message:** Beyond accuracy, we ask whether conditioning on a segment shifts the prediction in the *correct direction* — and we use that as a guardrail, not just a number.
- **Content:**
  - Metric `cf_alignment`: alignment between the **predicted** shift (segment − population) and the **ground-truth** shift. +1 = right direction · −1 = wrong · 0 = orthogonal · NaN when no real shift.
  - Used as a **non-regression gate**: a new system must keep `cf_alignment ≥ baseline` — accuracy can't be bought by getting directions wrong.
  - Honest finding: directional sensitivity is **modest and weakly differentiated** at this scale (~0.40–0.45 for the better model); distributional framings edge out first-person, but margins are small.
  - It rises with model quality, suggesting the ceiling here is also model-bound.
- **Visual:** A 2D shift-vector picture: ground-truth shift arrow vs predicted shift arrow, angle = alignment. Side panel: cf_alignment by system (small bars, overlapping CIs).
- **Source:** `src/scrye/scoring.py` (shift-alignment helper, line ~130); Stages 01/04/05/07 cf_alignment values.

---

# Act 5 — The result

## Slide 21 — Sealed-test headline

- **Key message:** On a test set untouched until the final run, the full lineage lifts the faithful baseline by ~+5.5, with every confidence interval excluding zero.
- **Content:**
  - 🔒 faithful **35.21 → ~40.9** (final system) on sealed test; all CIs exclude 0.
  - The router's +1.1 val edge **did not transfer** → we report the simpler **cc + abstain** system.
  - Honest decomposition restated: model +15.9, method stack +5.5.
- **Visual:** Score progression bar with CIs: baseline → +method stack → final, each with error bars. Flag the router/cc tie.
- **Source:** Stage 17. `docs/experiments/stage-17-final-test.md`.
- **VERIFY:** Reconcile 40.73 (router full-lineage) vs 40.93 (cc+abstain) and state which is the reported headline.

## Slide 22 — Validity check: we reproduce the paper

- **Key message:** Before trusting any of our own numbers, we re-derived the paper's published baseline — and our harness brackets it.
- **Content:**
  - faithful @ Qwen2.5-72B: our **S = 26.83 [24.4, 29.4]** brackets the paper's reported **27.61**. ✓
  - Same prompt, same model, same scoring → matches → the harness is faithful.
  - This is the credibility anchor for every other number in the deck.
- **Visual:** A single number line: our CI [24.4, 29.4] as a band with 26.83 marked, and the paper's 27.61 falling inside it.
- **Source:** Stage 17 paper-reproduction section.

## Slide 23 — The required-question predictions

- **Key message:** For the three mandated questions, here are our predicted distributions against the human ground truth.
- **Content:**
  - Trust in president · Gay rights · Internet-use frequency.
  - Show predicted vs actual bars per question + the per-question score.
  - Note these were **pinned to test** — first contact at final evaluation.
- **Visual:** Three small predicted-vs-actual bar-pairs (one per required question), each with its S score badge.
- **Source:** Notebook 03 / `best_analysis_run`; Stage 17. **VERIFY:** pull the three predicted distributions + scores from the final-test run before rendering.

---

# Act 6 — Reflection

## Slide 24 — Limitations & risks

- **Key message:** The system is honest about where it's still weak: spread is slightly too diffuse, behavioral and Global-South slices remain hard, and we lean on a single base model.
- **Content:**
  - Entropy still ~0.06 too diffuse — predictions are marginally over-confident in spread.
  - Behavioral tasks (gambles, moral dilemmas) and Global-South barometers resist survey-style prompting.
  - Bottom-up simulation underperforms a direct distributional ask (4 independent confirmations).
  - Single-model dependence: most of the gain is model-bound, so the result moves with the frontier.
- **Visual:** A short risk register table (risk · evidence · mitigation/next).
- **Source:** Stages 13, 16, 18; `docs/part-ii-feedback-architecture.md` → "Honest Scope".

## Slide 25 — What's next: agentic web-evidence

- **Key message:** The live frontier is giving the predictor *tools* — guarded web search/fetch to gather evidence before answering — with leakage controls built in.
- **Content:**
  - `WebAgentPredictor`: a bounded tool-calling agent over disk-cached, guarded web_search/web_fetch (Predictor ABC unchanged).
  - Leakage guards: identity stripping + domain blocklist + query guard + audit trail.
  - Decision rule preregistered: must beat a closed-book twin *and* the champion *and* pass a leakage audit.
  - We also report a closed line: the persona/electorate direction is location-limited and was retired.
- **Visual:** Agent loop: predictor ↔ guarded tools (search/fetch) ↔ cache, with a "leakage audit" gate on the output.
- **Source:** `Scrye_Project-60fk` bean; `docs/experiments/stage-18-agentic-web-evidence.md` (PLANNED).

## Slide 26 — Closing thesis

- **Key message:** We beat the baseline on sealed test, validated against the literature, and — most importantly — built a search that knows which of its own wins are real. That discipline is the transferable asset.
- **Content:**
  - Result: ~+5.5 method gain on sealed test, all CIs excluding 0; paper reproduced.
  - Honesty: model dominance stated; non-transferring wins demoted; negatives kept.
  - The apparatus (preregistration · leave-family-out · Ladder gate · ledger) generalizes to any LLM-evaluation problem.
  - Bridge: *Part II takes this from research benchmark to a commercial feedback loop.*
- **Visual:** Callback to the Slide 1 distribution motif, now resolved (prediction overlaid on truth). One-line thesis across the bottom.
- **Source:** Synthesis.

---

# Appendix *(pull-on-demand for the discussion)*

- **A1 — Full stage table (01–18):** every stage, hypothesis, and headline result. *Source: `docs/experiments/README.md`.*
- **A2 — Split mechanics:** family definition, fractions/seed, required-Q pinning, val-touch log. *Source: `src/scrye/splits.py`, manifests.*
- **A3 — Ladder gate / η derivation:** bootstrap → η → accept rule; Blum & Hardt bound. *Source: `src/scrye/ladder.py`.*
- **A4 — Per-dataset score breakdown:** pop vs grouped by dataset; where we beat / lose to uniform. *Source: notebook 03, `docs/figures/score_by_dataset__*.png`.*
- **A5 — Prompt-strategy gallery:** the five `PromptStrategy` styles, verbatim. *Source: `src/scrye/persona.py`.*
- **A6 — Cost & cache:** tokens, $ per stage, cache hit-rate (re-runs are free). *Source: `llm.py` `Usage`; `data/cache/`.*

---

## Design direction for Claude Design *(optional, delete if you have your own)*

- **Tone:** research-credible, not salesy. Lots of whitespace; one idea per slide.
- **Motif:** the uniform→peaked distribution morph as a recurring visual language (flat = baseline, peaked = signal).
- **Color coding:** reserve one accent for ✅ transferable wins, a muted/grey for ❌ negatives, an amber for ⚠️ didn't-transfer. Keep it consistent across the scoreboard and the graveyard slide.
- **Numbers:** every stat slide should be able to show its **Source** as a small footnote — reviewers will ask "where's that from," and the answer is on the slide.
- **Charts over bullets:** Slides 5, 15, 17, 19, 21, 22 are inherently visual — lead with the chart, demote the text.
