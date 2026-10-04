# Stage 14 — Task-context prompting

**Status:** preregistered (dev-only) · **Date:** 2026-06-21 · **Model:** `gemini-3.1-flash-lite`

## Motivation

SimBench presents every item **atomized**: one question + a one-line group
prompt (paper §2.2, p23 — system: *"You are a group of individuals with these
shared characteristics…"*, user: *"{question}. Estimate what percentage of your
group would choose each option."*). But the original respondents answered inside
a **task context** the benchmark strips:

- **Psychometric batteries** — OSPsychMACH is the 20-item MACH-IV; Big5, RWAS
  likewise. The latent construct that shapes the population distribution is
  defined *across* the items; a single item out of context is ambiguous.
- **Repeated-trial behavior** — Choices13k (many gambles per session),
  MoralMachine (many dilemmas).
- **Multi-topic surveys** — OpinionQA, ESS, ISSP.

This information was available to the respondents, so restoring it is faithful,
not a benchmark trick. OSPsychMACH / Choices13k / MoralMachine are exactly our
three abstain (below-uniform) datasets, so this is a candidate fix for tasks the
model currently fails on.

## Hypothesis

Giving the model the task context the respondents had improves the predicted
distribution on context-dependent tasks (psychometric batteries especially),
without hurting the already-strong opinion surveys.

Three variants, all on top of the `calibrated_commitment` system prompt and
distributional ask (only the *context* varies):

- **V1 `task_context_brief`** — prepend a faithful one-line description of the
  instrument (`persona.TASK_BRIEFS`, drawn from the paper appendix).
- **V2 `task_context_items`** — show K=6 sibling items from the same instrument
  (the "include other items" idea). Fully general; questions only, no answers.
- **V3 `task_context_both`** — brief + siblings.

Control: `calibrated_commitment` (current system prompt, atomized).

## Configs

- **Predictor:** `ZeroShotPredictor` with `TaskContextStrategy` (new), K=6 siblings.
- **Data:** dev only, `make_split(full, seed=0, unit="question")`. Per-dataset
  subsample (≤40), deterministic. Datasets: weak/psychometric — OSPsychMACH,
  Choices13k, MoralMachine, OSPsychBig5, OSPsychRWAS; strong controls — OpinionQA
  (pop), ESS (grouped). Normalizers on the full split (Eq. 2 Z).
- **Budget:** ≤ $3 (≈ 4 systems × ~250 items, cached on re-run).

## Decision rule (fixed before results)

- **WIN for a variant** iff, on the **weak/psychometric** datasets pooled, it
  beats `calibrated_commitment` by more than the bootstrap noise floor, AND it
  does not lose on the strong controls by more than noise. The best winning
  variant is carried forward (and, if it also helps a current abstain dataset
  enough to clear uniform, that dataset stops abstaining).
- **NO WIN:** atomized `calibrated_commitment` stands; record as negative.
- Dev only; no val/test contact. Required questions (test-pinned, in
  ESS/LatinoBarometro/ISSP) are untouched.

---

## Results

### Round 1 — variants on weak datasets + strong controls (dev)

Per-dataset subsample (≤40), `gemini-3.1-flash-lite`. Run file:
`outputs/runs/2026-06-21-task-context.results.json`.

| dataset | n | cc (control) | brief | **items** | both |
|---|---|---|---|---|---|
| OSPsychMACH | 40 | −54.99 | −51.04 | −50.04 | −53.28 |
| Choices13k | 40 | **22.76** | 4.24 | 2.09 | 1.74 |
| MoralMachine | 40 | −17.62 | −17.71 | −12.53 | −19.47 |
| OSPsychBig5 | 40 | 24.21 | 28.77 | 18.20 | 25.07 |
| OSPsychRWAS | 10 | 45.94 | 40.72 | 43.39 | 26.88 |
| **POOLED weak** | | **−3.33** | −6.01 | −7.40 | −9.23 |
| OpinionQA *(control)* | 40 | 59.13 | 59.13 | **72.86** | 72.86 |
| ESS *(control)* | 40 | 22.25 | 22.25 | **36.13** | 36.13 |
| **POOLED control** | | **40.69** | 40.69 | **54.49** | 54.49 |

**Preregistered verdict (weak datasets): REFUTED.** Every context variant loses
to atomized `calibrated_commitment` on the weak pool (Δ −2.7 to −5.9, within
noise but trending negative). Task context does **not** rescue the
behavioral/psychometric tasks, and it badly hurts Choices13k (22.8 → 2.1).
(`brief` == control on ESS/OpinionQA because no brief is authored for them, so
`both` == `items` there.)

**Unexpected, much larger signal on the strong controls — `items` context lifts
the grouped opinion surveys +13.8** (OpinionQA +13.7, ESS +13.9, consistent).
Showing sibling survey questions as context evidently disambiguates the survey
and re-calibrates the spread. These are SimBenchGrouped datasets — the topline
driver and home of the graded test questions.

**Discipline note:** this gain is on the *control* group, **not** the
preregistered hypothesis — hypothesis-generating, not a confirmed win (recording
it as a win would be HARKing). It motivates a dedicated, preregistered grouped
confirmation (Round 2) on all five grouped datasets before any carry-forward.
The weak-dataset abstain set (Stage 11/12) is unchanged.

### Round 2 — preregistration: grouped-survey confirmation (dev)

**Hypothesis (new, from Round 1):** `task_context_items` (K=6 sibling items)
beats `calibrated_commitment` on the **grouped** split, where context
disambiguates multi-topic surveys.

**Config:** dev grouped only, all five datasets (ESS, OpinionQA, Afrobarometer,
ISSP, LatinoBarometro), ≤120 deterministic items each. Two systems: cc (control)
vs items. `gemini-3.1-flash-lite`, normalizers on full split.

**Decision rule (fixed):** WIN iff `items` beats `cc` on grouped-pooled dev by
more than the bootstrap noise floor (and is positive on a majority of the five
datasets). A win → `task_context_items` becomes the grouped-system
val-confirmation candidate (a much larger lever than Stages 10–12). No val/test
contact; grouped dev has no required questions (test-pinned).

### Round 2 — results

Dev grouped, 120/dataset (600 recs). Run:
`outputs/runs/2026-06-21-task-context-grouped.results.json`.

| dataset | n | cc | task_context_items | Δ |
|---|---|---|---|---|
| **ESS** | 120 | 33.64 | 46.52 | **+12.88** |
| **OpinionQA** | 120 | 64.39 | 70.40 | **+6.01** |
| ISSP | 120 | 43.32 | 43.55 | +0.23 |
| Afrobarometer | 120 | 54.13 | 51.33 | −2.81 |
| LatinoBarometro | 120 | 4.38 | −0.06 | −4.44 |
| **POOLED** | 600 | 39.97 | **42.35** | **+2.37** |

**Verdict: WIN (rule met).** Paired Δ = +2.37, 95% CI **[+0.77, +3.96]** (excludes
0), wins 3/5. Smaller than Round 1's +13.8 (small-control-sample regression), but
statistically clean — a stronger grouped result than `calibrated_commitment`
itself (Stage 10's +1.8 was within noise; this is not). **Heterogeneous:** big
help on the US/European surveys (ESS, OpinionQA), a drag on the Global-South
barometers (Afrobarometer, LatinoBarometro) — plausibly the model has weaker
grounding there, so sibling items add noise.

### Round 3 — leakage-safe per-dataset routing (free, cached)

The heterogeneity invites the Stage-11 routing trick: learn per dataset on a
dev-FIT half whether `items` beats `cc`, apply to the dev-EVAL half (2-fold).
Run: `outputs/runs/2026-06-21-task-context-select.results.json`.

- Selector reliably routes **ESS, OpinionQA → items**; **Afrobarometer,
  LatinoBarometro → cc** (ISSP a near-tie).
- Eval-half pooled: cc 39.97 → items 42.35 → **selective 43.68**.
- **Selective Δ vs cc = +3.71, 95% CI [+2.71, +4.75]** — excludes 0, larger and
  tighter than pure-items.

### Stage verdict & carry-forward

- **Preregistered weak-dataset hypothesis: refuted** (context doesn't rescue the
  behavioral/psychometric tasks; hurts Choices13k).
- **Confirmed dev win on grouped:** `task_context_items` (+2.37 pooled), and
  **selective task-context** (+3.71, leakage-safe) — the largest, cleanest
  grouped lever found so far. Concentrated in ESS (+12.9) and OpinionQA (+6.0).
- **Mechanism:** restoring the survey context the respondents actually had
  (sibling questions) re-calibrates the predicted spread on multi-topic opinion
  surveys. Faithful, not a benchmark artifact (questions only, no answers).
- **`TaskContextStrategy` + `build_item_corpus` ship as reusable primitives.**

**Decision for the human (gated):** selective task-context is a strong
val-confirmation candidate for the *grouped* side — orthogonal to the pop-side
system (cc + abstain). Routing (items on ESS/OpinionQA, cc elsewhere) is learned
on dev by dataset identity (leakage-safe, datasets reused in val). Recommend one
val-gated confirmation. No val/test contact yet; required questions untouched.

### Follow-up — does task-context help the POP split? (dev)

`task_context_items` vs `cc` across all 20 pop datasets (≤40/dataset).
Log: `tmp/ctx_pop.log`.

- **Pooled pop: cc 31.49 → items 31.83, Δ +0.34, 95% CI [−2.00, +2.65]** — a
  wash. Leakage-safe selective routing: **+0.76, CI [−0.88, +2.43]** — still
  within noise.
- **Same survey/knowledge mechanism, same sign as grouped:** helps OSPsychMGKT
  +8.2, NumberGame +7.3, ESS +6.2, GlobalOpinionQA +6.0, MoralMachine +5.1,
  TISP +3.6 (12/20 datasets helped >+1) — but cancelled by sharp losses on
  self-contained behavioral choices: **Choices13k −20.7**, OSPsychBig5 −6.0,
  Jester −4.9, DICES −3.2. Context helps thematic surveys/batteries, hurts
  one-shot choice tasks (other gambles as context just add noise).
- **Cross-split flip:** LatinoBarometro/Afrobarometer *help* on pop (+2.8, +2.2)
  but *hurt* on grouped (−4.4, −2.8) — same datasets, opposite sign. The only
  difference is grouped conditions on a demographic segment, so sibling-item
  context appears to **interfere with demographic conditioning**.

**Net:** task-context is a **grouped-side win, pop-side wash.** It is a
"survey-context" method — promote on grouped (and optionally route onto specific
pop survey datasets for a small extra), keep it away from behavioral-choice
tasks. The pop-side system (cc + abstain, + optional Choices13k→voting) is
unaffected.
