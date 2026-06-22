# Stage 13 — Discrete-vote simulation for the weak pop tasks

**Status:** preregistered (dev-only) · **Date:** 2026-06-21 · **Model:** `gemini-3.1-flash-lite`

## Motivation

Stage 11 found three pop datasets where the model scores *below* uniform, so the
confirmed system abstains (predicts uniform) on them: **Choices13k**,
**MoralMachine**, **OSPsychMACH**. A base-rate prior would beat uniform on these,
but the gain on the binary ones (MoralMachine, Choices13k) is a **questionnaire
artifact** — it exploits that SimBench does not randomize A/B option order, not
any understanding of the task. We reject that on principle: we want methods that
are *reasonable as simulation*, not benchmark-order tricks.

The standing diagnosis (Stages 06–10): the model, asked to *introspect a
distribution*, regresses to medium entropy and is confidently wrong on the mode
for non-survey behavioral tasks. Stage 03 already refuted the obvious simulation
fix — `MonteCarloPredictor`, which averages each individual's self-reported
*distribution* (pop 5.0, worst of all systems): uniform-weighted ideology
archetypes over-disperse, and averaging hedged individuals stays hedged.

## Hypothesis

A **discrete-vote** ensemble fixes both of Stage 03's failure modes and is
simulation-faithful: draw K synthetic individuals varied along **task-agnostic
dispositions** (risk attitude, moral lean, temperament, values — not political
archetypes), have each commit to **one option**, and let the group distribution
be the **vote tally**. Spread is then endogenous — sharp where sampled people
agree, diffuse where they disagree — instead of an averaging artifact.

- **H1 (primary):** on the three abstain datasets, `voting_ensemble` beats
  uniform (the current fallback, ≈0) on dev.

## Configs

- **Predictor:** `VotingEnsemblePredictor` (new, `src/scrye/predict.py`),
  `n_individuals=16`, `alpha=0.5` Laplace smoothing, `temperature=0`, dispositions
  from `persona.sample_disposition` / `persona.voter_messages` (new).
- **Comparators:** `uniform` (the bar) and `calibrated_commitment` single-call
  (what abstain replaced — known < uniform here).
- **Data:** dev only, `make_split(full, seed=0, unit="question")`. Subsample per
  dataset: Choices13k 60, MoralMachine 60, OSPsychMACH all (~45). Deterministic
  (seed 0). Normalizers built on the **full** split (dataset-level Eq. 2 Z).
- **Budget:** ≤ $3 (16 votes × ~165 items ≈ 2.6k calls; cached on re-run).

## Decision rule (fixed before results)

- **WIN:** `voting_ensemble` mean score > uniform by more than the per-dataset
  bootstrap noise floor on **≥ 2 of 3** datasets, AND its pooled (3-dataset) mean
  is positive. A win replaces the uniform fallback with `voting_ensemble` on
  those datasets and proceeds to Round 2 (does it hurt the strong grouped tasks?
  + tune K). 
- **NO WIN:** abstention-to-uniform stands; record as a negative result.
- No val/test contact this stage. Required questions are pinned to test and are
  not in any of these datasets, so they are untouched regardless.

---

## Results

### Round 1 — discrete-vote vs uniform on the three abstain datasets (dev)

Subsample: Choices13k 60, MoralMachine 60, OSPsychMACH 40 (160 recs · 2,560 votes ·
0 parse failures). `voting_ensemble` N=16, α=0.5. Run file:
`outputs/runs/2026-06-21-voting-sim.results.json`.

| dataset | n | uniform (bar) | calibrated_commitment | **voting_ensemble** [95% CI] |
|---|---|---|---|---|
| Choices13k | 60 | −7.14 | 1.68 | **+15.61** [−4.1, 33.8] |
| MoralMachine | 60 | +3.19 | −31.16 | **−163.98** [−188.6, −137.3] |
| OSPsychMACH | 40 | −7.20 | −54.99 | **−26.13** [−41.8, −10.2] |
| **POOLED** | 160 | −3.28 | −24.80 | **−62.17** |

**H1 verdict: REFUTED.** Voting beats uniform on only **1 of 3** datasets (needed
≥2) and the pooled mean is strongly negative — fails the decision rule. The
uniform-abstain fallback stands as the default.

**But the failure is structured and informative — discrete voting is mode-gated:**

- **Choices13k (+15.6 vs uniform −7): a genuine, artifact-free win.** On gambles
  the model reasons sensibly per simulated agent (expected value × risk
  attitude), so a heterogeneous risk panel produces a realistic split matching
  human choice proportions. CI lower bound (−4.1) means it is not yet
  significant at n=60, but the point estimate is well above both uniform and the
  single-call. This is the principled, non-artifact gain we were after.
- **MoralMachine (−164): catastrophic.** The model is confidently *wrong-mode* —
  its agents vote "stay the course" while humans overwhelmingly swerve. Discrete
  commitment turns a wrong belief into near-certainty against the truth; far
  worse than either the hedged single-call (−31) or uniform (+3). Varied
  dispositions do **not** break the consensus, because the error is in the
  model's knowledge, not its spread.
- **OSPsychMACH (−26 vs −7): loses**, same wrong-mode mechanism on a 5-point
  Likert trait scale.

**Lesson:** confident simulation helps only where the model's mode is right and
is dangerous where it is wrong — the same mode-dependence found in Stages 10–11.
A single global "vote everywhere" is therefore wrong; the principled system must
be **selective**, routing to voting only where a leakage-safe dev signal says it
beats uniform. Round 2 tests exactly that on Choices13k.

### Round 2 — is the Choices13k win real out-of-sample? (full dev)

Full dev for all three datasets (Choices13k 237, MoralMachine 246, OSPsychMACH 40
· 523 recs · 8,368 votes · 0 parse failures). Run file:
`outputs/runs/2026-06-21-voting-r2.results.json`.

| dataset | n | uniform | **voting** [95% CI] |
|---|---|---|---|
| **Choices13k** | 237 | −0.77 | **+12.64 [+3.2, +21.4]** |
| MoralMachine | 246 | −2.73 | −157.86 [−172.8, −143.4] |
| OSPsychMACH | 40 | −7.20 | −26.13 [−41.8, −10.2] |

- **Choices13k is now significant:** at n=237 the voting CI lower bound (+3.2)
  excludes 0 and clears uniform (−0.77). The +13 gain is real and out-of-sample
  robust — a genuine, artifact-free simulation win on the gamble task.
- MoralMachine / OSPsychMACH confirm the Round-1 catastrophe (wrong-mode
  knowledge gap; not a spread problem, not prompt-fixable).

**Leakage-safe per-dataset selector (2-fold):** learn per dataset on a dev-FIT
half whether voting beats uniform, apply to the disjoint dev-EVAL half:

| | eval-half mean (pooled over 3 datasets) |
|---|---|
| pure uniform-abstain (current system) | −2.19 |
| selective (vote where fit-validated, else uniform) | **+2.25**  (Δ **+4.44**) |

The selector reliably routes Choices13k→vote and MoralMachine→uniform; it
misrouted OSPsychMACH→vote in one fold purely from its tiny n=40 (a small-sample
noise issue — a noise guard or a minimum-n threshold fixes it). The net gain is
carried almost entirely by Choices13k.

### Conclusion & decision

- **Preregistered H1 (global voting > uniform): REFUTED** (1/3 datasets, pooled
  strongly negative).
- **A real, principled, artifact-free win exists but is narrow:** heterogeneous-
  agent discrete voting beats uniform on **Choices13k** by ~13 (significant).
  Estimated effect at the topline: **≈ +0.84 dev pop** (Choices13k = 6.6% of dev
  pop), ≈ +0.3 pooled — modest, at 16× inference cost on that dataset.
- **The binding constraint is unchanged:** where the model's mode/knowledge is
  wrong (MoralMachine, OSPsychMACH), no spread mechanism helps; abstention to
  uniform remains the honest fallback there.

**Carry-forward / decision for the human:** the confirmed Stage 12 system is
unchanged by default. Optionally promote a **selective** improvement — route only
Choices13k from uniform-abstain to `voting_ensemble` (add a min-n / noise guard
to the selector) — for a small but principled pop gain. Promotion would need the
usual single val-gated confirmation. Given the ≈+0.3 pooled effect, this is a
judgment call, not an automatic carry-forward. The `VotingEnsemblePredictor`,
`sample_disposition`, and `voter_messages` ship as a validated, reusable
simulation primitive regardless.
