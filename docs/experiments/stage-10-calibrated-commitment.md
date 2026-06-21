# Stage 10 — Calibrated-commitment prompt iteration (dev)

- **Status:** **RUNNING** — preregistered; up to 4 dev prompt rounds.
- **Run name:** `2026-06-21-calibrated-commitment`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> `dev` only. Iterative prompt engineering: the *decision rule* is fixed before
> looking; the *wording* of each round is informed by the previous round's
> diagnostics (legitimate dev iteration). No `val`/`test` contact.

## Context

The error decomposition + entropy analysis (notebook 04, Stages 06–09) located the
problem precisely:

- The winner `anti_flattening` wins on **location** (covers the right options) but
  is the **most over-diffuse** strategy — it refuses to commit. On consensus
  questions (truth entropy < 0.4) it goes sharp only **17%** of the time
  (vs `contextualized` 28%, `faithful` 21%), with mean pred entropy 0.59 vs a truth
  of 0.25. Its "never flatten / keep minority mass" instruction is **one-directional**.
- Post-hoc calibration can't fix this (the spread fix is gated behind mode
  accuracy). But **the prompt can** — and consensus questions are the sweet spot:
  they are both where over-spreading is worst *and* where the mode is most likely
  right, so a sharper distribution there is not gated away.

## Hypothesis

A prompt that **combines** the strengths of `contextualized` (socio-cultural
context) and `anti_flattening` (diversity / anti-stereotype) and adds the two
missing ingredients — **consensus/entropy awareness** and **licensed commitment**
(it is correct to be sharp when a group genuinely agrees) — in **direct-answer
format** (no reasoning-first; CoT lost in Stage 07) beats `anti_flattening` on dev
grouped, by lowering the entropy floor on consensus questions without sacrificing
the location/diversity win.

Four design elements every variant must carry (per the brief):
1. diversity of perspectives, 2. don't stereotype, 3. consensus/entropy awareness,
4. commitment to a specific choice is sometimes the best answer.

## Method — up to 4 rounds

Each round adds 1–2 candidate prompt strategies (defined in the run script,
direct-answer format), scored on the **fixed dev-eval sample** against the
incumbents. After each round, read the diagnostics and refine the wording for the
next round. Stop early if a round clearly wins or if two consecutive rounds stall.

**Fixed eval set:** `stratified_sample(grouped, 700) + stratified_sample(pop, 300)`,
`seed=42` — the standard dev-eval used by Stages 07–09. Incumbents:
`anti_flattening` (grouped ≈ 45.0 on this set) and `contextualized`.

**Per-candidate diagnostics (mechanism check):** grouped mean + 95% CI, pop,
mean pred entropy, entropy floor (p10), slope of pred_H~truth_H (contestedness
tracking), **consensus-commit rate** (frac of truth_H<0.4 items with pred_H<0.45),
and mode (top-option) accuracy.

## Decision rule (preregistered — fixed before looking)

- **Primary:** dev-eval **grouped** mean SimBench, 95% bootstrap CI (n=700).
  Noise floor = half-width of the larger compared CI (~3).
- A candidate is a **dev win** iff its grouped beats `anti_flattening` beyond the
  noise floor. A candidate is **promising** (carry to next round's refinement) iff
  it beats incumbent OR moves the mechanism in the right direction (consensus-commit
  rate ↑, floor ↓, slope ↑) at equal score.
- **Mechanism guard:** report whether any score gain comes with the predicted
  diagnostic shift (committing more on consensus) and *not* from losing the
  location/diversity win (watch mode accuracy and pop).
- **Adaptive-search caveat:** iterating wording on one dev set across ≤4 rounds
  risks dev-overfitting. A dev winner is therefore a **val-confirmation candidate
  only** — promoted to "the system" (and the notebook default) **only** after a
  separate one-query val confirmation. `test` stays locked.
- **Negative:** if 4 rounds don't beat `anti_flattening` beyond noise, the prompt
  framing lever is exhausted; the remaining headroom needs the mode/location axis
  (predictor/model), not wording.

---

## Results

Fixed dev-eval n=990 (grouped 700 / pop 300, seed 42). `gemini-3.1-flash-lite`,
direct-answer format. Incumbent `anti_flattening` grouped = 45.01.

### Round 1 — `commit_v1` (balanced) vs `commit_v2` (aggressive)

| system | grouped | 95% CI | pop | floor p10 | slope | commit@cons | mode acc |
|---|---|---|---|---|---|---|---|
| `anti_flattening` (inc.) | 45.01 | [42.0, 47.9] | 31.6 | 0.61 | 0.53 | 0.15 | 0.61 |
| `contextualized` (inc.) | 44.39 | [40.9, 47.5] | 30.6 | 0.53 | 0.59 | 0.23 | 0.62 |
| **`commit_v1`** | **45.80** | [42.4, 49.0] | 25.0 | 0.53 | 0.60 | 0.28 | **0.64** |
| `commit_v2` (aggressive) | 44.60 | [41.0, 47.9] | 22.4 | 0.52 | 0.64 | 0.45 | 0.61 |

**Mechanism confirmed:** both candidates lower the entropy floor, raise the slope,
and commit much more on consensus questions. **`commit_v1`** is best grouped (+0.79,
within noise) *and* lifts mode accuracy to 0.64. **`commit_v2`** over-commits
(commit@cons 0.45) and grouped *drops* — re-confirming the entanglement (sharpening
when mode is 61% right backfires). **Both tank pop** (−7 to −9): the commitment
framing over-sharpens the genuinely-diffuse unconditioned population.
→ Round 2: refine `commit_v1`; recover pop by distinguishing a broad population
(usually divided) from a specific subgroup (more often unified).

### Pop vs grouped are different *task mixes* (diagnostic)

| split | n | truth_H mean / std | % consensus (<0.4) | datasets |
|---|---|---|---|---|
| `grouped` | 2566 | 0.70 / 0.21 | 9% | 5 demographic opinion surveys |
| `pop` | 3585 | 0.70 / 0.26 | 15% | **20** datasets incl. non-survey: NumberGame, MoralMachine, ChaosNLI, Choices13k, WisdomOfCrowds, Jester, OSPsych… |

Same mean entropy, but `pop` is a heterogeneous mix of **task types**, many not
opinion surveys at all. The demographic/consensus/"consider this group's social
circumstances" framing is right for surveys but nonsensical for a number game or
joke ratings — *that* is why commitment prompts tank pop. Implication: one
regime-aware prompt that applies the demographic framing **only when a segment is
present**.

### Round 2 — `commit_v3` (broad-vs-subgroup) and `commit_v4` (mode-targeting)

| system | grouped | 95% CI | pop | floor | slope | commit@cons | mode acc |
|---|---|---|---|---|---|---|---|
| `anti_flattening` (inc.) | 45.01 | [42.0, 47.9] | 31.6 | 0.61 | 0.53 | 0.15 | 0.61 |
| `commit_v1` | 45.80 | [42.4, 49.0] | 25.0 | 0.53 | 0.60 | 0.28 | 0.64 |
| `commit_v3` | 46.64 | [43.5, 49.7] | 28.2 | 0.53 | 0.59 | 0.21 | 0.63 |
| **`commit_v4`** | **47.00** | [43.9, 49.9] | **34.7** | 0.57 | 0.55 | 0.08 | **0.64** |

**`commit_v4` wins on both splits** (grouped +1.99, pop *above* incumbent) at the
best mode accuracy — and it does so through **location, not commitment**
(commit@cons just 0.08). The mode-first framing ("get the leading answer right; a
clear plurality when one option leads") is the lever *and* is task-agnostic enough
to keep pop healthy. Gains still within the n=700 noise floor.
→ Round 3: **regime-aware** — v4's mode-first base for all records, plus the
demographic/diversity/consensus framing only when `record.segment` is non-empty.

### Round 3 — regime-aware branching (does NOT help)

| system | grouped | 95% CI | pop | mode acc |
|---|---|---|---|---|
| `commit_v4` (no branching) | **47.00** | [43.9, 49.9] | **34.7** | 0.64 |
| `regime_v5` (branch, mode-first cond.) | 46.52 | [43.4, 49.6] | 27.4 | 0.63 |
| `regime_v6` (branch + commitment cond.) | 45.58 | [42.4, 48.6] | 28.0 | 0.62 |

Branching **underperformed the single v4 prompt** on both splits — notably it
*hurt* pop (27 vs 34.7). v4's "survey-methodologist + representative-sample +
get-the-leading-answer-right" framing already generalizes to the heterogeneous pop
tasks better than a hand-written "neutral broad-population" branch. Conclusion: the
regime distinction is real but **one mode-first prompt already absorbs it** — no
explicit branch needed.
→ Round 4: push `commit_v4` itself (add a light socio-cultural context anchor), then
confirm the winner on a larger dev sample to tighten the CI.

### Round 4 — context anchor + larger-sample confirmation (grouped 1200 / pop 400)

| system | grouped | 95% CI | pop | mode acc | commit@cons |
|---|---|---|---|---|---|
| `anti_flattening` (inc.) | 45.53 | [43.3, 47.7] | 30.9 | 0.60 | 0.10 |
| `contextualized` (inc.) | 44.81 | [42.3, 47.3] | 29.8 | 0.61 | 0.23 |
| **`commit_v4`** | **47.36** | [45.2, 49.6] | **33.0** | **0.62** | 0.05 |
| `commit_v7` (v4 + context) | 47.15 | [44.9, 49.4] | 30.1 | 0.61 | 0.09 |

`commit_v7`'s context anchor does **not** help. **`commit_v4` is confirmed best** on
the larger sample — grouped +1.83, pop +2.1, mode acc +0.02 — replicating round 2.

## Conclusion

**Best prompt: `commit_v4`** — a single, regime-agnostic, *mode-first* prompt that
carries all four requested elements (diversity, anti-stereotype, consensus/entropy
awareness, licensed commitment) and improves **both** grouped (+1.8) and pop (+2.1)
over `anti_flattening`, replicated across two dev samples, with the best mode
accuracy.

Key learnings from the four rounds:
1. **The lever is location, not commitment.** v4 wins via "get the leading answer
   right / give a clear plurality when one option leads" (commit@cons just 0.05),
   exactly the mode unlock identified in Stages 08–09. Pure entropy-commitment
   (v1/v2) over-sharpened and hurt pop.
2. **One prompt beats regime-branching.** v4's framing already generalizes across
   the heterogeneous pop task mix; an explicit `segment`-based branch (v5/v6) was
   worse, especially on pop.
3. **Context/aggressive-commitment add-ons don't help** (v2, v6, v7 all ≤ v4).

**Status / carry-forward:** the +1.8 grouped gain is **within the dev noise floor**
— a consistent, mechanism-validated improvement but **not** a clean beyond-noise dev
win. So `commit_v4` is promoted to a registered strategy **`calibrated_commitment`**
for reuse and is a **val-confirmation candidate**, but the locked system
(`anti_flattening`) and the notebook default are **unchanged**, and **no val/test
query is spent**. A future stage can run the one-query val confirmation if desired.
