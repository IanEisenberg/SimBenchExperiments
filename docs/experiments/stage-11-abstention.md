# Stage 11 — Abstention / uniform-fallback on hard tasks (dev)

- **Status:** **DONE** (ran 2026-06-21) — **win:** per-dataset abstention lifts dev
  pop (+3.1 on `calibrated_commitment` → 39.1, the best combo); the mode-first
  strategy is confidently *wrong* on unknowable tasks, so abstention and commitment
  are complementary. Per-item ceiling (oracle +15) needs a confidence signal.
- **Run name:** `2026-06-21-abstention`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> `dev` only. The abstention rule is **fit on a disjoint dev-fit half** and applied
> to the dev-eval half. No `val`/`test` contact.

## Context — the per-task breakdown

`anti_flattening`'s pop average (27.2) hides huge per-task variance. A handful of
**high-entropy, intrinsically-hard** tasks score **below uniform** (negative):
OSPsychMACH (−13.5, mode acc 0.18), Choices13k (−6.5), MoralMachine (−5.5),
Jester (−4.0), ConspiracyCorr (−2.2) — plus `LatinoBarometro` **grouped** (−9.4).
Drop the 5 worst pop tasks and pop rises 27.2 → 33.9, ≈ grouped (34.1). So pop is
not structurally worse; a few pathological tasks drag it down. On a task where the
model cannot beat uniform, **predicting uniform scores ~0 instead of negative** —
free upside.

This stage also reports the per-pop-task breakdown for the current best strategy
**`calibrated_commitment`** (Stage 10) — does it fix or worsen the hard tasks?

## Hypothesis

A deployable **abstention** rule — learn on dev which datasets the model scores
*below uniform* on, and **predict uniform** (the benchmark's own baseline) for those
— lifts the pop (and pooled) mean by clipping the negative tasks, without harming
the datasets where the model is good.

## Configs

Post-hoc on **cached/cheap** predictions. Two prediction systems compared:
`anti_flattening` (cached) and `calibrated_commitment` (Stage 10 winner).

| system | what it is |
|---|---|
| `model` | the strategy's raw prediction (control) |
| `abstain_oracle` | **reference** — per item, take `max(S, 0)` (predict uniform whenever the model would score below uniform). Per-item ceiling, not deployable. |
| **`abstain_perdataset`** | deployable: on the dev-**fit** half, mark each dataset whose mean score < 0; on the dev-**eval** half, replace those datasets' predictions with uniform. |

Abstention uses **dataset identity only** (known at inference), never the truth —
it is reliability calibration, valid because SimBench test reuses the same datasets.

## Data & budget

- Full dev **pop** (n≈3585, all 20 datasets), split 50/50 per dataset into
  fit/eval. Grouped reported too (for `LatinoBarometro`).
- `anti_flattening` cached; `calibrated_commitment` predictions are new live calls
  (~$0.4). `val`/`test` untouched. Normalizers = Eq. 2 from the full dev split.

## Decision rule (preregistered — fixed before looking)

- **Primary:** dev-eval **pop** mean SimBench (and pooled), with bootstrap CI.
- **`abstain_perdataset` WINS** iff dev-eval pop mean beats the no-abstention `model`
  beyond the noise floor, **and** does not reduce the grouped mean.
- **Report** the per-task table (anti vs calibrated_commitment), the `abstain_oracle`
  ceiling, the fraction of the ceiling captured, and exactly which datasets the rule
  abstains on.
- **Carry-forward:** if it wins, abstention is a deployable post-hoc wrapper to stack
  on the chosen system (val-confirm later). `test` stays locked.
- **Negative:** if learned abstention does not beat the model (e.g. the bad datasets
  are not separable on dev, or the fit/eval decision is too noisy), report it.

---

## Results

Ran 2026-06-21 on full dev pop (n=3585), split 50/50 per dataset into fit/eval.
`gemini-3.1-flash-lite`. Run files: `outputs/runs/2026-06-21-abstention.*`.
(Numbers are on the *current* 3.1 system — more negative on the hard tasks than the
2.5 baseline figures, because more-committal models are more confidently wrong.)

### Per-pop-task — `calibrated_commitment` vs `anti_flattening`

| dataset | uniform | anti_flattening | calibrated_commitment | cc − anti |
|---|---|---|---|---|
| OSPsychMACH | −7.2 | −34.2 | **−55.0** | **−20.8** |
| MoralMachine | −2.7 | −13.9 | **−29.4** | **−15.5** |
| Choices13k | −0.8 | −6.9 | −6.5 | +0.4 |
| Jester | −0.6 | 14.4 | 16.0 | +1.6 |
| NumberGame | −3.0 | 17.6 | 22.6 | +5.0 |
| LatinoBarometro (pop) | −2.0 | 29.9 | 34.3 | +4.4 |
| ChaosNLI | −0.7 | 39.5 | 34.7 | −4.8 |
| ConspiracyCorr | −14.8 | 26.2 | 36.2 | +10.0 |
| ESS / ISSP / Afrobarometer / GlobalOpinionQA | ~1 | 42–47 | 42–47 | ~0 to +2 |
| OSPsychMGKT | −0.9 | 58.9 | 61.5 | +2.6 |
| OpinionQA | −0.8 | 60.5 | 63.2 | +2.7 |
| DICES | 0.0 | 72.2 | **75.4** | +3.2 |

**`calibrated_commitment` helps on most tasks (+2 to +13 on surveys/easy items) but
is far *worse* on the two unknowable, high-entropy tasks** (OSPsychMACH −20.8,
MoralMachine −15.5): its "commit to the leading answer" framing makes it confidently
wrong where the model has no basis. Commitment is a double-edged sword → pairs
naturally with abstention.

### Abstention on dev-eval pop (per-dataset rule fit on disjoint dev-fit half)

| system | model | + abstention | Δ | oracle (per-item) | captured |
|---|---|---|---|---|---|
| `anti_flattening` | 36.0 [33.4, 38.4] | 37.8 [35.7, 40.0] | +1.9 | 47.6 | 16% |
| **`calibrated_commitment`** | 36.0 [33.3, 38.6] | **39.1 [36.8, 41.4]** | **+3.1** | 50.7 | 21% |

Both abstain on **{Choices13k, MoralMachine, OSPsychMACH}** (the datasets whose
dev-fit mean is below uniform). Grouped is untouched (these are pop-only tasks).

### Verdict

- **Abstention WINS** (deployable, dev-fit→dev-eval): it lifts pop without touching
  the good datasets. **`calibrated_commitment` + abstention = 39.1 is the best
  combination** — abstention helps the committal strategy *more* (+3.1 vs +1.9),
  because it removes exactly the confident-wrong failures commitment creates.
- **Most of the prize is per-item, not per-dataset.** The oracle (clip every
  below-uniform *item* to uniform) reaches ~50; the per-dataset rule captures only
  ~20%. Within "good" datasets many individual items still score negative.

### Implications / next

1. Ship per-dataset abstention as a post-hoc wrapper (`AbstainCalibrator`): cheap,
   safe, +3.1 pop, composes with any strategy. Val-confirm later; `test` locked.
2. The real headroom is a **per-item confidence signal** ("does the model know this
   one?") → abstain per item, not per dataset. Candidate next stage (e.g. an
   LLM-elicited confidence, self-consistency across paraphrases, or a learned
   item-difficulty model gating the uniform fallback).
3. `LatinoBarometro` **grouped** (−9.4) would also benefit — extend abstention across
   splits, not just pop.
