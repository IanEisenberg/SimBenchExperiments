# Stage 09 — Feature-predicted target entropy (dev)

- **Status:** **DONE** (ran 2026-06-21) — **negative, with the mechanism nailed:**
  even a good target-entropy model (eval R²=0.58) can't convert the +9.5 oracle into
  score, because the oracle's gain is **gated behind mode correctness** — it helps
  +16.8 where the top option is right and *hurts* on the 39% of items where it's
  wrong. Spread is entangled with location; post-hoc entropy correction is closed.
- **Run name:** `2026-06-21-feature-entropy`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> `dev` only — fit on the **disjoint** dev-fit sample, evaluate on the standard
> dev-eval sample (same split as Stage 08). No `val`/`test` contact.

## Context

Stage 08: the +9.5 grouped oracle (temper to truth entropy) is real, but **not**
recoverable from the prediction's own entropy (`entropy_target` fit to gain 0).
The prediction's entropy is compressed to ~constant, so it can't say which
questions are contested. A quick diagnostic shows **question features predict the
truth's entropy at out-of-sample R² ≈ 0.58** — vs 0.33 from the prediction's own
entropy. So an *external* contestedness signal exists. This stage tries to convert
it into score.

## Hypotheses

- **H1 (features beat own-entropy):** a calibrator that predicts each item's target
  entropy from **question features** and tempers to it beats identity on dev-eval
  grouped beyond the noise floor (where Stage 08's own-entropy version could not).
- **H2 (which features):** ablations locate the signal — dataset identity + option
  count (`meta_only`) vs the prediction's *shape* (`shape_only`) vs both (`full`).
- **H3 (partial capture):** `full` captures a meaningful fraction (target ≥ ~30%) of
  the +9.5 oracle headroom — not all, since R² ≈ 0.58 < 1.

## Configs

Post-hoc on **cached** `anti_flattening` @ `gemini-3.1-flash-lite`. All fittable
calibrators are **fit on dev-fit, applied to disjoint dev-eval** (same samples as
Stage 08: eval grouped 700 + pop 300 seed=42; fit grouped 500 + pop 250 seed=7).

| system | calibrator | features |
|---|---|---|
| `identity` | — | control |
| `own_entropy` | `EntropyTargetCalibrator` | prediction's own entropy (Stage-08 ref) |
| **`feat_full`** | `FeatureEntropyTargetCalibrator` | dataset + option count + conditioning + prediction shape |
| `feat_meta_only` | `…(use_pred_shape=False)` | dataset + option count + conditioning |
| `feat_shape_only` | `…(use_dataset=False)` | option count + conditioning + prediction shape |
| `entropy_oracle` | reference | tempers to the **truth** entropy (ceiling, not deployable) |

`FeatureEntropyTargetCalibrator` ridge-regresses `truth_H` on the features
(continuous features z-scored on the fit set), then grid-fits a blend `gain`; it
tempers each prediction to `H_pred + gain·(Ĥ_feat − H_pred)`.

## Data & budget

Same as Stage 08. `anti_flattening` predictions cached/cheap; calibrators CPU.
`val`/`test` untouched. Normalizers = Eq. 2 from the full dev split.

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** dev-eval **grouped** mean SimBench, 95% bootstrap CI (n=700).
- **Noise floor:** half-width of the larger compared CI (~3 at n=700).
- **`feat_full` WINS** iff dev-eval grouped beats `identity` beyond the noise floor.
  (Stage 08 already showed the global-temp control loses, so identity is the bar.)
- **Report** the oracle ceiling and the **fraction of oracle headroom captured** by
  each arm, plus the fitted gain and the feature-model out-of-sample R² on the eval
  truths. Ablations attribute the signal (meta vs shape).
- **Carry-forward:** if `feat_full` wins, it stacks on `anti_flattening` as a
  candidate for a later **val**-confirmation round (one query). `test` stays locked.
- **Negative result:** if no feature arm beats identity beyond noise, the spread
  headroom is not convertible into TVD by entropy-only tempering at this fidelity —
  the remaining error needs the *shape/location* axis, not just the entropy.

---

## Results

Ran 2026-06-21. Fit on dev-fit (n=715), scored on dev-eval (n=990; grouped n=700).
Run files: `outputs/runs/2026-06-21-feature-entropy.{topline.csv,meta.json}`.

**Feature model:** out-of-sample (eval) `truth_H` R² = **0.576** — features genuinely
predict contestedness. Yet every arm fit **gain = 0**:

| system | grouped | 95% CI | vs identity | captured |
|---|---|---|---|---|
| `identity` | **45.01** | [42.0, 47.9] | — | — |
| `own_entropy` (Stage-08) | 45.01 | [42.0, 47.9] | +0.0 | 0% |
| **`feat_full`** | 45.01 | [42.0, 47.9] | +0.0 | 0% |
| `feat_meta_only` | 45.01 | [42.0, 47.9] | +0.0 | 0% |
| `feat_shape_only` | 45.01 | [42.0, 47.9] | +0.0 | 0% |
| `entropy_oracle` *(ref)* | **54.47** | [51.0, 57.7] | +9.5 | 100% |

### Why a good entropy model still captures nothing (the mechanism)

A diagnostic forced-`gain` sweep on **eval** (not selection — explanation) shows
`gain=0` is correct, not a conservative fit-set artifact: every positive gain
**monotonically hurts**.

| forced gain | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---|---|---|---|---|---|
| `feat_full` grouped | 45.01 | 45.07 | 44.21 | 42.58 | 40.28 | 37.51 |

Decomposing the **oracle** by whether the prediction's top option matches the
truth's:

| item set | n (of 700) | identity | oracle | lift |
|---|---|---|---|---|
| **mode-correct** (top option right) | 428 (61%) | 59.6 | 76.3 | **+16.8** |
| **mode-wrong** (top option wrong) | 272 (39%) | 22.1 | 20.1 | **−2.0** |

The oracle's entire +9.5 lives on **mode-correct** items. On the **39%** of grouped
items where the predicted top option is wrong, fixing the entropy **hurts** —
sharpening toward the right spread on the *wrong* options just concentrates mass in
the wrong place. A deployable calibrator cannot tell the two apart, so applying any
entropy correction indiscriminately nets ≤ 0 → `gain = 0`.

### Verdict

- **H1 (features beat own-entropy) — FALSE in score terms.** Features predict
  entropy far better (R² 0.58 vs 0.33), but it doesn't matter: tempering to the
  target can't beat identity (it loses for any gain > 0). No arm clears the bar.
- **The real finding:** the spread error and the **location** (mode) error are
  *entangled*. The +9.5 spread headroom is **locked behind mode correctness** —
  unrecoverable by any post-hoc entropy transform, because ~40% of items have the
  wrong peak and would be made worse.
- **Post-hoc calibration is now closed** (Stages 06, 08, 09 all negative, now with a
  mechanism). Incumbent `anti_flattening` @ `gemini-3.1-flash-lite` unchanged;
  `test` untouched.

### Implication for next stage

The binding constraint is **mode/location accuracy** — getting the *right top
option* on the 39% of grouped items where we currently miss it. That is a
predictor / prompt / model problem (e.g. post-stratification to assemble the right
options, a stronger/reasoning model, or retrieval), **not** a calibration problem.
Once the mode is right, the entropy headroom (+16.8 on those items) becomes
reachable — calibration may be worth revisiting *after* a location fix, not before.
