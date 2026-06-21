# Stage 08 — Entropy de-compression recalibration (dev)

- **Status:** **DONE** (ran 2026-06-21) — **negative-with-direction:** the +9.5
  oracle headroom is **not recoverable from the prediction's own entropy**
  (`entropy_target` fit to gain 0 = identity); global temps hurt. Round 2 must
  estimate the target spread from question features. Incumbent unchanged.
- **Run name:** `2026-06-21-entropy-recalib`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> `dev` only — fit on a **disjoint** dev sample, evaluate on the standard dev eval
> sample. No `val`/`test` contact.

## Context — the diagnostic (notebook 04 / `scrye.decompose`)

The error decomposition showed most of our grouped error is **concentration**
(wrong spread), not location. Digging into the spread axis on the winner
(`anti_flattening` @ `gemini-3.1-flash-lite`):

- Predicted vs true normalized entropy regresses with **slope ≈ 0.48**
  (`pred_H ≈ 0.46 + 0.48·truth_H`). The model's spread is **compressed toward a
  constant (~0.8)** — it barely tracks how contested a question actually is.
- Net it is **too diffuse** (mean entropy gap +0.09, 78% of items over-spread),
  but this is dominated by **consensus** questions it over-spreads; contested
  questions are roughly right. The error is therefore **bidirectional**, not a
  constant offset — which is exactly why Stage 06's global/one-sided calibrators
  failed.
- **Oracle ceiling:** tempering each prediction to the *truth's* entropy (keeping
  the option ranking fixed) lifts grouped **+8.4** (41.8 → 50.2) on
  `anti_flattening` — a prize larger than the whole strategy win. The headroom is
  real *if* we can estimate the target spread without the truth.

## Hypotheses

- **H1 (de-compression helps):** a calibrator that estimates each item's target
  entropy from the prediction's *own* entropy (inverting the compression) and
  tempers the distribution to that target beats **both** identity and the best
  **global** temperature on dev-eval grouped.
- **H2 (global can't):** the best single global temperature — including
  *sharpening* (T<1) — does **not** clear the noise floor over identity. The error
  is a compression, not an offset; a global knob trades consensus accuracy for
  contested accuracy.
- **H3 (direction/gain):** the fitted de-compression expands entropy variance
  (gain > 0); a full inversion (gain ≈ 1) is near-optimal.

## Configs

Post-hoc calibrators on **cached** `anti_flattening` @ `gemini-3.1-flash-lite`
predictions (no new LLM calls). Each fittable calibrator is **fit on the dev-fit
sample and applied to the disjoint dev-eval sample**.

| system | what it is | fit? |
|---|---|---|
| `identity` | `anti_flattening` raw (control) | — |
| `global_temp` | best global temperature `T` (grid incl. T<1 sharpening) | dev-fit |
| `entropy_temp` | Stage 06's `T=1+slope·H(pred)` (one-sided ref) | dev-fit |
| **`entropy_target`** | estimate target H by regressing truth_H on pred_H, temper each item to `H_pred + gain·(Ĥ−H_pred)`; `gain` grid-fit | dev-fit |
| `entropy_oracle` | **reference only** — temper to the *truth's* entropy (not deployable) | — |

`entropy_target` is a new `Calibrator` (`EntropyTargetCalibrator`). Adding it is
the off-registry calibrator this stage introduces (human-authorized for this
exploration).

## Data & budget

- **dev-eval (scored):** `stratified_sample(grouped, 700) + stratified_sample(pop, 300)`,
  `seed=42` — the standard eval set used by prior stages; grouped CI on n=700.
- **dev-fit (calibrator fitting only):** a **disjoint** dev sample,
  `stratified_sample(grouped, 500) + stratified_sample(pop, 250)`, `seed=7`,
  drawn from dev records **not** in the eval set. Calibrators never see eval
  truths at fit time.
- **val/test:** untouched. **Normalizers:** Eq. 2 from the full dev split.
- **Cost:** ~$0 — `anti_flattening` predictions are cached; calibrators are CPU.

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** dev-eval **grouped** mean SimBench, 95% bootstrap CI (n=700).
- **Noise floor:** half-width of the larger compared CI (~3 at n=700). "Beats" =
  mean gap > noise floor.
- **`entropy_target` WINS** iff it beats **both** `identity` and `global_temp` on
  dev-eval grouped beyond the noise floor. Beating identity but not `global_temp`
  means it is only a global shift in disguise — not a win.
- **Report** the `entropy_oracle` ceiling and the **fraction of oracle headroom
  captured** by each method, plus the fitted regression (slope/intercept) and the
  chosen `gain`/`T`.
- **Carry-forward:** if `entropy_target` wins, it becomes a candidate to stack on
  `anti_flattening` for a later val-confirmation round. `test` stays locked.
- **Negative result:** if no calibrator beats identity beyond noise, post-hoc
  entropy correction cannot capture the oracle headroom — the spread error is not
  recoverable from the prediction's own entropy, and Round 2 must estimate the
  target spread from question content instead (or move to a predictor-level fix).

---

## Results

Ran 2026-06-21. Calibrators fit on the disjoint dev-fit sample (n=715), scored on
dev-eval (n=990; grouped n=700). `anti_flattening` predictions cached/cheap.
Run files: `outputs/runs/2026-06-21-entropy-recalib.{topline.csv,meta.json}`.

**Fitted params:** `global_temp T=1.5`; `entropy_temp slope=1.0`;
**`entropy_target gain=0.0`** with regression `truth_H ≈ 0.07 + 0.80·pred_H`.

| system | pooled | **grouped** | 95% CI | pop | vs identity | vs global |
|---|---|---|---|---|---|---|
| `identity` (control) | 41.08 | **45.01** | [42.0, 47.9] | 31.59 | — | — |
| `global_temp` (T=1.5) | 37.75 | 40.89 | [38.4, 43.3] | 30.16 | −4.1 | — |
| `entropy_temp` (slope 1.0) | 35.81 | 38.69 | [36.1, 41.0] | 28.85 | −6.3 | −2.2 |
| **`entropy_target`** (gain 0) | 41.08 | 45.01 | [42.0, 47.9] | 31.59 | +0.0 | +4.1 |
| `entropy_oracle` *(ref, uses truth)* | 52.15 | **54.47** | [51.0, 57.7] | 46.56 | **+9.5** | — |

### Verdict against the decision rule

- **H1 (de-compression from own entropy) — FALSE.** `entropy_target`'s grid chose
  **gain = 0**, collapsing to identity: no positive de-compression gain improved
  the fit-set score, so it captured **0%** of the +9.5 oracle headroom. (It "beats"
  global_temp only by *being* identity.)
- **H2 (global can't) — TRUE.** Both `global_temp` and `entropy_temp` fell **below**
  identity (−4.1, −6.3). A global/one-sided knob can only flatten, but on average we
  are already too diffuse, so it actively hurts — re-confirming Stage 06 with a
  finer, two-sided sweep.
- **Net: post-hoc entropy correction from the prediction alone cannot move the
  number.** Incumbent `anti_flattening` unchanged; `test` untouched.

### Why (the mechanism, and the pointer to Round 2)

The oracle proves a large spread prize (+9.5). `entropy_target` can't claim it
because it estimates the target entropy from the prediction's **own** entropy —
and that entropy is compressed *precisely because it is nearly constant* (~0.8
regardless of the question). The fitted `truth_H ≈ 0.07 + 0.80·pred_H` has little
spread on the input, so the implied target barely moves from the prediction, and a
larger gain just amplifies noise (sharpening genuinely-contested items by mistake)
— hence gain 0.

**Implication:** to capture the headroom we need an **independent signal of
contestedness** — one not derived from the model's own (compressed) output. That
is Round 2 (Stage 09): predict the target entropy from **question features**
(dataset, option count, conditioning, prediction *shape* features), then temper to
it.
