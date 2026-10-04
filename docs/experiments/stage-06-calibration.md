# Stage 06 — Calibration sweep on anti_flattening (dev, n=1000)

- **Status:** **DONE** (ran 2026-06-21) — negative result: no calibrator beats identity
- **Run name:** `2026-06-21-calibration`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> Decision rules are fixed **before** the run. Append results below the line.

## Context

Stages 01–05 locked the system: **`anti_flattening` @ `gemini-3.1-flash-lite`**
(dev grouped 48.81, val grouped 53.02 — confirmed). The three registered
calibrators in `levers.py` have never been evaluated; all prior runs used the
`IdentityCalibrator` (no-op). This stage explores whether post-hoc calibration
can squeeze further TVD reduction out of the existing predictions.

No new LLM calls are needed — all `anti_flattening` dev predictions are cached
from Stage 04's full-dev run. This is pure post-processing.

## Hypothesis

Instruct-tuned LLMs are known to be over-confident (mode-seeking, RLHF-induced).
The three calibrators each address a different failure mode:

- **H1 (TempScaling, T > 1):** The predictions are systematically too peaked; a
  global temperature > 1 flattens them toward the true human distribution and
  reduces TVD.
- **H2 (EntropyTempScaling, slope > 0):** The over-sharpness is
  entropy-conditional — the model is most over-confident on diffuse (high-entropy)
  items (SimBench r = −0.942). An entropy-proportional temperature corrects this
  selectively without touching already-well-calibrated items.
- **H3 (DirichletCalibrator, alpha > 0):** Minority response options get zero or
  near-zero predicted mass; additive smoothing restores plausible tail mass and
  reduces TVD on multi-modal items.

## Configs

**Base system (control):** `anti_flattening` + `IdentityCalibrator` (T=1 / slope=0 / alpha=0).

Fixed hyperparameter grid — no fitting on the eval data.

| calibrator | hyperparameter values |
|---|---|
| `TempScaling` | T ∈ {0.7, 1.25, 1.5, 2.0, 3.0} |
| `EntropyTempScaling` | slope ∈ {0.5, 1.0, 2.0, 3.0} |
| `DirichletCalibrator` | alpha ∈ {0.01, 0.05, 0.1, 0.2} |

**Total configs:** 1 (identity) + 5 + 4 + 4 = **14**

All on `gemini-3.1-flash-lite`, `anti_flattening` strategy. The calibrator is
applied per-prediction after the (cached) LLM call, as a pure transform.

## Data & budget

- **Subsample of dev (n=1000):** `stratified_sample(grouped, 700) +
  stratified_sample(pop, 300)`, `seed=42`. Separate grouped/pop stratification
  preserves coverage of both split types. Same 1000 records for all configs.
- **val/test:** untouched.
- **Normalizers:** Eq. 2 scalars from the FULL dev split (as in all prior stages).
- **Cost:** ~$0 — all LLM calls served from cache (Stage 04 ran full dev with
  this exact model+strategy). Calibrators are CPU-only transforms.

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** grouped mean SimBench score, 95% bootstrap CI on n=700.
- **Per-calibrator type:** the best hyperparameter value within each type is
  the "type winner". A type winner **beats identity** if its grouped mean exceeds
  identity's by more than the noise floor (half-width of the larger CI; expected
  ~3–4 at n=700).
- **Overall winner:** the best config across all 14 — if it beats identity beyond
  the noise floor, it is a **calibration candidate** carried to a val confirmation
  round (one additional val query). If it does not clear the noise floor, identity
  is unchanged — no calibration improvement is claimed.
- **Negative result:** if all 14 configs are within noise of identity (or worse),
  this is a clean negative: post-hoc calibration with fixed hyperparameters does
  not help `anti_flattening` on this dataset and model.
- Report the full hyperparameter curves (score vs. T, slope, alpha) to inform
  where calibration pushes the distribution.

---

## Results

- **Subsample:** dev grouped=700, pop=290 (n=990 total) · **Cost:** $0.00 (100% cache hits) · **Elapsed:** ~6 s · model `gemini-3.1-flash-lite`

### Grouped score by configuration (primary) — mean [95% CI]

| system | grouped | 95% CI | pop | pooled | frac_below_uniform |
|---|---|---|---|---|---|
| **identity** (control) | **45.01** | [42.04, 47.86] | 31.59 | 41.08 | 0.149 |
| dirichlet_a0.01 | 44.59 | [41.71, 47.40] | 31.68 | 40.81 | 0.142 |
| temp_T1.25 | 43.66 | [40.99, 46.23] | 31.65 | 40.14 | 0.130 |
| entropy_slope0.5 | 42.66 | [40.01, 45.19] | 31.10 | 39.28 | 0.136 |
| dirichlet_a0.05 | 42.00 | [39.33, 44.58] | 30.53 | 38.64 | 0.138 |
| temp_T1.5 | 40.89 | [38.39, 43.28] | 30.16 | 37.75 | 0.132 |
| temp_T0.7 | 40.74 | [37.32, 44.05] | 25.42 | 36.25 | 0.199 |
| entropy_slope1.0 | 38.69 | [36.14, 41.00] | 28.85 | 35.81 | 0.146 |
| dirichlet_a0.1 | 38.55 | [36.00, 40.97] | 28.00 | 35.46 | 0.147 |
| temp_T2.0 | 35.31 | [32.89, 37.58] | 25.68 | 32.49 | 0.151 |
| dirichlet_a0.2 | 32.90 | [30.52, 35.13] | 22.92 | 29.98 | 0.158 |
| entropy_slope2.0 | 31.80 | [29.39, 34.03] | 24.12 | 29.55 | 0.171 |
| entropy_slope3.0 | 27.06 | [24.61, 29.34] | 20.19 | 25.05 | 0.194 |
| temp_T3.0 | 27.47 | [25.02, 29.75] | 18.10 | 24.73 | 0.192 |

> **Noise floor (grouped):** ≈2.91 (half-width of identity's CI [42.04, 47.86]).

### Verdicts — all calibrators LOST or tied

- **H1 (TempScaling > identity): REFUTED.** Best temperature config is T=1.25
  (grouped 43.66 vs 45.01, −1.35) — within noise but negative. T<1 (sharpening)
  and T≥2 (heavy flattening) are substantially worse. The predictions are not
  over-peaked in a way that global temperature scaling can fix.
- **H2 (EntropyTempScaling > identity): REFUTED.** Best is slope=0.5 (42.66 vs
  45.01, −2.35) — within noise but negative. Higher slopes get progressively
  worse. Entropy-adaptive flattening does not help here.
- **H3 (DirichletCalibrator > identity): REFUTED.** Best is alpha=0.01 (44.59
  vs 45.01, −0.42) — within noise and essentially tied. Tiny smoothing is
  harmless but larger alpha is harmful. There is no tail-mass deficit that Dirichlet
  smoothing can recover.
- **Overall: no calibration candidate to carry forward.** Identity wins across all
  14 configurations. The `anti_flattening` strategy's predictions do not benefit
  from post-hoc distribution adjustment.

### Key finding

A clean negative. The `anti_flattening` prompt strategy already produces
well-calibrated distributional predictions — the within-group-diversity framing
effectively does what calibration would try to do. Post-hoc correction adds
nothing and generally degrades performance monotonically with the strength of
the intervention. The system to carry into any future test-set evaluation is
**`anti_flattening` + identity calibrator** exactly as locked in Stage 05.

**Run files:** `outputs/runs/2026-06-21-calibration.{topline.csv, meta.json}`
