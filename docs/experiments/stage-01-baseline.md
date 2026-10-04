# Stage 01 — Strategy baseline

- **Status:** **DONE** (ran 2026-06-20)
- **Run name:** `2026-06-20-baseline`
- **Preregistered:** 2026-06-20
- **Owner:** Ian + Claude

> Decision rules in this file are fixed **before** the run. Append results below
> the line; do not edit the preregistration above it after running.

## Hypothesis

SimBench's headline finding is that **first-person demographic conditioning
degrades** group-level scores. So:

- **H1 (conditioning style):** the three *distributional* framings
  (`representative_sample`, `anti_flattening`, `contextualized`) beat the
  first-person control (`simbench_faithful`) on dev mean SimBench score.
- **H2 (flattening):** `anti_flattening` — which explicitly preserves
  within-group minority mass — has the lowest `frac_below_uniform` (it should
  most reduce the catastrophic-collapse tail).
- **H3 (direction sensitivity):** distributional framings have higher
  `cf_alignment` (they get the *direction* of demographic shifts right more
  often) than the first-person persona.

This stage establishes the **baseline to beat** for the new persona mechanisms
(Stage 02: diversity-elicitation CoT and Monte-Carlo individuals).

## Configs to run

All systems share one model and the `zero_shot` predictor; only the persona
`PromptStrategy` varies. Run via `experiment.compare()` so all systems see the
identical dev records and one normalizer set.

| System label | predictor | strategy | role |
|---|---|---|---|
| `uniform` | `uniform` | — | floor (S ≈ 0 by construction) |
| `faithful` | `zero_shot` | `simbench_faithful` | **control** (paper baseline) |
| `representative_sample` | `zero_shot` | `representative_sample` | distributional |
| `persona_embodiment` | `zero_shot` | `persona_embodiment` | immersive first-person |
| `anti_flattening` | `zero_shot` | `anti_flattening` | distributional + minority mass |
| `contextualized` | `zero_shot` | `contextualized` | distributional + context anchor |

- **Model:** `gemini-2.5-flash-lite` (`DEFAULT_MODEL`).
- **Sampling:** `temperature=0`, `seed=0` (deterministic; cache-stable).

## Data & budget

- **Split:** `make_split(seed=0, unit="question")`, default fractions
  `{dev:0.5, val:0.25, test:0.25}`. Required questions pinned to **test**.
- **Evaluate on:** `dev` **only**. `val`/`test` are not touched this stage.
- **n (dev records):** _filled at run time from `split.summary()`._
- **Normalizers:** `build_normalizers()` on the **full split** (Eq. 2
  denominator must be dataset-level).
- **Cost cap:** **$10** for the whole stage. Confirm against the estimate
  `dev_n × 6 systems × per-call cost` before launching; on a cached re-run the
  marginal cost is $0.

## Decision rule (preregistered — fixed before looking)

- A non-control system **wins** if its dev `mean_score` exceeds `faithful`'s by
  **more than the bootstrap noise floor** (the half-width of the larger of the
  two 95% bootstrap CIs). A gap inside the noise floor is "no difference."
- **H1/H2/H3** are judged on the table below: which systems clear the noise
  floor, the `frac_below_uniform` ranking, and the `cf_alignment` ranking.
- The **carry-forward winner** into Stage 02 is the single highest dev
  `mean_score` among the systems that also have `cf_alignment ≥ faithful`
  (we want accuracy that isn't bought by getting directions wrong). Ties broken
  by lower `frac_below_uniform`.

## Artifacts this stage will produce

- `data/cache/<sha>.json` — every LLM call (always).
- `outputs/runs/2026-06-20-baseline.parquet` — the topline table + per-system
  frames _(pending the `compare()` persistence helper; until then the table is
  pasted below and the run lives in cache + this file)._

---

## Results

- **Dev n:** 6151 · **Calls:** 30,536 (219 cache hits) · **Cost:** **$1.78** · **Elapsed:** 22.5 min
- **Config:** `gemini-2.5-flash-lite`, temp 0, seed 0, `compare()` on dev, 16 workers.

### Pooled — preregistered headline

| system | mean_score | 95% CI | frac_below_uniform | cf_alignment |
|---|---|---|---|---|
| **anti_flattening** | **30.09** | [28.80, 31.30] | 0.227 | 0.217 |
| persona_embodiment | 26.84 | [25.43, 28.20] | 0.243 | 0.254 |
| representative_sample | 26.73 | [25.29, 28.09] | 0.254 | 0.208 |
| contextualized | 24.96 | [23.44, 26.31] | 0.250 | 0.220 |
| faithful (control) | 24.78 | [23.32, 26.18] | 0.260 | 0.217 |
| uniform | 2.08 | [1.09, 3.10] | 0.449 | — |

### By split — pop vs grouped (the conditioning read)

| system | pop mean [CI] (n=3585) | grouped mean [CI] (n=2566) |
|---|---|---|
| anti_flattening | 27.25 [25.45, 29.09] | 34.07 [32.20, 35.96] |
| persona_embodiment | 24.58 [22.57, 26.59] | 30.01 [28.04, 31.86] |
| representative_sample | 22.35 [20.20, 24.32] | 32.85 [30.92, 34.79] |
| contextualized | 18.14 [15.97, 20.40] | **34.50** [32.70, 36.32] |
| faithful | 23.63 [21.59, 25.56] | 26.38 [24.30, 28.41] |
| uniform | −0.34 [−1.68, 1.05] | 5.45 [4.11, 6.90] |

### Hypotheses

- **H1 (distributional > faithful): CONFIRMED on grouped.** All four non-control
  strategies beat faithful on the grouped (conditioning) task by **+3.6 to +8.1**
  points, every gap above the **2.06** noise floor. Pooled: 3 of 4 win;
  contextualized is a pooled ~tie only because it underperforms on pop.
- **H2 (anti_flattening lowest collapse): CONFIRMED.** anti_flattening has the
  lowest `frac_below_uniform` (0.227 vs faithful 0.260).
- **H3 (distributional higher cf_alignment): NOT CONFIRMED.** cf_alignment does
  not track framing — persona_embodiment leads (0.254), representative_sample is
  lowest (0.208), the rest cluster at faithful's 0.217. Accuracy did not buy
  directional sensitivity; cf_alignment is weak/undifferentiated at this scale.

### Key finding

On the task where conditioning applies (grouped), **better demographic
conditioning beats the SimBench first-person baseline decisively**:
contextualized (34.50) ≈ anti_flattening (34.07) > representative_sample (32.85)
> persona_embodiment (30.01) ≫ faithful (26.38). This sharpens SimBench's
"conditioning hurts" into **"_naive_ first-person conditioning hurts;
distributional / diversity-aware conditioning helps a lot."** On pop (little/no
demographic content) the elaborate framings add noise — contextualized (18.14)
and representative_sample (22.35) fall *below* faithful (23.63); only
anti_flattening (27.25) and persona_embodiment (24.58) hold up.
**anti_flattening is the most robust across both tasks.**

### Carry-forward into Stage 02

- **Preregistered rule (pooled mean, gated cf_alignment ≥ faithful):** →
  **persona_embodiment** (highest pooled mean among cf-eligible
  {persona_embodiment, contextualized, faithful}).
- **⚠ Gate artifact:** the cf-alignment gate excluded **anti_flattening** — best
  overall (30.09 pooled), co-best on grouped, H2 collapse-resistance winner — by a
  **0.0002** cf_alignment margin (0.217235 vs 0.217461). That is pure noise.
- **Recommendation (amendment, justified):** carry forward **anti_flattening** as
  the primary Stage 02 base (most robust across pop+grouped; its
  within-group-diversity thesis is exactly what Stage 02's Monte-Carlo-individuals
  and diversity-elicitation mechanisms amplify), with **contextualized** as a
  secondary base (top on grouped). Final base confirmed in the Stage 02 prereg.

**Run files:** `outputs/runs/2026-06-20-baseline.{topline.csv, results.json, meta.json, log}`;
all 30,536 calls cached in `data/cache/` (re-run is free).
