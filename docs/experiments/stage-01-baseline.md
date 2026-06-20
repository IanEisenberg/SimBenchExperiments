# Stage 01 — Strategy baseline

- **Status:** PLANNED → _RUNNING_ → _DONE_
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

## Results _(appended after the run)_

- **Status when run:** _pending_
- **Dev n:** _pending_  ·  **Total cost:** _pending_  ·  **Cache hits:** _pending_

| system | model | n | mean_score | ci_low | ci_high | frac_below_uniform | cf_alignment |
|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | |

**Noise floor (half-width):** _pending_

**H1 (distributional > faithful):** _pending_
**H2 (anti_flattening lowest frac_below_uniform):** _pending_
**H3 (distributional higher cf_alignment):** _pending_

**Carry-forward winner into Stage 02:** _pending_

**What we learned / implications for Stage 02:** _pending_

**Run files:** `outputs/ledger/` or `outputs/runs/2026-06-20-baseline.*` — _pending_
