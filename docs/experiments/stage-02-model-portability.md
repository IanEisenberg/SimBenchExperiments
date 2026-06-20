# Stage 02 — Model portability of the top conditioning strategies

- **Status:** PLANNED → _RUNNING_ → _DONE_
- **Run name:** `2026-06-20-model-sweep`
- **Preregistered:** 2026-06-20
- **Owner:** Ian + Claude

> Decision rules are fixed **before** the run. Append results below the line.

## Hypothesis

Stage 01 found that on the **grouped** (demographic-conditioning) task, better
conditioning beats the SimBench first-person baseline by +6–8 points on
`gemini-2.5-flash-lite`. This stage asks whether that result is a **property of
the method or of the model**:

- **H1 (portability):** the conditioning advantage holds on newer/stronger
  models — on each model, `anti_flattening` and `contextualized` still beat
  `faithful` on grouped by more than the noise floor.
- **H2 (capability scaling):** grouped SimBench score rises with model
  capability (`gemini-2.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.5-flash`)
  for the winning strategies.
- **H3 (conditioning as crutch?):** does a stronger model **shrink** the
  `faithful`→best gap (good conditioning matters less when the model is better),
  or **widen** it (better models exploit good conditioning more)?

## Configs to run

3 systems × 3 models. The current model is the cached reference (free); the two
new models are the live spend.

| | systems (predictor / strategy) |
|---|---|
| **systems (3)** | `faithful` (control), `anti_flattening`, `contextualized` |

| model key | OpenRouter id | role | $/Mtok (in/out) |
|---|---|---|---|
| `gemini-flash-lite` | google/gemini-2.5-flash-lite | reference (cached from Stage 01) | 0.10 / 0.40 |
| `gemini-3.1-flash-lite` | google/gemini-3.1-flash-lite | new | 0.25 / 1.50 |
| `gemini-3.5-flash` | google/gemini-3.5-flash | new (flagship) | 1.50 / 9.00 |

- **Sampling:** `temperature=0`, `seed=0`, `zero_shot` predictor with the named strategy.

## Data & budget

- **Subsample of dev** (cost control): stratified by dataset, `seed=0`, target
  **~1500 records**, both pop and grouped represented. The *same* subsample is
  scored on every model (seeded → identical records → comparable).
- `val`/`test` untouched.
- **Normalizers:** Eq. 2 scalars from the FULL split (as in Stage 01).
- **Realized subsample n / pop / grouped:** _filled at run time._
- **Cost cap:** **$15** (estimate ≈ $6: 3.1-flash-lite ~$0.9 + 3.5-flash ~$5.2;
  2.5-flash-lite is cache hits = $0). Pre-launch smoke test (1 call/model)
  confirms the new model ids return parseable output + native cost.

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** grouped mean SimBench score per (system × model), with 95%
  bootstrap CIs on the subsample. Pooled + pop/grouped reported as standard.
- **H1 verdict:** "portable" if, on **each** model, the best strategy beats
  `faithful` on grouped by more than the noise floor (half-width of the larger CI).
- **H2 verdict:** read the grouped score trend across the three models for the
  winning strategies; "scales" if monotone increasing beyond noise.
- **H3 verdict:** compare the `faithful`→best grouped gap across models.
- **Carry-forward (what to use for the eventual test-set number):** the
  (model, strategy) with the highest grouped score — **but cost-adjusted**: if
  `gemini-3.5-flash` (≈22× the cost of flash-lite) beats `gemini-3.1-flash-lite`
  by less than the noise floor on grouped, prefer the cheaper model.

## Artifacts

- `outputs/runs/2026-06-20-model-sweep.{topline.csv, results.json, meta.json}`
- Ledger: appended to `outputs/ledger/2026-06-20-baseline.jsonl` as `model.swap`
  branches off the `anti_flattening` and `contextualized` nodes (the growing tree).
- `data/cache/` — all calls.

---

## Results _(appended after the run)_

- **Subsample n / pop / grouped:** _pending_ · **Cost:** _pending_ · **Cache hits:** _pending_

### Grouped score by system × model (primary)

| system | 2.5-flash-lite | 3.1-flash-lite | 3.5-flash |
|---|---|---|---|
| _pending_ | | | |

**H1 (portable):** _pending_
**H2 (scales with capability):** _pending_
**H3 (faithful→best gap vs model):** _pending_
**Carry-forward (cost-adjusted):** _pending_

**Run files:** _pending_
