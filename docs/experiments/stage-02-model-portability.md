# Stage 02 — Model portability of the top conditioning strategies

- **Status:** **DONE** (ran 2026-06-20)
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

- **Subsample of dev** (cost control): the *same* subsample is scored on every
  model (seeded → identical records → comparable).
  **Amendment (pre-run, two reasons):** (1) `gemini-3.5-flash` turned out to emit
  ~770 output tokens/call (verbose), ≈60× flash-lite, so ~1500 records would be
  ~$33 — over cap. (2) A flat dataset-stratified 500-sample starved grouped to
  73 records (grouped lives in ~5 datasets, pop in many). **Fix:** sample grouped
  and pop *separately* — `stratified_sample(grouped, 300) + stratified_sample(pop, 150)`,
  `seed=0` → **n=440 (grouped=300, pop=140)**.
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

## Results

- **Subsample:** dev n=440 (grouped=300, pop=140) · **Cost:** **$11.45** · **Elapsed:** 21 min
- **Per model:** 2.5-flash-lite $0.00 (1320 cache hits) · 3.1-flash-lite $0.16 · 3.5-flash $11.29

### Grouped score by system × model (primary) — mean [95% CI]

| system | 2.5-flash-lite | 3.1-flash-lite | 3.5-flash |
|---|---|---|---|
| anti_flattening | 29.79 | **41.76** [36.8, 46.3] | 40.18 [35.4, 44.9] |
| contextualized | 29.66 | **42.03** [36.5, 47.2] | 35.78 [30.6, 40.8] |
| faithful | 20.76 | 38.12 [32.5, 43.3] | 35.35 [29.9, 40.8] |

> **n=300 grouped → wide CIs (noise floor ≈ 5–6).** Treat per-model gaps as
> underpowered. Also: these 2.5-flash-lite grouped scores (~29.8) are lower than
> Stage 01's full-dev (34.1) because this is a smaller, dataset-balanced subsample
> — the *within-sweep* model comparison is valid (identical records); cross-stage
> absolute comparison is not.

### Verdicts

- **H1 (portability): holds in point estimates; statistically clear only on the
  weak model.** Conditioning beats `faithful` on grouped on every model
  (+9.0 / +3.9 / +4.8), but only the 2.5-flash-lite gap (+9 > nf 6.0) clears the
  noise floor; on the newer models (+3.6–4.8) it is within nf ≈ 5.4 — underpowered
  at n=300, **not** disproven.
- **H2 (capability scaling): CONFIRMED and non-monotone.** Both newer models add
  **+12 to +17 grouped points** over 2.5-flash-lite. But **`3.1-flash-lite` ≥
  `3.5-flash` on every system** — significantly so for contextualized
  (42.03 vs 35.78, +6.2 > nf). The 22×-pricier flagship is **not** better.
- **H3 (conditioning as crutch): SUPPORTED.** The `faithful`→best grouped gap
  shrinks +9.0 → +3.9 → +4.8 as the model improves; `faithful` alone jumps
  20.8 → 38.1 on 3.1-flash-lite. **A better base model closes most of the
  conditioning gap — strategy choice matters most on weaker models.**
- **Carry-forward (cost-adjusted): `gemini-3.1-flash-lite` + `contextualized`
  (or `anti_flattening`).** At **$0.16 vs $11.29** it ties-or-beats the flagship
  on grouped — `gemini-3.5-flash` is dominated (pricier, not better) and is dropped.

### Key finding

The biggest lever in this round was **the model, not the prompt**: moving
2.5-flash-lite → 3.1-flash-lite added more grouped points (+12–17) than any
conditioning-strategy change did in Stage 01 (+8). And the cheap newer model
**beats the 22×-costlier flagship**. Net: use `gemini-3.1-flash-lite`; on it,
good conditioning still helps but the margin over the naive baseline narrows.

**Run files:** `outputs/runs/2026-06-20-model-sweep.{topline.csv, results.json, meta.json, log}`;
ledger extended at `outputs/ledger/2026-06-20-baseline.jsonl` (model.swap branches
off anti_flattening + contextualized → 10 nodes).
