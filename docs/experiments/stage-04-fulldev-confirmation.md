# Stage 04 — Full-dev confirmation of the top systems

- **Status:** **DONE** (ran 2026-06-20) — anti_flattening confirmed best; contextualized within noise of faithful
- **Run name:** `2026-06-20-fulldev-confirm`
- **Preregistered:** 2026-06-20
- **Owner:** Ian + Claude

> Decision rule fixed **before** the run. Append results below the line.

## Why

Stages 02–03 ran the top systems on **dev subsamples** (≤500 grouped), where the
`contextualized`/`anti_flattening` advantage over `faithful` (~+3.5 grouped) sat
**inside the noise floor** (~4–5). This run evaluates all three on the **full dev
grouped set (n=2,566)** to shrink the CIs and settle whether the conditioning
advantage is real, before we spend any `val`/`test` budget.

## Configs

All on `gemini-3.1-flash-lite`, `temperature=0`, `zero_shot` predictor.

| system | strategy | role |
|---|---|---|
| `faithful` | simbench_faithful | control (SimBench first-person baseline) |
| `anti_flattening` | anti_flattening | candidate |
| `contextualized` | contextualized | candidate (Stage 02/03 point-estimate leader) |

## Data & budget

- **Full dev:** all 6,151 records (grouped=2,566, pop=3,585). No subsample.
- `val` (3,767) and `test` (3,592) **untouched** — this is still iteration on dev.
- Normalizers from the FULL split (Eq. 2).
- **Cost cap:** $10 (estimate ≈ $3; ~18.5k calls on 3.1-flash-lite, single-call,
  partly cached from Stages 02–03; throttled workers + `max_retries=10`).

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** grouped mean SimBench score per system, 95% bootstrap CI on
  n=2,566. Pooled + pop/grouped reported as standard.
- **"Beats faithful":** a candidate's grouped mean exceeds `faithful`'s by more
  than the noise floor (half-width of the larger of the two CIs — now ~1–1.5).
- **Carry-forward to the val/test step:** the highest grouped scorer among the
  candidates that beat `faithful`; ties broken by lower `frac_below_uniform`. If
  neither candidate clears the noise floor over `faithful`, carry **`faithful`**
  (the simplest system) and report that conditioning gave no measurable gain at
  this scale on this model.

---

## Results

- **n:** grouped=2,566, pop=3,585 · **Cost:** $1.91 · **15,803 calls** (2,650 cache hits) · **52.5 min**

| system | grouped | 95% CI | pop | pooled | cf_alignment |
|---|---|---|---|---|---|
| **anti_flattening** | **48.81** | [47.33, 50.38] | 35.10 | 40.82 | 0.388 |
| contextualized | 47.60 | [45.78, 49.40] | 32.49 | 38.79 | 0.408 |
| faithful (control) | 46.01 | [44.16, 47.77] | 29.69 | 36.49 | 0.374 |

**Noise floor (grouped):** ≈1.81 (half-width of the larger CI).

### Verdict

- **`anti_flattening` BEATS `faithful`: +2.81 > noise floor.** The first clean,
  statistically-resolved confirmation that conditioning helps — and it also wins
  on pop (35.10) and has cf_alignment ≥ faithful.
- **`contextualized` is WITHIN noise of `faithful`: +1.59 < 1.81.** Its lead on
  the small Stage 02/03 subsamples (where it nosed ahead of anti_flattening) does
  **not** hold at full dev — it is not a statistically clear improvement over the
  baseline here.
- `anti_flattening` vs `contextualized`: +1.22 (within noise; anti_flattening leads).

### Carry-forward to the val/test step

**`anti_flattening` on `gemini-3.1-flash-lite`** — the only candidate that beats
`faithful` beyond the noise floor, best on pop, and cf_alignment ≥ faithful.
**This revises the earlier carry-forward** (subsamples favored `contextualized`;
at full dev `anti_flattening` is the confirmed winner).

### Key finding

The conditioning advantage is **real but modest** (~+2.8 grouped on the winning
model). Combined with Stages 01–03: the big lever was the **model** (+17 from
2.5→3.1-flash-lite); the **strategy** adds a smaller, now-confirmed ~+3 — and
`anti_flattening` (preserve within-group minority mass) is the system to lock in.

**Run files:** `outputs/runs/2026-06-20-fulldev-confirm.{topline.csv, results.json, meta.json, log}`
