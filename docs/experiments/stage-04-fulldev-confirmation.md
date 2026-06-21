# Stage 04 — Full-dev confirmation of the top systems

- **Status:** PLANNED → _RUNNING_ → _DONE_
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

## Results _(appended after the run)_

- **n grouped / pop:** _pending_ · **Cost:** _pending_ · **Calls:** _pending_

| system | grouped | 95% CI | pop | pooled | cf_alignment |
|---|---|---|---|---|---|
| _pending_ | | | | | |

**Noise floor (grouped):** _pending_
**Does contextualized / anti_flattening beat faithful beyond noise?** _pending_
**Carry-forward to val/test:** _pending_

**Run files:** _pending_
