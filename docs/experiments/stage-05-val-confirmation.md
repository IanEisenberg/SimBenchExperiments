# Stage 05 — Validation-set confirmation (first val touch)

- **Status:** **DONE** (ran 2026-06-20) — selection **CONFIRMED**; advantage held and grew on val
- **Run name:** `2026-06-20-val-confirm`
- **Preregistered:** 2026-06-20
- **Owner:** Ian + Claude

> Decision rule fixed **before** the run. This is the **first and only** `val`
> touch for this selection. `test` stays locked.

## Why

Selection on dev (Stage 04) chose **`anti_flattening` @ gemini-3.1-flash-lite**
(beats `faithful` +2.81 grouped, clears noise). This confirms the choice on the
held-out **val** set — does the dev-measured advantage replicate out-of-sample, or
was it dev-overfitting? Confirming the *advantage* needs the baseline too, so we
query the winner **and** `faithful` (2 configs, one confirmation round).

## Configs

`gemini-3.1-flash-lite`, `temperature=0`, full `val` (n=3,767; grouped=1,903).

| system | role |
|---|---|
| `anti_flattening` | selected winner |
| `faithful` | baseline reference |

`test` (3,592) is **not** touched. After this query, no further val tuning.

## Decision rule (preregistered — fixed before looking)

The selection is **CONFIRMED** iff **both** hold on val grouped:

1. **Level holds:** `anti_flattening` val grouped is consistent with its dev value
   (~48.8) — no large drop that would signal dev-overfitting (within ~1 CI width).
2. **Advantage holds:** `anti_flattening` still beats `faithful` on val grouped
   (gap > 0; ideally clears the val noise floor).

- If confirmed → **`anti_flattening` @ gemini-3.1-flash-lite is locked** as the
  system for any future one-time `test` number.
- If the advantage vanishes/reverses on val → the dev edge did not generalize;
  fall back to `faithful` and report conditioning gave no held-out gain.

---

## Results

- **n:** grouped=1,903, pop=1,864 · **Cost:** $0.91 · **7,526 calls** (8 cache hits — val was never queried before) · **26.5 min**

| system | val grouped | 95% CI | dev grouped | pop | cf_alignment |
|---|---|---|---|---|---|
| **anti_flattening** | **53.02** | [51.50, 54.65] | 48.81 | 33.64 | 0.419 |
| faithful (baseline) | 46.98 | [45.03, 48.84] | 46.01 | 26.61 | 0.397 |

- **Level holds?** ✅ anti_flattening val 53.02 ≥ dev 48.81 — no overfit drop (in
  fact higher on val).
- **Advantage holds?** ✅ **+6.03 on val, clears the noise floor (1.91)** — and
  exceeds the dev gap (+2.81). cf_alignment 0.419 ≥ faithful 0.397.
- **Selection CONFIRMED.**

### Outcome

**`anti_flattening` @ `gemini-3.1-flash-lite` is locked** as the system. The
within-group-diversity conditioning advantage over the SimBench first-person
baseline is real and **generalized out-of-sample** (held and grew on val). Both
systems scored a touch higher on val than dev (different family split), but the
*gap* is robust.

No further val tuning (this was the single confirmation round, 2 configs).
`test` (3,592, incl. the three required questions) remains **locked and untouched**
— reserved for a future one-time headline number, not run in this session.

**Run files:** `outputs/runs/2026-06-20-val-confirm.{topline.csv, results.json, meta.json, log}`
