# Stage 05 — Validation-set confirmation (first val touch)

- **Status:** PLANNED → _RUNNING_ → _DONE_
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

## Results _(appended after the run)_

- **n:** grouped=1,903, pop=1,864 · **Cost:** _pending_

| system | val grouped | 95% CI | dev grouped | pop | cf_alignment |
|---|---|---|---|---|---|
| _pending_ | | | | | |

**Level holds (anti_flattening val ≈ dev)?** _pending_
**Advantage holds (anti_flattening > faithful on val)?** _pending_
**Selection confirmed?** _pending_

**Run files:** _pending_
