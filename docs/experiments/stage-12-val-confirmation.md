# Stage 12 — Val confirmation: calibrated_commitment + abstention (2nd val touch)

- **Status:** **DONE** (ran 2026-06-21) — **CONFIRMED.** Challenger `cc+abstain`
  passes all three rules (grouped +0.66, pop +3.26, pooled +1.95). New system:
  **`calibrated_commitment` @ `gemini-3.1-flash-lite` + `AbstainCalibrator`**.
  (Grouped gain within noise → confirm not "strong"; abstention carries the win.)
- **Run name:** `2026-06-21-val-confirm-cc`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> This is the **second and (for now) final** `val` touch — the first was Stage 05.
> `test` stays locked. The challenger's only fitted component (the abstain-set) is
> learned on **dev** and merely *applied* to val.

## Why

Dev evidence (Stages 10–11) favors a new stack over the val-confirmed champion:

- **Champion (Stage 05):** `anti_flattening` @ `gemini-3.1-flash-lite` — val grouped
  53.0, the only config that has passed val.
- **Challenger:** `calibrated_commitment` @ `gemini-3.1-flash-lite` **+ `AbstainCalibrator`**.
  On dev it beats the champion on both splits (grouped +1.8, pop +2.1) and abstention
  adds +3.1 pop — but the grouped gain was *within* dev noise. This query tests
  whether that multi-axis dev advantage **replicates out-of-sample**.

## Configs (val)

`gemini-3.1-flash-lite`, full `val` (grouped + pop).

| system | predictor | calibrator | notes |
|---|---|---|---|
| `anti_flattening` (champion) | anti_flattening | identity | **cached** from Stage 05 |
| `calibrated_commitment` | calibrated_commitment | identity | prompt effect alone |
| **`cc+abstain` (challenger)** | calibrated_commitment | `AbstainCalibrator` | abstain-set **fit on full dev** |

The `AbstainCalibrator` is fit on full-dev `calibrated_commitment` predictions
(predict uniform on datasets where the model's mean TVD exceeds uniform's), then
applied unchanged to val. `cc+abstain` shares `cc`'s LLM calls (abstention is a
deterministic post-hoc transform), so this is effectively one new prediction set.

## Decision rule (preregistered — fixed before looking)

Primary metric: **val grouped** mean SimBench (95% bootstrap CI). Secondary: pop, pooled.

**CONFIRM** the challenger as the new system (and update the notebook winner) iff **all**:
1. **Grouped advantage replicates:** `cc+abstain` val grouped ≥ `anti_flattening` val
   grouped (gap ≥ 0; the dev gap was ~+1.8).
2. **Overall improves:** `cc+abstain` val **pooled** > `anti_flattening` val pooled.
3. **No pop regression:** `cc+abstain` val pop ≥ `anti_flattening` val pop.

- **STRONG confirm** if the grouped gap clears the val noise floor (half-width of the
  larger CI).
- **REJECT** (champion stays) if the challenger regresses on grouped beyond noise, or
  fails the pooled/pop guards — i.e. the dev gain was dev-overfitting.
- Report `cc` vs `cc+abstain` to separate the prompt's effect from abstention's.
- After this query: **no further val tuning**; `test` remains untouched.

---

## Results

Ran 2026-06-21. Full `val` (n=3767; grouped 1903, pop 1864). `gemini-3.1-flash-lite`.
Abstain-set fit on full dev = `{Choices13k, MoralMachine, OSPsychMACH}` (same as
Stage 11). Run files: `outputs/runs/2026-06-21-val-confirm-cc.*`.

| system | grouped | 95% CI | g half-width | pop | pooled |
|---|---|---|---|---|---|
| `anti_flattening` (champion) | 53.02 | [51.4, 54.7] | 1.62 | 33.64 | 43.43 |
| `calibrated_commitment` (prompt only) | 53.68 | [52.1, 55.2] | 1.53 | 31.70 | 42.81 |
| **`cc+abstain` (challenger)** | **53.68** | [52.1, 55.2] | 1.53 | **36.90** | **45.38** |

**Challenger vs champion:** grouped **+0.66** (floor 1.62), pop **+3.26**, pooled **+1.95**.

### Verdict — CONFIRMED

All three preregistered rules hold: grouped ≥ champion ✓, pooled > champion ✓, pop ≥
champion ✓. Not "strong" (the grouped gap is within the val noise floor), so the
confirmation rests on pop + pooled, not grouped.

**New confirmed system: `calibrated_commitment` @ `gemini-3.1-flash-lite` +
`AbstainCalibrator`** (replaces `anti_flattening` as the locked best).

### What the val data shows (honest read)

- **The prompt holds grouped out-of-sample but doesn't move it.** `calibrated_commitment`
  grouped 53.68 vs `anti_flattening` 53.02 (+0.66) — the dev gap (+1.8) shrank and is
  within noise, but it did **not** reverse. The mode-first prompt is at worst neutral
  on grouped out-of-sample.
- **The prompt alone *hurts* pop** (31.70 vs 33.64): committing confidently on the
  unknowable pop tasks backfires out-of-sample too — exactly the Stage 11 mechanism.
- **Abstention is the MVP.** It lifts the challenger's pop 31.70 → 36.90 (+5.2) and is
  what turns the stack into a net win (pooled +1.95). The Stage 11 dev result (+3.1
  pop) **replicated and grew on val** (+5.2 here over cc-alone, +3.26 over champion).

### Carry-forward

- Promote `calibrated_commitment` + `AbstainCalibrator` to the system; point the
  notebook winner at it. The `AbstainCalibrator` ships with its dev-fit abstain-set
  `{Choices13k, MoralMachine, OSPsychMACH}`.
- **No further val tuning.** `test` remains **untouched** — available for a future
  one-time headline number on this confirmed system.
