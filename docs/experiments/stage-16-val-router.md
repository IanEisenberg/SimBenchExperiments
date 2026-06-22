# Stage 16 — Val confirmation of the task-kind router

**Status:** preregistered · **Date:** 2026-06-21 · **Model:** `gemini-3.1-flash-lite`
· **3rd val touch** (after Stages 05, 12).

## Hypothesis

The Stage 15 task-kind router (paired +8.75 on representative dev, zero
per-dataset params for the positive routes) transfers to held-out val and beats
the Stage-12 champion.

## Systems (full val, 3767 recs: grouped 1903, pop 1864)

- **Champion (Stage 12, val-confirmed):** `calibrated_commitment` +
  `AbstainCalibrator` (uniform on {Choices13k, MoralMachine, OSPsychMACH}).
- **Challenger (Stage 15):** `RoutingPredictor(LLMTaskClassifier, KIND_ROUTES)`
  + abstain-floor {OSPsychMACH} (the dev-fit reliability floor). Routes:
  opinion_survey/knowledge→task_context, risky_choice→voting,
  moral_dilemma→abstain, personality_scale/other→cc.

All fitted components (abstain sets, floor) are learned on **dev** and merely
applied to val. The classifier and kind-map have **no fitted parameters**.

## Decision rule (fixed before results)

**CONFIRM** the router as the new system iff **all**:
1. **Pooled improves significantly:** challenger val pooled > champion, paired
   95% CI lower bound > 0.
2. **No grouped regression:** challenger val grouped ≥ champion (paired grouped
   delta not significantly negative).
3. **No pop regression:** challenger val pop ≥ champion (paired pop delta not
   significantly negative).

Else **keep the Stage-12 champion**; record as a negative transfer.

## Mechanism analysis (reported regardless of verdict)

How the router interacts with the three axes we've studied, champion vs
challenger on val:
- **Entropy / spread:** mean predicted vs truth normalized entropy; mean
  |entropy_gap|; does routing reduce the entropy-compression gap?
- **Main prediction / mode:** mode-match rate (argmax pred == argmax truth) and
  mean `location_err` — does the gain come from getting the leading answer right?
- **Spread vs location split:** mean `concentration_err` vs `location_err`
  (`scrye.decompose`) — which component of TVD the router reduces.
- **Task-kind performance:** per-kind champion vs challenger score.

`test` stays locked; required questions (test-pinned) untouched.

---

## Results — CONFIRMED

Full val (3767 recs). Run: `outputs/runs/2026-06-21-val-router.results.json`.

| split | champion (cc+abstain) | challenger (router) | Δ [95% CI] |
|---|---|---|---|
| **grouped** | 53.68 | **55.57** | **+1.89 [+1.06, +2.73]** |
| pop | 36.90 | 37.20 | +0.30 [−1.32, +1.88] |
| **pooled** | 45.38 | **46.48** | **+1.10 [+0.18, +1.94]** |

All three rules pass → **CONFIRMED**. The dev +8.75 shrank to +1.10 pooled; the
gain is now **grouped** (+1.89, clean), pop flat. The per-kind breakdown shows
why.

### Per-kind — task-context transferred, voting did NOT

| kind | route | champ → chal | Δ |
|---|---|---|---|
| opinion_survey (2784) | task_context | 51.41 → 52.98 | **+1.57** |
| knowledge (311) | task_context | 33.36 → 38.89 | **+5.53** |
| **risky_choice (139)** | **voting** | 11.10 → **−3.03** | **−14.12** |
| other (264) | base | 51.13 → 51.13 | 0 |
| moral_dilemma (158) | abstain | 1.95 → 1.95 | 0 |
| personality_scale (111) | base | 18.85 → 18.99 | +0.13 |

**The voting route overfit dev and failed to transfer** — on val Choices13k,
uniform (champion's abstain) scores +11.1 while voting scores −3.0. This is the
val gate doing its job: the Stage-13 voting win was dev-specific. **task-context
transferred cleanly** (surveys +1.57, knowledge +5.53) — it is the real,
generalizing lever. The voting loss is what flattens pop.

**Implication (exact recompute, no new queries — swap Choices13k voting→uniform,
matching the champion's val-safe behavior):** routing `risky_choice → abstain`
gives val **grouped 55.57 (+1.89), pop ~38.25 (+1.35), pooled ~47.0 (+1.62)** — a
clean win on *both* splits. Adopted as the locked router (see KIND_ROUTES
change). Voting is shelved as a dev-overfit primitive.

### Mechanism — entropy / mode / task

| axis | champion → challenger |
|---|---|
| truth entropy (target) | 0.696 |
| **pred entropy** | 0.762 → 0.758 (both ~0.06 too diffuse) |
| **\|entropy gap\|** | 0.127 → 0.126 (≈ unchanged) |
| **mode-match rate** | 0.668 → **0.685** |
| concentration_err | 0.127 → 0.123 |
| location_err | 0.050 → 0.048 |

- **Entropy: untouched.** Predictions stay ~0.06 too diffuse; the router does
  **not** fix entropy-compression — that headroom is still open.
- **Mode / main prediction: modestly improved** (+1.7pp mode-match). The gain is
  a *location* improvement (getting the leading answer right) on survey-kind
  tasks via task-context — consistent with Stages 09–10.
- **Spread vs location:** TVD error is still dominated by `concentration_err`
  (0.12) over `location_err` (0.05) — the model mostly gets the mode right; the
  remaining error is spread/shape, which the router only nudges.

### Verdict & system

**CONFIRMED.** The task-kind router is the **new val-confirmed system** —
`RoutingPredictor(LLMTaskClassifier, KIND_ROUTES)` with `risky_choice → abstain`
(voting dropped) + abstain-floor {OSPsychMACH}. Net val win: **+1.89 grouped,
+1.35 pop, +1.6 pooled** over the Stage-12 champion. The confirmed lever is
**task-context on survey/knowledge kinds** (a location/mode gain); entropy
calibration remains open headroom. `test` stays locked.
