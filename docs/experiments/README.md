# Experiment log (preregistration)

This is the **human** layer of the experiment loop — the place we plan, discuss,
and record experiments *before and after* running them. It sits on top of the
machine artifacts (`outputs/ledger/<run>.jsonl`, `<run>.manifest.json`,
`data/cache/`), which are the receipts; this folder is the plan and the story.

One markdown file per **stage**. A stage is a batch of experiments sharing one
hypothesis and one preregistered decision rule. Each file is written and
approved *before* the run (so the decision rule is fixed before we see the
data), then results are appended *after*.

## The loop

1. **Preregister** — draft `stage-NN-<name>.md`: hypothesis, configs to run,
   data + budget, and the decision rule. Get sign-off. *(no val/test contact yet)*
2. **Run + iterate** — execute on `dev`. Every LLM call caches to `data/cache/`,
   so re-runs are free and deterministic. Iterate a few times as needed.
3. **Record** — append the topline results, the run-file pointers, what we
   learned, and what it implies for the next stage.
4. **Next round** — preregister `stage-NN+1`, informed by the result.

See the root `CLAUDE.md` → "Experiment workflow (preregistration)" for the rules
this folder enforces.

## Stages

| Stage | Name | Status | Headline result |
|---|---|---|---|
| [01](stage-01-baseline.md) | Strategy baseline | DONE | Better conditioning beats SimBench first-person baseline on grouped (+8 pts); anti_flattening most robust |
| [02](stage-02-model-portability.md) | Model portability | DONE | gemini-3.1-flash-lite is a big step up (+12–17 grouped pts) and beats the 22×-pricier 3.5-flash; better model narrows the conditioning gap |
| [03](stage-03-persona-mechanisms.md) | Persona mechanisms | DONE | Negative result: Monte-Carlo individuals (−15.7) and diversity-CoT (−3.1) both lose to the simple single-call framing; contextualized/anti_flattening stays best |
| [04](stage-04-fulldev-confirmation.md) | Full-dev confirmation | DONE | At full dev (n=2566) anti_flattening beats faithful +2.81 (clears noise); contextualized within noise. Winner: anti_flattening @ 3.1-flash-lite |
| [05](stage-05-val-confirmation.md) | Val confirmation (first val touch) | DONE | CONFIRMED on held-out val: anti_flattening 53.0 vs faithful 47.0 (+6.0, clears noise, > dev gap). anti_flattening @ 3.1-flash-lite locked; test untouched |
| [06](stage-06-calibration.md) | Calibration sweep | DONE | Negative: all 3 calibrators (TempScaling, EntropyTemp, Dirichlet) lose to identity; anti_flattening needs no post-hoc correction |
| [07](stage-07-superforecaster.md) | Superforecaster prompting | DONE | Negative: outside-view / entropy-first / full superforecaster all lose to the no-CoT incumbent and cluster with generic CoT (~−2–3 grouped). 2nd negative on "make this model reason" — direct distributional ask beats elicited reasoning |
| [08](stage-08-entropy-recalibration.md) | Entropy de-compression | DONE | Negative: +9.5 grouped oracle headroom (predictions are entropy-compressed, slope 0.48) is **not** recoverable from the prediction's own entropy; global temps hurt. Needs an external contestedness signal |
| [09](stage-09-feature-entropy.md) | Feature-predicted entropy | DONE | Negative with mechanism: features predict truth-entropy (R²=0.58) but tempering still can't beat identity — the oracle's gain is **gated behind mode correctness** (+16.8 where top option right, −2.0 on the 39% where wrong). Spread entangled with location; post-hoc calibration closed |
| [10](stage-10-calibrated-commitment.md) | Calibrated-commitment prompt | DONE | 4 dev rounds → **`calibrated_commitment`**: a mode-first prompt improving both grouped (+1.8) and pop (+2.1) over anti_flattening via better *location* (not entropy commitment). Within dev noise → registered + val-confirmation candidate; system unchanged. One prompt beat regime-branching |
| [11](stage-11-abstention.md) | Abstention / uniform-fallback | DONE | Per-task: a few high-entropy tasks (OSPsychMACH, MoralMachine, Choices13k) score *below uniform*, and the committal prompt is *more* wrong there. **`AbstainCalibrator`** (predict uniform on datasets the model fails on, learned on dev) lifts pop +3.1 → 39.1; per-item oracle (+15) needs a confidence signal |

_Statuses: PLANNED → RUNNING → DONE. Update this row when a stage closes._
