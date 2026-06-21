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
| [04](stage-04-fulldev-confirmation.md) | Full-dev confirmation | RUNNING | — |

_Statuses: PLANNED → RUNNING → DONE. Update this row when a stage closes._
