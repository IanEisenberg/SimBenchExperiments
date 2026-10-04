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
| [12](stage-12-val-confirmation.md) | Val confirmation (cc+abstain) | DONE | **CONFIRMED** (2nd val touch): `calibrated_commitment` + `AbstainCalibrator` beats champion on val — grouped +0.66 (within noise), pop +3.26, pooled +1.95. New locked system = **calibrated_commitment @ gemini-3.1-flash-lite + abstain**; abstention is the MVP. test untouched |
| [13](stage-13-voting-simulation.md) | Discrete-vote simulation (weak pop tasks) | DONE | Replace uniform-abstain with *principled* simulation, no questionnaire artifacts. **`VotingEnsemblePredictor`** (K dispositional agents each cast ONE vote; tally = dist). Global H1 REFUTED (1/3 datasets). But a real, artifact-free win on **Choices13k** (+12.6 vs uniform, CI excludes 0) — heterogeneous risk panel matches gamble choices; catastrophic where mode is wrong (MoralMachine −158). Selective routing nets +4.4 dev / ≈+0.8 pop. System unchanged by default; optional selective promotion needs val. Mode/knowledge remains the binding constraint |

_Statuses: PLANNED → RUNNING → DONE. Update this row when a stage closes._
| [14](stage-14-task-context.md) | Task-context prompting | DONE | SimBench prompts atomized (1 Q + 1-line group); humans had instrument/session context. New **`TaskContextStrategy`** (brief / **include-other-items** / both) on top of cc. Weak-dataset hypothesis REFUTED (context doesn't rescue behavioral tasks; hurts Choices13k). But **`task_context_items` is a confirmed grouped win**: +2.37 pooled (CI [+0.77,+3.96]), driven by ESS +12.9 / OpinionQA +6.0; hurts the Global-South barometers. **Selective routing** (items where dev-validated) → **+3.71 dev** (CI [+2.71,+4.75]) — biggest clean grouped lever found. Val-gated confirmation recommended; pop side unchanged |
| [15](stage-15-task-kind-routing.md) | Task-kind routing (unified rule) | DONE | Replace per-dataset routing with an **upfront LLM task classifier** → fixed kind→intervention map (**`RoutingPredictor`** + `taskkind.py`): opinion_survey/knowledge→task_context, risky_choice→voting, moral_dilemma→abstain, personality/other→cc. **WIN**: routed+abstain-floor beats base cc by paired **+8.75** (95% CI [+5.94,+11.67], representative dev), driven by pop; beats the per-dataset oracle using **zero per-dataset params** for the positive routes → generalization by construction. New unified candidate; val-gated confirmation recommended |
| [16](stage-16-val-router.md) | Val confirmation: task-kind router | DONE | **CONFIRMED** (3rd val touch): router beats Stage-12 champion — grouped **+1.89** [+1.06,+2.73], pooled **+1.10** [+0.18,+1.94]. Per-kind: **task_context transferred** (surveys +1.57, knowledge +5.53); **`risky_choice→voting` did NOT** (val Choices13k −3.0 vs abstain +11.1) → voting dropped, risky_choice→abstain (clean both-split win: grouped +1.89, pop +1.35). Mechanism: gain is **location/mode** (+1.7pp mode-match) via task-context; **entropy untouched** (still ~0.06 too diffuse). New val-confirmed system; test locked |
| [17](stage-17-final-test.md) | One-shot TEST + full lineage + paper reproduction | DONE | **Final headline.** Sealed test, full lineage @ gemini-3.1: faithful **35.21 → 40.73** (+5.52, all CIs exclude 0). **Two honest findings:** (1) the **model is the dominant lever** (faithful 2.5→3.1 = +15.9 overall / +25.4 grouped) — the method stack adds +5.5 on top; (2) on test the **router ties `cc+abstain`** (the +1.1 val edge did NOT transfer) → reporting leads with the simpler **`cc+abstain`**. **Pipeline reproduces the paper**: faithful @ Qwen2.5-72B S=26.83 [24.4,29.4] brackets 27.61 ✓. Notebook 03 + `best_analysis_run` (dev-only) updated |
| [18](stage-18-nemotron-personas.md) | Grounded persona electorate (Nemotron-Personas-USA) | DONE | Bottom-up sim from a **census-grounded synthetic electorate** (1M NVIDIA personas) on OpinionQA pop dev, developed across 5 rounds. **Discrete voting over-concentrates** (12.7) → **distribution+population-averaging recovers the spread** (46.3, entropy near-calibrated) → **diagnosis**: personas barely disagree (78% mode-agreement) because Nemotron has **no ideology axis** (the model differentiates 0.54 on explicit archetypes but 0.21 on bare personas) → **worldview enrichment + ideology-calibration** to OpinionQA's real POLIDEOLOGY marginal lifts standalone **+5.5 → 51.8** and is decorrelated from the champion (r=0.56). **But still −11 vs `calibrated_commitment` (63.2); ensemble only +0.47 (noise).** Location-limited; bottom-up < direct ask (4th confirmation). New: `nemotron.PersonaBank`, `worldview`, `Grounded{Voting,Averaging}/Enriched/Ensemble` predictors. **Persona line closed; system unchanged** |
| [19](stage-19-equivalent-model-paper.md) | Method portability to frontier Claude (equivalent-model paper test) | DONE (concluded on Sonnet-4.5) | Does the gemini-tuned `cc+abstain` *beat the paper* when run on a model in the paper's top tier? Claude-3.7-Sonnet (paper's best, 40.80) is **retired on OpenRouter** → used the Sonnet-4 line as the equivalent successor. **Both hypotheses FAIL on Sonnet-4.5:** faithful→cc+abstain delta **+0.93 [−0.73,+2.75]** (n.s.) — the cc prompt is even **net-negative** alone (33.70 < faithful 34.68); cc+abstain ≈ **38 paper-equiv < 40.80**. **The +5.5 "method" lift is substantially gemini-specific and does not port.** Sonnet-4.6 unmeasured for the method (OpenRouter key cap; concluded by choice). A preregistered cross-model test that **refuted** the hoped-for "exceeds the paper" claim — honest headline stays *cc+abstain @ gemini-3.1 ties 40.80 on a far cheaper model*. New: Sonnet-4.x registered in `config.py`; `scripts/run_stage19_claude.py` |
