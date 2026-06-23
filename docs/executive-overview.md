# Scrye — Executive Overview

*Predicting the distribution of human survey responses, and doing it as a
disciplined, AI-driven scientific search.*

This document is the plain-language tour of the project: **what problem we are
solving**, **how we are solving it** — including the unusual *meta*-method of
letting an AI run a guarded scientific search — **which specific levers we
pulled** and what each one bought us, and **which promising directions we tried and
rejected or deliberately deferred**. For the interactive version (diagrams,
module map, live experiment replay) see [`overview.html`](overview.html); for the
full preregistered record see [`experiments/`](experiments/README.md).

---

## 1. The problem space

### What we are predicting

Most "LLMs as simulated humans" work asks a model to give *an answer* — to play a
persona and pick an option. **Scrye predicts a distribution.** Given a survey
question, a set of options, and a demographic group, we predict *what fraction of
that group chooses each option*, and compare it against the real empirical
distribution of human responses.

The benchmark is **[SimBench](https://huggingface.co/datasets/pitehu/SimBench)**
(Hu et al., arXiv 2510.17516) — 20 survey datasets, 130+ countries, millions of
question–group pairs drawn from instruments like the European Social Survey,
Afrobarometer, LatinoBarómetro, OpinionQA and ISSP. The unit of truth is a
*histogram over options*, not a single label.

### How success is measured

Every prediction is scored by how far its distribution **Q** sits from the human
truth **P**, normalized against the **uniform** guess **U**:

```
S = 100 × ( 1 − TVD(P, Q) / TVD(P, U) )
```

where TVD is total-variation distance (½·Σ|pᵢ − qᵢ|). The scale is anchored, not
abstract:

- **S = 0** → no better than predicting "everything is equally likely" (the naive
  uniform baseline — *the bar to beat*).
- **S = 100** → perfect match to the human distribution.
- **S < 0** → actively *worse* than uniform (confidently wrong).

The published state of the art on SimBench is only **~41/100** — even the best
LLM closes well under half the gap to ground truth, and ten of the models tested
score *below* uniform. This is a genuinely unsolved problem, which is what makes
it a good target.

### Why it is hard

Three structural difficulties shape everything we do:

1. **Demographic conditioning backfires.** Telling a model "you are a 35-year-old
   woman from Finland" *degrades* group-level accuracy in SimBench — by 1 to 10
   points across every category tested. Models collapse a diverse group into a
   single stereotyped voice and lose the within-group spread that the truth
   contains. So naïve persona engineering is a trap, not a lever.
2. **The value–action gap.** Stated attitudes ("I care about sustainability") are
   far more predictable than behavior under real stakes (who actually pays the
   green premium). Across the literature, attitudes predict at ~0.83 while
   behavior lands nearer ~0.66. The hard, high-value questions are exactly the
   ones where the baseline is weakest.
3. **Spread is entangled with location.** Models are systematically *miscalibrated*
   on these tasks — overconfident on consensus questions, too diffuse on contested
   ones — and, as we found the hard way, you cannot fix the spread without first
   getting the *leading answer* right.

---

## 2. The approach

### 2.1 The prediction pipeline

The core runtime is a small, swappable chain:

```
Record  →  Predictor  →  Calibrator  →  scored distribution
```

- a **Record** is one typed survey item (question, options, group, empirical
  truth);
- a **Predictor** turns the record into a predicted distribution — e.g. a
  zero-shot conditioning prompt, a routing predictor, or a simulation ensemble;
- a **Calibrator** optionally post-processes the distribution (temperature,
  abstention, …);
- the result is scored with the SimBench metric, with **bootstrap confidence
  intervals** on every headline number so we never compare bare point estimates.

Every LLM call routes through OpenRouter and is cached on disk by a hash of
`(model, messages, sampling params)`. Re-running any experiment against the same
calls is therefore **free and deterministic** — which is what makes the search
loop below affordable.

### 2.2 The meta-approach: AI-driven scientific discovery

The more distinctive half of the project is *how* we improve the predictor. Rather
than hand-tuning prompts and eyeballing results, Scrye runs a **preregistered,
leakage-controlled search** in which an AI proposes interventions and statistical
guardrails decide what counts as a real gain. The goal is not just a good score —
it is a *defensible* one, where no result is an artifact of overfitting or of
peeking at the answer.

Building that apparatus was itself a deliberate goal. A meaningful share of the
project's effort went not into any single prediction trick but into a **reusable,
scalable meta-science tool for crafting experiments** — the preregistration
workflow, the lever registry, the Ladder gate, and the ledger/manifest record
described below — treated as a first-class deliverable in its own right (see §5).

Four mechanisms make this honest:

**(a) Preregistration — plan before you look.** Work proceeds in **stages**. Each
stage is a markdown file written *before* the run: the hypothesis, the exact
configs to try, the data and budget, and — crucially — the **decision rule** (what
will count as a win), all fixed before any results are seen and signed off by a
human. Results are appended afterward to the same file. This is the layer where
we "plan and discuss before running, and record after," and it keeps the science
separate from the engineering.

**(b) Leakage discipline — never tune on the answer.** The dataset is split
**leave-family-out**: all variants of one question (grouped by `(dataset,
template)`) move together into one of three buckets — **dev** (free iteration),
**val** (gated selection), **test** (sealed, scored once). A random row split
would scatter siblings across buckets and let the model be "evaluated" on
questions it was effectively tuned on; family-based splitting forbids that. The
three required headline questions are *pinned to test* before the run, so the
number we ultimately report is never informed by a design decision. Assignment is
deterministic (seeded hashing), so splits are stable and reproducible.

**(c) The Ladder gate — protect the validation set from yourself.** The val set is
the only thing standing between dev and test, and adaptive search will overfit it
if allowed to. So every promotion to val passes through a **Ladder gate** (Blum &
Hardt 2015, *adaptive data analysis*): a new candidate is accepted only if it
beats the running best by more than **η**, a noise threshold derived by
**bootstrapping the val score's own sampling wobble** before the search begins. The
theorem guarantees the held-out estimate stays valid no matter how many candidates
are tried — it forces the loop to find *real* gains, not lucky ones. η is fixed up
front (no p-hacking), and the count of val queries is budgeted and tracked
globally.

**(d) Reproducibility apparatus — every run is auditable.** Three layers record
each run:
- a **content-addressed `PipelineSpec`** (SHA-256 of model + predictor +
  calibrators) so identical pipelines hit cache and never re-spend budget;
- an append-only **ledger DAG** (`ExperimentNode`s) capturing every proposal — dev
  score, val score, η applied, accept/reject verdict, lever, rationale, cost;
- a frozen **manifest** sidecar (split seed, fractions, allowed levers, η,
  budgets, model, dataset fingerprint, timestamp).

Together with the LLM cache, these make a run **byte-for-byte reproducible and
fully audited**: the same seed reproduces the same split, the same fingerprint the
same records, the same manifest the same calls.

**The loop, in one sentence:** an AI proposes a registered lever → it is scored
freely on **dev** → only dev-winners spend one **Ladder-gated** query against
**val** → accepted winners become the new incumbent → **test** is touched exactly
once, at the very end. Human judgment sets the hypotheses and decision rules; the
machinery enforces that the conclusions are earned.

---

## 3. The specific levers pursued

Improvement directions are pre-registered as **levers** — theory-motivated
interventions, each carrying its hypothesis and expected direction. Adding an
off-registry lever is a deliberate, human-approved act. Across seventeen
preregistered stages we worked through the levers below (an eighteenth explored a
different direction entirely — see §5). The honest summary: **most clever ideas lost; a few
simple ones won; and the biggest lever wasn't a method at all.**

| Lever | What it is | Verdict |
|---|---|---|
| **Model swap** (gemini-2.5 → 3.1-flash-lite) | Run the same prompt on a stronger foundation model | ✅ **Dominant.** +12–17 grouped points on an identical prompt; a 22×-pricier flagship was *not* better. |
| **Distributional framings** (`representative_sample`, `anti_flattening`, `contextualized`) | Replace first-person "be this person" with a third-person "what does this *group* look like," explicitly preserving minority mass | ✅ Real and out-of-sample robust. `anti_flattening` beat the faithful baseline +2.8 on dev, **+6.0 on val**. Reverses SimBench's headline failure. |
| **Monte-Carlo individuals** | Simulate 20 synthetic people, average their answers | ❌ Catastrophic (−15.7). Over-disperses; the single distributional call wins. |
| **Diversity / superforecaster reasoning** (CoT, outside-view, entropy-first, premortem) | Make the model reason before answering | ❌ Consistently −2 to −3. Reasoning-first is a *tax* here; the direct distributional ask beats elicited reasoning. |
| **Post-hoc calibration** (temperature, entropy-tempering, Dirichlet) | Sharpen/flatten the output distribution after the fact | ❌ All lose to identity. The model's own entropy carries no usable signal; spread can't be fixed from spread alone. |
| **Feature-predicted entropy** | Predict each question's true spread from features (R²=0.58) and temper to it | ❌ Captures ~0% of the available headroom — the oracle gain is **gated behind getting the mode right** (+16.8 where the top option is correct, −2.0 where it isn't). |
| **Calibrated-commitment prompt** | A mode-first prompt: nail the leading answer / give a clear plurality when one option leads | ✅ +1.8 grouped / +2.1 pop over `anti_flattening`. Improves *location*, not entropy. Became the new prompt. |
| **Abstention / uniform fallback** | Predict uniform on the few datasets the model reliably fails (e.g. MoralMachine, Choices13k) | ✅ +3.1 pop. A committal prompt is *more* wrong on unknowable tasks; abstaining there recovers the loss. The pop-side MVP. |
| **Discrete-vote simulation** | A panel of dispositional agents each cast one vote; the tally is the distribution | ⚠️ A real, artifact-free win on risky-choice gambles (+12.6 vs uniform), but catastrophic where the model's mode is wrong (MoralMachine −158) — and the win **did not transfer to val**. Shelved. |
| **Task-context prompting** | Show sibling items from the same survey so the model sees the respondent's real context | ✅ Grouped win (+2.4 pooled), driven by opinion surveys (ESS +12.9, OpinionQA +6.0); hurts behavioral tasks. |
| **Task-kind routing** | An upfront LLM classifier dispatches each item to the right intervention (surveys→context, moral→abstain, etc.) with *zero per-dataset parameters* | ✅ on dev (+8.75) and val (+1.1 pooled, gate-confirmed) — but **tied** the simpler `cc+abstain` on sealed test. |

A few cross-cutting findings fall out of this table:

- **Reasoning-first prompting reliably hurts** this model/task — the opposite of
  the usual intuition.
- **Spread and location are entangled.** The big calibration prize (+9.5 grouped
  from perfect entropy) is *unreachable* unless the leading option is already
  correct, which closed off the entire post-hoc-calibration family and redirected
  effort toward getting the mode right.
- **Commitment and abstention are complements:** a prompt that commits harder helps
  where the model knows the answer and hurts where it doesn't, so pairing it with
  a "know when to shrug" fallback is what made the combination robust.

---

## 4. What we found — the headline result

The sealed **test** set was queried exactly once, at the end, with the full
lineage measured for an honest accounting:

| System | Model | Overall S |
|---|---|---|
| faithful (paper baseline) | gemini-2.5-flash-lite | 19.27 |
| faithful | Qwen2.5-72B-Instruct | 25.40 |
| faithful | **gemini-3.1-flash-lite** | **35.21** |
| anti_flattening | gemini-3.1 | 39.68 |
| **calibrated_commitment + abstain** | gemini-3.1 | **40.93** |
| task-kind router (val-confirmed) | gemini-3.1 | 40.73 |

Three honest headlines:

1. **The model is the dominant lever.** Swapping gemini-2.5 → 3.1-flash-lite alone
   moved the faithful baseline **+15.9 overall / +25.4 grouped** — the required
   opinion questions jumped from −4.5 to +51.5 on a model swap with *no method
   change*. Our entire method stack then adds **+5.5 on top** (CI-clean over
   faithful@3.1) — real, but roughly a third of the model's contribution.
2. **The simpler method wins ties.** On val, the task-kind router edged
   `calibrated_commitment + abstain` by +1.1; on sealed test that edge **vanished**
   — the two are statistically tied. Per leakage discipline we don't re-select on
   test, so the router stays the system *of record*, but the **reporting leads with
   the simpler `cc+abstain`**: one prompt plus a uniform fallback, matching the
   router's accuracy without a per-item classifier. (This val→test wash-out is the
   textbook reason a sealed test set exists.)
3. **The pipeline reproduces the benchmark.** Running our faithful predictor on
   Qwen2.5-72B (a model the paper evaluated) scores **26.83 [24.4, 29.4]**, which
   brackets SimBench's reported **27.61** — validating that the harness, prompts,
   parsing and normalization are faithful before we trust any of our own numbers.

**Bottom line:** Scrye beats SimBench's uniform baseline decisively (final
**~40.9/100**, far above 0 and in the range of the paper's best published systems
— though our number is on a newer model, gemini-3.1-flash-lite, that postdates the
paper's evaluation, so it isn't a same-model comparison). More importantly, the
result is *defensible*: every gain is preregistered, gated against overfitting,
and reproducible, with the model-vs-method contributions honestly separated.

---

## 5. Directions considered but not adopted

A faithful overview is as clear about the roads not taken as the ones that paid
off. Three are worth calling out.

**Pursued in depth, then rejected — a grounded persona "electorate" (Nemotron).**
The most thorough negative result was an attempt to predict a group's answers
*bottom-up* — by simulating individual people and tallying them — instead of asking
the model to report the distribution directly. Earlier synthetic-persona attempts
had failed because the personas were generic and over-dispersed; this stage replaced
them with **Nemotron-Personas-USA**: a million *census-grounded* synthetic US adults,
each with a rich life narrative, representative of the real population by
construction. Over five rounds we cleanly separated and fixed two failure modes —
a discrete-vote panel *over-concentrated* (every persona argmaxed onto the same
option → score 12.7 on OpinionQA), but moving the aggregation to the population
level (each persona gives a *distribution*, then average) **solved the spread
problem** (predicted entropy 0.644 vs. truth 0.692 — the first persona simulator in
the project to get the *shape* right), and a worldview-enrichment step supplied the
missing ideology axis (→ 51.8). Yet even fully developed, the grounded electorate
stayed **~11 points below the direct `calibrated_commitment` ask** (63.2 on the same
slice), and blending the two added nothing beyond noise (+0.5). The binding
constraint turned out to be *location* — which option carries the mass — and a direct
distributional ask simply places it better than a bottom-up average. This was the
project's **fourth confirmation** that elaborate bottom-up / elicited methods lose to
the direct distributional ask, and the most informative one: it proved the residual
error is location, not spread. The persona line of inquiry was **closed**; the
champion system is unchanged. (It did leave a reusable, validated grounded-simulation
toolkit for any future *location*-targeted work.)

**Considered and specified, but deferred for time — agentic web retrieval.** The
benchmark brief explicitly permits tools, including internet access, to gather
information before answering. We designed a serious contender around this: a
**`WebAgentPredictor`** — a bounded tool-calling agent over guarded, disk-cached web
search/fetch, with identity-stripping and a domain blocklist so the model can't
simply look up the survey's published results, plus a closed-book twin as a control
to isolate the value of web access from the scaffolding. It was fully preregistered
(hypothesis, leakage guards, decision rule) but **held, not run** — deferred for
time, not for lack of promise. It remains the most natural next lever to pick up.

**Chosen deliberately — investing in the meta-science tool itself.** Finally, one of
the project's primary "directions" was not a prediction trick at all but the
**experiment-crafting machinery** of §2.2. Building a reusable, scalable tool for
proposing, gating, recording and reproducing experiments was a conscious trade:
effort spent on the apparatus is effort not spent squeezing a few more points out of
this one benchmark — but it is what makes *every* future experiment fast (cached),
honest (gated), and reproducible, and it is the part most likely to outlast this
specific task.

---

## 6. Beyond the benchmark (Part II)

A companion design document,
[`part-ii-feedback-architecture.md`](part-ii-feedback-architecture.md), sketches
how this methodology extends from static, population-level survey distributions to
a **commercial system predicting individual behavioral outcomes** from transaction
logs (purchases, churn, engagement) — the high-value end of the value–action gap.
It carries forward the same discipline (stamped predictions before outcomes are
known, gated model updates, explicit abstention when out-of-distribution) into a
multi-tenant feedback loop with compounding memory. It is a forward-looking design,
not part of the Part I results above.

---

### Where to go next

| If you want… | Go to |
|---|---|
| The interactive tour (diagrams, module map, replay) | [`overview.html`](overview.html) |
| The preregistered experiment record, stage by stage | [`experiments/README.md`](experiments/README.md) |
| The code architecture and how to run things | [`../README.md`](../README.md) and [`../CLAUDE.md`](../CLAUDE.md) |
| The literature this builds on | [`literature_review/`](literature_review/) |
| The commercial feedback-loop design | [`part-ii-feedback-architecture.md`](part-ii-feedback-architecture.md) |
