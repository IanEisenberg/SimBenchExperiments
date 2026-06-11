# Scrye Mini-Project — Gameplan & Direction Map

**Date:** 2026-06-10 · **Status:** First brainstorm — options document, not a converged spec
**Recommended spine:** Approach A (calibrated simulator), with other directions preserved as live options, ablations, or framed extensions.

---

## 1. The assignment

Two parts, two weeks (clock running):

- **Part I — Simulation implementation.** Given a survey question, its answer choices, and an optional demographic segment, predict the empirical human response distribution from SimBench. Must be LLM-based, must beat the paper's naive baseline, must generalize to unseen questions/segments, must not leak ground truth. Required questions: trust-in-president (LatinoBarometro 2023), gay-rights (ESS 2016), internet-use (ESS 2016). Metrics: SimBench score S + a **self-designed counterfactual sensitivity score**.
- **Part II — Feedback mechanism architecture.** Systems design only (no code): a sports e-commerce simulator starts receiving revealed behavior (transactions) and stated responses (new survey waves); design the architecture that compounds this into improving predictions. Must address: compounding memory, the retrieval→fine-tuning spectrum, revealed-vs-stated signal handling, multi-tenancy/generalization, guardrails & update validation.
- Plus: infrastructure discussion, presentation for 60–90 min defense.

**Performance criteria:** creativity/novelty, technical depth, empirical validation, implementation quality, systems reasoning.

## 2. Constraints

- **Time:** 5–10 hours total, part-time alongside a full job.
- **Budget:** ~$100 API spend; cheap models (Gemini Flash, Together+Qwen) — consistent with the model-as-commodity thesis.
- **Posture:** *Rigor as the product*, but the score must meaningfully improve; everything defensible with good practices; limitations framed as a clear improvement roadmap, not confessions.

## 3. SimBench ground truth (verified)

- **Paper:** Hu et al., arXiv 2510.17516; dataset HF `pitehu/SimBench` (CC-BY-NC-SA-4.0).
- **Score:** S = 100·(1 − TVD(P,Q)/TVD(P,U)), U = uniform. **Naive baseline (uniform) = 0 by construction** — trivially beatable; the real bar is strong zero-shot (~35–41; best Claude-3.7-Sonnet **40.8**).
- **Splits:** Pop (7,167 cases, 20 source datasets) and Grouped (6,343 cases, 5 surveys; conditioning = country × one attribute, natural-language prompts). **No train/dev split** — holdout discipline is ours to construct. Ground-truth distributions are public → the leakage protocol is ours to own and document.
- **Findings we build on:**
  - Verbalized distributions ≫ first-token logprobs for instruct models (settles elicitation).
  - CoT/reasoning budget does not help (slightly hurts) — saves money, prunes a direction.
  - **Diagnosed failure mode: mode-seeking.** Instruct models win on low-entropy consensus questions, lose on high-entropy diverse ones (r = −0.94 between IT benefit and response entropy).
  - Grouped conditioning degrades scores (ΔS ≈ −1.3 to −4.6), worst for religiosity/political-ideology.
- All three required questions confirmed present, with 67–91 grouped variants each — enough for real per-question counterfactual analysis.

## 4. Strategic read

1. **This is the case-study stage the interview-prep thesis anticipated.** Part II's scenario (revealed transactions vs. stated surveys) is the attitudes-vs-behavior evidence map restated as a design exercise. The prep-doc architecture (tiered confidence, calibration layer, reliability ceiling, honest no-flywheel multi-tenancy) can be deployed nearly verbatim.
2. **The self-designed counterfactual metric is an open invitation** to do construct-validity work — the signature differentiator.
3. **The hollow-demo trap is the central risk.** The winning shape: one or two well-chosen ideas with airtight ablations and an honest, roadmap-shaped limitations section — not a kitchen sink.
4. **A cheap model beating frontier zero-shot via method** would itself dramatize the model-as-commodity thesis.

## 5. The direction space (divergent map)

Families are composable; any spine draws from several.

### A. Elicitation — getting a distribution out of an LLM
| Option | Assessment |
|---|---|
| Verbalized distributions (model states probabilities) | **Default.** Paper shows it beats logprobs; works with any API. |
| First-token logprob readout | Dominated per the paper; needs logit access. Skip, or one-line ablation. |
| Monte Carlo persona sampling (N synthetic respondents vote) | Natural dispersion + coherent segments; 20–50× the calls. See Approach B. |
| Prompt-paraphrase / multi-model ensembling | Cheap variance reduction; prompt sensitivity is a known noise source. Include small (3–5 paraphrase) ensemble with ablation. |

### B. Conditioning — how segments enter
| Option | Assessment |
|---|---|
| Direct segment prompting | The baseline the paper shows degrades scores. Our comparison point. |
| **Delta modeling** — predict the segment's *shift* from the population prediction | Decouples conditioning direction from prior accuracy — exactly the decomposition the counterfactual metric asks for. Recommended. |
| Post-stratification — predict per-cell, reweight by marginals; population = weighted sum of segments | Elegant coherence-by-construction; needs marginal weights per country×attribute, which SimBench's group sizes partially supply. Strong option if delta modeling underwhelms. |
| Persona-pool filtering | Belongs to Approach B. |

### C. Calibration & structural correction — where the cheap points likely live
| Option | Assessment |
|---|---|
| **Entropy-aware recalibration** (temperature/Dirichlet mapping fit on held-out families, targeting mode-seeking) | The headline treatment, aimed at the paper's own diagnosis. Recommended. |
| Ordinal/latent-scale modeling (ordered-probit: predict location+spread on Likert scales, discretize) | Principled dispersion fix; survey-methodology depth competitors won't bring. Medium effort — stretch goal or roadmap item. |
| Response-style priors (acquiescence, central tendency) as correction terms | Cheap to describe, fiddly to fit in-budget. Roadmap item. |
| Shrinkage toward uniform/empty-persona with learned weight | Degenerate case of recalibration; comes for free as an ablation point. |

### D. Information augmentation
| Option | Assessment |
|---|---|
| Retrieval-anchored few-shot (neighboring items' distributions from held-out families) | Probably the biggest raw score gain; lives or dies on a bulletproof leakage protocol. See Approach C. |
| Tool/web context (country, year, current events) | Plausible for time-sensitive items (trust-in-president); small targeted ablation at most. |
| SimBench train-time use (exemplars/calibration data) | Permitted; our self-constructed dev split *is* this, with leave-family-out discipline. |

### E. Fine-tuning (LoRA to emit distributions)
Highest effort, least distinctive, worst hour-fit. **Listed to argue against**; appears in Part II as the far end of the spectrum and in the roadmap.

### F. Measurement & evaluation — the signature layer (in every variant)
- **Self-designed counterfactual sensitivity score:** compare predicted vs. true *delta vectors* (segment minus population) — directional agreement + magnitude correlation; ordinal-aware where scales are ordered.
- **Reliability-ceiling normalization:** ground-truth distributions are finite-sample estimates (group_size is published). Bootstrap the sampling noise to get a per-question ceiling on achievable S; report normalized scores. The Park/Kinzinger test-retest logic applied to this benchmark — no other candidate will do this.
- **Error decomposition:** accuracy (TVD) / rank-order / dispersion per prediction.
- **Holdout discipline:** dev/eval split by source dataset family; all tuning on dev only; documented leakage protocol.
- **Uncertainty:** bootstrap CIs on all scores; stratified evaluation subsample (~300–500 Pop + ~300–500 Grouped cases incl. all required-question variants).

## 6. Candidate spines

### Approach A — Calibrated simulator with coherent conditioning ⭐ recommended
Verbalized elicitation (small paraphrase ensemble) → entropy-aware recalibration fit on held-out families → delta-modeled segment conditioning → full F-layer.
**Narrative:** "The paper diagnoses mode-seeking; we treat the diagnosis and prove the treatment on held-out question families."
**Fit:** lowest compute; every component has a one-variable ablation; on-thesis. **Risk:** modest method novelty — novelty must come from the calibration design and measurement layer (matches chosen posture).

### Approach B — Silicon post-stratification (persona Monte Carlo)
Persona pool (Nemotron/census-derived) → per-persona answers → post-stratified aggregation; segments = reweighting the same pool.
**Fit:** higher novelty, natural dispersion, coherent conditioning by construction. **Risk:** 20–50× API calls; literature says personalization signal is modest in the attitude regime; persona quality is an uncontrolled confound. Honest estimate: 15–25 hours done properly. **Role:** limitations/roadmap — the natural next architecture, connecting to the agent tier of the prep-doc stack.

### Approach C — Retrieval-anchored prediction
Nearest-neighbor survey items + their distributions as few-shot anchors, leave-family-out.
**Fit:** likely biggest raw S gain. **Risk:** "you fed it neighboring ground truth" is the first attack an evaluator runs; the defense consumes scarce presentation time. **Role:** optional small ablation if hours remain (anchors strictly from held-out families); otherwise roadmap.

## 7. Part II direction sketch (written from conviction, ~1–2h + diagram)

- **Compounding memory:** tiered — raw outcome/event store → segment-level posterior store → cached calibrators/classifiers per tenant. What's stored: outcomes keyed to predictions made (prediction ledger), enabling honest backtesting.
- **Method spectrum:** retrieval/ICL (cold, low data) → cached per-tenant calibration layers (warm) → per-tenant adapters/fine-tunes (hot, only past data-volume and stability gates). Gate transitions by data volume, drift, and measured lift on holdout.
- **Revealed vs. stated:** different epistemic classes. Revealed behavior = calibration labels and ground truth for the prediction ledger; stated responses = persona priors and cold-start conditioning. Never average them; route them. (The evidence-map thesis, operationalized.)
- **Multi-tenancy:** the honest position — per-tenant data improves per-tenant predictions; share *structure* (calibrators' functional form, harness, ontology), not data. No cross-customer flywheel claimed.
- **Guardrails:** champion/challenger shadow evaluation; updates ship only on held-out lift; drift monitors on input and outcome distributions; the reliability ceiling as the hard stopping condition for tuning.

## 8. Infrastructure sketch (brief, per the brief's ask)

- **Compute:** laptop + hosted APIs (Gemini Flash / Together Qwen). No GPUs needed at this scale; state the scale-up path (vLLM on reserved instances for open-weights bulk inference) and the on-demand-vs-reserved tradeoff.
- **Models:** hosted APIs behind a thin provider-agnostic client (swappable-by-design; one-line model swap is itself an ablation).
- **Data:** SimBench CSVs → parquet locally; deterministic seeded subsampling; cached raw model outputs (JSONL) so recalibration/ablation reruns are free.
- **Experiment tracking:** config-hashed run directories + a results table (or lightweight W&B); every figure regenerable from cached outputs.

## 9. Proposed hour budget (Approach A spine)

| Phase | Hours | Output |
|---|---|---|
| 1. Harness + data + zero-shot baseline on subsample | 2–3 | S for baseline; the eval machinery (the rigor layer is built *first*) |
| 2. Recalibration + conditioning + ablations | 2–3 | Headline result + ablation table |
| 3. Counterfactual metric + reliability ceiling + required-question deep-dives | 1–2 | The signature analyses |
| 4. Part II writeup + diagram | 1–2 | Architecture section |
| 5. Presentation assembly | 1–2 | Slides |

Total: 7–12h — the 5h floor is achievable by trimming phase 2 to recalibration-only and phase 3 to the required questions.

## 10. Open questions (for next session)

1. Subsample size vs. full-split evaluation — how much do we trade CI width for spend/wall-clock?
2. Exact recalibration functional form (global temperature vs. entropy-conditioned vs. Dirichlet) — decide after seeing dev-split error decomposition.
3. Which model is the workhorse (Gemini Flash vs. Qwen-72B-class on Together) — pick by a 50-case pilot, not by debate.
4. Does the presentation lead with Part I results or with the evidence-map framing? (Sequencing question for later.)
