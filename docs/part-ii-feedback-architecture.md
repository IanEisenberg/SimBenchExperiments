# Part II — Feedback Mechanism Architecture

**Date:** 2026-06-21 · **Status:** Design document (no implementation required)
**Context:** Commercial deployment of the Scrye population simulator; real-world outcomes begin arriving.

---

## Executive Summary

Part I built and validated a methodology for predicting survey response distributions using LLMs. The validated system — a task-kind router that dispatches to the right prompt intervention based on question type, confirmed on held-out val at **55.57 grouped** (+1.89 over the prior champion) — is the empirical foundation for this design. But it is important to be precise about what that validates: a method for matching population-level opinion distributions over discrete survey options. The commercial system described here deals with a qualitatively richer problem — individual-level behavioral prediction from transaction logs — where the data granularity, signal complexity, and modeling requirements are all different.

What transfers from Part I is not the specific numbers but the *methodology*: the principle that different task types warrant different interventions (routing), the discipline of locking decisions before seeing outcomes (prediction ledger), the insistence on held-out validation before shipping updates (gated pipeline), and the honest accounting of where the model is already well-calibrated vs. where it needs correction.

Three structural bets define this architecture:

**1. Task/environment description as universal input enrichment — not routing dispatch.** Part I's routing gain came from giving the model better contextual framing of the decision environment (sibling survey items, task structure) — not from the classification machinery itself. The classifier was fragile; the enrichment principle is durable. In the commercial system, every prediction receives a rich description of the task structure, decision environment, and relevant contextual factors — the model conditions on this context rather than being switched between strategies. Abstain-on-uncertainty is preserved as a separate coverage policy: knowing when not to predict (risky choice, moral dilemma) is distinct from knowing how to enrich a prediction. The enrichment configuration itself — what contextual factors to include, how to structure the task description — is a learned artifact, discovered empirically as outcome data accumulates.

**2. Data normalization layer: all inputs refashioned as single-choice behavioral records.** Rather than maintaining separate pipelines for surveys, transactions, and experimental data, a normalization layer converts every input type into a canonical single-choice behavioral record. The survey-to-behavioral bridge (sampling synthetic individuals from population distributions) is one component of this layer; behavioral ETL is another; experimental data a third. This unified representation is what enables Centaur to train on heterogeneous sources, what makes the memory tiers (L1–L4) format-agnostic, and what allows the same prediction stack to handle cold-start via synthetic data and hot-path via real behavioral outcomes without special-casing.

**3. Centaur enriched by task/environment encoding — a primary experimental lever.** Following Binz et al. (2024), a foundation model fine-tuned on behavioral outcome data learns structural properties of human decision-making that generalize across tasks and populations — encoding this in weights rather than records, which is the only principled form of cross-tenant learning. The additional hypothesis — that Centaur's predictions further improve when inputs include rich task/environment descriptions — is the architecture's primary experimental lever for Phase 1. The cross-domain generalizability from cognitive tasks (Psych-101) to e-commerce behavior is also empirical; the architecture mandates testing both before asserting either.

---

## 1. What Part I Established — and Its Scope

### The Validated System

The final val-confirmed system (Stage 16) is:

```
RoutingPredictor(LLMTaskClassifier, KIND_ROUTES)
+ abstain-floor {OSPsychMACH}
@ gemini-3.1-flash-lite
```

Where:
```
KIND_ROUTES = {
  opinion_survey  → task_context       (sibling survey items as context)
  knowledge       → task_context
  risky_choice    → abstain            (uniform; voting overfit dev, dropped Stage 16)
  moral_dilemma   → abstain
  personality_scale → calibrated_commitment
  other           → calibrated_commitment
}
```

Val results vs. Stage 12 champion (calibrated_commitment + abstain):

| split | champion | router | Δ [95% CI] |
|---|---|---|---|
| grouped | 53.68 | **55.57** | +1.89 [+1.06, +2.73] |
| pop | 36.90 | 37.20 | +0.30 [−1.32, +1.88] |
| pooled | 45.38 | **46.48** | +1.10 [+0.18, +1.94] |

The gain is concentrated on grouped (survey-conditioned) predictions via task-context on opinion and knowledge items. The pop gain is flat — routing doesn't hurt, but the pop challenge (predicting across heterogeneous self-contained tasks) is not resolved by task-context alone.

Key per-kind breakdown:

| kind | route | Δ |
|---|---|---|
| opinion_survey (2784) | task_context | +1.57 |
| knowledge (311) | task_context | +5.53 |
| risky_choice (139) | abstain | saved −14.12 by NOT using voting |
| moral_dilemma (158) | abstain | 0 (already abstaining) |
| personality_scale (111) | base | +0.13 |

The Stage 16 val gate did its job: the `risky_choice → voting` route looked good on dev (+12.6 on Choices13k) but failed on val (−3.0 vs uniform's +11.1). This is the canonical example of why gated validation is non-negotiable.

Stage 06 established a separate important fact: all post-hoc calibrators (temperature scaling, entropy-adaptive temperature, Dirichlet smoothing — 14 configurations total) failed to improve on the identity calibrator. The `anti_flattening` prompt strategy already produces well-calibrated distributional predictions on its native domain. This tells us precisely where calibration is NOT needed (stated preference survey distributions) and therefore where it IS: revealed economic behavior, which is out-of-distribution for a text-pretrained model.

### Methodological Lessons That Transfer

Four lessons from Part I carry forward regardless of domain:

1. **Environmental enrichment beats generic prompting.** Providing the model with richer contextual description of the decision environment — sibling items, task framing, option structure — consistently outperforms generic prompting. The Part I routing system delivered this enrichment through a task classifier, but the gain was in the enrichment itself, not the classification machinery. In a commercial system with diverse and novel task types, universal enrichment is more defensible than a brittle dispatch map.

2. **The abstain principle.** When the model is confidently wrong (moral dilemmas, certain risky choice frames), predicting uniform is strictly better than predicting the model's output. Every deployment needs explicit failure-mode identification and abstention.

3. **Calibration is domain-specific.** Post-hoc calibration is only warranted where the model's prior is systematically wrong. On its native domain (text expressing attitudes), the LLM is already calibrated by good prompting. The calibration investment belongs where the model is genuinely OOD.

4. **The prediction ledger discipline.** Stamp predictions before outcomes arrive. This is what makes calibration errors computable honestly and what makes the update gate meaningful. Without the ledger, all evaluation is retrospective and subject to hindsight contamination.

### Honest Scope of the SimBench Results

SimBench is a distribution-over-discrete-options prediction task: given a survey question and demographic segment, predict what fraction of respondents chose each answer. This is a real and difficult problem, and the results are meaningful — but it is not behavioral prediction, and the commercial system must not be designed as if it were.

What SimBench measures:
- Population-level distributional accuracy over a fixed, known option set
- Ability to condition on a demographic segment defined in text
- Generalization across survey instruments and topic areas (within the survey domain)

What commercial behavioral data adds — and SimBench cannot prepare us for:
- **Individual-level event sequences**: who bought what, when, in response to what stimulus. Not a distribution over options but a stream of timestamped individual events. The learning target is fundamentally different.
- **Temporal dynamics**: drift, churn, re-engagement cycles. Survey distributions are relatively stable within a wave. Consumer behavior shifts week to week.
- **Confound richness**: price, availability, placement, promotion interacted with individual purchase history. The raw behavioral signal conflates propensity with opportunity in ways survey responses do not.
- **No fixed option set**: the "choices" include browse-without-purchase, cart abandonment, return, delayed purchase, repeat purchase. The outcome space is open-ended.
- **Untested routing**: the KIND_ROUTES validated in Part I cover survey task kinds. The commercial kinds (`behavioral_propensity`, `offer_response`, `churn_risk`) have no validated routes yet. Routing here will require empirical discovery from behavioral outcome data.

The transition from SimBench to commercial data is a domain shift, not an extrapolation. The methodology validates — routing, prediction ledger, gated updates, signal routing by type. The specific scores do not. A system achieving 55.57 grouped on SimBench may be completely wrong about purchase propensity, and a system that predicts purchase propensity well may score poorly on SimBench. They are measuring different things.

---

## 2. From Survey Prediction to Behavioral Simulation

### The Scope Expansion

The commercial simulator must predict behavioral outcomes, not opinion distributions. These share a deep structure — both are about what humans do when facing a set of options — but differ in almost every surface feature.

**Granularity.** SimBench ground truth is an aggregate distribution: "37% said 'a lot', 29% said 'some', ...". Commercial ground truth is individual events: "user 4821 browsed product P3 for 14 seconds, added to cart, then abandoned; 6 hours later purchased P3 at a 15% discount." The individual level unlocks personalization but requires modeling of individual heterogeneity, not just segment statistics.

**Signal richness.** Survey options are clean ordinal labels. Transaction logs are multivariate event streams with continuous features (price, time-since-last-visit, session depth), categorical features (product category, channel, device), and implicit signals (dwell time, scroll depth, search terms). The feature space is orders of magnitude larger.

**Confound structure.** Survey responses are confounded by social desirability and hypothetical framing — known, stable biases that good prompting can partially address (what anti_flattening does). Transaction logs are confounded by price dynamics, promotion effects, inventory availability, seasonality, and individual purchase history. These confounds are not stable and must be modeled and removed before the signal can be used for calibration.

**Temporal dynamics.** Survey distributions are relatively stable within a wave. Behavioral patterns drift — consumer preferences shift with trends, competitive actions, seasonality, and life events. The simulator must track drift, not just fit a static distribution.

### What This Unlocks

The richer data enables things SimBench cannot:

- **Causal counterfactuals**: "if we had offered this segment a 10% discount instead of 15%, what would purchase propensity have been?" Survey distributions don't have offer-context variation; transaction logs do (natural experiments from historical promotions).
- **Individual-level personalization**: predict for a specific individual (or a micro-segment) rather than a demographic group. The commercial value is often in the tail — identifying the high-propensity individuals, not the population mean.
- **Sequence and recency effects**: how does a recent browse event change purchase probability in the next 24 hours? Survey waves are cross-sectional; transaction logs are longitudinal.
- **Cross-category dynamics**: does browsing category A increase purchase probability in category B? Only visible with individual-level sequence data.

### The Central New Challenge

The value-action gap is already the central constraint in the survey domain — SimBench shows behavioral tasks (economic games) predict at r ≈ 0.66 while attitude tasks predict at r ≈ 0.83. At the individual behavioral level, this gap is wider: individual purchase decisions are noisier than population distributions, more confounded, and more sensitive to unobservable individual state (mood, intent, competing considerations).

This is not a reason to avoid behavioral prediction. It is the reason the architecture must be built around honest calibration, rigorous holdout, and explicit uncertainty — and why the Centaur-style behavioral fine-tuning (a model trained specifically on behavioral outcomes, not just text about attitudes) is the architectural investment that pays off most at scale.

---

## 3. Architecture Overview

```
══════════════════════════════════════════════════════════════════════════════
  DATA INGESTION
══════════════════════════════════════════════════════════════════════════════

  Revealed Behavior           Stated Responses          External Signals
  (transaction logs)          (survey waves)            (market context,
       │                           │                     seasonality, etc.)
       ▼                           ▼                           │
  ┌──────────────────┐      ┌───────────────────┐             │
  │  Behavioral ETL  │      │    Survey ETL     │             │
  │  - normalize     │      │  - aggregate →    │             │
  │    offer context │      │    distributions  │             │
  │  - dedup,        │      │  - quality meta   │             │
  │    deconfound    │      │    (n, recency)   │             │
  └────────┬─────────┘      └────────┬──────────┘             │
           │                         │                         │
           │                         ▼                         │
           │              ┌──────────────────────┐            │
           │              │ Survey→Behavioral     │            │
           │              │ Bridge                │            │
           │              │ - sample N synthetic  │            │
           │              │   individuals per dist│            │
           │              │ - assign demographic  │            │
           │              │   + behavioral profile│            │
           │              │ - construct pseudo-   │            │
           │              │   transaction logs    │            │
           │              └──────────┬────────────┘            │
           │                         │ (synthetic behavioral)   │
           ▼                         ▼                         ▼

══════════════════════════════════════════════════════════════════════════════
  COMPOUNDING MEMORY  (per-tenant, data-isolated)
══════════════════════════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────────────────────┐
  │  L1 — Raw Outcome Store  (append-only, permanent)                      │
  │  ├── behavioral_events: {user_id, item_id, action, offer_ctx, ts}      │
  │  ├── survey_responses:  {wave_id, item_id, dist, segment_meta, n}      │
  │  └── synthetic_events:  {bridge_run_id, provenance, pseudo_event, ts}  │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  L2 — Prediction Ledger  (stamped before outcomes arrive)              │
  │  ├── prediction: {id, spec_hash, task_kind, segment, pred_dist, ts}    │
  │  └── outcome:    {prediction_id, ground_truth, outcome_source}  ← late │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  L3 — Segment Posterior Store  (aggregated, EMA-weighted)              │
  │  ├── empirical_dists: per-segment × task-kind × context               │
  │  └── calibration_residuals: {segment, kind, tvd_err, entropy_gap, n}  │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │  when update gate passes
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  L4 — Calibrator Cache  (versioned; rollback-able)                     │
  │  ├── route_params: per-kind calibration params (kind→calibrator)       │
  │  └── adapter_weights: LoRA / Centaur weights (human-gated)            │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼

══════════════════════════════════════════════════════════════════════════════
  BEHAVIORAL FOUNDATION  (shared across tenants; weights, not data)
══════════════════════════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────────────────────┐
  │  Centaur Foundation Model                                              │
  │  Training corpus:                                                      │
  │    Phase 0 (day 0):   synthetic behavioral from survey bridge          │
  │    Phase 1 (months):  real behavioral (anonymized cross-tenant agg)    │
  │    Phase 2 (scale):   synthetic + real, weighted by provenance         │
  │  Training method: periodic batch fine-tune (LoRA on backbone LLM)     │
  │  Validation: same update gate as L4; held-out cross-tenant holdout     │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼

══════════════════════════════════════════════════════════════════════════════
  PREDICTION PIPELINE  (at inference time)
══════════════════════════════════════════════════════════════════════════════

  Query: {question, segment, offer_context, individual_history?}
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Task/Environment Enrichment  (configurable, versioned) │
  │  Augments query with contextual description:            │
  │   - decision environment: option set, stakes, channel   │
  │   - task structure: type, format, framing               │
  │   - behavioral priors from L3 (k-NN retrieved)         │
  │  Enrichment config is a learned artifact (gated, v'd)   │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ├── abstain policy: risky choice / moral dilemma
                           │          └──→ return uniform (coverage policy)
                           ▼
          ┌──────────┬──────────┐
          ▼          ▼          ▼
      COLD PATH  WARM PATH  HOT PATH
      (n<100)   (n≥100)    (n≥10k, stable)
      ICL from  Centaur+   Centaur + LoRA
      L3        L4 calibr  adapter (human-gated)
          │          │          │
          └──────────┴──────────┘
                     │
                     ▼
            Predicted distribution
                     │
                     ▼
            ← stamp to L2 Ledger  (the closed loop)

══════════════════════════════════════════════════════════════════════════════
  UPDATE GATE  (before any L4 or Centaur change ships)
══════════════════════════════════════════════════════════════════════════════

  Shadow eval  →  Lift test (Blum-Hardt η)  →  Ceiling check
       →  Drift check (PSI)  →  Human sign-off (fine-tune only)
       │                                                    │
       PASS: promote challenger, retire old version         │
       FAIL: keep champion; log challenger to L2 ───────────┘
```

---

## 4. The Data Normalization Layer

### The Normalization Principle

The system ingests three distinct data types — revealed behavior (transaction logs), stated preferences (survey waves), and experimental outcomes. These have different formats, different granularities, and different epistemic statuses. The normalization layer converts all of them into a canonical single-choice behavioral record before any downstream processing. This unification is what makes the memory tiers (L1–L4) format-agnostic, what enables Centaur to train on heterogeneous sources, and what allows the prediction stack to operate identically regardless of input origin.

The canonical behavioral record:

```json
{
  "agent_id":   "individual or segment identifier",
  "context":    "decision environment: option set, stakes, framing, channel",
  "choice":     "selected option (or sampled from distribution)",
  "outcome":    "downstream result if observed",
  "provenance": "behavioral | survey_bridge | experimental",
  "weight":     "provenance-based confidence (1.0 real, 0.4–0.7 bridge)"
}
```

Three normalization paths produce this format.

### Path A — Behavioral ETL

Transaction logs arrive as individual events: `{user_id, item_id, action, price, placement, channel, timestamp}`. The ETL step:
1. **Deconfounds**: strips offer effects (price promotion, placement boost) from the raw action signal to isolate behavioral propensity
2. **Normalizes**: maps item/category identifiers to a stable taxonomy
3. **Tags**: adds provenance = "behavioral" and weight = 1.0

These records are the ground-truth training signal. LLMs are genuinely OOD on this domain; calibration is required.

### Path B — Survey-to-Behavioral Bridge

Survey waves arrive as population distributions over discrete options. The bridge converts these into synthetic individual behavioral records:

**Step 1 — Individual sampling.** Draw N synthetic agents from the survey distribution. Assign each a demographic profile (from the segment definition) and a latent behavioral propensity initialized as a noisy function of their opinion score, scaled by the empirical attitude-behavior correlation (r ≈ 0.4–0.6).

**Step 2 — Behavioral event generation.** For each agent, generate plausible events consistent with their propensity: high-propensity agents get purchase-like events; low-propensity get browse or abandon events. Event timing follows realistic inter-event distributions. Noise is calibrated to the known attitude-behavior gap.

**Step 3 — Record construction.** Assemble events into canonical behavioral records tagged provenance = "survey_bridge" with weight < 1.0 (reflecting the fidelity ceiling). The bridge corpus is available from day one before any real outcomes exist.

**Fidelity ceiling.** Bridge fidelity is bounded by the attitude-behavior correlation and social desirability bias in the source survey. As real behavioral data arrives, the gap is measurable: compare Centaur fine-tuned on bridge-only vs. real outcomes. If the gap is large and persistent, bridge generation needs revision. If it closes quickly, the bridge was a good warm-start.

### Path C — Experimental and Public Behavioral Data

Psych-101 (Binz et al.'s Centaur training corpus) and SimBench-derived data are also normalized into this format. These records have high provenance quality for their domain (cognitive and attitude tasks) and serve as Phase 0 Centaur training data before any client data arrives. Provenance = "experimental"; weight reflects domain distance to the target prediction task.

### What Normalization Enables

By reducing all input types to the same record schema before they enter L1, the architecture achieves:
- **Centaur training agnosticism**: one fine-tuning pipeline handles all data sources; provenance weights control their relative influence
- **L1 simplicity**: the raw store holds one event type, not three
- **Consistent calibration**: the same calibration error computation (predicted vs. actual choice) applies regardless of whether the ground truth came from a transaction or a bridge-generated record
- **Graceful degradation**: as real data accumulates, bridge records' weights decline automatically; the pipeline behavior is continuous, not a hard switch
## 5. The Compounding Memory

Four tiers, each serving a distinct purpose.

### L1 — Raw Outcome Store (append-only, permanent)

The audit log. Every event lands here in normalized but minimally processed form. L1 is append-only: nothing is overwritten or deleted. It is the source of truth for retrospective backtesting, drift forensics, and reprocessing if downstream aggregation logic changes.

Three event types:
- **behavioral_events**: individual transaction records from the client, normalized for offer context (price, promotion flag, placement position, channel). The raw purchase signal is confounded; the ETL strips promotion effects before downstream use, but the original event with offer context is preserved in L1 so the deconfounding model can be iterated.
- **survey_responses**: aggregate distributions from survey waves, with quality metadata (n, recency, fielding method, response rate). Stored at the wave-question-segment grain.
- **synthetic_events**: pseudo-transaction logs from the survey-to-behavioral bridge, tagged with provenance. Stored in the same schema as behavioral_events so the pipeline can process them identically, with the provenance flag allowing differential weighting.

### L2 — Prediction Ledger (stamped before outcomes arrive)

The key abstraction preventing data snooping. Every prediction is recorded before its outcome is known:

```
prediction: {
  id:           UUID,
  spec_hash:    SHA256 of pipeline config (model, strategy, calibrator version),
  task_kind:    classifier output (behavioral_propensity / opinion_survey / ...),
  segment:      demographic conditioning,
  context:      offer context at prediction time,
  pred_dist:    the predicted distribution (array of (option, probability)),
  timestamp:    when the prediction was made
}
```

When the outcome arrives (transaction event, new survey wave), it is linked:

```
outcome: {
  prediction_id:   UUID matching the prediction,
  ground_truth:    the realized distribution or event,
  outcome_source:  "behavioral" | "survey" | "synthetic",
  link_timestamp:  when the outcome was observed
}
```

This enables honest calibration error computation: error = f(predicted_dist, ground_truth) where ground_truth was unknown at prediction time. Without the ledger, all calibration fitting is retrospective and trivially gameable.

### L3 — Segment Posterior Store (aggregated on schedule)

The living summary of learned distributions. Two components:

- **empirical_dists**: per-segment × task-kind × offer-context-bucket distributions, aggregated from L1, weighted by an exponential moving average (λ = 0.9 weekly, so ~10 weeks half-life — tunable per client based on known drift rates). This provides the retrieval target for the cold path and the baseline for drift detection.
- **calibration_residuals**: structured error signals from L2 matching — how far the current pipeline's predictions are from ground truth, broken down by segment, task kind, and offer context bucket. This is the input to calibrator fitting for the warm path.

L3 is recomputed on a schedule (daily for behavioral events, per-wave for surveys) or triggered by data volume thresholds. The EMA weighting means L3 naturally tracks trend without requiring explicit drift detection in the retrieval path.

### L4 — Calibrator Cache (gated, versioned, rollback-able)

The operational artifact. Per-tenant, versioned, and subject to the update gate before promotion:

- **route_params**: per-kind calibration parameters (temperature coefficients, segment shift vectors, abstain thresholds). These are the warm-path artifacts — lightweight, fast to fit, low blast radius.
- **adapter_weights**: LoRA weights from per-tenant fine-tuning or the Centaur foundation model weights. These are the hot-path artifacts — expensive to produce, hard to roll back, require human sign-off.

Every L4 version is retained for 30 days after promotion. Rollback is deterministic (re-serve the previous version) and automated when the production TVD alarm fires.

---

## 6. Method Spectrum

The method for any prediction is determined by: (a) the task kind (from the classifier), and (b) the data tier (from the volume of available outcomes for that segment × kind). These two axes jointly determine which intervention applies.

### Cold Path — ICL + Task-Kind Routing (n < 100 outcomes)

The default for new tenants, new segments, and new task kinds. The task classifier determines the retrieval strategy:

- **opinion_survey / knowledge**: retrieve sibling items (same survey instrument, same segment) from L3. Use them as few-shot context in the prompt, exactly as validated in the Part I task-context route. This is the highest-validated path for opinion prediction.
- **behavioral_propensity**: retrieve the k nearest segments from L3 by demographic similarity (k-NN in demographic embedding space). Use their empirical behavioral distributions as the prior context in the prompt. The Centaur backbone is already doing some of this work if it has been trained on similar domains.
- **risky_choice / moral_dilemma**: abstain (return uniform). The Part I lesson — validated on val — is that the model is confidently wrong on these task kinds and uniform strictly dominates. This applies equally in the commercial setting.
- **offer_response**: retrieve historical A/B test results from L3 by similar segment × offer-context. Use as anchors for the LLM prediction.

### Warm Path — Calibrated + Routing (n ≥ 100 outcomes)

Once 100+ labeled outcomes exist for a segment × kind, fit a calibrator from L2. The calibration functional form is kind-specific:

- **opinion_survey**: entropy-conditioned temperature (higher temperature on high-entropy predictions, lower on consensus ones). This is what Stage 06 showed is NOT needed for well-prompted survey predictions — but may be needed if a commercial survey instrument differs from SimBench's norms.
- **behavioral_propensity**: logistic recalibration of the LLM's propensity estimate against realized purchase rates. The LLM's zero-shot behavioral estimate is genuinely OOD and will have systematic bias; a simple platt-scaling logistic calibrator per segment is the first correction.
- **offer_response**: calibrate by offer-context bucket; the LLM tends to overestimate promotional lift for unfamiliar offer structures.

### Task/Environment Enrichment as a Continuously Learned Artifact

The enrichment configuration — what contextual factors to include, how to structure the task description, what priors to retrieve from L3 — is not a static config. It is a first-class gated artifact, versioned and updated through the same five-step protocol as calibrators.

The principle from Part I: the enrichment gain (task_context on survey/knowledge items: +1.57 and +5.53 on val) came from better environmental description, not from the classification mechanism. What transfers to the commercial system is the principle that richer, more structured task descriptions produce better predictions. The specific content of those descriptions — which factors matter for behavioral propensity vs. offer response vs. segment comparison — must be discovered from outcome data, not assumed.

For each prediction context, the optimal enrichment *e\*(c)* is:

```
e*(c) = argmax_e E[TVD_improvement(e) | context_type = c, n_outcomes ≥ threshold]
```

evaluated on held-out outcomes from L2. When sufficient labeled outcomes exist for a context type, the enrichment choice is testable. A new enrichment structure (e.g., including competitive context for offer-response tasks, or recency-weighted history for churn prediction) is introduced as a challenger, shadowed against the current configuration, and promoted if it clears the Blum-Hardt lift threshold.

The starting enrichment configuration for commercial contexts is an informed prior:

| context type | starting enrichment (prior) | basis |
|---|---|---|
| `behavioral_propensity` | individual history + segment prior from L3 | LLM needs behavioral grounding |
| `offer_response` | A/B context + offer parameters + segment empirical | analogous to task_context on surveys |
| `segment_comparison` | contrast framing + both segments' priors | delta prediction avoids distorting priors |
| `churn_risk` | abstain until n ≥ 500 | high-stakes; cold enrichment is unreliable |
| `opinion_survey` | sibling items as context (val-confirmed from Part I) | direct transfer |

Each configuration will be confirmed, modified, or replaced as behavioral outcome data accumulates. The abstain cases (`churn_risk` at low n) are coverage policy decisions, not enrichment decisions — they are fixed until volume justifies a prediction attempt.

### Hot Path — Centaur Fine-Tune (n ≥ 10k, stable, ceiling)

Only when: (a) outcome volume exceeds the threshold, (b) the calibration layer has plateaued (improvement from warm path < η over last update cycle), (c) the Centaur foundation backbone is available and stable, and (d) the update gate passes including human sign-off.

The LoRA adapter per tenant encodes the residual between the Centaur backbone's predictions and the tenant-specific outcome distribution. This is a much smaller tuning target than fine-tuning from scratch — the Centaur backbone has already absorbed most of the behavioral structure; the adapter corrects for what is idiosyncratic to this tenant's population.

**The risk**: the adapter overfits to a specific epoch of the tenant's customer behavior. If the customer base shifts (new market segment, product line extension, competitor pricing change), the adapter may make the predictions worse. The drift detector (PSI on outcome distributions) triggers re-evaluation, and the rollback contract ensures the previous adapter is available.

---

## 7. Signal Types

Three distinct epistemic classes. They must be routed to different roles in the architecture — never averaged, never treated as the same kind of evidence.

### Revealed Behavior (Transaction Logs) → Calibration Labels

- **What it measures**: what people *did* under real stakes, real prices, real availability
- **LLM nativeness**: low — the LLM was pretrained on text; economic choices with real monetary consequences are genuinely OOD
- **Confound burden**: high — a purchase conflates propensity, availability, price sensitivity, promotion exposure, and individual state. The ETL must isolate propensity from opportunity before this signal can serve as a calibration label.
- **Role**: primary ground truth for L2 ledger outcomes; the signal that calibrates the warm and hot paths; the training target for the Centaur behavioral fine-tune
- **What it unlocks that surveys can't**: individual heterogeneity, temporal sequences, cross-item effects, causal identification via natural promotion experiments

### Stated Preferences (Survey Waves) → Persona Prior and Cold-Start

- **What it measures**: what people *say* they value and believe, under hypothetical framing
- **LLM nativeness**: high — surveys are exactly what the model was pretrained on (attitudes, opinions, self-reports in text)
- **Biases**: social desirability, hypothetical frame, non-response
- **Role**: soft prior for the cold path; source of conditioning for opinion/attitude predictions; input to the survey-to-behavioral bridge; L3 posterior updates via survey waves
- **Epistemic limitation**: cannot serve as ground truth for behavioral calibration; systematically predicts behavior at r ≈ 0.4–0.6 at the individual level

### Synthetic Behavioral (Bridge Output) → Centaur Training Bootstrap

- **What it is**: pseudo-transaction logs generated by sampling from survey distributions and applying the attitude-behavior correlation model
- **LLM nativeness**: n/a — this is generated data, not a model input
- **Provenance**: always tagged; differentially weighted below real behavioral data
- **Role**: Phase 0 training corpus for Centaur; cold-path retrieval target when L3 is empty for a new segment; degrades gracefully as real data accumulates
- **Fidelity ceiling**: bounded by the attitude-behavior correlation and the social-desirability/hypothetical biases in the source survey data

### How They Interact

```
Signal Type          → Role in System             → NOT used for
─────────────────────────────────────────────────────────────────────
Revealed behavior    → calibration labels (L2)    → prior conditioning
                     → Centaur training (real)    → cold-start prior
                     
Stated preferences  → persona prior               → calibration target
                     → cold-start conditioning    → Centaur training alone
                     → survey bridge input
                     
Synthetic behavioral → Centaur Phase 0 training   → calibration labels
                     → cold-path L3 bootstrap      → direct predictions
```

The value-action gap makes these boundaries load-bearing. Blurring them — treating survey responses as calibration targets for behavioral prediction, or using synthetic behavioral data as ground truth — would corrupt the calibration and produce overconfident predictions on a distribution the model has never been honestly tested against.

---

## 8. Behavioral Foundation Model (Centaur-Style)

### Motivation

The text-pretrained LLM carries a strong prior over stated preferences (its native domain) but a weak and systematically biased prior over revealed behavioral outcomes. The warm-path calibration corrects for the bias per tenant, but it starts from a weak baseline. A foundation model additionally trained on behavioral outcome data would start from a better prior — requiring less calibration to correct, and generalizing better to new segments and new task kinds within the behavioral domain.

Binz et al. (2024) — *Centaur: A Foundation Model of Human Cognition* — demonstrated this approach on cognitive behavioral data: fine-tuning Llama-3.1-70B on the Psych-101 dataset (~60k participants across thousands of cognitive experiments) produced strong cross-task generalization to held-out cognitive tasks and populations. The resulting model "thinks like humans" in a way that zero-shot LLMs do not — not because it was instructed to, but because it was trained on behavioral outcomes.

### Training Data Composition

The Centaur training corpus is built in phases, with explicit provenance tracking:

| Phase | Data source | Provenance weight | When available |
|---|---|---|---|
| 0 | Synthetic behavioral (bridge output) | 0.3 | Day 1 |
| 0 | Psych-101 / public cognitive behavioral | 0.5 | Day 1 |
| 0 | SimBench-derived pseudo-behavioral | 0.2 | Day 1 |
| 1 | Anonymized cross-tenant real behavioral | 0.7 (rising) | After 3 months |
| 1 | Synthetic (declining share) | 0.3 (falling) | Ongoing |
| 2 | Primarily real behavioral | >0.9 | At scale |

The provenance weights are not hyperparameters to tune on the validation metric — they reflect the epistemic quality ordering: real behavioral outcomes are the ground truth, synthetic is a prior. The weight schedule is predetermined and changes on a calendar schedule, not in response to evaluation metrics.

### The Cross-Domain Generalizability Question

The Centaur result holds for cognitive tasks (memory, reasoning, perceptual judgment, risky choice under laboratory conditions). Whether it transfers to commercial behavioral data (e-commerce purchases) is an open empirical question, not a design assumption.

The domain gap is real and non-trivial:
- Psych-101 tasks are controlled experiments with clean stimuli; transaction logs are messy, confounded, and ecologically complex
- Laboratory risky-choice tasks (lotteries with stated probabilities) are different from real purchase decisions under uncertain promotion effects
- Cognitive tasks are typically one-shot; purchase behavior has strong path dependence and recency effects

The architecture's response: run the Centaur backbone against a held-out cross-tenant behavioral test set (minimum two distinct engagement types; one sports retail, one from a different vertical if available) before asserting cross-domain transfer. The result of this test determines whether the Centaur backbone improves predictions relative to the text-pretrained baseline. If it does not, it is treated as a negative result and the warm-path calibration remains the primary learning mechanism.

### Incremental Updating

The Centaur foundation model is re-trained periodically (monthly initially; more frequently at scale) on the growing cross-tenant corpus. Each re-training:
- Incorporates new real behavioral data from all tenants (anonymized and aggregated)
- Reduces the synthetic share proportionally
- Is validated on the cross-tenant holdout before being promoted
- Goes through the same update gate as L4 changes, including a human sign-off

The re-training uses LoRA (parameter-efficient fine-tuning) to avoid full-model training costs. The LoRA rank is a function of the corpus size: smaller rank at Phase 0 (weak signal, avoid overfitting), larger rank at Phase 2 (strong signal, more capacity needed).

---

## 9. Generalization and Multi-tenancy

### What Does and Doesn't Generalize

**Does not generalize (per-tenant, data-isolated):**
- L1, L2, L3: raw outcomes, prediction ledger, segment posteriors are tenant-specific. Privacy law and commercial data-sharing agreements require this; the architecture enforces it.
- L4 calibrator params: fit on tenant-specific outcome distributions; not exchangeable across tenants with different customer bases.

**Generalizes via weights (Centaur foundation):**
- The Centaur backbone encodes cross-tenant behavioral structure in model weights. This is legitimate: it encodes *structure* (how demographic segments respond to offer types, how high-entropy behavioral states decompose) not *records* (who bought what). No raw customer data crosses tenant boundaries.

**Generalizes via method (validated protocol):**
- The routing architecture, the prediction ledger schema, the update gate protocol, the calibration functional forms — these are validated methods that cold-start any new tenant from a better prior than random initialization. The routing table KIND_ROUTES is the Part I result; it is available to all tenants as the default and refined per-tenant on their dev data.
- The reliability ceiling methodology (bootstrap CI on ground-truth finite-n noise) is a universal tool; the computed ceilings are per-tenant.

### The Two Flywheels

The honest version of the multi-tenancy flywheel has two components:

**Flywheel 1 — Method validation.** Each new engagement tests the routing architecture, the calibration protocol, and the update gate against real behavioral outcomes. The results inform what works and what doesn't. A failed route in one engagement (as voting failed on val in Stage 16) is a signal to remove or constrain that route for future engagements. The method compound by becoming more validated, not by pooling data.

**Flywheel 2 — Centaur weight sharing.** Each new engagement's behavioral outcomes (anonymized, aggregated) contribute to the next Centaur re-training cycle. The foundation model improves because it has seen more diverse behavioral distributions; this improvement benefits all tenants on the next cycle. This is the legitimate data flywheel — operating at the model-weight level, not the raw-record level.

### The Honest No-Flywheel Claim

A new sports retailer engagement does not make us better at predicting behavior for a pre-existing sports retailer client. The per-tenant calibrators are independent. The Centaur improvement from a new engagement is diffuse — it improves the foundation's behavioral prior in general, not the specific tenant's calibration.

The claim is: *a new engagement benefits from the validated method and the Centaur foundation model. It does not benefit from other tenants' raw behavioral data. Its own data, once accumulated, improves its own predictions.*

---

## 10. Guardrails and Evaluation

### Preventing Overfitting

**Volume gates.** Cold path (n < 100) uses retrieval only — no calibration fitting. Warm path (n ≥ 100) fits a calibrator. Hot path (n ≥ 10k) allows fine-tuning. The gate thresholds are not hyperparameters; they are derived from the minimum sample size needed to detect a signal above noise at α = 0.05 with 80% power, given the expected effect size from the domain literature.

**Reliability ceiling as hard stop.** Ground-truth distributions are finite-sample estimates. The bootstrap CI on the sampling noise gives an irreducible ceiling on achievable accuracy. Any update claiming improvement beyond this ceiling is fitting noise and is blocked. This is the one bound that cannot be gamed — it is a property of the data-generating process.

**Ensemble calibrators.** Fit N = 5 calibrators on different data windows (1-month, 3-month, 6-month, all-time, leave-last-wave-out). Ship only if they agree within bootstrap CIs. Disagreement across windows signals overfitting to a temporal slice.

**Leave-one-wave-out.** The calibrator is always fit on a held-out wave (the most recent survey wave or behavioral batch is withheld). The held-out wave is the primary validation set for the lift test.

### Preventing Drift

**Input drift (PSI).** Population Stability Index on the distribution of task kinds, segment frequencies, and offer contexts. Alert if PSI > 0.2 since last calibrator training. This is the signal that the input distribution has shifted enough that the calibrator may no longer be valid.

**Outcome drift.** Monitor empirical purchase rates by segment in a trailing window. Alert if any segment's rate shifts beyond 2σ of the calibrator's training-time baseline.

**Temporal windowing in L3.** The segment posterior store uses EMA weighting (recent outcomes count more). The system naturally tracks trend; drift detection is a backstop for sudden shifts, not a substitute for adaptive weighting.

### The Five-Step Update Gate

Before any L4 or Centaur change ships:

**Step 1 — Shadow evaluation.** The challenger runs in shadow on all live predictions for a burn-in (≥1,000 predictions or two calendar weeks, whichever is larger). Outputs are logged to L2 alongside the champion's but do not affect the client response.

**Step 2 — Lift test on the L2 holdout.** Using outcomes from the held-out wave (never used in calibration), compute TVD for champion and challenger on matched predictions. The challenger must show statistically significant reduction (bootstrap CI lower bound > 0) AND the raw Δ must exceed η — the noise floor derived from bootstrap variance on the holdout. This is the Blum-Hardt ladder condition: the same principle operationalized in Part I's experiment protocol, which prevents accepting updates that are within noise of no-improvement.

**Step 3 — Reliability ceiling check.** The claimed Δ must not exceed the reliability ceiling computed from ground-truth finite-n noise. If the challenger claims to exceed the ceiling, it is fitting noise.

**Step 4 — Drift check.** PSI on the challenger's input distribution vs. the champion's training-time distribution. A large PSI means the challenger was trained on a different distribution than it is being evaluated on — an artifact, not a real improvement.

**Step 5 — Human sign-off for fine-tune ships.** Calibrator parameter updates (L4 route_params) are automated once steps 1–4 pass. LoRA adapter and Centaur backbone updates require human review: blast radius is larger, rollback is harder, and subtle distributional artifacts are more likely.

**Rollback contract.** Every L4 version retained 30 days post-promotion. If production TVD rises by >2σ in a trailing window of 500 predictions, the system auto-rolls back and opens an incident. Re-promotion requires human review.

---

## 11. Infrastructure

### Compute

**Inference (prediction serving):** hosted API inference via OpenRouter (OpenAI-compatible gateway, single key, any vendor model accessible by model ID string). At low volume, on-demand API calls are cheaper than reserved GPU capacity. The OpenRouter architecture from Part I is the right answer here: one client, one request schema, one API key — vendor switching is a config change, not an integration. The cross-model portability Part I demonstrated (method gain is roughly model-invariant) means vendor lock-in is not a risk.

At higher tenant volume (>10k predictions/day), the latency and cost of hosted API inference justifies self-hosting: vLLM on reserved GPU instances (A100 or H100 for large models; A10G for smaller models like gemini-equivalent open-weights). The OpenRouter interface abstracts this: the same code points at a self-hosted vLLM endpoint with an OpenAI-compatible API.

**Centaur fine-tuning:** GPU spot instances (A100 40GB for LoRA fine-tuning of 7–13B models; H100 for 70B). Spot pricing is appropriate because fine-tuning runs are scheduled batch jobs, not latency-sensitive. One fine-tuning run per month at Phase 1; more frequent at Phase 2 if behavioral corpus grows fast. Cost estimate: ~$200–500/run for a 13B LoRA fine-tune on 1M behavioral events (12–24 hours on 4×A100 spot).

### Model Access

OpenRouter as the production-inference gateway. The same provider-agnostic pattern from Part I:
- One `LLMClient` with one `OPENROUTER_API_KEY`; `model=` string swaps vendor
- On-disk SHA-256 cache keyed by `hash(model, messages, sampling_params)` — re-runs are free and deterministic
- Production can pin to a specific provider (e.g., `google/gemini-flash-1.5-8b` directly) for latency SLA; the codebase is unchanged

For self-hosted inference: vLLM with the same OpenAI-compatible API; the `LLMClient` points at `localhost:8000` instead of OpenRouter. Zero code change.

### Data Storage

**L1 (raw outcomes):** append-only object storage (S3-compatible). Parquet partitioned by tenant × date × event_type. Cheap, durable, query-able via Athena/DuckDB. No deletions; GDPR compliance via per-tenant encryption keys (key revocation = effective deletion without modifying the store).

**L2 (prediction ledger):** columnar format (Parquet) in object storage, with an index on `prediction_id` and `timestamp`. The backtest query pattern (join prediction to outcome, filter by timestamp range, aggregate by segment) is efficient in columnar. Consider a lightweight analytical DB (DuckDB, MotherDuck) for interactive queries.

**L3 (segment posteriors):** key-value store keyed by `tenant_id:segment_hash:kind`. Redis for low-latency cold-path retrieval; S3 for persistent backup. Evict on a TTL matching the EMA half-life.

**L4 (calibrators + adapters):** versioned object storage. Calibrator params as JSON (small, fast to load). LoRA adapter weights as safetensors. Metadata catalog (DynamoDB or a simple SQLite) tracks version, training timestamp, holdout metrics, and rollback target.

**Centaur training corpus:** versioned data lake (Iceberg or Delta Lake format). Provenance column on every row. Fine-tuning jobs consume a specific snapshot of the lake, tagged to the Centaur model version they produced.

### Experiment Tracking

**For inference/calibration experiments:** config-hashed run directories (the pattern from Part I: `outputs/runs/<date>-<name>/`). Every run produces a `meta.json` (frozen config), `topline.csv` (aggregate scores), `results.json` (per-record scores). The ledger DAG tracks the sequence of updates and the champion/challenger history.

**For Centaur fine-tuning:** ML experiment tracking (Weights & Biases or MLflow). Each run tagged with: training corpus version, LoRA rank, base model version, holdout metrics, data provenance breakdown. Model artifacts registered with the above metadata so any Centaur version is reproducible from the corpus snapshot and the training config.

**The invariant:** every prediction in production is traceable to a spec_hash in L2, which resolves to a config, which resolves to a model version and a corpus snapshot. No prediction is unattributable.

---

## 12. Open Questions and Research Agenda

These are the dimensions where the architecture makes bets that have not yet been empirically tested, in priority order for future investigation.

### 1. Centaur Cross-Domain Generalizability

**The question:** does a model fine-tuned on cognitive behavioral data (Psych-101) and/or SimBench-derived pseudo-behavioral data produce better zero-shot priors for e-commerce purchase behavior than the text-pretrained baseline?

**Why it matters:** if yes, the Centaur foundation is a genuine flywheel and the investment in behavioral fine-tuning pays off from early engagements. If no, the foundation is just a text-pretrained LLM with extra training, and the warm-path calibration carries all the load.

**How to test:** run the Centaur backbone (fine-tuned on Phase 0 corpus) vs. the base LLM on a held-out cross-tenant behavioral test set from at least two distinct engagement types. Measure TVD improvement on purchase propensity predictions before any per-tenant calibration. A positive result > η is evidence of transfer; a null result means calibration is doing the work, not the foundation.

### 2. Survey-to-Behavioral Bridge Fidelity

**The question:** how good are synthetic behavioral pseudo-populations as training data for the Centaur model, and how much real data is needed before the synthetic bootstrap is no longer contributing positively?

**How to test:** at 3 months post-engagement, the first real behavioral data is available. Run an ablation: Centaur trained on (Phase 0 only) vs. (Phase 0 + real behavioral) vs. (real behavioral only). Measure on the held-out behavioral test set. The Phase 0 vs. Phase 0+real gap quantifies how much the real data adds; the Phase 0 vs. real-only gap quantifies how much the bridge contributed to warm-start vs. training from scratch on real data.

### 3. Individual vs. Population-Level Prediction

**The question:** the SimBench work predicts population distributions; the commercial system may need individual-level predictions (which specific users to target, not what fraction of the population will convert). Individual-level prediction is a harder problem — it requires modeling heterogeneity within segments, not just segment means.

**Why it matters:** the routing architecture and calibration framework in this document are built around population/segment-level distributions. Extending to individual-level prediction requires either (a) a much larger feature space per individual or (b) a probabilistic model of within-segment heterogeneity.

**Approach:** pilot individual-level prediction on a single segment with rich transaction history. Compare segment-mean prediction vs. individual-level prediction on held-out outcomes. The gap quantifies the value of individual-level modeling and informs whether the investment is warranted for the commercial product.

### 4. Entropy Headroom from Stage 16

Stage 16 mechanism analysis showed that predictions remain ~0.06 too diffuse (predicted entropy 0.76 vs. truth entropy 0.70) and this headroom was not closed by the routing intervention. Concentration error (0.12) dominates location error (0.05) in the remaining TVD. This is an open avenue for improvement.

**Approach:** investigate entropy calibration methods that are conditioned on the task kind rather than applied globally (which Stage 06 showed is ineffective). A task-kind-specific entropy correction (e.g., tighter distributions for knowledge items where ground truth is more peaked; wider for opinion items where heterogeneity is expected) is the natural extension.

### 5. Longer-Horizon Behavioral Dynamics

**The question:** all current architecture is essentially static (predict propensity at a point in time). Commercial value often comes from longer-horizon dynamics: when will a customer churn? how does a customer's purchase probability evolve across a session? what is the 30-day LTV given a first purchase?

**Approach:** this requires sequence modeling rather than distribution-over-options prediction — a qualitative extension of the architecture. The Centaur backbone could in principle handle this if trained on behavioral sequences rather than single events. Longer-horizon modeling is outside the current scope but is the natural Phase 2 research agenda.

---

## Design Decisions and Tradeoffs

**Why universal enrichment, not routing dispatch.** The Part I evidence shows the gain came from better contextual description of the decision environment, not from the classification mechanism. The `task_context` route worked because it gave the model richer environmental framing — sibling survey items as context — not because it dispatched to a different strategy. The dev-to-val reversal on `risky_choice → voting` is the clearest illustration of routing's brittleness: a route that looks good on dev can fail systematically on held-out data, and a commercial system with novel task types would face this failure mode constantly. Universal enrichment — always conditioning on rich task/environment description — is more defensible: it doesn't require a correct taxonomy, doesn't break on unseen task types, and the enrichment configuration itself is the gated learning artifact. Abstain-on-uncertainty remains as a coverage policy for known-unreliable prediction contexts (risky choice, moral dilemma at low n).

**Why the SimBench numbers are framing, not targets.** The commercial problem differs from SimBench in ways that matter structurally: individual-level data vs. population distributions, temporal sequences vs. cross-sectional snapshots, confounded behavioral signals vs. clean survey options, open outcome spaces vs. fixed option sets. A system optimized to maximize SimBench scores would make the wrong tradeoffs for commercial behavioral prediction. The Part I numbers are cited here as proof that the methodology works — routing generalizes, the prediction ledger enforces honest evaluation, gated validation catches overfitting — not as performance targets for the commercial system.

**Why the bridge, not waiting for real data.** Waiting for real behavioral data before starting Centaur training means months of cold-start on a weak prior. The bridge produces a synthetic corpus that is better than nothing, has known provenance, and degrades gracefully as real data accumulates. The cost of the bridge (generating synthetic data) is negligible relative to the cost of deploying a behavioral simulator on a weak prior for months.

**Why not fine-tune per-tenant from day one.** Stage 06 showed that even targeted calibration (14 configurations of calibrators) on well-prompted survey predictions added no value. The LLM's native calibration for its training domain is better than post-hoc correction. For behavioral data, the LLM's native calibration is genuinely poor — but the correction should start with calibration (cheap, low blast radius) and escalate to fine-tuning only when calibration has plateaued. Volume gates enforce this escalation ladder.

**Why shared weights, not shared data.** Per-tenant data cannot cross tenant boundaries (privacy, commercial, legal). Model weights can, because they encode learned structure, not records. This is the same argument that makes foundation model pretraining on internet text legitimate: the weights encode patterns, not individuals. The Centaur approach extends this to behavioral data.

