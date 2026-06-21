# Part II — Feedback Mechanism Architecture

**Date:** 2026-06-21  
**Status:** Design document (no implementation required)  
**Context:** Scrye sports e-commerce simulator receiving real-world outcomes for the first time.

---

## The Scenario

The simulator has been running as a research benchmark (Part I): given a survey question and demographic segment, predict the population-level response distribution. Now the commercial engagement goes live: a sports e-commerce partner begins returning **revealed behavior** (transaction logs) and **stated responses** (new survey waves). The system must compound these signals into continuously improving predictions.

A note from Part I that grounds the design: Stage 06 showed that post-hoc temperature scaling and Dirichlet calibrators did *not* improve the survey distribution predictor — the `anti_flattening` prompt strategy was already well-calibrated on its native domain (stated preferences). This negative result is load-bearing for Part II: the LLM's zero-shot distribution estimates are good when the signal lives in its training distribution (text about attitudes and opinions). The calibration gap opens where it doesn't — revealed economic behavior. That asymmetry drives the entire architecture.

---

## Architecture Overview

```
══════════════════════════════════════════════════════════════════
  DATA INGESTION
══════════════════════════════════════════════════════════════════

  Revealed Behavior          Stated Responses
  (transaction logs)         (survey waves)
       │                           │
       ▼                           ▼
  ┌─────────────────┐       ┌──────────────────┐
  │ Behavioral ETL  │       │  Survey ETL       │
  │ - normalize     │       │  - aggregate to   │
  │   offer context │       │    distributions  │
  │ - flag promotns │       │  - attach quality │
  │ - dedup         │       │    metadata (n,   │
  └────────┬────────┘       │    recency, wave) │
           │                └───────┬────────── ┘
           │                        │
           ▼                        ▼

══════════════════════════════════════════════════════════════════
  COMPOUNDING MEMORY (per-tenant, data-isolated)
══════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────┐
  │  L1 — Raw Outcome Store (append-only, permanent)         │
  │  ├── behavioral_events: {user_id, item_id, action,       │
  │  │                        offer_context, timestamp}       │
  │  └── survey_responses:  {wave_id, item_id, dist,         │
  │                           segment_metadata, n}            │
  └──────────────────────┬───────────────────────────────────┘
                         │ triggers (on schedule or volume)
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  L2 — Prediction Ledger (written at prediction time)     │
  │  ├── prediction:  {id, spec_hash, question, segment,     │
  │  │                  predicted_dist, timestamp}            │
  │  └── outcome:     {prediction_id, ground_truth,          │
  │                    outcome_source}  ← linked on arrival  │
  └──────────────────────┬───────────────────────────────────┘
                         │ aggregated on schedule
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  L3 — Segment Posterior Store                            │
  │  ├── empirical_dists: per-segment × question-family      │
  │  │    (exponential moving window; recent = higher weight) │
  │  └── calibration_residuals: {segment, family,            │
  │         tvd_error, entropy_gap, n_outcomes}              │
  └──────────────────────┬───────────────────────────────────┘
                         │ when update gate passes
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  L4 — Calibrator Cache (versioned, rollback-able)        │
  │  ├── calibrator_params: entropy-conditioned temperature  │
  │  │    + segment shift vectors (per tenant)               │
  │  └── adapter_weights: LoRA weights (only if fine-tune    │
  │       gate has been passed; rare; human-gated)           │
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼

══════════════════════════════════════════════════════════════════
  PREDICTION PIPELINE (at inference time)
══════════════════════════════════════════════════════════════════

  Query arrives: {question, segment, offer_context}
       │
       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  Route by signal type                                      │
  │                                                            │
  │  Stated preference  ──────────────┐                        │
  │  (survey-style question)          │                        │
  │                                   ▼                        │
  │                          ┌─────────────────┐              │
  │                          │ COLD / WARM PATH │              │
  │                          │ ICL + LLM prior  │              │
  │                          │ (already well-   │              │
  │                          │  calibrated;     │              │
  │                          │  Stage 06 result)│              │
  │                          └────────┬─────────┘              │
  │                                   │                        │
  │  Revealed behavior  ──────────────┤                        │
  │  (purchase propensity)            │                        │
  │         │                         │                        │
  │         ▼                         │                        │
  │  ┌──────────────────────┐         │                        │
  │  │  Data volume check   │         │                        │
  │  │  against L3 + L4     │         │                        │
  │  └──────┬───────────────┘         │                        │
  │         │                         │                        │
  │    n < 100 outcomes               │                        │
  │         │                         │                        │
  │         ▼                         │                        │
  │  ┌──────────────────┐             │                        │
  │  │  COLD PATH       │             │                        │
  │  │  Retrieve nearest│             │                        │
  │  │  segment from L3 │             │                        │
  │  │  (k-NN by demo-  │             │                        │
  │  │  graphic profile)│             │                        │
  │  └──────┬───────────┘             │                        │
  │         │                         │                        │
  │   n ≥ 100 outcomes               │                        │
  │         │                         │                        │
  │         ▼                         │                        │
  │  ┌──────────────────┐             │                        │
  │  │  WARM PATH       │             │                        │
  │  │  LLM zero-shot   │             │                        │
  │  │  + L4 calibrator │             │                        │
  │  │  (temp scaling + │             │                        │
  │  │  segment delta)  │             │                        │
  │  └──────┬───────────┘             │                        │
  │         │                         │                        │
  │  n ≥ 10k AND stable               │                        │
  │  AND calibration ceiling          │                        │
  │         │                         │                        │
  │         ▼                         │                        │
  │  ┌──────────────────┐             │                        │
  │  │  HOT PATH        │             │                        │
  │  │  LoRA adapter    │             │                        │
  │  │  (human-gated    │             │                        │
  │  │  ship)           │             │                        │
  │  └──────┬───────────┘             │                        │
  │         └─────────────────────────┘                        │
  └────────────────────────────────────────────────────────────┘
       │
       ▼
  Predicted distribution → log to L2 Prediction Ledger

══════════════════════════════════════════════════════════════════
  UPDATE GATE (before any L4 change ships)
══════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────┐
  │  1. Shadow evaluation (challenger runs in shadow for     │
  │     ≥ 1k predictions or 2 weeks alongside champion)      │
  │                                                          │
  │  2. Lift test on L2 holdout (never used in calibration)  │
  │     Δ TVD must exceed eta = f(bootstrap variance on     │
  │     holdout) — the Blum-Hardt ladder condition           │
  │                                                          │
  │  3. Reliability ceiling check: claimed improvement must  │
  │     not exceed human sampling noise floor (bootstrap CI  │
  │     on the ground-truth distribution's finite-n noise)   │
  │                                                          │
  │  4. Drift check: PSI on input distribution and outcome   │
  │     drift since calibrator was last trained              │
  │                                                          │
  │  5. Fine-tune ships only: human sign-off required        │
  └──────────────────────────────────────────────────────────┘
       │
       ├── PASS → promote challenger to champion, retire old L4
       └── FAIL → keep champion; log challenger result to L2
```

---

## 1. The Compounding Memory

Four tiers, each serving a distinct purpose:

**L1 — Raw Outcome Store (always written, never deleted)**  
The audit log. Every transaction event and every survey wave lands here in its original form, normalized just enough to be query-able. The key invariant: L1 is append-only. It is the source of truth for retrospective backtesting and drift forensics. Nothing is overwritten here — if the segment-posterior or calibration logic later changes, L1 can be reprocessed from scratch.

Behavioral events carry offer context (price, promotion, placement) because the raw purchase signal conflates propensity with opportunity. The ETL strips promotion effects before downstream aggregation — but the original event with the offer context is preserved in L1, so the promotion attribution model can be iterated without losing the raw signal.

**L2 — Prediction Ledger (written at prediction time, outcome linked later)**  
The key abstraction that prevents data snooping. Every prediction is stamped *before* the outcome arrives — question, segment, the predicted distribution, the pipeline version. When the outcome arrives (a transaction, a new survey wave), it is linked to the existing ledger record. This means all calibration errors are computable against predictions made under real conditions, not from retrospective reconstruction. Without this ledger, there is no honest backtesting.

**L3 — Segment Posterior Store (computed periodically from L1 + L2)**  
The living summary of what has been learned. Two components:

- *Empirical distributions*: per-segment, per-question-family aggregate distributions, weighted by an exponential moving average (recent outcomes count more, naturally tracking trend without requiring explicit drift detection in the retrieval path).
- *Calibration residuals*: the structured error signal — how far the LLM's zero-shot prediction is from ground truth, broken down by segment and question family. This is the direct input to the calibrator fitting step.

L3 is what the retrieval/cold path draws from: when we have no outcomes for a new segment, find the nearest segment in L3 by demographic similarity and use its posterior as the prior.

**L4 — Calibrator Cache (updated only when the update gate passes)**  
The operational artifact: versioned, rollback-able. Calibrator parameters (temperature coefficients, segment shift vectors) plus, optionally, LoRA adapter weights. Each version is kept for 30 days after replacement to enable rollback without retraining. Promotion of a new L4 version is gated — never automatic.

---

## 2. Method Spectrum

The ordering from least to most data-hungry reflects a Bayesian intuition: start with the LLM's pretrained prior (strong, free), then move the posterior as evidence accumulates, replacing the prior only when it demonstrably fails and you have enough data to justify the replacement.

| Tier | Mechanism | Data threshold | Who controls |
|---|---|---|---|
| Cold | ICL / retrieval from L3 | n < 100 labeled outcomes | Automated |
| Warm | LLM zero-shot + L4 calibrator | n ≥ 100 | Automated (gated) |
| Warm+ | Segment-conditioned calibration | n ≥ 500 per segment | Automated (gated) |
| Hot | LoRA adapter fine-tuning | n ≥ 10k, stable, ceiling | Human sign-off |

**Cold (ICL/retrieval):** The new-tenant, new-segment default. Retrieve the most demographically similar segment from L3 (k-NN on a small demographic feature space: age band, income band, sport affiliation). Use its empirical posterior as a few-shot anchor in the LLM prompt. This is already what the anti_flattening strategy implicitly does — remind the model that real populations are diverse, not mode-seeking. The cold path makes that anchor explicit and tenant-specific.

**Warm (calibrated zero-shot):** Past 100 labeled outcomes, fit an entropy-conditioned temperature calibrator from L2. The functional form: higher temperature (flatten/spread) on high-entropy predictions, lower on consensus ones. This corrects the systematic mode-seeking failure diagnosed in the SimBench paper. Note: Stage 06 showed this calibration was *not* useful for pure survey distribution prediction (anti_flattening prompting already handles it). But for behavioral outcomes — purchase propensity — the LLM's zero-shot is genuinely out-of-distribution, and a calibration layer is critical.

**Warm+ (segment-conditioned calibration):** With 500+ outcomes per segment, fit separate calibrators per segment rather than a global one. Encode segment effects as shift vectors (delta from the population calibrator) — matching the delta-modeling structure from Part I. This prevents the segment information from distorting population-level accuracy, which the SimBench paper showed is the failure mode of naive persona conditioning.

**Hot (LoRA fine-tuning):** A last resort, not a default. Justified only when: (a) volume exceeds ~10k labeled examples, (b) the calibration layer has plateaued, (c) the zero-shot LLM prior is structurally wrong for this domain, and (d) the update gate passes. The risk is overfitting to one tenant's distribution and losing zero-shot generalization — a serious problem if the tenant's customer base shifts. The human sign-off gate exists because fine-tune ships are expensive to roll back and hard to audit.

---

## 3. Signal Types: Revealed Behavior vs. Stated Preference

These are different epistemic classes. They must be routed to different roles in the architecture — not averaged.

**Revealed behavior (transaction logs) → calibration labels and prediction ledger ground truth**

- Ecological validity: high. This is what people *actually did* — the commercial variable of interest.
- LLM nativeness: low. The LLM was trained on text; economic choices under specific offer conditions are OOD.
- Confounds: high. A purchase is entangled with price, availability, placement, and promotional context. The ETL must isolate propensity from opportunity before using this as a calibration label.
- Role in the architecture: primary ground truth. Every Prediction Ledger entry linked to a behavioral outcome provides a direct training signal for the calibration layer. This is the signal that *tests the simulator's predictions* against reality.

**Stated preferences (survey waves) → persona prior and cold-start conditioning**

- Ecological validity: moderate. Social desirability bias, hypothetical framing, and mode-seeking from respondents.
- LLM nativeness: high. Surveys are exactly what the LLM was trained on — attitudes and opinions in text.
- Role in the architecture: soft prior. New survey waves update L3 posteriors and improve retrieval-path accuracy. They are the *source of conditioning* for the simulator, not the *label* it is validated against.

**Why they must not be averaged:** The value-action gap is well-documented in simulation research (Park et al.: behavioral tasks r ≈ 0.66 vs. attitude tasks r ≈ 0.83). Survey responses systematically overstate socially desirable preferences and understate variance. Averaging stated and revealed signals would corrupt both: the behavioral signal would be diluted by an optimistic prior, and the survey signal would be treated as ground truth for something it was never designed to measure. Route them separately; let each inform what it can.

Practically: for a new segment with no transaction history, the architecture uses survey-derived priors (cold path, ICL from L3 survey posteriors) and flags these predictions as prior-dominated with wider confidence intervals. When behavioral outcomes begin arriving, they gate the warm path and progressively override the survey prior for behavioral predictions — while the survey prior continues to inform pure preference and attitude predictions.

---

## 4. Generalization and Multi-tenancy

**Ground truth from one engagement does not directly improve predictions for another.** This is the honest position. The tempting claim is a "flywheel" where more customers make the system smarter for everyone — but population simulation predicts the behavior of *specific populations*, and those distributions are not exchangeable across customer bases.

**What does not transfer (data-isolated by design):**
- L1: raw outcomes are tenant-specific and subject to data-sharing agreements and privacy law
- L3: segment posteriors encode the specific population served
- L4: calibrators and adapter weights are fit on tenant-specific data

**What does transfer (structural):**
- The inference harness, prediction ledger schema, and evaluation protocol: universal
- The backbone LLM: shared across all tenants (the pretrained prior)
- The calibration *method* (entropy-conditioned temperature): validated on one tenant, available for cold-start on others
- The reliability ceiling computation: the methodology is universal; the ceiling values are tenant-specific
- The update gate logic and thresholds: shared defaults, tuned per tenant over time

**The legitimate cross-tenant benefit:** a new tenant benefits from *method validation*, not data transfer. If entropy-conditioned calibration worked on previous sports retailer clients, that is evidence it will work as the warm-path starting point for a new one. The cold-start calibrator is initialized to parameters that worked elsewhere — not to flat priors. This is the honest flywheel: the method becomes better-validated and faster to initialize. Not: "we know more about your customers because we served other customers."

**The optional shared-prior shard:** if two tenants explicitly consent and their customer populations genuinely overlap (e.g., both serve the same demographic), a shared L3 shard can be constructed. This is gated by explicit agreement, not the default. The architecture supports it as a configuration option; the default is full tenant isolation.

---

## 5. Guardrails and Evaluation

**Preventing overfitting:**

- The calibrator is fit on L2 with leave-one-wave-out cross-validation: outcomes from the most recent survey wave or behavioral batch are held out from calibration and used for validation. The calibrator never sees the holdout during training.
- Volume gates (100 outcomes for warm path, 10k for fine-tuning) prevent calibration when data is too thin to distinguish signal from noise.
- The **reliability ceiling** is the hard upper bound. Ground-truth distributions are finite-sample estimates — there is irreducible noise from the finite n of any survey or transaction batch. Bootstrap CI over this noise gives the ceiling on what *any* predictor can achieve. An update claiming to exceed the ceiling is overfit by definition and is blocked at the gate.
- Ensemble calibrators: fit N=5 calibrators on different time windows (1-month, 3-month, 6-month, all-time, leave-last-wave-out). Ship only if they agree within bootstrap CIs. Disagreement across windows signals overfitting to a specific temporal slice.

**Preventing drift:**

- Input drift detector: Population Stability Index (PSI) on incoming question types, segment frequencies, and offer contexts. Alert if PSI exceeds threshold since last calibrator training.
- Outcome drift detector: monitor empirical purchase rates by segment in a trailing window. Alert if any segment's rate shifts beyond 2σ of the calibrator's training-time baseline.
- Temporal windowing on L3: the segment posterior store uses exponential moving average weights, so recent outcomes count more. The system naturally tracks trend without requiring explicit recalibration on every new batch.

**Validating an update before it ships:**

1. **Shadow evaluation.** The challenger runs in shadow on all live predictions for a burn-in period (≥1,000 predictions or two calendar weeks, whichever is larger). Its outputs are logged to L2 alongside the champion's but do not affect the client response. At burn-in end, L2 contains paired champion and challenger predictions for real outcomes.

2. **Lift test on the holdout slice.** Using the L2 holdout (outcomes not used in calibration), compute TVD for champion and challenger on matched predictions. The challenger must show statistically significant reduction (bootstrap CI) and the raw Δ must exceed η — the noise floor derived from bootstrap variance on the holdout. This is the Blum-Hardt ladder condition: the same principle used in Part I's LadderGate, which bounds generalization error under repeated adaptive evaluation.

3. **Reliability ceiling check.** The claimed improvement Δ must not approach the reliability ceiling from the ground-truth distributions' finite-n noise. If the challenger is claiming improvements near the ceiling, it is fitting noise.

4. **Human sign-off for fine-tune ships.** Calibrator parameter updates are automated once gates 1–3 pass. LoRA adapter ships require human review: the blast radius is larger, rollback requires re-serving the previous adapter, and fine-tunes can encode subtle distributional artifacts not visible in aggregate metrics but apparent on sample inspection.

5. **Rollback contract.** Every L4 version is retained for 30 days after promotion. If production TVD rises by more than 2σ in a trailing window of 500 predictions, the system automatically rolls back and opens an incident. The rollback is deterministic — no human action required to execute it, though human review is required to re-promote.

---

## 6. Extension: Behavioral Foundation Model (Centaur-Style)

The architecture above treats the backbone LLM as fixed — a text-pretrained prior that is either calibrated (warm path) or fine-tuned per-tenant (hot path). A more ambitious extension makes the backbone itself a living component of the feedback loop.

Binz et al. (2024) — *Centaur: A Foundation Model of Human Cognition* — fine-tuned Llama-3.1-70B on the Psych-101 dataset (~60k participants across thousands of cognitive experiments). The resulting model predicted held-out human behavior substantially better than zero-shot, with strong generalization to tasks and populations not present in training. The key result: a foundation model fine-tuned on behavioral data learns structural properties of human decision-making that transfer across domains, not just task-specific patterns.

**The revised stack:**

```
Pretrained LLM (text prior)
    ↓
[existing architecture]

becomes:

Pretrained LLM (text prior)
    ↓
Behavioral foundation fine-tune   ← cross-tenant, anonymized aggregate distributions
    ↓                                 updated periodically; validated before shipping
Per-tenant calibration (L4 warm/hot path)
    ↓
Prediction
```

**What this changes and why it is defensible:**

Section 4 argued that shared data across tenants is not a flywheel, because population-level behavioral distributions are not exchangeable across customer bases. That claim holds for *raw data*. Model weights are a different matter: if you aggregate anonymized behavioral patterns (purchase propensity distributions by demographic profile × product category × offer context) into fine-tuning data, the knowledge is encoded in weights — not in individual records. The same logic that makes Centaur privacy-defensible applies: what transfers is learned structure of human decision-making, not who bought what.

This rehabilitates the multi-tenancy argument in a principled way. The flywheel is real, but it operates at the model-weight level rather than the data level:

- *What transfers:* learned priors over how demographic segments respond to offers, how entropy varies across product types, how context modulates propensity — the structural features of behavioral distributions.
- *What does not transfer:* the specific joint distribution of any one tenant's customer base, which remains the domain of the per-tenant calibration layer.

The behavioral foundation model improves the cold path materially: even a new tenant with zero outcomes benefits from a backbone that already knows something about how humans make purchasing decisions, not merely what they write about purchasing. It also reduces the work the warm-path calibrator needs to do — the prior is better, so the residual to correct is smaller.

**Incremental updating as data accumulates:**

The behavioral foundation model is not trained once. It is periodically retrained on the growing cross-tenant corpus, with increasingly fine-grained behavioral segmentation as data density allows — finer product categories, more nuanced demographic intersections, recency weighting on recent behavioral waves. At low data volumes the fine-tuning is coarse (broad behavioral priors). As the corpus grows, the fine-tuning captures more specific behavioral structure. This is the legitimate version of "the system gets smarter over time" — grounded in weight updates on aggregate data, not raw customer pooling.

Each updated behavioral foundation model is validated through the same update gate as L4 changes: shadow evaluation, lift test with the Blum-Hardt η condition, reliability ceiling check, and human sign-off (a foundation model update has a larger blast radius than a per-tenant calibrator).

**What is genuinely unknown (empirical test required):**

Centaur's generalization holds across *cognitive tasks* — memory, reasoning, choice under uncertainty, perceptual judgment. Whether that generalization extends to *commercial behavioral data* is untested. The domain gap may be substantial: Psych-101 is controlled laboratory experiments with clean stimuli; transaction logs are messy, confounded by price dynamics and promotion effects, and ecologically very different from a psychology study.

Two specific risks:
1. **Domain mismatch:** behavioral fine-tuning on sports e-commerce data may not improve predictions for, say, media consumption or financial products. The cross-domain generalization claim must be tested, not assumed.
2. **Confound absorption:** if the fine-tuning corpus is not rigorously promotion-normalized (§3 ETL step), the behavioral foundation model learns "people buy more when there are discounts" rather than genuine preference structure. The anonymized fine-tuning data must be conditioned on offer context before aggregation, for the same reason as the per-tenant calibration labels.

The defensible posture: propose the Centaur-style layer as the architecture, implement the cross-tenant fine-tuning in parallel with per-tenant calibration, and run a held-out generalization test across at least two distinct engagement types before claiming cross-domain transfer. The test is cheap (the foundation model is already being trained); the claim is only made if the data supports it.

---

## Design Decisions and Tradeoffs

**Why not fine-tune first?** Stage 06 showed the LLM with good prompting is already well-calibrated for its native domain (stated preferences). For behavioral outcomes, the value-action gap makes the LLM's zero-shot prior imprecise but not useless — still better than no prior. A calibration layer correcting for systematic OOD error is lower-risk and lower-cost than a fine-tune that replaces the prior. Fine-tuning is reserved for when calibration plateaus, which requires data volumes most tenants will not reach for months.

**Why the Prediction Ledger must exist.** Without an explicit record of what was predicted before outcomes arrived, it is impossible to compute calibration errors honestly. All retrospective analyses would be subject to hindsight contamination — effectively, data snooping on the outcome. The ledger makes the prediction-outcome link explicit and auditable, enforcing the same leakage discipline as the Part I dev/val/test protocol.

**Why shared data across tenants is not a flywheel, but shared weights can be.** Population-level behavioral distributions are not exchangeable across customer bases — raw data pooling conflates "more training examples" with "more training examples about *you*." But a behavioral foundation model (§6) encodes cross-tenant learning in weights rather than data, which is the legitimate flywheel: the backbone gets better at predicting human behavior in general, while the per-tenant calibration layer handles what is specific to each customer base.

**Why the reliability ceiling is the hard stop.** Any metric can be gamed under adaptive evaluation. The reliability ceiling — irreducible noise from finite-n sampling — is the one bound that cannot be gamed: it is a property of the data-generating process, not the model. Using it as a hard gate ensures the system does not mistake sampling noise for model improvement, which is the central risk of an adaptive evaluation loop.
