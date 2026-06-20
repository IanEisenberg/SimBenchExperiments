# Research Overview: Behavioral Simulation for Scrye

*A literature map and hypothesis set to ground the [simulation brief](./scrye_simulation_brief.md). Its primary job is to fix one gap the brief had — it never incorporated **SimBench** (Hu et al., ICLR 2026), the field's first standardized benchmark — and to use SimBench to sharpen, correct, and stress-test the brief's thesis.*

*Compiled 2026-06-13. Sources: the five core papers in `docs/papers/` + the brief's two anchor papers, plus ~25 supporting papers from a structured web sweep (cited inline).*

---

## 0. TL;DR — what changed after reading SimBench

The brief got the **big picture right**: attitudes are a solved regime (~0.83 normalized), behavior-under-stakes is the open bet, the reliability-denominator is the field's evaluation backbone, personalization is modest, and direction generalizes while magnitude inflates. SimBench independently corroborates all of this. But it **corrects three things and adds one mechanism the brief missed**:

1. **"The model is a commodity layer" is half-wrong.** SimBench shows capable models matter a lot (best model 40.80/100; *ten of 45 models score below a random baseline*), and simulation ability correlates almost perfectly with knowledge-reasoning (MMLU-Pro **r=0.94**). More importantly, the *kind* of post-training matters: there is an **alignment–simulation tradeoff** (below) that turns model choice from a commodity into a genuine technical lever. → This is an *opportunity* the brief left on the table.
2. **Personalization isn't just "modest" — group-level demographic conditioning can be net-negative.** On SimBenchGrouped, conditioning on a demographic persona *degrades* simulation accuracy across **every** category tested (ΔS from −1.27 to −9.91). This strongly reinforces — and arguably mandates — the brief's "persona vectors as a validity probe, not the engine" reframe.
3. **The attitude-vs-behavior gap is now benchmark-confirmed, not a Park artifact.** The brief's single load-bearing risk rested on Park's 5 broken economic games (n.s.). SimBench reproduces the same gap across 20 datasets: models are best on attitudes/self-assessment, systematically worse on *behavioral choice* (Choices13k, MoralMachine) — the "value–action gap." The risk is real and external.
4. **New mechanism — the alignment–simulation tradeoff.** Instruction tuning helps on consensus (low-entropy) questions but *actively hurts* on diverse (high-entropy) ones (**r=−0.942**); past entropy ≈0.8 an aligned model is a *worse* simulator than its own base model. The mechanistic story (mode-seeking vs. mass-covering KL) unifies magnitude inflation, homogenization, and the behavior gap as facets of one thing: **miscalibration**. The brief treated these as separate observations; they are one.

**Net:** the brief's architecture (telemetry classifier zoo + cold-start LLM + governance instrumentation) survives intact and is *strengthened*. The main revisions: (a) treat distribution-preserving model engineering as a real moat, not a commodity; (b) elevate the persona-as-probe idea from "wedge" to load-bearing, because the benchmark says naive conditioning backfires; (c) make the value–action gap the explicit central hypothesis to test against RedBird telemetry.

---

## 1. The field is young enough to map almost completely

This literature is ~3 years old: Argyle et al.'s "silicon sampling" (2023) → SimBench (2026). It is small enough that the map below is close to comprehensive for the question Scrye cares about. The honest summary:

- **Top-left of the evidence map (cheap *or* expensive data → stated attitudes) is crowded and "solved."**
- **The whole field shares one methodological spine** (normalize against a baseline; predict *distributions*, not points).
- **The bottom-right cell — accumulated behavioral data predicting what people *do* under stakes — is genuinely empty.** No paper tests it. Scrye's telemetry position is the only thing that *could* fill it, and that is both the opportunity and the unvalidated risk.

|                              | Stated attitudes (regime A)        | Behavior under stakes / revealed (regime B) |
|------------------------------|------------------------------------|---------------------------------------------|
| **Expensive bespoke data**   | Park ~0.83 ✓                        | Park ~0.66, std 2.83, n.s. ✗                 |
| **Cheap pre-existing panel** | Kinzinger 0.79 / r=0.59 ✓           | **untested by anyone**                       |
| **Benchmark consensus**      | SimBench: attitudes are the easy tasks ✓ | SimBench: behavioral-choice tasks are the hard tasks ✗ |

---

## 2. Paper-by-paper: what each establishes, and what it does *not*

### 2.1 SimBench (Hu, Baumann, Lupo, Collier, Hovy, Röttger — Cambridge/Stanford/Bocconi/Oxford). ICLR 2026, arXiv:2510.17516. **[The gap the brief missed]**

**What it is.** The first large-scale standardized benchmark for *group-level* human-behavior simulation. 20 harmonized datasets (moral dilemmas, economic/risky choice, psychometrics, opinion surveys), >130 countries, 45 LLMs (0.5B–405B, base + instruct, open + closed). 10,930,271 question–group targets; two evaluation splits: **SimBenchPop** (7,167 cases, broad population) and **SimBenchGrouped** (6,343 cases, demographic conditioning over 5 large survey datasets).

**Metric.** SimBench score `S = 100·(1 − TVD(P,Q)/TVD(P,U))` — Total Variation Distance between the model's predicted response *distribution* Q and the human ground-truth distribution P, normalized against a uniform baseline U. 100 = perfect, 0 = no better than random. **Crucially this is a *distributional* metric** (does the model reproduce the full spread of human responses), not point accuracy — which is exactly the right target for Scrye, where variance preservation is the whole game.

**Headline findings (treat as the strongest evidence in the field):**
- **Modest ceiling for everyone.** Best model = **Claude-3.7-Sonnet, 40.80/100**. Top open-weights = DeepSeek-R1, 34.52. The best models close only ~40% of the gap to ground truth. **Ten of 45 models score below 0** — worse than uniform random.
- **Scaling: yes with size, no with thinking.** Log-linear improvement with parameter count. But **inference-time compute does nothing** (o4-mini low→high: 27.77→28.99; CoT *hurts* GPT-4.1: 34.55→33.11). Reasoning/deliberation is the wrong tool — it forces over-rational deliberation onto heuristic human responses.
- **The alignment–simulation tradeoff (the key new result).** Binning questions by human-response entropy, the gain from instruction tuning vs. its base model is a near-perfect *negative* line (**r=−0.942**): post-training helps up to +40 S-points on consensus questions but crosses into *harm* past entropy ≈0.8. Mechanism (formalized via RL-as-inference): pretraining minimizes *mass-covering* KL (covers all human modes); RLHF minimizes *mode-seeking* KL (collapses to the single "best" mode). Causal mediation: instruction tuning has a **+6.46 direct effect** (instruction-following) and a **−1.74 indirect effect** via entropy reduction.
- **Behavior is harder than attitudes (confirms the brief's central risk).** Best on opinion/self-assessment (OpinionQA, Afrobarometer); degrades on *behavioral choice* (Choices13k risky choice, MoralMachine) — explicitly the "value–action gap" (cites Shen et al. 2025).
- **Atypical perspectives are hardest.** Worst on Machiavellianism, conspiratorial beliefs, humor (Jester) — often below uniform. Alignment filters suppress counter-normative simulation.
- **Demographic conditioning backfires.** On SimBenchGrouped, ΔS (grouped − ungrouped) is **negative for every category**: religiosity/practice −9.91, political ideology −4.97, religion −4.83 … gender −1.24, age −1.50. Asking the model to "be" a specific group makes it *worse*, not better.
- **Simulation ≈ knowledge-reasoning.** Correlates with MMLU-Pro **r=0.94**, GPQA r=0.86; weaker with general chat (Chatbot Arena r=0.71) and narrow math (AIME r=0.48). Neither conversational polish nor problem-solving is sufficient.
- **A constructive nuance:** specialist *cognitive* tuning (Centaur, Binz et al. 2025) preserves distributional fidelity better path-for-path than general instruction tuning at matched scale — evidence that distribution-preserving post-training is achievable.

**What it does NOT establish.** It is **group-level, static, and non-interactive** (authors flag all three in Appendix A). It predicts *response distributions for a demographic group*, not individual trajectories, not behavior over time, not multi-turn dynamics. It is not a behavior-under-stakes-with-real-consequences test (the games are still stated choices). So it sharpens the brief's risk but does not resolve the empty cell.

**Why its bibliography matters to Scrye.** SimBench's reference list is, in effect, a pre-curated reading list of the failure-mode literature (value–action gap, persona effect, miscalibration, distributional alignment, diversity collapse). The "papers to acquire" section below is drawn largely from it.

### 2.2 Park et al. 2024, "Generative Agent Simulations of 1,000 People" (arXiv:2411.10109) — *the brief's bedrock; unchanged*

N=1,052 person-specific agents from 2-hour AI interviews. Normalized accuracy = agent ÷ participant's own 2-week test–retest consistency. Interview agents: **0.83 normalized GSS** (0.86 +survey) vs. 0.74 demographic / 0.71 persona. Economic games (the only behavior-under-stakes probe): **~0.66, std 2.83, non-significant**; reproduces sign/rank-order (r=0.91–0.99) but **inflates magnitude**. SimBench generalizes the games result: behavior really is the hard regime.

### 2.3 Kinzinger & Hartmann 2026, "Synthetic Personalities" (arXiv:2606.04592) — *the cheap-data validator; unchanged*

Twins from pre-existing heterogeneous panel data (SOEP stand-in). 60-cell grid, 2.1M+ scored responses. Best cell **0.788 accuracy, r=0.590** — matches bespoke-instrument literature from data that costs nothing. Entropy-ranked item selection ≈ random (p≈0.32): **volume drives quality, cleverness doesn't.** Never tested behavior. SimBench is consistent: model size/data scale help; construction tricks don't.

### 2.4 Park et al. 2023, "Generative Agents" (UIST) — *generative, not predictive*

The memory-stream + reflection + planning architecture (recency·importance·relevance retrieval; reflection trigger; recursive planning). **Its dependent variable is *believability*, ranked by human observers (TrueSkill), plus descriptive emergent-behavior demos — never predictive accuracy against what a real person did.** Headline d=8.16 vs. prior SOTA is a *believability* effect. Failure modes (hallucination/embellishment ~1.3%; retrieval misses; degradation as memory grows) map exactly onto what would corrupt a *prediction*. **Strategic read: this is scaffolding, not evidence. Citing it as proof simulation predicts behavior is a category error** — believable ≠ calibrated. Scrye's thesis lives in the gap this paper explicitly left open.

### 2.5 AgentSociety (Piao et al., Tsinghua; SSRN preprint, DeepSeek-V3) — *scale demo, weak validation*

>10,000 agents, ~5M interactions; psychological "mind" (emotion/needs/cognition) + real environment (OSM/SafeGraph/Census) + Ray/MQTT engine scaling to 1M agents. **But validation is overwhelmingly face-validity and engineering throughput.** Only one of four showcases touches real data (Hurricane Dorian mobility), and that is *aggregate, qualitative line-chart overlap* with author-admitted magnitude/timing errors *at the peak-stakes moment*. **Zero held-out predictive-accuracy metrics against individual outcomes.** Useful as a *reference design* for telemetry-rich simulation at scale; useless as evidence of predictive validity. Cautionary data point: even the best case degrades hardest exactly when stakes spike.

---

## 3. Settled findings (treat as ground truth across labs)

1. **Attitudes are a solved regime.** Park 0.83, Kinzinger 0.79, SimBench's easy tasks. Three independent labs, two data regimes.
2. **The evaluation backbone is baseline-normalization.** Park: test–retest ceiling. Kinzinger: empty-persona floor. SimBench: uniform-random baseline. *Same spirit — raw accuracy is meaningless without subtracting a baseline.* (Nuance: Park/Kinzinger normalize against a **human** floor/ceiling; SimBench against uniform random — a skill-above-chance normalizer, not a human-noise ceiling. For a calibration-credible product, prefer the human-reliability denominator; SimBench's is the weaker of the two.)
3. **Predict *distributions*, not points.** SimBench's TVD metric is the field's most honest target and the one that *exposes* variance collapse. Scrye should adopt distributional fidelity, not point accuracy, as a primary metric.
4. **Direction generalizes; magnitude inflates — robustly, across literatures.** Park (sign r=0.91–0.99, magnitude off); Bisbee et al. 2024 (**~7× polarization inflation**, *Political Analysis*); the 156-experiment replication (Nature Comp. Sci. 2025: effect sizes **2–3× larger**, significant effects on **68–83%** of true nulls); Mei et al. 2024 PNAS (systematic **prosocial bias** in economic games); a social-preference benchmark (overestimates altruism, underestimates self-interest). **For a product selling "X% better," magnitude inflation is the dangerous half — clients A/B test and notice.**
5. **Grounding in real behavioral traces measurably helps.** "Beyond Believability" (arXiv:2503.20749): fine-tuning on real behavior improves *action-trace* accuracy ~21–25% over prompt-only agents. This is the constructive counterpoint and the empirical kernel of Scrye's telemetry thesis.
6. **Alignment is mechanistically at odds with diversity.** Mode-seeking RLHF → confident low-entropy outputs → variance collapse. SimBench (r=−0.942) + Tjuatja et al. 2024 (RLHF models fail to reproduce human response biases) + Park et al. 2024b (diminished diversity-of-thought).

---

## 4. Open / contested / unresolved

- **Is "model = commodity" true?** *Contested.* Park found model tier barely moves attitude accuracy; SimBench found a 40-point spread and r=0.94 with MMLU-Pro. Reconciliation: above a capability threshold frontier models cluster on *attitudes*, but the *floor is high* (weak models are worse than random) and the *alignment dimension* is a real lever. → Treat model/post-training choice as a **moat candidate**, not a settled commodity.
- **Do response biases transfer?** *Unsettled.* Tjuatja 2024 (TACL): RLHF models *don't* reliably show human response biases. Newer work (arXiv:2507.07188, 2509.08480) partially contests this. Live question for survey-style use.
- **Does steering/persona-conditioning help or hurt?** *Leaning hurt.* SimBench: group conditioning degrades. Hu & Collier 2024 ("Quantifying the persona effect") and the flattening literature (Wang/Morgenstern/Dickerson, Nature MI 2025) agree naive personas caricature. → Supports persona-as-probe over persona-as-engine.
- **The empty cell: does accumulated behavioral data predict staked behavior?** *Untested by anyone.* This is Scrye's bet and its largest risk.

---

## 5. The empty cell, restated with SimBench context

Scrye sells prediction of what audiences *do*: finish, re-watch, churn, share, pay. The combined literature now says, with more force than the brief had:

- Stated content *preferences/attitudes* → likely the ~0.83 regime. High confidence, say so loudly.
- *Behavioral* content outcomes under real stakes → SimBench's *hard* regime (value–action gap), Park's broken-games regime. **The benchmark evidence predicts this is closer to ~0.66 than ~0.83.**
- For genuinely *novel* content (cold start, no telemetry): simulation ability tracks knowledge-reasoning (MMLU-Pro r=0.94), so semantic extrapolation has *some* footing — but with no labels, magnitude is unknowable. Stays the lowest-confidence tier.

The honest position: **the brief's load-bearing empirical question is now better-supported as a real risk.** That is good news for product credibility — it makes "refuse the magnitude, ship the calibrated rank" the evidence-backed default rather than a hedge.

---

## 6. Hypotheses Scrye should pursue (testable against RedBird telemetry)

Ordered by leverage. Each is framed to be falsifiable on logged behavior.

**H1 — The value–action gap holds for content (the load-bearing test).**
*Content-engagement behavior (finish/re-watch/churn/pay) is predicted markedly worse than stated content preference, with the gap resembling SimBench's attitude→behavior drop.*
Test: pair stated-preference prediction vs. revealed-behavior prediction on the same titles/audiences; measure both in normalized/distributional terms. If the gap is large → validates the tiered-confidence architecture and the magnitude refusal. If small → Scrye's defensible surface is bigger than the literature implies (upside).
Evidence base: Park games, SimBench behavioral-choice, Shen et al. value–action gap.

**H2 — Telemetry-grounded fine-tuning closes the behavior gap where prompting cannot.**
*A classifier/model fine-tuned on RedBird behavioral traces beats prompt-only generative agents on held-out action prediction by a margin comparable to the ~21–25% "Beyond Believability" result.*
Test: prompt-only agent vs. telemetry-tuned model on held-out completion/churn. This is the core constructive bet and the empirical justification for tier-1 of the architecture.

**H3 — Distribution-preserving model engineering is a real moat (model ≠ commodity).**
*On the high-entropy / heterogeneous slices of audience prediction, base or distribution-preserving models (model-merged or contrastive-decoded) outperform aligned frontier models — i.e., the alignment–simulation tradeoff shows up in content data.*
Test: base vs. instruct vs. merged on high- vs. low-entropy audience segments. Evidence: SimBench r=−0.942; Hu/Minixhofer/Collier model-merging; Dong et al. contrastive system-prompt control; Centaur.

**H4 — Persona/control vectors are better validity *probes* than prediction *engines*.**
*A "stereotype-direction" projection (representation-engineering style) predicts *when* an agent's output is miscalibrated (collapsed toward a demographic prior) better than persona conditioning improves accuracy — consistent with SimBench's finding that conditioning degrades.*
Test: regress prediction error on the magnitude of an agent's projection onto extracted stereotype directions; compare against the accuracy delta from naive persona conditioning. This operationalizes Ian's persona-vectors side-project as the governance layer's instrument. Evidence: Hu & Collier persona effect; Anthropic persona vectors; Zou RepE; Wang/Morgenstern flattening.

**H5 — Rank-order is shippable; magnitude is not.**
*Across held-out RedBird A/B tests, simulated rankings of content correlate strongly with realized rankings (high rank-order r) while absolute magnitude predictions are systematically inflated and poorly calibrated.*
Test: rank-order correlation vs. magnitude error on past A/B tests. If confirmed (the literature strongly predicts it will), the product spec writes itself: **calibrated rank + explicit confidence tier + refused point-magnitude.**

**H6 — Cold-start semantic extrapolation has measurable-but-bounded signal.**
*For novel content with no logged history, an LLM placing the item semantically against telemetry-anchored neighbors beats chance but cannot be magnitude-calibrated without labels.*
Test: hold out whole content clusters; measure rank recovery from semantic placement alone. Sets the honest ceiling for the cold-start tier.

---

## 7. Papers to acquire next (prioritized)

**Tier 1 — directly informs H1–H5, get first:**
- **Shen, Clark, Mitra 2025, "Mind the value–action gap: Do LLMs act in alignment with their values?"** EMNLP 2025. → the behavior-gap mechanism behind H1.
- **"Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs"** arXiv:2503.20749. → the telemetry-tuning result behind H2 (the constructive core).
- **Hu & Collier 2025, "iNews: a multimodal dataset for modeling personalized affective responses to news."** ACL 2025. → **closest published analog to Scrye's content-response problem** (personalized affective response to media). High priority — possibly a direct methodological template and a baseline to beat.
- **Hu & Collier 2024, "Quantifying the persona effect in LLM simulations."** ACL 2024. → quantifies exactly the persona signal H4 reframes.
- **Hu, Minixhofer, Collier 2025, "Navigating the alignment–calibration trade-off via model merging."** arXiv:2510.17426. → the distribution-preserving moat for H3.

**Tier 2 — sharpens the magnitude/calibration story and the governance frame:**
- **Mei, Xie, Yuan, Jackson 2024, PNAS, "A Turing test of whether AI chatbots are behaviorally similar to humans."** → strongest behavior-under-stakes (economic games) evidence; benchmarks GPT-4 vs. ~88k humans, finds systematic prosocial bias.
- **"Can LLMs Replace Human Subjects?"** (156-experiment replication; Nature Comp. Sci. 2025, arXiv:2409.00128). → the cleanest hard numbers on magnitude inflation + false positives.
- **Hu, Zhang, Hovy, Collier 2026, "LLM failures from hallucination to homogenization are different facets of miscalibration."** → the unifying frame that ties H3/H4/H5 together; likely the conceptual spine for the governance layer's narrative.
- **Meister, Guestrin, Hashimoto 2025, "Benchmarking distributional alignment of LLMs."** NAACL 2025. → distributional-metric methodology to adopt alongside SimBench's TVD.
- **Liu et al. 2025a, "LLMs assume people are more rational than we really are."** ICLR 2025. → the hyper-rationality bias; relevant to why staked/heuristic behavior is mispredicted.

**Tier 3 — instruments and incumbent baselines:**
- **Anthropic 2025, "Persona Vectors"** (arXiv:2507.21509) + **Zou et al. 2023, "Representation Engineering"** (arXiv:2310.01405). → the activation-steering machinery for H4's validity probe.
- **Dong et al. 2026, "Steer model beyond assistant: controlling system-prompt strength via contrastive decoding"** (arXiv:2601.06403). → situational conditioning without retraining (the one place vectors touch the hard problem).
- **Wang, Morgenstern, Dickerson 2025, Nature MI, "LLMs that replace human participants can harmfully misportray and flatten identity groups"** + **Bisbee et al. 2024, "The Perils of LLMs"** (Political Analysis). → the caricature/flattening + 7× inflation citations for the governance argument.
- **Incumbent recommender baselines (for positioning the classifier-zoo tier):** Zhao et al. 2019 (YouTube multitask ranking, RecSys); Ma et al. 2018 (MMoE, KDD); CREAD (watch-time as ordinal classification, WWW 2024). → frames what the non-LLM telemetry tier already does well and where it's brittle (cold-start, no counterfactual reasoning).

---

## 8. Implications for the brief's architecture

The three-tier architecture **stands**, with these adjustments:

- **Tier 1 (classifier zoo over telemetry):** unchanged and validated by "Beyond Believability." Adopt **distributional** evaluation (TVD-style), not point accuracy. Test H2/H5 here first — it's where calibrated prediction lives and where RedBird's data is densest.
- **Tier 2 (cold-start LLM):** unchanged. SimBench's MMLU-Pro correlation (r=0.94) says the LLM's comparative advantage really is knowledge-grounded semantic extrapolation. But **do not buy "more reasoning helps"** — SimBench shows CoT/inference-compute don't improve simulation. Cache-and-route, don't deliberate.
- **Tier 3 (simulated agents):** lowest confidence, unchanged — but now with benchmark backing that this regime tops out around the 0.66/40-point ceiling. Surface the tier loudly.
- **Governance / instrumentation layer (the wedge):** *upgraded from "wedge" to load-bearing.* Two SimBench findings make it non-optional: (a) demographic conditioning degrades accuracy, so an automated **collapse/caricature detector** (H4) is needed to know when *not* to trust a conditioned agent; (b) the alignment–simulation tradeoff means the governance layer must also choose *which model/post-training* to trust per entropy regime (H3). Frame the layer as **calibration governance**, anchored by a **human-reliability denominator** (stronger than SimBench's uniform baseline) and the miscalibration lens (Hu et al. 2026).
- **The product promise:** lead with **revealed behavior + validation as the spine** (H1, H2, H5). Deliver **calibrated rank + explicit confidence tier + refused point-magnitude.** Bring the adaptive-probe / control-vector depth (H4, H6) only for the "how to outfit it" / situational question. Depth on tap, not on transmit.

---

## 9. The one-line thesis, re-grounded

*The field has solved attitudes and left behavior-under-stakes empty; SimBench now proves the gap is real and structural (mode-seeking alignment collapses the very diversity that staked behavior requires). Scrye's defensible position is not a better simulator — it is the **calibration-governance layer** that knows which tier to trust, refuses the magnitude it can't justify, and is the only thing in the stack grounded in revealed behavioral telemetry the literature never had.*
