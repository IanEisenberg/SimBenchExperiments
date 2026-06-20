# Behavioral Simulation: Evidence, Architecture, and the Scrye Thesis

A synthesis of two foundational papers and the architecture and strategy they imply for a vertical, telemetry-rich simulation company inside RedBird's network.

---

## The two papers

**Park et al. (2026), "LLM Agents Grounded in Self-Reports" (arXiv 2411.10109).** The Stanford team (Park, Bernstein, Liang) — Simile's scientific bedrock. Person-specific generative agents for a stratified US sample (N=1,052), each built from a two-hour AI-conducted voice interview (~6,500 words) plus the GSS, BFI-44, five incentivized economic games, and five experiment replications. The headline metric is **normalized accuracy = agent accuracy ÷ the participant's own two-week test–retest consistency** — i.e., self-consistency as the reliability ceiling. Interview agents hit 0.83 normalized on the GSS (0.86 with survey added) vs. 0.74 demographic and 0.71 persona baselines.

**Kinzinger & Hartmann (2026), "Synthetic Personalities" (arXiv 2606.04592).** TUM team. Asks whether twins can be built from *pre-existing heterogeneous panel data* (the messy CRM/loyalty/repeat-survey data firms already hold) rather than purpose-built instruments. Uses the German Socio-Economic Panel as the stand-in. A 60-cell grid (3 open-weights models × 5 information depths × 2 embeddings × 2 reasoning modes) over 2.1M+ scored responses, on three metrics: accuracy, rank-order correlation, dispersion fidelity. Best cell 0.788 accuracy, r=0.590 — matching or beating the bespoke-data literature from data that costs nothing to acquire.

## The evidence map

|                              | Stated attitudes              | Behavior under stakes (revealed) |
|------------------------------|-------------------------------|----------------------------------|
| **Expensive bespoke data**   | Park ~0.83 ✓                   | Park ~0.66, std 2.83, n.s. ✗      |
| **Cheap pre-existing panel** | Kinzinger 0.79 / r=0.59 ✓       | **untested by either paper**      |

Both papers crowd into the top-left. The bottom-right cell — cheap accumulated data predicting what people actually *do* under stakes — is empty in the literature, and it is exactly Scrye's bet.

## What the two papers jointly establish (treat as settled)

- **The reliability denominator is the field's convergent standard.** Park normalizes against test–retest consistency; Kinzinger uses an empty-persona floor. Same logic, two independent labs: raw accuracy is meaningless without subtracting the noisy-target / generic-respondent baseline. This is the construct-reliability framework Ian published in PNAS — not adjacent to his background, the literal center of how this field evaluates itself.
- **The model is a commodity layer.** Park: model tier barely moves results (GPT-5/o3/o1 ≈ GPT-4o). Kinzinger: 14–30B open-weights models hit 0.788. Moat is not the model.
- **The expensive interview moat is unnecessary.** Cheap accumulated data matches bespoke instruments. This validates the vertical/telemetry thesis *and* dissolves Simile's apparent data moat (their bedrock paper is the expensive one).
- **The genuine personalization signal is modest.** Park: ~9–12 normalized points over demographics, against an ~80% human ceiling. Kinzinger: ~12 accuracy points over a 0.65 baseline. Useful, not magical.
- **Construction cleverness doesn't pay.** Kinzinger's entropy-ranked item selection was statistically indistinguishable from random (p≈0.32); volume drives quality. Preserve raw response granularity (dialog > lossy summary on scaled items); don't over-engineer selection.

## The live risk

Every headline number in both papers is the **attitude** regime. The only probe of **behavior under stakes** — Park's economic games — broke (~0.66, std 2.83, non-significant across architectures, public-goods went negative). The cheap-data paper offers no cover; it never tested behavior. Scrye sells prediction of what audiences *do* — finish, re-watch, churn, share, pay. Whether "content enjoyment" behaves like the GSS (~0.83) or like the games (~0.66) is the single load-bearing empirical question for the company, and the combined literature warns it could be the latter.

A second, commercially specific failure: Park's agents reproduce the **sign and rank-order** of effects (r=0.91–0.99) but systematically **inflate magnitude**. Direction generalizes; calibrated magnitude does not. For a product selling "this content will outperform that by X%," magnitude inflation is the dangerous half, because clients A/B test and notice.

## The architecture that emerged

A confidence-tiered system, with the tier always known and surfaced:

1. **Classifier zoo over revealed-behavior telemetry** — self-maintaining on streaming data. Handles the dense, in-distribution regime. This is where calibrated prediction lives; RedBird's logged behavior (completion, re-watch, churn, second-screen, fantasy/betting engagement) is the substrate neither paper used.
2. **LLM for cold start and semantic extrapolation** — novel content with no logged history. On query, the LLM searches for trainable data; if found, it builds/maintains a cached classifier and calls it as a tool; if not, it falls back to the simulated agent. The LLM's comparative advantage is semantic generalization to genuinely new items — the regime a tree model can't touch.
3. **Simulated agents as lowest-confidence fallback** — dispositional priors when no behavioral data exists.

Two disciplines on the build-classifiers-on-demand idea: it must *cache and route* (a model library grown offline), not train per query — 2.1M responses cannot be 2.1M training jobs. And it does **not** fill the empty cell: a supervised model needs labels, and the novel-content regime has none, so dynamic builders enrich the furnished rooms and leave the empty one empty.

## The wedge — the part that is distinctively Ian's

Across all tiers sits a **mechanistic instrumentation-and-governance layer**: the thing that decides what the system is allowed to build, trust, and ship, anchored by a fixed human-designed reliability harness (the test–retest / empty-persona ceiling). This is the layer no ML-native founder builds and the one the whole system's trustworthiness depends on.

- **Persona / control vectors belong here as instruments, not as the engine.** Steering a base model into "being a fan" is the weak, cute version — coarse, and it courts the exact stereotyping/diversity-collapse failures the field is escaping. The strong version: control vectors as a *validity probe* (detect mechanistically when an agent collapses toward a stereotype prior vs. conditions on individual data — a richer empty-persona ablation, and squarely RSA/representational-geometry work), as a *diversity-preservation control*, and speculatively as a *situational-context lever*.
- This reframes Ian's persona-vectors/valence side project from "can I steer a persona" to "can I detect and quantify when a simulated agent distorts" — more novel, and directly load-bearing.

The test for whether this is a mandate or a hobby: **would a Scrye customer pay for trustworthiness instrumentation, or just want the number?** If they'd pay, it's a defensible CAO mandate.

## Three speculative enrichments — all low-probability, second-order

The core (classifier zoo + cold-start LLM + governance instrumentation) is settled. These are live options for the second-order edge once the core works:

- **Deep phenotyping / richer self-report.** Bounded value, bound unknown. Jumps tiers only if shown to close the *behavior* gap rather than the solved attitude gap — untested.
- **Control vectors for situational conditioning.** The one place vectors touch the hard problem (injecting situation without retraining) rather than the solved one.
- **Adaptive expert interviewing.** Intellectually the most interesting idea here, and strategically the least aligned: it's slow and costly per head — a play for telemetry-*poor* domains, the opposite of Scrye's position.

## The generation–prediction synthesis

An adaptive expert interview is the same operation whether the subject is a human or a simulation: you don't query a high-dimensional preference model everywhere, you use a skilled adaptive probe to *find the directions worth evaluating*. The interview script is as important for extracting signal *out of* a good simulation as for getting data *into* it. This resolves the qualitative-research tension: its generative value (surfacing the unknown unknown) survives compression not as a prediction feature but as a **search policy over the simulation** — generation proposes the directions, prediction scores the ones with ground truth.

Caveat that keeps it honest: even a brilliant interview can't extract a fact the respondent doesn't have access to, and staked behavior is situationally contingent in ways people can't self-report. Qualitative depth is strongest in the already-solved attitude regime and ceiling-bound against the behavior problem specifically.

## Applying it to the case-study shape (script → good? → who → how)

The competence is in **holding three confidence tiers apart**, not fusing them into one simulated number:

- **"Is the script good"** → a *ranking* question, cold-start cell. Ground in telemetry, place the novel script semantically. Deliver a **calibrated rank with an explicit confidence tier; refuse the point-estimate magnitude.** That refusal, done well, is the CAO mandate enacted.
- **"Who to market it to"** → the *solved* regime (segment-level attitude/preference matching, the 0.83 cell). High confidence — say so loudly. Calibration means being clear about what you trust, too.
- **"How to outfit it"** → the *situational* problem, the empty cell. Use the adaptive probe to interrogate the simulation for directions of largest preference movement, then test only those against telemetry. Directions-to-test, not answers.

### Traps to avoid

- **The hollow demo.** A clean end-to-end pipeline that's epistemically empty because nothing was validated against held-out behavior. Make validation the *spine* of the answer, not the appendix — lead with "here's how I'd know this is right before trusting it."
- **Leading with the cute thing.** The adaptive probe is a second-order edge; lead with it and you sound like a design-thinker who hasn't grounded in data. Lead with revealed behavior and validation; bring the probe in for the outfitting question specifically.
- **Researcher, not founder.** A founding-CAO conversation tests scoping to what a $10M build-to-sell company ships in year one, and whether you'll fight for the seat. Have the full map in your head; in the room, **lead with the smallest thing that's real and validated** (ranking-against-telemetry for one vertical) and reveal depth on probe. Depth on tap, not depth on transmit.

---

*Calibration note: this maps the technical and strategic content, which is in hand. It does not model the actual shape of tomorrow's conversation or how the founders will weight it. The unmodeled, load-bearing piece is conversational sequencing and surfacing the leadership/seat question.*
