# Stage 18 — Grounded persona electorate (Nemotron-Personas-USA)

**Status:** preregistered (dev-only) · **Date:** 2026-06-22 · **Model:** `gemini-3.1-flash-lite`

## Motivation

Every system so far predicts a group's answer distribution by asking the model to
**introspect** it — one call that reports "how would this population answer."
Stage 03 tested the obvious bottom-up alternative — `MonteCarloPredictor`, which
averages many *synthetic* individuals' self-reported distributions — and it
**lost decisively** (pop 5.0, worst of all systems). The diagnosis: the synthetic
personas were drawn from hand-built, uniformly-weighted ideology/demographic axis
pools (`persona.IDEOLOGY_AXES`), so they were generic, mutually uncorrelated, and
**over-dispersed**. Stage 13's discrete-vote ensemble fixed the averaging artifact
but kept the same synthetic, ungrounded individuals, and only won where the
model's mode was already right (Choices13k).

The persona pool was the weak link, never the aggregation. **Nemotron-Personas-USA**
(NVIDIA, 1M synthetic US adults) replaces it with the opposite kind of pool:

- **Census-grounded.** Demographics are sampled from a PGM aligned to the ACS, so
  the corpus *as a whole is a representative sample of the US population* — joint
  demographics in real proportion, not uniform archetypes.
- **Richly described.** Each persona carries a multi-paragraph life narrative
  (work, culture, hobbies, values), far more concrete than a one-clause
  ideology tag.

This is the clean causal test the project has been missing: **does grounding the
persona pool in a real representative sample rescue the bottom-up simulation that
synthetic personas failed at?**

## The method: a fixed representative electorate

Use Nemotron exactly as designed — as a standing representative sample of US
adults — instead of reconstructing a population per question:

1. Draw a **fixed panel of K = 50 personas** once (seeded uniform sample of the
   corpus → census-representative by construction). The same electorate answers
   every question, so it is cacheable and auditable.
2. For each question, condition the model on **each persona's narrative** and ask
   that one person to **commit to a single option** (discrete vote; reuses the
   Stage-13 `voter` framing). Calls run in parallel.
3. The group prediction is the **vote tally** (+ Laplace `alpha` smoothing). Spread
   is endogenous: sharp where the panel agrees, diffuse where it splits.

**Scope: OpinionQA only** — the one US-anchored SimBench dataset, so a US electorate
is the honest grounding.

- **Headline (Round 1):** the whole-corpus panel on **OpinionQA pop** (500
  unconditioned US-population questions). No demographic conditioning needed — the
  panel *is* the US population — so Nemotron's missing structured axes
  (race/income/religion/politics) do not bite at all.
- **Extension (Round 2):** a **demographically-filtered** panel on the OpinionQA
  grouped segments whose axis Nemotron carries — **age / sex / region / education /
  marital** (405 of 984 grouped rows). Segments pinned on an axis Nemotron lacks
  (RACE/INCOME/RELIG/RELIGATTEND/POLPARTY/POLIDEOLOGY → 579 rows) cannot be
  matched, so the predictor **falls back to the champion single-call prompt** for
  those records (mirrors `PostStratificationPredictor`'s no-decomposition
  fallback), and the coverage split is logged.

This stage evaluates the persona electorate **on its own**. Ensembling it with the
champion is a *separate, later* step, pursued only if Round 1 is promising.

## Configs

- **New predictor:** `GroundedVotingPredictor` (`src/scrye/predict.py`) — a fixed
  Nemotron panel, discrete vote per persona, tally + `alpha = 0.5` smoothing,
  `temperature = 0`, K-call parallelism. `K = 50`, model `gemini-3.1-flash-lite`.
- **Persona bank:** `src/scrye/nemotron.py` — loads a slim local cache of the
  corpus, draws the fixed panel, filters by matchable axes (`PersonaBank`).
  Persona→message via `persona.persona_voter_messages`.
- **Comparators (the bar = "naive LLM"):**
  - `uniform` — the score-0 floor.
  - `simbench_faithful` — the paper's first-person baseline (true naive).
  - `calibrated_commitment` — the champion single-call (strongest introspection).
- **Data:** OpinionQA **dev** only, `make_split(pop+grouped, seed=0, unit="question")`.
  Round 1 = all OpinionQA pop dev rows. Round 2 = matchable OpinionQA grouped dev
  rows. Normalizers (Eq. 2 Z) built on the **full** OpinionQA split.
- **Budget:** K=50 × OpinionQA dev rows, cached on re-run. Est. ≤ $8 first run.

## Decision rule (fixed before results)

Primary contrast is **`grounded_voting` vs `calibrated_commitment` (champion
single-call) on OpinionQA pop dev**, by mean SimBench score with bootstrap CIs.

- **PROMISING → pursue the ensemble step (and later a val-gated confirmation):**
  `grounded_voting` ≥ `calibrated_commitment` within the bootstrap noise floor on
  OpinionQA pop **and** clearly beats `uniform`/`simbench_faithful`. (A grounded
  electorate that *matches* a tuned single-call while being a genuinely different,
  bottom-up signal is already promising — uncorrelated errors are what make an
  ensemble pay off.)
- **STRONG WIN:** beats `calibrated_commitment` by more than the noise floor →
  carry forward as a candidate system in its own right; proceed to Round 2 +
  ensemble + val.
- **NEGATIVE:** below `uniform`, or far below the champion with correlated errors →
  record as a negative result (grounding does not rescue bottom-up simulation for
  opinion surveys); the champion stands.
- **Round 2 (extension), reported regardless:** on matchable grouped segments,
  does demographic filtering move the prediction in the **right direction**
  (counterfactual-sensitivity alignment) vs the unconditioned panel — i.e. does a
  grounded electorate actually use the demographics?

No val/test contact this stage. OpinionQA's required questions are pinned to test
by `splits.py` and are untouched. Nemotron carries no survey answers, so there is
no label leakage — personas are input conditioning only.

## Implementation plan

- `scripts/prepare_nemotron.py` — one-time: download `nvidia/Nemotron-Personas-USA`,
  write a slim seeded subsample (`data/nemotron/personas_slim.parquet`) with only
  the columns we use: `uuid, sex, age, marital_status, education_level, state,
  occupation, persona`. Prints categorical value vocabularies so the axis maps are
  built from the real data.
- `src/scrye/nemotron.py` — `PersonaBank`: load slim cache; `panel(k, seed)` (fixed
  representative electorate); `matched_panel(record, k, seed)` (filter by matchable
  axes); axis maps (age→bins, sex, state→Census region, education_level→OpinionQA
  EDUCATION, marital_status→OpinionQA MARITAL) with explicit "unmatchable" returns.
- `src/scrye/persona.py` — `persona_voter_messages(record, persona_text)`: the
  Stage-13 voter ask, conditioned on a full narrative instead of a disposition
  clause (reuses `VOTER_SYSTEM`).
- `src/scrye/predict.py` — `GroundedVotingPredictor`, reusing `_parse_vote` and the
  tally/smoothing logic; optional `fallback` predictor for unmatchable segments;
  internal `ThreadPoolExecutor` for the K calls.
- `experiment.PREDICTOR_REGISTRY` — register `grounded_voting` (lazy bank load).
- Tests (`tests/test_nemotron.py`, offline): axis mapping & bucketing, matchable vs
  unmatchable detection, panel determinism under seed, tally/smoothing correctness,
  fallback path. LLM calls are cached; pure logic is tested without the network.

---

## Results

### Round 1 — grounded discrete-vote electorate (OpinionQA pop dev, n=248)

K=50, `gemini-3.1-flash-lite`, all 248 grounded (0 fallback — pop has no segment).
Run file: `outputs/runs/2026-06-22-stage18-r1.results.json`.

| system | mean_score | 95% CI | frac<uniform | cf_align |
|---|---|---|---|---|
| calibrated_commitment | **63.24** | [60.45, 65.91] | 0.01 | −0.13 |
| simbench_faithful | 59.66 | [56.66, 62.38] | 0.02 | — |
| **grounded_voting (K=50)** | **12.67** | [6.80, 18.37] | **0.39** | −0.86 |
| uniform | −0.81 | [−3.64, 2.15] | — | — |

**Decision-rule verdict: NEGATIVE for the discrete-vote variant.** The grounded
electorate beats uniform (CI excludes 0) but is crushed by the naive single-call
(13 vs 63) and falls below uniform on 39% of items.

**Mechanism (diagnosed before the run, confirmed by it): over-concentration.**
Forced to commit to one option, most personas argmax onto the *same* modal answer,
so the vote tally is far too peaked and discards the real human spread. Example
(truth → grounded):

- "…job replaced by robots…": truth C .38 / B .32 / D .19 / A .10 → grounded **.89 C**.
- "China's COVID response": truth B .32 / D .28 / C .26 / A .12 → grounded .64 D / .30 C.

The bottleneck is not the persona pool's demographics (they are census-grounded and
do vary) but the **per-individual collapse**: argmax-then-tally throws away each
person's internal uncertainty. This is the discrete-vote analogue of Stage 13's
over-confidence, now on opinion surveys where the human distribution is genuinely
spread. Errors are also *correlated* with `calibrated_commitment` (same mode), so
a naive ensemble of the two is unpromising.

### Round 2 — soft voters: per-persona distribution + average (preregistered)

**Hypothesis.** The over-concentration is caused by collapsing each persona to a
single option *before* aggregating. Move the collapse to the population level:
ask each persona for *their own* probability over the options (a calibrated read of
that individual, `PERSONA_DIST_SYSTEM` / `persona_dist_messages`) and **average**
across the panel. Averaging is the unbiased estimator of the population
choice-fraction (sampling a vote only adds variance; argmax discards within-person
uncertainty), so if the grounding lets the model give *decided-but-honest*
per-person distributions, the group spread should re-inflate toward the truth.

This is the untested cell of the 2×2 (aggregation × persona source): Stage 03 was
synthetic + average (over-dispersed, lost); Round 1 is grounded + argmax
(over-concentrated). Round 2 is **grounded + average**. The risk it trades into is
*over-dispersion* if the model hedges each persona toward uniform — which the
calibrated prompt and the concrete narratives are designed to prevent.

**Config.** `GroundedAveragingPredictor`, same fixed K=50 panel, same model/data
(OpinionQA pop dev, n=248), normalizers on the full split. Reported head-to-head
with Round 1's systems plus the **entropy diagnostic** (mean predicted vs true
normalized entropy) — the direct read on whether spread was recovered.

**Decision rule (fixed before results).**
- **PROMISING (pursue ensemble + val):** `grounded_averaging` beats
  `grounded_voting` decisively *and* closes most of the gap to
  `calibrated_commitment` (within a few points or better), with predicted entropy
  moving from Round-1's over-concentration toward the truth entropy (not past it).
- **STRONG:** matches or beats `calibrated_commitment` (CI overlap or better) →
  a standalone grounded system worth val-gating, and a strong ensemble candidate
  if its errors decorrelate from the single-call.
- **NEGATIVE:** still far below `calibrated_commitment`, or over-disperses past the
  truth entropy (the Stage-03 failure reappears with grounding) → record that
  bottom-up persona simulation does not beat introspection for opinion surveys.

**Results (OpinionQA pop dev, n=248, K=50).** Run file:
`outputs/runs/2026-06-22-stage18-r1.results.json` (combined table).

| system | mean_score | 95% CI | frac<uniform | pred_H | truth_H | cf_align |
|---|---|---|---|---|---|---|
| calibrated_commitment | 63.24 | [60.45, 65.91] | 0.01 | 0.771 | 0.692 | −0.13 |
| simbench_faithful | 59.66 | [56.66, 62.38] | 0.02 | 0.768 | 0.692 | — |
| **grounded_averaging (K=50)** | **46.32** | [42.35, 50.39] | 0.11 | **0.644** | 0.692 | +0.08 |
| grounded_voting (K=50) | 12.67 | [6.80, 18.37] | 0.39 | 0.425 | 0.692 | −0.86 |
| uniform | −0.81 | [−3.64, 2.15] | 0.46 | 1.000 | 0.692 | — |

**Verdict: hypothesis CONFIRMED; standalone still below champion; residual gap is
location, not spread.**

- **+33.6 over discrete voting** (46.3 vs 12.7) purely from moving the collapse to
  the population level (per-persona distribution + average). The mechanism is the
  per-individual argmax, exactly as diagnosed.
- **Spread recovered and well-calibrated.** Predicted entropy moves 0.425 → 0.644
  vs truth 0.692 — near-calibrated and, crucially, **not** over-dispersed (the
  Stage-03 risk). Averaging's spread is actually *closer* to truth than the
  champion's (CC 0.771 is slightly too diffuse).
- **Still ~17 below `calibrated_commitment`.** Since averaging's entropy is better
  than CC's, the remaining gap is **location** (which options carry the mass), not
  spread. The grounded electorate gets the shape right but mis-locates the mode
  more than introspection.
- **Decorrelated signal.** `grounded_averaging` cf_align is **+0.08** (conditioning
  direction weakly correct) where CC is **−0.13** — a genuinely different error
  profile, which makes an ensemble worth testing (Round 3).

### Round 3 — ensemble with the champion (free; both systems cached)

`grounded_averaging` contributes well-calibrated spread + correct conditioning
direction; `calibrated_commitment` contributes better location. Their predictions
are already cached on the 248 dev records, so a weighted blend
`w·averaging + (1−w)·CC` is scored at zero marginal LLM cost. Sweep `w` on dev
(decision: a dev-best `w` that beats CC by more than the bootstrap noise floor →
val-gated confirmation; the grounded signal "adds something" only if best `w` > 0
with a score above CC).

**Results (OpinionQA pop dev, n=248, weight sweep, zero new LLM calls).**

| w (on averaging) | mean_score | 95% CI | pred_H |
|---|---|---|---|
| 0.0 (pure calibrated_commitment) | 63.24 | [60.45, 65.91] | 0.771 |
| **0.1 (best)** | **63.57** | [60.71, 66.18] | 0.764 |
| 0.3 | 62.80 | [59.82, 65.47] | 0.748 |
| 0.5 | 60.29 | [57.14, 63.35] | 0.726 |
| 1.0 (pure grounded_averaging) | 46.32 | [42.35, 50.39] | 0.644 |

**Verdict: NEGATIVE.** The dev-best blend (w=0.1) beats CC by **+0.33** — far inside
the bootstrap noise floor (CIs essentially identical), and any larger grounded
weight strictly hurts. The grounded electorate's *location* errors are not better
than the champion's, so blending cannot lift it. No val query is warranted.

## Conclusion

The Stage-18 arc cleanly separates the two failure modes of bottom-up survey
simulation and resolves one of them:

1. **Discrete-vote electorate → over-concentration** (12.7). Argmax-then-tally
   discards each persona's internal uncertainty; the tally is far too peaked.
2. **Per-persona distribution + average → spread solved** (46.3, **+33.6**).
   Collapsing at the *population* level instead of per individual recovers the
   spread to near-calibrated entropy (0.644 vs truth 0.692) **without**
   over-dispersing — the first persona-simulation system in this project that gets
   the *shape* right. Census grounding is what makes the per-person distributions
   decided rather than hedged (the Stage-03 failure does not recur).
3. **But location is the binding constraint, and introspection wins it.** Even with
   well-calibrated spread, the grounded electorate stays ~17 below the
   `calibrated_commitment` single-call, and ensembling adds nothing (+0.33, noise).
   The residual error is *which* options carry the mass — and a direct distributional
   ask still places the mode better than a bottom-up persona average.

This is the project's fourth confirmation (cf. Stages 03, 07, 13) that bottom-up /
elicited approaches lose to the direct distributional ask — but the most
informative one: it shows the loss is now **entirely a location problem**, not a
spread problem, because population-level averaging solved spread. The system
(`calibrated_commitment` @ gemini-3.1 + router) is unchanged.

### Is K=50 too small? (free K-sensitivity check)

The 50 per-persona distributions are cached, so the score for any K′≤50 is scored
at no new cost by averaging random K′-subsets (`scripts/k_sensitivity.py`).

| K′ | 5 | 10 | 25 | 40 | 50 |
|---|---|---|---|---|---|
| mean_score | 41.69 | 44.54 | 45.79 | 46.18 | **46.32** |

The curve **plateaus**: K=10 captures almost all the value, and 40→50 adds only
0.14 (≈0.014/persona, still declining). Extrapolating, K=250 would add at most
~0.5–1 point (≈46.5–47) — still ~16 below the champion. **The system is
location-limited, not sample-limited:** more personas estimate the same biased
location more precisely; they cannot fix a systematic persona→opinion mis-mapping.
A larger panel is therefore *not* the lever; location accuracy is.

**Note on per-slice magnitude.** All Stage-18 scores are **OpinionQA-only, pop,
dev** — the model's strongest dataset — so they sit well above the multi-dataset
test headline (Stage 17, 40.73 pooled over all datasets incl. the hard behavioral
ones). The slice is high by construction (Nemotron is US-only → OpinionQA), not by
any scoring change; normalizers are the standard full-split Eq. 2.

**Reusable artifacts shipped:** `scrye.nemotron.PersonaBank` (census-grounded
electorate with OpinionQA axis matching), `GroundedVotingPredictor`,
`GroundedAveragingPredictor`, `EnsemblePredictor`, and the persona prompts — a
validated grounded-simulation primitive for future location-targeted work.

**Not run (optional next step):** the Round-2 *extension* on OpinionQA **grouped**
segments — does demographic *filtering* of the panel move predictions in the right
direction (counterfactual sensitivity)? It is a distinct question (conditioning,
not population estimation) but costs K×(matchable grouped segments) new calls and
will not change the standalone-vs-champion verdict, so it is deferred pending a
decision to spend on it.

---

## Rounds 4-5 — why the personas don't bite, and the fix

Rounds 1-3 left a 17-point location gap. Diagnostics traced it to a single cause.

**Persona-bite diagnostic (free, from cache).** On the bare-persona summary panel,
the 50 personas barely disagree: mean pairwise inter-persona TVD **0.205**, and
**78% pick the same option** (a 5-option survey where reality puts only ~30-40% on
the top option). The "population estimate" is largely the model's own consensus
answer with persona-flavored noise — which also explains the K-plateau (the
personas are near-redundant, so 10 ≈ 50).

**Controlled sensitivity probe.** But the model is *not* persona-insensitive:
maximally-contrasting hand-written archetypes (progressive-urban vs
conservative-rural …) diverge by **0.54 TVD**, all in sensible directions
(same-sex-marriage 0.75, gun-TV 0.68). So the bottleneck is the **persona
representation**: Nemotron describes lifestyle/work/temperament but has **no
ideology/politics/religion fields**, and opinion answers hinge on that axis. The
model has nothing to differentiate on and falls back to the mode.

**Round 4 — richer Nemotron fields (negative).** Adding `professional_persona`,
`cultural_background`, `hobbies` + explicit demographics barely moved it: +1.2
score, mode-agreement 0.76→0.73. The extra narratives are mostly *apolitical*, so
they supply little of the missing axis.

**Round 5 — worldview enrichment + ideology calibration (the final system).**
Two-stage pipeline (`scrye.worldview`, `EnrichedGroundedPredictor`): (1) infer an
expressive *worldview* per persona (political lean, religiosity, institutional
trust, moral outlook), sampled (temp 0.7) and grounded in their profile, not
argmax'd; (2) post-stratify the panel's inferred-ideology mix onto OpinionQA's
**real `POLIDEOLOGY` marginal** so the added axis is representative by
construction, then answer + weighted-average.

| system (OpinionQA pop dev, n=248) | score | 95% CI | pred_H |
|---|---|---|---|
| calibrated_commitment (champion) | **63.24** | [60.45, 65.91] | 0.771 |
| enriched (calibrated) | 51.84 | [47.79, 55.59] | 0.663 |
| enriched (uncalibrated) | 51.28 | [47.25, 55.02] | 0.655 |
| grounded_avg summary (Round 2) | 46.32 | — | 0.644 |

- **Standalone: real but modest.** Enrichment lifts the grounded electorate **+5.5**
  over the bare-persona baseline (46.3 → 51.8, CI clears it) — the best standalone
  grounded system, and it confirms the missing axis was ideology. Still ~11 below
  the champion; the residual is still location.
- **Representativeness check (a finding in itself).** The enrichment model softens
  everyone toward the center: inferred panel **74% moderate vs real 37%**, with
  **0% in either ideological tail** (real 9%+9%). Calibration can only reshuffle
  the middle (empty tails), so it adds just +0.56 (within noise).
- **Independence: yes.** Per-record correlation with the champion is **r = 0.56** —
  a genuinely decorrelated signal.
- **Ensemble: within noise.** Best blend `0.2·enriched + 0.8·champion` = **63.71**,
  only **+0.47** over the champion — inside the noise floor. Decorrelated but too
  much worse (52 vs 63) to pay off.

## Final verdict (persona-based exploration closed)

The full arc — discrete vote (12.7) → distribution+average (46.3, spread solved) →
worldview-enriched + ideology-calibrated (51.8, axis supplied) — fixed each
predicted failure in turn and produced a census-grounded, decorrelated simulator.
**But even fully developed, the bottom-up persona electorate neither beats nor
meaningfully improves the `calibrated_commitment` champion on OpinionQA opinion
prediction.** It is location-limited, and its decorrelation is too weak to help in
ensemble (+0.47, noise). No val/test contact is warranted; the champion system is
unchanged. This closes the persona-based line of inquiry.

New artifacts: `scrye.worldview` (worldview enrichment + ideology calibration to
OpinionQA's real marginal), `EnrichedGroundedPredictor`, `EnsemblePredictor`, the
persona-bite / sensitivity / K-sensitivity diagnostics. A reusable grounded-
simulation toolkit, even though the headline is negative.
