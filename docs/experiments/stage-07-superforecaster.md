# Stage 07 — Superforecaster prompting strategies (dev, n=1000)

- **Status:** **DONE** (ran 2026-06-21) — **negative result (H0):** no forecasting
  strategy beats the incumbent; structured CoT clusters with generic CoT, all
  ~2–3 pts below the no-CoT direct ask. Incumbent unchanged.
- **Run name:** `2026-06-21-superforecaster`
- **Preregistered:** 2026-06-21
- **Owner:** Ian + Claude

> Decision rule is fixed **before** the run. Append results below the line.
> No `val`/`test` contact in this stage — `dev` only.

## Context

The system is locked at **`anti_flattening` @ `gemini-3.1-flash-lite`** (dev grouped
48.81, val grouped 53.02 — confirmed in Stages 04–05). Two follow-ups have since
come back **negative**:

- **Stage 03** — Monte-Carlo individuals (−15.7) and a generic "reason about the
  range first" CoT strategy (`diversity_elicitation`, −3.1) both lost to the
  simple single-call distributional framing.
- **Stage 06** — all three post-hoc calibrators lost to identity.

So both *external aggregation* and *post-hoc correction* are exhausted. This stage
goes back to the **prompt** — the project's main experiment surface — and asks a
sharper question than Stage 03 did.

## Hypothesis

Superforecasting (Tetlock) is not "think harder" — it is a **specific debiasing
pipeline**. Its moves map onto SimBench's two documented LLM failure modes:

| Superforecaster move | LLM failure it targets here |
|---|---|
| **Outside view first** (estimate the base rate, then adjust) | **over-conditioning** — SimBench's core finding is that demographic conditioning *degrades* group scores. Anchoring on the population marginal the model knows well, then adjusting modestly, should resist that. |
| **Estimate the spread first** (commit to "consensus vs contested" before the numbers) | **over-sharpening** — the model is worst on high-entropy items (SimBench r=−0.942). Forcing an explicit spread commitment before the distribution should flatten contested items. |
| **Premortem** ("which option am I under-weighting?") | **flattening** — too little mass on real minority options. |

**Why this differs from the failed `diversity_elicitation` (Stage 03):** that was
*unstructured* free-text ("think about the range, then answer"). The
superforecaster version forces **structured intermediate commitments** — an
explicit population base-rate estimate and an explicit spread rating — that
constrain the final numbers. The open question: is that structure enough to stop
the model from saying "contested!" in prose and then still emitting a peak?

- **H1 (entropy_first):** committing to a spread rating, then matching it, lifts
  grouped score by reducing over-sharpening on high-entropy items.
- **H2 (outside_view):** base-rate anchoring + modest adjustment lifts grouped
  score by reducing over-conditioning.
- **H3 (superforecaster):** the full pipeline (H1 + H2 + premortem) beats both
  ablations and the incumbent.
- **H0 (null):** structured forecasting prompts do not beat `anti_flattening` —
  a second clean negative on "make this model reason about distributions."

## Configs

All on `gemini-3.1-flash-lite`, `temperature=0` (deterministic, cache-stable).
Three **new** strategies (one `PromptStrategy` subclass + registry entry each,
auto-registering as predictors), plus two reference systems already cached on the
same sample. Each new strategy writes reasoning first, JSON object **last**
(same contract as `diversity_elicitation`).

| system | role | new? | mechanism |
|---|---|---|---|
| `anti_flattening` | **incumbent control** | cached (Stage 06) | distributional, no CoT |
| `diversity_elicitation` | **generic-CoT reference** | new on this sample | "reason about range" CoT, no forecasting structure |
| `outside_view` | ablation | **new** | base-rate first → modest demographic adjustment |
| `entropy_first` | ablation | **new** | spread rating (1–5) → distribution matched to it |
| `superforecaster` | full pipeline | **new** | outside view → spread → adjust → premortem |

The two reference systems isolate the confound: `anti_flattening` is the
no-CoT incumbent; `diversity_elicitation` is generic CoT with no forecasting
structure. A forecasting strategy only earns the claim "the *structure* helps"
if it beats **both** — otherwise any gain is just "reasoning-first."

### Strategy designs (exact prompt structure)

Shared scaffolding (existing helpers): `_population_phrase(record)` →
"a large, representative sample of {who}{year_clause}"; `_json_instruction(options)`
→ the sum-to-1 JSON contract. System prompts below; user message ends with
"Write your reasoning first, then the JSON object LAST."

**`outside_view`** — base-rate anchoring (isolates H2)
- *System:* "You are an expert forecaster of survey responses. You always start
  from the base rate — how the general population answers — and adjust only as much
  as the specific group genuinely justifies, never over-reacting to a group label."
- *User steps:* (1) Ignoring their specific characteristics, estimate how the
  **general population** answers — the base rate. (2) Now adjust for *this* group:
  which way, and how much, does being {descriptor} shift each option? Make only the
  adjustments the demographic actually justifies; keep shifts modest absent a strong
  reason. (3) Give the final distribution.

**`entropy_first`** — spread-before-distribution (isolates H1; the most novel)
- *System:* "You are an expert forecaster of survey responses. You first judge how
  divided a group is on a question, then give a distribution whose sharpness matches
  that judgment — you never output a confident peak on a question you judged contested."
- *User steps:* (1) Rate how divided this group is, **1–5**: 1 = near-unanimous (one
  option dominates), 3 = leaning but contested, 5 = evenly split across options.
  State the number. (2) Give a distribution whose spread **matches** that rating —
  a low rating is peaked, a high rating is flat.

**`superforecaster`** — full pipeline (H3)
- *System:* "You are a superforecaster estimating how a population subgroup answers a
  survey. You reason in steps: outside view first, calibrate the spread, adjust for the
  group, and check what you might be under-weighting — then commit to numbers."
- *User steps:* (1) **Outside view:** base rate for the general population. (2)
  **Spread:** rate 1–5 how divided *this* group is, given its real internal diversity.
  (3) **Adjust:** shift the base rate for this group, only as justified. (4)
  **Premortem:** which option might you be under-weighting? Keep real mass on plausible
  minority views. (5) Final distribution consistent with all four steps.

## Data & budget

- **Subsample of dev (n=1000):** `stratified_sample(grouped, 700) +
  stratified_sample(pop, 300)`, `seed=42` — **identical** to Stage 06, so the
  `anti_flattening` control is already cached and directly comparable, and the
  grouped CI is computed on n=700.
- **val/test:** untouched.
- **Normalizers:** Eq. 2 scalars from the **full** dev split (as in all prior stages).
- **Cost:** `anti_flattening` = $0 (cached). Three new CoT strategies +
  `diversity_elicitation` ≈ 4 × 1000 calls; CoT on `gemini-3.1-flash-lite`
  ($0.25/$1.50 per M tok) ≈ **$3–4 total**. Re-runs are free from cache.

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** grouped mean SimBench score, 95% bootstrap CI on n=700.
- **Noise floor:** half-width of the larger of the two CIs being compared
  (expected ~3–4 at n=700). "Beats" = mean gap > noise floor.
- **A forecasting strategy wins** iff it beats **both** references on dev grouped
  beyond the noise floor: `anti_flattening` (no-CoT incumbent) **and**
  `diversity_elicitation` (generic CoT). Beating only `anti_flattening` but not
  `diversity_elicitation` is attributed to "reasoning-first," not forecasting
  structure, and does **not** count as a structure win.
- **Attribution (which mechanism carries it):**
  - `superforecaster` wins but neither ablation does → the *combination* matters.
  - An ablation matches `superforecaster` → that single mechanism carries the effect.
  - `entropy_first` alone wins → the shape-first idea is the driver (cleanest result).
- **Carry-forward:** the single best winner (if any clears the bar) becomes a
  **candidate** for a val confirmation round (Stage 08, one val query). If it
  also beats the current incumbent `anti_flattening`, it would be the new system
  pending that val confirmation. `test` stays locked.
- **Negative result (H0):** if every forecasting strategy is within noise of
  `anti_flattening` (or worse), structured forecasting prompts do not help this
  model — recorded as a clean second negative, incumbent unchanged.
- **Diagnostics to report regardless:** pop scores (expect methods to bunch —
  pop is near-unconditioned); score-vs-entropy curves for `entropy_first` vs
  `anti_flattening` (does the spread commitment actually help high-entropy items?);
  and a few example distributions.

---

## Results

Ran 2026-06-21 on the n=990 dev subsample (700 grouped / 290 pop, seed=42;
the pop pool stratified to 290). All systems saw the identical records.
`gemini-3.1-flash-lite`, `temperature=0`, max_workers=8. 21 min, **$2.53**.
Run files: `outputs/runs/2026-06-21-superforecaster.{topline.csv,results.json,meta.json}`.

### Topline (grouped is the deciding metric)

| system | role | pooled | **grouped** | grouped 95% CI | pop | cf_align |
|---|---|---|---|---|---|---|
| **`anti_flattening`** | **incumbent (no CoT)** | 41.08 | **45.01** | [42.04, 47.86] | 31.59 | 0.452 |
| `diversity_elicitation` | generic-CoT ref | 38.83 | 42.93 | [39.27, 46.17] | 28.93 | 0.471 |
| `outside_view` | ablation | 38.67 | 42.77 | [39.31, 46.17] | 28.77 | 0.445 |
| `entropy_first` | ablation | 39.01 | 42.53 | [39.29, 45.66] | 30.49 | 0.414 |
| `superforecaster` | full pipeline | 38.72 | 41.55 | [37.86, 44.95] | 31.91 | 0.416 |

Gap vs incumbent on grouped (noise floor = larger half-width, ~3.2–3.6):

| system | gap | noise floor | verdict |
|---|---|---|---|
| `diversity_elicitation` | −2.08 | 3.45 | within noise (loses) |
| `outside_view` | −2.24 | 3.43 | within noise (loses) |
| `entropy_first` | −2.48 | 3.18 | within noise (loses) |
| `superforecaster` | −3.46 | 3.55 | within noise (loses) |

### Verdict against the decision rule

**Negative result (H0).** The rule required a forecasting strategy to beat **both**
`anti_flattening` *and* `diversity_elicitation` beyond the noise floor. **None beats
either.** Every forecasting strategy lands *below* the no-CoT incumbent and sits on
top of the generic-CoT reference. No carry-forward; **incumbent `anti_flattening`
@ `gemini-3.1-flash-lite` is unchanged.** No val query spent; `val`/`test` untouched.

### What we learned

1. **A roughly constant "CoT tax."** All four reasoning-first systems (generic +
   all three forecasting) collapse into a tight **41.5–42.9** grouped band, ~2–3.5
   points under the direct distributional ask (45.01). Whatever the reasoning's
   *structure*, eliciting it first costs about the same on this model+task. This
   replicates Stage 03's `diversity_elicitation` finding on a fresh sample.
2. **More structure was slightly worse, not better.** Monotonic with scaffolding:
   `superforecaster` (4 steps, 41.55) < the 2-step ablations (42.5–42.8) <
   `diversity_elicitation` (loose CoT, 42.93). The premortem / multi-step pipeline
   pulls the model *further* from its better direct estimate rather than debiasing it.
3. **The specific superforecaster moves did not fire as theorized.** `entropy_first`'s
   explicit spread commitment did **not** lift grouped score and *lowered*
   `cf_alignment` (0.414 vs incumbent 0.452) — committing to "contested" in prose did
   not translate into a better-calibrated distribution. `outside_view`'s base-rate
   anchoring likewise gave no conditioning benefit (cf_align 0.445).
4. **Second clean negative on "make this model reason about distributions"** (Stage 03
   was the first). Two independent negatives now point the same way: for
   `gemini-3.1-flash-lite` on SimBench, the single-call **direct** distributional
   judgment beats any **elicited** verbalized reasoning. Superforecaster debiasing —
   built for human cognition — does not transfer to this LLM's distributional
   forecasting here.

### Implications for next stage

- The **prompt-reasoning** axis is now well-explored and exhausted on this model:
  conditioning *framing* helps (Stages 01/04/05), but added *reasoning* (Monte-Carlo,
  diversity CoT, superforecaster) consistently hurts. Stop pushing CoT here.
- Open, still-unexplored directions: (a) a **stronger/reasoning-native model** —
  the CoT tax may be model-specific; a model that reasons natively might invert this
  (would be a Stage 02-style model sweep of the *best* forecasting strategy, not the
  faithful one); (b) **post-stratification** (`PostStratificationPredictor` exists but
  is untested in a stage); (c) declare the method search converged and spend the
  one-time **`test`** number on the locked `anti_flattening` @ `gemini-3.1-flash-lite`.
