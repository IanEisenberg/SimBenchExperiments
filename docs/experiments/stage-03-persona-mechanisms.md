# Stage 03 — Persona mechanisms (Monte-Carlo individuals + diversity-elicitation CoT)

- **Status:** **DONE** (ran 2026-06-20) — both mechanisms **lost** (negative result)
- **Run name:** `2026-06-20-persona-mechanisms`
- **Preregistered:** 2026-06-20
- **Owner:** Ian + Claude

> Decision rules are fixed **before** the run. Append results below the line.

## Hypothesis

Stages 01–02 found the best *single-call group framings* (anti_flattening,
contextualized). This stage tests two mechanisms that change *how* the
distribution is produced, on the carry-forward model `gemini-3.1-flash-lite`:

- **Monte-Carlo individuals** (`monte_carlo`, N=20): draw 20 synthetic
  within-group individuals, get each one's distribution, and average. The group
  spread emerges from cross-draw variation.
  - **Pre-run finding (amendment):** the original design relied on
    temperature/seed sampling for variation. A live diagnostic showed
    gemini-3.1-flash-lite is **near-deterministic across seeds** — all 20 draws
    came back byte-identical (and confidently wrong), collapsing MC to one
    over-confident call (~uniform score). **Fix:** inject variation ourselves —
    each draw samples a synthetic person (a worldview/ideology lean + the
    demographic attributes the segment leaves open) from a **seeded RNG**
    (`scrye.persona.sample_persona`), at `temperature=0` for reproducibility.
    Diagnostic on one item: aggregate moved from `{A:.85,B:.10}` (broken) to
    `{A:.57,B:.38}` vs truth `{A:.48,B:.51}`.
  - **H1:** external aggregation of sampled individuals beats the best single-call
    group framing on grouped — i.e. `monte_carlo` > max(anti_flattening,
    contextualized) by more than the noise floor.
- **Diversity-elicitation CoT** (`diversity_elicitation`): one call, model reasons
  about the *range* of in-group views before emitting the distribution.
  - **H2:** the reasoning step beats its non-reasoning parents (anti_flattening /
    contextualized) on grouped.
- **H3 (cost/benefit):** MC costs N=20× the calls. Its grouped gain must justify
  the cost (report score-per-dollar); a within-noise gain does not.

## Configs to run

All on `gemini-3.1-flash-lite`, `temperature=0`. `monte_carlo` draws N=20
seeded synthetic personas (variation from the sampler, not the model RNG).

| system | predictor | role |
|---|---|---|
| `faithful` | zero_shot / simbench_faithful | control |
| `anti_flattening` | zero_shot / anti_flattening | base (Stage 01/02 winner) |
| `contextualized` | zero_shot / contextualized | base (Stage 02 best grouped) |
| `diversity_elicitation` | zero_shot / diversity_elicitation | **new** |
| `monte_carlo` | MonteCarloPredictor(N=20) | **new** |

## Data & budget

- **Subsample of dev**, sampled separately for guaranteed grouped coverage
  (as amended in Stage 02): `stratified_sample(grouped, 500) +
  stratified_sample(pop, 100)`, `seed=0`. Same records for every system.
- `val`/`test` untouched. Normalizers from the FULL split (Eq. 2).
- **Realized n / pop / grouped:** _filled at run time._
- **Cost cap:** **$10** (estimate ≈ $2–3; `monte_carlo` at N=20 dominates —
  ~12k calls on 3.1-flash-lite. Single-call systems are cheap / partly cached
  from Stage 02). Throttled workers + `max_retries=10` for the 300-rpm limit.

## Decision rule (preregistered — fixed before looking)

- **Primary metric:** grouped mean SimBench score per system, 95% bootstrap CI.
  Pooled + pop/grouped reported as standard.
- **H1 verdict:** `monte_carlo` **wins** if its grouped mean beats
  `max(anti_flattening, contextualized)` on this sample by more than the noise
  floor (half-width of the larger CI).
- **H2 verdict:** `diversity_elicitation` wins if it beats both parents on grouped
  by more than the noise floor.
- **H3 / carry-forward:** the mechanism carried into the final test-set number is
  the one with the highest grouped score — **cost-adjusted**: a `monte_carlo`
  win below ~2 SimBench points does not justify 20× the calls; prefer the
  single-call winner then.

## Artifacts

- `outputs/runs/2026-06-20-persona-mechanisms.{topline.csv, results.json, meta.json}`
- Ledger: new mechanism nodes branched off the `gemini-3.1-flash-lite`
  carry-forward node in `outputs/ledger/2026-06-20-baseline.jsonl`.

---

## Results

- **Subsample:** dev grouped=500, pop=100 · **Cost:** $2.08 · **13,153 calls** (1,247 cache hits) · **46 min** · model `gemini-3.1-flash-lite`

### Grouped score by system (primary) — mean [95% CI]

| system | grouped | 95% CI | pop | pooled | frac_below_uniform |
|---|---|---|---|---|---|
| **contextualized** (base) | **44.95** | [41.1, 48.8] | 19.24 | 40.67 | 0.157 |
| **anti_flattening** (base) | 44.51 | [41.0, 48.0] | 21.47 | 40.67 | 0.157 |
| faithful (control) | 41.49 | [37.3, 45.6] | 18.84 | 37.71 | 0.190 |
| diversity_elicitation *(new)* | 41.85 | [37.8, 45.9] | 23.42 | 38.78 | 0.167 |
| monte_carlo, N=20 *(new)* | 29.28 | [25.1, 33.2] | 5.01 | 25.24 | 0.267 |

### Verdicts — both new mechanisms LOST

- **H1 (monte_carlo > best single-call): REFUTED, decisively.** `monte_carlo`
  grouped 29.28 vs best base 44.95 → **−15.7**, far beyond the noise floor (~4).
  Even after fixing the seed-determinism collapse, externally aggregating
  *uniformly-sampled* synthetic individuals is much worse than the model's own
  single-call group estimate — we don't know the true persona mixing weights, and
  uniform weighting over 12 ideology archetypes over-disperses. (MC's pop score
  5.0 and highest `frac_below_uniform` 0.267 confirm it injects noise.)
- **H2 (diversity_elicitation > parents): REFUTED.** 41.85 vs anti_flattening 44.51
  / contextualized 44.95 → **−2.7 to −3.1** (within noise, but trending *worse* and
  regressing to ~faithful). The reasoning preamble did not help.
- **H3 / carry-forward: unchanged — `contextualized` (≈`anti_flattening`) on
  `gemini-3.1-flash-lite`.** Neither new mechanism is carried forward.

### Key finding

A rigorous test of the headline "simulate individuals and aggregate" hypothesis:
**it does not work here.** The simple single-call distributional framing
(contextualized/anti_flattening) remains best; the model's internal group
estimate beats our hand-built Monte-Carlo mixture and a reason-first prompt. A
clean negative result — the method search converges on the Stage 01/02 winner.

> **Caveat:** n=500 grouped → noise floor ~4. The base conditioning advantage
> over faithful (+3.5) is itself at the noise edge here (consistent with Stage 02);
> the *mechanism* losses (esp. MC, −15.7) are well clear of it.

**Run files:** `outputs/runs/2026-06-20-persona-mechanisms.{topline.csv, results.json, meta.json, log}`;
ledger extended to 12 nodes (`predictor.swap → diversity_elicitation` / `→ monte_carlo`
off the `gemini-3.1-flash-lite` contextualized node).
