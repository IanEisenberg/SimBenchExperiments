# Stage 03 — Persona mechanisms (Monte-Carlo individuals + diversity-elicitation CoT)

- **Status:** PLANNED → _RUNNING_ → _DONE_
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

## Results _(appended after the run)_

- **Subsample n / pop / grouped:** _pending_ · **Cost:** _pending_

### Grouped score by system (primary) — mean [95% CI]

| system | grouped | pop | pooled |
|---|---|---|---|
| _pending_ | | | |

**H1 (monte_carlo > best single-call):** _pending_
**H2 (diversity_elicitation > parents):** _pending_
**H3 (cost-adjusted carry-forward):** _pending_

**Run files:** _pending_
