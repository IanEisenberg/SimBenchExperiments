# Persona Factory: Demographic Conditioning + Post-Stratification

**Date:** 2026-06-19
**Status:** Approved, implementing

## Goal

Build a "persona factory" that, from any SimBench segment (country and/or
demographic attributes), derives well-constructed demographic-conditioning
prompts in several research-backed *styles*; supplies per-task demographic
distribution tables; and uses those distributions to **recombine** subgroup
inferences into population/marginal estimates (post-stratification).

## Motivating tension

The literature (SimBench Hu et al.; Hu & Collier; Wang et al.) finds that
*naive* demographic conditioning **degrades** group-level scores. But two facts
keep the door open, and the design treats them as empirical questions rather
than settled:

1. SimBench's own `group_prompt` is **first-person persona embodiment**
   ("You are from Finland. Your gender is female."). That is exactly the framing
   the negative result was measured on. Third-person *distributional* framings
   ("a representative sample of people from Finland…") are untested here.
2. SimBench averaged over many (including weak) models with one naive prompt.
   Stronger models + better prompts may beat the paper. We do **not**
   pre-condemn direct personas — one style is a seriously-engineered persona
   built to win.

So the factory emits a **registry of ablatable styles**, and we measure which
help vs. hurt rather than assuming.

## Data facts (verified against the grouped split)

- 5 datasets: OpinionQA (US-only, country implicit), ESS (28 EU countries),
  Afrobarometer (39), ISSP (35), LatinoBarometro (17). ~120 distinct countries.
- Conditioning is mostly `country × one attribute` (4,959 of 6,343 records);
  1,364 single-var; 20 unconditioned.
- `group_size` = respondent count per cell. Verified: the group_size-weighted
  average of attribute-subgroup *truths* reproduces the country marginal truth
  to mean TVD 0.0003 (max 0.0046) over 200 country-questions. This identity is
  the correctness backbone for post-stratification and a key test.
- Segment values are human-readable but uneven: descriptive strings
  (`eisced` = "lower secondary education (ES-ISCED II)"), plus a few numeric
  scales (`lrscale`, `topbot`, `rlgdgr`, `political_group`).

## Architecture

### 1. `src/simbench_exp/persona.py` — the factory

- **`COUNTRIES`**: per-dataset country catalog (the full paper coverage),
  addressable by name.
- **`VARIABLE_DICTIONARY`**: maps each grouping variable key to a *third-person*
  verbalization fragment + human label, covering all ~40 keys observed, with a
  generic fallback (`"whose {humanized key} is {value}"`) for anything unknown.
  Numeric-scale keys get scale-aware phrasing.
- **`verbalize_segment(segment, year=None, locale=None)`**: builds a third-person
  descriptor noun phrase, e.g. "from Finland, in the 30-49 age group". Country
  and year are surfaced first; OpinionQA's implicit US locale is recovered from
  the dataset / `group_prompt`.
- **Strategy registry** `STRATEGIES: dict[str, PromptStrategy]`. Each strategy's
  `build_messages(record) -> list[{role, content}]` returns chat messages. A
  shared `distribution_instruction(options)` helper carries the verbalized-
  distribution elicitation (JSON, sums to 1) so styles differ only in framing:
  - `simbench_faithful` — uses `record.group_prompt` verbatim (first-person
    persona). Reproduces the current baseline exactly; the control.
  - `representative_sample` — third-person "estimate the distribution across a
    representative sample of [descriptor]"; decouples from role-play.
  - `persona_embodiment` — a strong, immersive first-person persona engineered
    to answer authentically; the serious direct-persona contender.
  - `anti_flattening` — representative-sample + explicit instruction to preserve
    within-group heterogeneity and resist stereotype collapse (counters Wang
    et al. flattening).
  - `contextualized` — representative-sample + brief year/place context anchor.

### 2. `src/simbench_exp/distributions.py` — post-stratification weights

- **`SegmentWeights`** built from grouped records, indexed by
  `(dataset, input_template)`. Methods:
  - `children(record, over=None)` → list of `(weight, child_record)` decomposing
    `record`'s segment by adding one variable (or auto-pick a variable with the
    most coverage). Weights = `group_size / Σ siblings`.
  - `decompose_variables(record)` → which variables admit a decomposition.
- **`distribution_table(records)`** → `DistributionRow`s: the per-subset-task
  demographic distribution artifact (every (dataset, question, grouping) subset
  with each cell's respondent share).
- **`WeightSource`** protocol (`children(record, over)`), with `SegmentWeights`
  as the SimBench-counts implementation. This protocol is the extensibility
  seam: an external census source — or a finer-than-SimBench subdivision (US →
  states) that returns synthetic child cells — plugs in behind it with no change
  to the combiner. A nested `PopulationCell` tree is deferred until a finer
  level actually exists (YAGNI).

### 3. `PostStratificationPredictor` (in `predict.py`)

Implements the existing `Predictor` ABC; drops into the current
`Pipeline`/`evaluate` harness unchanged.

- At `predict(record)`: look up child cells via `SegmentWeights`; if none, fall
  back to the base predictor on `record` directly. Otherwise predict each child
  record with the base predictor and return `Σ wᵢ · predᵢ`, aligned to
  `record.options`.
- First cut: decompose a country marginal over one attribute (validated against
  the real `('country',)` truth). Country→world combination needs external
  population weights and is deferred behind `WeightSource`.

### Refactor: `ZeroShotPredictor`

Becomes strategy-driven: takes a `PromptStrategy` (default `simbench_faithful`,
preserving current behavior) and builds messages via the strategy instead of a
hardcoded prompt. Uses `client.complete(messages)` to allow system messages.
Parsing/uniform-fallback logic is unchanged.

## Testing

- Strategies emit valid, non-empty messages for population, single-var, and
  two-var records; `simbench_faithful` reproduces the prior baseline prompt
  content.
- `verbalize_segment` handles country, descriptive, numeric-scale, and unknown
  keys without crashing.
- `SegmentWeights.children` weights sum to 1; the group_size-weighted
  recombination of child *truths* reproduces the marginal truth within tolerance
  (uses real data, skipped if absent) — plus a synthetic unit test for the math.
- `PostStratificationPredictor` with a fixed fake base predictor recombines
  correctly and falls back when no children exist.

## Out of scope (now)

- Finer-than-SimBench subdivision (US → states): structure supports it; not
  populated.
- External/census weight sources: interface only.
- Live LLM ablation sweep across styles: a demo script wires it up; the full run
  is a follow-up.
