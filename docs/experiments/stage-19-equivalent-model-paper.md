# Stage 19 — Method portability to frontier Claude (equivalent-model paper comparison)

- **Owner:** Ian + Claude
- **Status:** DONE (concluded on Sonnet-4.5; Sonnet-4.6 not completed — OpenRouter
  key cap, user chose to conclude rather than raise it)
- **Date:** 2026-06-25
- **Preceding:** Stage 17 (sealed test; cc+abstain @ gemini-3.1-flash-lite = 40.93 overall;
  faithful @ gemini-3.1 = 35.21). Stage 18 (nemotron, closed below cc).

## Motivation

Our shipped result (40.9) only *ties* the paper's best published single model
(**Claude-3.7-Sonnet, 40.80**, Hu et al. Table 1). Two honest reasons it isn't a
clean "method beats the paper" claim:

1. **Different model.** 40.9 is on gemini-3.1-flash-lite, a model that postdates
   the paper. Under the paper's *same* faithful method, gemini-3.1-flash-lite scores
   **35.21 — below Claude-3.7's 40.80.** Our method only lifts the *cheap* model to
   parity; we never ran our method on a model in the paper's top tier.
2. **We never benchmarked against the paper's best model.** Our entire sweep was
   cheap/open models (Gemini/Qwen/DeepSeek). Claude was never in it.

This stage closes that gap: run **faithful** *and* our **method (cc+abstain)** on a
frontier Claude model and ask whether (a) the method's gain *ports* to a strong
model, and (b) the resulting system *clears* the paper's best published score.

## Model availability constraint (must be recorded)

**Claude-3.7-Sonnet — the paper's #1 — is retired on OpenRouter** (confirmed
2026-06-25 against the live `/models` list; only `claude-sonnet-4`, `-4.5`, `-4.6`
remain in the Sonnet line). We therefore **cannot** reproduce the paper's exact
40.80 on the same model. We substitute the **Sonnet-4 line** (4.5, 4.6) — the
direct, same-vendor, same-tier *successor* to 3.7 Sonnet, i.e. an "equivalent or
stronger" model. This is an honest substitution, not a same-model reproduction, and
every claim below is hedged accordingly.

## Hypotheses

- **H1 (method portability).** The faithful→cc+abstain gain (~+5.5 on gemini-3.1)
  is not a quirk of one cheap model: on a frontier Claude model, `cc+abstain` beats
  `faithful` on overall S, with the paired-delta bootstrap CI excluding 0.
- **H2 (clears the paper).** On the paper-comparable metric (split-avg S), the best
  Claude `cc+abstain` system exceeds the paper's best published single-model score
  (40.80), CI lower bound above 40.80.
- **H3 (faithful anchor).** `faithful` @ frontier Claude lands at/above the paper's
  strong-Claude regime (≈40), confirming the harness puts a top-tier Claude where the
  paper's top-tier Claude was — so the method delta is measured from a fair baseline.

## Configs to run

| Model | Systems |
|---|---|
| `gemini-3.1-flash-lite` (reference, mostly cached) | faithful · calibrated_commitment · cc+abstain |
| `claude-sonnet-4.5` | faithful · calibrated_commitment · cc+abstain |
| `claude-sonnet-4.6` | faithful · calibrated_commitment · cc+abstain |

- **Prompt strategies:** `simbench_faithful` (paper baseline, byte-faithful) and
  `calibrated_commitment` (our winner). `cc+abstain` = cc predictions with
  `AbstainCalibrator` overlay (no extra LLM calls).
- **Abstain fit:** `AbstainCalibrator().fit(...)` on a **stratified val subsample
  (n=2000, seed=0)**, per model (the abstain-set is model-specific). Uses only
  per-dataset reliability vs uniform; never eval truth.
- **Eval basis:** **stratified dev subsample (n=3000, seed=0)**, full-split
  normalizers (`build_normalizers(dev+val+test)`). **Sealed test is NOT touched** —
  Stage 17's "test touched once, ever" property is preserved. Dev uses the same
  datasets + Eq. 2 normalization as the paper, so absolute S is paper-*comparable*
  (not paper-identical; the family-split dataset mix differs slightly).
- **Why dev, not test:** this is a cross-model portability measurement, not a new
  shipped-system selection. Per leakage discipline we keep the sealed test sealed and
  measure on dev; the gemini-3.1 reference row (same dev subsample) makes the Claude
  comparison apples-to-apples on an identical basis.

## Metrics

Per (model × system): **overall** (record-pooled mean S), **grouped**, **pop**, and
**split-avg** (= mean of grouped-mean and pop-mean; the paper's Table-1 aggregation),
with bootstrap CIs on overall and on the faithful→cc+abstain paired delta.

## Decision rule (fixed before looking at results)

- **H1 PASS** iff, on **both** Claude models, `cc+abstain` overall > `faithful`
  overall AND the paired-delta 95% bootstrap CI excludes 0.
- **H2 PASS** iff the **best** Claude `cc+abstain` split-avg S > **40.80** with the
  95% CI lower bound > 40.80. (If the point estimate clears 40.80 but the CI straddles
  it → "matches/edges, not decisively beats".)
- **H3 PASS** iff `faithful` @ each Claude model split-avg S ≥ ~38 (within noise of
  the paper's 40.80 strong-Claude regime).
- **Headline claim is earned** only if H1 ∧ H2 hold. If H1 holds but H2 doesn't, the
  honest claim stays "method ports to frontier models; ties (not beats) the paper."

## Budget

~8k Claude calls/model (faithful 3000 + cc 3000 eval + cc 2000 val-fit) ≈ **$14/model**,
**~$29 total** for the two Sonnet models at $3/$15 per Mtok. Gemini reference ≈ free
(cached). All calls cache to `data/cache/`, so re-runs are free.

## Results (2026-06-25, partial — key-limit blocked on sonnet-4.6)

Run: `scripts/run_stage19_claude.py` → `outputs/runs/2026-06-25-stage19-claude.*`.
Eval = stratified **dev** subsample n=2306 (grouped 481, pop 1825); abstain-fit =
stratified val subsample n=1480. (`stratified_sample` caps at one row per dataset ×
`n//n_datasets`, so the realized n is below the 3000/2000 targets — fine for power.)

| model | system | overall | 95% CI | grouped | pop | **split-avg** |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | faithful | 30.74 | [28.34, 33.28] | 43.92 | 27.26 | 35.59 |
| gemini-3.1-flash-lite | calibrated_commitment | 35.04 | [32.91, 37.29] | 50.98 | 30.84 | 40.91 |
| gemini-3.1-flash-lite | **cc+abstain** | 37.97 | [36.01, 39.96] | 50.98 | 34.54 | **42.76** |
| claude-sonnet-4.5 | faithful | 34.68 | [32.41, 36.93] | 44.60 | 32.07 | 38.34 |
| claude-sonnet-4.5 | calibrated_commitment | 33.70 | [31.40, 36.07] | 47.44 | 30.08 | 38.76 |
| claude-sonnet-4.5 | **cc+abstain** | 35.62 | [33.59, 37.68] | 47.44 | 32.50 | **39.97** |
| claude-sonnet-4.6 | faithful (partial n=731) | 38.42 | [34.19, 42.35] | 41.02 | 37.77 | 39.40 |
| claude-sonnet-4.6 | cc / cc+abstain | — | KEY LIMIT (n=0) | — | — | — |

Faithful→cc+abstain paired delta: gemini **+7.24** [+5.24, +9.30] · sonnet-4.5
**+0.93** [−0.73, +2.75] · sonnet-4.6 n/a.

**Basis note (critical for the paper comparison).** This dev subsample runs *hotter*
than the sealed test for cc+abstain: gemini cc+abstain = **42.76** split-avg here vs
**40.8** on Stage-17 test (+~2). Faithful is basically unchanged (35.59 here vs 35.21
on test). So to compare a Claude cc+abstain number on *this* basis to the paper's
40.80, **deflate by ~2**. Sonnet-4.5 cc+abstain 39.97 → ~**38 paper-equivalent**.

### Decision-rule verdict (sonnet-4.5; sonnet-4.6 pending)

- **H1 (method ports): FAIL on sonnet-4.5.** Delta +0.93 with CI **including 0**.
  Worse: `calibrated_commitment` *alone* (33.70) is **below** `faithful` (34.68) on
  overall — the prompt that gives gemini +4–7 is net-negative on Claude-4.5; only the
  per-model abstain step claws back a hair. The method's gain is **model-specific**,
  not a portable lever.
- **H2 (beats the paper): FAIL on sonnet-4.5.** cc+abstain split-avg 39.97 (~38
  paper-equivalent after the basis correction) is **below 40.80**, CI nowhere near
  clearing it.
- **H3 (faithful anchor ≥38): marginal.** faithful @ 4.5 split-avg 38.34 — below the
  paper's Claude-3.7 regime (40.80). Claude-4.5's *faithful* is a touch weaker than
  Claude-3.7 was; Claude-4.6 faithful (partial 39.40, wide CI) looks stronger.

### Blocker

OpenRouter **key total-spend limit exceeded** mid-run (403) → sonnet-4.6 cc/cc+abstain
never ran. Spend so far: sonnet-4.5 $9.03 + sonnet-4.6 partial $4.74 = **$13.77**.
Finishing 4.6 reuses cache (the 731 faithful + partial fit already cached); ~$9–10
more. **Need the key limit raised to complete 4.6** — the one model whose stronger
faithful might still tell a different story.

### Honest read so far

The experiment was meant to *earn* the "method beats the paper with an equivalent
model" claim. On the one frontier Claude model we fully measured (Sonnet-4.5) it does
the **opposite**: the method neither beats its own faithful baseline (n.s.) nor the
paper (39.97 < 40.80). This **strengthens** the honest framing — the +5.5 "method"
lift is substantially co-adapted to gemini-3.1 and does not transfer — and makes the
CV's "exceeding the paper" line *less* defensible, not more. Final verdict pending
sonnet-4.6.

## Conclusion (concluded on Sonnet-4.5 by decision)

**H1 (method ports): FALSE on Sonnet-4.5.** This is the robust, basis-independent
finding: the gemini-tuned `calibrated_commitment + abstain` does not significantly
beat its own faithful baseline on Claude (+0.93 [−0.73, +2.75]), and the cc prompt is
net-negative there. The +5.5 we credited to "method" on gemini-3.1 is **substantially
model-specific** — it does not transfer to a different model family unchanged.

**H2 (beats the paper): NOT DEMONSTRATED.** Sonnet-4.5 cc+abstain ≈ 38 paper-equivalent
< 40.80. We did **not** earn "method beats the paper with an equivalent model."

**Caveats (stated honestly, both directions):**
- This tests the *fixed, gemini-tuned* method transferred to Claude — not a method
  re-optimized per-model (Stage-19 option 3, declined). It shows the shipped artifact
  doesn't transfer/beat the paper; it does **not** prove no Claude method could.
- We lack a same-basis Claude-3.7-Sonnet anchor (retired on OpenRouter) and our
  dev-subsample mix differs from the paper's full eval, so the absolute "vs 40.80"
  comparison is approximate. The *within-experiment* method-portability result needs
  no such anchor and stands on its own.
- **Sonnet-4.6 was not measured for the method** (key cap; concluded by choice). Its
  partial faithful (39.40, wide CI) hints at a stronger base, but the cc prompt's
  net-negativity on 4.5 makes a method win on 4.6 unlikely. Genuinely open, untested.

**Net.** Stage 19 does the job preregistration is for: a fixed, adversarial cross-model
test **refuted** the hoped-for claim and revealed the method's gemini-specificity. The
honest headline is unchanged from Stage 17 — *cc+abstain @ gemini-3.1 ties the best
published system (40.80) on a far cheaper model* — and the "exceeding the paper" CV
phrasing is now positively contraindicated, not merely unsupported.
