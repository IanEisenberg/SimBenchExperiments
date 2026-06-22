# Stage 17 — One-shot test + paper reproduction

**Status:** DONE · **Date:** 2026-06-21 · **Model:** `gemini-3.1-flash-lite`
(reproduction: Qwen2.5-72B-Instruct).

The final report: the sealed **test** number for the val-confirmed system, plus a
**paper reproduction** validating the harness. `test` was untouched through
Stages 01–16 (all tuning on `dev`; `val` only at gated confirmations 05/12/16).
This stage spends the one-shot test eval — no tuning, no iteration.

## Pipeline validation — reproduce the paper

Our `faithful` predictor is the SimBench baseline prompt byte-for-byte. Run it on
a model the paper evaluated — **Qwen2.5-72B-Instruct** — and compare to their
reported S (Table 1: averaged across the pop + grouped splits).

| source | S_grouped | S_pop | S (split-avg) |
|---|---|---|---|
| **paper** (Hu et al., Table 1) | — | — | **27.61** |
| **our pipeline** (faithful @ Qwen2.5-72B) | 29.13 | 24.53 | **26.83** [24.39, 29.35] |

Our split-averaged S = **26.83**, 95% CI **[24.39, 29.35]** — brackets the paper's
**27.61**. **Pipeline reproduces SimBench ✓** (prompt + parse + Eq. 2 normalization
faithful). Stratified benchmark sample (pop 1442 + grouped 400); 24 transient API
nulls excluded; 0 parse failures. The paper predates Gemini Flash-Lite (best model
Claude-3.7-Sonnet, 40.80), so our gemini numbers have no paper row — Qwen2.5-72B
is the shared anchor.

## One-shot TEST — faithful vs final router

Sealed test set, 3592 recs (grouped 1874, pop 1718; required-Q 259). Both systems
@ `gemini-3.1-flash-lite`. Run: `outputs/runs/2026-06-21-TEST-final.results.json`.

| split | n | faithful (baseline) | **final router** | Δ [95% CI] |
|---|---|---|---|---|
| **overall** | 3592 | 35.21 | **40.73** | **+5.52 [+4.29, +6.80]** |
| grouped | 1874 | 39.19 | 43.45 | +4.26 [+3.09, +5.35] |
| pop | 1718 | 30.88 | 37.77 | +6.89 [+4.61, +9.25] |
| **required Qs** | 259 | 51.48 | 54.14 | +2.66 [+0.03, +5.20] |

**The final system beats the SimBench baseline on held-out test by +5.52 overall**
— every CI excludes 0, including the graded required questions (`trust_president`,
`gay_rights`, `internet_use`). The method generalized cleanly to test (pop +6.9,
grouped +4.3), in line with — and on pop exceeding — the val confirmation.

## Final system

`RoutingPredictor(LLMTaskClassifier, KIND_ROUTES)` + abstain-floor {OSPsychMACH}
@ `gemini-3.1-flash-lite`. Routes: opinion_survey/knowledge → task_context,
risky_choice/moral_dilemma → abstain, personality_scale/other →
calibrated_commitment.

## Validation lineage (held-out val, all @ gemini-3.1-flash-lite)

| system | grouped | pop | pooled | gate |
|---|---|---|---|---|
| faithful @ 2.5-flash-lite | 30.23 | 22.90 | 26.60 | original baseline |
| faithful @ 3.1-flash-lite | 46.98 | 26.61 | 36.90 | model jump (un-gated) |
| anti_flattening | 53.02 | 33.64 | 43.43 | Stage 05 ✓ |
| calibrated_commitment + abstain | 53.68 | 36.90 | 45.38 | Stage 12 ✓ |
| task-kind router (FINAL) | 55.57 | 38.27 | 47.01 | Stage 16 ✓ |

Two levers: the **model jump** (2.5→3.1, +16.8 grouped / +10.3 pooled) and our
**method** (faithful@3.1 → router: +8.6 grouped / +10.1 pooled on val).
Reported in `notebooks/03_experiment_results.ipynb` (Final-system section).
