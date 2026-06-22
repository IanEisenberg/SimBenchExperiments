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

(On the test family-split specifically, faithful @ Qwen2.5-72B gives split-avg
S = 25.04 — directionally consistent but lower and noisier, with a high 14%
transient-API-failure rate excluded; the dedicated stratified sample above, at
1.3% failures, is the clean reproduction.)

## One-shot TEST — full lineage

Sealed test set, 3592 recs (grouped 1874, pop 1718; required-Q 259). Runs:
`outputs/runs/2026-06-21-TEST-{final,lineage,faithful-models}.results.json`.
The intermediary systems and the two baseline models were measured on test as
**completeness measurement** — the final system was fixed on val before any test
contact, so this is not selection-on-test.

| system | model | overall | grouped | pop | required |
|---|---|---|---|---|---|
| faithful | gemini-2.5-flash-lite | 19.27 | 13.77 | 25.28 | −4.51 |
| faithful | Qwen2.5-72B-Instruct | 25.40 | 20.97 | 29.10 | −1.26 |
| faithful | gemini-3.1-flash-lite | 35.21 | 39.19 | 30.88 | 51.48 |
| anti_flattening | gemini-3.1 | 39.68 | 42.69 | 36.39 | 51.28 |
| **calibrated_commitment + abstain** | gemini-3.1 | **40.93** | **44.02** | 37.55 | **55.29** |
| **task-kind router (val-confirmed)** | gemini-3.1 | 40.73 | 43.45 | **37.77** | 54.14 |

**Two honest headlines:**

1. **The model is the dominant lever.** faithful 2.5 → 3.1-flash-lite is
   **+15.9 overall / +25.4 grouped** on test; the required (grouped opinion)
   questions jump from **−4.5 → +51.5** purely from the model swap. gemini-3.1 is
   dramatically stronger on the demographic surveys than both 2.5 and Qwen-72B.
   The entire method stack (faithful@3.1 → final) adds **+5.5 overall** on top of
   that — real and CI-clean over faithful@3.1, but an order of magnitude smaller
   than the model lever.

2. **The router does NOT beat cc+abstain on test — they are statistically tied.**
   cc+abstain is marginally ahead on overall (+0.20), grouped (+0.57), and the
   required questions (+1.15); the router is ahead only on pop (+0.22) — all well
   within the bootstrap noise (~±1.5). **The router's val advantage (+1.10 pooled,
   Stage 16) did not transfer**; the task-context routing washed out on held-out
   test. Per leakage discipline we do **not** re-select the system on test — the
   router remains the val-confirmed system — but the honest read is that the
   simpler **cc+abstain** is equivalent here, and the router's extra machinery (a
   per-item LLM classifier + task-context corpus) buys nothing measurable on test.

**vs faithful @ 3.1 (CI, the headline win):** router overall **+5.52
[+4.29, +6.80]**, grouped +4.26, pop +6.89, required +2.66 [+0.03, +5.20] — all
exclude 0. cc+abstain is essentially identical (+5.72 overall).

## Final system — the simpler method comes to the fore

Two systems are statistically equivalent on test (≈40.8 overall):

- **`calibrated_commitment` + `AbstainCalibrator`** (Stage 12) — one prompt + a
  per-dataset uniform fallback. **No classifier, no per-item routing.**
- **task-kind router** (Stage 16) — an upfront LLM classifier dispatching to
  `task_context` / `voting`→`abstain` / `cc` per kind.

The router was the more interesting idea and it **won on val** (+1.10 pooled,
Stage 16). But that lift **did not transfer to test** — the two tie, with
`cc+abstain` marginally ahead overall. So for a deployable recommendation the
**simpler `cc+abstain` leads**: it matches the router's test accuracy at a
fraction of the inference cost (one call per item, no task classification, no
sibling-item corpus).

**The val ↔ test conflict** is the honest story here. On dev/val the task-context
route reliably helped the opinion surveys; the router packaged that into a
generalizable rule and cleared the val gate. On the sealed test family-split that
edge vanished — task-context neither helped nor hurt grouped beyond noise. This is
the textbook reason a held-out test set exists: a small val win, accumulated over
three val touches, can fail to replicate. The robust, replicated result is the
**+5.5 method gain over `faithful@3.1`** (shared by both systems) on top of the
**+15.9 model gain** — not the router-over-`cc+abstain` increment.

Per leakage discipline we do not re-select on test, so the **router remains the
val-confirmed system of record**; but the **reporting leads with `cc+abstain`** as
the simpler, equally-good method. Both @ `gemini-3.1-flash-lite`.

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
