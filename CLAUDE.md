# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Scrye predicts the empirical distribution of human survey responses in [SimBench](https://huggingface.co/datasets/pitehu/SimBench) (Hu et al., arXiv 2510.17516). The goal is to beat the benchmark's naive uniform baseline using LLM-based methods. All LLM calls go through OpenRouter (OpenAI-compatible API, any vendor via a single model id string).

## Setup

```bash
uv sync --extra dev                 # create .venv and install all deps
uv run nbstripout --install         # one-time git filter for notebooks
cp .env.example .env                # then add OPENROUTER_API_KEY
```

The `.env` is gitignored. `OPENROUTER_API_KEY` is required for any live LLM call; scoring and parsing tests work without it.

## Common commands

```bash
# Tests
uv run pytest                       # all tests
uv run pytest tests/test_scoring.py # single file
uv run pytest -k "test_tvd"         # single test by name

# Scripts
uv run python scripts/download_data.py   # download SimBench CSVs to data/raw/
uv run python scripts/smoke_llm.py       # one live LLM call + cache check

# Deployable CLI — predict a distribution for any freeform question (the shipped
# best method: calibrated_commitment @ gemini-3.1-flash-lite; see scrye/ask.py)
uv run scrye-ask "Will remote work keep growing?" \
    --options "Agree;Somewhat agree;Somewhat disagree;Disagree" --as "US tech workers"

# Notebooks
uv run jupyter lab                  # select the .venv kernel
```

## Architecture

The core prediction loop is: **Record → Predictor → Calibrator → scored distribution**.

```
data.py          SimBenchRecord (typed row from the dataset)
  └── predict.py     Predictor ABC → ZeroShot / Uniform / Routing / PostStratification /
                                     Voting / Grounded{Voting,Averaging} / Enriched / ...
        └── calibrate.py  Calibrator ABC → Identity / Abstain / *TempScaling /
                                           EntropyTarget / Dirichlet / Chain / ...
              └── pipeline.py  Pipeline (composes one Predictor + one Calibrator;
                                         predict_batch runs threaded IO)
                    └── evaluate.py   evaluate() → tidy DataFrame per record
                          └── scoring.py  simbench_score(), bootstrap_ci(), TVD helpers
```

**Current best system** (Stage 17, sealed test): `calibrated_commitment` prompt
strategy @ `gemini-3.1-flash-lite` + `AbstainCalibrator` — faithful 35.21 → 40.73.
Two honest findings drive everything: (1) **the model is the dominant lever**
(faithful 2.5→3.1-flash-lite is +15.9 overall); the prompt/calibration stack adds
~+5.5 on top. (2) `AbstainCalibrator` (predict uniform on datasets the model fails)
is the MVP intervention. The task-kind router (`RoutingPredictor` + `taskkind.py`)
*tied* cc+abstain on test, so reporting leads with the simpler cc+abstain. This is
what `scrye/ask.py` ships.

**Key modules:**

- `config.py` — all paths (`DATA_DIR`, `CACHE_DIR`, `FIGURES_DIR`), `.env` loading, `OPENROUTER_MODELS` short-key → full-id map, `DEFAULT_MODEL`.
- `data.py` — `SimBenchRecord` dataclass + `load_all()` which returns `{"pop": [...], "grouped": [...]}`.
- `llm.py` — `LLMClient`: SHA-256 keyed on-disk JSON cache in `data/cache/`; `Usage` tracker; thread-safe.
- `persona.py` — `PromptStrategy` registry (`STRATEGIES`): ten swappable demographic-conditioning styles (`simbench_faithful`, `representative_sample`, `persona_embodiment`, `anti_flattening`, `contextualized`, `calibrated_commitment`, `diversity_elicitation`, `outside_view`, `entropy_first`, `superforecaster`), plus the dynamic `TaskContextStrategy`. `simbench_faithful` reproduces the original paper's baseline exactly; `calibrated_commitment` is the validated winner. **This is the main experiment-direction surface** — a new conditioning style is one `PromptStrategy` subclass + a registry entry, and it is then available to both the notebooks and the search loop (each strategy auto-registers as a predictor in `experiment.PREDICTOR_REGISTRY`).
- `splits.py` — leave-family-out train/dev/val/test split engine. Required questions are pinned to test. Split by `(dataset_name, input_template)` family — never random row split.
- `experiment.py` — `build_pipeline()` factory, `PREDICTOR_REGISTRY`, `compare()` (multi-system topline table), `model_sweep()`.
- `spec.py` — `PipelineSpec`: serializable, SHA-256 content-addressed pipeline description used by the search loop.
- `levers.py` — `LEVER_REGISTRY`: pre-registered theory-motivated interventions (calibrator wraps, model swaps, predictor swaps). Adding off-registry levers requires human approval.
- `search.py` — `run_search()`: guarded autonomous search loop. Steps: propose lever → score on dev → if dev-best, one Ladder-guarded val query. Halts on budget / off-registry proposal / gate fire.
- `ladder.py` — `LadderGate` (Blum & Hardt 2015): val queries accepted only when improvement > eta (noise guard derived from bootstrap). Bounds generalization error under adaptive search.
- `ledger.py` — `Ledger` / `ExperimentNode`: DAG of all search steps, serialized to disk.
- `manifest.py` — `RunManifest`: the frozen, machine-readable record of a run's data selection + params (split seed/fractions/unit, goal, allowed levers, budgets, eta, model, dataset fingerprint). Written beside the ledger as `<run>.manifest.json`; makes a run reproducible.
- `results.py` — query helpers over a loaded `Ledger` (`nodes_frame`, `best_progression`, `spec_from_node`).
- `decompose.py` — error decomposition: splits each item's `TVD(P,Q)` into `concentration_err` (wrong spread/profile) + `location_err` (right shape, wrong options), both ≥ 0 and summing to TVD, plus signed facets (entropy gap, mode accuracy). `decompose_records()` turns a run's `results.json` into a per-item frame.
- `predict.py` — the 11 `Predictor` subclasses. Beyond the simple ones: `RoutingPredictor` (dispatch by task kind), `VotingEnsemblePredictor` (K dispositional agents each cast one discrete vote), and the grounded-electorate family (`GroundedVoting`/`GroundedAveraging`/`EnrichedGrounded`/`Ensemble`) built on `nemotron.py`.
- `taskkind.py` — upfront task-kind classifier (`LLMTaskClassifier` + `heuristic_task_kind` fallback) + `KIND_ROUTES` map (opinion/knowledge→task_context, risky_choice→abstain, moral→abstain, else→cc). Classifies from question *content* so it generalizes to unseen datasets; consumed by `RoutingPredictor`.
- `nemotron.py` — `PersonaBank` over a local slim cache of `nvidia/Nemotron-Personas-USA` (built by `scripts/prepare_nemotron.py`): a census-grounded synthetic US electorate, sampled whole-corpus or matched to a SimBench segment. Feeds the grounded predictors.
- `worldview.py` — worldview enrichment + ideology calibration (Stage 18): infers a sampled values/ideology axis per persona and post-stratifies the panel onto OpinionQA's real `POLIDEOLOGY` marginal. The persona line is closed (still below `calibrated_commitment`); kept for reference.
- `distributions.py` — post-stratification weight tables: `SegmentWeights` from SimBench `group_size` counts behind a swappable `WeightSource` protocol; consumed by `PostStratificationPredictor`.
- `ask.py` / `cli_ask.py` — the deployable: `ask()` runs the shipped best method on a *cold* freeform question (no dataset / no ground truth, so abstention and routing reduce to one `calibrated_commitment` call), exposed as the `scrye-ask` console script.
- `viz.py` — diagnostic matplotlib figures from an `evaluate()` frame (score histogram, score-by-dataset, score-vs-entropy, calibration scatter). `report.py` — production `score_fn` for the search loop + `final_report()` (test-once headline with Ladder-widened CI). `tree_view.py` — ASCII renderer of the search DAG (✓ accepted / ✗ Ladder-rejected / · dev-only).

**Scoring:** `S = 100 × (1 − TVD(P,Q) / TVD(P,U))`. Score 0 = uniform baseline; 100 = perfect match; negative = worse than uniform. Always use `build_normalizers()` on the **full split** before scoring subsamples — the Eq. 2 denominator must be dataset-level, not subsample-level.

**Caching:** every LLM completion is keyed by `hash(model, messages, sampling_params)`. Re-running any experiment against the same calls is free and deterministic. `data/cache/` is gitignored; `outputs/` is gitignored (scratch); `docs/figures/` is tracked (deliverable-grade visuals only). `deliverables/` is tracked and holds the final hand-off artifacts (Part I/II HTML decks, the SimBench explorer, the per-question predictions CSV), built by `scripts/build_deck*.py`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `00_starter.ipynb` | Initial exploration, sanity checks |
| `00_dataexploration.ipynb` | Dataset deep-dive; also uses `scrye.decompose` for the concentration-vs-location TVD split |
| `01_experiments.ipynb` | Main system comparisons and ablations |
| `02_persona_factory.ipynb` | Persona strategy ablation |
| `03_experiment_results.ipynb` | Analysis of search results from the Ledger |

Notebooks import `scrye` and delegate logic to the library; reusable code lives in `src/scrye/`, not in cells. `nbstripout` strips outputs before commit.

## Leakage discipline

The three required questions (trust_president, gay_rights, internet_use) are **always pinned to the test bucket** via `splits.py`. The `val` bucket is used only for Ladder-gated model/hyperparameter selection. Never pass `val` or `test` data to `compare()` during iterative development — use `dev` only.

## Experiment workflow (preregistration)

Experiments run as **preregistered stages**, tracked in the human-facing log at
`docs/experiments/` (one markdown file per stage + a `README.md` index). A stage
is a batch of experiments sharing one hypothesis and one decision rule. This is
the layer where we *plan and discuss before running*, and *record after*.

The loop, every round:

1. **Preregister** — draft `docs/experiments/stage-NN-<name>.md` *before* running:
   hypothesis, the exact configs to run, data + budget, and the **decision rule**
   (what counts as a win). The decision rule is fixed before looking at results.
   Get user sign-off. No `val`/`test` contact at this point — see Leakage
   discipline above.
2. **Run + iterate** — execute on `dev`. Every LLM call caches to `data/cache/`,
   so re-runs are free and deterministic; iterate a few times freely. Search-loop
   runs also append nodes to `outputs/ledger/<run>.jsonl` with a
   `<run>.manifest.json` sidecar (the machine-level frozen config).
3. **Record** — append results to the *same* stage file: topline table, run-file
   pointers, what was learned, and what it implies for the next stage. Update the
   status row in `docs/experiments/README.md`.
4. **Next round** — preregister `stage-NN+1`, informed by the result. Do **not**
   start a new stage's runs before its preregistration is written and approved.

**Artifacts per run** (all gitignored except the stage markdown):

- `data/cache/<sha>.json` — every raw LLM call (text, tokens, cost). Always written.
- `outputs/ledger/<run>.jsonl` + `<run>.manifest.json` — search-loop tree + frozen config.
- `docs/experiments/stage-NN-*.md` — the human plan + results (tracked in git).

`docs/experiments/` tracks the **science** (hypotheses, decision rules, results);
beans track the **work** (tasks, status). Keep them separate.
