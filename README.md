# Scrye — SimBench Survey-Distribution Prediction

Scrye predicts the **empirical distribution of human survey responses** in
[SimBench](https://huggingface.co/datasets/pitehu/SimBench) (Hu et al., arXiv
2510.17516). Given a survey question, its answer options, and a demographic group,
we predict *what fraction of that group picks each option* — a histogram over
options, not a single label — and score it against the real human distribution.

The benchmark is hard: published state of the art is only ~41/100, and many tested
models score *below* the naive uniform baseline. The work here is run as a
**disciplined, AI-driven scientific search** — preregistered stages, a held-out
val/test ladder, and a guarded autonomous lever loop — so the numbers are honest,
not overfit.

**Headline result (sealed test, 3,592 records):** the shipped system —
`calibrated_commitment` prompt + dataset-level abstention on `gemini-3.1-flash-lite`
— scores **40.9 overall** (grouped 44.0, pop 37.6). Two honest reads: (1) the
**model is the dominant lever** — swapping `gemini-2.5` → `3.1-flash-lite` alone is
+15.9 overall; the entire method stack adds +5.5 on top, real but an order of
magnitude smaller; (2) the harness **reproduces the paper** — our byte-faithful
baseline on Qwen2.5-72B gives split-avg S = 26.83 [24.39, 29.35], bracketing the
paper's 27.61. See [`docs/experiments/stage-17-final-test.md`](docs/experiments/stage-17-final-test.md).

## Where to start reading

- **Plain-language tour** → [`docs/executive-overview.md`](docs/executive-overview.md)
  — the problem, the method, every lever we pulled and what it bought, and what we
  tried and rejected. **Read this first.** Interactive version: [`docs/overview.html`](docs/overview.html).
- **The science** → [`docs/experiments/README.md`](docs/experiments/README.md) — the
  preregistered experiment log: one markdown file per stage (hypothesis + decision
  rule written *before* the run, results appended after). 18 stages, baseline → sealed test.
- **The work** → `beans list --ready` — this project tracks tasks with the `beans`
  CLI (an agentic issue tracker), not scattered TODOs.
- **Contributor / agent guidance** → [`CLAUDE.md`](CLAUDE.md) — architecture deep-dive,
  conventions, and the leakage rules every change must respect.
- **Design specs & gameplan** → [`docs/superpowers/specs/`](docs/superpowers/specs/).
- **Deliverables** (Part I / Part II decks, interactive explorer, prediction CSV) →
  [`deliverables/`](deliverables/) and [`docs/presentation/`](docs/presentation/).

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                 # create venv, install deps
uv run nbstripout --install         # one-time: strip notebook outputs on commit
cp .env.example .env                # then add your OPENROUTER_API_KEY
```

`nbstripout --install` configures a local git filter (recorded in `.gitattributes`)
so executed notebook outputs and execution counts are stripped before commit — git
tracks only source-cell changes, while your working copy keeps its rendered outputs.
The filter lives in `.git/config`, which isn't committed, so each fresh clone runs
this once.

The `.env` file is gitignored and holds the OpenRouter key. All LLM access goes
through OpenRouter (one OpenAI-compatible endpoint, any vendor by model id).
`OPENROUTER_API_KEY` is required for any live LLM call; scoring and parsing tests
run without it.

## Quick start

```bash
uv run python scripts/download_data.py   # download SimBench + print summary
uv run python scripts/smoke_llm.py       # one live call, verify cache works
uv run pytest                            # unit tests (no API key needed)
uv run jupyter lab                       # notebooks (select the .venv kernel)
```

**Try the shipped system on your own question.** `scrye-ask` runs the best method
(the `calibrated_commitment` prompt on `gemini-3.1-flash-lite`) for any freeform
question — a cold question has no ground truth, so the method reduces cleanly to one
calibrated call parsed into a distribution:

```bash
uv run scrye-ask "Will remote work keep growing?" \
    --options "Agree;Somewhat agree;Somewhat disagree;Disagree" \
    --as "US tech workers in 2025"

uv run scrye-ask                         # bare → interactive wizard
uv run scrye-ask ... --json              # strict JSON for scripting
```

## Architecture

The core prediction loop is **Record → Predictor → Calibrator → scored distribution**:

```
data.py        SimBenchRecord (typed row: question, options, segment, truth P)
  └── predict.py     Predictor ABC → ZeroShot / Uniform / PostStratification /
                                     Voting / Routing
        └── calibrate.py  Calibrator ABC → Identity / TempScaling / Abstain / ...
              └── pipeline.py  Pipeline = one Predictor + one Calibrator
                                          (predict_batch runs threaded IO)
                    └── evaluate.py   evaluate() → tidy per-record DataFrame
                          └── scoring.py  simbench_score(), bootstrap_ci(), TVD
```

### Package layout (`src/scrye/`)

| Module | Purpose |
|---|---|
| `config.py` | Paths (`DATA_DIR`, `CACHE_DIR`, `FIGURES_DIR`), `.env` loading, validated `OPENROUTER_MODELS` short-key → id map, `DEFAULT_MODEL`. |
| `data.py` | Download SimBench; `SimBenchRecord` dataclass; `load_all()` → `{"pop": [...], "grouped": [...]}`. |
| `scoring.py` | SimBench score `S = 100·(1 − TVD(P,Q)/TVD(P,U))`, TVD helpers, bootstrap CIs, response entropy. |
| `llm.py` | OpenRouter client: SHA-256 on-disk JSON cache in `data/cache/`, retries, thread-safe `Usage` tracker. |
| `persona.py` | `PromptStrategy` registry (`STRATEGIES`) — swappable demographic-conditioning styles (`simbench_faithful`, `anti_flattening`, `calibrated_commitment`, …). **The main experiment surface.** |
| `predict.py` | `Predictor` ABC and implementations (zero-shot, uniform, post-stratification, voting ensemble, task-kind router). |
| `calibrate.py` | `Calibrator` ABC: identity, entropy-target scaling, Dirichlet, `AbstainCalibrator`, chaining. |
| `pipeline.py` / `evaluate.py` | Compose predictor + calibrator; run batched threaded IO; `build_normalizers()` + a tidy scored frame. |
| `splits.py` | Leave-family-out dev/val/test engine. Splits by `(dataset_name, input_template)` family — never random rows. Required questions pinned to test. |
| `experiment.py` | `build_pipeline()` factory, `PREDICTOR_REGISTRY`, `compare()` topline table, `model_sweep()`. |
| `spec.py` / `manifest.py` | Serializable, content-addressed `PipelineSpec`; `RunManifest` frozen run config (seed, fractions, budgets, eta, model, dataset fingerprint). |
| `levers.py` / `search.py` | Preregistered `LEVER_REGISTRY`; `run_search()` guarded autonomous loop (propose lever → score on dev → one Ladder-gated val query if dev-best). |
| `ladder.py` | `LadderGate` (Blum & Hardt 2015): val queries accepted only when improvement > noise eta — bounds generalization error under adaptive search. |
| `ledger.py` / `results.py` | DAG of all search steps serialized to disk; query helpers over a loaded `Ledger`. |
| `decompose.py` | Error decomposition: split each `TVD(P,Q)` into `concentration_err` (wrong spread) + `location_err` (wrong options), plus signed entropy/mode facets. |
| `ask.py` / `cli_ask.py` | The deployable `scrye-ask` core + CLI — best method on a cold freeform question. |

### Notebooks (`notebooks/`)

Notebooks stay thin: they `import scrye` and call the library; reusable logic
graduates into `src/scrye/`.

| Notebook | Purpose |
|---|---|
| `00_dataexploration.ipynb` | SimBench dataset exploration |
| `00_starter.ipynb` | Initial sanity checks |
| `01_experiments.ipynb` | Main system comparisons and ablations |
| `02_persona_factory.ipynb` | Persona / conditioning-strategy ablation |
| `03_experiment_results.ipynb` | Analysis of search results from the Ledger |

## Scoring

```
S = 100 × ( 1 − TVD(P, Q) / TVD(P, U) )      TVD = ½·Σ|pᵢ − qᵢ|
```

- **S = 100** → perfect match; **S = 0** → no better than uniform (the naive
  baseline, *the bar to beat*); **S < 0** → confidently worse than uniform.
- Best zero-shot in the paper ≈ 40.8 (Claude-3.7-Sonnet), so the real target is
  improving on strong zero-shot, not merely beating uniform.
- Always call `build_normalizers()` on the **full split** before scoring a
  subsample — the Eq. 2 denominator is dataset-level, not subsample-level.

### Key data facts (verified against the downloaded CSVs)

- **Pop split:** 7,167 cases across 20 source datasets (default group prompt per question).
- **Grouped split:** 6,343 cases across 5 surveys (ESS, Afrobarometer, LatinoBarómetro, OpinionQA, ISSP); segments are country × one attribute, encoded in `record.segment` (e.g. `{"cntry": "Finland", "age_group": "30-49"}`).
- No train/dev split ships with the benchmark — any holdout is constructed by us. Ground-truth distributions are public, so leakage discipline is documented and enforced.

## Leakage discipline

The three required questions (`trust_president`, `gay_rights`, `internet_use`) are
**always pinned to the test bucket** via `splits.py`. The `val` bucket is touched
only at Ladder-gated confirmation steps; **`test` is opened exactly once**, at the
end (Stage 17). During iterative development use **`dev` only** — never pass `val`
or `test` to `compare()`.

## Experiment workflow (preregistration)

Experiments run as **preregistered stages** in [`docs/experiments/`](docs/experiments/):
draft `stage-NN-<name>.md` (hypothesis + exact configs + decision rule) and get
sign-off *before* running; iterate on `dev`; then append results to the same file
and update the status row in the index. The decision rule is fixed before anyone
looks at the data. `docs/experiments/` tracks the **science**; beans track the
**work** — keep them separate.

## Caching & reproducibility

Every LLM completion is keyed by `hash(model, messages, sampling params)` and
cached as JSON under `data/cache/`. Re-running an experiment or ablation that
touches the same calls is free and deterministic — reported numbers regenerate
from cache without re-spending tokens. `data/cache/` and `outputs/` are gitignored
(everything there regenerates); `docs/figures/` is tracked (deliverable-grade
visuals only).
