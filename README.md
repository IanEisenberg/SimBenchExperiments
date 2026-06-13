# Scrye — SimBench Survey-Distribution Prediction

Mini-project: predict the empirical distribution of human survey responses in
[SimBench](https://huggingface.co/datasets/pitehu/SimBench) (Hu et al., arXiv
2510.17516), improving on the benchmark's naive baseline, with a counterfactual
sensitivity metric and a feedback-architecture design (Part II).

See `docs/superpowers/specs/` for the gameplan and design.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                 # create venv, install deps
cp .env.example .env                # then add your OPENROUTER_API_KEY
```

The `.env` file is gitignored and holds the OpenRouter key. All LLM access goes
through OpenRouter (one OpenAI-compatible endpoint, any vendor by model id).

## Quick start

```bash
uv run python scripts/download_data.py   # download SimBench + print summary
uv run python scripts/smoke_llm.py       # one live call, verify cache works
uv run pytest                            # unit tests (scoring + parsing)
```

## Package layout (`src/scrye/`)

| Module | Purpose |
|---|---|
| `config.py` | Paths, `.env` loading, SimBench coords, validated OpenRouter model ids. |
| `data.py` | Download SimBench; parse rows into typed `SimBenchRecord` (question, options, segment, empirical distribution). |
| `scoring.py` | SimBench score `S = 100·(1 − TVD(P,Q)/TVD(P,U))`, TVD helpers, bootstrap CIs. |
| `llm.py` | Provider-agnostic OpenRouter client with on-disk response caching, retries, usage tracking. |

### Key data facts (verified against the downloaded CSVs)

- **Pop split:** 7,167 cases across 20 source datasets (default group prompt per question).
- **Grouped split:** 6,343 cases across 5 surveys (ESS, Afrobarometer, LatinoBarometro, OpinionQA, ISSP); segments are country × one attribute, encoded in `record.segment` (e.g. `{"cntry": "Finland", "age_group": "30-49"}`).
- No train/dev split ships with the benchmark — any holdout is constructed by us. Ground-truth distributions are public, so leakage discipline is documented in the design.

### Scoring anchors

- `S = 100` → perfect match; `S = 0` → no better than uniform (the naive baseline); `S < 0` → worse than uniform.
- Best zero-shot model in the paper ≈ 40.8 (Claude-3.7-Sonnet), so the real target is improving on strong zero-shot, not beating uniform.

## Caching & reproducibility

Every LLM completion is keyed by `hash(model, messages, sampling params)` and
cached as JSON under `data/cache/`. Re-running an experiment or ablation that
touches the same calls is free and deterministic — reported numbers regenerate
from cache without re-spending tokens.
