"""Project paths and environment configuration.

Loads the gitignored `.env` (OpenRouter key, base URL) and centralizes the
directory layout so every module agrees on where data and caches live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repo root = two levels up from this file (src/scrye/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the repo root if present. Never hard-fail here; some commands
# (e.g. scoring tests) need no API key.
load_dotenv(REPO_ROOT / ".env")

# Directory layout (all gitignored except where noted).
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded SimBench CSVs (gitignored)
CACHE_DIR = DATA_DIR / "cache"      # on-disk LLM response cache (gitignored)
OUTPUTS_DIR = REPO_ROOT / "outputs"  # generated run artifacts/scratch (gitignored)
# Presentation-final, deliverable-grade figures live here and ARE tracked, so
# git versions only curated visuals while regenerable churn stays in outputs/.
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

for _d in (RAW_DIR, CACHE_DIR, OUTPUTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Put it in the gitignored .env at the "
                "repo root (see .env.example) or export it in your shell."
            )
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return cls(api_key=key, base_url=base_url)


# HuggingFace dataset coordinates for SimBench.
SIMBENCH_REPO_ID = "pitehu/SimBench"
SIMBENCH_FILES = {
    "pop": "SimBenchPop.csv",
    "grouped": "SimBenchGrouped.csv",
}

# Curated, validated-on-OpenRouter model ids spanning tiers/vendors, for the
# cross-model invariance ablation. Prices (per 1M tokens, prompt/completion) are
# indicative as of 2026-06 and will drift; verify before a large run.
OPENROUTER_MODELS = {
    "gemini-flash-lite": "google/gemini-2.5-flash-lite",   # ~$0.10 / $0.40
    "qwen-7b": "qwen/qwen-2.5-7b-instruct",                # ~$0.04 / $0.10
    "qwen-72b": "qwen/qwen-2.5-72b-instruct",              # ~$0.36 / $0.40
    "deepseek-chat": "deepseek/deepseek-chat-v3-0324",     # ~$0.20 / $0.77
}

# Default workhorse for development (cheap, fast, reliable).
DEFAULT_MODEL = OPENROUTER_MODELS["gemini-flash-lite"]

# The three questions the assignment requires us to report on, matched by a
# distinctive case-insensitive substring of the question text. Verified present
# in the downloaded CSVs (trust -> LatinoBarometro; the other two -> ESS).
REQUIRED_QUESTIONS = {
    "trust_president": "trust in the president",
    "gay_rights": "free to live their own life",
    "internet_use": "use the internet",
}

# Per-model OpenRouter prices, USD per 1M tokens (prompt, completion). Used ONLY
# as a fallback when a response omits native usage.cost. Keyed on the exact ids
# from OPENROUTER_MODELS; an unknown model estimates to $0 (native cost still wins).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "qwen/qwen-2.5-72b-instruct": (0.36, 0.40),
    "deepseek/deepseek-chat-v3-0324": (0.20, 0.77),
}


def price_for(model: str, prices: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    """(prompt, completion) $/Mtok for a model; (0.0, 0.0) if unknown."""
    table = MODEL_PRICES if prices is None else prices
    return table.get(model, (0.0, 0.0))
