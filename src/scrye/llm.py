"""Provider-agnostic LLM client over OpenRouter, with on-disk response caching.

OpenRouter exposes an OpenAI-compatible API, so we drive it with the `openai`
SDK pointed at the OpenRouter base URL. A single `model` string selects any
vendor (Gemini, Qwen, DeepSeek, Claude, GPT, ...), which makes the cross-model
ablation a config sweep rather than N integrations.

Caching is the rigor lever: every completion is keyed by a hash of (model,
messages, sampling params) and stored as JSON on disk. Re-running an experiment
or an ablation that touches the same calls is then free and deterministic, so
the reported numbers are reproducible from cache without re-spending tokens.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from .config import CACHE_DIR, OpenRouterConfig, price_for


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str,
                  prices: dict | None = None) -> float:
    """USD cost from token counts and a per-Mtok price table."""
    p_in, p_out = price_for(model, prices)
    return prompt_tokens / 1e6 * p_in + completion_tokens / 1e6 * p_out


def cost_of_record(record: dict, prices: dict | None = None) -> float:
    """Prefer OpenRouter's native `cost`; else estimate from tokens."""
    native = record.get("cost")
    if native is not None:
        return float(native)
    return estimate_cost(record.get("prompt_tokens", 0), record.get("completion_tokens", 0),
                         record.get("model", ""), prices)


@dataclass
class Usage:
    """Running totals for one client instance."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool
    raw: dict = field(default_factory=dict)


class LLMClient:
    """Cached chat-completion client.

    Args:
        model: OpenRouter model id (e.g. "google/gemini-2.0-flash-001",
            "qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat").
        temperature, max_tokens, top_p, seed: default sampling params; can be
            overridden per call.
        cache_dir: where to store cached responses (defaults to data/cache).
        use_cache: set False to bypass the cache entirely.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        seed: int | None = 0,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 4,
        timeout: float = 60.0,
    ) -> None:
        cfg = OpenRouterConfig.from_env()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._lock = threading.Lock()
        self._client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    # -- caching -----------------------------------------------------------
    def _request_payload(self, messages: list[dict], **overrides) -> dict:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "top_p": overrides.get("top_p", self.top_p),
        }
        seed = overrides.get("seed", self.seed)
        if seed is not None:
            params["seed"] = seed
        return params

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, key: str, value: dict) -> None:
        try:
            self._cache_path(key).write_text(json.dumps(value, ensure_ascii=False))
        except OSError:
            pass

    # -- public API --------------------------------------------------------
    def complete(self, messages: list[dict], **overrides) -> LLMResponse:
        """Run one chat completion, returning the assistant text.

        Identical (params, messages) hit the on-disk cache and cost nothing.
        """
        payload = self._request_payload(messages, **overrides)
        key = _cache_key(payload)

        if self.use_cache:
            hit = self._read_cache(key)
            if hit is not None:
                with self._lock:
                    self.usage.cache_hits += 1
                return LLMResponse(
                    text=hit.get("text", ""), model=self.model, cached=True, raw=hit
                )

        completion = self._client.chat.completions.create(
            **payload, extra_body={"usage": {"include": True}}
        )
        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        native_cost = getattr(usage, "cost", None)
        if native_cost is None and usage is not None:
            native_cost = (getattr(usage, "model_extra", None) or {}).get("cost")
        record = {
            "text": text,
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cost": native_cost,  # may be None -> estimated below
            "params": {k: v for k, v in payload.items() if k != "messages"},
        }

        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += record["prompt_tokens"]
            self.usage.completion_tokens += record["completion_tokens"]
            self.usage.cost_usd += cost_of_record(record)  # new spend only

        if self.use_cache:
            self._write_cache(key, record)

        return LLMResponse(text=text, model=self.model, cached=False, raw=record)

    def prompt(self, user: str, system: str | None = None, **overrides) -> str:
        """Convenience wrapper for a single user (and optional system) message."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.complete(messages, **overrides).text
