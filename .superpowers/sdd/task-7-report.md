# Task 7 Report: Per-call OpenRouter cost capture in `LLMClient`

## What was implemented

### `src/simbench_exp/config.py`
- Added `MODEL_PRICES: dict[str, tuple[float, float]]` — 4 seeded models (gemini-2.0-flash-001, gemini-2.0-flash-lite-001, qwen-2.5-72b-instruct, deepseek-chat) with prompt/completion $/Mtok rates.
- Added `price_for(model, prices=None) -> tuple[float, float]` — returns `(0.0, 0.0)` for unknown models.

### `src/simbench_exp/llm.py`
- Updated import: `from .config import CACHE_DIR, OpenRouterConfig, price_for`
- Added `cost_usd: float = 0.0` field to `Usage` dataclass.
- Updated `Usage.as_dict()` to include `cost_usd`.
- Added module-level `estimate_cost(prompt_tokens, completion_tokens, model, prices=None) -> float`.
- Added module-level `cost_of_record(record, prices=None) -> float` — prefers `record["cost"]` (native), falls back to token estimate.
- Updated `complete()` to pass `extra_body={"usage": {"include": True}}` to `.create()` (outside the cache payload).
- Updated `complete()` to extract native cost from `usage.cost` / `usage.model_extra["cost"]`.
- Updated `complete()` to store `"cost": native_cost` in `record` and accumulate `self.usage.cost_usd += cost_of_record(record)`.

### `tests/test_llm_cost.py`
New test file with 5 tests covering all specified interfaces.

## TDD Evidence

**RED (before implementation):**
```
ERROR tests/test_llm_cost.py
ImportError: cannot import name 'price_for' from 'simbench_exp.config'
```

**GREEN (after implementation):**
```
tests/test_llm_cost.py .....  5 passed in 0.35s
```

## Files Changed
- `src/simbench_exp/config.py` — added `MODEL_PRICES`, `price_for`
- `src/simbench_exp/llm.py` — updated import, `Usage`, added helpers, updated `complete`
- `tests/test_llm_cost.py` — new test file (5 tests)

## Full Suite Summary
```
138 passed, 2 skipped in 0.49s
```

## Invariant Verification

### 1. Cache hit = $0 new spend
The cache-hit early return at `llm.py` lines ~157-163:
```python
if self.use_cache:
    hit = self._read_cache(key)
    if hit is not None:
        with self._lock:
            self.usage.cache_hits += 1
        return LLMResponse(...)   # <-- EARLY RETURN here
```
The cost accumulation `self.usage.cost_usd += cost_of_record(record)` only executes on the NON-cached path, after the live API call. Cache hits return before this block and never touch `cost_usd`.

### 2. Native cost preferred, token-estimate fallback
`cost_of_record` in `llm.py`:
```python
def cost_of_record(record: dict, prices: dict | None = None) -> float:
    native = record.get("cost")
    if native is not None:
        return float(native)
    return estimate_cost(...)
```
`price_for` returns `(0.0, 0.0)` for unknown models, so `estimate_cost` returns 0.0 rather than crashing.

### 3. Cache-key stability
`extra_body={"usage": {"include": True}}` is passed directly to `.create()`, NOT via `_request_payload()` or the `payload` dict that feeds `_cache_key()`. The cache key hash is computed from `payload` alone before the `.create()` call, so native cost opt-in never affects cache key computation.

### 4. `Usage.cost_usd` in `as_dict()`
```python
def as_dict(self) -> dict:
    return {
        ...
        "cost_usd": self.cost_usd,
    }
```
Defaults to `0.0` (dataclass field default). Confirmed by `test_usage_dict_includes_cost`.

---

## Scientific-Integrity Fix Report (2026-06-20)

Three changes applied to fix MODEL_PRICES key-mismatch and add spend-visibility guards.

### Fix 1 — Align MODEL_PRICES keys to real model ids (`src/simbench_exp/config.py`)

Replaced the old `MODEL_PRICES` dict (keyed on stale ids like `google/gemini-2.0-flash-001` and `deepseek/deepseek-chat`) with a new dict keyed on the EXACT ids from `OPENROUTER_MODELS`, using prices from the inline comments:

```python
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "qwen/qwen-2.5-72b-instruct": (0.36, 0.40),
    "deepseek/deepseek-chat-v3-0324": (0.20, 0.77),
}
```

Updated the explanatory comment above to reference "exact ids from OPENROUTER_MODELS" rather than old ids.

### Fix 2 — Warn on unmeasurable spend (`src/simbench_exp/llm.py`)

Added `import warnings` at the top. On the cache-MISS path, before the cost-accumulation lock block, added:

```python
if record.get("cost") is None and price_for(self.model) == (0.0, 0.0):
    warnings.warn(
        f"Spend for model '{self.model}' cannot be measured: no native cost from "
        "OpenRouter and no fallback entry in MODEL_PRICES. Cost recorded as $0; "
        "the cost cap may under-count actual spend.",
        stacklevel=2,
    )
```

This fires ONLY on cache misses where native cost is absent AND the model has no price-table entry. After Fix 1, the four registry models are all in MODEL_PRICES so the warning does NOT fire for them.

### Fix 3 — Pin cache-hit = $0 invariant (`tests/test_llm_cost.py`)

Added `test_cache_hit_costs_zero` — uses `monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-for-offline-test")` to construct `LLMClient` offline, pre-seeds cache with `_write_cache` (non-zero cost record), then calls `complete()` and asserts `resp.cached is True`, `client.usage.cache_hits == 1`, and `client.usage.cost_usd == 0.0`.

### Commands run and output

```
pytest tests/test_llm_cost.py -v
# 6 passed in 0.41s

pytest -q
# 139 passed, 2 skipped in 0.55s
```

No stray warnings in summary. New warning did not fire in offline suite.
