"""Cost capture: native-cost preference, token-price fallback, cache-hit = $0."""

from scrye.config import price_for
from scrye.llm import LLMClient, Usage, _cache_key, cost_of_record, estimate_cost


def test_estimate_cost_uses_price_table():
    prices = {"m": (1.0, 2.0)}  # $1/Mtok in, $2/Mtok out
    # 1e6 prompt tokens -> $1; 1e6 completion -> $2; total $3.
    assert abs(estimate_cost(1_000_000, 1_000_000, "m", prices) - 3.0) < 1e-9


def test_unknown_model_estimates_zero():
    assert price_for("nope", {"m": (1.0, 2.0)}) == (0.0, 0.0)
    assert estimate_cost(1_000_000, 1_000_000, "nope", {"m": (1.0, 2.0)}) == 0.0


def test_cost_of_record_prefers_native_cost():
    rec = {"cost": 0.42, "prompt_tokens": 9, "completion_tokens": 9, "model": "m"}
    assert cost_of_record(rec, {"m": (1000.0, 1000.0)}) == 0.42  # native wins over estimate


def test_cost_of_record_falls_back_to_estimate():
    rec = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "model": "m"}
    assert abs(cost_of_record(rec, {"m": (1.0, 2.0)}) - 1.0) < 1e-9


def test_usage_dict_includes_cost():
    u = Usage()
    u.cost_usd = 1.25
    assert u.as_dict()["cost_usd"] == 1.25


def test_cache_hit_costs_zero(monkeypatch, tmp_path):
    """Cache hits must add $0 to usage.cost_usd even if the cached record has a cost."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-for-offline-test")

    model = "google/gemini-2.5-flash-lite"
    client = LLMClient(model, cache_dir=tmp_path, use_cache=True)

    messages = [{"role": "user", "content": "hello"}]
    payload = client._request_payload(messages)
    key = _cache_key(payload)
    # Pre-seed the cache with a record that has a non-zero cost.
    client._write_cache(key, {
        "text": "hi",
        "model": model,
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "cost": 0.01,
    })

    resp = client.complete(messages)
    assert resp.cached is True
    assert client.usage.cache_hits == 1
    assert client.usage.cost_usd == 0.0  # cache hit adds NO spend
