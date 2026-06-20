"""Cost capture: native-cost preference, token-price fallback, cache-hit = $0."""

from scrye.config import price_for
from scrye.llm import Usage, cost_of_record, estimate_cost


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
