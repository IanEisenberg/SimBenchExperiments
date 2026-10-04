"""Tests for the pre-registered lever set: purity, composition, provenance."""

import pytest

from simbench_exp.levers import LEVER_REGISTRY, apply_lever
from simbench_exp.spec import PipelineSpec


def test_global_temp_appends_calibrator_and_is_pure():
    base = PipelineSpec(predictor="zero_shot")
    out = apply_lever(base, "recalib.global_temp", {"T": 1.5})
    assert out.calibrators == [{"name": "temp", "kwargs": {"T": 1.5}}]
    assert base.calibrators == []  # original untouched (pure)
    assert out.lever_path == ["recalib.global_temp"]
    assert out.config_hash() != base.config_hash()


def test_model_swap_changes_model_only():
    base = PipelineSpec(model="gemini-flash-lite", predictor="zero_shot")
    out = apply_lever(base, "model.swap", {"model": "qwen-72b"})
    assert out.model == "qwen-72b" and out.predictor == "zero_shot"


def test_predictor_swap_changes_predictor():
    base = PipelineSpec(predictor="zero_shot")
    out = apply_lever(base, "predictor.swap", {"predictor": "uniform"})
    assert out.predictor == "uniform"


def test_levers_compose_and_accumulate_path():
    s = PipelineSpec(predictor="zero_shot")
    s = apply_lever(s, "recalib.global_temp", {"T": 2.0})
    s = apply_lever(s, "recalib.dirichlet", {"alpha": 0.05})
    assert [c["name"] for c in s.calibrators] == ["temp", "dirichlet"]
    assert s.lever_path == ["recalib.global_temp", "recalib.dirichlet"]


def test_off_registry_raises():
    with pytest.raises(KeyError):
        apply_lever(PipelineSpec(), "not.a.lever", {})


def test_registry_levers_have_required_metadata():
    for lever in LEVER_REGISTRY.values():
        assert lever.hypothesis and lever.mechanism and lever.expected_direction
