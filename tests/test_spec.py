"""Tests for PipelineSpec: stable content-addressing and pipeline assembly."""

from scrye.spec import PipelineSpec, build_from_spec


def test_config_hash_is_stable_and_short():
    s = PipelineSpec(model="gemini-flash-lite", predictor="zero_shot")
    h = s.config_hash()
    assert isinstance(h, str) and len(h) == 16
    assert s.config_hash() == h  # deterministic across calls


def test_config_hash_ignores_lever_path_but_tracks_resolved_config():
    # lever_path is provenance only — two specs with the same resolved config
    # but different provenance are the SAME experiment (same hash).
    a = PipelineSpec(predictor="zero_shot", lever_path=["x"])
    b = PipelineSpec(predictor="zero_shot", lever_path=["y", "z"])
    assert a.config_hash() == b.config_hash()


def test_config_hash_changes_with_calibrator():
    a = PipelineSpec(predictor="zero_shot")
    b = PipelineSpec(predictor="zero_shot", calibrators=[{"name": "temp", "kwargs": {"T": 1.5}}])
    assert a.config_hash() != b.config_hash()


def test_build_from_spec_uniform_needs_no_client():
    # The "uniform" predictor builds with no client/network (registry thunk).
    pipe = build_from_spec(PipelineSpec(predictor="uniform"))
    assert pipe.name.startswith("uniform")
