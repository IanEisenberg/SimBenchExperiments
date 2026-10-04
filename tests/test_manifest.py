"""Tests for RunManifest — round-trip and fingerprint determinism."""

from simbench_exp.manifest import RunManifest, dataset_fingerprint


def _make_manifest(**kw):
    base = dict(
        registry_version="v1",
        split_seed=0,
        split_fractions={"dev": 0.5, "val": 0.25, "test": 0.25},
        split_unit="question",
        pin_required_to="test",
        goal="improve S",
        allowed_levers=["recalib.global_temp", "recalib.dirichlet"],
        autonomy_budget=10,
        val_query_budget=5,
        eta=0.5,
        cost_cap_usd=100.0,
        model="gemini-flash-lite",
        temperature=0.0,
        seed=None,
        dataset_fingerprint="abc123456789abcd",
        created_at="2026-06-20T00:00:00",
    )
    base.update(kw)
    return RunManifest(**base)


def test_manifest_roundtrip(tmp_path):
    """Save and load equality."""
    m = _make_manifest()
    p = tmp_path / "run.manifest.json"
    m.save(p)
    loaded = RunManifest.load(p)
    assert loaded == m
    assert loaded.registry_version == "v1"
    assert loaded.split_fractions == {"dev": 0.5, "val": 0.25, "test": 0.25}
    assert loaded.pin_required_to == "test"
    assert loaded.seed is None


def test_manifest_roundtrip_none_fields(tmp_path):
    """Manifest with None pin_required_to and seed survives roundtrip."""
    m = _make_manifest(pin_required_to=None, seed=42)
    p = tmp_path / "run.manifest.json"
    m.save(p)
    loaded = RunManifest.load(p)
    assert loaded == m
    assert loaded.pin_required_to is None
    assert loaded.seed == 42


class _FakeRecord:
    def __init__(self, dataset_name, input_template):
        self.dataset_name = dataset_name
        self.input_template = input_template


def test_fingerprint_determinism():
    """Same records (regardless of order) -> same fingerprint."""
    recs1 = [
        _FakeRecord("ds_a", "tmpl_1"),
        _FakeRecord("ds_a", "tmpl_2"),
        _FakeRecord("ds_b", "tmpl_1"),
    ]
    # Duplicate + shuffled order
    recs2 = [
        _FakeRecord("ds_b", "tmpl_1"),
        _FakeRecord("ds_a", "tmpl_2"),
        _FakeRecord("ds_a", "tmpl_1"),
        _FakeRecord("ds_a", "tmpl_1"),  # duplicate
    ]
    fp1 = dataset_fingerprint(recs1)
    fp2 = dataset_fingerprint(recs2)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_differs_for_different_records():
    """Different (dataset_name, input_template) pairs -> different fingerprint."""
    recs_a = [_FakeRecord("ds_a", "tmpl_1")]
    recs_b = [_FakeRecord("ds_a", "tmpl_2")]
    assert dataset_fingerprint(recs_a) != dataset_fingerprint(recs_b)
