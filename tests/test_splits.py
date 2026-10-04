"""Tests for the leave-family-out split engine.

These are pure unit tests over synthetic records — no network, no dataset
download required. They lock in the three guarantees that matter:
  1. no question family ever straddles two buckets (leakage),
  2. the partition is deterministic and stable across runs,
  3. required questions land in the pinned bucket.
"""

from __future__ import annotations

import pytest

from simbench_exp.data import SimBenchRecord
from simbench_exp.splits import (
    DEFAULT_FRACTIONS,
    family_key,
    is_required_question,
    make_split,
)


def _rec(
    dataset: str,
    template: str,
    *,
    split: str = "grouped",
    segment: dict | None = None,
) -> SimBenchRecord:
    return SimBenchRecord(
        dataset_name=dataset,
        split=split,
        input_template=template,
        options=("A", "B"),
        human_answer={"A": 0.5, "B": 0.5},
        group_prompt="",
        segment=segment or {},
        num_grouping_vars=len(segment or {}),
    )


def _question_family(dataset: str, template: str, n_variants: int) -> list[SimBenchRecord]:
    """A population row plus n_variants grouped variants — one question family."""
    recs = [_rec(dataset, template, split="pop")]
    for i in range(n_variants):
        recs.append(_rec(dataset, template, segment={"cntry": f"C{i}"}))
    return recs


# -- family identity -------------------------------------------------------
def test_family_key_question_binds_variants():
    pop = _rec("ESS", "Q1?", split="pop")
    var = _rec("ESS", "Q1?", segment={"cntry": "Finland"})
    # Same question text => same family, regardless of split or segment.
    assert family_key(pop) == family_key(var) == ("ESS", "Q1?")


def test_family_key_dataset_unit():
    a = _rec("ESS", "Q1?")
    b = _rec("ESS", "Q2?")
    assert family_key(a, unit="dataset") == family_key(b, unit="dataset") == ("ESS",)


def test_family_key_unknown_unit_raises():
    with pytest.raises(ValueError):
        family_key(_rec("ESS", "Q?"), unit="bogus")


# -- core guarantees -------------------------------------------------------
def test_no_family_leakage_across_buckets():
    records = []
    for q in range(40):
        records += _question_family("DS", f"Q{q}?", n_variants=5)
    split = make_split(records, seed=0)
    split.check_disjoint()  # raises on leakage
    # Each family's records all share one bucket.
    seen: dict[tuple, str] = {}
    for bucket, recs in split.subsets.items():
        for r in recs:
            seen.setdefault(family_key(r), bucket)
            assert seen[family_key(r)] == bucket


def test_deterministic_across_runs():
    records = [r for q in range(50) for r in _question_family("DS", f"Q{q}?", 3)]
    a = make_split(records, seed=7)
    b = make_split(records, seed=7)
    assert a.family_bucket == b.family_bucket


def test_seed_changes_partition():
    records = [r for q in range(50) for r in _question_family("DS", f"Q{q}?", 3)]
    a = make_split(records, seed=1)
    b = make_split(records, seed=2)
    assert a.family_bucket != b.family_bucket


def test_stable_when_records_added():
    base = [r for q in range(40) for r in _question_family("DS", f"Q{q}?", 3)]
    extra = [r for q in range(40, 60) for r in _question_family("DS", f"Q{q}?", 3)]
    a = make_split(base, seed=0)
    b = make_split(base + extra, seed=0)
    # Families present in both must keep their assignment (hash-based stability).
    for key, bucket in a.family_bucket.items():
        assert b.family_bucket[key] == bucket


def test_fractions_respected_over_families():
    records = [r for q in range(400) for r in _question_family("DS", f"Q{q}?", 2)]
    split = make_split(records, fractions={"dev": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    fam_counts = {b: len({family_key(r) for r in recs}) for b, recs in split.subsets.items()}
    total = sum(fam_counts.values())
    assert abs(fam_counts["dev"] / total - 0.5) < 0.03
    assert abs(fam_counts["val"] / total - 0.25) < 0.03
    assert abs(fam_counts["test"] / total - 0.25) < 0.03


def test_two_way_split():
    records = [r for q in range(100) for r in _question_family("DS", f"Q{q}?", 2)]
    split = make_split(records, fractions={"dev": 0.7, "test": 0.3}, seed=0)
    assert set(split.subsets) == {"dev", "test"}
    assert len(split.val) == 0


# -- required-question pinning --------------------------------------------
def test_required_questions_pinned_to_test():
    records = [r for q in range(60) for r in _question_family("DS", f"Q{q}?", 3)]
    # Inject a required question (matches REQUIRED_QUESTIONS substring).
    required = _question_family("LatinoBarometro", "How much trust in the president?", 8)
    records += required
    split = make_split(records, seed=0)
    # Every required-question record is in test.
    for r in required:
        assert split.bucket_of(r) == "test"
    assert split.summary().set_index("bucket").loc["test", "required_q_records"] == len(required)


def test_pinning_can_be_disabled():
    required = _question_family("LatinoBarometro", "trust in the president now?", 5)
    records = [r for q in range(50) for r in _question_family("DS", f"Q{q}?", 2)] + required
    split = make_split(records, seed=0, pin_required_to=None)
    buckets = {split.bucket_of(r) for r in required}
    assert len(buckets) == 1  # still one family, but not forced to test
    # It is *possible* (not guaranteed) to be outside test; just assert it's valid.
    assert buckets.issubset(set(split.subsets))


def test_pin_target_must_be_a_bucket():
    records = _question_family("DS", "Q?", 2)
    with pytest.raises(ValueError):
        make_split(records, fractions={"dev": 0.8, "test": 0.2}, pin_required_to="val")


def test_is_required_question():
    assert is_required_question(_rec("X", "Do you have trust in the president?"))
    assert not is_required_question(_rec("X", "What is your favorite color?"))


# -- summary ---------------------------------------------------------------
def test_summary_columns_and_totals():
    records = [r for q in range(30) for r in _question_family("DS", f"Q{q}?", 4)]
    split = make_split(records, seed=0)
    summ = split.summary()
    assert list(summ.columns) == ["bucket", "families", "records", "required_q_records"]
    assert summ["records"].sum() == len(records)
    assert summ["families"].sum() == 30


def test_default_fractions_unchanged():
    # Guard against accidental mutation of the module-level default.
    assert DEFAULT_FRACTIONS == {"dev": 0.5, "val": 0.25, "test": 0.25}
