"""Leave-family-out dev/val/test split engine.

SimBench ships only evaluation splits — no train/dev. We *iterate on algorithms*
(prompt design, persona-pool construction, model choice, calibration
parameters), so even though nothing is gradient-trained, every one of those
decisions is fit to data. If we make those decisions while looking at the test
set, we leak. This module constructs the holdout discipline that keeps the
honest headline number honest.

The one non-obvious rule: **split by question family, not by row.** SimBench's
grouped split contains many near-duplicate variants of the same survey item
(one per demographic segment). A random row split scatters siblings across
dev and test, so the model would be "evaluated" on questions it was effectively
tuned on. The unit of assignment is therefore the *family*:

    family = (dataset_name, input_template)

which binds a question's population row and all its segment variants into a
single indivisible block. Whole families move together; no variant ever
straddles the dev/test boundary. (Set ``unit="dataset"`` for the stronger
generalize-to-an-unseen-survey test, at the cost of a coarser split.)

Assignment is deterministic and stable: each family is hashed (seeded SHA-256)
to a value in [0, 1) and placed by fixed cumulative-fraction thresholds. A
family's bucket is therefore a pure function of its own key and the seed — so
the same seed always yields the same partition, and adding or removing other
records never moves an existing family. Proportions are approximate (the law
of large numbers makes them tight: <1% off with thousands of families) rather
than exact, which is the right trade for leakage-safe stability. The three
required assignment questions are *pinned to test* — they are the numbers we
must report, so they must never inform a design decision.

Typical use::

    from scrye.data import load_all
    from scrye.splits import make_split

    recs = load_all()
    split = make_split(recs["pop"] + recs["grouped"])
    split.summary()          # families / records / required-Qs per bucket
    dev = split.dev          # iterate here
    test = split.test        # touch once, at the very end
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .config import REQUIRED_QUESTIONS
from .data import SimBenchRecord

# Default three-way partition. Roles, not just sizes:
#   dev  — free iteration, error analysis, comparing simulation systems
#          (the experiments notebook runs here)
#   val  — model / hyperparameter selection and any parameter fitting
#          (e.g. a calibration temperature), kept separate from `dev` so
#          selection isn't contaminated by the data you eyeballed
#   test — locked; scored once for the final report; required questions live here
DEFAULT_FRACTIONS: dict[str, float] = {"dev": 0.5, "val": 0.25, "test": 0.25}


def family_key(record: SimBenchRecord, unit: str = "question") -> tuple[str, ...]:
    """The indivisible grouping key for a record.

    ``unit="question"`` (default) → ``(dataset_name, input_template)``: every
    segment variant of one survey item stays together. ``unit="dataset"`` →
    ``(dataset_name,)``: whole surveys are held out (stronger generalization
    claim, coarser split). The key deliberately excludes ``split`` so a
    question's population row and its grouped variants share one family.
    """
    if unit == "dataset":
        return (record.dataset_name,)
    if unit == "question":
        return (record.dataset_name, record.input_template)
    raise ValueError(f"Unknown unit {unit!r}; expected 'question' or 'dataset'.")


def is_required_question(record: SimBenchRecord) -> bool:
    """True if the record is one of the three assignment-required questions.

    Matched by the same case-insensitive substring rule as
    :data:`~scrye.config.REQUIRED_QUESTIONS`.
    """
    text = record.input_template.lower()
    return any(needle in text for needle in REQUIRED_QUESTIONS.values())


_HASH_SCALE = float(1 << 64)


def _hash_unit(key: tuple[str, ...], seed: int) -> float:
    """Map a family key to a stable value in [0, 1).

    Uses SHA-256 (not Python's salted ``hash``) so the value is identical
    across runs and machines for a given seed. The bucket a family lands in
    depends only on this value, never on the rest of the dataset.
    """
    payload = f"{seed}::" + "::".join(key)
    h = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return h / _HASH_SCALE


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for :func:`make_split`."""

    fractions: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FRACTIONS))
    unit: str = "question"
    seed: int = 0
    pin_required_to: str | None = "test"


@dataclass(frozen=True)
class DataSplit:
    """An assignment of records to named, family-disjoint subsets.

    Attributes:
        subsets: bucket name -> list of records.
        family_bucket: family key -> bucket name (the authoritative assignment).
        config: the :class:`SplitConfig` used.
    """

    subsets: dict[str, list[SimBenchRecord]]
    family_bucket: dict[tuple[str, ...], str]
    config: SplitConfig

    # -- accessors ---------------------------------------------------------
    def subset(self, name: str) -> list[SimBenchRecord]:
        if name not in self.subsets:
            raise KeyError(f"No subset {name!r}; have {list(self.subsets)}.")
        return self.subsets[name]

    @property
    def dev(self) -> list[SimBenchRecord]:
        return self.subsets.get("dev", [])

    @property
    def val(self) -> list[SimBenchRecord]:
        return self.subsets.get("val", [])

    @property
    def test(self) -> list[SimBenchRecord]:
        return self.subsets.get("test", [])

    def bucket_of(self, record: SimBenchRecord) -> str:
        """Which bucket a record belongs to (by its family)."""
        return self.family_bucket[family_key(record, self.config.unit)]

    # -- diagnostics -------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """One row per bucket: family count, record count, required-Q records.

        Fractions are exact over *families*; record counts vary because grouped
        families carry many segment variants while population families carry
        one. Inspect this to confirm the test bucket is large enough to score.
        """
        fams_per_bucket: dict[str, set] = {b: set() for b in self.subsets}
        for key, bucket in self.family_bucket.items():
            fams_per_bucket[bucket].add(key)
        rows = []
        order = list(self.config.fractions) + [
            b for b in self.subsets if b not in self.config.fractions
        ]
        for bucket in order:
            recs = self.subsets.get(bucket, [])
            rows.append(
                {
                    "bucket": bucket,
                    "families": len(fams_per_bucket.get(bucket, set())),
                    "records": len(recs),
                    "required_q_records": sum(is_required_question(r) for r in recs),
                }
            )
        return pd.DataFrame(rows)

    def check_disjoint(self) -> None:
        """Assert no family appears in more than one bucket. Raises on leakage."""
        seen: dict[tuple[str, ...], str] = {}
        for bucket, recs in self.subsets.items():
            for r in recs:
                key = family_key(r, self.config.unit)
                prev = seen.setdefault(key, bucket)
                if prev != bucket:
                    raise AssertionError(
                        f"Family {key!r} leaks across {prev!r} and {bucket!r}."
                    )


def make_split(
    records: Sequence[SimBenchRecord],
    *,
    fractions: Mapping[str, float] | None = None,
    unit: str = "question",
    seed: int = 0,
    pin_required_to: str | None = "test",
) -> DataSplit:
    """Partition records into leave-family-out buckets.

    Args:
        records: any flat list of records (typically pop + grouped combined).
        fractions: bucket -> share, summing to ~1. Defaults to
            ``{"dev": 0.5, "val": 0.25, "test": 0.25}``. Keys define the bucket
            names; pass ``{"dev": 0.7, "test": 0.3}`` for a two-way split.
        unit: family granularity, ``"question"`` (default) or ``"dataset"``.
        seed: controls the deterministic family ordering.
        pin_required_to: bucket that the three required questions are forced
            into (default ``"test"``); ``None`` disables pinning. Pinned
            families are removed from the pool first, so realized fractions
            apply to the remaining (non-pinned) families.

    Returns:
        A :class:`DataSplit`. Guaranteed family-disjoint across buckets.
    """
    fracs = dict(fractions) if fractions is not None else dict(DEFAULT_FRACTIONS)
    if not fracs:
        raise ValueError("fractions must be non-empty.")
    total = sum(fracs.values())
    if total <= 0:
        raise ValueError("fractions must sum to a positive value.")
    if pin_required_to is not None and pin_required_to not in fracs:
        raise ValueError(
            f"pin_required_to={pin_required_to!r} is not one of the bucket "
            f"names {list(fracs)}."
        )

    cfg = SplitConfig(
        fractions=fracs, unit=unit, seed=seed, pin_required_to=pin_required_to
    )

    # Group records by family, and learn which families are required (pinned).
    fam_records: dict[tuple[str, ...], list[SimBenchRecord]] = {}
    pinned_families: set[tuple[str, ...]] = set()
    for r in records:
        key = family_key(r, unit)
        fam_records.setdefault(key, []).append(r)
        if pin_required_to is not None and is_required_question(r):
            pinned_families.add(key)

    family_bucket: dict[tuple[str, ...], str] = {}
    for key in pinned_families:
        family_bucket[key] = pin_required_to  # type: ignore[assignment]

    # Place each remaining family by its own hash against cumulative-fraction
    # thresholds. Assignment depends only on the family key + seed, so it is
    # stable under additions/removals elsewhere in the dataset.
    bucket_names = list(fracs)
    edges: list[tuple[float, str]] = []
    cum = 0.0
    for name in bucket_names:
        cum += fracs[name] / total
        edges.append((cum, name))
    edges[-1] = (1.0, bucket_names[-1])  # guard against float drift at the top

    for key in fam_records:
        if key in pinned_families:
            continue
        u = _hash_unit(key, seed)
        for threshold, name in edges:
            if u < threshold:
                family_bucket[key] = name
                break
        else:  # pragma: no cover - u in [0,1) and last edge is 1.0
            family_bucket[key] = bucket_names[-1]

    subsets: dict[str, list[SimBenchRecord]] = {b: [] for b in bucket_names}
    for key, recs in fam_records.items():
        subsets[family_bucket[key]].extend(recs)

    split = DataSplit(subsets=subsets, family_bucket=family_bucket, config=cfg)
    split.check_disjoint()
    return split
