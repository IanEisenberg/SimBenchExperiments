"""Demographic distribution tables for post-stratification.

SimBench's grouped split conditions almost every question on ``country × one
attribute`` and records a ``group_size`` (the respondent count) per cell. Those
counts are the population weights we need to **recombine** subgroup predictions
into a coarser estimate (post-stratification): predict each cell, then average
weighted by its share of respondents.

Verified property (the correctness backbone): the group_size-weighted average of
the attribute-subgroup *truths* reproduces the country-marginal truth to mean
TVD ≈ 0.0003. :class:`SegmentWeights` exposes that decomposition; the
:class:`~simbench_exp.predict.PostStratificationPredictor` consumes it.

The :class:`WeightSource` protocol keeps the weight origin swappable —
:class:`SimBenchWeights` derives weights from the benchmark's own counts today;
an external census source can plug in behind the same interface later. The same
hook is where finer-than-SimBench subdivision (e.g. US → states) would live.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .data import SimBenchRecord


def _question_key(record: SimBenchRecord) -> tuple[str, str]:
    """Identity of the underlying question (shared across its segments)."""
    return (record.dataset_name, record.input_template)


def _group_size(record: SimBenchRecord) -> float:
    return float(record.group_size or 0)


@runtime_checkable
class WeightSource(Protocol):
    """Supplies the finer cells (and weights) a record decomposes into."""

    def children(
        self, record: SimBenchRecord, over: str | None = None
    ) -> list[tuple[float, SimBenchRecord]]:
        ...


class SegmentWeights:
    """Post-stratification weights derived from SimBench ``group_size`` counts.

    Built from the grouped records. For a coarser `record`, :meth:`children`
    finds the records that add exactly one demographic variable, matched on the
    parent's existing segment values, and weights them by respondent share.
    """

    def __init__(self, records: list[SimBenchRecord]) -> None:
        self._by_question: dict[tuple[str, str], list[SimBenchRecord]] = defaultdict(list)
        for r in records:
            self._by_question[_question_key(r)].append(r)

    # -- decomposition ----------------------------------------------------- #
    def _candidates(
        self, record: SimBenchRecord
    ) -> dict[str, list[SimBenchRecord]]:
        """Group same-question records that extend `record` by one variable."""
        parent = record.segment
        parent_keys = set(parent.keys())
        out: dict[str, list[SimBenchRecord]] = defaultdict(list)
        for cand in self._by_question.get(_question_key(record), ()):
            cand_keys = set(cand.segment.keys())
            if len(cand_keys) != len(parent_keys) + 1:
                continue
            if not parent_keys.issubset(cand_keys):
                continue
            if any(cand.segment.get(k) != v for k, v in parent.items()):
                continue
            new_var = next(iter(cand_keys - parent_keys))
            out[new_var].append(cand)
        return out

    def decompose_variables(self, record: SimBenchRecord) -> list[str]:
        """Variables that admit a 2+-cell decomposition of `record`, sorted."""
        cands = self._candidates(record)
        return sorted(v for v, recs in cands.items() if len(recs) >= 2)

    def children(
        self, record: SimBenchRecord, over: str | None = None
    ) -> list[tuple[float, SimBenchRecord]]:
        """Return ``[(weight, child_record), ...]`` decomposing `record`.

        If `over` is None, auto-pick the variable whose cells cover the most
        respondents. Weights are ``group_size / Σ siblings`` and sum to 1; if
        counts are missing, falls back to equal weights. Returns ``[]`` when no
        decomposition exists (caller should fall back to direct prediction).
        """
        cands = self._candidates(record)
        if not cands:
            return []
        if over is None:
            over = max(cands, key=lambda v: sum(_group_size(r) for r in cands[v]))
        recs = cands.get(over, [])
        if not recs:
            return []
        total = sum(_group_size(r) for r in recs)
        if total <= 0:
            w = 1.0 / len(recs)
            return [(w, r) for r in recs]
        return [(_group_size(r) / total, r) for r in recs]


def build_segment_weights(records: list[SimBenchRecord]) -> SegmentWeights:
    """Convenience constructor (mirrors the loader-style helpers)."""
    return SegmentWeights(records)


# --------------------------------------------------------------------------- #
# Distribution table (the "demographic distribution per subset task" artifact)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DistributionRow:
    """One cell's share within a (dataset, question, grouping) subset."""

    dataset: str
    grouping_keys: tuple[str, ...]
    segment: dict[str, str]
    group_size: int | None
    weight: float
    question: str = field(default="", compare=False)


def distribution_table(records: list[SimBenchRecord]) -> list[DistributionRow]:
    """Population shares for every demographic subset present in the data.

    Groups records by (dataset, question, grouping-keys) and normalizes
    ``group_size`` within each group, yielding the share of each segment value.
    This is the per-subset-task demographic distribution used to weight and
    combine inferences across groups.
    """
    groups: dict[tuple, list[SimBenchRecord]] = defaultdict(list)
    for r in records:
        groups[(r.dataset_name, r.input_template, r.grouping_keys)].append(r)

    rows: list[DistributionRow] = []
    for (dataset, question, grouping_keys), recs in groups.items():
        total = sum(_group_size(r) for r in recs)
        for r in recs:
            weight = _group_size(r) / total if total > 0 else 1.0 / len(recs)
            rows.append(
                DistributionRow(
                    dataset=dataset,
                    grouping_keys=grouping_keys,
                    segment=dict(r.segment),
                    group_size=r.group_size,
                    weight=weight,
                    question=question,
                )
            )
    return rows
