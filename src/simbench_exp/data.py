"""SimBench dataset download and loading.

Downloads the two SimBench evaluation CSVs from the HuggingFace hub and parses
each row into a typed :class:`SimBenchRecord`. The benchmark ships only
evaluation splits (no train/dev), so any dev/holdout split is constructed
downstream by us (see notes in the design doc).

Record fields normalize the messy CSV representation:
  * `human_answer` percentages -> probabilities summing to 1
  * the option labels are recovered from the answer distribution keys
  * dict-like string columns are parsed safely

The CSV columns are documented in the SimBench paper (Hu et al., arXiv
2510.17516). The two files have slightly different schemas (Grouped adds
`grouping_keys` etc.), so loading is defensive about column presence.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from .config import RAW_DIR, SIMBENCH_FILES, SIMBENCH_REPO_ID


def _parse_mapping(value) -> dict:
    """Parse a CSV cell that should hold a dict.

    Handles native dicts, JSON, and Python-literal strings; returns {} for
    empty/NaN cells.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and pd.isna(value):
        return {}
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            out = parser(text)
            if isinstance(out, dict):
                return out
        except (ValueError, SyntaxError):
            continue
    return {}


def _parse_keyset(value) -> tuple[str, ...]:
    """Parse the `grouping_keys` cell, stored as a Python set literal.

    Examples: "set()" -> (), "{'age_group', 'cntry'}" -> ('age_group','cntry').
    Returns a sorted tuple for a stable, hashable order.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    if isinstance(value, (set, frozenset, list, tuple)):
        return tuple(sorted(str(v) for v in value))
    text = str(value).strip()
    if not text or text in {"set()", "nan", "None"}:
        return ()
    try:
        out = ast.literal_eval(text)
        if isinstance(out, (set, frozenset, list, tuple)):
            return tuple(sorted(str(v) for v in out))
    except (ValueError, SyntaxError):
        pass
    return ()


def _normalize_distribution(raw_answer: dict) -> dict[str, float]:
    """Convert a {label: percentage} map to {label: probability} summing to 1.

    SimBench stores `human_answer` as percentages summing to ~100. We coerce
    values to float and renormalize defensively (some rows may not sum to
    exactly 100 due to rounding).
    """
    numeric: dict[str, float] = {}
    for label, pct in raw_answer.items():
        try:
            numeric[str(label)] = float(pct)
        except (TypeError, ValueError):
            continue
    total = sum(numeric.values())
    if total <= 0:
        # Degenerate row; fall back to uniform over whatever labels exist.
        n = len(numeric) or 1
        return {k: 1.0 / n for k in numeric} if numeric else {}
    return {k: v / total for k, v in numeric.items()}


@dataclass(frozen=True)
class SimBenchRecord:
    """One SimBench test case.

    Attributes:
        dataset_name: source survey (e.g. "ESS", "OpinionQA", "LatinoBarometro").
        split: "pop" or "grouped".
        input_template: the full question prompt including lettered options.
        options: ordered option labels (e.g. ["A", "B", "C", "D"]).
        answer_options: {label: answer text} when available (grouped split).
        human_answer: empirical distribution as {label: probability}, sums to 1.
        group_prompt: the rendered population / segment conditioning prompt.
        segment: the segment's conditioning values, e.g.
            {"cntry": "Finland", "age_group": "30-49"}; empty for the
            unconditioned population. This is the field the counterfactual
            sensitivity metric conditions on.
        grouping_keys: the demographic variable *names* defining the segment.
        num_grouping_vars: number of grouping variables (0 = population).
        group_size: number of human respondents behind the distribution.
        raw: the original CSV row, for any field not surfaced above.
    """

    dataset_name: str
    split: str
    input_template: str
    options: tuple[str, ...]
    human_answer: dict[str, float]
    group_prompt: str
    answer_options: dict[str, str] = field(default_factory=dict)
    segment: dict[str, str] = field(default_factory=dict)
    grouping_keys: tuple[str, ...] = ()
    num_grouping_vars: int = 0
    group_size: int | None = None
    raw: dict = field(default_factory=dict)

    @property
    def num_options(self) -> int:
        return len(self.options)

    @property
    def is_population(self) -> bool:
        """True for the unconditioned full-population case (no segment)."""
        return self.num_grouping_vars == 0 and not self.segment


def download_simbench(force: bool = False) -> dict[str, Path]:
    """Download the SimBench CSVs from HuggingFace into data/raw.

    Returns a {split: local_path} mapping. Idempotent: hf_hub_download caches,
    and we copy into RAW_DIR for a stable local path.
    """
    paths: dict[str, Path] = {}
    for split, filename in SIMBENCH_FILES.items():
        local = hf_hub_download(
            repo_id=SIMBENCH_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
            force_download=force,
        )
        paths[split] = Path(local)
    return paths


def _row_to_record(row: dict, split: str) -> SimBenchRecord | None:
    raw_answer = _parse_mapping(row.get("human_answer"))
    human_answer = _normalize_distribution(raw_answer)
    if not human_answer:
        return None  # skip unusable rows
    options = tuple(human_answer.keys())

    group_size = row.get("group_size")
    try:
        group_size = int(group_size) if group_size is not None and not pd.isna(group_size) else None
    except (TypeError, ValueError):
        group_size = None

    segment = _parse_mapping(row.get("group_prompt_variable_map"))
    grouping_keys = _parse_keyset(row.get("grouping_keys"))
    num_grouping_vars = row.get("num_grouping_vars")
    try:
        num_grouping_vars = (
            int(num_grouping_vars)
            if num_grouping_vars is not None and not pd.isna(num_grouping_vars)
            else len(segment)
        )
    except (TypeError, ValueError):
        num_grouping_vars = len(segment)

    return SimBenchRecord(
        dataset_name=str(row.get("dataset_name", "")),
        split=split,
        input_template=str(row.get("input_template", "")),
        options=options,
        human_answer=human_answer,
        group_prompt=str(row.get("group_prompt_template", "") or ""),
        answer_options=_parse_mapping(row.get("answer_options")),
        segment=segment,
        grouping_keys=grouping_keys,
        num_grouping_vars=num_grouping_vars,
        group_size=group_size,
        raw=row,
    )


def load_split(split: str, download: bool = True) -> list[SimBenchRecord]:
    """Load one SimBench split ("pop" or "grouped") as a list of records."""
    if split not in SIMBENCH_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SIMBENCH_FILES)}")
    path = RAW_DIR / SIMBENCH_FILES[split]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; run download_simbench() first.")
        download_simbench()
    df = pd.read_csv(path)
    records: list[SimBenchRecord] = []
    for row in df.to_dict(orient="records"):
        rec = _row_to_record(row, split)
        if rec is not None:
            records.append(rec)
    return records


def load_all() -> dict[str, list[SimBenchRecord]]:
    """Load both splits."""
    return {split: load_split(split) for split in SIMBENCH_FILES}
