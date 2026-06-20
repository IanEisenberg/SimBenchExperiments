"""RunManifest — stores the knobs needed to re-select the exact same data + params.

A RunManifest is a frozen record of everything that was deterministic at search
time: the registry version, the split configuration (seed, fractions, unit,
pin_required_to), the search contract parameters, model settings, a dataset
fingerprint, and the timestamp. Together with a cache of LLM responses (temp=0),
these knobs make a re-run byte-for-byte identical.

Usage::

    from scrye.manifest import RunManifest, dataset_fingerprint
    manifest = RunManifest(
        registry_version=REGISTRY_VERSION,
        split_seed=0, split_fractions={"dev": 0.5, "val": 0.25, "test": 0.25},
        split_unit="question", pin_required_to="test",
        goal="improve S", allowed_levers=["recalib.global_temp"],
        autonomy_budget=10, val_query_budget=5, eta=0.5, cost_cap_usd=100.0,
        model="gemini-flash-lite", temperature=0.0, seed=None,
        dataset_fingerprint=dataset_fingerprint(records),
        created_at="2026-06-20T00:00:00",
    )
    manifest.save("outputs/ledger/run.manifest.json")
    loaded = RunManifest.load("outputs/ledger/run.manifest.json")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def dataset_fingerprint(records) -> str:
    """SHA-256 (16 hex chars) over sorted unique (dataset_name, input_template) pairs.

    The same set of (dataset_name, input_template) pairs yields the same fingerprint
    regardless of record order; a different set yields a different one.
    """
    pairs = sorted({(r.dataset_name, r.input_template) for r in records})
    payload = json.dumps(pairs, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RunManifest:
    """Frozen record of the knobs that make a search run reproducible."""

    registry_version: str
    split_seed: int
    split_fractions: dict
    split_unit: str
    pin_required_to: Optional[str]
    goal: str
    allowed_levers: list
    autonomy_budget: int
    val_query_budget: int
    eta: float
    cost_cap_usd: float
    model: str
    temperature: float
    seed: Optional[int]
    dataset_fingerprint: str
    created_at: str

    def save(self, path: str | Path) -> None:
        """Write the manifest to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """Load a manifest from a JSON file."""
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)
