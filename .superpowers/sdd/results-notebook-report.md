# Results Notebook Implementation Report

## Status: DONE

## Summary

Implemented the full reproducibility layer + experiment-results query/visualization
notebook on top of the existing SimBench lever search system.

## Parts Implemented

### Part A — Reproducibility Plumbing

**A1: `ExperimentNode` schema additions** (`src/simbench_exp/ledger.py`)
- Added `params: dict` and `spec_config: dict` trailing fields with defaults
- Added `field` to dataclasses import
- New test `test_params_spec_config_roundtrip` in `tests/test_ledger.py` verifies JSONL round-trip

**A2: `run_search` now logs params and spec_config** (`src/simbench_exp/search.py`)
- `params=dict(proposal.params)` and `spec_config=new_spec.resolved()` passed to ExperimentNode
- New test `test_node_carries_params_and_spec_config` in `tests/test_search.py` verifies fields

**A3: New `src/simbench_exp/manifest.py`**
- `RunManifest` frozen dataclass with all 16 required fields
- `save(path)` writes JSON, `load(path)` classmethod reads it
- `dataset_fingerprint(records)` — sha256 (16 hex chars) over sorted unique (dataset_name, input_template) pairs
- `tests/test_manifest.py`: round-trip equality, None field handling, fingerprint determinism

### Part B — Query + Reproduction Helpers (`src/simbench_exp/results.py`)

- `nodes_frame(ledger)` — DataFrame with 11 columns per node
- `node_report(ledger, node_id)` — full dict; raises KeyError if missing
- `format_node_report(ledger, node_id)` — human-readable multi-line string
- `best_progression(ledger)` — Ladder climb DataFrame with is_new_best logic
- `spec_from_node(ledger, node_id)` — PipelineSpec(**node.spec_config); raises on empty spec_config
- `replay_proposals(ledger)` — list of Proposal objects in append order

`tests/test_results.py`: 13 tests covering shape, depth, score logic, new-best logic (explicitly tests lower score -> is_new_best=False), missing-id raises, empty spec_config raises.

### Part C — Notebook (`notebooks/03_experiment_results.ipynb`)

8 cells (2 markdown, 6 code):
1. Title + where-results-live markdown
2. Setup: imports, RUN_PATH, auto-generates demo ledger if file absent (offline stub with two real registry levers)
3. Load ledger + print tree
4. nodes_frame display + dev vs val scatter (colored by accepted)
5. format_node_report for NODE_ID (parameterized)
6. best_progression display + Ladder climb line plot
7. Reproducibility markdown explanation
8. Reproducibility code: load manifest if present, offline check: replay_proposals -> fresh ledger -> assert lever sequence + global_k match -> print "REPRODUCED ✓"

## Test Results

- Before: 144 passed, 2 skipped
- After: 163 passed, 2 skipped (+19 new tests, 0 failures)
- Notebook JSON validates; all 6 code cells run without error; "REPRODUCED ✓" prints

## Files Changed

- `src/simbench_exp/ledger.py` — added params/spec_config fields + `field` import
- `src/simbench_exp/search.py` — pass params/spec_config to ExperimentNode
- `src/simbench_exp/manifest.py` — NEW: RunManifest + dataset_fingerprint
- `src/simbench_exp/results.py` — NEW: query/reproduction helpers
- `tests/test_ledger.py` — +1 test for roundtrip
- `tests/test_search.py` — +1 test for params/spec_config in logged nodes
- `tests/test_manifest.py` — NEW: 4 tests
- `tests/test_results.py` — NEW: 13 tests
- `notebooks/03_experiment_results.ipynb` — NEW

## Concerns

None. All backward-compatible (new fields have defaults).
