# Task 9 Report: Tree Renderer + Offline Integration Test

## What Was Implemented

### `src/scrye/tree_view.py`
A 37-line ASCII experiment-tree renderer. Public API: `tree_lines(ledger, depth=0, parent_id=None) -> list[str]`. Renders the DAG depth-first from roots with:
- `✓` for accepted nodes (val_score not None, accepted=True)
- `✗` for rejected nodes (val_score not None, accepted=False)
- `·` for dev-only nodes (val_score is None)
- Two-space indent per depth level
- Format: `{mark} {lever_id} (dev={dev_score:.2f}[ val={val_score:.2f}])[ ${cost_usd:.2f}]`

### `tests/test_tree_view.py`
Single test `test_tree_lines_marks_status_and_nests` building a 3-node tree (root + 2 children at depth 1): verifies correct markers for all three status types and that children are indented deeper than their parent.

### `tests/test_search_integration.py`
End-to-end offline integration test `test_full_loop_offline` that:
1. Builds a `Ledger` and `SearchContract` allowing two levers
2. Injects an offline `score_fn` returning `(20 + 5*depth, {"by_entropy": {}, "cost_usd": 1.0})` — no network
3. Runs `run_search` to completion via `propose_fn` that exhausts a 2-item iterator
4. Asserts `stop_reason == "proposer_done"`, `global_k() == 2`, `lever_path == ["recalib.global_temp", "recalib.dirichlet"]`, `total_cost() == 4.0`
5. **Cost-consistency invariant** (additional required assertion): `abs(led.total_cost() - led.nodes[-1].cum_cost_usd) < 1e-9` — confirms the per-node running total agrees with the independent sum
6. Calls `final_report` on a uniform predictor (no network) and asserts `global_k == 2` and `total_cost_usd == 4.0`

## TDD Evidence

**RED:** `pytest tests/test_tree_view.py tests/test_search_integration.py -v` failed with `ModuleNotFoundError: No module named 'scrye.tree_view'` (collection error before any test ran).

**GREEN:** After creating `src/scrye/tree_view.py`, both tests passed: `2 passed in 0.53s`.

## Files Changed
- Created: `/Users/ian/Projects/Scrye_Project/.claude/worktrees/lever-search-spec/src/scrye/tree_view.py`
- Created: `/Users/ian/Projects/Scrye_Project/.claude/worktrees/lever-search-spec/tests/test_tree_view.py`
- Created: `/Users/ian/Projects/Scrye_Project/.claude/worktrees/lever-search-spec/tests/test_search_integration.py`

## Full-Suite Summary Line (exact)
`144 passed, 2 skipped in 0.53s`

## Self-Review Findings

**Tree indentation/markers correct:** Yes. `"  " * depth` gives 2 spaces per level; the test verifies parent (depth 0, no leading spaces) vs children (depth 1, 2 leading spaces). Markers match spec exactly.

**Integration test genuinely end-to-end:** Yes. Exercises `run_search` → `Ledger` → `final_report` with no mocks — only `score_fn`, `propose_fn`, and `now_fn` are injected as offline substitutes. `final_report` calls `evaluate` on a real `uniform` predictor against a live `SimBenchRecord`.

**Cost-consistency assertion meaningful:** Yes. The `score_fn` returns `cost_usd=1.0` per call; each accepted step calls `score_fn` twice (dev + val), so `cum_cost_usd` on the last node (`4.0`) should equal `total_cost()` (`sum(n.cost_usd for n in nodes)`). The assertion `abs(led.total_cost() - led.nodes[-1].cum_cost_usd) < 1e-9` passes, confirming no drift between the two accounting paths.

**YAGNI:** `tree_view.py` is 37 lines — no extra utilities, no public exports beyond `tree_lines`. Tests import only what they use.

**Test hygiene:** Each test uses `tmp_path` (pytest fixture) for ledger isolation. No shared state. No prints. No network calls.

**No concerns.** All assertions pass and implementation is a faithful transcription of the brief.

---

## Hardening Fixes (post-review, applied after original task-9 work)

### Fix 1 — eta<=0 warning in `run_search` (search.py)
Added `import warnings` at top of file. At the start of `run_search`, before the loop, added a `warnings.warn(...)` that fires when `contract.eta <= 0.0`. All existing tests use `eta=0.5` so the warning does NOT fire in the suite.

### Fix 2 — Cache-hit val_best regression (search.py)
Changed `state.val_best = gate.best` (on the accepted cache-hit path) to `state.val_best = max(gate.best, val_score)`. On a cache hit `gate.consider` is never called, so `gate.best` can be stale/lower than the cached candidate's `val_score`. The ledger and `gate.best` are unchanged; this is an in-memory consistency fix only.

### Fix 3 — v1 no-dev-fit decision recorded (spec.py, levers.py)
- `build_from_spec` docstring in `spec.py`: noted that calibrators are constructed from fixed kwargs and are NOT `.fit()`-ed; auto dev-fit is a deferred v2 decision.
- `levers.py` module docstring: added one sentence — the three recalibration levers apply FIXED hyperparameters supplied per-proposal; they do not auto-fit on dev in v1.
No code behavior changed.

### Commands run and output
```
pytest tests/test_search.py -v   →  7 passed in 0.70s
pytest -q (full suite)           →  144 passed, 2 skipped in 0.52s
```
No warnings in either summary. Eta warning does not fire in the suite.

### Commit
SHA `5237707` — `harden: warn on eta<=0; fix cache-hit val_best; record v1 no-dev-fit decision`
