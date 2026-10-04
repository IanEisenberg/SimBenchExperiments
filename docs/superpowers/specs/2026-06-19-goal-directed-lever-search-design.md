# Goal-Directed Lever Search with Ladder-Guarded Holdout — Design Spec

**Date:** 2026-06-19 · **Status:** Approved design, ready for implementation planning
**Spine:** Sits on the existing harness (`splits.py`, `scoring.py`, `calibrate.py`, `pipeline.py`, `experiment.py`, `evaluate.py`).
**Research grounding:** the anti-overfitting design is drawn from the deep-research report on adaptive data analysis, prompt-optimizer overfitting, and SimBench levers. Verified claims cited inline as **[D]**.

---

## 1. Purpose & thesis

Build a disciplined, **semi-autonomous** loop that discovers *which theory-motivated interventions actually generalize* on SimBench — not a system that maximizes the SimBench score `S` directly.

The distinction is the whole point. An automated S-maximizer is mechanically an overfitting machine: every time a loop selects its next move from a held-out score, it spends some of that set's statistical independence. With enough automated probes, reported gains become selection noise rather than real signal.

- **[D]** Reusing a holdout adaptively overfits it and invalidates the estimates (Dwork et al., *Science* 2015, `science.aaa9375`).
- **[D]** The "Ladder" attack shows an adaptive loop can climb to a public MSE of 0.4 while the true MSE stays 1.0 — large apparent gains **from pure noise, even under full disclosure of every query** (Blum & Hardt 2015, `arXiv:1607.00091`). This is precisely how an automated SimBench maximizer would lie to us.

So the system's objective is reframed from *"maximize S on val"* to *"discover generalizing levers, measured under a provable anti-overfitting guard."* That reframe is on-thesis with the project posture — **rigor over headline numbers** — and the guard itself becomes a signature artifact of the work.

### Non-goals (explicitly out of scope; roadmap)

- **DSPy / MIPROv2 / GEPA / OPRO / TextGrad automated optimizers.** They target accuracy-style scalar rewards and are documented to overfit distributionally — **[D]** TextGrad showed 96% train vs 69% test (`arXiv:2309.03409`). They need many iterations (each a holdout query), maximizing both cost and overfitting exactly where a small-budget study can least afford it. They belong on the "heavy-machinery end of the spectrum" as future work, not in this spine.
- **Retrieval-anchored few-shot.** Yield-vs-leakage is unquantified in the literature; optional ablation only, with strict leave-family-out, never the spine.
- **Persona Monte Carlo, fine-tuning.** Roadmap.

---

## 2. The three-way split discipline (foundation)

`make_split` already produces family-disjoint buckets by fractions. This system **mandates a three-way split** and assigns each bucket a fixed role that never changes:

| Bucket | Role | Access policy |
|---|---|---|
| **dev** | Fit calibrators, explore levers, iterate freely | **Unlimited** probing. Outputs cached by config hash → re-runs free. |
| **val** | Adjudicate whether a lever generalizes | **Metered.** Touched only through the Ladder gate (§5). Every *new* config scored here spends one unit of the global query budget K. |
| **test** | The single reported number | **Touched exactly once**, at the very end, on the frozen final pipeline. No lever ever sees test. |

Required questions stay pinned to `test` (`pin_required_to="test"`), so the headline required-question results are reported from never-searched data. `DataSplit.check_disjoint()` is asserted before any run.

**Split sizing:** val must be large enough that the Ladder threshold η (§5) is smaller than the lever effects we care about. The implementation plan must include a pre-run check: bootstrap the val-S noise floor and confirm it is below the smallest lever ΔS worth claiming; widen val (shrink dev) if not. This is an open parameter — see §10.

---

## 3. Component A — Lever registry (`levers.py`, new)

A **pre-registered, version-controlled** list of candidate interventions. Pre-registration before any val contact is what converts the "garden of forking paths" into a fixed, countable comparison set — the multiple-comparisons surface is frozen and auditable.

### 3.1 What a lever is

A lever is a pure, declarative transform over the existing `Pipeline = Pipeline(Predictor, Calibrator)` composition. It does **not** mutate global state; it returns a new pipeline spec.

```python
@dataclass(frozen=True)
class Lever:
    id: str                      # stable, e.g. "recalib.entropy_temp"
    hypothesis: str              # what we expect to improve and why
    mechanism: str               # the literature-grounded reason (mode-seeking, etc.)
    expected_direction: str      # "↑ high-entropy S", "↓ consensus regression risk"
    params: dict                 # the lever's own knobs (frozen per instantiation)
    apply: Callable[[PipelineSpec], PipelineSpec]   # spec -> spec, pure
```

A `PipelineSpec` is a serializable description (predictor name + kwargs, ordered calibrator stages + kwargs, model id) from which a concrete `Pipeline` is built. Levers compose by chaining `apply`; the **ordered list of applied levers + their params is the experiment state** (see §6).

### 3.2 Seeded levers (the frozen v1 registry)

Each is one-variable and maps to a named mechanism from the research:

| Lever id | Hypothesis | Mechanism / **[D]** evidence |
|---|---|---|
| `elicit.verbalized` | Verbalized distribution ≫ logprob readout for instruct models | **[D]** SimBench: verbalized beats logits for instruct models (`arXiv:2510.17516v4`). Default-on; logprob variant is the ablation. |
| `ensemble.paraphrase` | 3–5 prompt paraphrases reduce prompt-sensitivity variance | Cheap variance reduction; prompt sensitivity is a known noise source. |
| `recalib.global_temp` | One global temperature corrects mode-seeking sharpness | Mode-seeking RLHF → over-sharp distributions. |
| `recalib.entropy_temp` | Entropy-conditioned temperature — correct *more* where humans disagree | **[D]** Instruct helps consensus, hurts high-entropy, r=−0.942 (`arXiv:2510.17516v4`). The headline treatment. |
| `recalib.dirichlet` | Dirichlet-smoothing map fit on dev | Principled dispersion fix; degenerate-to-uniform shrinkage falls out free. |
| `condition.delta` | Predict the segment's *shift* from population, not "be this group" | **[D]** Demographic conditioning degrades all models (religion −9.91, political −4.97; `arXiv:2510.17516v4`). Delta-modeling dodges the trap. |
| `model.base_vs_instruct` | A base/less-aligned model is a better high-entropy simulator | Alignment–simulation tradeoff; mass-covering > mode-seeking on diverse items. |

New levers may be *proposed* during a run but adding one is a gated, human-approved act (§4) precisely because it enlarges K's comparison surface.

---

## 4. Component B — The goal + scope contract (`search.py`, new)

The operator hands the loop a **contract** that defines a bounded autonomous exploration. This is what lets "human judgment come in flexibly": the operator sets *where the gates sit* per run.

```python
@dataclass(frozen=True)
class SearchContract:
    goal: str                        # e.g. "improve high-entropy S without regressing consensus"
    allowed_levers: list[str]        # subset of the frozen registry the loop may explore
    autonomy_budget: int             # max autonomous DEV steps before mandatory check-in (e.g. 8)
    val_query_budget: int            # the Ladder K cap — max NEW val queries before check-in (e.g. 3)
    cost_cap_usd: float = 1000.0     # hard ceiling on cumulative NEW OpenRouter spend (§9); loop halts before exceeding it
    gate_policy: GatePolicy          # which events pause for a human (below)
    eta_rule: str = "bootstrap"      # how the Ladder threshold is set (§5)
```

### 4.1 Gate policy — the flexible human-in-the-loop

```python
@dataclass(frozen=True)
class GatePolicy:
    gate_off_registry_proposal: bool = True   # ALWAYS pause before a non-registered lever
    gate_on_val_query: bool = False           # pause before each val query (strictest)
    gate_at_val_budget_frac: float = 1.0      # pause when this fraction of K cap is spent
    gate_on_surprise: bool = True             # pause on an unexpected accept/reject
    gate_after_dev_steps: int | None = None   # pause every N autonomous dev steps
    gate_at_cost_frac: float = 0.8            # pause when cumulative spend reaches this fraction of cost_cap_usd (§9)
```

**Invariant that cannot be disabled:** proposing a lever **not** on the pre-registered list always halts for human approval (`gate_off_registry_proposal` is forced `True`). Adding to the comparison set is the one action that changes the overfitting accounting, so it is never autonomous.

**Shipped default ("bounded autonomy"):** autonomous on dev up to `autonomy_budget`; auto-spend val queries up to `val_query_budget`; hard-stop at the cap, on any off-registry proposal, or on a surprising result. Tighten by setting `gate_on_val_query=True`; loosen by raising the budgets.

### 4.2 The loop

Each autonomous step:

1. **Propose** — Claude selects the next lever (or a param refinement) from `allowed_levers`, writes a rationale referencing the lever's hypothesis/mechanism.
2. **Fit & score on dev** — build the pipeline via `Pipeline`, run `predict_batch` on `dev` (cached by config hash → free if seen), compute dev S via `simbench_score` / `evaluate`. The scorer also returns the **new OpenRouter spend** for this step (cache hits = $0), which is logged on the node and added to the running total (§9).
3. **Decide promotion** — if the dev result clears the lever's expected direction, request a val query; else log the dead end on the tree and continue (no K spent).
4. **Gate check** — consult `GatePolicy`; pause and surface the trail if any trigger fires.
5. **Val query (metered)** — run the Ladder gate (§5): accept into the running pipeline iff ΔS > η. Spend one K. Log node with cumulative global K.
6. **Repeat** until `autonomy_budget` or `val_query_budget` is reached, then check in.

When the loop checks in it presents the **trail** (every node, rationale, dev/val outcome), not a single opaque jump. The operator redirects, approves, rewinds (§6), or extends the budget.

---

## 5. Component C — The Ladder gate (extends `evaluate.py`)

The val set is queried **only** through a Ladder rule (Blum & Hardt 2015), chosen over Thresholdout because **[D]** Thresholdout's differential-privacy constants are too weak to be practical at our n (`arXiv:1506.02629`); the Ladder is simpler and fits a maximizer loop natively.

**Rule.** Maintain the running-best val S, `S*`. A candidate config with val score `S_c` is **accepted only if `S_c > S* + η`**; then `S*` ← `S_c`. Otherwise rejected and `S*` unchanged.

- **[D]** Releasing a number only on a significant improvement bounds the generalization error to ~`(log(kn)/n)^{1/3}` regardless of how many candidates k are tried (Blum & Hardt 2015, `arXiv:1607.00091`; `blum15.pdf`).
- **η (threshold):** derived from the data, not guessed. Default `eta_rule="bootstrap"`: bootstrap the val ground-truth sampling noise (group sizes are published) to get the per-comparison noise floor; η = a small multiple of that floor. This ties the gate to SimBench's own finite-sample reliability — the same reliability-ceiling logic the project already adopts.

**Global K accounting.** K is the total count of *distinct* configs ever scored on val across the **entire experiment tree** — not per branch. Branching does not reset it. Revisiting a config hash already queried returns its cached val score and costs **no new K**. The final reported confidence band is widened for the realized global K (§7).

---

## 6. Component D — The experiment-tree ledger (`ledger.py`, new)

An **append-only DAG** that records the full search path and makes any node a resumption point.

### 6.1 Node schema

```python
@dataclass(frozen=True)
class ExperimentNode:
    node_id: str                 # short stable id
    config_hash: str             # content-address of the PipelineSpec (ordered levers+params+model)
    parent_id: str | None        # the state this was derived from
    lever_id: str                # lever applied to reach this node
    rationale: str               # Claude's written reason for trying it
    dev_score: float             # S on dev (always present)
    dev_breakdown: dict          # entropy-binned S, counterfactual sensitivity, etc.
    val_score: float | None      # S on val — only if a val query was spent
    eta: float | None            # the Ladder threshold at query time
    accepted: bool | None        # Ladder verdict (None if not queried)
    global_k_at_query: int | None  # cumulative distinct val queries when this was scored
    cost_usd: float              # NEW OpenRouter spend to produce this node (dev + any val); cache hits = 0.0
    cum_cost_usd: float          # running total new spend across the whole tree at this node (§9)
    timestamp: str
```

The `config_hash` is the content address of the `PipelineSpec`; because state is the *ordered composition of levers*, a node's full configuration is reconstructable from the path to root. Cached LLM outputs are keyed by `config_hash`, so re-walking explored territory is free.

### 6.2 Capabilities

- **Resume from any node** — `search.py --from <node_id>` opens a new branch off that state. Fork three recalibration ideas off one parent and compare them as sibling branches.
- **Free dev replay** — revisiting a `config_hash` returns cached predictions; no API spend, no new K.
- **Honest global K** — the ledger is the source of truth for total distinct val queries; the final band (§7) reads K from it.
- **Cost ledger** — each node carries its own new spend (`cost_usd`) and the running total (`cum_cost_usd`), so cost-per-experiment is queryable directly from the tree and the run's total spend is the last node's `cum_cost_usd` (§9).
- **Visualization** — the notebook renders the tree (nodes colored by accept/reject/dead-end; edges labeled by lever), so the search path is legible at a glance. This rendered tree + the query ledger **is** the rigor artifact for the presentation.

### 6.3 Persistence

JSONL append-only under `outputs/ledger/<run>.jsonl` (gitignored, like other run artifacts), plus a small committed `manifest` recording the frozen registry version and split config used, so a run is reproducible.

---

## 7. Reporting (extends `evaluate.py`)

- **Headline number from `test`, touched once**, on the frozen final pipeline — reported both as raw S and **ceiling-normalized** S (the project's reliability-denominator move).
- **K-corrected band.** The confidence band on the selected-pipeline val→test claim is widened to reflect the realized global K (Ladder bound). State K explicitly: "selected from K=__ distinct val queries across the search tree."
- **The query ledger and rendered tree** are presented as evidence the search did not launder noise — the every-query-disclosed Ladder discipline, made visible.
- **Counterfactual sensitivity** (delta-vector directional agreement + magnitude correlation) reported as a signature metric, on test.
- **Cost summary** — total OpenRouter spend for the whole search (the tree's final `cum_cost_usd`), plus a per-experiment cost table (cost vs. ΔS), so the spend that bought the reported number is itself disclosed (§9).

---

## 8. Data flow (end to end)

```
make_split (3-way, family-disjoint, required→test)
   │
   ▼
SearchContract (goal + allowed levers + budgets + gate policy)
   │
   ▼
search.py loop ──► propose lever ──► fit+score on DEV (cached, free, unlimited)
   │                                        │
   │                                  promote? ──no──► log dead-end node, continue
   │                                        │yes
   │                                  GatePolicy check ──► (pause for human if triggered)
   │                                        │
   │                                  Ladder gate on VAL (spend 1 global K) ──► accept iff ΔS>η
   │                                        │
   ▼                                  append ExperimentNode to ledger DAG
check-in at budget ──► human redirects / approves / rewinds-to-node / extends
   │
   ▼ (when search frozen)
final pipeline ──► score ONCE on TEST ──► raw S + ceiling-normalized S + K-corrected band
```

No lever ever sees test; val is reached only through the η-gate; dev is free and unlimited. Every scored step also records its new OpenRouter spend on the node; the loop halts before cumulative spend would exceed `cost_cap_usd` (§9).

---

## 9. Compute & cost model

This makes explicit *what runs where* and *what it costs* — both the who-pays boundary and per-experiment dollar accounting.

### 9.1 Three layers, three cost profiles

| Layer | What it does | Cost |
|---|---|---|
| **Claude Code (subscription)** | The search *intelligence*: proposes the next lever from theory + observed results, writes rationales, decides when to spend a val query, when to branch/stop. Drives the loop by calling the rails below. | Flat-rate subscription. No per-experiment charge. |
| **Python rails (`spec/levers/ladder/ledger/search/report`)** | Deterministic bookkeeping and guardrails: Ladder gate, global-K accounting, content-addressed cache, tree persistence, K-corrected bands, cost accounting. | $0. No LLM, no network. |
| **OpenRouter (metered)** | The only paid layer: inside `score_fn` → `evaluate` → `Pipeline.predict_batch` → `LLMClient`, the simulator model is called on data to produce distributions. | Per-token $; the thing we cap and track. |

The search loop is therefore cheap to *run* and only spends money where it scores a candidate on data. Cache hits (re-walked configs) cost **$0** of new spend.

### 9.2 Capturing cost

`LLMClient.Usage` currently tracks calls/cache-hits/tokens but **not dollars**. We extend it:

- Request native cost from OpenRouter by sending `extra_body={"usage": {"include": True}}`; read `usage.cost` (USD) off each completion and store it in the cached record alongside the token counts.
- **Fallback:** if a response omits `cost`, estimate from `prompt_tokens`/`completion_tokens` against a small per-model price table in `config.py` (keyed by model id). The estimate is marked so reports can flag any model lacking native cost.
- `Usage` gains `cost_usd: float`, accumulating **new** spend only (a cache hit adds $0, matching the K-accounting philosophy that re-walking explored territory is free).

### 9.3 Attribution & enforcement

- **Per experiment:** `score_fn` snapshots `client.usage.cost_usd` before and after scoring a `PipelineSpec` and returns the delta in its breakdown. That delta is written to the node's `cost_usd`; `cum_cost_usd` is the running tree total. Cost-per-experiment is thus a first-class, queryable field — the primary thing the operator asked to track.
- **Enforcement:** before any step that would spend (a dev or val score that isn't a pure cache hit), the loop checks `cum_cost_usd + projected ≤ cost_cap_usd` (default **$1000**). It pauses for human approval at `gate_at_cost_frac` of the cap and hard-halts rather than cross it. Projected cost uses a cheap heuristic (records × recent mean cost-per-record for that model).
- **Reporting:** the final report states total spend and a per-experiment cost table (§7), so the dollar cost that produced the headline number is disclosed alongside it.

---

## 10. Open parameters (decide during implementation, not now)

1. **val size / η calibration** — pick val fraction so the bootstrap η is below the smallest lever ΔS worth claiming. Check empirically on a pilot before the real search.
2. **η multiple** — how many noise-floor units define "significant." Start conservative (favor false rejects over false accepts; an over-strict gate costs a real lever, an under-strict gate costs the honesty claim).
3. **Default budgets** — `autonomy_budget`, `val_query_budget` starting values; tune after one dry run on dev only.
4. **Surprise detector** — what counts as a "surprising" accept/reject for `gate_on_surprise` (e.g., dev-predicted direction contradicted by val).
5. **Price-table fallback** — per-model $/token entries used only when OpenRouter omits native `usage.cost`. Seed with the models the search is allowed to swap in; native cost is preferred whenever present.

---

## 11. Effort fit

| Piece | Est. | Notes |
|---|---|---|
| `levers.py` registry + seeded levers | ~1.5h | Most levers wrap existing predictors/calibrators. |
| `ledger.py` tree + persistence + viz | ~1.5h | JSONL + a tree-render cell. |
| `search.py` contract + loop + Ladder gate | ~1.5h | Gate logic is small; Ladder is a few lines. |
| Cost capture in `LLMClient` + cap enforcement | ~0.5h | `usage.cost` capture + price-table fallback + per-node attribution. |
| Wiring + pilot η check | ~1h | Confirm val sizing before real search. |

~6h to stand up; leaves the remaining time budget for actually *running* levers.

---

## 12. Why this is defensible (one-paragraph summary)

The literature says an automated benchmark maximizer overfits its holdout and can show large gains from pure noise even under full disclosure. So we don't build a maximizer — we build a *guarded explorer*: a pre-registered lever set, a Claude-driven loop with bounded autonomy and flexible human gates, every val query metered through a Ladder rule with a data-derived threshold, and a single global K-counter that keeps the overfitting story honest across the entire branching search. The reported number comes from test data touched exactly once. The guard is not overhead — it is the artifact that proves the result is real.
