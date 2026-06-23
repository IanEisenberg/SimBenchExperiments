# `scrye-ask` — CLI for predicting a survey-response distribution

**Date:** 2026-06-23 · **Status:** Design (approved) · **Bean:** Scrye_Project-y7ea

## Purpose

A tiny CLI that runs the project's **best deployable method** to predict how a
population would answer a multiple-choice question. Run it bare and a wizard walks
you through the question; pass arguments and it runs non-interactively so it is
easy to script. Output is the predicted probability **distribution** over the
answer options, rendered as text.

## What "best method" means here

The final report (`docs/experiments/stage-17-final-test.md`) lands on two
statistically-tied systems, and leads with the simpler one:
**`calibrated_commitment`** (the Stage-10 prompt-search winner) on
**`gemini-3.1-flash-lite`** (the dominant lever, +15.9 overall on test).

The other moving parts of the system of record do **not** apply to a freeform
question and are deliberately dropped:

- **`AbstainCalibrator`** decides abstention *per dataset*, by comparing the
  model's vs. uniform's TVD against ground truth at fit time, then emitting
  uniform for every record from a flagged dataset (`src/scrye/calibrate.py`). A
  typed-in question has no `dataset_name` and no ground truth, so the calibrator
  can never fire. Per the decision below, the CLI does **no abstention** — it
  always answers.
- The **task-kind router** / **task-context** corpus need SimBench sibling items
  and washed out on held-out test anyway. Not used.

So for a cold question the best method reduces cleanly to:
`calibrated_commitment` prompt → parse JSON distribution → render.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Output | Population distribution over the options, rendered as text. `--json` for strict JSON. |
| Conditioning | Free-text audience string (`--as`, or wizard prompt), woven into the prompt's population phrase. |
| Options | User-provided; **blank ⇒ 4-point Likert** (`A: Agree, B: Somewhat agree, C: Somewhat disagree, D: Disagree`). |
| Model | `gemini-3.1-flash-lite` default; `--model` override (accepts a `config.OPENROUTER_MODELS` key or a raw id). |
| Abstention | **None** for freeform questions — always answer. |
| Command | `scrye-ask` console entry point. |
| `--options` syntax | Semicolon-separated list. |
| Structure | Library core (`src/scrye/ask.py`) + thin CLI (`src/scrye/cli_ask.py`). |

## Architecture

```
scrye/cli_ask.py          CLI: arg parsing + wizard + rendering  (entry: scrye-ask)
  └── scrye.ask.ask(...)  core: build record → calibrated_commitment predict → AskResult
        ├── scrye.persona  reuse CalibratedCommitmentStrategy system prompt + ask
        ├── scrye.predict.ZeroShotPredictor (strategy="calibrated_commitment")
        └── scrye.experiment.make_client / config.OPENROUTER_MODELS
```

### Core — `src/scrye/ask.py`

```python
@dataclass(frozen=True)
class AskResult:
    question: str
    audience: str | None
    model: str
    options: dict[str, str]        # {"A": "Agree", ...}  label -> text
    distribution: dict[str, float] # {"A": 0.44, ...}      label -> probability

DEFAULT_LIKERT = ("Agree", "Somewhat agree", "Somewhat disagree", "Disagree")
DEFAULT_ASK_MODEL = "gemini-3.1-flash-lite"

def ask(
    question: str,
    options: Sequence[str] | None = None,   # None/empty -> DEFAULT_LIKERT
    audience: str | None = None,            # None -> unconditioned population
    *,
    model: str = DEFAULT_ASK_MODEL,
    client: LLMClient | None = None,        # inject for tests / shared cache
) -> AskResult: ...
```

Behaviour:

1. **Options** → assign labels `A, B, C, …` to the option texts. Empty ⇒
   `DEFAULT_LIKERT`. Require ≥ 2 options.
2. **Build a `SimBenchRecord`** faithful to the real data shape:
   - `input_template = f"{question}\n\nOptions:\n(A): {t0}\n(B): {t1}\n…"`
   - `options = ("A", "B", "C", …)` (bare letters — matches the dataset's
     `human_answer` keys)
   - `human_answer` = uniform placeholder (unused; only scoring reads it)
   - `dataset_name="__ask__"`, `split="pop"`, empty `segment`/`group_prompt`.
3. **Predict** with `ZeroShotPredictor(client, strategy="calibrated_commitment")`
   via the pipeline, so parsing/normalization/uniform-fallback behaviour is
   identical to the evaluated system.
4. **Audience injection.** `CalibratedCommitmentStrategy` derives its "who" from a
   *structured segment*, not free text, so a small refactor exposes the population
   descriptor as an override:
   - Add an optional `audience: str | None` parameter threaded into the
     calibrated_commitment message builder (default `None`).
   - `audience=None` ⇒ the prompt is **byte-identical** to the current
     `CalibratedCommitmentStrategy` output for an unconditioned record (guarded by
     a test).
   - `audience="US tech workers in 2025"` ⇒ the opening line becomes
     *"Consider a large, representative sample of US tech workers in 2025."* The
     system prompt and the distributional ask are unchanged.
   - Implementation: factor the shared system prompt + ask text so both the
     registered strategy and the CLI path use one source of truth (no copy-paste
     drift).
5. Return `AskResult`.

### CLI — `src/scrye/cli_ask.py` (entry point `scrye-ask`)

The CLI lives inside the importable package (a `console_scripts` entry needs an
importable target, and `scripts/` is not a package). Added to `pyproject.toml`:

```toml
[project.scripts]
scrye-ask = "scrye.cli_ask:main"
```

After `uv sync`, run as `uv run scrye-ask ...`.

**Wizard** (no question argument, interactive TTY) — prompts in order, each with
its default in brackets, Enter accepts the default:

```
Question: Will remote work keep growing?
Answer options [Enter = 4-point Likert: Agree / Somewhat agree / Somewhat disagree / Disagree]:
Audience / population [Enter = general population]: US tech workers in 2025
Model [gemini-3.1-flash-lite]:
```

**Non-interactive:**

```
scrye-ask "Will remote work keep growing?" \
    --options "Agree;Somewhat agree;Somewhat disagree;Disagree" \
    --as "US tech workers in 2025" \
    --model gemini-3.1-flash-lite \
    --json
```

- Question: positional or `--question`.
- `--options` semicolon-separated; omitted ⇒ Likert.
- `--as` / `--audience`: free-text; omitted ⇒ unconditioned.
- `--model`: key or raw id; default `gemini-3.1-flash-lite`.
- `--json`: strict JSON instead of the text block.

**Text output (default):**

```
Q: Will remote work keep growing?   ·  audience: US tech workers in 2025   ·  model: gemini-3.1-flash-lite
Distribution:
  A  Agree              0.44
  B  Somewhat agree     0.31
  C  Somewhat disagree  0.16
  D  Disagree           0.09
```

**JSON output (`--json`):**

```json
{
  "question": "Will remote work keep growing?",
  "audience": "US tech workers in 2025",
  "model": "google/gemini-3.1-flash-lite",
  "options": {"A": "Agree", "B": "Somewhat agree", "C": "Somewhat disagree", "D": "Disagree"},
  "distribution": {"A": 0.44, "B": 0.31, "C": 0.16, "D": 0.09}
}
```

### Errors & edge cases

- **Missing `OPENROUTER_API_KEY`** → a one-line, friendly error (point at
  `.env`), not a traceback. (A live call is required; the on-disk cache makes
  repeats free and deterministic.)
- **< 2 options** → error before any API call.
- **Unparseable model response** → the predictor already falls back to uniform
  and increments `n_parse_failures`; the CLI surfaces a brief
  "model response unparseable — showing uniform" note. (This is the predictor's
  existing robustness, not an abstention decision.)

## Testing — `tests/test_ask.py` (no API key)

The LLM client is stubbed (a fake `LLMClient.complete` returning a canned JSON
string), following the project's "parsing/scoring tests work without an API key"
convention.

- Option parsing + Likert default (blank/`None` ⇒ the four Likert texts, labels
  `A–D`).
- `< 2` options raises.
- **Prompt faithfulness:** with `audience=None`, the calibrated_commitment
  messages built by the CLI path equal those of the registered
  `CalibratedCommitmentStrategy` for an equivalent unconditioned record (locks the
  "byte-identical when unconditioned" guarantee).
- Audience injection: the audience string appears in the user message's opening
  population phrase; the system prompt is unchanged.
- Distribution parse → normalized, covers exactly the option labels.
- `AskResult` → text render and `--json` shape.

## Out of scope (YAGNI)

- No abstention / uniform-fallback selection logic for freeform questions.
- No multi-question batch mode, no file input.
- No persona / Monte-Carlo / router systems (calibrated_commitment only).
- No `scrye` multi-subcommand CLI — a single `scrye-ask` entry.
```
