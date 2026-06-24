---
# Scrye_Project-y7ea
title: ask CLI — predict survey-response distribution for a freeform question
status: completed
type: feature
priority: normal
created_at: 2026-06-23T15:01:26Z
updated_at: 2026-06-23T15:50:21Z
---

Tiny CLI that runs the project's best method (calibrated_commitment @ gemini-3.1-flash-lite) to predict how a population would answer a multiple-choice question.

Decisions (brainstorm):
- Output: population distribution rendered as text; --json for strict JSON.
- Conditioning: free-text audience (--as / wizard prompt), injected into the calibrated_commitment population phrase.
- Options: user-provided; blank => 4-point Likert (A Agree / B Somewhat agree / C Somewhat disagree / D Disagree).
- Model: gemini-3.1-flash-lite default, --model override.
- No abstention on freeform questions (dataset-level AbstainCalibrator can't judge a cold question); always answer.
- Invocation: naive run => step-by-step wizard; args => non-interactive.

## TODO
- [ ] Write design doc (docs/superpowers/specs)
- [ ] Implement core ask() library fn
- [ ] Implement CLI (wizard + args + output)
- [ ] Add console entry point
- [ ] Tests (no API key needed)
- [ ] Manual smoke test (live)

## Summary of Changes

- src/scrye/ask.py — ask() core: builds a faithful SimBenchRecord (letter labels + Options block, uniform placeholder for unused human_answer), runs calibrated_commitment, returns a normalized AskResult.
- src/scrye/cli_ask.py — scrye-ask CLI: bare run => wizard; args => non-interactive (--options semicolon list, --as/--audience, --model, --json). Friendly error on missing OPENROUTER_API_KEY.
- src/scrye/persona.py — CalibratedCommitmentStrategy gains optional audience override; byte-identical to the validated prompt when audience is None (locked by a test).
- pyproject.toml — [project.scripts] scrye-ask.
- tests/test_ask.py (11) + tests/test_cli_ask.py (9), no API key. Full suite green; live smoke test passed across conditioned/text, Likert/JSON, and custom-options modes.
- No abstention on freeform questions (dataset-level AbstainCalibrator inapplicable); always answers.
