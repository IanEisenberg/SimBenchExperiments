"""`simbench-ask` — CLI to predict a population's answer distribution for a question.

Run it bare and a wizard walks you through the question; pass arguments and it
runs non-interactively (easy to script). Output is the predicted distribution over
the answer options, as a text block or, with ``--json``, strict JSON.

    simbench-ask "Will remote work keep growing?" \\
        --options "Agree;Somewhat agree;Somewhat disagree;Disagree" \\
        --as "US tech workers in 2025"

The heavy lifting lives in :func:`simbench_exp.ask.ask`; this module is arg-parsing,
the interactive wizard, and rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .ask import DEFAULT_ASK_MODEL, DEFAULT_LIKERT, AskResult, ask
from .llm import LLMClient


def parse_options(raw: str | None) -> list[str] | None:
    """Split a ``--options`` string on semicolons; ``None`` for blank.

    Semicolons (not commas) separate options, so option text may contain commas.
    """
    if not raw or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(";")]
    parts = [p for p in parts if p]
    return parts or None


def render_text(res: AskResult) -> str:
    """Human-readable distribution block."""
    audience = res.audience or "general population"
    header = f"Q: {res.question}   ·  audience: {audience}   ·  model: {res.model}"
    width = max((len(t) for t in res.options.values()), default=0)
    lines = ["Distribution:"]
    for label, text in res.options.items():
        prob = res.distribution.get(label, 0.0)
        lines.append(f"  {label}  {text:<{width}}  {prob:.2f}")
    return header + "\n" + "\n".join(lines)


def render_json(res: AskResult) -> str:
    """Strict, machine-parseable JSON."""
    return json.dumps(
        {
            "question": res.question,
            "audience": res.audience,
            "model": res.model,
            "options": res.options,
            "distribution": res.distribution,
        },
        indent=2,
        ensure_ascii=False,
    )


def run(
    question: str,
    options: Sequence[str] | None,
    audience: str | None,
    model: str,
    as_json: bool,
    *,
    client: LLMClient | None = None,
) -> str:
    """Predict and render — the testable core of ``main`` (no I/O)."""
    res = ask(question, options=options, audience=audience, model=model, client=client)
    return render_json(res) if as_json else render_text(res)


def _wizard() -> tuple[str, list[str] | None, str | None, str]:
    """Interactively collect the question and conditioning. Enter accepts defaults."""
    likert = " / ".join(DEFAULT_LIKERT)
    question = input("Question: ").strip()
    options = parse_options(
        input(f"Answer options (semicolon-separated) [Enter = 4-point Likert: {likert}]: ")
    )
    audience = input("Audience / population [Enter = general population]: ").strip() or None
    model = input(f"Model [{DEFAULT_ASK_MODEL}]: ").strip() or DEFAULT_ASK_MODEL
    return question, options, audience, model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simbench-ask",
        description="Predict how a population would answer a multiple-choice question.",
    )
    p.add_argument("question", nargs="?", help="the question text (or use --question)")
    p.add_argument("--question", dest="question_opt", help="the question text")
    p.add_argument(
        "--options",
        help='semicolon-separated answer options; omit for the 4-point Likert default',
    )
    p.add_argument(
        "--as",
        "--audience",
        dest="audience",
        help='free-text population to condition on, e.g. "women aged 30-49 in Finland"',
    )
    p.add_argument("--model", default=DEFAULT_ASK_MODEL, help="model key or id")
    p.add_argument("--json", dest="as_json", action="store_true", help="emit strict JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    question = args.question or args.question_opt

    if question:
        options = parse_options(args.options)
        audience = args.audience
        model = args.model
    elif sys.stdin.isatty():
        question, options, audience, model = _wizard()
        if not question:
            parser.error("no question provided")
    else:
        parser.error("no question provided (pass one as an argument or run interactively)")

    try:
        print(run(question, options, audience, model, args.as_json))
    except RuntimeError as exc:  # e.g. OPENROUTER_API_KEY not set
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
