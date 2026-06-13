"""End-to-end smoke test for the OpenRouter LLM client.

Makes one cheap call, then repeats it to confirm the on-disk cache returns
the same text for free. Usage:

    uv run python scripts/smoke_llm.py [model_id]
"""

import sys

from scrye.llm import LLMClient

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    client = LLMClient(model=model, max_tokens=64)

    prompt = (
        "Respond with exactly one word: the capital of France. No punctuation."
    )
    r1 = client.prompt(prompt)
    print(f"model        : {model}")
    print(f"call 1 text  : {r1!r}  (cached={False})")

    r2 = client.complete([{"role": "user", "content": prompt}])
    print(f"call 2 text  : {r2.text!r}  (cached={r2.cached})")

    print(f"usage        : {client.usage.as_dict()}")
    assert r2.cached, "second identical call should hit the cache"
    print("OK: live call succeeded and cache works.")


if __name__ == "__main__":
    main()
