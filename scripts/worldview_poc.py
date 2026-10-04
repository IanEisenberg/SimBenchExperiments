"""Stage 5 proof-of-concept: does inferring an expressive WORLDVIEW make real
Nemotron personas differentiate on values questions?

Takes a handful of real personas, infers a rich worldview for each (grounded in
their profile, like the persona narratives themselves), then measures whether
their answers to values-laden questions spread out — comparing rich-only vs
rich+worldview. If enrichment lifts the inter-persona spread from ~0.22 toward the
archetype level (0.54), supplying the missing ideological axis is the fix.

    uv run python scripts/worldview_poc.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from simbench_exp.config import DATA_DIR
from simbench_exp.data import load_all
from simbench_exp.experiment import make_client
from simbench_exp.nemotron import PersonaBank, render_persona
from simbench_exp.persona import persona_dist_messages
from simbench_exp.predict import _extract_json_object
from simbench_exp.splits import make_split

MODEL = "gemini-3.1-flash-lite"
N_PERSONAS = 10

ENRICH_SYSTEM = (
    "You are a social scientist building a realistic profile of one real person. "
    "Given their demographics and life, infer their likely social and political "
    "worldview: where they probably fall on contested social and political issues, "
    "their religiosity, their trust in government and institutions, and the moral "
    "intuitions that guide them. Ground every inference in specifics of their life. "
    "Be realistic, not a caricature — real people are not uniformly partisan and "
    "often hold mixed views."
)


def enrich(client, persona_text, seed):
    user = (f"{persona_text}\n\nIn 3-4 sentences, describe this person's likely "
            "worldview and values — political leanings, religiosity, trust in "
            "institutions, and moral outlook — grounded in their background.")
    return client.complete([{"role": "system", "content": ENRICH_SYSTEM},
                            {"role": "user", "content": user}],
                           temperature=0.7, seed=seed).text.strip()


def dist(client, rec, text):
    resp = client.complete(persona_dist_messages(rec, text), temperature=0.0, seed=0).text
    p = _extract_json_object(resp)
    if not p:
        return None
    d = np.array([max(0.0, float(p.get(o, 0.0) or 0.0)) for o in rec.options])
    return d / d.sum() if d.sum() > 0 else None


def spread(dists):
    arr = [d for d in dists if d is not None]
    if len(arr) < 2:
        return float("nan")
    return float(np.mean([0.5 * np.abs(arr[i] - arr[j]).sum()
                          for i in range(len(arr)) for j in range(i + 1, len(arr))]))


def main() -> None:
    import pandas as pd
    df = pd.read_parquet(DATA_DIR / "nemotron" / "personas_slim.parquet")
    bank = PersonaBank(df, text_mode="rich")
    adults = bank.adults.sample(n=N_PERSONAS, random_state=7).to_dict("records")
    rich_texts = [render_persona(r, "rich") for r in adults]

    recs = load_all()
    allr = recs["pop"] + recs["grouped"]
    split = make_split(allr, seed=0, unit="question")
    pool = [r for r in split.dev if r.dataset_name == "OpinionQA" and r.split == "pop"]
    kw = ("government", "religio", "gun", "marriage", "church", "god", "moral", "immigra")
    questions = [r for r in pool if any(k in r.input_template.lower() for k in kw)][:4]

    client = make_client(MODEL, max_retries=10, timeout=90)

    # Stage 1: infer a worldview per persona (varied seeds = sampled, not argmax).
    with ThreadPoolExecutor(max_workers=5) as ex:
        worldviews = list(ex.map(lambda it: enrich(client, it[1], it[0]),
                                 enumerate(rich_texts)))
    print("=== inferred worldviews (first 3) ===")
    for i in range(3):
        print(f"\n[persona {i}] {rich_texts[i].splitlines()[0]}")
        print(f"  -> {worldviews[i]}")

    enriched_texts = [f"{t}\n\nWorldview and values: {w}"
                      for t, w in zip(rich_texts, worldviews)]

    print("\n=== inter-persona spread on values questions ===")
    print(f"{'question':<48} | {'rich-only':>9} | {'+worldview':>10}")
    base_s, enr_s = [], []
    for rec in questions:
        with ThreadPoolExecutor(max_workers=5) as ex:
            base = list(ex.map(lambda t: dist(client, rec, t), rich_texts))
            enr = list(ex.map(lambda t: dist(client, rec, t), enriched_texts))
        b, e = spread(base), spread(enr)
        base_s.append(b); enr_s.append(e)
        stem = rec.input_template.split("Options")[0].strip().replace("\n", " ")[:46]
        print(f"{stem:<48} | {b:9.3f} | {e:10.3f}")
    print(f"{'MEAN':<48} | {np.nanmean(base_s):9.3f} | {np.nanmean(enr_s):10.3f}")
    print("\n(archetype reference = 0.54; un-enriched real personas ~ 0.22)")


if __name__ == "__main__":
    main()
