"""Does the persona actually change the answer? A controlled sensitivity probe.

Runs a handful of MAXIMALLY CONTRASTING personas on the same values-laden
OpinionQA questions and prints each persona's distribution side by side. If even
extreme personas give near-identical answers, the model is persona-insensitive
(the core failure). If they diverge sensibly (e.g. the progressive vs the
conservative split on social issues), the model IS sensitive and the problem is
that the Nemotron panel isn't varied/extreme enough or the population mix is off.

    uv run python scripts/persona_sensitivity.py
"""

from __future__ import annotations

import numpy as np

from simbench_exp.data import load_all
from simbench_exp.experiment import make_client
from simbench_exp.persona import persona_dist_messages
from simbench_exp.predict import _extract_json_object
from simbench_exp.splits import make_split

MODEL = "gemini-3.1-flash-lite"

PERSONAS = {
    "progressive_urban":
        "Maya Chen is a 27-year-old woman living in Brooklyn, New York. She holds a "
        "master's degree in environmental policy and works for a climate-advocacy "
        "nonprofit. She is an atheist, identifies as very liberal and progressive, is "
        "active in social-justice movements, and consistently votes Democratic.",
    "conservative_rural":
        "Earl Whitfield is a 66-year-old man in rural Alabama. He has a high-school "
        "education and is a retired long-haul truck driver. He is a devout Southern "
        "Baptist who attends church weekly, identifies as very conservative, deeply "
        "distrusts the federal government, owns several firearms, and votes Republican.",
    "moderate_suburban":
        "James Romano is a 48-year-old man in suburban Columbus, Ohio. He owns a small "
        "HVAC business, is Catholic, and identifies as a political moderate and "
        "independent — fiscally conservative but socially moderate.",
    "immigrant_liberal":
        "Aisha Rahman is a 33-year-old woman in Minneapolis. She immigrated from "
        "Pakistan as a child, is Muslim, holds an engineering degree, is "
        "community-oriented, and leans liberal on social issues.",
    "generic": "an adult member of the U.S. public",
}

KEYWORDS = ("government", "religio", "immigra", "gun", "climate", "discrimin",
            "abortion", "marriage", "gender", "church", "god", "moral", "welfare",
            "tax", "police", "racial", "vaccine")


def _dist(client, rec, text):
    resp = client.complete(persona_dist_messages(rec, text), temperature=0.0, seed=0).text
    p = _extract_json_object(resp)
    if not p:
        return None
    d = np.array([max(0.0, float(p.get(o, 0.0) or 0.0)) for o in rec.options])
    return d / d.sum() if d.sum() > 0 else None


def main() -> None:
    recs = load_all()
    allr = recs["pop"] + recs["grouped"]
    split = make_split(allr, seed=0, unit="question")
    pool = [r for r in split.dev if r.dataset_name == "OpinionQA" and r.split == "pop"]
    questions = [r for r in pool if any(k in r.input_template.lower() for k in KEYWORDS)][:6]
    client = make_client(MODEL, max_retries=10, timeout=90)

    all_spreads = []
    for rec in questions:
        stem = rec.input_template.split("Options")[0].strip()
        print("\n" + "=" * 88)
        print(stem[:300])
        print(f"options: {dict(enumerate(rec.options))}")
        dists = {}
        for name, text in PERSONAS.items():
            d = _dist(client, rec, text)
            if d is not None:
                dists[name] = d
                print(f"  {name:20s} " + " ".join(f"{o}:{p:.2f}" for o, p in zip(rec.options, d)))
        # spread across the 4 real archetypes (exclude generic)
        arch = [dists[n] for n in PERSONAS if n != "generic" and n in dists]
        if len(arch) >= 2:
            pairs = [0.5 * np.abs(a - b).sum() for i, a in enumerate(arch) for b in arch[i + 1:]]
            spread = float(np.mean(pairs))
            all_spreads.append(spread)
            print(f"  --> mean pairwise TVD across archetypes: {spread:.3f}")

    print("\n" + "=" * 88)
    print(f"MEAN archetype spread over {len(all_spreads)} questions: "
          f"{np.mean(all_spreads):.3f}")
    print("(near 0 => model ignores persona; high => persona drives the answer)")


if __name__ == "__main__":
    main()
