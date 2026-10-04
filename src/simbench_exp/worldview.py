"""Worldview enrichment + ideology calibration (Stage 18, Round 5 / final).

The diagnosis: a census-grounded Nemotron electorate under-differentiates on
opinion questions because the personas describe lifestyle and temperament but not
*values* — and opinion answers hinge on the value/ideology axis (a controlled
probe showed the model differentiates by 0.54 TVD on explicit ideological
archetypes but only ~0.22 on the bare personas). The fix supplies the missing
axis without "randomly applying" it:

  1. **Enrich** — infer an expressive worldview per persona (political lean,
     religiosity, institutional trust, moral outlook), *grounded* in their full
     profile and *sampled* (temperature > 0, seeded) so within-cell diversity is
     preserved rather than collapsed to a stereotype mode.
  2. **Calibrate** — the model softens worldviews toward "moderate", so the
     panel's inferred-ideology mix need not match reality. Post-stratify the panel
     onto OpinionQA's *real* `POLIDEOLOGY` marginal (the Pew sample's ideology
     distribution) so the electorate is representative on the added axis by
     construction — not by trusting the inference.

Each enrichment also returns a structured ideology label (on the OpinionQA scale)
used only for the calibration weights.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .data import SimBenchRecord
from .llm import LLMClient
from .predict import _extract_json_object

#: OpinionQA POLIDEOLOGY scale (the calibration buckets).
IDEOLOGY_BUCKETS: tuple[str, ...] = (
    "very liberal", "liberal", "moderate", "conservative", "very conservative",
)

ENRICH_SYSTEM = (
    "You are a social scientist building a realistic profile of one real person. "
    "Given their demographics and life, infer their likely social and political "
    "worldview: where they probably fall on contested social and political issues, "
    "their religiosity, their trust in government and institutions, and the moral "
    "intuitions that guide them. Ground every inference in specifics of their life. "
    "Be realistic, not a caricature — real people are not uniformly partisan and "
    "often hold mixed views."
)


def enrich_persona(client: LLMClient, persona_text: str, seed: int,
                   temperature: float = 0.7) -> dict:
    """Infer one persona's worldview + self-placed ideology (one cached call).

    Sampled (``temperature`` > 0, seeded) so two similar personas can diverge,
    preserving within-cell spread. Returns ``{"worldview": str, "ideology": str}``;
    ``ideology`` is snapped to :data:`IDEOLOGY_BUCKETS` (defaulting to "moderate").
    """
    user = (
        f"{persona_text}\n\n"
        "Describe this person's likely worldview and values in 3-4 sentences — "
        "political leanings, religiosity, trust in institutions, and moral outlook "
        "— grounded in their background. Then place them on a political scale.\n"
        'Respond with ONLY JSON: {"worldview": "<3-4 sentences>", "ideology": '
        '"<one of: very liberal, liberal, moderate, conservative, very conservative>"}.'
    )
    resp = client.complete(
        [{"role": "system", "content": ENRICH_SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature, seed=seed,
    ).text
    obj = _extract_json_object(resp) or {}
    ideology = str(obj.get("ideology", "moderate")).strip().lower()
    if ideology not in IDEOLOGY_BUCKETS:
        ideology = next((b for b in IDEOLOGY_BUCKETS if b in ideology), "moderate")
    return {"worldview": str(obj.get("worldview", "")).strip(), "ideology": ideology}


def opinionqa_ideology_marginal(grouped_records) -> dict[str, float]:
    """The real US ideology distribution from OpinionQA's POLIDEOLOGY segments.

    Sums ``group_size`` over rows grouped solely by POLIDEOLOGY (the Pew sample's
    respondent counts) → population share per ideology bucket. This is population
    *structure*, not survey answers, so using it to make the panel representative
    carries no answer-label leakage (the same spirit as census post-stratification).
    """
    counts: dict[str, float] = defaultdict(float)
    for r in grouped_records:
        if r.grouping_keys == ("POLIDEOLOGY",):
            val = str(r.segment.get("POLIDEOLOGY", "")).strip().lower()
            if val in IDEOLOGY_BUCKETS:
                counts[val] += float(r.group_size or 0)
    total = sum(counts.values())
    return {b: counts.get(b, 0.0) / total for b in IDEOLOGY_BUCKETS} if total > 0 else {}


def calibration_weights(ideologies: list[str], target: dict[str, float]) -> list[float]:
    """Post-stratification weights aligning the panel's ideology mix to ``target``.

    ``w_i ∝ P_target(ideology_i) / P_panel(ideology_i)`` (0 if the panel lacks that
    bucket), normalized to sum 1. Personas in over-represented buckets are
    down-weighted so the weighted electorate matches the real ideology marginal.
    Falls back to equal weights if the inputs are degenerate.
    """
    n = len(ideologies)
    if n == 0:
        return []
    panel = Counter(ideologies)
    raw = [target.get(ide, 0.0) / (panel[ide] / n) if panel[ide] else 0.0
           for ide in ideologies]
    s = sum(raw)
    return [w / s for w in raw] if s > 0 else [1.0 / n] * n


def build_enriched_panel(
    client: LLMClient, panel_texts: list[str], grouped_records, *,
    base_seed: int = 0, calibrate: bool = True, max_workers: int = 5,
) -> dict:
    """Enrich a fixed panel and compute its ideology-calibration weights.

    Returns ``{"texts": [persona+worldview, ...], "ideologies": [...],
    "weights": [...], "panel_marginal": {...}, "target_marginal": {...}}``.
    With ``calibrate=False`` the weights are uniform (the ablation).
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        enr = list(ex.map(lambda it: enrich_persona(client, it[1], base_seed + it[0]),
                          enumerate(panel_texts)))
    texts = [f"{t}\n\nWorldview and values: {e['worldview']}"
             for t, e in zip(panel_texts, enr)]
    ideologies = [e["ideology"] for e in enr]
    target = opinionqa_ideology_marginal(grouped_records)
    weights = calibration_weights(ideologies, target) if calibrate else None
    n = len(panel_texts)
    panel_marginal = {b: sum(i == b for i in ideologies) / n for b in IDEOLOGY_BUCKETS}
    return {"texts": texts, "ideologies": ideologies, "weights": weights,
            "panel_marginal": panel_marginal, "target_marginal": target}
