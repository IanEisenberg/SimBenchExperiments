"""Is K=50 leaving score on the table? A FREE K-sensitivity diagnostic.

The 50 per-persona distributions for grounded_averaging are already cached, so we
can reconstruct them (cache hits only) and, for each panel size K' <= 50, average
random K'-subsets and score the result. If the score-vs-K' curve has flattened by
50, a larger panel (e.g. 250) will not help — the system is bias-limited, not
sample-limited. If it is still climbing, more personas are worth the new calls.

    uv run python scripts/k_sensitivity.py
"""

from __future__ import annotations

import random

import numpy as np

from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers
from simbench_exp.experiment import make_client
from simbench_exp.nemotron import default_bank
from simbench_exp.persona import persona_dist_messages
from simbench_exp.predict import _extract_json_object
from simbench_exp.splits import make_split

MODEL = "gemini-3.1-flash-lite"
K = 50


def _tvd(p: dict, q: dict, opts) -> float:
    return 0.5 * sum(abs(p.get(o, 0.0) - q.get(o, 0.0)) for o in opts)


def _persona_dist(client, record, text, i):
    resp = client.complete(persona_dist_messages(record, text), temperature=0.0, seed=i).text
    parsed = _extract_json_object(resp)
    if not parsed:
        return None
    draw = {o: max(0.0, float(parsed.get(o, 0.0) or 0.0)) for o in record.options}
    tot = sum(draw.values())
    return {o: draw[o] / tot for o in record.options} if tot > 0 else None


def main() -> None:
    recs = load_all()
    allrecs = recs["pop"] + recs["grouped"]
    norm = build_normalizers(allrecs)
    split = make_split(allrecs, seed=0, unit="question")
    target = [r for r in split.dev if r.dataset_name == "OpinionQA" and r.split == "pop"]

    client = make_client(MODEL, max_retries=10, timeout=90)
    bank = default_bank()

    # Reconstruct the 50 per-persona distributions per record (cache hits).
    per_record_dists = []
    truths, opts_list, zs = [], [], []
    for rec in target:
        panel = bank.panel_for(rec, K, base_seed=0)
        dists = [d for d in (_persona_dist(client, rec, t, i) for i, t in enumerate(panel))
                 if d is not None]
        if not dists:
            continue
        per_record_dists.append(dists)
        truths.append(rec.human_answer)
        opts_list.append(rec.options)
        zs.append(norm.get((rec.split, rec.dataset_name), 1.0) or 1.0)

    grid = [5, 10, 15, 20, 25, 30, 40, 50]
    n_draws = 25  # random subsets per K' to denoise (K'=50 uses the full panel)
    print(f"OpinionQA pop dev, n={len(per_record_dists)} records, "
          f"each with {K} cached persona distributions\n")
    print(" K'   mean_score   (avg of random K'-subsets, full panel at K'=50)")
    rng = random.Random(0)
    for kp in grid:
        rec_scores = []
        for dists, truth, opts, z in zip(per_record_dists, truths, opts_list, zs):
            if kp >= len(dists):
                subsets = [dists]
            else:
                subsets = [rng.sample(dists, kp) for _ in range(n_draws)]
            sub_scores = []
            for sub in subsets:
                agg = {o: float(np.mean([d[o] for d in sub])) for o in opts}
                s = 100.0 * (1.0 - _tvd(truth, agg, opts) / z)
                sub_scores.append(s)
            rec_scores.append(float(np.mean(sub_scores)))
        print(f"{kp:3d}     {np.mean(rec_scores):7.2f}")


if __name__ == "__main__":
    main()
