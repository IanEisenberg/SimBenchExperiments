"""Stage 18 Round 4 — does RICHER persona conditioning make personas bite?

Compares two renderings of the same fixed 50-persona panel on OpinionQA pop dev:
  * summary  — the one-sentence `persona` field (Rounds 1-3; fully cached)
  * rich     — explicit demographics + work + cultural background + interests

For each, reports the grounded_averaging score AND the persona-bite metrics
(inter-persona disagreement, mode-agreement). The hypothesis: rich conditioning
raises inter-persona disagreement toward the real human spread and improves
location. Re-runs are free from cache.

    uv run python scripts/run_rich.py
"""

from __future__ import annotations

import math
import random

import numpy as np

from simbench_exp.data import load_all
from simbench_exp.evaluate import build_normalizers
from simbench_exp.experiment import make_client
from simbench_exp.nemotron import PersonaBank
from simbench_exp.persona import persona_dist_messages
from simbench_exp.predict import _extract_json_object
from simbench_exp.splits import make_split

MODEL, K = "gemini-3.1-flash-lite", 50


def _norm_entropy(v: np.ndarray) -> float:
    v = v[v > 0]
    return float(-(v * np.log(v)).sum() / math.log(len(v))) if len(v) > 1 else 0.0


def _pdist(client, rec, text, i):
    resp = client.complete(persona_dist_messages(rec, text), temperature=0.0, seed=i).text
    p = _extract_json_object(resp)
    if not p:
        return None
    d = np.array([max(0.0, float(p.get(o, 0.0) or 0.0)) for o in rec.options])
    return d / d.sum() if d.sum() > 0 else None


def eval_mode(client, df, target, norm, mode, max_workers=5):
    from concurrent.futures import ThreadPoolExecutor

    bank = PersonaBank(df, text_mode=mode)
    rng = random.Random(0)
    scores, inter_tvd, mode_agree, pred_H = [], [], [], []
    for rec in target:
        panel = bank.panel_for(rec, K, base_seed=0)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            raw = list(ex.map(lambda it: _pdist(client, rec, it[1], it[0]), enumerate(panel)))
        M = np.array([d for d in raw if d is not None])
        if len(M) < 5:
            continue
        z = norm.get((rec.split, rec.dataset_name), 1.0) or 1.0
        avg = M.mean(0)
        truth = np.array([rec.human_answer.get(o, 0.0) for o in rec.options])
        scores.append(100.0 * (1.0 - 0.5 * np.abs(truth - avg).sum() / z))
        pred_H.append(_norm_entropy(avg))
        pairs = [(rng.randrange(len(M)), rng.randrange(len(M))) for _ in range(200)]
        inter_tvd.append(np.mean([0.5 * np.abs(M[a] - M[b]).sum() for a, b in pairs if a != b]))
        counts = np.bincount(M.argmax(1), minlength=M.shape[1])
        mode_agree.append(counts.max() / len(M))
    return {
        "mean_score": float(np.mean(scores)),
        "inter_persona_tvd": float(np.mean(inter_tvd)),
        "mode_agreement": float(np.mean(mode_agree)),
        "pred_H": float(np.mean(pred_H)),
    }


def main() -> None:
    import pandas as pd

    from simbench_exp.config import DATA_DIR
    df = pd.read_parquet(DATA_DIR / "nemotron" / "personas_slim.parquet")
    recs = load_all()
    allr = recs["pop"] + recs["grouped"]
    norm = build_normalizers(allr)
    split = make_split(allr, seed=0, unit="question")
    target = [r for r in split.dev if r.dataset_name == "OpinionQA" and r.split == "pop"]
    client = make_client(MODEL, max_retries=10, timeout=90)

    print(f"OpinionQA pop dev, n={len(target)}, truth_H=0.692, "
          f"calibrated_commitment=63.24 (reference)\n")
    print(f"{'mode':>9} | {'score':>6} | {'inter_persona_tvd':>17} | "
          f"{'mode_agreement':>14} | {'pred_H':>6}")
    for mode in ("summary", "rich"):
        m = eval_mode(client, df, target, norm, mode)
        print(f"{mode:>9} | {m['mean_score']:6.2f} | {m['inter_persona_tvd']:17.3f} | "
              f"{m['mode_agreement']:14.3f} | {m['pred_H']:6.3f}")


if __name__ == "__main__":
    main()
