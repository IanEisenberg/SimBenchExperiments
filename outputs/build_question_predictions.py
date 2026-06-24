"""Build one consolidated per-question CSV covering ALL of SimBench (13,510 Qs).

One row per SimBench question (pop + grouped, all dev/val/test buckets) with:
  * the raw human distribution (truth)
  * our system's prediction + the paper-faithful baseline + uniform reference
  * the SimBench score (Eq. 2) for each, computed with FULL-split normalizers
  * queryable metadata (dataset, task_kind, splits, segment, required-Q, ...)

Predictions are regenerated entirely FROM CACHE (network disabled: a miss raises),
so this costs nothing and is deterministic. anti_flattening and simbench_faithful
are 100%% cache-covered across all buckets; the final task-kind router is only
partially cached, so "our system" here is anti_flattening (S~=40.8, on par with the
router's held-out TEST S=40.7) -- our best system with full per-question coverage.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scrye.config import REQUIRED_QUESTIONS
from scrye.data import load_all
from scrye.evaluate import build_normalizers
from scrye.experiment import make_client
from scrye.llm import LLMResponse, _cache_key
from scrye.persona import get_strategy
from scrye.pipeline import Pipeline
from scrye.predict import ZeroShotPredictor
from scrye.scoring import response_entropy, simbench_score, tvd_to_uniform
from scrye.splits import make_split

MODEL = "gemini-3.1-flash-lite"
OUR_STRATEGY = "anti_flattening"      # our headline full-coverage system
OUT = Path("/Users/ian/Projects/Scrye_Project/outputs/simbench_question_predictions.csv")

# Task kind per source dataset. SimBench's group_prompt holds the persona ("You are
# from Czechia"), not the survey name, and options are bare letters -- so the content
# heuristic in scrye.taskkind can't fire. dataset_name is the reliable task signal.
DATASET_KIND = {
    "ESS": "opinion_survey", "Afrobarometer": "opinion_survey",
    "LatinoBarometro": "opinion_survey", "OpinionQA": "opinion_survey",
    "ISSP": "opinion_survey", "GlobalOpinionQA": "opinion_survey",
    "TISP": "opinion_survey", "ConspiracyCorr": "opinion_survey",
    "DICES": "opinion_survey",
    "NumberGame": "knowledge", "WisdomOfCrowds": "knowledge", "OSPsychMGKT": "knowledge",
    "Choices13k": "risky_choice",
    "MoralMachine": "moral_dilemma", "MoralMachineClassic": "moral_dilemma",
    "OSPsychBig5": "personality_scale", "OSPsychMACH": "personality_scale",
    "OSPsychRWAS": "personality_scale",
    "ChaosNLI": "other", "Jester": "other",
}


class CacheMiss(Exception):
    pass


def cache_only(client):
    """Disable network: cache hit -> return; miss -> raise CacheMiss (no spend)."""
    read = client._read_cache

    def complete(messages, **ov):
        key = _cache_key(client._request_payload(messages, **ov))
        hit = read(key)
        if hit is None:
            raise CacheMiss(key)
        client.usage.cache_hits += 1
        return LLMResponse(text=hit.get("text", ""), model=client.model, cached=True, raw=hit)

    client.complete = complete  # type: ignore[method-assign]
    return client


def _qid(rec) -> str:
    raw = json.dumps([rec.dataset_name, rec.split, rec.input_template,
                      sorted(rec.segment.items())], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _required_q(template: str) -> str:
    low = template.lower()
    for name, needle in REQUIRED_QUESTIONS.items():
        if needle in low:
            return name
    return ""


def _dist_json(d) -> str:
    return json.dumps({str(k): round(float(v), 6) for k, v in d.items()})


def main() -> None:
    allr = load_all()
    full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")  # required-Qs default-pinned to test
    normalizers = build_normalizers(split.dev + split.val + split.test)

    client = cache_only(make_client(MODEL, max_retries=10, timeout=90))
    our_pipe = Pipeline(ZeroShotPredictor(client, strategy=get_strategy(OUR_STRATEGY)))
    faithful_pipe = Pipeline(ZeroShotPredictor(client, strategy=get_strategy("simbench_faithful")))

    cols = [
        "question_id", "dataset", "task_kind", "eval_bucket", "simbench_file",
        "is_population", "num_segment_vars", "segment_vars", "segment",
        "n_options", "group_size", "required_question", "question_text", "options",
        "human_dist", "truth_entropy", "tvd_to_uniform", "normalizer_Z", "uniform_score",
        "our_system", "our_pred", "our_score",
        "faithful_pred", "faithful_score",
    ]

    rows, misses = [], 0
    for bucket in ("dev", "val", "test"):
        for rec in split.subset(bucket):
            truth, options = rec.human_answer, list(rec.options)
            z = normalizers.get((rec.split, rec.dataset_name))
            uni = {o: 1.0 / len(options) for o in options}
            try:
                our_pred = our_pipe.predict(rec)
                faithful_pred = faithful_pipe.predict(rec)
            except CacheMiss:
                misses += 1
                continue
            rows.append({
                "question_id": _qid(rec),
                "dataset": rec.dataset_name,
                "task_kind": DATASET_KIND.get(rec.dataset_name, "other"),
                "eval_bucket": bucket,
                "simbench_file": rec.split,
                "is_population": rec.is_population,
                "num_segment_vars": rec.num_grouping_vars,
                "segment_vars": "|".join(rec.grouping_keys),
                "segment": json.dumps(rec.segment, sort_keys=True),
                "n_options": rec.num_options,
                "group_size": rec.group_size if rec.group_size is not None else "",
                "required_question": _required_q(rec.input_template),
                "question_text": rec.input_template,
                "options": json.dumps(options),
                "human_dist": _dist_json(truth),
                "truth_entropy": round(response_entropy(truth), 6),
                "tvd_to_uniform": round(tvd_to_uniform(truth), 6),
                "normalizer_Z": round(z, 6) if z else "",
                "uniform_score": round(simbench_score(uni, truth, options=options, normalizer=z), 4),
                "our_system": OUR_STRATEGY,
                "our_pred": _dist_json(our_pred),
                "our_score": round(simbench_score(our_pred, truth, options=options, normalizer=z), 4),
                "faithful_pred": _dist_json(faithful_pred),
                "faithful_score": round(simbench_score(faithful_pred, truth, options=options, normalizer=z), 4),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    import statistics as st
    from collections import Counter
    print(f"wrote {OUT}  ({len(rows)} rows, {len(cols)} cols)  cache misses={misses}")
    print(f"  mean our_score      = {st.mean(r['our_score'] for r in rows):.2f}")
    print(f"  mean faithful_score = {st.mean(r['faithful_score'] for r in rows):.2f}")
    print(f"  mean uniform_score  = {st.mean(r['uniform_score'] for r in rows):.2f}")
    print("  eval_bucket:", dict(Counter(r['eval_bucket'] for r in rows)))
    print("  simbench_file:", dict(Counter(r['simbench_file'] for r in rows)))
    print("  task_kind:", dict(Counter(r['task_kind'] for r in rows)))
    print(f"  required-Q rows: {sum(1 for r in rows if r['required_question'])}")


if __name__ == "__main__":
    main()
