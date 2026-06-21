"""Stage 10 — calibrated-commitment prompt iteration on dev.

Direct-answer prompt variants combining contextualized + anti_flattening with
consensus/entropy awareness and licensed commitment. Scored on the fixed dev-eval
sample vs the incumbents, with mechanism diagnostics (entropy floor, slope,
consensus-commit rate, mode accuracy). Edit CANDIDATES and re-run per round.

Usage:
    uv run python scripts/run_commitment_prompts.py [round_tag]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scrye.data import load_all
from scrye.evaluate import build_normalizers, stratified_sample
from scrye.experiment import make_client
from scrye.persona import (
    PromptStrategy,
    _json_instruction,
    _population_phrase,
    get_strategy,
)
from scrye.predict import ZeroShotPredictor
from scrye.scoring import bootstrap_ci, response_entropy, simbench_score
from scrye.splits import make_split

MODEL = "gemini-3.1-flash-lite"
EVAL_SEED = 42
N_G = int(__import__("os").environ.get("N_G", 700))
N_P = int(__import__("os").environ.get("N_P", 300))
OUTPUTS = Path("outputs/runs"); OUTPUTS.mkdir(parents=True, exist_ok=True)


class AdHoc(PromptStrategy):
    """A candidate prompt: a system message + a direct (no reasoning-first) ask."""

    def __init__(self, name, system, body):
        self.name = name
        self._system = system
        self._body = body          # body(question, who, year_clause, options) -> user str

    def build_messages(self, record):
        who, yc = _population_phrase(record)
        user = self._body(record.input_template.strip(), who, yc, record.options)
        return [{"role": "system", "content": self._system},
                {"role": "user", "content": user}]


# ---- candidates for THIS round (edit per round) --------------------------
_V1_SYS = (
    "You are an expert survey methodologist estimating how a demographic group "
    "answers a survey question. Real groups hold a genuine diversity of views — "
    "never collapse a group to one stereotyped answer, and keep probability mass "
    "on minority views that truly exist. But groups are not always divided: on "
    "many questions there is a strong majority, and the honest estimate is then a "
    "concentrated distribution. Weigh the group's social, cultural, and economic "
    "circumstances, and match the true shape of opinion — commit to a sharp "
    "distribution when the group largely agrees, and spread the mass when they "
    "are genuinely split."
)


def _v1_body(q, who, yc, opts):
    return (f"Consider a large, representative sample of {who}{yc}. Reflect both "
            "their real internal diversity and the genuine degree of consensus on "
            f"this question.\n\n{q}\n\nEstimate how this whole group answers — sharp "
            "if they largely agree, spread if they are divided, keeping minority "
            f"views only where they genuinely exist.\n{_json_instruction(opts)}")


_V2_SYS = (
    "You are an expert survey methodologist. Your job is to match the true shape "
    "of a group's opinion. Some questions have a clear majority — then the right "
    "answer is a concentrated distribution, and hedging across all options is a "
    "mistake. Other questions genuinely divide the group — then preserve that "
    "spread and keep the minority views. Account for the group's social and "
    "cultural circumstances, never reduce them to a stereotype, but do not "
    "manufacture diversity that is not there."
)


def _v2_body(q, who, yc, opts):
    return (f"Consider a large, representative sample of {who}{yc}.\n\n{q}\n\nSense "
            "whether this is a consensus question or a divided one for this group, "
            "then give the distribution that matches — concentrated when they "
            f"largely agree, spread when they are split.\n{_json_instruction(opts)}")


# --- Round 2: refine v1 to recover pop + a mode-targeting variant ---------
_V3_SYS = (
    "You are an expert survey methodologist estimating how a group of people "
    "answers a survey question. Real groups hold a genuine diversity of views — "
    "never collapse them to one stereotyped answer, and keep probability mass on "
    "minority views that truly exist. But not every question divides a group: when "
    "there is a strong majority, the honest estimate is a concentrated "
    "distribution, so commit and be sharp. A broad, general population is usually "
    "divided across several options; a specific, well-defined subgroup is more "
    "often unified. Weigh the group's social, cultural, and economic "
    "circumstances, and match the true shape of opinion — sharp when they "
    "genuinely agree, spread when they are genuinely divided."
)


def _v3_body(q, who, yc, opts):
    return (f"Consider a large, representative sample of {who}{yc}. Reflect both "
            "their real internal diversity and the genuine degree of consensus on "
            "this question — remembering a broad population is usually more divided "
            f"than a narrowly-defined subgroup.\n\n{q}\n\nEstimate how this whole "
            "group answers — sharp if they largely agree, spread if they are "
            f"divided, keeping minority views only where they truly exist.\n"
            f"{_json_instruction(opts)}")


_V4_SYS = (
    "You are an expert survey methodologist. Estimate how a group answers in two "
    "respects at once: (1) which option is most common for this group — get the "
    "leading answer right — and (2) how concentrated or divided the group truly is "
    "around it. Real groups are diverse, so never zero out minority views that "
    "exist or reduce the group to a stereotype; but when one option clearly leads, "
    "give it a clear plurality rather than hedging evenly across options. A broad "
    "population is usually more split than a specific subgroup."
)


def _v4_body(q, who, yc, opts):
    return (f"Consider a large, representative sample of {who}{yc}.\n\n{q}\n\nGive "
            "the group's answer distribution: put the most mass on the option this "
            "group most likely favors, concentrate it when they largely agree and "
            "spread it when they are divided, and keep minority views where they "
            f"genuinely exist.\n{_json_instruction(opts)}")


# --- Round 3: regime-aware (branch on whether a demographic segment exists) --
class RegimeAware(PromptStrategy):
    """One strategy, two regimes: a conditioned (segment present) branch with the
    demographic/diversity/consensus framing, and an unconditioned (empty segment)
    branch with a task-agnostic mode-first framing that survives non-survey tasks."""

    def __init__(self, name, cond_sys, cond_body, uncond_sys, uncond_body):
        self.name = name
        self._cs, self._cb = cond_sys, cond_body
        self._us, self._ub = uncond_sys, uncond_body

    def build_messages(self, record):
        who, yc = _population_phrase(record)
        conditioned = bool(record.segment)
        sysmsg = self._cs if conditioned else self._us
        body = (self._cb if conditioned else self._ub)
        return [{"role": "system", "content": sysmsg},
                {"role": "user", "content": body(record.input_template.strip(), who, yc, record.options)}]


# unconditioned branch: task-agnostic mode-first (works for NumberGame, Jester, …)
_UNCOND_SYS = (
    "You are an expert at predicting how a broad population responds to a question "
    "or task. Estimate the distribution of responses: which option is most common "
    "— get the leading answer right — and how concentrated or divided people are. "
    "When one option clearly leads, give it a clear plurality; when responses are "
    "genuinely split, spread the mass to match. Do not force agreement that is not "
    "there, and do not manufacture division that is not there."
)


def _uncond_body(q, who, yc, opts):
    return (f"Consider the full range of people responding to the following.\n\n{q}"
            "\n\nGive the distribution of their responses: most mass on the most "
            "common answer, concentrated when there is broad agreement and spread "
            f"when responses are genuinely divided.\n{_json_instruction(opts)}")


# conditioned branch A (v5): v4 mode-first, demographic-aware
_COND_SYS_A = (
    "You are an expert survey methodologist estimating how a specific demographic "
    "group answers a survey question. Judge two things at once: (1) which option "
    "this group most favors — get the leading answer right — and (2) how united or "
    "divided they are around it. Real groups are diverse: never reduce them to a "
    "stereotype or zero out minority views that exist. But a specific, well-defined "
    "subgroup often has a clear shared leaning — when one option clearly leads, "
    "give it a clear plurality rather than hedging. Weigh the group's social, "
    "cultural, and economic circumstances."
)


def _cond_body_A(q, who, yc, opts):
    return (f"Consider a large, representative sample of {who}{yc}.\n\n{q}\n\nGive "
            "the group's answer distribution: put the most mass on the option this "
            "group most likely favors, concentrate it when they largely agree and "
            "spread it when they are genuinely divided, keeping minority views "
            f"where they truly exist.\n{_json_instruction(opts)}")


# conditioned branch B (v6): A + stronger consensus/commitment emphasis
_COND_SYS_B = (
    _COND_SYS_A + " When this group genuinely agrees, commit to a sharp "
    "distribution; hedging evenly across options when there is a real majority is a "
    "mistake."
)


# --- Round 4: v4 + light socio-cultural context anchor --------------------
_V7_SYS = (
    _V4_SYS + " When the group is specific, weigh their social, cultural, and "
    "economic circumstances in judging which option they favor."
)


CANDIDATES = [
    AdHoc("commit_v4", _V4_SYS, _v4_body),       # round-2 best (cached reference)
    AdHoc("commit_v7", _V7_SYS, _v4_body),       # v4 + context anchor
]
INCUMBENTS = ["anti_flattening", "contextualized"]


def raw_preds(strategy, records, workers=8):
    pred = ZeroShotPredictor(make_client(MODEL, max_retries=10, timeout=90),
                             name=strategy.name, strategy=strategy)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(pred.predict, records))


def diagnostics(records, preds, normalizers):
    g, p = [], []
    pH, tH, mode_ok = [], [], []
    for rec, pred in zip(records, preds):
        z = normalizers.get((rec.split, rec.dataset_name))
        s = simbench_score(pred, rec.human_answer, options=list(rec.options), normalizer=z)
        (g if rec.split == "grouped" else p).append(s)
        if rec.split == "grouped":
            pH.append(response_entropy(pred)); tH.append(response_entropy(rec.human_answer))
            mode_ok.append(max(pred, key=pred.get) == max(rec.human_answer, key=rec.human_answer.get))
    pH, tH = np.array(pH), np.array(tH)
    gm, glo, ghi = bootstrap_ci(g)
    cons = tH < 0.4
    return {
        "grouped": round(gm, 2), "g_lo": round(glo, 2), "g_hi": round(ghi, 2),
        "g_hw": round((ghi - glo) / 2, 2),
        "pop": round(float(np.mean(p)), 2) if p else None,
        "predH": round(float(pH.mean()), 2), "floor_p10": round(float(np.percentile(pH, 10)), 2),
        "slope": round(float(np.polyfit(tH, pH, 1)[0]), 2),
        "commit@consensus": round(float(np.mean(pH[cons] < 0.45)), 2) if cons.any() else None,
        "mode_acc": round(float(np.mean(mode_ok)), 2),
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "round"
    t0 = time.time()
    allr = load_all(); full = allr["grouped"] + allr["pop"]
    split = make_split(full, seed=0, unit="question")
    normalizers = build_normalizers(split.dev + split.val + split.test)
    dev_g = [r for r in split.dev if r.split == "grouped"]
    dev_p = [r for r in split.dev if r.split == "pop"]
    eval_recs = stratified_sample(dev_g, N_G, seed=EVAL_SEED) + \
                stratified_sample(dev_p, N_P, seed=EVAL_SEED)
    print(f"[commit:{tag}] eval n={len(eval_recs)} (g={N_G},p={N_P})")

    strategies = [get_strategy(n) for n in INCUMBENTS] + CANDIDATES
    rows = []
    for strat in strategies:
        preds = raw_preds(strat, eval_recs)
        d = diagnostics(eval_recs, preds, normalizers); d["system"] = strat.name
        rows.append(d)
        print(f"  {strat.name:16s} grouped={d['grouped']:5.1f} [{d['g_lo']},{d['g_hi']}] "
              f"floor={d['floor_p10']} slope={d['slope']} commit@cons={d['commit@consensus']} "
              f"modeAcc={d['mode_acc']}")

    tbl = pd.DataFrame(rows).set_index("system")
    base = tbl.loc["anti_flattening", "grouped"]
    print(f"\nincumbent anti_flattening grouped={base}")
    for name in [s.name for s in CANDIDATES]:
        g = tbl.loc[name, "grouped"]
        floor = max(tbl.loc[name, "g_hw"], tbl.loc["anti_flattening", "g_hw"])
        v = "BEATS" if g - base > floor else ("ties" if g - base > -floor else "loses")
        print(f"  {name:16s} grouped={g:5.1f}  vs anti_flattening {g-base:+.2f}  ({v})")

    tbl.to_csv(OUTPUTS / f"2026-06-21-calibrated-commitment.{tag}.csv")
    print(f"\n[commit:{tag}] {time.time()-t0:.0f}s; saved .{tag}.csv")
    print(tbl[["grouped", "g_lo", "g_hi", "pop", "predH", "floor_p10", "slope",
               "commit@consensus", "mode_acc"]].to_string())


if __name__ == "__main__":
    main()
