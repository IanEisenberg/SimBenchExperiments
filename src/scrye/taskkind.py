"""Task-kind classification — the upfront classifier that contextualizes the
prompt strategy (Stage 15).

Across Stages 11–14 we learned that the *best intervention depends on the kind
of task*, not the dataset identity:

  * opinion/attitude surveys & knowledge  → task-context (sibling items help)
  * risky monetary choice                 → discrete-vote simulation
  * moral dilemmas                        → abstain (model is wrong-mode)
  * personality self-trait scales         → base calibrated_commitment
  * other                                 → base calibrated_commitment

This module classifies a record into one of those kinds from the *question
content* (so it generalizes to unseen datasets), and :data:`KIND_ROUTES` maps
each kind to the intervention name a :class:`~scrye.predict.RoutingPredictor`
dispatches to. Two classifiers are provided: an :class:`LLMTaskClassifier`
(robust, cached per unique question stem) and :func:`heuristic_task_kind` (a
deterministic content-pattern fallback, also the test oracle).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .data import SimBenchRecord

#: The task kinds. Stable strings — keys of KIND_ROUTES and classifier outputs.
KINDS: tuple[str, ...] = (
    "opinion_survey",
    "knowledge",
    "risky_choice",
    "moral_dilemma",
    "personality_scale",
    "other",
)

#: Decision rule: task kind -> intervention name. The RoutingPredictor maps each
#: name to a concrete Predictor. Derived from the Stage 11–14 dev evidence by
#: *mechanism*, validated on dev in Stage 15, and val-confirmed in Stage 16.
#:
#: Stage 16 (val): task_context transferred cleanly (surveys +1.57, knowledge
#: +5.53), but the dev-only `risky_choice -> voting` route FAILED to transfer
#: (val Choices13k: voting −3.0 vs uniform/abstain +11.1). Voting was dropped;
#: risky_choice now routes to `abstain` (the champion's val-safe behavior).
KIND_ROUTES: dict[str, str] = {
    "opinion_survey": "task_context",
    "knowledge": "task_context",
    "risky_choice": "abstain",        # was "voting" — dev-overfit, dropped Stage 16
    "moral_dilemma": "abstain",
    "personality_scale": "base",
    "other": "base",
}


# --------------------------------------------------------------------------- #
# Heuristic classifier (deterministic; fallback + test oracle)
# --------------------------------------------------------------------------- #

_ORDINAL_HINTS = ("disagree", "agree", "very good", "very bad", "strongly",
                  "somewhat", "approve", "oppose")


def heuristic_task_kind(record: SimBenchRecord) -> str:
    """Label a record's task kind from surface content patterns (deterministic)."""
    t = record.input_template.lower()
    g = (record.group_prompt or "").lower()
    opts = " ".join(str(o).lower() for o in record.options)

    # risky monetary choice: lotteries/gambles with stated probabilities
    if (("gambling machine" in t or "lottery" in t or "gamble" in t)
            or ("$" in record.input_template and re.search(r"\d+(\.\d+)?%|chance", t))):
        return "risky_choice"
    # moral dilemma
    if any(w in t for w in ("moral dilemma", "self-driving", "brake failure",
                            "swerve", "pedestrian", "moral machine")):
        return "moral_dilemma"
    # introspective personality self-assessment
    if "indicate your level of agreement" in t or "openpsychometrics" in g:
        return "personality_scale"
    # factual / general-knowledge judgments
    if "number game" in t or "wisdom of crowds" in g or "general knowledge" in t:
        return "knowledge"
    # opinion / attitude survey: ordinal attitude options or a survey context
    if any(h in opts for h in _ORDINAL_HINTS) or any(
            s in g for s in ("survey", "european social survey", "afrobarometer",
                             "latinobarómetro", "latinobarometro", "issp", "opinion")):
        return "opinion_survey"
    return "other"


# --------------------------------------------------------------------------- #
# LLM classifier (primary; one cached call per unique question stem)
# --------------------------------------------------------------------------- #

_SYSTEM = (
    "You label survey and behavioral-science items by task kind for a research "
    "pipeline. Answer with exactly one label and nothing else."
)
_TAXONOMY = (
    "- opinion_survey: an item from a coherent multi-question survey or "
    "questionnaire — attitudes, opinions, OR factual/behavioral/biographical "
    "questions a survey asks its respondents (e.g. national social surveys, "
    "opinion polls). Seeing the survey's other questions would help interpret "
    "this one.\n"
    "- knowledge: a factual or general-knowledge question with correct answers, "
    "or a numeric generalization judgment, from a test/quiz instrument\n"
    "- risky_choice: a choice between monetary gambles or lotteries with stated "
    "probabilities and payoffs\n"
    "- moral_dilemma: a moral dilemma choosing whom or what to sacrifice, or "
    "which harmful outcome is more acceptable\n"
    "- personality_scale: an introspective self-assessment of one's own "
    "personality, traits, or values (e.g. agree/disagree with a self-statement)\n"
    "- other: a self-contained stimulus that stands alone — its meaning does not "
    "depend on other items (e.g. rating one joke, a single sentence-pair "
    "judgment)"
)


def _classify_messages(stem: str, context: str = "") -> list[dict]:
    ctx = f"Respondent context: {context}\n\n" if context else ""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content":
            "Classify this item into exactly one task kind:\n\n"
            f"{_TAXONOMY}\n\n{ctx}Item:\n{stem}\n\nKind:"},
    ]


class LLMTaskClassifier:
    """Classify records into task kinds with one cached LLM call per unique stem.

    Deterministic (``temperature=0``) and memoized in-process by question stem,
    on top of the on-disk LLM cache — so a full dev/val pass costs at most one
    call per distinct question, and re-runs are free.
    """

    def __init__(self, client, fallback: Callable[[SimBenchRecord], str] = heuristic_task_kind):
        self.client = client
        self.fallback = fallback
        self._memo: dict[str, str] = {}

    def __call__(self, record: SimBenchRecord) -> str:
        return self.classify(record)

    def classify(self, record: SimBenchRecord) -> str:
        stem = record.input_template.strip()
        if stem in self._memo:
            return self._memo[stem]
        context = (record.group_prompt or "").strip()
        text = self.client.complete(_classify_messages(stem, context),
                                    temperature=0, max_tokens=8).text.lower()
        kind = next((k for k in KINDS if k in text), None)
        if kind is None:  # model returned something off-taxonomy
            kind = self.fallback(record)
        self._memo[stem] = kind
        return kind
