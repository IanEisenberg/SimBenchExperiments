"""Persona factory — demographic-conditioning prompt strategies.

Turns a :class:`~simbench_exp.data.SimBenchRecord`'s structured segment (country and/or
demographic attributes) into chat messages that condition an LLM on that group.

The benchmark ships a first-person `group_prompt` ("You are from Finland. Your
gender is female."). SimBench's headline finding — that conditioning *degrades*
group-level scores — was measured on exactly that first-person framing. So this
module exposes a **registry of swappable styles** rather than a single prompt:

  * ``simbench_faithful``      — reproduces the paper's first-person persona
                                 verbatim (the control / current baseline).
  * ``representative_sample``  — third-person distributional framing.
  * ``persona_embodiment``     — a strong, immersive first-person persona.
  * ``anti_flattening``        — distributional + preserve within-group diversity.
  * ``contextualized``         — distributional + socio-cultural context anchor.

Which styles help vs. hurt is an empirical question we measure, not assume.

The :data:`VARIABLE_DICTIONARY` verbalizes each grouping variable into a
*third-person* descriptor fragment, with a generic fallback for unknown keys, so
the factory never crashes on an unfamiliar variable. :data:`COUNTRIES` is the
full per-dataset country catalog from the SimBench grouped split.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .data import SimBenchRecord

# --------------------------------------------------------------------------- #
# Country catalog (full coverage from the SimBench grouped split)
# --------------------------------------------------------------------------- #

COUNTRIES: dict[str, tuple[str, ...]] = {
    "OpinionQA": ("the United States",),
    "ESS": (
        "Austria", "Belgium", "Bulgaria", "Cyprus", "Czechia", "Denmark",
        "Estonia", "Finland", "France", "Germany", "Hungary", "Iceland",
        "Ireland", "Israel", "Italy", "Lithuania", "Montenegro", "Norway",
        "Poland", "Portugal", "Russia", "Serbia", "Slovenia", "Spain",
        "Sweden", "Switzerland", "the Netherlands", "the United Kingdom",
    ),
    "Afrobarometer": (
        "Angola", "Benin", "Botswana", "Burkina Faso", "Cabo Verde",
        "Cameroon", "Congo-Brazzaville", "Côte d'Ivoire", "Eswatini",
        "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Kenya", "Lesotho",
        "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
        "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Senegal",
        "Seychelles", "Sierra Leone", "South Africa", "Sudan",
        "São Tomé and Príncipe", "Tanzania", "Togo", "Tunisia", "Uganda",
        "Zambia", "Zimbabwe",
    ),
    "ISSP": (
        "Australia", "Austria", "Bulgaria", "China", "Croatia",
        "Czech Republic", "Denmark", "Finland", "France", "Georgia",
        "Germany", "Great Britain", "Hungary", "India", "Israel", "Italy",
        "Japan", "Lithuania", "Mexico", "New Zealand", "Norway",
        "Philippines", "Russia", "Slovakia", "Slovenia", "South Korea",
        "Spain", "Suriname", "Sweden", "Switzerland", "Taiwan", "Thailand",
        "Turkey", "United States", "Venezuela",
    ),
    "LatinoBarometro": (
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica",
        "Dominican Republic", "Ecuador", "El Salvador", "Guatemala",
        "Honduras", "Mexico", "Panama", "Paraguay", "Peru", "Uruguay",
        "Venezuela",
    ),
}

#: All distinct countries across datasets, sorted.
ALL_COUNTRIES: tuple[str, ...] = tuple(
    sorted({c for cs in COUNTRIES.values() for c in cs})
)


# --------------------------------------------------------------------------- #
# Variable dictionary: segment key -> third-person verbalization
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VariableSpec:
    """How to verbalize one grouping variable in the third person.

    `template` is a sentence fragment with a ``{value}`` placeholder that reads
    naturally after "a representative sample of people ...", e.g. "from Finland"
    or "in the 30-49 age group". `order` controls fragment ordering (country and
    age lead). `label` is a short human concept name used in tables.
    """

    label: str
    template: str
    order: int = 5


#: Keys whose values are 0-N numeric scale positions (verbalized specially).
VARIABLE_DICTIONARY: dict[str, VariableSpec] = {
    # Geography
    "country": VariableSpec("country", "from {value}", 0),
    "cntry": VariableSpec("country", "from {value}", 0),
    "CREGION": VariableSpec("region", "from the {value} region", 1),
    # Age
    "age_group": VariableSpec("age", "in the {value} age group", 2),
    "age": VariableSpec("age", "in the {value} age group", 2),
    "AGE": VariableSpec("age", "in the {value} age group", 2),
    # Sex / gender
    "gender": VariableSpec("gender", "who are {value}", 3),
    "gndr": VariableSpec("gender", "who are {value}", 3),
    "SEX": VariableSpec("gender", "who are {value}", 3),
    "RACE": VariableSpec("race", "whose race is {value}", 4),
    # Education
    "education": VariableSpec("education", "whose education level is {value}"),
    "EDUCATION": VariableSpec("education", "whose highest level of education is {value}"),
    "eisced": VariableSpec("education", "whose education level is {value}"),
    "highest_education": VariableSpec("education", "with {value}"),
    # Income / economic standing
    "INCOME": VariableSpec("income", "with an annual income of {value}"),
    "income": VariableSpec("income", "whose income level is {value}"),
    "household_income": VariableSpec("income", "whose household finances are such that {value}"),
    "subjective_income": VariableSpec("income strain", "who report their income falls short {value}"),
    "topbot": VariableSpec("social standing", "who place themselves at {value} on a 0-10 social-standing scale"),
    # Marital status
    "MARITAL": VariableSpec("marital status", "whose marital status is {value}"),
    "marital_status": VariableSpec("marital status", "whose marital status is {value}"),
    "maritalb": VariableSpec("marital status", "whose marital status is {value}"),
    # Religion
    "religion": VariableSpec("religion", "who identify religiously as {value}"),
    "RELIG": VariableSpec("religion", "who identify religiously as {value}"),
    "religiosity": VariableSpec("religiosity", "who are {value}"),
    "rlgdgr": VariableSpec("religiosity", "whose religiosity is {value} on a 0-10 scale"),
    "RELIGATTEND": VariableSpec("religious attendance", "who attend religious services {value}"),
    # Politics
    "POLIDEOLOGY": VariableSpec("political ideology", "who consider themselves {value}"),
    "POLPARTY": VariableSpec("political party", "who identify politically as {value}"),
    "political_group": VariableSpec("political group", "in political group {value}"),
    "lrscale": VariableSpec("political orientation", "who place themselves at {value} on a 0-10 left-right scale"),
    "discuss_politics": VariableSpec("political engagement", "who discuss politics {value}"),
    # Employment
    "employment": VariableSpec("employment", "whose work status is {value}"),
    "employment_status": VariableSpec("employment", "who are {value}"),
    "work_status": VariableSpec("employment", "whose work status is {value}"),
    "mnactic": VariableSpec("employment", "whose main activity is {value}"),
    # Locality
    "domicil": VariableSpec("locality", "who live in {value}"),
    "urban_rural": VariableSpec("locality", "from a {value} area"),
    "city_size": VariableSpec("city size", "who live somewhere that {value}"),
}


def _humanize(key: str) -> str:
    return key.replace("_", " ").strip().lower()


def _fallback_spec(key: str) -> VariableSpec:
    return VariableSpec(_humanize(key), "whose " + _humanize(key) + " is {value}", 9)


def _clean_value(value) -> str:
    """Tidy a segment value for prose (drop trailing ``.0`` on scale codes)."""
    s = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def verbalize_segment(
    segment: dict[str, str],
    locale: str | None = None,
) -> str:
    """Render a segment as a third-person descriptor noun phrase.

    Example: ``{"cntry": "Finland", "age_group": "30-49"}`` ->
    "from Finland, in the 30-49 age group". `locale` supplies an implicit country
    (e.g. OpinionQA's US) when the segment carries none. Returns "" for the
    unconditioned population.
    """
    fragments: list[tuple[int, str]] = []
    for key, value in segment.items():
        spec = VARIABLE_DICTIONARY.get(key) or _fallback_spec(key)
        fragments.append((spec.order, spec.template.format(value=_clean_value(value))))
    has_country = any(k in segment for k in ("country", "cntry"))
    if locale and not has_country:
        fragments.append((0, f"from {locale}"))
    fragments.sort(key=lambda t: t[0])
    return ", ".join(frag for _, frag in fragments)


# --------------------------------------------------------------------------- #
# Context extraction
# --------------------------------------------------------------------------- #


def extract_year(group_prompt: str) -> str | None:
    """Pull "The year is YYYY" out of a rendered group prompt, if present."""
    m = re.search(r"[Tt]he year is (\d{4})", group_prompt or "")
    return m.group(1) if m else None


def extract_locale(record: SimBenchRecord) -> str | None:
    """Best-effort country/locale: from the segment, else the group prompt."""
    for key in ("country", "cntry"):
        if record.segment.get(key):
            return record.segment[key]
    m = re.search(r"[Yy]ou are from ([^.]+)\.", record.group_prompt or "")
    return m.group(1).strip() if m else None


def _population_phrase(record: SimBenchRecord) -> tuple[str, str]:
    """Return ("people <descriptor>", " as of YYYY") for a record."""
    locale = extract_locale(record)
    descriptor = verbalize_segment(record.segment, locale=locale)
    who = f"people {descriptor}" if descriptor else "people"
    year = extract_year(record.group_prompt)
    year_clause = f" as of {year}" if year else ""
    return who, year_clause


def _json_instruction(options) -> str:
    opts = ", ".join(options)
    return (
        f"Respond with ONLY a JSON object mapping each option label [{opts}] to "
        "its probability (between 0 and 1). The probabilities must sum to 1. "
        "No other text."
    )


# --------------------------------------------------------------------------- #
# Prompt strategies
# --------------------------------------------------------------------------- #


class PromptStrategy:
    """Builds chat messages that condition the model on a record's segment.

    Subclasses set `name` and implement :meth:`build_messages`, returning a list
    of ``{"role": ..., "content": ...}`` dicts ready for ``LLMClient.complete``.
    """

    name: str = "strategy"

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        raise NotImplementedError


def faithful_prompt(record: SimBenchRecord) -> str:
    """The exact prompt string used by the original zero-shot baseline.

    Kept as a standalone function so the baseline is byte-stable (cache-safe)
    and testable. First-person persona block + verbalized-distribution ask.
    """
    opts = ", ".join(record.options)
    persona = record.group_prompt.strip()
    persona_block = f"{persona}\n\n" if persona else ""
    return (
        f"{persona_block}{record.input_template.strip()}\n\n"
        "Estimate how a large, representative sample of such people would "
        "answer. Give the probability (between 0 and 1) that a randomly "
        "sampled person chooses each option.\n"
        f"Respond with ONLY a JSON object mapping each option label "
        f"[{opts}] to its probability. The probabilities must sum to 1. "
        "No other text."
    )


class FaithfulStrategy(PromptStrategy):
    """Reproduces the SimBench first-person persona baseline exactly."""

    name = "simbench_faithful"

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        return [{"role": "user", "content": faithful_prompt(record)}]


class RepresentativeSampleStrategy(PromptStrategy):
    """Third-person distributional framing — reason over the whole group."""

    name = "representative_sample"
    SYSTEM = (
        "You are an expert survey methodologist. You estimate how an entire "
        "population subgroup answers survey questions by reasoning about the "
        "full distribution of views within the group, never collapsing it to a "
        "single 'typical' answer."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Estimate how this whole group answers — the probability that a "
            "randomly sampled member chooses each option.\n"
            f"{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class PersonaEmbodimentStrategy(PromptStrategy):
    """A strong first-person persona — inhabit the group, still distributional.

    The serious direct-persona contender: richer immersion than the faithful
    baseline, grounded in lived experience rather than stereotype.
    """

    name = "persona_embodiment"
    SYSTEM = (
        "You are simulating members of a specific demographic group for survey "
        "research. Inhabit their perspective fully and authentically, grounded "
        "in the real, varied lived experience of that group rather than "
        "stereotypes."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        persona = record.group_prompt.strip()
        persona_block = f"{persona}\n\n" if persona else ""
        user = (
            f"{persona_block}Thinking as people in this group genuinely would, "
            "consider the question below.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Estimate how a large, representative sample of such people would "
            "answer.\n"
            f"{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class AntiFlatteningStrategy(PromptStrategy):
    """Distributional framing that explicitly preserves within-group diversity.

    Counters the documented failure mode where LLMs flatten a group to a single
    stereotyped response and drop mass on minority views that really exist.
    """

    name = "anti_flattening"
    SYSTEM = (
        "You are an expert survey methodologist. Real demographic groups contain "
        "wide internal disagreement. Reproduce that genuine heterogeneity: never "
        "flatten a group to one stereotyped response, and keep probability mass "
        "on the minority views that really exist within it."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}. "
            "This group holds a real diversity of views.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Estimate the full distribution of their answers, preserving the "
            "real internal spread of opinion in this group.\n"
            f"{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class ContextualizedStrategy(PromptStrategy):
    """Distributional framing with a socio-cultural context anchor."""

    name = "contextualized"
    SYSTEM = RepresentativeSampleStrategy.SYSTEM

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Survey context: respondents are {who}, surveyed{year_clause}. "
            "Consider the social, cultural, and economic circumstances of this "
            "group when reasoning about their views.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Estimate how this whole group answers.\n"
            f"{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class CalibratedCommitmentStrategy(PromptStrategy):
    """Mode-first distributional framing — the Stage 10 prompt search winner.

    Combines the four design goals in one direct-answer prompt: (1) diversity of
    perspectives and (2) no stereotyping (keep minority mass), (3) consensus/entropy
    awareness (concentrate when the group agrees, spread when divided), and (4)
    *licensed commitment* — when one option clearly leads, give it a clear plurality
    rather than hedging evenly. Its real lever is **mode/location** (get the leading
    answer right), which the error decomposition (notebook 04) and the calibration
    stages (08-09) identified as the binding constraint. On dev it improves both the
    grouped (+1.8) and pop (+2.1) splits over ``anti_flattening`` and lifts mode
    accuracy, though the grouped gain is within the dev noise floor — a
    val-confirmation candidate, not yet a confirmed replacement.
    """

    name = "calibrated_commitment"
    SYSTEM = (
        "You are an expert survey methodologist. Estimate how a group answers in "
        "two respects at once: (1) which option is most common for this group — get "
        "the leading answer right — and (2) how concentrated or divided the group "
        "truly is around it. Real groups are diverse, so never zero out minority "
        "views that exist or reduce the group to a stereotype; but when one option "
        "clearly leads, give it a clear plurality rather than hedging evenly across "
        "options. A broad population is usually more split than a specific subgroup."
    )

    def __init__(self, audience: str | None = None) -> None:
        """`audience` overrides the population descriptor with a free-text noun
        phrase (e.g. "US tech workers in 2025"), used by the ``simbench-ask`` CLI to
        condition on an arbitrary group. ``None`` (the default, and the registered
        instance) keeps the segment-derived phrasing, so the prompt stays
        byte-identical to the validated strategy."""
        self.audience = audience

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        if self.audience:
            who, year_clause = self.audience, ""
        else:
            who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Give the group's answer distribution: put the most mass on the option "
            "this group most likely favors, concentrate it when they largely agree "
            "and spread it when they are divided, and keep minority views where they "
            f"genuinely exist.\n{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class DiversityElicitationStrategy(PromptStrategy):
    """Single-call framing that makes the model *reason about the range of views*
    before committing to a distribution.

    A cheap cousin of Monte-Carlo simulation: rather than externally aggregating
    sampled individuals, it elicits the within-group spread inside one call by
    making the model enumerate which subgroups lean which way first.
    """

    name = "diversity_elicitation"
    SYSTEM = (
        "You are an expert survey methodologist. Before estimating a group's "
        "answer distribution, you explicitly reason about the full range of "
        "views inside the group — which subgroups hold which positions and how "
        "common each is — and never collapse the group to one typical answer."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}.\n\n"
            f"{record.input_template.strip()}\n\n"
            "First, think step by step about the RANGE of views within this "
            "group: which subgroups lean toward which options, and roughly how "
            "common each view is. Then give the group's overall answer "
            "distribution.\n"
            f"{_json_instruction(record.options)} "
            "Write your reasoning first, then the JSON object LAST."
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class OutsideViewStrategy(PromptStrategy):
    """Superforecaster move: outside view first, then a modest adjustment.

    Step 1 estimates the general-population base rate (which the model tends to
    represent well); step 2 shifts it for the specific group only as much as the
    demographic justifies. Isolates base-rate anchoring — the antidote to
    SimBench's finding that demographic conditioning *degrades* group scores.
    """

    name = "outside_view"
    SYSTEM = (
        "You are an expert forecaster of survey responses. You always start from "
        "the base rate — how the general population answers — and adjust only as "
        "much as the specific group genuinely justifies, never over-reacting to a "
        "group label."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Forecast in two steps.\n"
            "1. OUTSIDE VIEW: Ignoring this group's specific characteristics, "
            "estimate how the general population answers this question — the base "
            "rate.\n"
            "2. ADJUST: Now adjust for this specific group — which way, and how "
            "much, do their characteristics shift each option? Make only the "
            "adjustments the group genuinely justifies; keep shifts modest unless "
            "there is a strong reason.\n"
            "Then give the group's final answer distribution.\n"
            f"{_json_instruction(record.options)} "
            "Write your reasoning first, then the JSON object LAST."
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class EntropyFirstStrategy(PromptStrategy):
    """Superforecaster move: commit to the spread before the distribution.

    Step 1 forces an explicit 1–5 "how divided is this group" rating; step 2
    requires a distribution whose sharpness matches it. Isolates the
    spread-before-numbers idea, directly targeting over-sharpening on the
    high-entropy items SimBench shows LLMs handle worst (r = −0.942).
    """

    name = "entropy_first"
    SYSTEM = (
        "You are an expert forecaster of survey responses. You first judge how "
        "divided a group is on a question, then give a distribution whose "
        "sharpness matches that judgment — you never output a confident peak on a "
        "question you judged contested."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Forecast in two steps.\n"
            "1. SPREAD: First rate how divided this group is on this question, on "
            "a 1-5 scale: 1 = near-unanimous (one option dominates), 3 = leaning "
            "but contested, 5 = evenly split across options. State the number.\n"
            "2. DISTRIBUTION: Now give a distribution whose spread MATCHES your "
            "rating — a low rating must be peaked, a high rating must be spread "
            "out. Do not output a confident peak if you rated the question "
            "contested.\n"
            f"{_json_instruction(record.options)} "
            "Write your reasoning first, then the JSON object LAST."
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class SuperforecasterStrategy(PromptStrategy):
    """The full superforecaster pipeline: outside view → spread → adjust →
    premortem → distribution.

    Combines :class:`OutsideViewStrategy` (base-rate anchoring) and
    :class:`EntropyFirstStrategy` (spread calibration) with a premortem step that
    guards minority mass (counters flattening). The kitchen-sink contender — if
    structured forecasting helps this model at all, this should show it.
    """

    name = "superforecaster"
    SYSTEM = (
        "You are a superforecaster estimating how a population subgroup answers a "
        "survey. You reason in steps — outside view first, calibrate the spread, "
        "adjust for the group, and check what you might be under-weighting — then "
        "commit to numbers."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Consider a large, representative sample of {who}{year_clause}.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Forecast like a superforecaster, in four steps.\n"
            "1. OUTSIDE VIEW: Estimate the base rate — how the general population "
            "answers, ignoring this group's specifics.\n"
            "2. SPREAD: Rate how divided THIS group is on a 1-5 scale (1 = "
            "near-unanimous, 5 = evenly split), given its real internal "
            "diversity.\n"
            "3. ADJUST: Shift the base rate for this group, only as much as its "
            "characteristics justify.\n"
            "4. PREMORTEM: Which option are you most likely UNDER-weighting? Keep "
            "real probability mass on plausible minority views.\n"
            "Then give the group's final distribution, consistent with all four "
            "steps and with the spread you rated.\n"
            f"{_json_instruction(record.options)} "
            "Write your reasoning first, then the JSON object LAST."
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


class IndividualStrategy(PromptStrategy):
    """Frame ONE specific, randomly drawn individual from the group.

    Used by :class:`~simbench_exp.predict.MonteCarloPredictor`: called many times with
    different sampling seeds, each draw imagines a different concrete person, so
    averaging the draws reconstructs the group's spread from the bottom up rather
    than asking one call to self-report its diversity. Not registered as a
    standalone simulation system — a single individual is a poor group estimate;
    it is only meaningful inside the Monte-Carlo average.
    """

    name = "individual"
    SYSTEM = (
        "You simulate one specific, randomly sampled member of a demographic "
        "group for survey research. Inhabit the particular person described — "
        "their stated leanings and background — and answer as that one person, "
        "not as the group average."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        # Base (no injected sub-persona); MonteCarloPredictor uses the sampled
        # variant via `individual_messages` instead.
        return individual_messages(record, "is a typical member of this group")


# --------------------------------------------------------------------------- #
# Synthetic within-group individuals (for Monte-Carlo simulation)
# --------------------------------------------------------------------------- #
#
# The model is near-deterministic across sampling seeds, so temperature alone
# produces identical draws — useless for Monte-Carlo. Instead we inject the
# variation ourselves: each draw samples a synthetic individual (a worldview
# lean plus unspecified demographic attributes) from these pools with a seeded
# RNG, so draws genuinely differ yet stay reproducible.

IDEOLOGY_AXES: tuple[str, ...] = (
    "is strongly economically left-wing",
    "is strongly economically right-wing",
    "is a political moderate",
    "leans libertarian",
    "is socially conservative and traditionalist",
    "is socially progressive",
    "deeply distrusts government and institutions",
    "broadly trusts government and institutions",
    "favors free markets and low taxes",
    "favors public services and redistribution",
    "is politically disengaged and undecided",
    "holds populist, anti-establishment views",
)

#: concept -> pool of clauses that read after "they ...".
DEMOGRAPHIC_AXES: dict[str, tuple[str, ...]] = {
    "age": ("are a young adult", "are middle-aged", "are older or retired"),
    "education": ("have little formal schooling", "finished secondary school",
                  "have a university degree"),
    "locality": ("live in a major city", "live in a small town",
                 "live in a rural area"),
    "income": ("are on a low income", "are on a comfortable middle income",
               "are on a high income"),
}

#: a sampled axis is suppressed when the segment already pins that concept
#: (matched against :data:`VARIABLE_DICTIONARY` labels), so we never contradict
#: the group definition.
_PINNED_ALIASES: dict[str, set[str]] = {
    "age": {"age"},
    "education": {"education"},
    "locality": {"locality", "city size", "region"},
    "income": {"income", "income strain", "social standing"},
}


def sample_persona(record: SimBenchRecord, seed: int) -> str:
    """Sample one synthetic within-group individual as a descriptor clause.

    Draws a worldview lean plus the demographic attributes the segment leaves
    unspecified, from a seeded RNG (so the same seed reproduces the same person).
    Returns a clause that reads after "... who", e.g.
    "is socially progressive; they are middle-aged, have a university degree".
    """
    rng = random.Random(seed)
    pinned = {
        (VARIABLE_DICTIONARY.get(k) or _fallback_spec(k)).label
        for k in record.segment
    }
    ideology = rng.choice(IDEOLOGY_AXES)
    demos = [
        rng.choice(pool)
        for concept, pool in DEMOGRAPHIC_AXES.items()
        if not (pinned & _PINNED_ALIASES[concept])
    ]
    return f"{ideology}; they {', '.join(demos)}" if demos else ideology


def individual_messages(record: SimBenchRecord, persona: str) -> list[dict]:
    """Chat messages asking the model to answer as one described individual."""
    who, year_clause = _population_phrase(record)
    user = (
        f"Imagine ONE specific person from {who}{year_clause} who {persona}.\n\n"
        f"{record.input_template.strip()}\n\n"
        "Answer as that single person: give the probability that THIS person "
        "chooses each option (it is fine to be fairly decided).\n"
        f"{_json_instruction(record.options)}"
    )
    return [{"role": "system", "content": IndividualStrategy.SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Discrete-vote simulation (for VotingEnsemblePredictor)
# --------------------------------------------------------------------------- #
#
# MonteCarloPredictor asks each synthetic individual for a *distribution* and
# averages — which double-blurs (a hedging individual stays hedged) and, with
# uniform ideology weights, over-disperses (Stage 03, decisively refuted). The
# voting ensemble instead asks each individual to commit to ONE option; the
# group distribution is the *tally* of votes, so spread is fully endogenous —
# it appears only when sampled people genuinely disagree.
#
# The variation axes are deliberately TASK-AGNOSTIC dispositions (risk
# attitude, moral lean, temperament, values) rather than the political/
# demographic axes above, which are irrelevant to gambles, moral dilemmas, and
# psychometric items. They induce realistic disagreement on judgment/choice
# tasks without encoding any single dataset's peculiarities.

VOTER_SYSTEM = (
    "You simulate one specific, randomly drawn member of the public taking part "
    "in a study. Real people differ enormously in their values, risk tolerance, "
    "moral intuitions, and temperament. Fully inhabit the particular person "
    "described and decide as they would — not as an average or a hedge. Commit "
    "to the single option this one person would choose."
)

#: Task-agnostic dispositional clauses; each reads after "who ...".
DISPOSITION_AXES: tuple[str, ...] = (
    "is cautious and strongly risk-averse",
    "is a bold risk-taker who is comfortable with uncertainty",
    "is moderately risk-tolerant and pragmatic",
    "decides quickly on gut instinct and first impressions",
    "deliberates slowly and weighs the analytical details",
    "weighs outcomes and the greater good above all else",
    "holds firm moral rules and duties regardless of consequences",
    "prioritizes loyalty to their own family and community",
    "is highly empathetic and puts others' feelings first",
    "is self-reliant, competitive, and individualistic",
    "is conventional, traditional, and wary of change",
    "is unconventional, curious, and open to new experiences",
    "is trusting and tends to give people the benefit of the doubt",
    "is skeptical, cynical, and questions stated motives",
    "is agreeable and inclined to go along with the group",
    "is contrarian and comfortable holding a minority view",
    "is optimistic and expects things to work out",
    "is anxious and tends to expect the worst",
)


def sample_disposition(record: SimBenchRecord, seed: int) -> str:
    """Sample one synthetic individual's disposition as a descriptor clause.

    Draws two distinct task-agnostic dispositional traits from
    :data:`DISPOSITION_AXES` with a seeded RNG (same seed → same person).
    Reproducible and model-independent, so the ensemble's diversity comes from
    us, not the (near-deterministic) sampler.
    """
    rng = random.Random(seed)
    a, b = rng.sample(DISPOSITION_AXES, 2)
    return f"{a}, and {b}"


def voter_messages(record: SimBenchRecord, disposition: str) -> list[dict]:
    """Chat messages asking the simulated person to cast ONE discrete vote."""
    who, year_clause = _population_phrase(record)
    from_clause = f" from {who}{year_clause}" if who and who != "people" else ""
    labels = ", ".join(str(o) for o in record.options)
    user = (
        f"Imagine ONE specific person{from_clause} who {disposition}.\n\n"
        f"{record.input_template.strip()}\n\n"
        "Decide which single option this one person would choose. Do not hedge "
        "or give probabilities — make the one choice this person commits to.\n"
        f'Respond with only JSON: {{"choice": "<one of: {labels}>"}}.'
    )
    return [{"role": "system", "content": VOTER_SYSTEM},
            {"role": "user", "content": user}]


def persona_voter_messages(record: SimBenchRecord, persona_text: str) -> list[dict]:
    """Discrete-vote messages conditioned on a full Nemotron persona narrative.

    The Stage-13 voter ask, but the simulated person is a real census-grounded
    individual (their multi-paragraph life narrative) rather than a synthetic
    disposition clause. The persona *replaces* the group prompt: the demographic
    grounding lives in the description, so we do not restate the segment.
    """
    _, year_clause = _population_phrase(record)
    labels = ", ".join(str(o) for o in record.options)
    user = (
        "Here is a description of one specific person:\n\n"
        f"{persona_text.strip()}\n\n"
        f"This person is one respondent in a survey{year_clause}.\n\n"
        f"{record.input_template.strip()}\n\n"
        "Decide which single option THIS person would most likely choose, based on "
        "who they are. Do not hedge or give probabilities — make the one choice "
        "this person commits to.\n"
        f'Respond with only JSON: {{"choice": "<one of: {labels}>"}}.'
    )
    return [{"role": "system", "content": VOTER_SYSTEM},
            {"role": "user", "content": user}]


#: System prompt for the per-persona *distribution* elicitation (Stage 18 Round 2).
#: The collapse to a single number happens at the POPULATION level (we average
#: these distributions across the panel), so each person is allowed to carry
#: genuine within-person uncertainty — but only their *real* uncertainty, not the
#: model's ignorance about them (which, hedged toward uniform and averaged, would
#: over-disperse the group estimate, the Stage-03 failure).
PERSONA_DIST_SYSTEM = (
    "You simulate one specific, real person taking part in a study. Estimate how "
    "THIS person would answer. If who they are clearly points to one answer, put "
    "most of the probability on it — commit to your best reading of them. Only "
    "spread probability across options when this particular person would genuinely "
    "be of two minds. Do not flatten toward an even split merely because you are "
    "unsure about them; give your most informed read of this individual."
)


def persona_dist_messages(record: SimBenchRecord, persona_text: str) -> list[dict]:
    """Per-persona *distribution* messages conditioned on a Nemotron narrative.

    The Round-2 "soft voter": rather than forcing one option (which discards the
    person's internal uncertainty and over-concentrates the tally), ask for this
    person's own probability over the options. The population distribution is the
    average of these across the panel — the collapse to one number happens once,
    at the group level, not per individual.
    """
    _, year_clause = _population_phrase(record)
    user = (
        "Here is a description of one specific person:\n\n"
        f"{persona_text.strip()}\n\n"
        f"This person is one respondent in a survey{year_clause}.\n\n"
        f"{record.input_template.strip()}\n\n"
        "Estimate the probability that THIS specific person would choose each "
        "option, based on who they are. Be decided where their background clearly "
        "points one way; keep probability on more than one option only where this "
        "person would truly be torn.\n"
        f"{_json_instruction(record.options)}"
    )
    return [{"role": "system", "content": PERSONA_DIST_SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Task-context prompting (Stage 14)
# --------------------------------------------------------------------------- #
#
# SimBench presents every item *atomized* — one question + a one-line group
# prompt (paper §2.2, p23). But the original respondents answered inside a task
# context: a full psychometric instrument (MACH-IV is 20 items), a session of
# many gambles/dilemmas, a multi-topic survey. The construct that shapes the
# population distribution lives *across* the items. These strategies restore the
# context the respondents actually had — a faithful task brief and/or sibling
# items from the same instrument — while keeping the calibrated_commitment
# system prompt and distributional ask fixed, so the only thing that varies is
# context. No human answers are ever shown (sibling *questions* only), so this
# is a pure prompt method with no label leakage.

#: Faithful one-line task descriptions, drawn from the SimBench paper's
#: per-dataset appendix (the instrument's real purpose — context the
#: respondents had). Used by the `brief`/`both` variants.
TASK_BRIEFS: dict[str, str] = {
    "OSPsychMACH": "the MACH-IV, a 20-item personality scale measuring how much a "
    "person endorses Machiavellian views — that manipulation and self-interest "
    "can outweigh morality — answered on a 5-point Disagree–Agree scale",
    "OSPsychBig5": "the Big Five personality inventory; its items measure five "
    "broad personality traits and are answered on a Disagree–Agree scale",
    "OSPsychRWAS": "the Right-Wing Authoritarianism Scale; its items measure "
    "submission to established authority and adherence to convention, answered "
    "on a Disagree–Agree scale",
    "Choices13k": "a long series of choices between two monetary gambles, made "
    "for a real cash bonus, run to study how people decide under risk",
    "MoralMachine": "a series of self-driving-car moral dilemmas judged on the "
    "public Moral Machine website, choosing which outcome is more acceptable",
}


def build_item_corpus(records) -> dict[str, list[str]]:
    """Map each dataset to its list of unique item stems (insertion order).

    The sibling pool the `items`/`both` variants draw from. Questions only — no
    human answers — so it carries no label information.
    """
    corpus: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for r in records:
        ds = r.dataset_name
        stem = r.input_template.strip()
        pool = corpus.setdefault(ds, [])
        marks = seen.setdefault(ds, set())
        if stem not in marks:
            marks.add(stem)
            pool.append(stem)
    return corpus


def _question_stem(input_template: str, max_chars: int = 220) -> str:
    """A compact one-line rendering of an item for use as sibling context."""
    t = input_template.strip()
    idx = t.find("Options:")
    if idx == -1:
        idx = t.find("Options\n")
    if idx != -1:
        t = t[:idx]
    t = " ".join(t.split())
    return t[:max_chars] + ("…" if len(t) > max_chars else "")


class TaskContextStrategy(PromptStrategy):
    """calibrated_commitment + the task context the respondents actually had.

    ``mode``:
      * ``"brief"`` — prepend a faithful one-line description of the instrument
        (from :data:`TASK_BRIEFS`), if available for the dataset.
      * ``"items"`` — show ``k`` sibling items from the same instrument as
        context (your "include other items" idea), deterministically chosen.
      * ``"both"`` — brief and siblings.

    Constructed with an item ``corpus`` (from :func:`build_item_corpus`). The
    system prompt and the final distributional ask are copied verbatim from
    :class:`CalibratedCommitmentStrategy`, so any score change is attributable
    to the added context alone.
    """

    SYSTEM = CalibratedCommitmentStrategy.SYSTEM

    def __init__(
        self,
        corpus: dict[str, list[str]],
        mode: str = "items",
        briefs: dict[str, str] | None = None,
        k: int = 6,
        name: str | None = None,
    ) -> None:
        if mode not in ("brief", "items", "both"):
            raise ValueError(f"mode must be brief|items|both, got {mode!r}")
        self.corpus = corpus
        self.mode = mode
        self.briefs = TASK_BRIEFS if briefs is None else briefs
        self.k = k
        self.name = name or f"task_context_{mode}"

    def _siblings(self, record: SimBenchRecord) -> list[str]:
        target = record.input_template.strip()
        pool = [s for s in self.corpus.get(record.dataset_name, []) if s != target]
        # deterministic: stable sort, take the first k
        chosen = sorted(pool)[: self.k]
        return [_question_stem(s) for s in chosen]

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        parts = [f"Consider a large, representative sample of {who}{year_clause}."]

        if self.mode in ("brief", "both"):
            brief = self.briefs.get(record.dataset_name)
            if brief:
                parts.append(
                    f"Context: these people are taking part in {brief}. They see "
                    "the whole task, not just this one item."
                )
        if self.mode in ("items", "both"):
            sibs = self._siblings(record)
            if sibs:
                listed = "\n".join(f"  - {s}" for s in sibs)
                parts.append(
                    "This question is one item from that larger set; other items "
                    f"the same people answer include:\n{listed}"
                )

        parts.append(record.input_template.strip())
        parts.append(
            "Give the group's answer distribution: put the most mass on the option "
            "this group most likely favors, concentrate it when they largely agree "
            "and spread it when they are divided, and keep minority views where they "
            f"genuinely exist.\n{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)}]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

STRATEGIES: dict[str, PromptStrategy] = {
    s.name: s
    for s in (
        FaithfulStrategy(),
        RepresentativeSampleStrategy(),
        PersonaEmbodimentStrategy(),
        AntiFlatteningStrategy(),
        ContextualizedStrategy(),
        CalibratedCommitmentStrategy(),
        DiversityElicitationStrategy(),
        OutsideViewStrategy(),
        EntropyFirstStrategy(),
        SuperforecasterStrategy(),
    )
}

DEFAULT_STRATEGY = FaithfulStrategy.name


def get_strategy(name: str) -> PromptStrategy:
    """Look up a strategy by name; raise a helpful error if unknown."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy {name!r}; available: {sorted(STRATEGIES)}"
        ) from None
