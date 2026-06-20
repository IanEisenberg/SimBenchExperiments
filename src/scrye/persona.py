"""Persona factory — demographic-conditioning prompt strategies.

Turns a :class:`~scrye.data.SimBenchRecord`'s structured segment (country and/or
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


class IndividualStrategy(PromptStrategy):
    """Frame ONE specific, randomly drawn individual from the group.

    Used by :class:`~scrye.predict.MonteCarloPredictor`: called many times with
    different sampling seeds, each draw imagines a different concrete person, so
    averaging the draws reconstructs the group's spread from the bottom up rather
    than asking one call to self-report its diversity. Not registered as a
    standalone simulation system — a single individual is a poor group estimate;
    it is only meaningful inside the Monte-Carlo average.
    """

    name = "individual"
    SYSTEM = (
        "You simulate one specific, randomly sampled member of a demographic "
        "group for survey research. Each time, imagine a different concrete "
        "person — give them a particular, realistic background, life situation, "
        "and set of opinions a real individual in that group might hold — then "
        "answer as that one person, not as the group average."
    )

    def build_messages(self, record: SimBenchRecord) -> list[dict]:
        who, year_clause = _population_phrase(record)
        user = (
            f"Imagine ONE specific person randomly drawn from {who}{year_clause}. "
            "Picture their particular circumstances and views as a real "
            "individual.\n\n"
            f"{record.input_template.strip()}\n\n"
            "Answer as that single person: give the probability that THIS "
            "person chooses each option (it is fine to be fairly decided).\n"
            f"{_json_instruction(record.options)}"
        )
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]


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
        DiversityElicitationStrategy(),
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
