"""Nemotron-Personas-USA persona bank — a grounded representative electorate.

Loads a slim local cache of `nvidia/Nemotron-Personas-USA` (built by
``scripts/prepare_nemotron.py``) and exposes it as a sampleable population of US
adults. Two uses, both feeding :class:`~scrye.predict.GroundedVotingPredictor`:

  * **Whole-corpus panel** — a fixed, seeded, uniform sample of K personas. Because
    the corpus is census-aligned (ACS-grounded PGM), a uniform draw is a
    representative sample of the US adult population, so it answers an
    *unconditioned* (pop-split) question with no demographic prompt at all.
  * **Matched panel** — the same draw, filtered to a SimBench segment on the axes
    Nemotron carries structurally (age / sex / region / education / marital). A
    segment pinned on an axis Nemotron lacks (race / income / religion / politics)
    returns ``None`` so the predictor can fall back to a prompt method.

The axis maps below are built from the **observed** slim-cache vocabularies (see
``prepare_nemotron.py`` output), not guessed. Minors (~21% of v1.1 rows, despite
the card's "adults only") are dropped: survey respondents are adults.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd

from .config import DATA_DIR
from .data import SimBenchRecord

SLIM_PATH = DATA_DIR / "nemotron" / "personas_slim.parquet"
MIN_ADULT_AGE = 18

# --------------------------------------------------------------------------- #
# Axis maps: SimBench (OpinionQA) segment value -> Nemotron structured field.
# A key or value absent here is treated as UNMATCHABLE (-> fall back to a prompt).
# --------------------------------------------------------------------------- #

#: OpinionQA AGE bucket -> inclusive [lo, hi] over Nemotron integer `age`.
AGE_BINS: dict[str, tuple[int, int]] = {
    "18-29": (18, 29), "30-49": (30, 49), "50-64": (50, 64), "65+": (65, 200),
}

#: OpinionQA SEX -> Nemotron `sex`.
SEX_MAP: dict[str, str] = {"female": "Female", "male": "Male"}

#: OpinionQA EDUCATION -> set of Nemotron `education_level` values.
EDUCATION_MAP: dict[str, set[str]] = {
    "less than high school": {"less_than_9th", "9th_12th_no_diploma"},
    "high school graduate": {"high_school"},
    "some college, no degree": {"some_college"},
    "associate's degree": {"associates"},
    "college graduate/some postgrad": {"bachelors"},
    "postgraduate": {"graduate"},
}

#: OpinionQA MARITAL -> Nemotron `marital_status` (None = no faithful equivalent).
MARITAL_MAP: dict[str, str | None] = {
    "married": "married_present",
    "never been married": "never_married",
    "divorced": "divorced",
    "widowed": "widowed",
    "living with a partner": None,  # Nemotron has no cohabiting category
}

#: US state (USPS) -> Census region. Puerto Rico (PR) is deliberately absent →
#: PR personas never match an OpinionQA CREGION (Midwest/Northeast/South/West).
_REGION_STATES: dict[str, tuple[str, ...]] = {
    "Northeast": ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    "Midwest": ("IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "South": ("DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY",
              "MS", "TN", "AR", "LA", "OK", "TX"),
    "West": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI",
             "OR", "WA"),
}
CENSUS_REGION: dict[str, str] = {
    s: region for region, states in _REGION_STATES.items() for s in states
}

#: OpinionQA segment keys Nemotron can match structurally.
MATCHABLE_AXES: frozenset[str] = frozenset({"AGE", "SEX", "CREGION", "EDUCATION", "MARITAL"})


def _seed_for(segment: dict[str, str], base_seed: int) -> int:
    """Stable per-segment seed so each segment gets a fixed (reusable) panel."""
    payload = f"{base_seed}::" + "::".join(f"{k}={segment[k]}" for k in sorted(segment))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


#: Prose renderings for the underscored categorical codes (for `rich` conditioning).
_MARITAL_PROSE = {
    "married_present": "married", "never_married": "never married",
    "divorced": "divorced", "widowed": "widowed", "separated": "separated",
}
_EDU_PROSE = {
    "less_than_9th": "less than a 9th-grade education",
    "9th_12th_no_diploma": "some high school but no diploma",
    "high_school": "a high-school diploma", "some_college": "some college",
    "associates": "an associate's degree", "bachelors": "a bachelor's degree",
    "graduate": "a graduate degree",
}


def render_persona(row: dict, mode: str = "summary") -> str:
    """Render one persona row to conditioning text.

    ``summary`` (Rounds 1–3): the one-sentence ``persona`` field only — what the
    original runs used. ``rich`` (Round 4): an explicit demographic header (age,
    sex, marital status, education, occupation, location — the structured fields we
    were discarding) plus the longer narrative fields, to test whether stronger
    contextualization makes the persona actually bite.
    """
    persona = str(row.get("persona", "") or "").strip()
    if mode == "summary":
        return persona
    sex = str(row.get("sex", "") or "").lower()
    age = int(row.get("age", 0) or 0)
    occ = str(row.get("occupation", "") or "").replace("_", " ")
    marital = _MARITAL_PROSE.get(str(row.get("marital_status", "")),
                                 str(row.get("marital_status", "") or "").replace("_", " "))
    edu = _EDU_PROSE.get(str(row.get("education_level", "")),
                         str(row.get("education_level", "") or "").replace("_", " "))
    city, state = str(row.get("city", "") or ""), str(row.get("state", "") or "")
    _nonjob = {"not in workforce", "unemployed", "retired", "homemaker", "student", "disabled"}
    if occ and occ not in _nonjob:
        occ_clause = f", working as a {occ}"
    elif occ:
        occ_clause = f", {occ}"
    else:
        occ_clause = ""
    header = (f"A {age}-year-old {sex}, {marital}, with {edu}{occ_clause}, "
              f"living in {city}, {state}.")
    parts = [header, persona]
    for label, key in (("Work", "professional_persona"),
                       ("Background", "cultural_background"),
                       ("Interests", "hobbies_and_interests")):
        v = str(row.get(key, "") or "").strip()
        if v:
            parts.append(f"{label}: {v}")
    return "\n".join(p for p in parts if p)


class PersonaBank:
    """A grounded, sampleable population of Nemotron US-adult personas."""

    def __init__(self, df: pd.DataFrame | None = None, *, min_age: int = MIN_ADULT_AGE,
                 text_mode: str = "summary") -> None:
        if df is None:
            if not SLIM_PATH.exists():
                raise FileNotFoundError(
                    f"{SLIM_PATH} not found; run `uv run python scripts/prepare_nemotron.py`."
                )
            df = pd.read_parquet(SLIM_PATH)
        self.adults = df[df["age"] >= min_age].reset_index(drop=True)
        self.text_mode = text_mode  # "summary" | "rich" — how personas are rendered
        # Cache sampled panels per (segment-key, k, text_mode) so a segment's
        # electorate is identical across all questions sharing it (fixed panel).
        self._panel_cache: dict[tuple[str, int, str], list[str]] = {}

    # -- axis matching ----------------------------------------------------- #
    def _axis_mask(self, key: str, value: str) -> "pd.Series | None":
        """Boolean mask over ``self.adults`` for one pinned axis, or None if the
        axis/value is not structurally matchable in Nemotron."""
        a = self.adults
        if key == "AGE":
            rng = AGE_BINS.get(value)
            return None if rng is None else (a["age"] >= rng[0]) & (a["age"] <= rng[1])
        if key == "SEX":
            nem = SEX_MAP.get(value)
            return None if nem is None else a["sex"] == nem
        if key == "CREGION":
            states = [s for s, r in CENSUS_REGION.items() if r == value]
            return None if not states else a["state"].isin(states)
        if key == "EDUCATION":
            nem = EDUCATION_MAP.get(value)
            return None if nem is None else a["education_level"].isin(nem)
        if key == "MARITAL":
            nem = MARITAL_MAP.get(value, "__missing__")
            if nem in (None, "__missing__"):
                return None
            return a["marital_status"] == nem
        return None  # RACE / INCOME / RELIG / RELIGATTEND / POLPARTY / POLIDEOLOGY

    def is_matchable(self, segment: dict[str, str]) -> bool:
        """True iff every pinned axis in the segment can be matched in Nemotron."""
        return self._pool_index(segment) is not None

    def _pool_index(self, segment: dict[str, str]) -> "pd.Index | None":
        """Row index of adults matching the whole segment, or None if unmatchable."""
        if not segment:
            return self.adults.index
        mask = pd.Series(True, index=self.adults.index)
        for key, value in segment.items():
            m = self._axis_mask(key, str(value))
            if m is None:
                return None
            mask &= m
        return self.adults.index[mask]

    # -- panel sampling ---------------------------------------------------- #
    def panel_for(
        self, record: SimBenchRecord, k: int, base_seed: int = 0, min_pool: int = 10
    ) -> list[str] | None:
        """The persona-narrative panel for a record's segment.

        Returns a list of up to ``k`` persona texts (a fixed, seeded, representative
        draw — identical for every record sharing the segment), or ``None`` when the
        segment is unmatchable or its matched pool is below ``min_pool`` (caller
        should fall back). An empty segment (pop split) → the whole-corpus
        electorate.
        """
        seg = dict(record.segment)
        cache_key = (_seg_key(seg), k, self.text_mode)
        if cache_key in self._panel_cache:
            return self._panel_cache[cache_key]
        rows = self.panel_rows(record, k, base_seed=base_seed, min_pool=min_pool)
        if rows is None:
            return None
        personas = [render_persona(r, self.text_mode) for r in rows]
        self._panel_cache[cache_key] = personas
        return personas

    def panel_rows(
        self, record: SimBenchRecord, k: int, base_seed: int = 0, min_pool: int = 10
    ) -> list[dict] | None:
        """The raw persona rows (dicts) for a record's segment — same fixed, seeded
        draw as :meth:`panel_for`, but with all columns (for worldview enrichment).
        Returns ``None`` when unmatchable or the matched pool is below ``min_pool``."""
        seg = dict(record.segment)
        idx = self._pool_index(seg)
        if idx is None or len(idx) < min_pool:
            return None
        pool = self.adults.loc[idx]
        n = min(k, len(pool))
        return pool.sample(n=n, random_state=_seed_for(seg, base_seed)).to_dict("records")


def _seg_key(segment: dict[str, str]) -> str:
    return "|".join(f"{k}={segment[k]}" for k in sorted(segment)) or "<pop>"


@lru_cache(maxsize=1)
def default_bank() -> PersonaBank:
    """Process-wide singleton bank loaded from the slim cache (lazy)."""
    return PersonaBank()
