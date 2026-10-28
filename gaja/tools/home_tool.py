"""Home Assistant REST API local tool for fast room status, AC, and media-player control.

Reads ``HASS_URL`` and ``HASS_TOKEN`` strictly from environment variables (never ``.env`` files or
``~/.hermes``); when either variable is absent, the tool reports itself disabled.

The household's real entity ids, spoken aliases, ``write_allowlist``, spoken reply forms and
boiler markers live in ``[home] household_file`` (default ``data/home.toml``, gitignored — never
in this repo; schema in ``docs/home.example.toml``). Without the file: no aliases and an empty
allowlist, i.e. reads only through configured aliases and no writes at all (NEW-6).

Safety invariants:
- Writes require an allowlist match in the household's ``write_allowlist`` (empty by default:
  no writes). Boiler entities (``home.boiler_id_substrings`` plus ``water_heater.*``) and
  ``button.press``, ``number.set_value``, ``select.*`` are read-only unless the exact
  ``entity_id`` is explicitly listed (wildcards never unlock them). Wildcard patterns may only
  unlock the ``media_player`` domain; every other domain needs the exact ``entity_id``.
- A write executes only when the utterance itself names the target entity (one of its aliases
  present with inflection-tolerant token matching); otherwise Gaja asks back — between the
  candidates of the domain the utterance's nouns point at, or "Nie znam takiego urządzenia"
  with a handoff for an unknown named device. The LLM's ``device`` argument and raw entity ids
  never pick a write target; the only exception is an utterance with no device noun at all and
  exactly one entity of the action's domain (H1). Status reads keep the richer fallback.
- AC target temperatures must fall within the 16-30 °C sanity range (asks back outside it) and are
  also clamped to the entity's ``min_temp`` / ``max_temp``.
- Every write is logged at ``INFO`` with ``entity_id``, service and arguments (never the token).
- Every write polls ``GET /api/states/{entity_id}`` up to ~1.5 s and verifies the requested value
  before speaking success.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import httpx

from gaja.config import Settings
from gaja.text.normalize import normalize_for_tts, polish_declension, polish_noun_cases
from gaja.tools.base import ToolResult, clean_speakable, current_session_context

log = logging.getLogger(__name__)

SENSITIVE_DOMAINS: frozenset[str] = frozenset({"button", "number", "select", "water_heater"})
SENSITIVE_SERVICES: frozenset[str] = frozenset({"button.press", "number.set_value"})

# Code-level boiler marker floor (H3/M1e): generic heating words, not household data. They always
# apply on top of the household file's boiler_id_substrings, so a missing/misplaced file key can
# never leave a boiler writable.
BOILER_MARKER_FLOOR: frozenset[str] = frozenset({"boiler", "vitodens", "viessmann", "kociol", "kocioł", "piec"})

AC_SANITY_MIN_C = 16.0
AC_SANITY_MAX_C = 30.0
VERIFY_POLL_RETRIES = 3
VERIFY_POLL_INTERVAL_S = 0.5

# State-changing actions: an ambiguous utterance must ask back instead of falling back to the
# LLM's guessed device argument (NEW-3). Status reads keep the fallback.
WRITE_ACTIONS: frozenset[str] = frozenset(
    {"turn_on", "turn_off", "set_temperature", "set_hvac_mode", "volume_up", "volume_down", "volume_set", "mute", "unmute"}
)

_SENSITIVE_ATTR_SUBSTRINGS = ("token", "secret", "password", "api_key", "apikey", "webhook", "auth")
_STRIPPED_ATTR_EXACT = frozenset({"entity_picture", "entity_picture_local", "access_token"})

HVAC_MODE_ALIASES: dict[str, str] = {
    "cool": "cool",
    "chlodzenie": "cool",
    "klima": "cool",
    "zimno": "cool",
    "heat": "heat",
    "grzanie": "heat",
    "ogrzewanie": "heat",
    "cieplo": "heat",
    "fan_only": "fan_only",
    "fan": "fan_only",
    "nawiew": "fan_only",
    "wentylator": "fan_only",
    "wentylacja": "fan_only",
    "dry": "dry",
    "osuszanie": "dry",
    "auto": "auto",
    "automatyczny": "auto",
    "automatycznie": "auto",
    "heat_cool": "heat_cool",
    "off": "off",
    "wylacz": "off",
    "wylaczona": "off",
    "wylaczone": "off",
}

HVAC_MODE_LABELS_PL: dict[str, str] = {
    "cool": "chłodzenie",
    "heat": "grzanie",
    "fan_only": "nawiew",
    "dry": "osuszanie",
    "auto": "automatyczny",
    "heat_cool": "automatyczny",
    "off": "wyłączona",
}

_PL_TENS: dict[str, int] = {
    "dziesiec": 10,
    "jedenascie": 11,
    "dwanascie": 12,
    "trzynascie": 13,
    "czternascie": 14,
    "pietnascie": 15,
    "szesnascie": 16,
    "siedemnascie": 17,
    "osiemnascie": 18,
    "dziewietnascie": 19,
    "dwadziescia": 20,
    "trzydziesci": 30,
    "czterdziesci": 40,
}
_PL_UNITS: dict[str, int] = {
    "zero": 0,
    "jeden": 1,
    "dwa": 2,
    "trzy": 3,
    "cztery": 4,
    "piec": 5,
    "szesc": 6,
    "siedem": 7,
    "osiem": 8,
    "dziewiec": 9,
}

# Spoken nominative / accusative / genitive labels for natural Polish replies ("Włączyłam klimę w
# salonie") come from the household file ([home] spoken_forms, keyed by entity_id); the fallback
# declines the configured alias (or Home Assistant's friendly_name) best-effort, and an empty
# accusative/genitive slot means "unknown form" — callers speak a neutral sentence shape then.

_STOPWORDS: frozenset[str] = frozenset(
    {
        "w",
        "we",
        "u",
        "na",
        "do",
        "z",
        "ze",
        "od",
        "dla",
        "o",
        "czy",
        "jak",
        "jaka",
        "jaki",
        "jakie",
        "jest",
        "sa",
        "chodzi",
        "dziala",
        "wlacz",
        "wylacz",
        "ustaw",
        "zmien",
        "sprawdz",
        "podaj",
        "pokaz",
        "stan",
        "status",
        "prosze",
        "mi",
        "teraz",
        "stopni",
        "stopnie",
        "stopien",
        "procent",
        "glosnosc",
        "glosniej",
        "ciszej",
        "wycisz",
        "odcisz",
        "tryb",
    }
)

# Domains the tool can actually execute writes against (H1 ask-back candidates).
WRITABLE_DOMAINS: tuple[str, ...] = ("climate", "media_player")

# Generic Polish device-noun stems (folded, no household data): a token starting with one of
# these says "the utterance names a device" even when no alias token matches — so an unknown
# named device ("włącz zmywarkę w kuchni") is recognisable as such (H1). Inflection-tolerant via
# prefix stems because folded case endings vary ("zmywarkę" -> "zmywarke").
_DEVICE_NOUN_STEMS: frozenset[str] = frozenset(
    {
        "klima",
        "klimatyzacj",
        "klimatyzator",
        "telewizor",
        "tv",
        "amplituner",
        "wzmacniacz",
        "glosnik",
        "mikrofon",
        "piec",
        "kocio",
        "kotl",
        "bojler",
        "zasobnik",
        "ogrzewani",
        "palnik",
        "drukark",
        "konsol",
        "zmywark",
        "pralk",
        "lodowk",
        "odkurzacz",
        "wentylator",
        "osuszacz",
        "kamer",
        "ps5",
        "playstation",
    }
)


def _is_device_noun(token: str) -> bool:
    """Whether a folded canonical token starts like a generic Polish device noun."""
    return token.startswith(tuple(sorted(_DEVICE_NOUN_STEMS)))


# MED round 5: device nouns that name a writable domain. A candidate entity of the OTHER writable
# domain contradicts the utterance's own noun and must never resolve ("włącz telewizor u Ani" must
# not ride the "u Ani" room alias onto climate.ania). Boiler/heating nouns stay unmapped: their
# entities are read-only anyway and status questions keep the fallback resolution.
_DEVICE_NOUN_DOMAINS: tuple[tuple[str, str], ...] = (
    ("klima", "climate"),
    ("klimatyzacj", "climate"),
    ("klimatyzator", "climate"),
    ("telewizor", "media_player"),
    ("tv", "media_player"),
    ("amplituner", "media_player"),
    ("wzmacniacz", "media_player"),
    ("glosnik", "media_player"),
)


def _utterance_device_domains(text: str) -> frozenset[str]:
    """Writable domains named by the utterance's own device nouns ("telewizor" -> media_player)."""
    domains: set[str] = set()
    for tok in canonical_tokens(text):
        for stem, domain in _DEVICE_NOUN_DOMAINS:
            if tok.startswith(stem):
                domains.add(domain)
    return frozenset(domains)


# H1 round 5: pronouns and spoken fillers never name a device, so they may accompany the bare-verb
# single-candidate exception ("włącz to", "wyłącz ją"). Everything else — światło, lampka,
# grzejnik, rolety, muzyka, a room word — is content: it names a device the aliases don't know.
_WRITE_FILLER_TOKENS: frozenset[str] = frozenset(
    {"to", "go", "ja", "je", "mu", "jej", "im", "ich", "nas", "was", "cie", "mnie", "mna",
     "sobie", "siebie", "jeszcze", "juz", "no", "hej", "gajo"}
)

# Climate value words (folded ASCII): HVAC mode names plus the temperature noun — "ustaw 22
# stopnie", "ustaw temperaturę", "ustaw na chłodzenie" name a value, not a device. Digits are
# accepted by _utterance_names_no_device directly; media value words (głośność, procent) are
# _STOPWORDS. HVAC_MODE_ALIASES keys are folded ASCII already, so no fold call at import time.
_CLIMATE_VALUE_TOKENS: frozenset[str] = frozenset(HVAC_MODE_ALIASES) | {"temperatura", "temperatur"}


def _utterance_names_no_device(u_set: set[str], *, ac_action: bool, media_action: bool) -> bool:
    """H1: every token is a pronoun/filler or the action's own value — the utterance names no
    device at all, so the single-candidate exception may apply. Any other content token
    ("światło", "lampka", "muzyka", a room word) names a device the aliases don't know: the
    write must ask back or hand off, never execute the one writable entity on a guess."""
    for tok in u_set:
        if _is_device_noun(tok):
            return False
        if tok in _WRITE_FILLER_TOKENS:
            continue
        if (ac_action or media_action) and tok.isdigit():
            continue
        if ac_action and (tok.startswith("temperatur") or tok in _CLIMATE_VALUE_TOKENS):
            continue
        return False
    return True

# F1/F3: after a bare "u <person>" alias's tokens are removed from an utterance, these generic
# words still name no device — sensor quantities and units, climate value words, room words,
# conjunctions. Any OTHER token left over ("zmywarke", "pralka", "ekspres", "piekarnik") is
# content the household does not know: a person place must never stand in for it.
_ROOM_TOKENS: frozenset[str] = frozenset(
    {
        "salon", "salonie", "salonu", "sypialnia", "sypialni", "kuchnia", "kuchni",
        "pokoj", "pokoju", "lazienka", "lazience", "garaz", "garazu",
        "biuro", "biurze", "piwnica", "piwnicy", "dom", "domu",
    }
)
_PERSON_PLACE_FILLER_TOKENS: frozenset[str] = frozenset(
    _CLIMATE_VALUE_TOKENS
    | _ROOM_TOKENS
    | {
        "i", "oraz", "ale", "tez", "takze",
        "temperatura", "temperatur", "temperatury", "temperature",
        "wilgotnosc", "cisnienie", "cisnienia",
        "stopien", "stopnia", "stopnie", "stopni", "procent", "glosnosc",
        "cieplo", "zimno", "goraco", "chlodno",
    }
)


def _is_non_content_token(token: str) -> bool:
    """Whether a canonical token names a value, a place or neither — never a device."""
    return (
        token.isdigit()
        or token in _PERSON_PLACE_FILLER_TOKENS
        or token.startswith(("temperatur", "wilgotn", "cisnien", "stopn"))
    )


def _person_reference_tokens(aliases: dict[str, str]) -> frozenset[str]:
    """Canonical tokens of the household's bare person references: "u <person>" aliases ("u Ani",
    "u rodziców") plus single-word aliases that name the same person ("rodzice"). A subset hit
    whose evidence is only these tokens is a person place standing in, not a device match."""
    person_sets: set[tuple[str, ...]] = set()
    for alias in aliases:
        if re.fullmatch(r"u\s+\w+", alias.strip(), re.IGNORECASE):
            person_sets.add(tuple(canonical_tokens(alias)))
    tokens: set[str] = set()
    for toks in person_sets:
        tokens.update(toks)
    for alias in aliases:
        toks = canonical_tokens(alias)
        if len(toks) == 1 and any(toks[0] in ps for ps in person_sets):
            tokens.add(toks[0])
    return frozenset(tokens)


def _person_place_stands_in(
    q_set: set[str], person_tokens: frozenset[str], known_tokens: frozenset[str] = frozenset()
) -> bool:
    """F1/F3: whether an unknown content noun remains next to the household's person words —
    the utterance names a device the household does not know ("włącz zmywarkę u Ani"), so no
    bare person-place alias may resolve (or ask back about) the person's own entity. Tokens in
    ``known_tokens`` (the household's alias words, e.g. "telewizor") are never unknown nouns."""
    if not person_tokens or not (q_set & person_tokens):
        return False
    return any(
        not _is_non_content_token(tok) and tok not in known_tokens for tok in q_set - person_tokens
    )


def _alias_tokens(aliases: dict[str, str]) -> frozenset[str]:
    """Every canonical token of the household's aliases: the device words it knows."""
    return frozenset(tok for alias in aliases for tok in canonical_tokens(alias))

# Irregular and common Polish inflections -> canonical lemma for entity resolution.
# Household person names never live here: aliases are written the way the household speaks them
# ("klima u Ani"), so exact-token matching resolves their inflected forms already (NEW-6).
_WORD_LEMMAS: dict[str, str] = {
    "salon": "salon",
    "salonie": "salon",
    "salonu": "salon",
    "salonem": "salon",
    "sypialnia": "sypialnia",
    "sypialni": "sypialnia",
    "sypialnie": "sypialnia",
    "rodzice": "rodzice",
    "rodzicow": "rodzice",
    "rodzicom": "rodzice",
    "rodzicach": "rodzice",
    "pokoj": "pokoj",
    "pokoju": "pokoj",
    "pokojem": "pokoj",
    "klima": "klima",
    "klime": "klima",
    "klimy": "klima",
    "klimie": "klima",
    "klimatyzacja": "klima",
    "klimatyzacje": "klima",
    "klimatyzacji": "klima",
    "klimatyzator": "klima",
    "telewizor": "tv",
    "telewizora": "tv",
    "telewizorze": "tv",
    "telewizorem": "tv",
    "tv": "tv",
    "amplituner": "amplituner",
    "amplitunera": "amplituner",
    "amplitunerze": "amplituner",
    "wzmacniacz": "wzmacniacz",
    "wzmacniacza": "wzmacniacz",
    "onkyo": "onkyo",
    "glosnik": "glosnik",
    "glosnika": "glosnik",
    "glosniku": "glosnik",
    "piec": "piec",
    "pieca": "piec",
    "piecu": "piec",
    "piecem": "piec",
    "kociol": "kociol",
    "kotla": "kociol",
    "kotle": "kociol",
    "kotlem": "kociol",
    "bojler": "bojler",
    "bojlera": "bojler",
    "bojlerze": "bojler",
    "zasobnik": "zasobnik",
    "zasobnika": "zasobnik",
    "ciepla": "ciepla",
    "cieplej": "ciepla",
    "woda": "woda",
    "wody": "woda",
    "wode": "woda",
    "palnik": "palnik",
    "palnika": "palnik",
    "cisnienie": "cisnienie",
    "cisnienia": "cisnienie",
    "instalacja": "instalacja",
    "instalacji": "instalacja",
    "wilgotnosc": "wilgotnosc",
    "wilgotnosci": "wilgotnosc",
    "drukarka": "drukarka",
    "drukarki": "drukarka",
    "drukarce": "drukarka",
    "drukarke": "drukarka",
    "konsola": "konsola",
    "konsoli": "konsola",
    "konsole": "konsola",
    "ps5": "ps5",
    "playstation": "playstation",
    "strefa": "strefa",
    "strefy": "strefa",
    "strefie": "strefa",
    "zone": "strefa",
    "ogrzewanie": "ogrzewanie",
    "ogrzewania": "ogrzewanie",
    "dom": "dom",
    "domu": "dom",
}

_DISTINGUISHING_NOUNS: frozenset[str] = frozenset(
    {"tv", "glosnik", "wilgotnosc", "drukarka", "konsola", "amplituner", "piec", "kociol", "bojler", "palnik", "cisnienie"}
)


@dataclass(frozen=True)
class Resolution:
    entity_id: str | None
    error: str | None = None
    ask_back: str | None = None
    # F3: read-only entities (boilers) the utterance ALSO named while one writable entity won
    # ("wyłącz piec i klimę u Ani"). The allowed write executes; the reply must say out loud
    # that these cannot be changed instead of silently dropping them.
    read_only_named: tuple[str, ...] = ()


def home_credentials() -> tuple[str, str] | None:
    """Read ``(HASS_URL, HASS_TOKEN)`` strictly from ``os.environ``; ``None`` when unconfigured."""
    url = os.environ.get("HASS_URL", "").strip().rstrip("/")
    token = os.environ.get("HASS_TOKEN", "").strip()
    if not url or not token:
        return None
    return url, token


def home_configured() -> bool:
    """Whether ``HASS_URL`` and ``HASS_TOKEN`` are present in the environment."""
    return home_credentials() is not None


def _fold(text: str) -> str:
    text = text.lower().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _sanitize_attributes(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip token/picture-like fields from Home Assistant state attributes before returning."""
    if not attrs:
        return {}
    clean: dict[str, Any] = {}
    for key, val in attrs.items():
        k_low = str(key).lower()
        if k_low in _STRIPPED_ATTR_EXACT:
            continue
        if any(sub in k_low for sub in _SENSITIVE_ATTR_SUBSTRINGS):
            continue
        clean[key] = val
    return clean


def _extract_temperature_from_utterance(utterance: str) -> float | None:
    """Extract numeric or Polish spelled-out temperature before ``stopni/stopnie/stopień``."""
    m_num = re.search(r"(?<!\w)(-?\d{1,3}(?:[.,]\d)?)\s*stopn", utterance, re.IGNORECASE)
    if m_num:
        return float(m_num.group(1).replace(",", "."))
    folded = _fold(utterance)
    m_words = re.search(r"\b([a-z]+(?:\s+[a-z]+){0,3})\s+stopn", folded)
    if not m_words:
        return None
    words = m_words.group(1).split()
    has_half = False
    if len(words) >= 2 and words[-2:] == ["i", "pol"]:
        has_half = True
        words = words[:-2]
    if not words:
        return None
    # Scan from the tail of the captured words so prepositions like "na dwadziescia jeden" work.
    if len(words) >= 2 and words[-2] in ("dwadziescia", "trzydziesci", "czterdziesci") and words[-1] in _PL_UNITS:
        val = float(_PL_TENS[words[-2]] + _PL_UNITS[words[-1]])
        return val + (0.5 if has_half else 0.0)
    last = words[-1]
    if last in _PL_TENS:
        return float(_PL_TENS[last]) + (0.5 if has_half else 0.0)
    if last in _PL_UNITS:
        return float(_PL_UNITS[last]) + (0.5 if has_half else 0.0)
    return None


def _lemmatize_word(word: str) -> str:
    if word in _WORD_LEMMAS:
        return _WORD_LEMMAS[word]
    if len(word) >= 5:
        for suf in ("owi", "ami", "ach", "ego", "emu", "iej", "ie", "em", "om", "ow", "iu", "ia", "ze", "ce"):
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                stem = word[: -len(suf)]
                if stem in _WORD_LEMMAS:
                    return _WORD_LEMMAS[stem]
        for suf in ("a", "e", "i", "o", "u", "y"):
            if word.endswith(suf) and len(word) - len(suf) >= 4:
                stem = word[: -len(suf)]
                if stem in _WORD_LEMMAS:
                    return _WORD_LEMMAS[stem]
    return word


def canonical_tokens(phrase: str) -> list[str]:
    folded = _fold(phrase)
    raw_tokens = re.findall(r"[a-z0-9_]+", folded)
    out: list[str] = []
    for tok in raw_tokens:
        if tok in _STOPWORDS:
            continue
        lemma = _lemmatize_word(tok)
        if lemma not in _STOPWORDS:
            out.append(lemma)
    return out


def _short_label(entity_id: str, aliases: dict[str, str]) -> str:
    for alias, eid in aliases.items():
        if eid == entity_id:
            return alias
    return entity_id


def _format_ambiguity_question(candidates: list[str], aliases: dict[str, str], lang: str) -> str:
    labels = [_short_label(eid, aliases) for eid in candidates[:3]]
    if lang == "en":
        return f"Which device do you mean: {', '.join(labels)}?"
    if len(labels) == 2:
        return f"Które urządzenie masz na myśli: {labels[0]} czy {labels[1]}?"
    return f"Które urządzenie masz na myśli: {', '.join(labels[:-1])} czy {labels[-1]}?"


def _write_candidate_entities(
    utterance_tokens: set[str],
    aliases: dict[str, str],
    allowlist: list[str] | None,
    boiler_substrings: list[str] | None = None,
) -> dict[str, int]:
    """Every entity a write could target (alias targets + exact allowlist entries, writable
    domains only), scored by how strongly the utterance's words overlap its best alias.
    Read-only boiler entities are never offered (r5 6g): a write ask-back must not list what no
    allowlist pattern can unlock anyway."""
    exact_entries = {
        p.strip().lower()
        for p in (allowlist or [])
        if p and p.strip() and not any(ch in p for ch in "*?[")
    }
    entities: dict[str, int] = {}
    for alias, eid in aliases.items():
        dom = (eid or "").strip().lower().split(".", 1)[0]
        if dom not in WRITABLE_DOMAINS:
            continue
        eid = eid.strip().lower()
        # Only boilers the write would REFUSE are unlistable; an exactly-allowlisted boiler is
        # writable and may stay a candidate (is_write_allowed permits the exact id).
        if eid not in exact_entries and _is_boiler_entity(eid, boiler_substrings):
            continue
        score = 0
        for tok in set(canonical_tokens(alias)):
            if tok in utterance_tokens:
                score += 3 if _is_device_noun(tok) else 1
        entities[eid] = max(entities.get(eid, 0), score)
    for pattern in allowlist or []:
        p = pattern.strip().lower()
        if not p or "*" in p or "?" in p or "[" in p:
            continue
        if p.split(".", 1)[0] in WRITABLE_DOMAINS:
            entities.setdefault(p, 0)
    return entities


def _write_resolution_without_named_device(
    utterance: str,
    aliases: dict[str, str],
    *,
    action: str,
    lang: str,
    allowlist: list[str] | None = None,
    boiler_substrings: list[str] | None = None,
) -> Resolution:
    """H1: a write whose utterance did not resolve to one entity must ask back, never execute.

    The LLM's ``device`` argument (or a raw entity_id) is not trusted at all here. The only
    exception is an utterance with no device word whatsoever — just a verb plus pronouns/fillers
    ("włącz to", "wyłącz ją"), plus a temperature or HVAC-mode value for climate actions — and
    exactly one entity of the action's domain: then the target is unambiguous.
    """
    u_set = set(canonical_tokens(utterance))
    ac_action = action in ("set_temperature", "set_hvac_mode")
    media_action = action in ("volume_up", "volume_down", "volume_set", "mute", "unmute")
    noun_domains = _utterance_device_domains(utterance)

    # F1: same rule as the subset match — when the utterance's only entity evidence is the
    # person place and an unknown content noun remains, the write names a device the household
    # does not know: say so, never ask back about the person's own entity.
    if _person_place_stands_in(u_set, _person_reference_tokens(aliases), _alias_tokens(aliases)):
        ask = (
            "Nie znam takiego urządzenia."
            if lang == "pl"
            else "I don't know such a device."
        )
        return Resolution(entity_id=None, error="unknown_device", ask_back=ask)

    def domain_ok(eid: str) -> bool:
        dom = eid.split(".", 1)[0]
        if ac_action and dom != "climate":
            return False
        if media_action and dom != "media_player":
            return False
        if noun_domains and dom in WRITABLE_DOMAINS and dom not in noun_domains:
            # MED round 5: the utterance's own device noun names a different writable domain
            # ("włącz telewizor u Ani"): a climate candidate contradicts it, even as an ask-back.
            return False
        return True

    entities = _write_candidate_entities(u_set, aliases, allowlist, boiler_substrings)

    # The utterance's nouns point at specific domains: ask between those entities only, so a
    # TV noun can never resolve to (or even mention) a climate entity.
    named = sorted({eid for eid, score in entities.items() if score > 0 and domain_ok(eid)}, key=lambda e: (-entities[e], e))
    if named:
        if len(named) >= 2:
            return Resolution(
                entity_id=None,
                error="ambiguous_device",
                ask_back=_format_ambiguity_question(named, aliases, lang),
            )
        label = _short_label(named[0], aliases)
        # 6h: an open question, never a yes/no one. The pipeline carries no pending-clarification
        # state, so a bare "tak" cannot resolve to this candidate (it is classified fresh and the
        # home tool has no cross-turn memory); "Masz na myśli klimę u Ani?" + "tak" dead-ended.
        # An open "Które urządzenie: …?" invites the repeat that names the device more precisely.
        ask = f"Które urządzenie: {label}?" if lang == "pl" else f"Which device: {label}?"
        return Resolution(entity_id=None, error="ambiguous_device", ask_back=ask)

    if _utterance_names_no_device(u_set, ac_action=ac_action, media_action=media_action):
        # No device word at all ("wyłącz", "wyłącz to", "ustaw 22 stopnie"): with exactly one
        # entity of the action's domain the write target is unambiguous and may execute;
        # otherwise ask back.
        every = sorted({eid for eid in entities if domain_ok(eid)})
        if len(every) == 1:
            return Resolution(entity_id=every[0])
        if len(every) >= 2:
            return Resolution(
                entity_id=None,
                error="ambiguous_device",
                ask_back=_format_ambiguity_question(every, aliases, lang),
            )
    ask = "Nie znam takiego urządzenia." if lang == "pl" else "I don't know such a device."
    return Resolution(entity_id=None, error="no_match", ask_back=ask)


def _read_only_write_target(
    res: Resolution,
    utterance: str,
    aliases: dict[str, str],
    *,
    lang: str,
    boiler_substrings: list[str] | None,
) -> Resolution:
    """A write the utterance aims at a named read-only device (the boiler's hot water, a sensor)
    resolves to that entity, so the allowlist gate refuses it out loud ("Kocioł jest tylko do
    odczytu…") instead of "Nie znam takiego urządzenia" plus a handoff that lets Hermes try."""
    if res.entity_id is not None or res.error != "no_match":
        return res
    named = resolve_device(utterance, aliases, utterance="", action="status", lang=lang)
    eid = named.entity_id
    if eid is None:
        return res
    if eid.split(".", 1)[0] not in WRITABLE_DOMAINS or _is_boiler_entity(eid, boiler_substrings):
        return named
    return res


def resolve_device(
    phrase: str,
    aliases: dict[str, str],
    *,
    utterance: str = "",
    action: str = "status",
    lang: str = "pl",
    allowlist: list[str] | None = None,
    boiler_substrings: list[str] | None = None,
) -> Resolution:
    """Resolve a Polish device/room phrase to a Home Assistant ``entity_id`` using config aliases.

    The user's spoken ``utterance`` always takes precedence over the LLM's ``device`` argument
    (including raw ``domain.object_id`` strings like ``climate.salon``). A raw ``domain.object_id``
    pass-through is accepted only when the utterance did not resolve to a specific entity and the
    ``entity_id`` is a configured alias target or an exact allowlist entry.

    For every write action (``WRITE_ACTIONS``) the target must be named by the utterance itself
    (H1): if the utterance alone does not resolve to exactly one entity, the tool asks back —
    "Które urządzenie…?" between the candidates of the domain the utterance's nouns point at, or
    "Nie znam takiego urządzenia" when the named device is unknown. The LLM's ``device`` argument
    and a raw entity_id alone never pick a write target; the only exception is an utterance with
    no device noun at all and exactly one entity of the action's domain. Status reads keep the
    richer fallback resolution.
    """
    raw = (phrase or "").strip()

    # 0. The user's utterance always wins over the LLM's device argument (even if device="climate.salon").
    if utterance and raw:
        utterance_res = resolve_device(
            utterance,
            aliases,
            utterance="",
            action=action,
            lang=lang,
            allowlist=allowlist,
            boiler_substrings=boiler_substrings,
        )
        if utterance_res.entity_id is not None:
            return utterance_res
        if action in WRITE_ACTIONS:
            # H1 (write safety): for a write, the utterance naming no entity must ask back. Any
            # extra token ("na 22 stopnie", "w kuchni") used to turn the utterance into no_match,
            # after which the LLM's device guess (or a raw entity_id) was executed. Never combine
            # the two texts for a write: ask back instead.
            return _read_only_write_target(
                _write_resolution_without_named_device(
                    utterance, aliases, action=action, lang=lang, allowlist=allowlist,
                    boiler_substrings=boiler_substrings,
                ),
                utterance, aliases, lang=lang, boiler_substrings=boiler_substrings,
            )
    if utterance and not raw and action in WRITE_ACTIONS:
        # Same rule with no device argument at all: the utterance alone decides (H1).
        utterance_res = resolve_device(
            utterance, aliases, utterance="", action=action, lang=lang, allowlist=allowlist
        )
        if utterance_res.entity_id is not None:
            return utterance_res
        return _read_only_write_target(
            _write_resolution_without_named_device(
                utterance, aliases, action=action, lang=lang, allowlist=allowlist,
                boiler_substrings=boiler_substrings,
            ),
            utterance, aliases, lang=lang, boiler_substrings=boiler_substrings,
        )

    combined_text = f"{raw} {utterance}".strip()
    folded_combined = _fold(combined_text)
    wants_humidity = "wilgotn" in folded_combined
    wants_temp_or_ac = bool(re.search(r"\b(temperatur|klim|chlodz|grzan|ogrzewan|nawiew|osuszan)", folded_combined))
    is_media_action = action in ("volume_up", "volume_down", "volume_set", "mute", "unmute")
    is_ac_action = action in ("set_temperature", "set_hvac_mode")

    noun_domains = _utterance_device_domains(combined_text)

    def domain_compatible(eid: str) -> bool:
        domain = eid.split(".", 1)[0]
        if is_ac_action and domain != "climate":
            return False
        if is_media_action and domain != "media_player":
            return False
        if noun_domains and domain in WRITABLE_DOMAINS and domain not in noun_domains:
            # MED round 5: the utterance's device noun (telewizor/tv -> media_player, klima ->
            # climate, amplituner/głośnik -> media_player) forbids candidates of the other
            # writable domain ("włącz telewizor u Ani" never resolves the "u Ani" climate alias).
            return False
        if wants_humidity and eid.startswith("climate."):
            return False
        if wants_temp_or_ac and domain == "media_player":
            return False
        return True

    # Restricted pass-through: accept domain.object_id ONLY if it is a configured alias value or an
    # exact allowlist entry (never arbitrary entities, and only after checking utterance above).
    if re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", raw.lower()):
        candidate_eid = raw.lower()
        allowed_passthrough = {v.strip().lower() for v in aliases.values() if v and v.strip()}
        if allowlist:
            allowed_passthrough |= {
                p.strip().lower()
                for p in allowlist
                if p and p.strip() and "*" not in p and "?" not in p and "[" not in p
            }
        if candidate_eid in allowed_passthrough and domain_compatible(candidate_eid):
            return Resolution(entity_id=candidate_eid)
        ask = (
            f"Nie znalazłam w domu urządzenia pasującego do: {raw}. Podaj proszę dokładną nazwę lub pokój."
            if lang == "pl"
            else "I could not match that home device."
        )
        return Resolution(entity_id=None, error="no_match", ask_back=ask)

    q_tokens = canonical_tokens(raw)
    if wants_humidity and "wilgotnosc" not in q_tokens:
        q_tokens = ["wilgotnosc", *q_tokens]
    if not q_tokens and utterance:
        q_tokens = canonical_tokens(utterance)

    if not q_tokens:
        ask = (
            "O które urządzenie w domu pytasz?"
            if lang == "pl"
            else "Which home device do you mean?"
        )
        return Resolution(entity_id=None, error="no_match", ask_back=ask)

    q_set = set(q_tokens)

    # 1. Full alias match: every canonical token of the configured alias appears in the query.
    subset_hits: list[tuple[int, str, frozenset[str]]] = []
    for alias, eid in aliases.items():
        if not domain_compatible(eid):
            continue
        a_tokens = canonical_tokens(alias)
        if not a_tokens:
            continue
        a_set = frozenset(a_tokens)
        if a_set.issubset(q_set):
            if "strefa" in q_set and "strefa" not in a_set and any(
                "strefa" in canonical_tokens(other_a) for other_a in aliases
            ):
                continue
            subset_hits.append((len(a_set), eid, a_set))

    if subset_hits:
        best_len = max(score for score, _, _ in subset_hits)
        winners = [hit for hit in subset_hits if hit[0] == best_len]
        top_entities = list(dict.fromkeys(eid for _, eid, _ in winners))
        if len(top_entities) == 1:
            # F1/F3: a bare person place ("u Ani") must never stand in for a device the
            # household does not know. When every winning alias's evidence is only person
            # words and an unknown content noun remains ("włącz zmywarkę u Ani", "czy u Ani
            # działa pralka"), refuse instead of resolving the person's entity.
            winner_tokens: set[str] = set()
            for _, _, a_set in winners:
                winner_tokens.update(a_set)
            if winner_tokens <= _person_reference_tokens(aliases) and any(
                not _is_non_content_token(tok) for tok in q_set - winner_tokens
            ):
                ask = (
                    "Nie znam takiego urządzenia."
                    if lang == "pl"
                    else "I don't know such a device."
                )
                return Resolution(entity_id=None, error="unknown_device", ask_back=ask)
            if action in WRITE_ACTIONS:
                # F5: partial execution is symmetric. Whatever the word order or alias length,
                # the write targets the best subset hit the write gate would ALLOW, and every
                # refused boiler named next to it (a longer alias included — "kocioł gazowy w
                # piwnicy i klimę u Ani") is spoken in the same reply instead of winning the
                # longest match and silently dropping the allowed write. Nothing is ever
                # written to a boiler: an exactly-allowlisted boiler counts as writable.
                svc = {"turn_on": "turn_on", "turn_off": "turn_off"}.get(action, action)

                def _writable(eid: str) -> bool:
                    return is_write_allowed(
                        eid, eid.split(".", 1)[0], svc, allowlist or [], boiler_substrings
                    )

                writable = [(score, eid) for score, eid, _ in subset_hits if _writable(eid)]
                if writable:
                    wbest = max(score for score, _ in writable)
                    wtop = list(dict.fromkeys(eid for score, eid in writable if score == wbest))
                    if len(wtop) == 1:
                        read_only_named = tuple(sorted({
                            eid
                            for _, eid, _ in subset_hits
                            if eid != wtop[0]
                            and _is_boiler_entity(eid, boiler_substrings)
                            and not _writable(eid)
                        }))
                        return Resolution(entity_id=wtop[0], read_only_named=read_only_named)
                    return Resolution(
                        entity_id=None,
                        error="ambiguous_device",
                        ask_back=_format_ambiguity_question(wtop, aliases, lang),
                    )
                # No writable hit at all: keep the longest-match target so the write gate
                # refuses it out loud ("włącz piec" alone names only the read-only boiler).
            # Reads keep the longest match ("jaki jest stan pieca i klimy u Ani" answers the
            # boiler); a write that lands here names only read-only entities, which the write
            # gate refuses out loud before any Home Assistant call.
            return Resolution(entity_id=top_entities[0])
        return Resolution(
            entity_id=None,
            error="ambiguous_device",
            ask_back=_format_ambiguity_question(top_entities, aliases, lang),
        )

    # 2. Partial match: query tokens are a subset of alias tokens (e.g. bare "klima" or bare "telewizor").
    partial_hits: list[tuple[float, str]] = []
    for alias, eid in aliases.items():
        if not domain_compatible(eid):
            continue
        a_tokens = canonical_tokens(alias)
        if not a_tokens:
            continue
        a_set = set(a_tokens)
        if q_set.issubset(a_set):
            missing = a_set - q_set
            if missing & _DISTINGUISHING_NOUNS and not (q_set & _DISTINGUISHING_NOUNS):
                continue
            partial_hits.append((len(q_set) / len(a_set), eid))

    if partial_hits:
        best_entities = list(dict.fromkeys(eid for _, eid in sorted(partial_hits, key=lambda x: x[0], reverse=True)))
        if len(best_entities) == 1:
            return Resolution(entity_id=best_entities[0])
        return Resolution(
            entity_id=None,
            error="ambiguous_device",
            ask_back=_format_ambiguity_question(best_entities, aliases, lang),
        )

    ask = (
        f"Nie znalazłam w domu urządzenia pasującego do: {raw}. Podaj proszę dokładną nazwę lub pokój."
        if lang == "pl" and raw
        else ("Nie wiem, o które urządzenie w domu chodzi." if lang == "pl" else "I could not match that home device.")
    )
    return Resolution(entity_id=None, error="no_match", ask_back=ask)


# "wyłącz klimę u Ani i telewizor w salonie": the parts of one write joined by "i"/"oraz"/a comma.
_WRITE_PART_SPLIT = re.compile(r"(\s*,\s*|\s+(?:i|oraz|a\s+tak(?:ż|z)e)\s+)", re.IGNORECASE)


def _multi_write_parts(
    utterance: str,
    aliases: dict[str, str],
    *,
    action: str,
    lang: str,
    allowlist: list[str] | None,
    boiler_substrings: list[str] | None,
) -> tuple[list[tuple[str, str]], Resolution | None] | None:
    """Split a write that names several devices into ``(part, entity_id)`` pairs.

    The resolver scores one bag of words, so "wyłącz klimę u Ani i telewizor w salonie" kept one
    device and confirmed it as the whole command. A part counts when it names a household device
    word, or only a place ("i u Bartka", "oraz w sypialni"): then the previous part's device word
    is carried over ("klimę u Bartka"). Parts naming no device ("i ustaw na 22 stopnie") are not
    parts of the multi-device write. Returns None unless at least two distinct entities result;
    ``([], resolution)`` when a part does not resolve to exactly one entity, so nothing is written
    and that part is asked about.
    """
    # [piece, delimiter, piece, ...]: the delimiter decides whether a place-only piece is a new
    # part ("i u Bartka") or just STT punctuation inside one part ("wyłącz klimę, u Ani").
    split = _WRITE_PART_SPLIT.split(utterance)
    if len(split) < 3:
        return None
    folded = f" {_fold(utterance)} "
    if any(re.search(r"\s(?:i|oraz)\s|,", _fold(a)) and f" {_fold(a)} " in folded for a in aliases):
        return None  # an alias itself contains the conjunction
    known = _alias_tokens(aliases)
    places = _ROOM_TOKENS | _person_reference_tokens(aliases)
    parts: list[str] = []
    carried = ""
    for i in range(0, len(split), 2):
        piece = split[i].strip()
        if not piece:
            continue
        comma = i > 0 and "," in split[i - 1]
        nouns = [w for w in piece.split() if any(t in known and t not in places for t in canonical_tokens(w))]
        if nouns:
            carried = nouns[0]
            parts.append(piece)
        elif any(t in known for t in canonical_tokens(piece)):
            if comma and parts:
                parts[-1] = f"{parts[-1]} {piece}"
            elif carried:
                parts.append(f"{carried} {piece}")
    if len(parts) < 2:
        return None
    resolved: list[tuple[str, str]] = []
    for part in parts:
        res = resolve_device(
            part, aliases, utterance="", action=action, lang=lang,
            allowlist=allowlist, boiler_substrings=boiler_substrings,
        )
        if res.entity_id is None:
            return [], res
        if all(res.entity_id != eid for _, eid in resolved):
            resolved.append((part, res.entity_id))
    if len(resolved) < 2:
        return None
    return resolved, None


def _combine_multi_write(results: list[ToolResult]) -> ToolResult:
    """One reply for a multi-device write: the done parts first, the refused (read-only) ones last."""
    ordered = sorted(results, key=lambda r: r.error == "write_not_allowed")
    speak = " ".join(r.speak for r in ordered if r.speak)
    errors = [r.error for r in results if r.error]
    error = errors[0] if len(errors) == len(results) else None
    data: dict[str, Any] = {"results": [r.to_dict() for r in results]}
    refused = [str(r.data.get("entity_id")) for r in results if r.error == "write_not_allowed" and r.data.get("entity_id")]
    if refused:
        data["read_only_refused"] = refused  # same key as a single write that co-named a boiler (F3)
    return ToolResult(data=data, speak=speak, error=error)


def is_write_allowed(
    entity_id: str,
    domain: str,
    service: str,
    allowlist: list[str],
    boiler_substrings: list[str] | None = None,
) -> bool:
    """Check write permission against ``write_allowlist``, keeping boilers and sensitive services read-only.

    Wildcard patterns (``*`` ``?`` ``[``) may only unlock the ``media_player`` domain (H3/M1d):
    ``climate``, ``switch`` and every other domain need the exact ``entity_id``.
    """
    eid = entity_id.strip().lower()
    # r5 6f: the domain comes from the entity_id itself; the `domain` argument is kept only for
    # call compatibility and is never trusted (a stale/LLM-supplied domain must not widen the
    # check — a climate id must not pass on a media_player.* wildcard, or vice versa).
    dom = eid.split(".", 1)[0]
    svc = service.strip().lower()
    patterns = [p.strip().lower() for p in allowlist if p and p.strip()]
    exact_entries = {p for p in patterns if "*" not in p and "?" not in p and "[" not in p}
    wildcard_entries = [p for p in patterns if p not in exact_entries]

    is_sensitive = (
        _is_boiler_entity(eid, boiler_substrings)
        or dom in SENSITIVE_DOMAINS
        or f"{dom}.{svc}" in SENSITIVE_SERVICES
    )
    if is_sensitive:
        return eid in exact_entries

    if dom != "media_player":
        wildcard_entries = []  # wildcards never unlock a non-media_player domain
    return eid in exact_entries or any(fnmatch.fnmatchcase(eid, pat) for pat in wildcard_entries)


def _is_boiler_entity(entity_id: str, boiler_substrings: list[str] | None = None) -> bool:
    """Heating appliances stay read-only: ``water_heater.*`` plus the household's own boiler id
    markers (``[home] boiler_id_substrings`` in the household file) on top of the code-level
    generic floor (``BOILER_MARKER_FLOOR``)."""
    eid = entity_id.strip().lower()
    if eid.startswith("water_heater."):
        return True
    markers = BOILER_MARKER_FLOOR | {str(m).strip().lower() for m in (boiler_substrings or [])}
    return any(marker and marker in eid for marker in markers)


def _write_refusal_speak(
    entity_id: str,
    lang: str,
    boiler_substrings: list[str] | None = None,
) -> str:
    """The spoken refusal for a read-only device (the existing boiler wording, F3)."""
    if _is_boiler_entity(entity_id, boiler_substrings):
        return (
            "Kocioł jest tylko do odczytu i nie mogę zmienić jego ustawień."
            if lang == "pl"
            else "The boiler is read-only and its settings cannot be changed."
        )
    return (
        "To urządzenie jest tylko do odczytu i nie mogę zmienić jego ustawień."
        if lang == "pl"
        else "This device is read-only and cannot be modified."
    )


def _write_refusal_result(
    entity_id: str,
    domain: str,
    service: str,
    lang: str,
    boiler_substrings: list[str] | None = None,
) -> ToolResult:
    log.warning("home write refused by allowlist: entity=%s service=%s.%s", entity_id, domain, service)
    speak = _write_refusal_speak(entity_id, lang, boiler_substrings)
    return ToolResult(
        data={"error": "write_not_allowed", "entity_id": entity_id, "service": f"{domain}.{service}"},
        speak=speak,
        error="write_not_allowed",
    )


def clamp_temperature(target: float, attributes: dict[str, Any]) -> tuple[int | float, bool]:
    """Clamp requested AC temperature to the entity's ``min_temp`` / ``max_temp``.

    F6: a null attribute means the entity reports unknown limits; fall back to the defaults
    instead of raising ``TypeError`` on ``float(None)``."""
    raw_min = attributes.get("min_temp", 7.0)
    raw_max = attributes.get("max_temp", 35.0)
    min_temp = float(raw_min) if raw_min is not None else 7.0
    max_temp = float(raw_max) if raw_max is not None else 35.0
    step = float(attributes.get("target_temp_step", 1.0) or 1.0)
    clamped = max(min_temp, min(max_temp, float(target)))
    was_clamped = abs(clamped - float(target)) > 1e-6
    if step >= 1.0 and abs(step - round(step)) < 1e-6:
        rounded: int | float = int(round(clamped))
    else:
        rounded = round(round(clamped / step) * step, 1)
    return rounded, was_clamped


def _spoken_forms(
    entity_id: str,
    state_obj: dict[str, Any],
    aliases: dict[str, str],
    spoken_forms: dict[str, tuple[str, str, str]] | None = None,
) -> tuple[str, str, str]:
    configured = (spoken_forms or {}).get(entity_id.lower())
    if configured is not None:
        return configured
    friendly = str((state_obj.get("attributes") or {}).get("friendly_name") or "").strip()
    label = _short_label(entity_id, aliases) if entity_id in aliases.values() else (friendly or entity_id)
    if not friendly and label == entity_id:
        # No alias and no friendly_name: speak the object id's words, never the raw entity id
        # ("media_player.tv_salon" -> "tv salon").
        label = (entity_id.split(".", 1)[1] if "." in entity_id else entity_id).replace("_", " ").strip()
    # F7: an unknown label must never be spoken in the nominative where a case is needed
    # ("Włączyłam klima u Ani"). Decline it best-effort; an empty slot means the form is
    # unknown and the reply uses a neutral, case-free sentence shape.
    acc, gen = polish_noun_cases(label)
    return (label, acc, gen)


def _format_temp_pl(val: Any) -> str | None:
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if abs(num - round(num)) < 0.05:
        ival = int(round(num))
        word = polish_declension(abs(ival), "stopień", "stopnie", "stopni")
        return f"{ival} {word}"
    rounded = round(num, 1)
    return f"{rounded} stopnia"


def _cap_first(text: str) -> str:
    return (text[:1].upper() + text[1:]) if text else text


# F6: spoken agreement ("jest włączony/włączona/włączone") follows the label's head noun, derived
# from the nominative ending — consonant masculine, -a feminine, -o/-e/-um neuter — instead of a
# short prefix list ("Podgrzewacz CWU", "Klimatyzator" are masculine too). Abbreviations (words
# with no lowercase: CWU, TV), numbers and short function words never decide; -y/-i endings are
# undecidable and let the next word decide (Polish adjectives agree with the head noun, so the
# first decidable word carries the head's gender either way). The few nouns that break the
# ending rule get an explicit exception.
_GENDER_EXCEPTIONS: dict[str, str] = {
    "satelita": "m",  # masculine despite -a
}
_GENDER_PARTICIPLES: dict[str, tuple[str, str, str]] = {
    "m": ("włączony", "wyłączony", "ustawiony"),
    "f": ("włączona", "wyłączona", "ustawiona"),
    "n": ("włączone", "wyłączone", "ustawione"),
}


def _label_gender(nom: str) -> str | None:
    """The label's grammatical gender ('m'/'f'/'n') from its head noun's nominative ending;
    None when no word of the label decides (abbreviations, numbers)."""
    for word in re.findall(r"[^\W\d_]+", nom, re.UNICODE):
        if not any(ch.islower() for ch in word):
            continue  # abbreviations ("CWU", "TV", "TX") never decide
        folded = _fold(word)
        if len(folded) < 3:
            continue
        if folded in _GENDER_EXCEPTIONS:
            return _GENDER_EXCEPTIONS[folded]
        if folded.endswith("um"):
            return "n"
        last = folded[-1]
        if last in "oe":
            return "n"
        if last == "a":
            return "f"
        if last in "iyu":
            continue  # undecidable ("nowy", "pani") — the next word decides
        return "m"  # consonant-final, ł/ś/ż included ("podgrzewacz", "kocioł")
    return None


def _status_participles(nom: str) -> tuple[str, str, str]:
    """(on, off, set) participles agreeing with the spoken label; feminine is the fallback for
    labels whose gender no word decides."""
    return _GENDER_PARTICIPLES[_label_gender(nom) or "f"]


def _speak_status(
    entity_id: str,
    state_obj: dict[str, Any],
    aliases: dict[str, str],
    lang: str,
    spoken_forms: dict[str, tuple[str, str, str]] | None = None,
) -> str:
    state = str(state_obj.get("state") or "unknown")
    attrs = state_obj.get("attributes") or {}
    nom, _, _ = _spoken_forms(entity_id, state_obj, aliases, spoken_forms)
    cap_nom = _cap_first(nom)
    domain = entity_id.split(".", 1)[0]

    if state in ("unavailable", "unknown"):
        return f"Urządzenie {nom} jest teraz niedostępne." if lang == "pl" else f"{nom} is currently unavailable."

    if domain == "climate":
        cur_t = _format_temp_pl(attrs.get("current_temperature"))
        set_t = _format_temp_pl(attrs.get("temperature"))
        on_word, off_word, set_word = _status_participles(nom)
        gender = _label_gender(nom)
        if lang == "en":
            return f"{nom}: state {state}, current {attrs.get('current_temperature')} degrees."
        if state == "off":
            if cur_t and gender != "m":
                return f"{cap_nom} jest {off_word}, w pokoju jest {cur_t}."
            return f"{cap_nom} jest {off_word}."
        mode_pl = HVAC_MODE_LABELS_PL.get(state, state)
        parts = [f"{cap_nom} jest {on_word}, tryb {mode_pl}"]
        if set_t:
            parts.append(f"{set_word} na {set_t}")
        if cur_t:
            parts.append(f"obecnie {cur_t}")
        return ", ".join(parts) + "."

    if domain == "water_heater":
        cur_t = _format_temp_pl(attrs.get("current_temperature"))
        set_t = _format_temp_pl(attrs.get("temperature"))
        # "Bojler jest włączony" / "Ciepła woda jest włączona": follow the spoken label's gender.
        on_word, off_word, _ = _status_participles(nom)
        status_pl = on_word if state != "off" else off_word
        details = [f"{cap_nom} jest {status_pl}"]
        if cur_t:
            details.append(f"obecnie {cur_t}")
        if set_t:
            details.append(f"zadana {set_t}")
        return ", ".join(details) + "."

    if domain == "media_player":
        if state in ("off", "standby"):
            return f"{cap_nom} jest wyłączony." if lang == "pl" else f"{nom} is off."
        vol = attrs.get("volume_level")
        if isinstance(vol, (int, float)):
            pct = int(round(float(vol) * 100))
            pct_w = polish_declension(pct, "procent", "procenty", "procent")
            return f"{cap_nom} jest włączony, głośność {pct} {pct_w}."
        return f"{cap_nom} jest włączony." if lang == "pl" else f"{nom} is on."

    if domain == "binary_sensor":
        is_on = state == "on"
        return (
            # Present-tense verbs carry no grammatical gender ("Palnik działa", "Konsola nie działa").
            f"{cap_nom} {'teraz działa' if is_on else 'teraz nie działa'}."
            if lang == "pl"
            else f"{nom} is {'on' if is_on else 'off'}."
        )

    unit = str(attrs.get("unit_of_measurement") or "").strip()
    if unit == "%":
        try:
            ival = int(round(float(state)))
            pct_w = polish_declension(ival, "procent", "procenty", "procent")
            return f"{cap_nom} wynosi {ival} {pct_w}."
        except ValueError:
            pass
    if unit == "°C":
        t_str = _format_temp_pl(state)
        if t_str:
            return f"{cap_nom}: {t_str}."
    state_pl = {"idle": "w stanie gotowości", "printing": "drukuje", "stopped": "zatrzymana"}.get(state, state)
    suffix = f" {unit}" if unit else ""
    return f"{cap_nom}: {state_pl}{suffix}."


def _finalize(result: ToolResult, lang: str) -> ToolResult:
    if result.speak:
        result.speak = normalize_for_tts(clean_speakable(result.speak), lang)
    return result


async def _get_state(client: httpx.AsyncClient, base_url: str, headers: dict[str, str], entity_id: str) -> httpx.Response:
    return await client.get(f"{base_url}/api/states/{entity_id}", headers=headers)


def _state_matches_write(
    *,
    domain: str,
    action: str,
    v_state: str,
    v_attrs: dict[str, Any],
    initial_attrs: dict[str, Any],
    clamped_temp: int | float | None,
    mapped_mode: str | None,
    pct_target: int | None,
) -> bool:
    """Return True iff the re-read Home Assistant state reflects every requested write parameter."""
    if v_state in ("unavailable", "unknown"):
        return False

    if domain == "climate":
        expect_off = action == "turn_off" or (action == "set_hvac_mode" and mapped_mode == "off")
        if expect_off:
            return v_state in ("off", "standby")
        if v_state in ("off", "standby"):
            return False
        if mapped_mode is not None:
            if mapped_mode == "auto" and v_state in ("auto", "heat_cool"):
                pass
            elif v_state != mapped_mode:
                return False
        if clamped_temp is not None:
            raw_t = v_attrs.get("temperature")
            if raw_t is None:
                return False
            try:
                actual_t = float(raw_t)
            except (TypeError, ValueError):
                return False
            if abs(actual_t - float(clamped_temp)) > 0.25:
                return False
        return True

    if domain == "media_player":
        if action == "turn_off":
            return v_state in ("off", "standby")
        if action == "turn_on":
            return v_state not in ("off", "standby")
        if action == "volume_set":
            v_lvl = v_attrs.get("volume_level")
            if not isinstance(v_lvl, (int, float)) or pct_target is None:
                return False
            actual_pct = int(round(float(v_lvl) * 100))
            return abs(actual_pct - pct_target) <= 2
        if action == "volume_up":
            init_vol = initial_attrs.get("volume_level")
            v_lvl = v_attrs.get("volume_level")
            if isinstance(init_vol, (int, float)):
                if not isinstance(v_lvl, (int, float)):
                    return False
                if float(init_vol) >= 0.99:
                    return float(v_lvl) >= 0.99
                return float(v_lvl) > float(init_vol) + 1e-4
            return v_state not in ("off", "standby")
        if action == "volume_down":
            init_vol = initial_attrs.get("volume_level")
            v_lvl = v_attrs.get("volume_level")
            if isinstance(init_vol, (int, float)):
                if not isinstance(v_lvl, (int, float)):
                    return False
                if float(init_vol) <= 0.01:
                    return float(v_lvl) <= 0.01
                return float(v_lvl) < float(init_vol) - 1e-4
            return v_state not in ("off", "standby")
        if action == "mute":
            return v_attrs.get("is_volume_muted") is True
        if action == "unmute":
            return v_attrs.get("is_volume_muted") is False

    return True


def _speak_partial_write(
    completed: list[str],
    failed_service: str,
    acc: str,
    gen: str,
    lang: str,
    nom: str = "",
) -> str:
    if lang != "pl":
        return f"Partially applied ({', '.join(completed)}), but {failed_service} failed."
    first = completed[0] if completed else ""
    # F7: an empty acc/gen slot is an unknown form — speak a neutral, case-free shape instead.
    if first == "climate.turn_on" and failed_service == "climate.set_temperature":
        if acc:
            return f"Włączyłam {acc}, ale nie udało się ustawić temperatury."
        return f"{_cap_first(nom)}: włączono, ale nie udało się ustawić temperatury."
    if first == "climate.set_hvac_mode" and failed_service == "climate.set_temperature":
        if gen:
            return f"Zmieniłam tryb {gen}, ale nie udało się ustawić temperatury."
        return f"{_cap_first(nom)}: tryb zmieniony, ale nie udało się ustawić temperatury."
    if gen:
        return f"Wykonałam część polecenia dla {gen}, ale kolejne ustawienie się nie powiodło."
    return f"{_cap_first(nom)}: wykonano część polecenia, ale kolejne ustawienie się nie powiodło."


async def home(
    action: str = "status",
    device: str = "",
    temperature: int | float | None = None,
    hvac_mode: str | None = None,
    level: int | float | None = None,
    lang: str = "pl",
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    verify_delay_s: float = VERIFY_POLL_INTERVAL_S,
) -> ToolResult:
    """Query or control allowlisted Home Assistant entities via the REST API."""
    cfg = settings if settings is not None else Settings()
    if not cfg.home.enabled:
        return ToolResult(data={"error": "disabled"}, speak="", error="disabled", handoff=True)

    creds = home_credentials()
    if creds is None:
        return ToolResult(data={"error": "disabled"}, speak="", error="disabled", handoff=True)

    base_url, token = creds
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    act = (action or "status").strip().lower()
    ctx = current_session_context.get()
    utterance = str(ctx.get("utterance") or "") if ctx else ""

    # Recover temperature / hvac_mode from utterance when the small selector omitted them
    if temperature is None and act in ("set_temperature", "turn_on") and utterance:
        temperature = _extract_temperature_from_utterance(utterance)
    if hvac_mode is None and act == "set_hvac_mode" and utterance:
        for word in re.findall(r"[a-ząćęłńóśźż_]+", utterance.lower()):
            folded_w = _fold(word)
            if folded_w in HVAC_MODE_ALIASES:
                hvac_mode = HVAC_MODE_ALIASES[folded_w]
                break

    allowlist = cfg.home.write_allowlist
    if act in WRITE_ACTIONS and utterance:
        multi = _multi_write_parts(
            utterance, cfg.home.aliases, action=act, lang=lang, allowlist=allowlist,
            boiler_substrings=cfg.home.boiler_id_substrings,
        )
        if multi is not None:
            targets, failure = multi
            if failure is not None:
                # All or ask: one part names no single device, so nothing is written.
                return _finalize(
                    ToolResult(
                        data={"error": failure.error or "no_match", "device": device},
                        speak=failure.ask_back or "O które urządzenie chodzi?",
                        error=failure.error or "no_match",
                    ),
                    lang,
                )
            raw = (device or "").strip()
            if raw and _fold(raw) != _fold(utterance):
                # The model split the command into one call per device: this call writes only
                # the device its argument names, and only if the utterance itself named it (H1).
                own = resolve_device(
                    raw, cfg.home.aliases, action=act, lang=lang, allowlist=allowlist,
                    boiler_substrings=cfg.home.boiler_id_substrings,
                )
                if any(eid == own.entity_id for _, eid in targets):
                    targets = [(part, eid) for part, eid in targets if eid == own.entity_id]
            results: list[ToolResult] = []
            for part, _ in targets:
                part_temp = _extract_temperature_from_utterance(part) if act in ("set_temperature", "turn_on") else None
                tok = current_session_context.set({**(ctx or {}), "utterance": part})
                try:
                    results.append(await home(
                        action=act, device=part,
                        temperature=part_temp if part_temp is not None else temperature,
                        hvac_mode=hvac_mode, level=level, lang=lang, settings=cfg,
                        transport=transport, verify_delay_s=verify_delay_s,
                    ))
                finally:
                    current_session_context.reset(tok)
            return results[0] if len(results) == 1 else _combine_multi_write(results)

    resolution = resolve_device(
        device,
        cfg.home.aliases,
        utterance=utterance,
        action=act,
        lang=lang,
        allowlist=allowlist,
        boiler_substrings=cfg.home.boiler_id_substrings,
    )
    if resolution.entity_id is None:
        err_code = resolution.error or "no_match"
        if err_code == "unknown_device":
            # F1/F3: the utterance named a device the household does not know next to a person
            # place. The honest answer is spoken for reads and writes alike; a write still
            # hands the turn to Hermes (it may know the device), like any unknown H1 write,
            # while a read is answered here — the household config is the authority at home.
            return _finalize(
                ToolResult(
                    data={"error": err_code, "device": device},
                    speak=resolution.ask_back
                    or ("Nie znam takiego urządzenia." if lang == "pl" else "I don't know such a device."),
                    error=err_code,
                    handoff=act in WRITE_ACTIONS,
                ),
                lang,
            )
        if act == "status" and err_code == "no_match":
            return ToolResult(
                data={"error": "no_match", "device": device},
                speak="",
                error="no_match",
                handoff=True,
            )
        # H1: a write whose named device is unknown speaks "Nie znam takiego urządzenia" and hands
        # the turn to Hermes (it may still know the device) instead of guessing locally.
        write_unknown = act in WRITE_ACTIONS and err_code == "no_match" and bool(utterance)
        return _finalize(
            ToolResult(
                data={"error": err_code, "device": device},
                speak=resolution.ask_back or "O które urządzenie chodzi?",
                error=err_code,
                handoff=write_unknown,
            ),
            lang,
        )

    entity_id = resolution.entity_id
    domain = entity_id.split(".", 1)[0]
    if act in WRITE_ACTIONS and (
        domain not in WRITABLE_DOMAINS or _is_boiler_entity(entity_id, cfg.home.boiler_id_substrings)
    ):
        # A write aimed at a read-only device (boiler, hot water, sensors) is refused before any
        # Home Assistant call; an exactly allowlisted boiler entity still goes through the gates below.
        service = {"turn_on": "turn_on", "turn_off": "turn_off"}.get(act, act)
        if not is_write_allowed(entity_id, domain, service, allowlist, cfg.home.boiler_id_substrings):
            return _finalize(
                _write_refusal_result(entity_id, domain, service, lang, cfg.home.boiler_id_substrings), lang
            )
    completed_writes: list[str] = []
    current_write_service = ""
    # F3: the utterance co-named read-only boiler(s) next to the resolved writable target. The
    # allowed write below proceeds; the final reply appends the boiler refusal so nothing the
    # user asked for is dropped silently.
    read_only_refused = list(dict.fromkeys(resolution.read_only_named))
    refusal_notes = " ".join(
        _write_refusal_speak(eid, lang, cfg.home.boiler_id_substrings)
        for eid in read_only_refused
    )

    nom_fallback, acc_fallback, gen_fallback = _spoken_forms(
        entity_id, {}, cfg.home.aliases, cfg.home.spoken_forms
    )

    try:
        async with httpx.AsyncClient(timeout=cfg.home.timeout_s, transport=transport) as client:
            state_resp = await _get_state(client, base_url, headers, entity_id)
            if state_resp.status_code == 404:
                if act == "status":
                    return ToolResult(
                        data={"error": "entity_not_found", "entity_id": entity_id},
                        speak="",
                        error="entity_not_found",
                        handoff=True,
                    )
                # Never read the raw entity_id out loud; name the device the way the household does.
                speak = (
                    "Nie znalazłam tego urządzenia w Home Assistant."
                    if lang == "pl"
                    else "I could not find that device in Home Assistant."
                )
                return _finalize(
                    ToolResult(data={"error": "entity_not_found", "entity_id": entity_id}, speak=speak, error="entity_not_found"),
                    lang,
                )
            if state_resp.status_code != 200:
                speak = "Nie udało się odczytać stanu urządzenia z Home Assistant." if lang == "pl" else "Failed to read Home Assistant state."
                return _finalize(
                    ToolResult(data={"error": "ha_http", "status": state_resp.status_code}, speak=speak, error="ha_http"),
                    lang,
                )

            initial_obj = state_resp.json()
            if act == "status":
                speak = _speak_status(
                    entity_id, initial_obj, cfg.home.aliases, lang, cfg.home.spoken_forms
                )
                return _finalize(
                    ToolResult(
                        data={
                            "entity_id": entity_id,
                            "state": initial_obj.get("state"),
                            "attributes": _sanitize_attributes(initial_obj.get("attributes")),
                        },
                        speak=speak,
                    ),
                    lang,
                )

            # ---- State-changing actions ----
            initial_state = str(initial_obj.get("state") or "")
            initial_attrs = initial_obj.get("attributes") or {}
            nom_fallback, acc_fallback, gen_fallback = _spoken_forms(
                entity_id, initial_obj, cfg.home.aliases, cfg.home.spoken_forms
            )
            clamped_temp: int | float | None = None
            was_clamped = False
            mapped_mode: str | None = None
            pct_target: int | None = None
            planned_writes: list[tuple[str, str, dict[str, Any]]] = []

            if domain == "climate":
                if act == "turn_off":
                    planned_writes.append(("climate", "turn_off", {"entity_id": entity_id}))

                elif act in ("turn_on", "set_temperature", "set_hvac_mode"):
                    if act == "set_temperature" and temperature is None:
                        speak = "Podaj temperaturę w stopniach." if lang == "pl" else "Please specify a temperature."
                        return _finalize(
                            ToolResult(data={"error": "missing_temperature"}, speak=speak, error="missing_temperature"),
                            lang,
                        )
                    if act == "set_hvac_mode" and not hvac_mode:
                        speak = "Podaj tryb klimatyzacji, na przykład chłodzenie albo grzanie."
                        return _finalize(
                            ToolResult(data={"error": "missing_hvac_mode"}, speak=speak, error="missing_hvac_mode"),
                            lang,
                        )

                    if hvac_mode:
                        raw_mode = _fold(hvac_mode.strip())
                        mapped_mode = HVAC_MODE_ALIASES.get(raw_mode, raw_mode)
                        supported_modes = [str(m) for m in (initial_attrs.get("hvac_modes") or [])]
                        if mapped_mode == "auto" and "auto" not in supported_modes and "heat_cool" in supported_modes:
                            mapped_mode = "heat_cool"
                        if supported_modes and mapped_mode not in supported_modes:
                            speak = f"Tryb {hvac_mode} nie jest obsługiwany przez to urządzenie."
                            return _finalize(
                                ToolResult(
                                    data={"error": "unsupported_hvac_mode", "supported": supported_modes},
                                    speak=speak,
                                    error="unsupported_hvac_mode",
                                ),
                                lang,
                            )

                    if temperature is not None and mapped_mode != "off":
                        # Check allowlist before sanity range so read-only boiler entities still return write_not_allowed
                        if not is_write_allowed(
                            entity_id, "climate", "set_temperature", allowlist, cfg.home.boiler_id_substrings
                        ):
                            return _finalize(
                                _write_refusal_result(
                                    entity_id, "climate", "set_temperature", lang, cfg.home.boiler_id_substrings
                                ),
                                lang,
                            )
                        t_val = float(temperature)
                        if t_val < AC_SANITY_MIN_C or t_val > AC_SANITY_MAX_C:
                            speak = (
                                "Podaj proszę temperaturę między 16 a 30 stopni — jaką ustawić?"
                                if lang == "pl"
                                else "Please specify an AC temperature between 16 and 30 degrees — what should I set?"
                            )
                            return _finalize(
                                ToolResult(
                                    data={
                                        "error": "temperature_out_of_range",
                                        "requested": t_val,
                                        "min": AC_SANITY_MIN_C,
                                        "max": AC_SANITY_MAX_C,
                                    },
                                    speak=speak,
                                    error="temperature_out_of_range",
                                ),
                                lang,
                            )
                        clamped_temp, was_clamped = clamp_temperature(t_val, initial_attrs)

                    if mapped_mode is not None:
                        planned_writes.append(
                            ("climate", "set_hvac_mode", {"entity_id": entity_id, "hvac_mode": mapped_mode})
                        )
                    elif act == "turn_on" or (act == "set_temperature" and initial_state == "off"):
                        planned_writes.append(("climate", "turn_on", {"entity_id": entity_id}))

                    if clamped_temp is not None and mapped_mode != "off":
                        planned_writes.append(
                            ("climate", "set_temperature", {"entity_id": entity_id, "temperature": clamped_temp})
                        )
                else:
                    speak = "Ta akcja nie jest dostępna dla klimatyzacji."
                    return _finalize(
                        ToolResult(data={"error": "unsupported_action", "action": act}, speak=speak, error="unsupported_action"),
                        lang,
                    )

            elif domain == "media_player":
                if act in ("turn_on", "turn_off", "volume_up", "volume_down"):
                    planned_writes.append(("media_player", act, {"entity_id": entity_id}))
                elif act == "volume_set":
                    if level is None:
                        speak = "Podaj poziom głośności od zera do stu."
                        return _finalize(
                            ToolResult(data={"error": "missing_level"}, speak=speak, error="missing_level"),
                            lang,
                        )
                    pct_target = max(0, min(100, int(round(float(level)))))
                    vol_float = round(pct_target / 100.0, 2)
                    planned_writes.append(
                        ("media_player", "volume_set", {"entity_id": entity_id, "volume_level": vol_float})
                    )
                elif act in ("mute", "unmute"):
                    planned_writes.append(
                        ("media_player", "volume_mute", {"entity_id": entity_id, "is_volume_muted": act == "mute"})
                    )
                else:
                    speak = "Ta akcja nie jest dostępna dla odtwarzacza."
                    return _finalize(
                        ToolResult(data={"error": "unsupported_action", "action": act}, speak=speak, error="unsupported_action"),
                        lang,
                    )

            else:
                service_map = {
                    "turn_on": "turn_on",
                    "turn_off": "turn_off",
                    "set_temperature": "set_temperature",
                }
                svc_name = service_map.get(act, act)
                if not is_write_allowed(
                    entity_id, domain, svc_name, allowlist, cfg.home.boiler_id_substrings
                ):
                    return _finalize(
                        _write_refusal_result(entity_id, domain, svc_name, lang, cfg.home.boiler_id_substrings), lang
                    )
                speak = "Sterowanie tym typem urządzenia nie jest obsługiwane lokalnie."
                return _finalize(
                    ToolResult(data={"error": "unsupported_domain", "entity_id": entity_id}, speak=speak, error="unsupported_domain"),
                    lang,
                )

            # Pre-authorize ALL planned writes before mutating any state on Home Assistant
            for dom, svc, _ in planned_writes:
                if not is_write_allowed(entity_id, dom, svc, allowlist, cfg.home.boiler_id_substrings):
                    return _finalize(
                        _write_refusal_result(entity_id, dom, svc, lang, cfg.home.boiler_id_substrings), lang
                    )

            # Execute writes in order, reporting truthfully if a multi-step sequence partially completes
            for dom, svc, payload in planned_writes:
                current_write_service = f"{dom}.{svc}"
                safe_args = {k: v for k, v in payload.items() if k != "entity_id"}
                log.info("home write: entity=%s service=%s.%s args=%s", entity_id, dom, svc, safe_args)
                resp = await client.post(f"{base_url}/api/services/{dom}/{svc}", headers=headers, json=payload)
                if resp.status_code != 200:
                    if completed_writes:
                        speak = _speak_partial_write(completed_writes, current_write_service, acc_fallback, gen_fallback, lang, nom=nom_fallback)
                        return _finalize(
                            ToolResult(
                                data={
                                    "error": "partial_write",
                                    "entity_id": entity_id,
                                    "completed": list(completed_writes),
                                    "failed": current_write_service,
                                    "status": resp.status_code,
                                },
                                speak=speak,
                                error="partial_write",
                            ),
                            lang,
                        )
                    return _finalize(
                        ToolResult(
                            data={"error": "service_failed", "entity_id": entity_id, "status": resp.status_code},
                            speak="Home Assistant odrzucił polecenie sterowania urządzeniem.",
                            error="service_failed",
                        ),
                        lang,
                    )
                completed_writes.append(current_write_service)
            current_write_service = ""

            # ---- Mandatory post-write verification (poll up to ~1.5 s for Daikin/Onkyo lag) ----
            verified: dict[str, Any] | None = None
            matched = False
            for attempt in range(VERIFY_POLL_RETRIES + 1):
                if attempt > 0 and verify_delay_s > 0:
                    await asyncio.sleep(verify_delay_s)
                verify_resp = await _get_state(client, base_url, headers, entity_id)
                if verify_resp.status_code != 200:
                    continue
                verified = verify_resp.json()
                cand_state = str(verified.get("state") or "unknown")
                cand_attrs = verified.get("attributes") or {}
                if _state_matches_write(
                    domain=domain,
                    action=act,
                    v_state=cand_state,
                    v_attrs=cand_attrs,
                    initial_attrs=initial_attrs,
                    clamped_temp=clamped_temp,
                    mapped_mode=mapped_mode,
                    pct_target=pct_target,
                ):
                    matched = True
                    break

            if verified is None:
                speak = "Wysłałam polecenie, ale nie udało się potwierdzić stanu urządzenia."
                return _finalize(
                    ToolResult(data={"error": "verify_failed", "entity_id": entity_id}, speak=speak, error="verify_failed"),
                    lang,
                )

            v_state = str(verified.get("state") or "unknown")
            v_attrs = verified.get("attributes") or {}
            clean_v_attrs = _sanitize_attributes(v_attrs)
            nom, acc, gen = _spoken_forms(entity_id, verified, cfg.home.aliases, cfg.home.spoken_forms)

            if v_state in ("unavailable", "unknown"):
                speak = f"Urządzenie {nom} jest niedostępne, nie udało się potwierdzić zmiany."
                return _finalize(
                    ToolResult(
                        data={"error": "verify_unavailable", "entity_id": entity_id, "state": v_state},
                        speak=speak,
                        error="verify_unavailable",
                    ),
                    lang,
                )

            result_data: dict[str, Any] = {
                "entity_id": entity_id,
                "action": act,
                "state": v_state,
                "attributes": clean_v_attrs,
            }
            if read_only_refused:
                result_data["read_only_refused"] = read_only_refused
            if was_clamped:
                result_data["clamped"] = True
                result_data["clamped_temperature"] = clamped_temp

            if not matched:
                speak = (
                    "Wysłałam polecenie, ale urządzenie jeszcze tego nie potwierdziło."
                    if lang == "pl"
                    else "Command sent, but the device has not confirmed the change yet."
                )
                if refusal_notes:
                    speak = f"{speak} {refusal_notes}"
                return _finalize(
                    ToolResult(
                        data={**result_data, "error": "verify_mismatch"},
                        speak=speak,
                        error="verify_mismatch",
                    ),
                    lang,
                )

            if domain == "climate":
                real_temp = _format_temp_pl(v_attrs.get("temperature"))
                mode_pl = HVAC_MODE_LABELS_PL.get(v_state, v_state)
                # F7: an empty acc/gen slot is an unknown form — speak a neutral, case-free
                # shape (impersonal or nominal) instead of a guessed case.
                if act == "turn_off" or (act == "set_hvac_mode" and mapped_mode == "off"):
                    speak = f"Wyłączyłam {acc}." if acc else f"{_cap_first(nom)}: wyłączono."
                elif act == "turn_on":
                    if acc:
                        speak = f"Włączyłam {acc}, {real_temp}." if real_temp else f"Włączyłam {acc}, tryb {mode_pl}."
                    else:
                        speak = f"{_cap_first(nom)}: włączono, {real_temp}." if real_temp else f"{_cap_first(nom)}: włączono, tryb {mode_pl}."
                elif act == "set_hvac_mode":
                    if acc:
                        speak = f"Ustawiłam {acc} na tryb {mode_pl}, {real_temp}." if real_temp else f"Ustawiłam {acc} na tryb {mode_pl}."
                    else:
                        speak = f"{_cap_first(nom)}: tryb {mode_pl}, {real_temp}." if real_temp else f"{_cap_first(nom)}: tryb {mode_pl}."
                else:  # set_temperature
                    confirmed_t = real_temp or _format_temp_pl(clamped_temp)
                    if was_clamped and confirmed_t:
                        speak = f"Ograniczyłam zakres i ustawiłam {acc} na {confirmed_t}." if acc else f"{_cap_first(nom)}: ograniczono zakres, ustawiono na {confirmed_t}."
                    elif confirmed_t:
                        speak = f"Ustawiłam {acc} na {confirmed_t}." if acc else f"{_cap_first(nom)}: ustawiono na {confirmed_t}."
                    else:
                        speak = f"Zmieniłam ustawienie {gen}." if gen else f"{_cap_first(nom)}: ustawienia zmienione."
            else:  # media_player
                if act == "turn_off":
                    speak = f"Wyłączyłam {acc}." if acc else f"{_cap_first(nom)}: wyłączono."
                elif act == "turn_on":
                    speak = f"Włączyłam {acc}." if acc else f"{_cap_first(nom)}: włączono."
                elif act in ("volume_set", "volume_up", "volume_down"):
                    v_lvl = v_attrs.get("volume_level")
                    if isinstance(v_lvl, (int, float)):
                        pct = int(round(float(v_lvl) * 100))
                        pct_w = polish_declension(pct, "procent", "procenty", "procent")
                        speak = f"Ustawiłam głośność {gen} na {pct} {pct_w}." if gen else f"Ustawiono głośność na {pct} {pct_w}."
                    else:
                        speak = f"Zmieniłam głośność {gen}." if gen else f"{_cap_first(nom)}: głośność zmieniona."
                elif act == "mute":
                    speak = f"Wyciszyłam {acc}." if acc else f"{_cap_first(nom)}: wyciszono."
                else:  # unmute
                    speak = f"Wyłączyłam wyciszenie {gen}." if gen else f"{_cap_first(nom)}: wyciszenie wyłączone."

            if refusal_notes:
                speak = f"{speak} {refusal_notes}"

            return _finalize(ToolResult(data=result_data, speak=speak), lang)

    except httpx.TimeoutException:
        log.warning("Home Assistant request timed out for %s", entity_id)
        if completed_writes:
            failed = current_write_service or "verify"
            if failed == "verify":
                speak = "Wysłałam polecenie, ale nie udało się potwierdzić stanu urządzenia na czas."
            else:
                speak = _speak_partial_write(completed_writes, failed, acc_fallback, gen_fallback, lang, nom=nom_fallback)
            return _finalize(
                ToolResult(
                    data={
                        "error": "partial_write",
                        "entity_id": entity_id,
                        "completed": list(completed_writes),
                        "failed": failed,
                    },
                    speak=speak,
                    error="partial_write",
                ),
                lang,
            )
        speak = "Home Assistant nie odpowiedział na czas." if lang == "pl" else "Home Assistant timed out."
        return _finalize(ToolResult(data={"error": "timeout", "entity_id": entity_id}, speak=speak, error="timeout"), lang)
    except httpx.HTTPError as exc:
        log.warning("Home Assistant HTTP error for %s: %s", entity_id, exc)
        if completed_writes:
            failed = current_write_service or "verify"
            speak = _speak_partial_write(completed_writes, failed, acc_fallback, gen_fallback, lang, nom=nom_fallback)
            return _finalize(
                ToolResult(
                    data={
                        "error": "partial_write",
                        "entity_id": entity_id,
                        "completed": list(completed_writes),
                        "failed": failed,
                    },
                    speak=speak,
                    error="partial_write",
                ),
                lang,
            )
        speak = "Nie udało się połączyć z Home Assistant." if lang == "pl" else "Could not connect to Home Assistant."
        return _finalize(ToolResult(data={"error": "network_error", "entity_id": entity_id}, speak=speak, error="network_error"), lang)
