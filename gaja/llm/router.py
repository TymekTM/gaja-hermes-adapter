from __future__ import annotations

import ast
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from collections.abc import Callable

from gaja.llm.client import LLMClient
from gaja.llm.parse import ToolCall
from gaja.llm.prompt import load_text_block
from gaja.devices.volume import DEFAULT_STEP as VOLUME_DEFAULT_STEP

PROMISE_PL = re.compile(
    r"\b(?:przeka[zż]\w*|zlec\w*|przesy?ł\w*|wysy?ł\w*|poprosz\w*|zapytam)\s+"
    r"(?:(?:to|ci|mu|tę|tego|zadanie|sprawę|temat|do|teraz|od\s+razu|już)\s+){0,3}hermes(?:owi|a|em|ie|u)?\b"
    r"|\bhermes\s+(?:już\s+)?(?:to\s+)?(?:sprawdza|sprawdzi|zajmie\s+się|się\s+tym\s+zajmie|jest\s+na\s+tym)"
    r"|\b(?:sprawdza|sprawdzi|zajmie\s+się|zajmuje\s+się)\s+(?:tym\s+|to\s+)?(?:już\s+)?hermes\b",
    re.IGNORECASE,
)
PROMISE_EN = re.compile(
    r"\b(?:hand|pass|send|forward)\w*\s+(?:(?:it|this|that|the\s+task)\s+)?(?:(?:over|off|along)\s+)?to\s+hermes\b"
    r"|\b(?:i'?ll|let\s+me|i\s+will)\s+ask\s+hermes\b|\bhermes\s+(?:is\s+on\s+it|will\s+(?:check|handle))",
    re.IGNORECASE,
)
# "Hermes" as the god, the brand or the subject of a question, not as the agent being addressed.
MYTHOLOGY = re.compile(
    r"mitolog|mytholog|\bbog(?:a|ów|iem|u)?\b|\bbóg\b|\bgod\b|olimp|olymp|zeus|posłaniec\s+bog|messenger\s+of|"
    r"grec|greek|herm[eè]s\s+paris|torebk|perfum|o\s+hermesie|about\s+hermes|kim\s+(?:jest|był)\s+hermes|"
    r"who\s+(?:is|was)\s+hermes",
    re.IGNORECASE,
)
EXPLICIT_HERMES = re.compile(
    r"\b(?:zapytaj|spytaj|popro[sś]|każ|zleć|przeka[zż]\w*|niech)\s+(?:\w+\s+){0,2}hermes\w*"
    r"|^\W*hermes(?:ie)?\s*[,:!]|(?:\b(?:ask|tell|have|let)\s+hermes\b)|\bhermes\s*,\s*(?:please|proszę)"
    r"|\b(?:w|przez|użyj|use|via|using|in)\s+hermes(?:ie|a|em)?\b",
    re.IGNORECASE,
)

# Imperatives only Hermes can carry out (write/send/store/control/search). In 3-class mode they win
# over TOOL and LOCAL, since the small model tends to file "napisz skrypt…" or "włącz światło…"
# under a tool. Creative asks ("napisz wierszyk") stay local.
HERMES_ACTION = re.compile(
    r"\b(?:napisz|wyślij|wyslij|odpisz|zapisz|zapamiętaj|zapamietaj|dodaj|usuń|usun|ustaw|włącz|wlacz|wyłącz|wylacz|"
    # "zamknij/otwórz" are Hermes jobs for objects (rolety, brama), never reflexive "zamknij się"
    # or "otwórz się sezamie" — those stay local as on main (NEW-12).
    r"zamknij(?!\s+si[eę])|otw[oó]rz(?!\s+si[eę])|otworz(?!\s+si[eę])|przyciemnij|rozjaśnij|zgaś|zgas|zamów|zamow|zarezerwuj|znajdź|znajdz|wyszukaj|poszukaj|porównaj|porownaj|"
    r"zaplanuj|przygotuj)\b"
    r"(?!\s+(?:mi\s+)?(?:\w+\s+)?(?:wiersz\w*|wierszyk\w*|haiku|limeryk\w*|rym\w*|żart\w*|zart\w*|dowcip\w*|zagadk\w*))"
    r"|\bprzypomnij\s+mi\b(?!\s+co)"
    r"|\b(?:write|draft)\s+(?:me\s+)?(?:an?\s+)?(?:\w+\s+){0,3}(?:script|program|function|code|e-?mail|letter|essay|contract|report)"
    r"|\b(?:send|remind\s+me|set\s+an?\s+(?:alarm|reminder|timer)|turn\s+(?:on|off)|schedule|book\s+a|order|"
    r"search\s+(?:the\s+web|online|for)|find\s+me)\b",
    re.IGNORECASE,
)

# Home sensors and devices without a local tool go to Hermes, even when the words look like
# weather ("temperatura w salonie").
# ---- Home cues whose "u <person>" places come from the household aliases (NEW-6) -----------------------
# Person names never live in this file: the server derives the spoken "u <word>" places from the
# loaded home aliases ("klima u Ani" -> "u ani") and injects them at start-up. With no household
# aliases configured the person alternation never matches anything.
_PERSON_NONE = r"(?!)"  # never matches


def _person_place(person_alt: str) -> str:
    return rf"\bu\s+(?:{person_alt})\b" if person_alt else _PERSON_NONE


def _home_cue_pattern(person_alt: str) -> str:
    # ASCII-folded device nouns included (L3 + r5 6d): "wlacz klime u rodzicow" must cue home too.
    person = _person_place(person_alt)
    return "|".join(
        [
            r"\bklim(?:a|ę|e|y|ie)\b|klimatyzacj|ch[lł]odzeni|ogrzewani|grzani|nawiew|osuszani|hvac",
            r"\btelewizor\w*|\btv\b|\bamplituner\w*|\bonkyo\b|\bbravia\b",
            r"\bpiec(?:a|u|owi|em|ie)\b|\b(?:jaki\s+)?(?:stan|sprawd[zź])\s+piec(?:a|u)?\b|\bpiec\b\s+(?:grzeje|dzia[lł]a|pracuje|jest|nie)\b",
            r"\bkocio[lł]\w*|\bcwu\b|ciep[lł]\w+\s+wod",
            r"(?:temperatur\w*|ile\s+stopni|ciep[lł]o|zimno)\s+(?:jest\s+)?(?:dzisiaj\s+|teraz\s+)?"
            r"(?:w\s+(?:domu|sypialni|salonie|pokoju)|" + person + r")",
            person,
            r"\bw\s+sypialni\b.*\bstopni",
        ]
    )


def _home_override_pattern(person_alt: str) -> str:
    # ASCII-folded device nouns included (r5 6d): "klime" folds onto the same override as "klimę".
    person = _person_place(person_alt)
    rooms = r"(?:sypialni|salonie|pokoju|kuchni|[lł]azience)"
    place = rf"(?:{person}|w\s+{rooms})"
    return "|".join(
        [
            r"\bklim(?:a|ę|e|y|ie|atyzacj\w*)\b|\btelewizor\w*|\btv\b|\bamplituner\w*|\bonkyo\b|\bbravia\b",
            r"\bpiec(?:a|u|owi|em|ie)?\b|\bkocio[lł]\w*|\bcwu\b|\bnawiew\w*|\bosuszacz\w*|\bhvac\b",
            r"\b(?:ogrzewani\w*|grzani\w*|ch[lł]odzeni\w*)\b[^.!?\n]*\b" + place + r"\b",
            r"\b" + place + r"\b[^.!?\n]*\b(?:ogrzewani\w*|grzani\w*|ch[lł]odzeni\w*)\b",
        ]
    )


def home_person_words(aliases: dict[str, str] | None) -> tuple[str, ...]:
    """The household's spoken 'u <person/place>' words, derived from its home aliases
    ("klima u Ani", "u rodziców" -> ("ani", "rodziców")); empty without household aliases."""
    words: list[str] = []
    for alias in aliases or {}:
        for match in re.finditer(r"\bu\s+(\S+)", alias, re.IGNORECASE):
            word = match.group(1).strip(",.!?").lower()
            if word and word not in words:
                words.append(word)
    return tuple(words)


def _fold_ascii(text: str) -> str:
    text = text.lower().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def set_home_person_words(words: tuple[str, ...] | list[str] | None) -> None:
    """Rebuild the person-dependent home cues from the household's alias words (NEW-6).

    Called at start-up (server.py) with ``home_person_words(settings.home.aliases)`` and by tests;
    ``None``/empty restores the name-free defaults. Cues also match ASCII-folded forms (L3).
    """
    folded: list[str] = []
    for word in words or ():
        folded_word = str(word).strip().lower()
        if folded_word and folded_word not in folded:
            folded.append(folded_word)
        asc = _fold_ascii(folded_word)
        if asc and asc not in folded:
            folded.append(asc)
    person_alt = "|".join(re.escape(w) for w in folded)
    TOOL_CUES["home"] = re.compile(_home_cue_pattern(person_alt), re.IGNORECASE)
    globals()["HOME_OVERRIDE_DEVICE"] = re.compile(_home_override_pattern(person_alt), re.IGNORECASE)


# A selected tool only runs if the utterance (or, for follow-ups like "a w Krakowie?", the previous
# user/assistant turn) mentions what the tool is about; otherwise the small model's pick is noise (e.g.
# "kiedy była bitwa pod Grunwaldem" -> get_weather) and the turn is answered locally.
TOOL_CUES: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "get_time": r"godzin|czas|dat[aęy]|dzie[nń]\b|dni\b|dzi[sś]|kt[oó]ry\s+(?:dzi|jest)|ile\s+dni|ile\s+(?:zosta[lł]o\s+)?dni|jaki\s+dzie[nń]|time|date|\bday\b|today|weekday|clock",
        "get_weather": r"pogod|deszcz|pada|padać|padac|temperatur|stopni|s[lł]o[nń]c|wiatr|[sś]nieg|burz|mr[oó]z|zimno|ciep[lł]o|"
        r"prognoz|parasol|pochmurn|weather|rain|snow|sun|temperature|forecast|degrees|wind|cold|warm|hot|"
        # "jak będzie w Zakopanem": a capitalised place only, so "jak jest w pracy" stays local.
        r"jak\s+(?:tam\s+)?(?:będzie|bedzie|jest|jutro|dzi[sś]|dzisiaj)\s+(?:w|we|na)\s+(?-i:[A-ZĄĆĘŁŃÓŚŹŻ])",
        "get_sun": r"wsch[oó]d|zach[oó]d|d[lł]ugo[sś][cć]\s+dnia|kiedy\s+wschodzi|kiedy\s+zachodzi|ile\s+trwa\s+dzie[nń]|sunrise|sunset|daylight",
# Digits only: the selector garbles spelled-out numbers ("dwadzieścia plus trzydzieści pięć" -> 20 + 33).
        "calculate": r"\d",
        "convert_units": r"\d|kilo|metr|mil[eia]?\b|stop[ay]|cal[ei]?\b|litr|galon|funt|uncj|stopni|celsjusz|fahrenheit|kelvin|"
        r"szklank|łyżk|lyzk|bajt|bit|byte|przelicz|ile\s+\w+\s+(?:to|ma)|convert|pound|ounce|inch|feet|foot|mile|gallon|cup|tbsp|tsp",
        "exchange_rate": r"kurs|walut|euro|dolar|funt|frank|z[lł]ot|pln|eur\b|usd|gbp|chf|currency|exchange|dollar|pound|franc|zloty",
        "wiki_summary": r"kim\s+(?:jest|by[lł])|kto\s+to|co\s+to|opowiedz|wikipedi|podsumuj|who\s+(?:is|was)|what\s+is|summary|tell\s+me\s+about",
        "server_status": r"serwer|server|gpu|ram\b|pami[eę]ć|obci[aą][zż]|uptime|dysk|procesor|cpu|us[lł]ug|system|maszyn|disk|load|memory|"
        r"karta\s+graficzn|karty\s+graficzn",
        "next_holiday": r"[sś]wi[eę]t|wolne|wolny|holiday|urlop|ferie|d[lł]ugi\s+weekend|bo[zż]\w*\s+narodzeni\w*|gwiazdk\w*|wigili\w*|wielkanoc\w*|nowy\s+rok|nowego\s+roku|sylwest\w*|trzech\s+kr[oó]l\w*|bo[zż]\w*\s+cia[lł]\w*|wszystkich\s+[sś]wi[eę]t\w*|niepodleg[lł]\w*|konstytucj\w*|maj[oó]wk\w*",
        "manage_timer": r"minutnik|stoper|przypomn|budzik|ile\s+zosta[lł]o|anuluj\s+minutnik|timer|reminder",
        "remember": r"zapami[eę]taj|pami[eę]taj|remember|zanotuj",
        "forget": r"zapomnij|wykasuj|usu[nń]\s+z\s+pami[eę]ci|forget",
        "set_speaker_volume": r"cisz|gło[sś]|glos|wycisz|podgło[sś]|podglos|volume|loud|quiet|mute",
        "spotify": (
            r"\bspotify\b"
            r"|\b(?:pu[sś][cć]|pusc)\b"
            r"|\b(?:zagraj(?!my)|odtw[oó]rz|odtworz|wzn[oó]w|wznow|zatrzymaj|wstrzymaj|spauzuj|pomi[nń]|pomin|cofnij)\b"
            r"[^.!?\n]*\b(?:muzyk\w*|piosenk\w*|utw[oó]r\w*|utwor\w*|kawa[lł]ek\w*|kawalek\w*|playlist\w*|album\w*|radi\w*|przeb[oó]j\w*|przeboj\w*)\b"
            r"|\b(?:muzyk\w*|piosenk\w*|utw[oó]r\w*|utwor\w*|kawa[lł]ek\w*|kawalek\w*|playlist\w*|album\w*|radi\w*|przeb[oó]j\w*|przeboj\w*)\b"
            r"[^.!?\n]*\b(?:zagraj(?!my)|odtw[oó]rz|odtworz|wzn[oó]w|wznow|zatrzymaj|wstrzymaj|spauzuj|pomi[nń]|pomin|cofnij)\b"
            r"|\b(?:nast[eę]pn\w*|poprzedni\w*|kolejn\w*)\s+(?:piosenk\w*|utw[oó]r\w*|utwor\w*|kawa[lł]ek\w*|kawalek\w*)"
            r"|\b(?:co\s+(?:teraz\s+)?gra|jaka\s+to\s+piosenka|jaki\s+to\s+utw[oó]r|co\s+leci)\b"
            r"|\b(?:muzyk[ęei]|muzyke|muzyki)\s+(?:troch[ęe]\s+|troszk[ęe]\s+|du[żz]o\s+|bardzo\s+)?(?:g[lł]o[sś]niej|glosniej|ciszej|[sś]cisz|scisz|przycisz|podg[lł]o[sś]\w*|na\s+\d)"
            r"|\b(?:g[lł]o[sś]niej|glosniej|ciszej|[sś]cisz|scisz|przycisz|podg[lł]o[sś]\w*|g[lł]o[sś]no[sś][cć]|glosnosc|w[lł][aą]cz|wlacz|wy[lł][aą]cz|wylacz)\s+(?:troch[ęe]\s+|du[żz]o\s+)?(?:muzyk[ęei]|muzyke|muzyki|piosenk\w*|utw[oó]r\w*|playlist\w*)"
        ),
        # The home cue: device/room nouns plus the household's own "u <person>" places. The person
        # alternation is derived from the loaded home aliases at start-up (set_home_person_words,
        # NEW-6) — no person names are hard-coded here; empty by default (no household file).
        "home": _home_cue_pattern(""),
    }.items()
}
FAST_PATH_TIME = re.compile(
    r"^\s*(?:hej\s+gaja[,\s]+|gaja[,\s]+)?(?:przepraszam\s*,?\s*)?"
    r"(?:kt[oó]ra\s+(?:jest\s+)?(?:teraz\s+)?godzina(?:\s+jest)?|"
    r"jaka\s+(?:jest\s+)?(?:teraz\s+)?godzina|"
    r"podaj\s+(?:mi\s+)?godzin[eę]|"
    r"jaki\s+(?:mamy\s+)?czas|"
    r"what\s+(?:time\s+is\s+it|is\s+the\s+time))\s*[?!.]*\s*$",
    re.IGNORECASE,
)

FAST_PATH_DATE = re.compile(
    r"^\s*(?:hej\s+gaja[,\s]+|gaja[,\s]+)?(?:przepraszam\s*,?\s*)?"
    r"(?:jaki\s+(?:jest\s+|mamy\s+)?(?:dzi[sś]|dzisiaj)?\s*(?:jest\s+)?dzie[nń](?:\s+tygodnia)?|"
    r"jaka\s+(?:jest\s+)?(?:dzi[sś]|dzisiaj)?\s*data|"
    r"kt[oó]ry\s+(?:(?:dzi[sś]|dzisiaj)\s+jest|(?:jest\s+|mamy\s+)?(?:dzi[sś]|dzisiaj)(?:\s+dzie[nń])?)|"
    r"kt[oó]rego\s+(?:dzi[sś]|dzisiaj\s+)?mamy|"
    r"what\s+(?:day\s+is\s+today|is\s+(?:today'?s\s+date|the\s+date)))\s*[?!.]*\s*$",
    re.IGNORECASE,
)

# ---- device volume fast path -------------------------------------------------------------------------
# "Hej Gaja, ciszej" / "głośność 50" / "dużo głośniej" / "wycisz" / "na maksa" (and English) are answered
# locally with zero LLM tokens: the wake phrase is already stripped on voice turns, so these must match
# whole utterances only (the optional "Gaja," prefix covers typed text). Diacritics-free spellings are
# matched too ("glosnosc 50"). Spelled-out numbers are not parsed: the explicit amount is digits only,
# and a bare amount needs a percent unit ("50 procent") so a plain number is never mistaken for volume.
_VOL_PREFIX = r"^\s*(?:hej\s+gaja\s*[,:!]\s*|hey\s+gaja\s*[,:!]\s*|gaja\s*[,:!]\s*)?"
_VOL_TAIL = r"\s*[?!.]*\s*$"
_VOL_AMOUNT_OPT = r"(?P<amount>\d{1,3})\s*(?:%|procent\w*|percent)?"
_VOL_AMOUNT_REQ = r"(?P<amount>\d{1,3})\s*(?:%|procent\w*|percent)"
_VOL_BY = r"(?:\s*(?:by|o)\s*" + _VOL_AMOUNT_OPT + r")?"
_VOL_SMALL = r"(?:trochę|troszkę|lekko|odrobinę|a\s+bit|a\s+little|slightly)\s*"
_VOL_BIG = r"(?:dużo|duzo|bardzo|mocno|znacznie|much|a\s+lot|way\s+|far\s+)"
_VOL_DOWN = (
    r"(?:ciszej|cisz|przycisz|przyciszaj|[sś]cisz|[sś]ciszaj|quieter|softer|"
    r"volume\s+down(?:\s+by)?|lower\s+the\s+volume|turn\s+(?:it|the\s+volume)\s+down)"
)
_VOL_UP = (
    r"(?:głośniej|glosniej|podgłośnij|podglosnij|podgłoś|podglos|louder|"
    r"volume\s+up(?:\s+by)?|raise\s+the\s+volume|turn\s+(?:it|the\s+volume)\s+up)"
)
FAST_PATH_VOLUME_MUTE = re.compile(
    _VOL_PREFIX
    + r"(?:wycisz(?:aj)?|mute|przycisz\s+całkowicie|całkowicie\s+(?:wycisz|przycisz))"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_VOLUME_MAX = re.compile(
    _VOL_PREFIX
    + r"(?:(?:ustaw(?:iaj)?\s+|set\s+)?(?:na\s+)?(?:maksa|maks(?:imum)?|maksymalnie)"
    r"|maksymaln[ae]\s+(?:głośność|glosnosc|volume)"
    r"|(?:ustaw(?:iaj)?\s+)?(?:głośność|glosnosc)\s+na\s+maks\w*|(?:full|max(?:imum)?)\s+volume)",
    re.IGNORECASE,
)
FAST_PATH_VOLUME_DOWN = re.compile(
    _VOL_PREFIX + r"(?:" + _VOL_SMALL + r")?(?:" + _VOL_BIG + r")?\s*" + _VOL_DOWN + _VOL_BY + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_VOLUME_UP = re.compile(
    _VOL_PREFIX + r"(?:" + _VOL_SMALL + r")?(?:" + _VOL_BIG + r")?\s*" + _VOL_UP + _VOL_BY + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_VOLUME_SET = re.compile(
    _VOL_PREFIX
    + r"(?:(?:ustaw(?:iaj)?|set)\s+)?"
    r"(?:głośność|glosnosc|volume|dźwięk|dzwiek|sound)"
    r"\s*(?:na|do|to|at|on)?\s*"
    + _VOL_AMOUNT_OPT
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_VOLUME_BARE_PERCENT = re.compile(
    _VOL_PREFIX + r"(?:(?:na|do|to)\s+)?" + _VOL_AMOUNT_REQ + _VOL_TAIL, re.IGNORECASE
)

# Media (music/Spotify/PC playback): with no ready local spotify tool it is Hermes', even though
# the words overlap with the local speaker volume: the fast paths above drive only the Atom speaker.
# Rules (device log 2026-10-26):
#   - a playback verb with a music/Spotify object ("Puść coś na Spotify", "puść muzykę"),
#   - a volume word with music/Spotify or a playback verb ("Puść muzykę głośniej", "wycisz Spotify"),
#   - a volume command pointing at an object right after a Hermes result ("Daj to głośniej").
# A whole-utterance command ("głośniej", "dużo ciszej", "głośność 50") stays on the fast path: that is
# the Atom speaker, which the fast path is for. With a ready local spotify tool only the generic
# mood/genre/"coś" jobs and Hermes-session follow-ups stay with Hermes (see media_handoff).
MUSIC_MEDIA = re.compile(
    r"\b(?:muzyk\w*|spotify\w*|utw[oó]r\w*|piosenk\w*|playlist\w*|radi\w*|album\w*|przeb[oó]j\w*|music|song|track)\b",
    re.IGNORECASE,
)
# main@7741180's verb set. "zatrzymaj"/"wstrzymaj" were added with the pause fast path, but they must
# not widen the Hermes handoff: with no ready local tool "zatrzymaj minutnik"/"wstrzymaj timer" stay
# local as on main (NEW-2). The pause fast path spells its own verbs (FAST_PATH_SPOTIFY_PAUSE).
PLAYBACK_VERB = re.compile(
    r"\b(?:pu[śs][cć]\w*|zagraj(?!my)\w*|odtwarzaj\w*|odtw[oó]rz\w*|wzn[oó]w\w*|pauz\w*|"
    r"play|pause|resume)\b",
    re.IGNORECASE,
)
VOLUME_WORD = re.compile(
    r"\b(?:podgłoś\w*|podglos\w*|głośn\w*|glosn\w*|głosn\w*|cisz\w*|przycisz\w*|[sś]cisz\w*|wycisz\w*|"
    r"podkr[eę][cć]\w*|volume|louder|quieter|loud\w*|quiet\w*|mute\w*)",
    re.IGNORECASE,
)
VOLUME_OBJECT = re.compile(r"\b(?:to|tego|temu|go|ją|je|ich|tym)\b", re.IGNORECASE)

# M4 (round 5): up to two arbitrary words may sit between the play verb and a whole-word genre —
# "puść mi trochę jazzu", "puść stary rock", "puść polski rap", "puść ulubiony jazz" — and the
# adjectival genre forms count as the genre token itself ("puść coś jazzowego"). The earlier M4
# form (only "mi"/"trochę" fillers plus a fixed mood-adjective list) missed all of these.
# Deliberately still narrow: a gap word must not be the utterance's real object — film/serial/
# bajka/game/device nouns are excluded by lookahead — so "puść film Rocky", "puść film rockowy",
# "zagraj w punkty", "włącz klimę na punkt" and "włącz telewizor indie" never match (M4 round 4
# fixed exactly these; the arbitrary gap must not reopen them).
_M4_GAP_WORD = (
    r"(?:(?!(?:film\w*|serial\w*|bajk\w*|bajeczk\w*|kresk[oó]wk\w*|anim\w*|gr\w*|rozgrywk\w*|"
    r"poziom\w*|punk?t\w*|telewizor\w*|tv\b|klim\w*|amplituner\w*|wzmacniacz\w*|"
    r"g[lł]o[sś]nik\w*|piec\w*|kocio[lł]\w*|rolet\w*|[sś]wiat[lł]\w*|lamp\w*|ogrzewani\w*|muzyk\w*)\s)"
    r"(?:[a-ząćęłńóśźż]+\s+)){0,2}"
)
_GENRE_AFTER_PLAY_VERB = (
    r"\b(?:pu[sś][cć]|pusc|zagraj|odtw[oó]rz|odtworz|w[lł][aą]cz|wlacz)\s+"
    + _M4_GAP_WORD +
    r"(?:jazz(?:u|em|ie|owi)?|rock(?:a|u|em|owi|i)?|metal(?:u|em|owi)?|techno|hous(?:e|a|u|'u|'em|'owi)|"
    r"hip-?hop(?:u|em|a|owi)?|rap(?:u|a|em|owi)?|reggae|blues(?:a|u|em|owi|ie)?|lofi|lo-?fi|"
    r"chill(?:u|a|em)?|chillout|ambien(?:t|tu|tem|towi)?|punk(?:a|u|iem|owi)?|indie|"
    r"jazzow\w*|rockow\w*|rapow\w*|metalow\w*|popow\w*)\b"
)
GENERIC_PLAY_REQUEST = re.compile(
    r"\b(?:pu[sś][cć]|pusc|zagraj|odtw[oó]rz|odtworz|w[lł][aą]cz|wlacz)\b[^.!?\n]*\b(?:co[sś]|cokolwiek)\b"
    r"|\b(?:pu[sś][cć]|pusc|zagraj|odtw[oó]rz|odtworz|w[lł][aą]cz|wlacz)\s+(?:mi\s+)?jak[aą][sś]\b"
    r"|\bmuzyk\w*\s+(?:do\s+(?:pracy|snu|spania|biegania|treningu|nauki|relaksu|czytania|jazdy|imprezy|t[lł]a)|w\s+tle)\b"
    r"|\b(?:spokojn\w*|relaksuj\w*|energetyczn\w*|energiczn\w*|smutn\w*|weso[lł]\w*|romantyczn\w*|nastrojow\w*|"
    r"delikatn\w*|klasyczn\w*|elektroniczn\w*)\s+(?:muzyk\w*|piosenk\w*|co[sś])\b"
    r"|" + _GENRE_AFTER_PLAY_VERB,
    re.IGNORECASE,
)

# The only generic requests that may hand off to Hermes with no ready local tool where main's
# media_handoff said False (NEW-2): the ones that genuinely name music — a music noun tied to an
# activity or a mood, or a genre directly behind a play verb (M4). Timer, alarm, film, message and
# game objects never count; an indefinite "coś" without a music noun does not either ("zagraj mi
# coś na gitarze").
GENERIC_MUSIC_PLAY_REQUEST = re.compile(
    r"\bmuzyk\w*\s+(?:do\s+(?:pracy|snu|spania|biegania|treningu|nauki|relaksu|czytania|jazdy|imprezy|t[lł]a)|w\s+tle)\b"
    r"|\b(?:spokojn\w*|relaksuj\w*|energetyczn\w*|energiczn\w*|smutn\w*|weso[lł]\w*|romantyczn\w*|nastrojow\w*|"
    r"delikatn\w*|klasyczn\w*|elektroniczn\w*)\s+(?:muzyk\w*|piosenk\w*|playlist\w*)\b"
    r"|" + _GENRE_AFTER_PLAY_VERB,
    re.IGNORECASE,
)

# How a delivered Hermes result is stored in the chat history: the pipeline's RESULT_PROMPT user
# message starts with this prefix. The router reads it to tell that the previous exchange was
# answered by Hermes — which is what "to" in "daj to głośniej" points at.
HERMES_RESULT_PREFIX = "Wynik zadania od Hermesa:"

VOLUME_MODIFIER_SMALL = 6  # "trochę ciszej"
VOLUME_MODIFIER_BIG = 24  # "dużo głośniej"


def _volume_modifier(utterance: str) -> int:
    if re.search(_VOL_SMALL, utterance, re.IGNORECASE):
        return VOLUME_MODIFIER_SMALL
    if re.search(_VOL_BIG, utterance, re.IGNORECASE):
        return VOLUME_MODIFIER_BIG
    return 0


def volume_fast_path(utterance: str) -> ToolCall | None:
    """Deterministic device-volume commands -> a ``set_speaker_volume`` call (no LLM round trip).

    Returns None for anything else, so unusual phrasings still reach the router/Hermes (which has the
    ``speaker_set_volume`` MCP tools).
    """
    if addresses_hermes(utterance):
        return None
    text = utterance.strip()
    if not text:
        return None
    if FAST_PATH_VOLUME_MUTE.match(text):
        return ToolCall("set_speaker_volume", {"action": "mute"})
    if FAST_PATH_VOLUME_MAX.match(text):
        return ToolCall("set_speaker_volume", {"action": "max"})
    for pattern, direction in ((FAST_PATH_VOLUME_DOWN, -1), (FAST_PATH_VOLUME_UP, 1)):
        match = pattern.match(text)
        if match is None:
            continue
        amount = match.group("amount")
        delta = int(amount) if amount is not None else VOLUME_DEFAULT_STEP
        if not 1 <= delta <= 100:
            return None
        if amount is None:
            delta = _volume_modifier(text) or delta
        return ToolCall("set_speaker_volume", {"action": "step", "delta": direction * delta})
    match = FAST_PATH_VOLUME_SET.match(text)
    if match is None:
        match = FAST_PATH_VOLUME_BARE_PERCENT.match(text)
    if match is not None:
        amount = int(match.group("amount"))
        if not 0 <= amount <= 100:
            return None
        arguments: dict[str, Any] = {"action": "set", "level": amount}
        if amount == 0:
            arguments = {"action": "mute"}
        return ToolCall("set_speaker_volume", arguments)
    return None


# ---- Spotify playback & music-volume fast path -------------------------------------------------------
# Explicit non-household playback targets (NEW-1): computer, PC, laptop, phone, tablet or
# headphones — including with up to two filler words (possessives, demonstratives, adjectives)
# between the preposition and the target ("na MOICH słuchawkach", "na TYM telefonie", "na moim
# NOWYM laptopie"; NEW-1 round 5). Targets now include smartfon/komórka/iPad/Mac/lapek/Windows/
# Chromebook and "moje głośniki" ("na moich głośnikach" is ambiguous — own vs household speakers —
# so it stays Hermes' job). The local spotify tool would play on the household device, so these
# are Hermes' jobs ready or not.
_PC_TARGET = (
    r"(?:komp(?:a|ie|em|ik\w*|owi)?\b|komputer\w*|pec\w*|pc\b|laptop\w*|lap(?:ek|ka|ki|kę|ku|ie|y|ami|ach)\b|"
    r"macbook\w*|ma[kc](?:a|u|iem|owi)?\b|desktop\w*|smartfon\w*|"
    r"telefon\w*|kom[oó]r[ck]\w*|iphon(?:e|ie|'ie|a|owi|em)?|ipad\w*|tablet\w*|tablecie\b|"
    r"windows\w*|chromebook\w*|"
    r"s[lł]uchawk\w*|sluchawk\w*|airpods\w*|g[lł]o[sś]nik\w*|glosnik\w*)"
)
# Up to two arbitrary words (possessive, demonstrative, adjective) may sit between the preposition
# and the target word; "na słuchawkach" (none) matches too.
_PC_POSSESSIVE = r"(?:(?:[a-ząćęłńóśźż0-9]+\s+){0,2})"
SPOTIFY_DEVICE_TRANSFER = re.compile(
    r"\b(?:prze[lł][aą]cz|przenie[sś])\s+(?:spotify|muzyk[ęe]|muzyke|odtwarzanie)\s+na\b"
    r"|\b(?:pu[sś][cć]|pusc|w[lł][aą]cz|wlacz|zagraj|odtw[oó]rz|odtworz|prze[lł][aą]cz|przenie[sś])\b"
    r"[^.!?\n]*\b(?:na|w|we|przez)\s+" + _PC_POSSESSIVE + _PC_TARGET + r"\b"
    r"|\b(?:muzyk\w*|spotify\w*|odtwarzanie|radi\w*|film\w*|bajk\w*|playlist\w*)\b"
    r"[^.!?\n]*\b(?:na|w|we|przez)\s+" + _PC_POSSESSIVE + _PC_TARGET + r"\b"
    r"|\b(?:pu[sś][cć]|pusc|w[lł][aą]cz|wlacz|zagraj)\b.*\bna\s+(?:g[lł]o[sś]nik\w*|glosnik\w*|amplituner\w*|telewizor\w*|salon\w*|sypialni\w*)\b"
    r"|\b(?:muzyk\w*|spotify\w*|odtwarzanie|radi\w*)\b.*\bw\s+(?:salonie|sypialni|kuchni|pokoju|[lł]azience|gara[zż]u)\b",
    re.IGNORECASE,
)
# How the pipeline records a local spotify tool reply in the chat history: the assistant message
# starts with this prefix. The router reads it to tell that the previous exchange was served by the
# local spotify tool — which is what a bare "głośniej"/"dalej" then points at (M1). Free-text
# mentions of Spotify in older turns must never license that.
SPOTIFY_TURN_PREFIX = "Spotify:"
FAST_PATH_SPOTIFY_PAUSE = re.compile(
    _VOL_PREFIX
    + r"(?:pauza|spauzuj(?:\s+(?:muzyk[ęe]|muzyke|odtwarzanie|spotify))?"
    r"|(?:zatrzymaj|wstrzymaj)\s+(?:muzyk[ęe]|muzyke|odtwarzanie|spotify))"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_RESUME = re.compile(
    _VOL_PREFIX
    + r"(?:(?:wzn[oó]w|wznow|kontynuuj)\s+(?:muzyk[ęe]|muzyke|odtwarzanie|spotify)"
    r"|odpauzuj(?:\s+(?:muzyk[ęe]|muzyke|odtwarzanie|spotify))?)"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_NEXT = re.compile(
    _VOL_PREFIX
    + r"(?:(?:nast[eę]pn|kolejn)(?:a\s+piosenka|y\s+utw[oó]r|y\s+utwor|y\s+kawa[lł]ek|y\s+kawalek)"
    r"|(?:pomi[nń]|pomin|prze[lł][aą]cz)\s+(?:utw[oó]r|utwor|piosenk[ęe]|piosenke|kawa[lł]ek|kawalek)"
    r"|skip(?:\s+(?:track|song))?)"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_DALEJ = re.compile(
    _VOL_PREFIX + r"(?:dalej|nast[eę]pn[ay]|next(?:\s+song|\s+track)?)" + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_PREV = re.compile(
    _VOL_PREFIX
    + r"(?:poprzedni(?:a\s+piosenka|\s+utw[oó]r|\s+utwor|\s+kawa[lł]ek|\s+kawalek)"
    r"|cofnij\s+(?:utw[oó]r|utwor|piosenk[ęe]|piosenke|kawa[lł]ek|kawalek))"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_NOW = re.compile(
    _VOL_PREFIX
    + r"(?:co\s+(?:teraz\s+)?(?:gra|leci)(?:\s+na\s+spotify)?"
    r"|jaka\s+to\s+piosenka|jaki\s+to\s+(?:utw[oó]r|utwor))"
    + _VOL_TAIL,
    re.IGNORECASE,
)
# An optional leading playback verb: "puść muzykę głośniej", "zagraj spotify ciszej".
_SPOT_VOL_LEAD = r"(?:(?:pu[sś][cć]|pusc|zagraj|odtw[oó]rz|odtworz)\s+)?"
FAST_PATH_SPOTIFY_VOL_UP = re.compile(
    _VOL_PREFIX
    + r"(?:(?:we[źz]|zr[oó]b|zrob|daj)\s+)?"
    + _SPOT_VOL_LEAD
    + r"(?:(?:muzyk[ęe]|muzyke|spotify|radio)\s+(?:" + _VOL_SMALL + r"|" + _VOL_BIG + r")?\s*" + _VOL_UP
    + r"|(?:" + _VOL_SMALL + r"|" + _VOL_BIG + r")?\s*(?:podg[lł]o[sś](?:nij)?|podglos(?:nij)?|g[lł]o[sś]niej|glosniej)\s+(?:muzyk[ęe]|muzyke|spotify|radio)"
    r"|zwi[eę]ksz\s+g[lł]o[sś]no[sś][cć]\s+(?:muzyki|spotify|radia)"
    r"|(?:pu[sś][cć]|pusc|zagraj|odtw[oó]rz|odtworz)\s+co[sś]\s+(?:" + _VOL_SMALL + r"|" + _VOL_BIG + r")?\s*" + _VOL_UP + r")"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_VOL_DOWN = re.compile(
    _VOL_PREFIX
    + r"(?:(?:we[źz]|zr[oó]b|zrob|daj)\s+)?"
    + _SPOT_VOL_LEAD
    + r"(?:(?:muzyk[ęe]|muzyke|spotify|radio)\s+(?:" + _VOL_SMALL + r"|" + _VOL_BIG + r")?\s*" + _VOL_DOWN
    + r"|(?:wycisz(?:aj)?|przycisz(?:aj)?|[sś]cisz(?:aj)?|scisz(?:aj)?)\s+(?:muzyk[ęe]|muzyke|spotify|radio)"
    + r"|(?:" + _VOL_SMALL + r"|" + _VOL_BIG + r")?\s*(?:[sś]cisz(?:aj)?|scisz(?:aj)?|przycisz(?:aj)?|ciszej)\s+(?:muzyk[ęe]|muzyke|spotify|radio)"
    r"|zmniejsz\s+g[lł]o[sś]no[sś][cć]\s+(?:muzyki|spotify|radia)"
    r"|(?:pu[sś][cć]|pusc|zagraj|odtw[oó]rz|odtworz)\s+co[sś]\s+(?:" + _VOL_SMALL + r"|" + _VOL_BIG + r")?\s*" + _VOL_DOWN + r")"
    + _VOL_TAIL,
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_VOL_SET = re.compile(
    _VOL_PREFIX
    + r"(?:(?:ustaw(?:iaj)?|daj)\s+)?"
    r"(?:g[lł]o[sś]no[sś][cć]\s+(?:muzyki|spotify|radia)|glosnosc\s+(?:muzyki|spotify|radia)|(?:muzyk[ęe]|muzyke|spotify|radio)(?:\s+g[lł]o[sś]no[sś][cć])?)"
    r"\s+(?:na|do)\s*" + _VOL_AMOUNT_OPT + _VOL_TAIL,
    re.IGNORECASE,
)
# "wycisz muzykę" / "wycisz Spotify": a mute of the music, i.e. volume 0 on the spotify device.
FAST_PATH_SPOTIFY_MUTE = re.compile(
    _VOL_PREFIX + r"wycisz\s+(?:muzyk[ęe]|muzyke|spotify|radio)" + _VOL_TAIL,
    re.IGNORECASE,
)
# A named play request ("puść Muse", "puść Dawida Podsiadło na Spotify") is served deterministically
# when the local tool is ready; only "puść" (the music verb) triggers it, never "włącz"/"zagraj",
# which also start kettles and games. Generic "coś"/mood queries are Hermes' and stay out (H2/B1).
# Objects that are not music (NEW-4): video, radio, fairy tales, alarms, timers and messages are
# Hermes jobs or the normal router's business — "puść film" must never become a spotify search.
NON_MUSIC_PLAY_OBJECT = re.compile(
    r"\b(?:film\w{0,3}|serial\w{0,2}|bajk\w{0,2}|bajeczk\w{0,2}|kresk[oó]wk\w{0,2}|anim[eę]|"
    r"radi(?:o|a|u|em|iem)|audycj\w{0,2}|podcast\w{0,2}|nagran\w{0,2}|"
    r"yt|youtub\w{0,2}|netfli\w{0,2}|hbo)\b",
    re.IGNORECASE,
)
NON_MUSIC_SOLO_OBJECT = re.compile(
    r"(?:budzik\w{0,2}|alarm\w{0,2}|minutnik\w{0,2}|timer\w{0,2}|przypomnieni\w{0,2}|"
    r"wiadomo[sś][cć]\w{0,2}|sms\w{0,2}|mail\w{0,2}|e-?mail\w{0,2}|notatk\w{0,2})"
    r"(?:\s+(?:na|do|o|po|w|we|za|od)\s+[^,]*)?",
    re.IGNORECASE,
)
FAST_PATH_SPOTIFY_PLAY = re.compile(
    _VOL_PREFIX + r"(?:pu[sś][cć]|pusc)\s+(?P<query>.+?)(?:\s+na\s+spotify)?" + _VOL_TAIL,
    re.IGNORECASE,
)
_PLAY_QUERY_LEAD_FILLER = re.compile(r"^(?:mi|mnie|prosz[ęe]|poprosz[ęe])\s+", re.IGNORECASE)

UNSUPPORTED_HOME = re.compile(
    r"[sś]wiat[lł]|swiatl|\blamp|\b[żz]ar[oó]wk|\bzarowk|\bzga[sś]\b|\bzgas\b|przyciemnij|rozja[sś]nij|\blight"
    r"|\brolet|\b[żz]aluzj|\bzaluzj|\bzas[lł]on|\bdrzwi|\bbram|\bgara[żz]|\bgaraz|\bodkurzacz",
    re.IGNORECASE,
)

# An imperative counts as a *home tool* override only with an explicit controllable device noun.
# Whole-house heating words ("wyłącz ogrzewanie w całym domu" = the boiler) and bare rooms
# ("włącz światło w salonie") are Hermes' business or a read-only refusal, never a local write.
# The "u <person>" places come from the household aliases via set_home_person_words (NEW-6).
HOME_OVERRIDE_DEVICE = re.compile(_home_override_pattern(""), re.IGNORECASE)

# Messages, alarms and calendar objects are Hermes actions by nature: a local cue match ("u Ani",
# "muzykę") must never let the tool override win over the Hermes action rule ("napisz wiadomość
# u Ani", "ustaw budzik u Ani na szóstą").
HERMES_MESSAGE_ALARM = re.compile(
    r"\b(?:napisz|wy[sś]lij|wyslij|odpisz|zapisz|dodaj|usu[nń]|ustaw|zam[oó]w|zamow|zarezerwuj|zaplanuj|przygotuj|zle[cć])\b"
    r"[^.!?\n]*\b(?:wiadomo[sś][cć]\w*|mail\w*|e-?mail\w*|sms\w*|esemes\w*|budzik\w*|alarm\w*|przypomnieni\w*|notatk\w*|kalendarz\w*|spotkan\w*|wydarzeni\w*|zadani\w*|termin\w*|dzwo[nń]\w*|telefon\w*)\b"
    r"|\b(?:wiadomo[sś][cć]\w*|mail\w*|e-?mail\w*|sms\w*|esemes\w*|budzik\w*|alarm\w*|przypomnieni\w*|notatk\w*|kalendarz\w*|spotkan\w*|wydarzeni\w*|zadani\w*|termin\w*)\b[^.!?\n]*\b(?:napisz|wy[sś]lij|wyslij|odpisz|zapisz|dodaj|usu[nń]|ustaw|zam[oó]w|zamow|zarezerwuj|zaplanuj|przygotuj)\b",
    re.IGNORECASE,
)


# Injected by the server at start-up (see server.py): the router must not import tool modules, both
# to keep import of gaja.llm.router light and to keep these deterministic paths unit-testable.
# Defaults mean "no local Spotify": every media request then goes to Hermes (see media_handoff).
_spotify_playing_probe: Callable[[], bool] = lambda: False


def set_spotify_playing_probe(probe: Callable[[], bool] | None) -> None:
    """Inject the "is Spotify playing" probe used by the spotify fast path (None restores the default)."""
    global _spotify_playing_probe
    _spotify_playing_probe = probe if probe is not None else (lambda: False)


def recent_spotify_turn(history: list[dict[str, Any]] | None) -> bool:
    """Whether the previous exchange was answered by the local spotify tool.

    The pipeline records those replies with the ``Spotify:`` marker (see SPOTIFY_TURN_PREFIX), so a
    bare "głośniej"/"dalej" right after one points at Spotify. A free-text mention of Spotify in an
    older turn ("puść jazz na Spotify" three turns ago) must never license that hijack.
    """
    if not history:
        return False
    for message in reversed(history[-2:]):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).startswith(SPOTIFY_TURN_PREFIX)
    return False


def spotify_fast_path(
    utterance: str,
    history: list[dict[str, Any]] | None = None,
    *,
    playing: bool | None = None,
) -> ToolCall | None:
    """Deterministic Spotify playback/volume fast path (no LLM round trip)."""
    if addresses_hermes(utterance) or SPOTIFY_DEVICE_TRANSFER.search(utterance):
        return None
    text = utterance.strip()
    if not text:
        return None
    if FAST_PATH_SPOTIFY_PAUSE.match(text):
        return ToolCall("spotify", {"action": "pause"})
    if FAST_PATH_SPOTIFY_RESUME.match(text):
        return ToolCall("spotify", {"action": "play"})
    if FAST_PATH_SPOTIFY_NEXT.match(text):
        return ToolCall("spotify", {"action": "next"})
    if FAST_PATH_SPOTIFY_PREV.match(text):
        return ToolCall("spotify", {"action": "previous"})
    if FAST_PATH_SPOTIFY_NOW.match(text):
        return ToolCall("spotify", {"action": "now_playing"})
    if FAST_PATH_SPOTIFY_MUTE.match(text):
        return ToolCall("spotify", {"action": "volume_set", "level": 0})
    if FAST_PATH_SPOTIFY_VOL_UP.match(text):
        return ToolCall("spotify", {"action": "volume_up"})
    if FAST_PATH_SPOTIFY_VOL_DOWN.match(text):
        return ToolCall("spotify", {"action": "volume_down"})
    match = FAST_PATH_SPOTIFY_VOL_SET.match(text)
    if match is not None:
        amount = int(match.group("amount"))
        if 0 <= amount <= 100:
            return ToolCall("spotify", {"action": "volume_set", "level": amount})
        return None

    play = FAST_PATH_SPOTIFY_PLAY.match(text)
    if play is not None and "query" in play.groupdict():
        query = _PLAY_QUERY_LEAD_FILLER.sub("", play.group("query").strip()).strip()
        # Home/unsupported-home objects ("puść klimę"? never mind), non-music objects (NEW-4, M3:
        # "puść film"/"puść budzik"; whole-word inflections, so "puść Radiohead" stays a search) and
        # generic jobs are not this path.
        if (
            query
            and not GENERIC_PLAY_REQUEST.search(text)
            and not NON_MUSIC_PLAY_OBJECT.search(query)
            and not NON_MUSIC_SOLO_OBJECT.fullmatch(query)
            and not TOOL_CUES["home"].search(text)
            and not UNSUPPORTED_HOME.search(text)
        ):
            return ToolCall("spotify", {"action": "play", "query": query})

    recent = recent_spotify_turn(history)
    active = _spotify_playing_probe() if playing is None else playing
    if (active or recent) and FAST_PATH_SPOTIFY_DALEJ.match(text):
        return ToolCall("spotify", {"action": "next"})
    if recent:
        if FAST_PATH_VOLUME_UP.match(text):
            return ToolCall("spotify", {"action": "volume_up"})
        if FAST_PATH_VOLUME_DOWN.match(text):
            return ToolCall("spotify", {"action": "volume_down"})
    return None


def local_tool_override(utterance: str, tool_names: set[str], *, spotify_ready: bool = True) -> bool:
    """Whether an imperative/at-home utterance is served by an enabled local tool instead of Hermes.

    ``spotify_ready=False`` (no logged-in local Spotify) disables the spotify-cue override too:
    the cue match must not mask the Hermes action rule, so "włącz muzykę"/"włącz Spotify" keep
    main@7741180's ``hermes:action`` route instead of being stolen by an unusable tool (readiness
    leak, r4)."""
    if re.search(r"\bminutnik|\bprzypomnij\s+mi\s+o\s+\d|\bzapami[eę]taj|\bzapomnij\b", utterance, re.IGNORECASE):
        return True
    if HERMES_MESSAGE_ALARM.search(utterance):
        return False
    if "spotify" in tool_names and spotify_ready and not SPOTIFY_DEVICE_TRANSFER.search(utterance):
        if TOOL_CUES["spotify"].search(utterance):
            return True
    if "home" in tool_names and not UNSUPPORTED_HOME.search(utterance):
        if TOOL_CUES["home"].search(utterance) and HOME_OVERRIDE_DEVICE.search(utterance):
            return True
    return False


def clock_fast_path(utterance: str) -> ToolCall | None:
    if addresses_hermes(utterance):
        return None
    if FAST_PATH_TIME.match(utterance):
        return ToolCall("get_time", {"kind": "time"})
    if FAST_PATH_DATE.match(utterance):
        return ToolCall("get_time", {"kind": "date"})
    return None


def recent_hermes_result(history: list[dict[str, Any]]) -> bool:
    """Whether the newest user turn in ``history`` is a delivered Hermes result, i.e. the previous
    exchange was handled by Hermes — what "to" in "daj to głośniej" then most likely points at."""
    for message in reversed(history):
        if message.get("role") == "user":
            return str(message.get("content", "")).startswith(HERMES_RESULT_PREFIX)
    return False


def hermes_media_followup(utterance: str, history: list[dict[str, Any]]) -> bool:
    """A media follow-up right after a delivered Hermes result stays Hermes', ready or not: the
    sound plays where Hermes put it ("Daj to głośniej", "Puść muzykę głośniej"). A bare
    whole-utterance volume command ("głośniej") is excluded: that is the Atom speaker's fast path."""
    if not recent_hermes_result(history) or not VOLUME_WORD.search(utterance):
        return False
    return VOLUME_OBJECT.search(utterance) is not None or MUSIC_MEDIA.search(utterance) is not None


def media_handoff(utterance: str, history: list[dict[str, Any]], *, spotify_ready: bool = False) -> bool:
    """Whether a media request belongs to Hermes rather than the local spotify tool.

    Without a ready local tool (``spotify_ready=False``, the default and the state during a router
    outage) this is main@7741180's rule set (NEW-2): playback with a music object ("Puść coś na
    Spotify", "puść muzykę"), a volume word with music or a playback verb ("Puść muzykę głośniej",
    "wycisz Spotify", "zagraj coś głośniej"), and a volume command pointing at an object ("Daj to
    głośniej") right after a Hermes result — plus the generic/mood/genre requests that genuinely
    name music ("muzyka do pracy", "spokojną muzykę", "włącz jazz"). Timer, alarm, film, message
    and game objects ("zatrzymaj minutnik", "puść film", "zagraj w szachy") never hand off.

    With a ready local tool only what Hermes must keep is handed off: mood, genre and indefinite
    ("coś") jobs — Spotify search cannot honour "puść coś spokojnego" — and media follow-ups right
    after a Hermes result. Simple playback and volume stay local then.
    """
    if hermes_media_followup(utterance, history):
        return True
    if spotify_ready:
        return GENERIC_PLAY_REQUEST.search(utterance) is not None
    # No local tool: main@7741180's handoff plus the genuinely-music generic jobs (NEW-2). A
    # playback verb hands off only with a music/Spotify object, a volume word only with music or
    # a playback verb; the generic exception must name music itself. "zatrzymaj minutnik",
    # "puść film", "zagraj w szachy" and friends stay local as on main.
    if GENERIC_MUSIC_PLAY_REQUEST.search(utterance):
        return True
    if MUSIC_MEDIA.search(utterance) and PLAYBACK_VERB.search(utterance):
        return True
    if not VOLUME_WORD.search(utterance):
        return False
    if MUSIC_MEDIA.search(utterance) or PLAYBACK_VERB.search(utterance):
        return True
    return VOLUME_OBJECT.search(utterance) is not None and recent_hermes_result(history)


AT_HOME = re.compile(
    r"\bw\s+(?:ca[lł]ym\s+)?(?:domu|salonie|sypialni|kuchni|pokoju|[lł]azience|gara[zż]u|biurze)\b"
    r"|\b(?:in|at)\s+(?:the\s+)?(?:house|home|living\s+room|bedroom|kitchen)\b",
    re.IGNORECASE,
)


def plausible(call: ToolCall, utterance: str, history: list[dict[str, Any]], *, spotify_ready: bool = True) -> bool:
    if call.name == "spotify":
        # Readiness leak, r4d follow-up: with Spotify NOT ready nothing may make a spotify call
        # plausible — neither the fast path nor the cue/history checks below (a "Spotify:" marker
        # in the history matches the cue and otherwise lets select_tool return a call to an
        # unusable tool instead of following main@7741180's local/Hermes route).
        if not spotify_ready:
            return False
        if spotify_fast_path(utterance, history) is not None:
            return True
    return cue_plausible(call.name, utterance, history)


def cue_plausible(name: str, utterance: str, history: list[dict[str, Any]]) -> bool:
    """plausible()'s cue/history half on its own: does the utterance (or, for follow-ups like "a w
    Krakowie?", the previous exchange) mention what the tool ``name`` is about; a tool without a cue
    is always plausible. classify() reuses this for the not-ready spotify handoff, where the
    fast-path half must NOT count (whole-utterance commands like "pauza" keep main@7741180's local
    route, see the parity table in tests/test_router.py)."""
    cue = TOOL_CUES.get(name)
    if cue is None:
        return True
    if cue.search(utterance):
        return True
    # Only the previous exchange: an old weather turn must not license every later short question.
    return len(utterance.split()) <= 5 and any(cue.search(str(m.get("content", ""))) for m in history[-2:])


def any_plausible(
    names: set[str],
    utterance: str,
    history: list[dict[str, Any]],
    *,
    spotify_ready: bool = True,
) -> bool:
    """Whether ``plausible`` could accept a call to any of ``names``; if not, selecting is wasted time."""
    return any(plausible(ToolCall(name, {}), utterance, history, spotify_ready=spotify_ready) for name in names)


Kind = Literal["local", "tool", "hermes"]
ROUTER_MESSAGE_CHARS = 240
SELECT_TOKENS = 64


class ToolUnavailable(RuntimeError):
    """select_tool(): the model picked a tool that cannot run right now — spotify without a ready
    local login. The pipeline turns this into a Hermes handoff with reason "tool-unavailable"
    instead of executing a doomed call or falling back to a local chat reply (r4e)."""


def promises_handoff(text: str) -> bool:
    return bool(PROMISE_PL.search(text) or PROMISE_EN.search(text)) and not MYTHOLOGY.search(text)


def addresses_hermes(text: str) -> bool:
    """The user explicitly asks Hermes (the agent) to do something."""
    return bool(EXPLICIT_HERMES.search(text)) and not MYTHOLOGY.search(text)


def router_prefix(system: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n"


# Prompt fragments that teach the optional tools; dropped by template_router_prompt when the tool
# is not registered so the small model is not trained towards TOOL for a missing tool (H4).
_PROMPT_TOOL_BULLETS: dict[str, re.Pattern[str]] = {
    "spotify": re.compile(r"; odtwarzanie muzyki na Spotify \([^)]*\)"),
    "home": re.compile(
        r"; klimatyzacja, temperatura w pokojach, stan pieca, telewizory i amplituner w domu \(Home Assistant\)"
    ),
}


def template_router_prompt(system: str, tool_names: set[str]) -> str:
    """Drop the prompt teachings of tools that are not registered (best effort, H4).

    Without HASS_* the ``home`` examples would keep steering home questions into TOOL although no
    local tool can answer them. The classify-level "tool-unavailable" route stays the safety net
    when this file and the prompt drift apart. Cheap on purpose: whole blank-line-separated blocks
    (the tool listing line and the few-shot examples whose ``Wywołanie:`` calls the tool) and the
    two TOOL-bullet fragments above.
    """
    missing = [name for name in ("spotify", "home") if name not in tool_names]
    if not missing:
        return system
    kept: list[str] = []
    for block in system.split("\n\n"):
        stripped = block.strip()
        call = re.search(r"^Wywołanie: (spotify|home)\(", block, re.MULTILINE)
        if call is not None and call.group(1) in missing:
            continue  # a few-shot example block for the missing tool
        lines = [
            line
            for line in block.splitlines()
            if not any(line.strip().startswith(f"{name}(") for name in missing)
        ]
        block = "\n".join(lines)
        for name in missing:
            block = _PROMPT_TOOL_BULLETS[name].sub("", block)
        kept.append(block)
    return "\n\n".join(kept)


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= ROUTER_MESSAGE_CHARS else text[: ROUTER_MESSAGE_CHARS - 1] + "…"


def router_context(system: str, history: list[dict[str, Any]]) -> str:
    """Everything of the router prompt before the new utterance: system, recent history, user header.

    Priming this (not just the system prefix) keeps the history tokens off the critical path; only
    the utterance itself is evaluated when the turn is classified.
    """
    parts = [router_prefix(system)]
    for message in history:
        if message["role"] == "user":
            parts.append(f"<|im_start|>user\nZapytanie: {_short(message['content'])}<|im_end|>\n")
        elif message["role"] == "assistant":
            parts.append(f"<|im_start|>assistant\n{_short(message['content'])}<|im_end|>\n")
    parts.append("<|im_start|>user\n")
    return "".join(parts)


def build_router_prompt(system: str, utterance: str, history: list[dict[str, Any]]) -> str:
    # The utterance is capped exactly like the history lines above it: a huge paste must not blow
    # the one-token classification prompt (the pipeline rejects typed turns over 4000 characters,
    # but transcribed audio and other callers are unbounded).
    return f"{router_context(system, history)}Zapytanie: {_short(utterance)}<|im_end|>\n<|im_start|>assistant\nKategoria:"


def class_masses(completion: dict[str, Any]) -> tuple[float, float, float]:
    """Normalised probability mass of the first answer token: (LOCAL, TOOL, HERMES) by first letter."""
    candidates = (completion.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    hermes = local = tool = 0.0
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip().lower()
        mass = math.exp(float(candidate.get("logprob", -99.0)))
        if token.startswith("h"):
            hermes += mass
        elif token.startswith("l"):
            local += mass
        elif token.startswith("t"):
            tool += mass
    total = hermes + local + tool
    if total <= 0:
        return 1.0, 0.0, 0.0
    return local / total, tool / total, hermes / total


def p_hermes(completion: dict[str, Any]) -> float:
    """Normalised probability mass of the first answer token starting with H (HERMES) vs L (LOCAL)."""
    local, _, hermes = class_masses(completion)
    total = hermes + local
    return hermes / total if total > 0 else 0.0


def tool_grammar(tools: list[dict[str, Any]]) -> str:
    """GBNF for one pythonic call `name(arg=value, ...)` restricted to the given tools and their parameters."""
    rules = ['root ::= " "? call', 'str ::= "\\"" [^"\\n]{0,60} "\\""', 'num ::= "-"? [0-9]{1,9} ("." [0-9]{1,6})?']
    alternatives: list[str] = []
    for index, tool in enumerate(tools):
        function = tool.get("function", tool)
        name = function["name"]
        params = list((function.get("parameters") or {}).get("properties") or {})
        alternatives.append(f"t{index}")
        if not params:
            rules.append(f't{index} ::= "{name}()"')
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        arguments: list[str] = []
        for param in params:
            enum = properties[param].get("enum") if isinstance(properties[param], dict) else None
            if enum:
                values = " | ".join(f'"\\"{value}\\""' for value in enum)
                arguments.append(f'"{param}=" ({values})')
            else:
                arguments.append(f'"{param}=" (str | num)')
        rules.append(f"a{index} ::= " + " | ".join(f"({argument})" for argument in arguments))
        rules.append(f't{index} ::= "{name}(" (a{index} (", " a{index}){{0,4}})? ")"')
    rules.insert(1, "call ::= " + " | ".join(alternatives))
    return "\n".join(rules)


def parse_call(text: str, allowed: set[str]) -> ToolCall | None:
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in allowed:
        return None
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
            return None
        arguments[keyword.arg] = keyword.value.value
    return ToolCall(node.func.id, arguments)


def build_profile_prompt(utterance: str, options: list[tuple[str, str]]) -> str:
    listing = "\n".join(f"{i}. {label}: {desc or label}" for i, (label, desc) in enumerate(options, 1))
    system = (
        "Wybierasz, który asystent wykona zadanie. Odpowiedz tylko numerem.\n"
        f"{listing}"
    )
    return (
        f"{router_prefix(system)}<|im_start|>user\nZadanie: {_short(utterance)}<|im_end|>\n"
        "<|im_start|>assistant\nNumer:"
    )


def option_masses(completion: dict[str, Any], count: int) -> list[float]:
    """Normalised probability of each numbered option (1-based tokens "1".."9") for the first token."""
    candidates = (completion.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    masses = [0.0] * count
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip()
        if token.isdigit() and 1 <= int(token) <= count:
            masses[int(token) - 1] += math.exp(float(candidate.get("logprob", -99.0)))
    total = sum(masses)
    return [m / total for m in masses] if total > 0 else [1.0 / count] * count


@dataclass(frozen=True)
class Route:
    kind: Kind
    p_local: float = 1.0
    p_tool: float = 0.0
    p_hermes: float = 0.0
    reason: str = "router"


class Router:
    """One 1-token logprob call on the pinned router slot decides LOCAL / TOOL / HERMES.

    With ``tools`` empty this is the 2-class Router v2 (TOOL mass, if any, counts as local).
    For TOOL, ``select_tool`` continues the very same prompt (``Kategoria: TOOL\\nWywołanie:``) with a
    grammar, so the call is a short constrained generation on the already cached prefix.
    """

    def __init__(
        self,
        llm: LLMClient,
        system: str,
        *,
        threshold: float,
        history_turns: int,
        timeout_s: float,
        slot: int = -1,
        tools: list[dict[str, Any]] | None = None,
        tool_threshold: float = 0.5,
        spotify_ready: Callable[[], bool] | None = None,
    ) -> None:
        self._llm = llm
        self._system = system
        self.threshold = threshold
        self.tool_threshold = tool_threshold
        self._history_turns = history_turns
        self._timeout_s = timeout_s
        self._slot = slot
        self._tools = list(tools or [])
        self._tool_names = {t.get("function", t)["name"] for t in self._tools}
        self._grammar = tool_grammar(self._tools) if self._tools else ""
        # Whether the local spotify tool can serve this turn *right now* (registered + token file
        # present). A callable, so a login completed after start-up takes effect without a restart.
        self._spotify_ready = spotify_ready

    @classmethod
    def from_file(cls, llm: LLMClient, path: Path, **kwargs: Any) -> Router:
        return cls(llm, load_text_block(path), **kwargs)

    @property
    def tools_enabled(self) -> bool:
        return bool(self._tools)

    def _spotify_ready_now(self) -> bool:
        """Whether the local spotify tool can serve this turn right now (registered + callable probe)."""
        return self._spotify_ready is not None and "spotify" in self._tool_names and self._spotify_ready()

    def _recent(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return history[-self._history_turns :] if self._history_turns > 0 else []

    def _prompt(self, utterance: str, history: list[dict[str, Any]]) -> str:
        return build_router_prompt(self._system, utterance, self._recent(history))

    async def classify(self, utterance: str, history: list[dict[str, Any]]) -> Route:
        if addresses_hermes(utterance):
            return Route("hermes", 0.0, 0.0, 1.0, reason="explicit")
        recent = self._recent(history)
        if self._tools and "get_time" in self._tool_names and clock_fast_path(utterance):
            return Route("tool", 0.0, 1.0, 0.0, reason="fast-path")
        # Media routing before the model: the small router keeps filing these under TOOL (the local
        # speaker), which is the wrong device, or under LOCAL, where the chat LLM then claims it did
        # something. With a ready local spotify tool it serves simple playback and volume itself;
        # without it (or during a router outage) Hermes gets every media request. Hermes always
        # keeps device transfers, mood/genre/"coś" jobs and volume right after a Hermes result.
        spotify_ready = self._spotify_ready_now()
        if not MYTHOLOGY.search(utterance) and media_handoff(utterance, history, spotify_ready=spotify_ready):
            return Route("hermes", 0.0, 0.0, 1.0, reason="media")
        if spotify_ready and spotify_fast_path(utterance, recent):
            return Route("tool", 0.0, 1.0, 0.0, reason="fast-path")
        if self._tools and "set_speaker_volume" in self._tool_names and volume_fast_path(utterance):
            return Route("tool", 0.0, 1.0, 0.0, reason="fast-path")
        completion = await self._llm.complete(self._prompt(utterance, history), timeout_s=self._timeout_s, slot=self._slot)
        local, tool, hermes = class_masses(completion)
        if not self._tools:
            local, tool = local + tool, 0.0
        mythology = MYTHOLOGY.search(utterance)
        if hermes >= self.threshold and not mythology:
            return Route("hermes", local, tool, hermes)
        # Readiness leak (r4): with Spotify NOT ready the spotify cue must not mask the Hermes
        # action rule — "włącz muzykę"/"włącz Spotify" keep main@7741180's hermes:action route.
        is_local_override = local_tool_override(utterance, self._tool_names, spotify_ready=spotify_ready)
        if (
            self._tools
            and not mythology
            and not is_local_override
            and (HERMES_ACTION.search(utterance) or AT_HOME.search(utterance) or SPOTIFY_DEVICE_TRANSFER.search(utterance))
        ):
            return Route("hermes", local, tool, hermes, reason="action")
        # A home/spotify question whose tool is not even registered (no HASS_* env, spotify disabled)
        # must go to Hermes: the prompt teaches these as TOOL, but with nothing selectable the local
        # chat model would answer from imagination (or the weather tool would use the wrong
        # thermometer for "temperatura w sypialni").
        if self._tools and not mythology and any(
            name not in self._tool_names and TOOL_CUES[name].search(utterance) for name in ("home", "spotify")
        ):
            return Route("hermes", local, tool, hermes, reason="tool-unavailable")
        if self._tools and tool >= self.tool_threshold:
            # Readiness leak (r4e): with the spotify tool registered but NOT ready, a TOOL-classified
            # turn that is about spotify goes to Hermes ("tool-unavailable", the twin of the H4 rule
            # above): select_tool would refuse the spotify call (plausible() says no to an unusable
            # tool, r4d) and the turn would fall to a local chat reply that hallucinates the action
            # ("Zatrzymałam…"). The cue check mirrors plausible()'s cue/history half WITHOUT the
            # spotify fast path, so whole-utterance commands with no music noun ("pauza", "dalej",
            # "wznów") keep main@7741180's local route (parity table in tests/test_router.py).
            if (
                not spotify_ready
                and not mythology
                and not is_local_override
                and "spotify" in self._tool_names
                and cue_plausible("spotify", utterance, recent)
            ):
                return Route("hermes", local, tool, hermes, reason="tool-unavailable")
            return Route("tool", local, tool, hermes)
        return Route("local", local, tool, hermes)

    async def select_tool(self, utterance: str, history: list[dict[str, Any]]) -> ToolCall | None:
        if not self._tools:
            return None
        recent = self._recent(history)
        spotify_ready = self._spotify_ready_now()
        fast = clock_fast_path(utterance) if "get_time" in self._tool_names else None
        if fast is not None:
            return fast
        # Readiness leak (r4): with Spotify NOT ready the pause/next fast paths must not fire here
        # either ("pauza" stayed local on main instead of calling an unusable spotify tool).
        if "spotify" in self._tool_names and spotify_ready:
            fast = spotify_fast_path(utterance, recent)
            if fast is not None:
                return fast
        if "set_speaker_volume" in self._tool_names:
            fast = volume_fast_path(utterance)
            if fast is not None:
                return fast
        if not any_plausible(self._tool_names, utterance, recent, spotify_ready=spotify_ready):
            return None
        prompt = self._prompt(utterance, history) + " TOOL\nWywołanie:"
        text = await self._llm.generate(
            prompt, n_predict=SELECT_TOKENS, grammar=self._grammar, timeout_s=self._timeout_s * 2, slot=self._slot
        )
        call = parse_call(text, self._tool_names)
        if call is not None and call.name == "spotify" and not spotify_ready:
            # Readiness leak (r4e): the model selected the one tool that cannot run right now.
            # Never execute it — raise so the pipeline hands the turn to Hermes with reason
            # "tool-unavailable" instead of falling back to a local reply (plausible() below
            # would silently turn this into None -> local).
            raise ToolUnavailable(call.name)
        if call is not None and not plausible(call, utterance, recent, spotify_ready=spotify_ready):
            return None
        return call

    async def pick_profile(self, utterance: str, options: list[tuple[str, str]]) -> tuple[int, float]:
        """Which of the person's Hermes profiles (label, description) fits the task: (index, probability).

        One 1-token logprob call like ``classify``; the router slot is re-primed after the turn anyway.
        """
        prompt = build_profile_prompt(utterance, options)
        completion = await self._llm.complete(prompt, timeout_s=self._timeout_s, slot=self._slot)
        masses = option_masses(completion, len(options))
        best = max(range(len(options)), key=masses.__getitem__)
        return best, masses[best]

    async def p_hermes(self, utterance: str, history: list[dict[str, Any]]) -> float:
        return (await self.classify(utterance, history)).p_hermes

    async def prime(self, history: list[dict[str, Any]] | None = None) -> None:
        """Cache the prompt up to the next utterance; ``history`` is the chat context without system."""
        prompt = router_context(self._system, self._recent(history or []))
        await self._llm.prime(prompt, timeout_s=self._timeout_s, slot=self._slot)

    def hands_off(self, probability: float) -> bool:
        return probability >= self.threshold
