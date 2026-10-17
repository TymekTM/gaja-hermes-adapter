from __future__ import annotations

import asyncio
import re
import time
import unicodedata
import urllib.parse
from typing import Any

import httpx

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import PLACEHOLDER_LOCATIONS, ToolResult, get_default_location, polish_declension

DEFAULT_LOCATION = get_default_location()

# Irregular or frequent Polish locative/genitive city forms -> canonical nominative name.
IRREGULAR_LOCATIVES: dict[str, str] = {
    "krakowie": "Kraków",
    "krakowa": "Kraków",
    "krakow": "Kraków",
    "kraków": "Kraków",
    "gdańsku": "Gdańsk",
    "gdansku": "Gdańsk",
    "gdańska": "Gdańsk",
    "gdanska": "Gdańsk",
    "gdansk": "Gdańsk",
    "gdańsk": "Gdańsk",
    "wrocławiu": "Wrocław",
    "wroclawiu": "Wrocław",
    "wrocławia": "Wrocław",
    "wroclawia": "Wrocław",
    "wroclaw": "Wrocław",
    "wrocław": "Wrocław",
    "zakopanem": "Zakopane",
    "zakopanego": "Zakopane",
    "zakopane": "Zakopane",
    "nowym targu": "Nowy Targ",
    "nowego targu": "Nowy Targ",
    "nowy targ": "Nowy Targ",
    "berlinie": "Berlin",
    "berlina": "Berlin",
    "berlin": "Berlin",
    "katowicach": "Katowice",
    "katowic": "Katowice",
    "katowice": "Katowice",
    "warszawie": "Warszawa",
    "warszawy": "Warszawa",
    "warszawe": "Warszawa",
    "warszawę": "Warszawa",
    "warszawa": "Warszawa",
    "poznaniu": "Poznań",
    "poznania": "Poznań",
    "poznan": "Poznań",
    "poznań": "Poznań",
    "łodzi": "Łódź",
    "lodzi": "Łódź",
    "lodz": "Łódź",
    "łódź": "Łódź",
    "szczecinie": "Szczecin",
    "szczecina": "Szczecin",
    "szczecin": "Szczecin",
    "bydgoszczy": "Bydgoszcz",
    "bydgoszcz": "Bydgoszcz",
    "lublinie": "Lublin",
    "lublina": "Lublin",
    "lublin": "Lublin",
    "białymstoku": "Białystok",
    "bialymstoku": "Białystok",
    "białegostoku": "Białystok",
    "bialegostoku": "Białystok",
    "bialystok": "Białystok",
    "białystok": "Białystok",
    "gdyni": "Gdynia",
    "gdynie": "Gdynia",
    "gdynię": "Gdynia",
    "gdynia": "Gdynia",
    "sopocie": "Sopot",
    "sopotu": "Sopot",
    "sopot": "Sopot",
    "częstochowie": "Częstochowa",
    "czestochowie": "Częstochowa",
    "częstochowy": "Częstochowa",
    "czestochowa": "Częstochowa",
    "częstochowa": "Częstochowa",
    "radomiu": "Radom",
    "radomia": "Radom",
    "radom": "Radom",
    "sosnowcu": "Sosnowiec",
    "sosnowca": "Sosnowiec",
    "sosnowiec": "Sosnowiec",
    "toruniu": "Toruń",
    "torunia": "Toruń",
    "torun": "Toruń",
    "toruń": "Toruń",
    "kielcach": "Kielce",
    "kielc": "Kielce",
    "kielce": "Kielce",
    "gliwicach": "Gliwice",
    "gliwic": "Gliwice",
    "gliwice": "Gliwice",
    "zabrzu": "Zabrze",
    "zabrza": "Zabrze",
    "zabrze": "Zabrze",
    "bytomiu": "Bytom",
    "bytomia": "Bytom",
    "bytom": "Bytom",
    "bielsku-białej": "Bielsko-Biała",
    "bielsku białej": "Bielsko-Biała",
    "bielsku-bialej": "Bielsko-Biała",
    "bielsku bialej": "Bielsko-Biała",
    "bielsko-biala": "Bielsko-Biała",
    "bielsko-biała": "Bielsko-Biała",
    "bielsku": "Bielsko-Biała",
    "olsztynie": "Olsztyn",
    "olsztyna": "Olsztyn",
    "olsztyn": "Olsztyn",
    "rzeszowie": "Rzeszów",
    "rzeszowa": "Rzeszów",
    "rzeszow": "Rzeszów",
    "rzeszów": "Rzeszów",
    "rybniku": "Rybnik",
    "rybnika": "Rybnik",
    "rybnik": "Rybnik",
    "tychach": "Tychy",
    "tychów": "Tychy",
    "tychy": "Tychy",
    "opolu": "Opole",
    "opola": "Opole",
    "opole": "Opole",
    "zielonej górze": "Zielona Góra",
    "zielonej gorze": "Zielona Góra",
    "zielona gora": "Zielona Góra",
    "zielona góra": "Zielona Góra",
    "jeleniej górze": "Jelenia Góra",
    "jeleniej gorze": "Jelenia Góra",
    "jelenia gora": "Jelenia Góra",
    "jelenia góra": "Jelenia Góra",
    "szklarskiej porębie": "Szklarska Poręba",
    "szklarskiej porebie": "Szklarska Poręba",
    "szklarska poreba": "Szklarska Poręba",
    "szklarska poręba": "Szklarska Poręba",
    "nowym sączu": "Nowy Sącz",
    "nowym saczu": "Nowy Sącz",
    "nowego sącza": "Nowy Sącz",
    "nowy sacz": "Nowy Sącz",
    "nowy sącz": "Nowy Sącz",
    "starym sączu": "Stary Sącz",
    "starym saczu": "Stary Sącz",
    "stary sącz": "Stary Sącz",
    "nowym jorku": "Nowy Jork",
    "nowego jorku": "Nowy Jork",
    "nowy jork": "Nowy Jork",
    "new york": "New York",
    "gorzowie wielkopolskim": "Gorzów Wielkopolski",
    "gorzowie": "Gorzów Wielkopolski",
    "gorzów wielkopolski": "Gorzów Wielkopolski",
    "ostrowie wielkopolskim": "Ostrów Wielkopolski",
    "ostrowie": "Ostrów Wielkopolski",
    "ostrów wielkopolski": "Ostrów Wielkopolski",
    "krynicy-zdroju": "Krynica-Zdrój",
    "krynicy zdroju": "Krynica-Zdrój",
    "krynicy": "Krynica-Zdrój",
    "krynica-zdrój": "Krynica-Zdrój",
    "jastrzębiu-zdroju": "Jastrzębie-Zdrój",
    "jastrzebiu-zdroju": "Jastrzębie-Zdrój",
    "jastrzębiu zdroju": "Jastrzębie-Zdrój",
    "jastrzębie-zdrój": "Jastrzębie-Zdrój",
    "kędzierzynie-koźlu": "Kędzierzyn-Koźle",
    "kedzierzynie-kozlu": "Kędzierzyn-Koźle",
    "kędzierzyn-koźle": "Kędzierzyn-Koźle",
    "świnoujściu": "Świnoujście",
    "swinoujsciu": "Świnoujście",
    "swinoujscie": "Świnoujście",
    "świnoujście": "Świnoujście",
    "kołobrzegu": "Kołobrzeg",
    "kolobrzegu": "Kołobrzeg",
    "kolobrzeg": "Kołobrzeg",
    "kołobrzeg": "Kołobrzeg",
    "helu": "Hel",
    "hel": "Hel",
    "wiśle": "Wisła",
    "wisle": "Wisła",
    "wisła": "Wisła",
    "wisla": "Wisła",
    "karpaczu": "Karpacz",
    "karpacz": "Karpacz",
    "ustce": "Ustka",
    "ustka": "Ustka",
    "łebie": "Łeba",
    "lebie": "Łeba",
    "łeba": "Łeba",
    "leba": "Łeba",
    "tarnowie": "Tarnów",
    "tarnowa": "Tarnów",
    "tarnow": "Tarnów",
    "tarnów": "Tarnów",
    "koszalinie": "Koszalin",
    "koszalina": "Koszalin",
    "koszalin": "Koszalin",
    "kaliszu": "Kalisz",
    "kalisza": "Kalisz",
    "kalisz": "Kalisz",
    "legnicy": "Legnica",
    "legnica": "Legnica",
    "płocku": "Płock",
    "plocku": "Płock",
    "plock": "Płock",
    "płock": "Płock",
    "elblągu": "Elbląg",
    "elblagu": "Elbląg",
    "elblag": "Elbląg",
    "elbląg": "Elbląg",
    "wałbrzychu": "Wałbrzych",
    "walbrzychu": "Wałbrzych",
    "walbrzych": "Wałbrzych",
    "wałbrzych": "Wałbrzych",
    "włocławku": "Włocławek",
    "wloclawku": "Włocławek",
    "wloclawek": "Włocławek",
    "włocławek": "Włocławek",
    "chorzowie": "Chorzów",
    "chorzowa": "Chorzów",
    "chorzow": "Chorzów",
    "chorzów": "Chorzów",
    "mysłowicach": "Mysłowice",
    "myslowicach": "Mysłowice",
    "mysłowice": "Mysłowice",
    "jaworznie": "Jaworzno",
    "jaworzno": "Jaworzno",
    "cieszynie": "Cieszyn",
    "cieszyn": "Cieszyn",
    "żywcu": "Żywiec",
    "zywcu": "Żywiec",
    "żywiec": "Żywiec",
    "zywiec": "Żywiec",
    "ustroniu": "Ustroń",
    "ustroń": "Ustroń",
    "ustron": "Ustroń",
    "suwałkach": "Suwałki",
    "suwalkach": "Suwałki",
    "suwałki": "Suwałki",
    "suwalki": "Suwałki",
    "zamościu": "Zamość",
    "zamosciu": "Zamość",
    "zamość": "Zamość",
    "zamosc": "Zamość",
    "przemyślu": "Przemyśl",
    "przemyslu": "Przemyśl",
    "przemyśl": "Przemyśl",
    "przemysl": "Przemyśl",
    "koninie": "Konin",
    "konin": "Konin",
    "pile": "Piła",
    "piła": "Piła",
    "siedlcach": "Siedlce",
    "siedlce": "Siedlce",
    "inowrocławiu": "Inowrocław",
    "inowroclawiu": "Inowrocław",
    "inowrocław": "Inowrocław",
    "gnieźnie": "Gniezno",
    "gnieznie": "Gniezno",
    "gniezno": "Gniezno",
    "lesznie": "Leszno",
    "leszno": "Leszno",
    "głogowie": "Głogów",
    "glogowie": "Głogów",
    "głogów": "Głogów",
    "tomaszowie": "Tomaszów",
    "piotrkowie": "Piotrków Trybunalski",
    "piotrkowie trybunalskim": "Piotrków Trybunalski",
    "mielcu": "Mielec",
    "mielec": "Mielec",
    "sanoku": "Sanok",
    "sanok": "Sanok",
    "kłodzku": "Kłodzko",
    "klodzku": "Kłodzko",
    "kłodzko": "Kłodzko",
    "wieliczce": "Wieliczka",
    "wieliczka": "Wieliczka",
    "bochni": "Bochnia",
    "bochnia": "Bochnia",
    "oświęcimiu": "Oświęcim",
    "oswiecimiu": "Oświęcim",
    "oświęcim": "Oświęcim",
    "wadowicach": "Wadowice",
    "wadowice": "Wadowice",
    "białce tatrzańskiej": "Białka Tatrzańska",
    "bialce tatrzanskiej": "Białka Tatrzańska",
    "bukowinie tatrzańskiej": "Bukowina Tatrzańska",
    "bukowinie tatrzanskiej": "Bukowina Tatrzańska",
    "londynie": "Londyn",
    "londynu": "Londyn",
    "londyn": "Londyn",
    "london": "London",
    "paryżu": "Paryż",
    "paryzu": "Paryż",
    "paryża": "Paryż",
    "paryz": "Paryż",
    "paryż": "Paryż",
    "paris": "Paris",
    "rzymie": "Rzym",
    "rzymu": "Rzym",
    "rzym": "Rzym",
    "rome": "Rome",
    "madrycie": "Madryt",
    "madrytu": "Madryt",
    "madryt": "Madryt",
    "madrid": "Madrid",
    "wiedniu": "Wiedeń",
    "wiednia": "Wiedeń",
    "wieden": "Wiedeń",
    "wiedeń": "Wiedeń",
    "vienna": "Vienna",
    "pradze": "Praga",
    "pragi": "Praga",
    "praga": "Praga",
    "prague": "Prague",
    "tokio": "Tokio",
    "tokyo": "Tokyo",
    "bratysławie": "Bratysława",
    "bratyslawie": "Bratysława",
    "bratysława": "Bratysława",
    "budapeszcie": "Budapeszt",
    "budapesztu": "Budapeszt",
    "budapeszt": "Budapeszt",
    "monachium": "Monachium",
    "barcelonie": "Barcelona",
    "barcelona": "Barcelona",
    "mediolanie": "Mediolan",
    "mediolan": "Mediolan",
    "amsterdamie": "Amsterdam",
    "amsterdam": "Amsterdam",
    "brukseli": "Bruksela",
    "bruksela": "Bruksela",
    "lizbonie": "Lizbona",
    "lizbona": "Lizbona",
    "atenach": "Ateny",
    "ateny": "Ateny",
    "sztokholmie": "Sztokholm",
    "sztokholm": "Sztokholm",
    "oslo": "Oslo",
    "kopenhadze": "Kopenhaga",
    "kopenhaga": "Kopenhaga",
    "dublinie": "Dublin",
    "dublin": "Dublin",
    "zurychu": "Zurych",
    "zurych": "Zurych",
    "genewie": "Genewa",
    "genewa": "Genewa",
    "ruda śląska": "Ruda Śląska",
    "ruda slaska": "Ruda Śląska",
    "rudzie śląskiej": "Ruda Śląska",
    "rudzie slaskiej": "Ruda Śląska",
    "dąbrowa górnicza": "Dąbrowa Górnicza",
    "dabrowa gornicza": "Dąbrowa Górnicza",
    "dąbrowie górniczej": "Dąbrowa Górnicza",
    "dabrowie gorniczej": "Dąbrowa Górnicza",
    "stargard": "Stargard",
    "stargardzie": "Stargard",
    "ostrowiec świętokrzyski": "Ostrowiec Świętokrzyski",
    "ostrowiec swietokrzyski": "Ostrowiec Świętokrzyski",
    "ostrowcu świętokrzyskim": "Ostrowiec Świętokrzyski",
    "ostrowcu swietokrzyskim": "Ostrowiec Świętokrzyski",
    "siemianowice śląskie": "Siemianowice Śląskie",
    "siemianowice slaskie": "Siemianowice Śląskie",
    "siemianowicach śląskich": "Siemianowice Śląskie",
    "siemianowicach slaskich": "Siemianowice Śląskie",
    "piotrków trybunalski": "Piotrków Trybunalski",
    "piotrkow trybunalski": "Piotrków Trybunalski",
    "grudziądz": "Grudziądz",
    "grudziadz": "Grudziądz",
    "grudziądzu": "Grudziądz",
    "grudziadzu": "Grudziądz",
    "słupsk": "Słupsk",
    "slupsk": "Słupsk",
    "słupsku": "Słupsk",
    "slupsku": "Słupsk",
    "lubin": "Lubin",
    "lubinie": "Lubin",
    "pabianice": "Pabianice",
    "pabianicach": "Pabianice",
    "międzyzdroje": "Międzyzdroje",
    "miedzyzdroje": "Międzyzdroje",
    "międzyzdrojach": "Międzyzdroje",
    "miedzyzdrojach": "Międzyzdroje",
}

KNOWN_NOMINATIVE_CITIES: frozenset[str] = frozenset(v.lower() for v in IRREGULAR_LOCATIVES.values())

# Ordered deterministic suffix rules for Polish locative -> nominative candidates.
_WORD_SUFFIX_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("owie", ("ów", "owo", "owa")),
    ("wiu", ("w", "wie")),
    ("niu", ("ń", "n", "nie")),
    ("miu", ("m", "mie")),
    ("biu", ("b", "bie")),
    ("piu", ("p", "pie")),
    ("czu", ("cz", "cze")),
    ("szu", ("sz", "sze")),
    ("rzu", ("rz", "rze", "r")),
    ("dżu", ("dż",)),
    ("żu", ("ż", "że")),
    ("cu", ("iec", "ec", "c")),
    ("lu", ("l", "le")),
    ("ku", ("k", "ko", "ek")),
    ("gu", ("g", "go")),
    ("chu", ("ch", "cho")),
    ("ach", ("e", "y", "i", "a")),
    ("em", ("e", "o")),
    ("nie", ("n", "no", "na")),
    ("mie", ("m", "ma", "mo")),
    ("wie", ("wa", "wo", "w")),
    ("bie", ("ba", "b")),
    ("pie", ("pa", "p")),
    ("fie", ("fa", "f")),
    ("dzie", ("d", "da", "do")),
    ("zie", ("za", "z")),
    ("sie", ("sa", "s")),
    ("cie", ("t", "ta", "to")),
    ("czce", ("czka",)),
    ("szce", ("szka",)),
    ("rce", ("rka",)),
    ("lce", ("lka",)),
    ("ce", ("ka", "ca")),
    ("dze", ("ga",)),
    ("sze", ("sza",)),
    ("rze", ("ra",)),
    ("że", ("ża",)),
    ("cy", ("ca", "ce", "k")),
    ("dzi", ("dź", "da")),
    ("ni", ("nia", "ń")),
    ("li", ("la",)),
    ("u", ("", "o", "e")),
)

_ADJ_LOCATIVE_RULES: tuple[tuple[str, str], ...] = (
    ("ich", "e"),
    ("ych", "e"),
    ("ym", "y"),
    ("im", "i"),
    ("ej", "a"),
)

_LEADING_PREP_RE = re.compile(r"^(?:w|we|na|z|ze|do|dla|in|for|at)\s+", re.IGNORECASE)

# Words that stop a city name phrase or must never be treated as a city name.
_STOP_AND_TEMPORAL_WORDS: frozenset[str] = frozenset(
    {
        "w",
        "we",
        "na",
        "dla",
        "do",
        "z",
        "ze",
        "od",
        "po",
        "o",
        "u",
        "przy",
        "nad",
        "pod",
        "za",
        "przez",
        "i",
        "oraz",
        "albo",
        "lub",
        "ale",
        "czy",
        "jak",
        "jaka",
        "jaki",
        "jakie",
        "kiedy",
        "gdzie",
        "ile",
        "co",
        "to",
        "tam",
        "tu",
        "tutaj",
        "będzie",
        "bedzie",
        "jest",
        "była",
        "byla",
        "było",
        "bylo",
        "będą",
        "beda",
        "ma",
        "być",
        "byc",
        "pogoda",
        "pogodę",
        "pogode",
        "pogody",
        "prognoza",
        "prognozę",
        "prognoze",
        "temperatura",
        "temperaturę",
        "temperature",
        "deszcz",
        "śnieg",
        "snieg",
        "wiatr",
        "słońce",
        "slonce",
        "wschód",
        "wschod",
        "zachód",
        "zachod",
        "wschodzi",
        "zachodzi",
        "pada",
        "padać",
        "padac",
        "stopni",
        "stopnie",
        "dziś",
        "dzis",
        "dzisiaj",
        "jutro",
        "pojutrze",
        "wczoraj",
        "teraz",
        "obecnie",
        "aktualnie",
        "rano",
        "wieczorem",
        "popołudniu",
        "popoludniu",
        "południe",
        "poludnie",
        "nocy",
        "nocą",
        "noca",
        "dzień",
        "dzien",
        "weekend",
        "weekendzie",
        "tydzień",
        "tydzien",
        "tygodniu",
        "miesiąc",
        "miesiac",
        "miesiącu",
        "miesiacu",
        "rok",
        "roku",
        "poniedziałek",
        "poniedzialek",
        "wtorek",
        "środę",
        "srode",
        "środa",
        "sroda",
        "czwartek",
        "piątek",
        "piatek",
        "sobotę",
        "sobote",
        "sobota",
        "niedzielę",
        "niedziele",
        "niedziela",
        "styczniu",
        "lutym",
        "marcu",
        "kwietniu",
        "maju",
        "czerwcu",
        "lipcu",
        "sierpniu",
        "wrześniu",
        "wrzesniu",
        "październiku",
        "pazdzierniku",
        "listopadzie",
        "grudniu",
        "today",
        "tomorrow",
        "now",
        "tonight",
        "morning",
        "evening",
        "weather",
        "forecast",
        "rain",
        "snow",
        "sun",
        "sunrise",
        "sunset",
        "in",
        "for",
        "at",
        "on",
        "the",
        "is",
        "will",
        "be",
    }
)

_NON_LOCATION_WORDS: frozenset[str] = frozenset(
    {
        "domu",
        "salonie",
        "sypialni",
        "kuchni",
        "pokoju",
        "łazience",
        "lazience",
        "garażu",
        "garazu",
        "biurze",
        "piwnicy",
        "strychu",
        "ogrodzie",
        "balkonie",
        "tarasie",
        "dworze",
        "zewnątrz",
        "zewnatrz",
        "środku",
        "srodku",
        "kalendarzu",
        "internecie",
        "sieci",
        "pliku",
        "pamięci",
        "pamieci",
        "minutniku",
        "hermesie",
        "wikipedii",
        "wiadomościach",
        "wiadomosciach",
        "telewizji",
        "radiu",
        "pracy",
        "szkole",
        "sklepie",
        "mieście",
        "miescie",
        "miejscowości",
        "miejscowosci",
        "tym",
        "tej",
        "tego",
        "przyszłym",
        "przyszlym",
        "zeszłym",
        "zeszlym",
        "następnym",
        "nastepnym",
        "house",
        "home",
        "room",
        "kitchen",
        "bedroom",
        "living",
        "office",
    }
)

_MULTIWORD_PREFIXES: frozenset[str] = frozenset(
    {
        "nowy",
        "nowym",
        "nowego",
        "nowa",
        "nowej",
        "stary",
        "starym",
        "starego",
        "stara",
        "starej",
        "wielki",
        "wielkim",
        "wielkiej",
        "wielka",
        "mały",
        "małym",
        "małej",
        "zielona",
        "zielonej",
        "jelenia",
        "jeleniej",
        "szklarska",
        "szklarskiej",
        "czarna",
        "czarnej",
        "czarny",
        "czarnym",
        "biała",
        "białej",
        "biały",
        "białym",
        "bielsku",
        "bielsko",
        "krynicy",
        "krynica",
        "jastrzębiu",
        "jastrzebiu",
        "gorzowie",
        "gorzów",
        "ostrowie",
        "ostrów",
        "kędzierzynie",
        "kedzierzynie",
        "bukowinie",
        "bukowina",
        "białce",
        "bialce",
        "piotrkowie",
        "ruda",
        "rudzie",
        "dąbrowa",
        "dąbrowie",
        "dabrowa",
        "dabrowie",
        "ostrowiec",
        "ostrowcu",
        "siemianowice",
        "siemianowicach",
        "new",
        "los",
        "las",
        "san",
        "saint",
        "st",
    }
)


def _strip_accents(text: str) -> str:
    pl_map = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    decomposed = unicodedata.normalize("NFKD", text.translate(pl_map))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _title_case_place(text: str) -> str:
    parts = text.strip().split()
    out: list[str] = []
    for part in parts:
        if "-" in part:
            out.append("-".join(seg[:1].upper() + seg[1:] for seg in part.split("-") if seg))
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out)


def _lemmatize_word(word: str) -> list[str]:
    low = word.lower()
    if low in IRREGULAR_LOCATIVES:
        return [IRREGULAR_LOCATIVES[low]]
    candidates: list[str] = []
    for suffix, replacements in _WORD_SUFFIX_RULES:
        if low.endswith(suffix) and len(low) > len(suffix) + 1:
            stem = low[: -len(suffix)]
            for repl in replacements:
                cand = _title_case_place(stem + repl)
                if cand and cand not in candidates:
                    candidates.append(cand)
    return candidates


def _lemmatize_phrase(phrase: str) -> list[str]:
    cleaned = _LEADING_PREP_RE.sub("", phrase.strip()).strip(" ?!.,;:\"'")
    if not cleaned:
        return []
    low = cleaned.lower()
    if low in IRREGULAR_LOCATIVES:
        return [IRREGULAR_LOCATIVES[low]]

    # Handle hyphenated two-part names (e.g. "Bielsku-Białej")
    if "-" in cleaned and " " not in cleaned:
        parts = [p for p in cleaned.split("-") if p]
        if len(parts) == 2:
            c1 = _lemmatize_word(parts[0]) or [_title_case_place(parts[0])]
            c2: list[str] = []
            for suf, repl in _ADJ_LOCATIVE_RULES:
                if parts[1].lower().endswith(suf) and len(parts[1]) > len(suf) + 1:
                    c2.append(_title_case_place(parts[1][: -len(suf)] + repl))
            c2.extend(_lemmatize_word(parts[1]))
            if not c2:
                c2.append(_title_case_place(parts[1]))
            combined: list[str] = []
            for p1 in c1[:2]:
                for p2 in c2[:2]:
                    cand = f"{p1}-{p2}"
                    if cand not in combined:
                        combined.append(cand)
            return combined

    words = cleaned.split()
    if len(words) == 1:
        return _lemmatize_word(words[0])
    if len(words) == 2:
        w1, w2 = words
        w1_low = w1.lower()
        w1_cands: list[str] = []
        for suf, repl in _ADJ_LOCATIVE_RULES:
            if w1_low.endswith(suf) and len(w1_low) > len(suf) + 1:
                w1_cands.append(_title_case_place(w1_low[: -len(suf)] + repl))
        if not w1_cands:
            w1_cands = _lemmatize_word(w1) or [_title_case_place(w1)]
        w2_cands: list[str] = []
        for suf, repl in _ADJ_LOCATIVE_RULES:
            if w2_low.endswith(suf) and len(w2_low) > len(suf) + 1:
                w2_cands.append(_title_case_place(w2_low[: -len(suf)] + repl))
        w2_cands.extend(_lemmatize_word(w2))
        if not w2_cands:
            w2_cands.append(_title_case_place(w2))
        combined = []
        for c1 in w1_cands[:2]:
            for c2 in w2_cands[:3]:
                cand = f"{c1} {c2}"
                if cand not in combined:
                    combined.append(cand)
        return combined
    return []


def normalize_location(name: str) -> str:
    """Normalize a Polish (possibly locative/inflected) place name to nominative."""
    cleaned = _LEADING_PREP_RE.sub("", (name or "").strip()).strip(" ?!.,;:\"'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    low = cleaned.lower()
    if low in IRREGULAR_LOCATIVES:
        return IRREGULAR_LOCATIVES[low]
    lemmas = _lemmatize_phrase(cleaned)
    if lemmas:
        return lemmas[0]
    return _title_case_place(cleaned)


def location_candidates(name: str) -> list[str]:
    """Return ordered geocoding candidates: name as given first, then normalized/lemmatized forms."""
    raw = (name or "").strip().strip(" ?!.,;:\"'")
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return []
    stripped = _LEADING_PREP_RE.sub("", raw).strip()
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(val: str) -> None:
        v = val.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            candidates.append(v)

    # 1. Try the name as given (without any accidental leading preposition)
    _add(stripped or raw)

    # 2. If already a known nominative city, only add ASCII fallback if needed
    low = (stripped or raw).lower()
    if low in IRREGULAR_LOCATIVES:
        _add(IRREGULAR_LOCATIVES[low])

    if low not in KNOWN_NOMINATIVE_CITIES:
        for lemma in _lemmatize_phrase(stripped or raw):
            _add(lemma)

    # 3. Also add ASCII-folded variant of the primary normalized form if it has diacritics
    primary = normalize_location(stripped or raw)
    _add(primary)
    ascii_primary = _strip_accents(primary)
    if ascii_primary != primary:
        _add(ascii_primary)

    return candidates


_PREP_ITER_RE = re.compile(r"\b(w|we|na|dla|in|for|at)\s+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+(?:-[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)*")


def extract_location_from_utterance(utterance: str) -> str | None:
    """Extract and normalize a city/place mentioned in a user utterance (e.g. 'w Krakowie', 'w Nowym Targu')."""
    text = (utterance or "").strip()
    if not text:
        return None
    # Strip trailing year expressions like "w 1997 roku" so years are never treated as places
    for match in _PREP_ITER_RE.finditer(text):
        tail = text[match.end() :]
        # Only inspect tokens up to the first sentence/clause punctuation
        clause = re.split(r"[?.!,;:()\[\]\"]", tail, maxsplit=1)[0]
        tokens = _TOKEN_RE.findall(clause)
        if not tokens:
            continue
        valid_tokens: list[str] = []
        for tok in tokens[:3]:
            t_low = tok.lower()
            if t_low in _STOP_AND_TEMPORAL_WORDS or t_low in _NON_LOCATION_WORDS or t_low in PLACEHOLDER_LOCATIONS:
                break
            valid_tokens.append(tok)
        if not valid_tokens:
            continue

        candidate_raw = valid_tokens[0]
        if len(valid_tokens) >= 2:
            two_word = f"{valid_tokens[0]} {valid_tokens[1]}"
            two_low = two_word.lower()
            if (
                two_low in IRREGULAR_LOCATIVES
                or valid_tokens[0].lower() in _MULTIWORD_PREFIXES
                or (valid_tokens[0][:1].isupper() and valid_tokens[1][:1].isupper())
            ):
                candidate_raw = two_word

        if candidate_raw.lower() in _NON_LOCATION_WORDS or candidate_raw.lower() in PLACEHOLDER_LOCATIONS:
            continue

        return normalize_location(candidate_raw)

    # Fallback: bare known city name at the end of the utterance (e.g. "prognoza pogody na jutro wroclaw")
    words = _TOKEN_RE.findall(text)
    if len(words) >= 2:
        last_two = f"{words[-2]} {words[-1]}".lower()
        if last_two in IRREGULAR_LOCATIVES:
            return IRREGULAR_LOCATIVES[last_two]
    if words:
        last_one = words[-1].lower()
        if last_one in IRREGULAR_LOCATIVES and last_one not in _STOP_AND_TEMPORAL_WORDS:
            return IRREGULAR_LOCATIVES[last_one]

    return None


_ASSISTANT_WEATHER_PREFIX_RE = re.compile(
    r"^\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\s\-]{1,40}?),\s+(?:obecnie|dzisiaj|jutro|currently|today|tomorrow)\s*:",
    re.IGNORECASE,
)
_ASSISTANT_SUN_PREFIX_RE = re.compile(
    r"(?:W\s+miejscowości|In)\s+([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\s\-]{1,40}?)[,\s]+(?:słońce\s+wschodzi|sunrise\s+is)",
    re.IGNORECASE,
)
_WEATHER_HISTORY_CUE_RE = re.compile(
    r"pogod|deszcz|pada|temperatur|stopni|wiatr|opady|wsch[oó]d|zach[oó]d|s[lł]o[nń]c|weather|forecast|degrees|sunrise|sunset|"
    r"jak\s+(?:tam\s+)?(?:będzie|bedzie|jest|jutro|dzi[sś]|dzisiaj)\s+(?:w|we|na)\s+",
    re.IGNORECASE,
)


HISTORY_LOCATION_MESSAGES = 4


def extract_location_from_history(history: list[dict[str, Any]] | None) -> str | None:
    """Find the city from the most recent weather/sun turn in conversation history."""
    if not history:
        return None
    # The last two exchanges only: "a jutro?" continues the current talk, a question hours later gets the default.
    history = history[-HISTORY_LOCATION_MESSAGES:]
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        role = msg.get("role")
        if role == "assistant":
            m_w = _ASSISTANT_WEATHER_PREFIX_RE.match(content)
            if m_w:
                return normalize_location(m_w.group(1))
            m_s = _ASSISTANT_SUN_PREFIX_RE.search(content)
            if m_s:
                return normalize_location(m_s.group(1))
        if _WEATHER_HISTORY_CUE_RE.search(content) or (
            idx + 1 < len(history) and _WEATHER_HISTORY_CUE_RE.search(str(history[idx + 1].get("content") or ""))
        ):
            loc = extract_location_from_utterance(content)
            if loc:
                return loc
    return None


def resolve_tool_location(
    llm_location: str | None,
    utterance: str,
    default_location: str | None = None,
    *,
    history: list[dict[str, Any]] | None = None,
    last_location: str | None = None,
) -> str:
    """Resolve the target location for get_weather / get_sun from LLM arg, utterance, history, and config default."""
    default_loc = (default_location or get_default_location()).strip() or "Katowice"
    utt_loc = extract_location_from_utterance(utterance)
    raw_llm = str(llm_location or "").strip()
    is_placeholder = raw_llm.lower() in PLACEHOLDER_LOCATIONS

    if utt_loc:
        if is_placeholder:
            return utt_loc
        norm_llm = normalize_location(raw_llm)
        # If the LLM biased to default_location (e.g. "Katowice") while the user asked for another city,
        # or if the LLM only extracted half of a multi-word city (e.g. "Targu" instead of "Nowy Targ"),
        # prefer the city extracted from the utterance.
        if norm_llm.lower() == default_loc.lower() and utt_loc.lower() != default_loc.lower():
            return utt_loc
        if " " in utt_loc and " " not in norm_llm and norm_llm.lower() in utt_loc.lower():
            return utt_loc
        return norm_llm

    # No place mentioned in the current utterance
    hist_loc = extract_location_from_history(history) or last_location
    if is_placeholder:
        if hist_loc:
            return hist_loc
        return default_loc

    norm_llm = normalize_location(raw_llm)
    if norm_llm.lower() == default_loc.lower() and hist_loc:
        return hist_loc
    return norm_llm


WMO_CODES_PL = {
    0: "słonecznie",
    1: "głównie bezchmurnie",
    2: "częściowo pochmurno",
    3: "pochmurno",
    45: "mgła",
    48: "mgła osadzająca szadź",
    51: "lekka mżawka",
    53: "umiarkowana mżawka",
    55: "gęsta mżawka",
    56: "marznąca mżawka",
    57: "gęsta marznąca mżawka",
    61: "słaby deszcz",
    63: "umiarkowany deszcz",
    65: "silny deszcz",
    66: "marznący deszcz",
    67: "silny marznący deszcz",
    71: "słaby śnieg",
    73: "umiarkowany śnieg",
    75: "silny śnieg",
    77: "ziarna śniegu",
    80: "lekki przelotny deszcz",
    81: "umiarkowany przelotny deszcz",
    82: "gwałtowny przelotny deszcz",
    85: "lekki przelotny śnieg",
    86: "obfity przelotny śnieg",
    95: "burza",
    96: "burza z lekkim gradem",
    99: "burza z silnym gradem",
}

WMO_CODES_EN = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

_WEATHER_CACHE: dict[tuple[str, str, str], tuple[float, ToolResult]] = {}
CACHE_TTL_S = 600.0  # 10 minutes


def _degree_word_pl(val: int | float) -> str:
    n = round(val)
    return polish_declension(n, "stopień", "stopnie", "stopni")


def _names_match(returned_name: str, expected_name: str) -> bool:
    r = _strip_accents(returned_name).lower().strip()
    e = _strip_accents(expected_name).lower().strip()
    return r == e or r.startswith(e + " ") or e.startswith(r + " ")


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """GET with one quick retry on a 5xx: when overloaded Open-Meteo answers 503 to every few requests.
    A timeout is not retried: a second 2.5 s wait would overrun the 4 s tool budget."""
    res = await client.get(url)
    if res.status_code < 500:
        return res
    await asyncio.sleep(0.2)
    return await client.get(url)


async def get_weather(
    location: str | None = None,
    when: str = "now",
    lang: str = "pl",
    *,
    client: httpx.AsyncClient | None = None,
) -> ToolResult:
    """Get weather forecast from Open-Meteo with 10-minute cache."""
    raw_loc = (location or "").strip()
    if not raw_loc or raw_loc.lower() in PLACEHOLDER_LOCATIONS:
        loc = get_default_location()
    else:
        loc = raw_loc

    when_norm = when.lower().strip() if when else "now"
    if when_norm not in ("now", "today", "tomorrow"):
        when_norm = "now"

    norm_loc = normalize_location(loc) or loc
    cache_key = (norm_loc.lower(), when_norm, lang)
    now_ts = time.time()
    if cache_key in _WEATHER_CACHE:
        cached_time, cached_result = _WEATHER_CACHE[cache_key]
        if now_ts - cached_time < CACHE_TTL_S:
            return cached_result

    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=2.5)
        own_client = True

    try:
        candidates = location_candidates(loc) or [loc]
        canonical = IRREGULAR_LOCATIVES.get(loc.lower())
        chosen_hit: dict[str, Any] | None = None
        fallback_hit: dict[str, Any] | None = None

        for idx, candidate in enumerate(candidates):
            geo_url = (
                f"https://geocoding-api.open-meteo.com/v1/search?"
                f"name={urllib.parse.quote(candidate)}&count=1&language={lang}&format=json"
            )
            geo_res = await _get(client, geo_url)
            if geo_res.status_code != 200:
                err = f"Open-Meteo geocoding error HTTP {geo_res.status_code}"
                return ToolResult(data={"error": err}, speak="Nie udało się pobrać informacji o pogodzie.", error=err)

            geo_data = geo_res.json()
            results = geo_data.get("results")
            if not results:
                continue

            hit = results[0]
            # Guard against Open-Meteo false prefix matches on inflected Polish forms
            # (e.g. "Krakowie" -> "Krakowiec" in UA, "Łodzi" -> "Łodzia"): if we have a known
            # irregular canonical name that differs from the raw input and this first hit does
            # not match the canonical name, try the normalized candidate next.
            if (
                idx == 0
                and canonical
                and canonical.lower() != candidate.lower()
                and len(candidates) > 1
                and not _names_match(str(hit.get("name", "")), canonical)
            ):
                fallback_hit = hit
                continue

            chosen_hit = hit
            break

        if chosen_hit is None:
            chosen_hit = fallback_hit

        if chosen_hit is None:
            err = f"Nie znaleziono lokalizacji '{loc}'"
            speak = f"Nie znalazłam lokalizacji {loc}." if lang == "pl" else f"Could not find location {loc}."
            return ToolResult(data={"error": err}, speak=speak, error=err)

        matched_loc = chosen_hit.get("name", norm_loc)
        lat = chosen_hit["latitude"]
        lon = chosen_hit["longitude"]

        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,"
            f"apparent_temperature_min,precipitation_probability_max,wind_speed_10m_max,sunrise,sunset&"
            f"timezone=auto"
        )
        f_res = await _get(client, forecast_url)
        if f_res.status_code != 200:
            err = f"Open-Meteo forecast error HTTP {f_res.status_code}"
            return ToolResult(data={"error": err}, speak="Nie udało się pobrać informacji o pogodzie.", error=err)

        f_data = f_res.json()
        current = f_data.get("current", {})
        daily = f_data.get("daily", {})

        if when_norm == "now":
            temp_raw = current.get("temperature_2m", 0.0)
            feels_raw = current.get("apparent_temperature", temp_raw)
            temp = round(temp_raw, 1)
            feels_like = round(feels_raw, 1)
            temp_int = round(temp_raw)
            feels_int = round(feels_raw)
            w_code = current.get("weather_code", 0)
            wind = round(current.get("wind_speed_10m", 0.0))

            cond_text = (
                WMO_CODES_PL.get(w_code, "umiarkowane warunki")
                if lang == "pl"
                else WMO_CODES_EN.get(w_code, "moderate conditions")
            )

            data = {
                "location": matched_loc,
                "when": "now",
                "temperature": temp,
                "apparent_temperature": feels_like,
                "conditions": cond_text,
                "wind_speed_kmh": wind,
            }

            if lang == "en":
                raw_speak = (
                    f"{matched_loc}, currently: {temp_int} degrees, feels like {feels_int} degrees, "
                    f"{cond_text}, wind {wind} km/h."
                )
            else:
                deg_temp = _degree_word_pl(temp_int)
                deg_feels = _degree_word_pl(feels_int)
                raw_speak = (
                    f"{matched_loc}, obecnie: {temp_int} {deg_temp}, odczuwalna {feels_int} {deg_feels}, "
                    f"{cond_text}, wiatr {wind} km/h."
                )
        else:
            day_idx = 1 if when_norm == "tomorrow" else 0
            times = daily.get("time", [])
            if day_idx >= len(times):
                day_idx = 0

            target_date = times[day_idx] if day_idx < len(times) else ""
            t_max_raw = daily.get("temperature_2m_max", [0])[day_idx]
            t_min_raw = daily.get("temperature_2m_min", [0])[day_idx]
            t_max = round(t_max_raw, 1)
            t_min = round(t_min_raw, 1)
            t_max_int = round(t_max_raw)
            t_min_int = round(t_min_raw)
            w_code = daily.get("weather_code", [0])[day_idx]
            precip_prob = round(daily.get("precipitation_probability_max", [0])[day_idx])
            wind_max = round(daily.get("wind_speed_10m_max", [0])[day_idx])

            sunrise_raw = daily.get("sunrise", [""])[day_idx]
            sunset_raw = daily.get("sunset", [""])[day_idx]
            sunrise = sunrise_raw.split("T")[-1] if "T" in sunrise_raw else sunrise_raw
            sunset = sunset_raw.split("T")[-1] if "T" in sunset_raw else sunset_raw

            cond_text = (
                WMO_CODES_PL.get(w_code, "umiarkowane warunki")
                if lang == "pl"
                else WMO_CODES_EN.get(w_code, "moderate conditions")
            )

            data = {
                "location": matched_loc,
                "when": when_norm,
                "date": target_date,
                "temp_min": t_min,
                "temp_max": t_max,
                "conditions": cond_text,
                "precipitation_probability": precip_prob,
                "wind_speed_kmh": wind_max,
                "sunrise": sunrise,
                "sunset": sunset,
            }

            day_word_pl = "jutro" if when_norm == "tomorrow" else "dzisiaj"
            day_word_en = "tomorrow" if when_norm == "tomorrow" else "today"

            if lang == "en":
                raw_speak = (
                    f"{matched_loc}, {day_word_en}: temperature from {t_min_int} to {t_max_int} degrees, "
                    f"{cond_text}, chance of rain {precip_prob} percent, sunrise at {sunrise}, sunset at {sunset}."
                )
            else:
                deg_max = _degree_word_pl(t_max_int)
                raw_speak = (
                    f"{matched_loc}, {day_word_pl}: od {t_min_int} do {t_max_int} {deg_max}, "
                    f"{cond_text}, szansa na opady {precip_prob} procent, wschód słońca o {sunrise}, zachód o {sunset}."
                )

        speak = normalize_for_tts(raw_speak, lang)
        result = ToolResult(data=data, speak=speak)
        _WEATHER_CACHE[cache_key] = (now_ts, result)
        return result
    except Exception as exc:
        err = f"Błąd pobierania pogody: {exc!r}"  # repr: httpx timeouts have an empty str()
        return ToolResult(data={"error": err}, speak="Nie udało się pobrać informacji o pogodzie.", error=err)
    finally:
        if own_client:
            await client.aclose()
