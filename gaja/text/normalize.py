from __future__ import annotations

import re
import unicodedata
from typing import Any

import num2words

# --- Polish Declension Tables & Helpers ---

POLISH_CENTURIES_NOM: dict[str, str] = {
    "I": "pierwszy",
    "II": "drugi",
    "III": "trzeci",
    "IV": "czwarty",
    "V": "piąty",
    "VI": "szósty",
    "VII": "siódmy",
    "VIII": "ósmy",
    "IX": "dziewiąty",
    "X": "dziesiąty",
    "XI": "jedenasty",
    "XII": "dwunasty",
    "XIII": "trzynasty",
    "XIV": "czternasty",
    "XV": "piętnasty",
    "XVI": "szesnasty",
    "XVII": "siedemnasty",
    "XVIII": "osiemnasty",
    "XIX": "dziewiętnasty",
    "XX": "dwudziesty",
    "XXI": "dwudziesty pierwszy",
    "XXII": "dwudziesty drugi",
}

POLISH_CENTURIES_GEN: dict[str, str] = {
    "I": "pierwszego",
    "II": "drugiego",
    "III": "trzeciego",
    "IV": "czwartego",
    "V": "piątego",
    "VI": "szóstego",
    "VII": "siódmego",
    "VIII": "ósmego",
    "IX": "dziewiątego",
    "X": "dziesiątego",
    "XI": "jedenastego",
    "XII": "dwunastego",
    "XIII": "trzynastego",
    "XIV": "czternastego",
    "XV": "piętnastego",
    "XVI": "szesnastego",
    "XVII": "siedemnastego",
    "XVIII": "osiemnastego",
    "XIX": "dziewiętnastego",
    "XX": "dwudziestego",
    "XXI": "dwudziestego pierwszego",
    "XXII": "dwudziestego drugiego",
}

POLISH_CENTURIES_LOC: dict[str, str] = {
    "I": "pierwszym",
    "II": "drugim",
    "III": "trzecim",
    "IV": "czwartym",
    "V": "piątym",
    "VI": "szóstym",
    "VII": "siódmym",
    "VIII": "ósmym",
    "IX": "dziewiątym",
    "X": "dziesiątym",
    "XI": "jedenastym",
    "XII": "dwunastym",
    "XIII": "trzynastym",
    "XIV": "czternastym",
    "XV": "piętnastym",
    "XVI": "szesnastym",
    "XVII": "siedemnastym",
    "XVIII": "osiemnastym",
    "XIX": "dziewiętnastym",
    "XX": "dwudziestym",
    "XXI": "dwudziestym pierwszym",
    "XXII": "dwudziestym drugim",
}

POLISH_DAYS_ORD_GEN: dict[int, str] = {
    1: "pierwszego",
    2: "drugiego",
    3: "trzeciego",
    4: "czwartego",
    5: "piątego",
    6: "szóstego",
    7: "siódmego",
    8: "ósmego",
    9: "dziewiątego",
    10: "dziesiątego",
    11: "jedenastego",
    12: "dwunastego",
    13: "trzynastego",
    14: "czternastego",
    15: "piętnastego",
    16: "szesnastego",
    17: "siedemnastego",
    18: "osiemnastego",
    19: "dziewiętnastego",
    20: "dwudziestego",
    21: "dwudziestego pierwszego",
    22: "dwudziestego drugiego",
    23: "dwudziestego trzeciego",
    24: "dwudziestego czwartego",
    25: "dwudziestego piątego",
    26: "dwudziestego szóstego",
    27: "dwudziestego siódmego",
    28: "dwudziestego ósmego",
    29: "dwudziestego dziewiątego",
    30: "trzydziestego",
    31: "trzydziestego pierwszego",
}

POLISH_DAYS_ORD_LOC: dict[int, str] = {
    1: "pierwszym",
    2: "drugim",
    3: "trzecim",
    4: "czwartym",
    5: "piątym",
    6: "szóstym",
    7: "siódmym",
    8: "ósmym",
    9: "dziewiątym",
    10: "dziesiątym",
    11: "jedenastym",
    12: "dwunastym",
    13: "trzynastym",
    14: "czternastym",
    15: "piętnastym",
    16: "szesnastym",
    17: "siedemnastym",
    18: "osiemnastym",
    19: "dziewiętnastym",
    20: "dwudziestym",
    21: "dwudziestym pierwszym",
    22: "dwudziestym drugim",
    23: "dwudziestym trzecim",
    24: "dwudziestym czwartym",
    25: "dwudziestym piątym",
    26: "dwudziestym szóstym",
    27: "dwudziestym siódmym",
    28: "dwudziestym ósmym",
    29: "dwudziestym dziewiątym",
    30: "trzydziestym",
    31: "trzydziestym pierwszym",
}

TENS_ORD_GEN: dict[int, str] = {
    20: "dwudziestego",
    30: "trzydziestego",
    40: "czterdziestego",
    50: "pięćdziesiątego",
    60: "sześćdziesiątego",
    70: "siedemdziesiątego",
    80: "osiemdziesiątego",
    90: "dziewięćdziesiątego",
}

TENS_ORD_LOC: dict[int, str] = {
    20: "dwudziestym",
    30: "trzydziestym",
    40: "czterdziestym",
    50: "pięćdziesiątym",
    60: "sześćdziesiątym",
    70: "siedemdziesiątym",
    80: "osiemdziesiątym",
    90: "dziewięćdziesiątym",
}


POLISH_HOURS_NOM: dict[int, str] = {
    0: "północ",
    1: "pierwsza",
    2: "druga",
    3: "trzecia",
    4: "czwarta",
    5: "piąta",
    6: "szósta",
    7: "siódma",
    8: "ósma",
    9: "dziewiąta",
    10: "dziesiąta",
    11: "jedenasta",
    12: "dwunasta",
    13: "trzynasta",
    14: "czternasta",
    15: "piętnasta",
    16: "szesnasta",
    17: "siedemnasta",
    18: "osiemnasta",
    19: "dziewiętnasta",
    20: "dwudziesta",
    21: "dwudziesta pierwsza",
    22: "dwudziesta druga",
    23: "dwudziesta trzecia",
}

POLISH_HOURS_LOC: dict[int, str] = {
    0: "północy",
    1: "pierwszej",
    2: "drugiej",
    3: "trzeciej",
    4: "czwartej",
    5: "piątej",
    6: "szóstej",
    7: "siódmej",
    8: "ósmej",
    9: "dziewiątej",
    10: "dziesiątej",
    11: "jedenastej",
    12: "dwunastej",
    13: "trzynastej",
    14: "czternastej",
    15: "piętnastej",
    16: "szesnastej",
    17: "siedemnastej",
    18: "osiemnastej",
    19: "dziewiętnastej",
    20: "dwudziestej",
    21: "dwudziestej pierwszej",
    22: "dwudziestej drugiej",
    23: "dwudziestej trzeciej",
}

# Instrumental (godzina is feminine): "przed osiemnastą", "między siedemnastą a osiemnastą" (F3)
POLISH_HOURS_INST: dict[int, str] = {
    0: "północą",
    1: "pierwszą",
    2: "drugą",
    3: "trzecią",
    4: "czwartą",
    5: "piątą",
    6: "szóstą",
    7: "siódmą",
    8: "ósmą",
    9: "dziewiątą",
    10: "dziesiątą",
    11: "jedenastą",
    12: "dwunastą",
    13: "trzynastą",
    14: "czternastą",
    15: "piętnastą",
    16: "szesnastą",
    17: "siedemnastą",
    18: "osiemnastą",
    19: "dziewiętnastą",
    20: "dwudziestą",
    21: "dwudziestą pierwszą",
    22: "dwudziestą drugą",
    23: "dwudziestą trzecią",
}

PL_MONTHS = (
    "stycznia",
    "lutego",
    "marca",
    "kwietnia",
    "maja",
    "czerwca",
    "lipca",
    "sierpnia",
    "września",
    "października",
    "listopada",
    "grudnia",
)

# --- Polish Cardinal Declension Engine (0 - 999 999 999) ---

UNITS_NOM: dict[int, Any] = {
    0: "zero",
    1: {"m": "jeden", "f": "jedna", "n": "jedno"},
    2: {"m": "dwa", "f": "dwie", "n": "dwa"},
    3: "trzy",
    4: "cztery",
    5: "pięć",
    6: "sześć",
    7: "siedem",
    8: "osiem",
    9: "dziewięć",
}

UNITS_GEN: dict[int, Any] = {
    0: "zera",
    1: {"m": "jednego", "f": "jednej", "n": "jednego"},
    2: "dwóch",
    3: "trzech",
    4: "czterech",
    5: "pięciu",
    6: "sześciu",
    7: "siedmiu",
    8: "ośmiu",
    9: "dziewięciu",
}

UNITS_LOC: dict[int, Any] = {
    0: "zerze",
    1: {"m": "jednym", "f": "jednej", "n": "jednym"},
    2: "dwóch",
    3: "trzech",
    4: "czterech",
    5: "pięciu",
    6: "sześciu",
    7: "siedmiu",
    8: "ośmiu",
    9: "dziewięciu",
}

UNITS_INST: dict[int, Any] = {
    0: "zerem",
    1: {"m": "jednym", "f": "jedną", "n": "jednym"},
    2: {"m": "dwoma", "f": "dwiema", "n": "dwoma"},
    3: "trzema",
    4: "czterema",
    5: "pięcioma",
    6: "sześcioma",
    7: "siedmioma",
    8: "ośmioma",
    9: "dziewięcioma",
}

TEENS_NOM: dict[int, str] = {
    10: "dziesięć",
    11: "jedenaście",
    12: "dwanaście",
    13: "trzynaście",
    14: "czternaście",
    15: "piętnaście",
    16: "szesnaście",
    17: "siedemnaście",
    18: "osiemnaście",
    19: "dziewiętnaście",
}

TEENS_GEN: dict[int, str] = {
    10: "dziesięciu",
    11: "jedenastu",
    12: "dwunastu",
    13: "trzynastu",
    14: "czternastu",
    15: "piętnastu",
    16: "szesnastu",
    17: "siedemnastu",
    18: "osiemnastu",
    19: "dziewiętnastu",
}

TEENS_INST: dict[int, str] = {
    10: "dziesięcioma",
    11: "jedenastoma",
    12: "dwunastoma",
    13: "trzynastoma",
    14: "czternastoma",
    15: "piętnastoma",
    16: "szesnastoma",
    17: "siedemnastoma",
    18: "osiemnastoma",
    19: "dziewiętnastoma",
}

TENS_NOM: dict[int, str] = {
    20: "dwadzieścia",
    30: "trzydzieści",
    40: "czterdzieści",
    50: "pięćdziesiąt",
    60: "sześćdziesiąt",
    70: "siedemdziesiąt",
    80: "osiemdziesiąt",
    90: "dziewięćdziesiąt",
}

TENS_GEN: dict[int, str] = {
    20: "dwudziestu",
    30: "trzydziestu",
    40: "czterdziestu",
    50: "pięćdziesięciu",
    60: "sześćdziesięciu",
    70: "siedemdziesięciu",
    80: "osiemdziesięciu",
    90: "dziewięćdziesięciu",
}

TENS_INST: dict[int, str] = {
    20: "dwudziestoma",
    30: "trzydziestoma",
    40: "czterdziestoma",
    50: "pięćdziesięcioma",
    60: "sześćdziesięcioma",
    70: "siedemdziesięcioma",
    80: "osiemdziesięcioma",
    90: "dziewięćdziesięcioma",
}

HUNDREDS_NOM: dict[int, str] = {
    100: "sto",
    200: "dwieście",
    300: "trzysta",
    400: "czterysta",
    500: "pięćset",
    600: "sześćset",
    700: "siedemset",
    800: "osiemset",
    900: "dziewięćset",
}

HUNDREDS_GEN: dict[int, str] = {
    100: "stu",
    200: "dwustu",
    300: "trzystu",
    400: "czterystu",
    500: "pięciuset",
    600: "sześciuset",
    700: "siedmiuset",
    800: "ośmiuset",
    900: "dziewięciuset",
}

HUNDREDS_INST: dict[int, str] = {
    100: "stoma",
    200: "dwustoma",
    300: "trzystoma",
    400: "czterystoma",
    500: "pięciuset",
    600: "sześciuset",
    700: "siedmiuset",
    800: "ośmiuset",
    900: "dziewięciuset",
}


def _pl_cardinal_sub1000(
    n: int, case: str = "nom", gender: str = "m", is_compound: bool = False
) -> str:
    """Inflect cardinal number 0 <= n < 1000 into Polish grammatical case."""
    if n == 0 and not is_compound:
        if case == "gen":
            return "zera"
        if case == "loc":
            return "zerze"
        if case == "inst":
            return "zerem"
        return "zero"

    parts: list[str] = []
    h = (n // 100) * 100
    rem = n % 100

    if h > 0:
        if case == "nom":
            parts.append(HUNDREDS_NOM[h])
        elif case in ("gen", "loc"):
            parts.append(HUNDREDS_GEN[h])
        elif case == "inst":
            parts.append(HUNDREDS_INST[h])

    if rem == 0:
        return " ".join(parts)

    compound_now = is_compound or (h > 0)

    if rem < 10:
        if rem == 1 and compound_now:
            parts.append("jeden")
        elif rem == 1:
            if case == "nom":
                parts.append(UNITS_NOM[1][gender])
            elif case == "gen":
                parts.append(UNITS_GEN[1][gender])
            elif case == "loc":
                parts.append(UNITS_LOC[1][gender])
            elif case == "inst":
                parts.append(UNITS_INST[1][gender])
        elif rem == 2:
            if case == "nom":
                parts.append(UNITS_NOM[2][gender])
            elif case in ("gen", "loc"):
                parts.append(UNITS_GEN[2])
            elif case == "inst":
                parts.append(UNITS_INST[2][gender])
        else:
            if case == "nom":
                parts.append(UNITS_NOM[rem])
            elif case in ("gen", "loc"):
                parts.append(UNITS_GEN[rem])
            elif case == "inst":
                parts.append(UNITS_INST[rem])
    elif 10 <= rem < 20:
        if case == "nom":
            parts.append(TEENS_NOM[rem])
        elif case in ("gen", "loc"):
            parts.append(TEENS_GEN[rem])
        elif case == "inst":
            parts.append(TEENS_INST[rem])
    else:
        t = (rem // 10) * 10
        u = rem % 10
        if case == "nom":
            parts.append(TENS_NOM[t])
        elif case in ("gen", "loc"):
            parts.append(TENS_GEN[t])
        elif case == "inst":
            parts.append(TENS_INST[t])
        if u > 0:
            if u == 1:
                parts.append("jeden")
            elif u == 2:
                if case == "nom":
                    parts.append(UNITS_NOM[2][gender])
                elif case in ("gen", "loc"):
                    parts.append(UNITS_GEN[2])
                elif case == "inst":
                    parts.append(UNITS_INST[2][gender])
            else:
                if case == "nom":
                    parts.append(UNITS_NOM[u])
                elif case in ("gen", "loc"):
                    parts.append(UNITS_GEN[u])
                elif case == "inst":
                    parts.append(UNITS_INST[u])
    return " ".join(parts)


def _pl_thousands_decl(th: int, case: str) -> str:
    """Inflect thousands component for Polish numbers."""
    if th == 1:
        if case == "nom":
            return "tysiąc"
        if case == "gen":
            return "tysiąca"
        if case == "loc":
            return "tysiącu"
        if case == "inst":
            return "tysiącem"

    if case == "nom":
        mod10 = th % 10
        mod100 = th % 100
        t_word = "tysiące" if (mod10 in (2, 3, 4) and mod100 not in (12, 13, 14)) else "tysięcy"
        th_str = _pl_cardinal_sub1000(th, "nom", "m")
        return f"{th_str} {t_word}"
    elif case == "gen":
        th_str = _pl_cardinal_sub1000(th, "gen", "m")
        return f"{th_str} tysięcy"
    elif case == "loc":
        th_str = _pl_cardinal_sub1000(th, "loc", "m")
        return f"{th_str} tysiącach"
    elif case == "inst":
        th_str = _pl_cardinal_sub1000(th, "inst", "m")
        return f"{th_str} tysiącami"
    return "tysięcy"


def pl_cardinal(n: int, case: str = "nom", gender: str = "m") -> str:
    """Inflect Polish cardinal number 0 <= n < 1 000 000 000 000 into case ('nom', 'gen', 'loc', 'inst')."""
    if n < 0:
        pos_word = pl_cardinal(abs(n), case=case, gender=gender)
        return f"minus {pos_word}"
    if n >= 1_000_000_000 and n < 1_000_000_000_000:
        # Miliards (F4): without this branch a ten-digit number read as "tysiąc milionów".
        mld = n // 1_000_000_000
        rem = n % 1_000_000_000
        if mld == 1:
            if case == "nom":
                mld_part = "miliard"
            elif case == "gen":
                mld_part = "miliarda"
            elif case == "loc":
                mld_part = "miliardzie"
            elif case == "inst":
                mld_part = "miliardem"
            else:
                mld_part = "miliard"
        else:
            if case == "nom":
                mod10 = mld % 10
                mod100 = mld % 100
                mld_word = "miliardy" if (mod10 in (2, 3, 4) and mod100 not in (12, 13, 14)) else "miliardów"
                mld_part = f"{pl_cardinal(mld, 'nom', 'm')} {mld_word}"
            elif case == "gen":
                mld_part = f"{pl_cardinal(mld, 'gen', 'm')} miliardów"
            elif case == "loc":
                mld_part = f"{pl_cardinal(mld, 'loc', 'm')} miliardach"
            elif case == "inst":
                mld_part = f"{pl_cardinal(mld, 'inst', 'm')} miliardami"
            else:
                mld_part = f"{pl_cardinal(mld, 'nom', 'm')} miliardów"
        if rem == 0:
            return mld_part
        return f"{mld_part} {pl_cardinal(rem, case=case, gender=gender)}"

    if n >= 1_000_000:
        mil = n // 1_000_000
        rem = n % 1_000_000
        if mil == 1:
            if case == "nom":
                mil_part = "milion"
            elif case == "gen":
                mil_part = "miliona"
            elif case == "loc":
                mil_part = "milionie"
            elif case == "inst":
                mil_part = "milionem"
            else:
                mil_part = "milion"
        else:
            if case == "nom":
                mod10 = mil % 10
                mod100 = mil % 100
                m_word = "miliony" if (mod10 in (2, 3, 4) and mod100 not in (12, 13, 14)) else "milionów"
                mil_part = f"{pl_cardinal(mil, 'nom', 'm')} {m_word}"
            elif case == "gen":
                mil_part = f"{pl_cardinal(mil, 'gen', 'm')} milionów"
            elif case == "loc":
                mil_part = f"{pl_cardinal(mil, 'loc', 'm')} milionach"
            elif case == "inst":
                mil_part = f"{pl_cardinal(mil, 'inst', 'm')} milionami"
            else:
                mil_part = f"{pl_cardinal(mil, 'nom', 'm')} milionów"
        if rem == 0:
            return mil_part
        rem_part = pl_cardinal(rem, case=case, gender=gender)
        return f"{mil_part} {rem_part}"

    if n < 1000:
        return _pl_cardinal_sub1000(n, case=case, gender=gender)
    th = n // 1000
    rem = n % 1000
    th_part = _pl_thousands_decl(th, case=case)
    if rem == 0:
        return th_part
    rem_part = _pl_cardinal_sub1000(rem, case=case, gender=gender, is_compound=True)
    return f"{th_part} {rem_part}"


def pl_cardinal_float(val: float, case: str = "nom", gender: str = "m") -> str:
    """Inflect Polish decimal number into words using colloquial 'przecinek' form."""
    sign = "minus " if val < 0 else ""
    val_abs = abs(val)
    val_str = f"{val_abs:.6f}".rstrip("0").rstrip(".")
    if "." not in val_str:
        return f"{sign}{pl_cardinal(int(val_abs), case=case, gender=gender)}"
    parts = val_str.split(".")
    int_p = int(parts[0])
    dec_p = int(parts[1])
    w_int = pl_cardinal(int_p, case=case, gender=gender)
    w_dec = pl_cardinal(dec_p, case=case, gender="m")
    return f"{sign}{w_int} przecinek {w_dec}"


def inflect_number(n: int | float, case: str = "nom", gender: str = "m") -> str:
    """High-level inflector for int or float numbers in Polish."""
    if isinstance(n, float) and not n.is_integer():
        return pl_cardinal_float(n, case=case, gender=gender)
    return pl_cardinal(int(n), case=case, gender=gender)


# --- Data-driven Polish Noun Declensions & Agreement Table ---
# Dict noun -> (sg_nom, pl_nom, pl_gen, sg_gen, gender)
NOUN_DECLENSIONS: dict[str, tuple[str, str, str, str, str]] = {
    # Tysiąc (m)
    "tysiąc": ("tysiąc", "tysiące", "tysięcy", "tysiąca", "m"),
    "tysiące": ("tysiąc", "tysiące", "tysięcy", "tysiąca", "m"),
    "tysięcy": ("tysiąc", "tysiące", "tysięcy", "tysiąca", "m"),
    # Milion (m)
    "milion": ("milion", "miliony", "milionów", "miliona", "m"),
    "miliony": ("milion", "miliony", "milionów", "miliona", "m"),
    "milionów": ("milion", "miliony", "milionów", "miliona", "m"),
    # Miliard (m): "2 mld osób" -> "dwa miliardy osób", "1,5 mld zł" -> "... miliarda złotych"
    "miliard": ("miliard", "miliardy", "miliardów", "miliarda", "m"),
    "miliardy": ("miliard", "miliardy", "miliardów", "miliarda", "m"),
    "miliardów": ("miliard", "miliardy", "miliardów", "miliarda", "m"),
    # Dzień (m)
    "dzień": ("dzień", "dni", "dni", "dnia", "m"),
    "dni": ("dzień", "dni", "dni", "dnia", "m"),
    "dnia": ("dzień", "dni", "dni", "dnia", "m"),
    # Stopień (m)
    "stopień": ("stopień", "stopnie", "stopni", "stopnia", "m"),
    "stopnie": ("stopień", "stopnie", "stopni", "stopnia", "m"),
    "stopni": ("stopień", "stopnie", "stopni", "stopnia", "m"),
    "stopnia": ("stopień", "stopnie", "stopni", "stopnia", "m"),
    # Procent (m)
    "procent": ("procent", "procent", "procent", "procent", "m"),
    "procenty": ("procent", "procent", "procent", "procent", "m"),
    "procenta": ("procent", "procent", "procent", "procent", "m"),
    # Minuta (f)
    "minuta": ("minuta", "minuty", "minut", "minuty", "f"),
    "minuty": ("minuta", "minuty", "minut", "minuty", "f"),
    "minut": ("minuta", "minuty", "minut", "minuty", "f"),
    # Godzina (f)
    "godzina": ("godzina", "godziny", "godzin", "godziny", "f"),
    "godziny": ("godzina", "godziny", "godzin", "godziny", "f"),
    "godzin": ("godzina", "godziny", "godzin", "godziny", "f"),
    # Kilometr (m)
    "kilometr": ("kilometr", "kilometry", "kilometrów", "kilometra", "m"),
    "kilometry": ("kilometr", "kilometry", "kilometrów", "kilometra", "m"),
    "kilometrów": ("kilometr", "kilometry", "kilometrów", "kilometra", "m"),
    "kilometra": ("kilometr", "kilometry", "kilometrów", "kilometra", "m"),
    # Złoty (m)
    "złoty": ("złoty", "złote", "złotych", "złotego", "m"),
    "złote": ("złoty", "złote", "złotych", "złotego", "m"),
    "złotych": ("złoty", "złote", "złotych", "złotego", "m"),
    "złotego": ("złoty", "złote", "złotych", "złotego", "m"),
    # Grosz (m)
    "grosz": ("grosz", "grosze", "groszy", "grosza", "m"),
    "grosze": ("grosz", "grosze", "groszy", "grosza", "m"),
    "groszy": ("grosz", "grosze", "groszy", "grosza", "m"),
    "grosza": ("grosz", "grosze", "groszy", "grosza", "m"),
    # Euro (n - invariant)
    "euro": ("euro", "euro", "euro", "euro", "n"),
    # Dolar (m)
    "dolar": ("dolar", "dolary", "dolarów", "dolara", "m"),
    "dolary": ("dolar", "dolary", "dolarów", "dolara", "m"),
    "dolarów": ("dolar", "dolary", "dolarów", "dolara", "m"),
    "dolara": ("dolar", "dolary", "dolarów", "dolara", "m"),
    # Korona (f) - CZK, NOK, SEK
    "korona": ("korona", "korony", "koron", "korony", "f"),
    "korony": ("korona", "korony", "koron", "korony", "f"),
    "koron": ("korona", "korony", "koron", "korony", "f"),
    # Jen (m) - JPY
    "jen": ("jen", "jeny", "jenów", "jena", "m"),
    "jeny": ("jen", "jeny", "jenów", "jena", "m"),
    "jenów": ("jen", "jeny", "jenów", "jena", "m"),
    "jena": ("jen", "jeny", "jenów", "jena", "m"),
    # Bajt (m)
    "bajt": ("bajt", "bajty", "bajtów", "bajta", "m"),
    "bajty": ("bajt", "bajty", "bajtów", "bajta", "m"),
    "bajtów": ("bajt", "bajty", "bajtów", "bajta", "m"),
    "bajta": ("bajt", "bajty", "bajtów", "bajta", "m"),
    # Bit (m)
    "bit": ("bit", "bity", "bitów", "bita", "m"),
    "bity": ("bit", "bity", "bitów", "bita", "m"),
    "bitów": ("bit", "bity", "bitów", "bita", "m"),
    "bita": ("bit", "bity", "bitów", "bita", "m"),
    # Kilobajt (m)
    "kilobajt": ("kilobajt", "kilobajty", "kilobajtów", "kilobajta", "m"),
    "kilobajty": ("kilobajt", "kilobajty", "kilobajtów", "kilobajta", "m"),
    "kilobajtów": ("kilobajt", "kilobajty", "kilobajtów", "kilobajta", "m"),
    "kilobajta": ("kilobajt", "kilobajty", "kilobajtów", "kilobajta", "m"),
    # Terabajt (m)
    "terabajt": ("terabajt", "terabajty", "terabajtów", "terabajta", "m"),
    "terabajty": ("terabajt", "terabajty", "terabajtów", "terabajta", "m"),
    "terabajtów": ("terabajt", "terabajty", "terabajtów", "terabajta", "m"),
    "terabajta": ("terabajt", "terabajty", "terabajtów", "terabajta", "m"),
    # Szklanka (f)
    "szklanka": ("szklanka", "szklanki", "szklanek", "szklanki", "f"),
    "szklanki": ("szklanka", "szklanki", "szklanek", "szklanki", "f"),
    "szklanek": ("szklanka", "szklanki", "szklanek", "szklanki", "f"),
    # Łyżka (f)
    "łyżka": ("łyżka", "łyżki", "łyżek", "łyżki", "f"),
    "łyżki": ("łyżka", "łyżki", "łyżek", "łyżki", "f"),
    "łyżek": ("łyżka", "łyżki", "łyżek", "łyżki", "f"),
    # Łyżeczka (f)
    "łyżeczka": ("łyżeczka", "łyżeczki", "łyżeczek", "łyżeczki", "f"),
    "łyżeczki": ("łyżeczka", "łyżeczki", "łyżeczek", "łyżeczki", "f"),
    "łyżeczek": ("łyżeczka", "łyżeczki", "łyżeczek", "łyżeczki", "f"),
    # Gigabajt (m)
    "gigabajt": ("gigabajt", "gigabajty", "gigabajtów", "gigabajta", "m"),
    "gigabajty": ("gigabajt", "gigabajty", "gigabajtów", "gigabajta", "m"),
    "gigabajtów": ("gigabajt", "gigabajty", "gigabajtów", "gigabajta", "m"),
    "gigabajta": ("gigabajt", "gigabajty", "gigabajtów", "gigabajta", "m"),
    # Megabajt (m)
    "megabajt": ("megabajt", "megabajty", "megabajtów", "megabajta", "m"),
    "megabajty": ("megabajt", "megabajty", "megabajtów", "megabajta", "m"),
    "megabajtów": ("megabajt", "megabajty", "megabajtów", "megabajta", "m"),
    "megabajta": ("megabajt", "megabajty", "megabajtów", "megabajta", "m"),
    # Rok (m)
    "rok": ("rok", "lata", "lat", "roku", "m"),
    "lata": ("rok", "lata", "lat", "roku", "m"),
    "lat": ("rok", "lata", "lat", "roku", "m"),
    "roku": ("rok", "lata", "lat", "roku", "m"),
    # Mila (f)
    "mila": ("mila", "mile", "mil", "mili", "f"),
    "mile": ("mila", "mile", "mil", "mili", "f"),
    "mil": ("mila", "mile", "mil", "mili", "f"),
    "mili": ("mila", "mile", "mil", "mili", "f"),
    # Kilogram (m)
    "kilogram": ("kilogram", "kilogramy", "kilogramów", "kilograma", "m"),
    "kilogramy": ("kilogram", "kilogramy", "kilogramów", "kilograma", "m"),
    "kilogramów": ("kilogram", "kilogramy", "kilogramów", "kilograma", "m"),
    "kilograma": ("kilogram", "kilogramy", "kilogramów", "kilograma", "m"),
    # Metr (m)
    "metr": ("metr", "metry", "metrów", "metra", "m"),
    "metry": ("metr", "metry", "metrów", "metra", "m"),
    "metrów": ("metr", "metry", "metrów", "metra", "m"),
    "metra": ("metr", "metry", "metrów", "metra", "m"),
    # Centymetr (m)
    "centymetr": ("centymetr", "centymetry", "centymetrów", "centymetra", "m"),
    "centymetry": ("centymetr", "centymetry", "centymetrów", "centymetra", "m"),
    "centymetrów": ("centymetr", "centymetry", "centymetrów", "centymetra", "m"),
    "centymetra": ("centymetr", "centymetry", "centymetrów", "centymetra", "m"),
    # Milimetr (m)
    "milimetr": ("milimetr", "milimetry", "milimetrów", "milimetra", "m"),
    "milimetry": ("milimetr", "milimetry", "milimetrów", "milimetra", "m"),
    "milimetrów": ("milimetr", "milimetry", "milimetrów", "milimetra", "m"),
    "milimetra": ("milimetr", "milimetry", "milimetrów", "milimetra", "m"),
    # Sekunda (f)
    "sekunda": ("sekunda", "sekundy", "sekund", "sekundy", "f"),
    "sekundy": ("sekunda", "sekundy", "sekund", "sekundy", "f"),
    "sekund": ("sekunda", "sekundy", "sekund", "sekundy", "f"),
    # Funt (m)
    "funt": ("funt", "funty", "funtów", "funta", "m"),
    "funty": ("funt", "funty", "funtów", "funta", "m"),
    "funtów": ("funt", "funty", "funtów", "funta", "m"),
    "funta": ("funt", "funty", "funtów", "funta", "m"),
    # Frank (m)
    "frank": ("frank", "franki", "franków", "franka", "m"),
    "franki": ("frank", "franki", "franków", "franka", "m"),
    "franków": ("frank", "franki", "franków", "franka", "m"),
    "franka": ("frank", "franki", "franków", "franka", "m"),
    # Litr (m)
    "litr": ("litr", "litry", "litrów", "litra", "m"),
    "litry": ("litr", "litry", "litrów", "litra", "m"),
    "litrów": ("litr", "litry", "litrów", "litra", "m"),
    "litra": ("litr", "litry", "litrów", "litra", "m"),
    # Mililitr (m)
    "mililitr": ("mililitr", "mililitry", "mililitrów", "mililitra", "m"),
    "mililitry": ("mililitr", "mililitry", "mililitrów", "mililitra", "m"),
    "mililitrów": ("mililitr", "mililitry", "mililitrów", "mililitra", "m"),
    "mililitra": ("mililitr", "mililitry", "mililitrów", "mililitra", "m"),
    # Książka (f)
    "książka": ("książka", "książki", "książek", "książki", "f"),
    "książki": ("książka", "książki", "książek", "książki", "f"),
    "książek": ("książka", "książki", "książek", "książki", "f"),
    # Osoba (f)
    "osoba": ("osoba", "osoby", "osób", "osoby", "f"),
    "osoby": ("osoba", "osoby", "osób", "osoby", "f"),
    "osób": ("osoba", "osoby", "osób", "osoby", "f"),
    # Gram (m)
    "gram": ("gram", "gramy", "gramów", "grama", "m"),
    "gramy": ("gram", "gramy", "gramów", "grama", "m"),
    "gramów": ("gram", "gramy", "gramów", "grama", "m"),
    "grama": ("gram", "gramy", "gramów", "grama", "m"),
    # Miligram (m)
    "miligram": ("miligram", "miligramy", "miligramów", "miligrama", "m"),
    "miligramy": ("miligram", "miligramy", "miligramów", "miligrama", "m"),
    "miligramów": ("miligram", "miligramy", "miligramów", "miligrama", "m"),
    "miligrama": ("miligram", "miligramy", "miligramów", "miligrama", "m"),
    # Tona (f)
    "tona": ("tona", "tony", "ton", "tony", "f"),
    "tony": ("tona", "tony", "ton", "tony", "f"),
    "ton": ("tona", "tony", "ton", "tony", "f"),
    # Uncja (f)
    "uncja": ("uncja", "uncje", "uncji", "uncji", "f"),
    "uncje": ("uncja", "uncje", "uncji", "uncji", "f"),
    "uncji": ("uncja", "uncje", "uncji", "uncji", "f"),
    # Cal (m)
    "cal": ("cal", "cale", "cali", "cala", "m"),
    "cale": ("cal", "cale", "cali", "cala", "m"),
    "cali": ("cal", "cale", "cali", "cala", "m"),
    "cala": ("cal", "cale", "cali", "cala", "m"),
    # Stopa (f)
    "stopa": ("stopa", "stopy", "stóp", "stopy", "f"),
    "stopy": ("stopa", "stopy", "stóp", "stopy", "f"),
    "stóp": ("stopa", "stopy", "stóp", "stopy", "f"),
    # Jard (m)
    "jard": ("jard", "jardy", "jardów", "jarda", "m"),
    "jardy": ("jard", "jardy", "jardów", "jarda", "m"),
    "jardów": ("jard", "jardy", "jardów", "jarda", "m"),
    "jarda": ("jard", "jardy", "jardów", "jarda", "m"),
    # Kelwin (m)
    "kelwin": ("kelwin", "kelwiny", "kelwinów", "kelwina", "m"),
    "kelwiny": ("kelwin", "kelwiny", "kelwinów", "kelwina", "m"),
    "kelwinów": ("kelwin", "kelwiny", "kelwinów", "kelwina", "m"),
    "kelwina": ("kelwin", "kelwiny", "kelwinów", "kelwina", "m"),
    # Galon (m)
    "galon": ("galon", "galony", "galonów", "galona", "m"),
    "galony": ("galon", "galony", "galonów", "galona", "m"),
    "galonów": ("galon", "galony", "galonów", "galona", "m"),
    "galona": ("galon", "galony", "galonów", "galona", "m"),
    # Pinta (f)
    "pinta": ("pinta", "pinty", "pint", "pinty", "f"),
    "pinty": ("pinta", "pinty", "pint", "pinty", "f"),
    "pint": ("pinta", "pinty", "pint", "pinty", "f"),
    # Węzeł (m)
    "węzeł": ("węzeł", "węzły", "węzłów", "węzła", "m"),
    "węzły": ("węzeł", "węzły", "węzłów", "węzła", "m"),
    "węzłów": ("węzeł", "węzły", "węzłów", "węzła", "m"),
    "węzła": ("węzeł", "węzły", "węzłów", "węzła", "m"),
    # Tydzień (m)
    "tydzień": ("tydzień", "tygodnie", "tygodni", "tygodnia", "m"),
    "tygodnie": ("tydzień", "tygodnie", "tygodni", "tygodnia", "m"),
    "tygodni": ("tydzień", "tygodnie", "tygodni", "tygodnia", "m"),
    "tygodnia": ("tydzień", "tygodnie", "tygodni", "tygodnia", "m"),
}

# Unit symbols map to noun dictionary keys and additional suffix
UNIT_SYMBOL_TO_NOUN: dict[str, tuple[str, str]] = {
    "tys.": ("tysiąc", ""),
    "tys": ("tysiąc", ""),
    "mln": ("milion", ""),
    "zł": ("złoty", ""),
    "pln": ("złoty", ""),
    "eur": ("euro", ""),
    "usd": ("dolar", ""),
    "gbp": ("funt", ""),
    "chf": ("frank", ""),
    "km/h": ("kilometr", "na godzinę"),
    "m/s": ("metr", "na sekundę"),
    "km": ("kilometr", ""),
    "kg": ("kilogram", ""),
    "cm": ("centymetr", ""),
    "mm": ("milimetr", ""),
    "m": ("metr", ""),
    "gb": ("gigabajt", ""),
    "gigabajt": ("gigabajt", ""),
    "gigabajty": ("gigabajt", ""),
    "gigabajtów": ("gigabajt", ""),
    "mib": ("megabajt", ""),
    "mb": ("megabajt", ""),
    "megabajt": ("megabajt", ""),
    "megabajty": ("megabajt", ""),
    "megabajtów": ("megabajt", ""),
    "h": ("godzina", ""),
    "godz": ("godzina", ""),
    "min": ("minuta", ""),
    "s": ("sekunda", ""),
    "sek": ("sekunda", ""),
    "mi": ("mila", ""),
    "mile": ("mila", ""),
    "mil": ("mila", ""),
    "d": ("dzień", ""),
    "dni": ("dzień", ""),
    "l": ("litr", ""),
    "ml": ("mililitr", ""),
}


def agree_noun(val: int | float, noun_key: str, case: str = "nom") -> str:
    """Select correct grammatical form of the Polish noun according to quantity and case."""
    clean_key = noun_key.lower().strip()
    decl = NOUN_DECLENSIONS.get(clean_key)
    if not decl:
        return noun_key

    sg_nom, pl_nom, pl_gen, sg_gen, _ = decl

    # Rule: For memory units (gigabajt, megabajt), in genitive phrases (e.g. "z 15,5 GB"),
    # plural genitive (gigabajtów) is the standard natural form.
    if clean_key in ("gigabajt", "megabajt") and case == "gen":
        return pl_gen

    # Rule: After decimals, Polish strictly takes genitive singular
    if isinstance(val, float) and not val.is_integer():
        return sg_gen

    n = abs(int(val))
    if case == "gen":
        if n == 1:
            return sg_gen
        return pl_gen

    # case == "nom" (default)
    if n == 1:
        return sg_nom
    mod10 = n % 10
    mod100 = n % 100
    if mod10 in (2, 3, 4) and mod100 not in (12, 13, 14):
        return pl_nom
    return pl_gen


def inflect_number_noun(val: int | float, noun_key: str, case: str = "nom") -> str:
    """Format quantity and noun together with full Polish grammatical agreement."""
    clean_key = noun_key.lower().strip()
    suffix = ""
    if clean_key in UNIT_SYMBOL_TO_NOUN:
        clean_key, suffix = UNIT_SYMBOL_TO_NOUN[clean_key]

    decl = NOUN_DECLENSIONS.get(clean_key)
    gender = decl[4] if decl else "m"
    noun_form = agree_noun(val, clean_key, case=case)

    num_str = inflect_number(val, case=case, gender=gender)
    if suffix:
        return f"{num_str} {noun_form} {suffix}".strip()
    return f"{num_str} {noun_form}".strip()


def format_number_pl(n: int | float, max_decimals: int = 2) -> str:
    """Format number into speech-friendly Polish string."""
    if isinstance(n, float):
        if n.is_integer():
            n = int(n)
        else:
            n = round(n, max_decimals)
    text = str(n)
    if "." in text:
        parts = text.split(".")
        return f"{parts[0]} przecinek {parts[1]}"
    return text


def polish_declension(
    n: int | float, one: str, few: str, many: str, sg_gen: str | None = None
) -> str:
    """Select correct grammatical form for Polish numbers (backward-compatible)."""
    if isinstance(n, float) and not n.is_integer():
        return sg_gen if sg_gen else few
    val = abs(int(n))
    if val == 1:
        return one
    mod10 = val % 10
    mod100 = val % 100
    if mod10 in (2, 3, 4) and mod100 not in (12, 13, 14):
        return few
    return many


# --- Polish device-label noun cases (F7) ---

# Trailing prepositional phrases ("klima u Ani", "tv w salonie") are copied verbatim; only the
# leading noun group (adjective(s) + head noun) is declined.
_PL_CASE_PREPS: frozenset[str] = frozenset(
    {"w", "we", "u", "na", "do", "z", "ze", "od", "o", "dla", "przy", "po", "za", "obok", "nad", "pod"}
)
# Closed list of common label adjectives that take -ą/-ej (folded). A word outside the list is
# never declined — the whole form comes back empty ("", "") and callers speak a neutral shape.
_PL_FEM_ADJECTIVES: frozenset[str] = frozenset(
    {
        "ciepla", "zimna", "goraca", "cieplna", "glowna", "boczna", "nowa", "stara", "mala",
        "duza", "sucha", "mokra", "ciemna", "jasna", "cicha", "glosna", "wilgotna",
        "elektryczna", "gazowa", "nascienna", "sufitowa", "przenosna", "stacjonarna",
    }
)
# Feminine -a genitive takes -i after k/g/l/j/c (drukarka->drukarki, konsola->konsoli), -y after
# hard stems (klima->klimy, woda->wody); -nia/-ja always takes -i (sypialnia->sypialni).
_PL_FEM_GEN_I_STEMS: frozenset[str] = frozenset({"k", "g", "l", "j", "c"})


def _pl_fold_word(word: str) -> str:
    text = word.lower().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def polish_noun_cases(nom: str) -> tuple[str, str]:
    """Best-effort accusative and genitive of a Polish device label ("klima u Ani" ->
    ``("klimę u Ani", "klimy u Ani")``).

    Generic vocabulary only: the trailing prepositional phrase is copied verbatim, the head noun
    is declined by its ending (feminine -a, -ść; masculine and indeclinable keep the nominative
    accusative), and a small closed adjective list covers "ciepła woda"-style labels. Any form
    the helper cannot derive confidently comes back as ``""`` — callers must speak a neutral,
    case-free sentence shape then, never a guessed form.
    """
    words = nom.split()
    if not words:
        return ("", "")
    tail_start = next((i for i, w in enumerate(words) if _pl_fold_word(w) in _PL_CASE_PREPS), len(words))
    head, tail = words[:tail_start], words[tail_start:]
    if not head:
        return ("", "")
    noun = head[-1]
    adjectives = head[:-1]
    low = _pl_fold_word(noun)
    if not low:
        return ("", "")

    noun_acc = noun_gen = ""
    if not low[-1].isalpha():
        noun_acc = noun_gen = noun  # digits and abbreviations ("ps5") are indeclinable
    elif noun.lower().endswith("ć"):
        noun_acc = noun  # wilgotność -> (acc) wilgotność
        noun_gen = noun[:-1] + "ci"  # wilgotność -> (gen) wilgotności
    elif low.endswith("a"):
        noun_acc = noun[:-1] + "ę"  # klima -> klimę
        if low.endswith("nia"):
            noun_gen = noun[:-2] + "i"  # sypialnia -> sypialni, Ania -> Ani
        elif low[-2] in _PL_FEM_GEN_I_STEMS:
            noun_gen = noun[:-1] + "i"  # drukarka -> drukarki, konsola -> konsoli
        else:
            noun_gen = noun[:-1] + "y"  # klima -> klimy, woda -> wody
    elif low.endswith("o"):
        noun_acc = noun  # radio -> (acc) radio
        noun_gen = ""  # radio -> radia: unpredictable, unknown
    elif low[-1] not in "aeiouyąęó":
        noun_acc = noun  # inanimate masculine: piec -> (acc) piec
        noun_gen = ""  # piec -> pieca vs dysk -> dysku: unpredictable, unknown
    # every other ending (i, y, u, e, ą, ę, ó): no confident form at all

    acc_parts: list[str] = []
    gen_parts: list[str] = []
    for w in adjectives:
        fw = _pl_fold_word(w)
        if fw in _PL_FEM_ADJECTIVES:
            acc_parts.append(w[:-1] + "ą")  # ciepła -> ciepłą
            gen_parts.append(w[:-1] + "ej")  # ciepła -> ciepłej
        elif len(fw) >= 3 and fw[-1] in "yi" and fw[-2].isalpha():
            acc_parts.append(w)  # stary -> (acc) stary
            gen_parts.append(w[:-1] + "ego")  # stary -> starego
        else:
            return ("", "")  # an unrecognised word: decline nothing, guess nothing
    if noun_acc:
        acc_parts.append(noun_acc)
    else:
        acc_parts = []
    if noun_gen:
        gen_parts.append(noun_gen)
    else:
        gen_parts = []
    acc = " ".join([*acc_parts, *tail]) if acc_parts else ""
    gen = " ".join([*gen_parts, *tail]) if gen_parts else ""
    return (acc, gen)


def _pl_year_ord(year: int, case: str = "gen") -> str:
    """Format year into Polish ordinal: 'gen' (dwudziestego) or 'loc' (dwudziestym)."""
    ord_days = POLISH_DAYS_ORD_LOC if case == "loc" else POLISH_DAYS_ORD_GEN
    tens_words = TENS_ORD_LOC if case == "loc" else TENS_ORD_GEN

    if year == 2000:
        return "dwutysięcznym" if case == "loc" else "dwutysięcznego"
    if 2001 <= year <= 2099:
        last2 = year % 100
        if last2 == 0:
            return "dwa tysiące setnym" if case == "loc" else "dwa tysiące setnego"
        if last2 in ord_days:
            return f"dwa tysiące {ord_days[last2]}"
        tens = (last2 // 10) * 10
        ones = last2 % 10
        if ones == 0:
            return f"dwa tysiące {tens_words.get(tens, '')}"
        return f"dwa tysiące {tens_words.get(tens, '')} {ord_days.get(ones, '')}".strip()
    if 1000 <= year <= 1999:
        rem = year % 1000
        hundreds = (rem // 100) * 100
        last2 = rem % 100
        h_words = {
            100: "sto",
            200: "dwieście",
            300: "trzysta",
            400: "czterysta",
            500: "pięćset",
            600: "sześćset",
            700: "siedemset",
            800: "osiemset",
            900: "dziewięćset",
        }
        h_part = f" {h_words[hundreds]}" if hundreds in h_words else ""
        if last2 == 0:
            if hundreds == 0:
                return "tysięcznym" if case == "loc" else "tysięcznego"
            hundreds_ord = {
                100: ("stym", "stego"),
                200: ("dwusetnym", "dwusetnego"),
                300: ("trzechsetnym", "trzechsetnego"),
                400: ("czterechsetnym", "czterechsetnego"),
                500: ("pięćsetnym", "pięćsetnego"),
                600: ("sześćsetnym", "sześćsetnego"),
                700: ("siedemsetnym", "siedemsetnego"),
                800: ("osiemsetnym", "osiemsetnego"),
                900: ("dziewięćsetnym", "dziewięćsetnego"),
            }
            ord_suff = hundreds_ord.get(hundreds, ("stym", "stego"))
            suff = ord_suff[0] if case == "loc" else ord_suff[1]
            return f"tysiąc {suff}".strip()
        if last2 in ord_days:
            return f"tysiąc{h_part} {ord_days[last2]}"
        tens = (last2 // 10) * 10
        ones = last2 % 10
        if ones == 0:
            return f"tysiąc{h_part} {tens_words.get(tens, '')}"
        return f"tysiąc{h_part} {tens_words.get(tens, '')} {ord_days.get(ones, '')}".strip()
    try:
        return num2words.num2words(year, lang="pl")
    except Exception:
        return str(year)


def _pl_year_ord_gen(year: int) -> str:
    """Format year into Polish ordinal genitive: 2026 -> dwa tysiące dwudziestego szóstego."""
    return _pl_year_ord(year, case="gen")


def _pl_year_ord_loc(year: int) -> str:
    """Format year into Polish ordinal locative: 2026 -> dwa tysiące dwudziestym szóstym."""
    return _pl_year_ord(year, case="loc")


# Phone numbers. The 3-3-3 shape alone is not a phone: "1 412 000 000 ludzi" and "12 345 678 901"
# are numbers, and the old fixed noun lookahead let every noun outside its list through (F4). The
# shape is decided by context in _sub_phone: a phone cue before it, a +48 prefix, or the bare
# 3-3-3 form with a non-zero leading group and no following word.
RE_PHONE_333 = re.compile(
    r"(?:(\+48|\(\+48\)|0048)[\s-]?)?(\b\d{3})[\s-](\d{3})[\s-](\d{3})\b",
    re.I,
)

RE_PHONE_322 = re.compile(
    r"(?:(\+48|\(\+48\)|0048)[\s-]?)?(?:\(?(\b\d{2})\)?[\s-]?)?(\b\d{3})-(\d{2})-(\d{2})\b",
    re.I,
)

# Words that introduce a phone number ("mój numer to ...", "telefon: ...", "zadzwoń pod ...").
RE_PHONE_CUE = re.compile(
    r"(?:\btel\.?|\btelefon\w*|\bkom[oó]rk\w*|\bnumer\w*|\bzadzwo[nń]\w*|\binfolini\w*|\bfax\w*)"
    r"\s*(?:to\s+|jest\s+|pod\s+|:)?\s*$",
    re.I,
)
# A digit + space right before the match means it is the tail of a longer spaced number.
RE_DIGIT_BEFORE_PHONE = re.compile(r"\d\s$")
# Any word after the match means the groups are a magnitude ("... ludzi"), not a phone.
RE_WORD_AFTER_PHONE = re.compile(r"\s+[a-ząćęłńóśźż]", re.I)


def _read_phone_group(g: str) -> str:
    g = g.strip()
    if not g:
        return ""
    if g == "0":
        return "zero"
    if all(c == "0" for c in g):
        return " ".join(["zero"] * len(g))
    if len(g) > 1 and g.startswith("0"):
        zeros_count = len(g) - len(g.lstrip("0"))
        zeros_part = " ".join(["zero"] * zeros_count)
        rem_word = pl_cardinal(int(g.lstrip("0")), case="nom")
        return f"{zeros_part} {rem_word}"
    return pl_cardinal(int(g), case="nom")


def _looks_like_phone_333(text: str, m: re.Match) -> bool:
    """Whether a 3-3-3 match is a phone number rather than the tail of a spaced magnitude (F4)."""
    if m.group(1):
        return True
    before = text[: m.start()]
    if RE_PHONE_CUE.search(before):
        return True
    if RE_DIGIT_BEFORE_PHONE.search(before):
        return False
    # A thousands group of zeros ("1 412 000 000") is a magnitude, never a phone.
    if any(group == "000" for group in (m.group(2), m.group(3), m.group(4))):
        return False
    # A bare 3-3-3 number reads as digit groups only when no word follows it ("... ludzi" is a noun).
    return RE_WORD_AFTER_PHONE.match(text, m.end()) is None


def _sub_phone(text: str) -> str:
    def _sub_333(m: re.Match) -> str:
        if not _looks_like_phone_333(text, m):
            return m.group(0)
        prefix = m.group(1)
        g1, g2, g3 = m.group(2), m.group(3), m.group(4)
        parts = []
        if prefix:
            parts.append("plus czterdzieści osiem")
        parts.extend([_read_phone_group(g1), _read_phone_group(g2), _read_phone_group(g3)])
        return " ".join(parts)

    def _sub_322(m: re.Match) -> str:
        prefix = m.group(1)
        area = m.group(2)
        g1, g2, g3 = m.group(3), m.group(4), m.group(5)
        parts = []
        if prefix:
            parts.append("plus czterdzieści osiem")
        if area:
            parts.append(_read_phone_group(area))
        parts.extend([_read_phone_group(g1), _read_phone_group(g2), _read_phone_group(g3)])
        return " ".join(parts)

    res = RE_PHONE_333.sub(_sub_333, text)
    res = RE_PHONE_322.sub(_sub_322, res)
    return res


# Precompiled regex patterns for performance (< 1 ms requirement)
RE_CLEAN_WS = re.compile(r"\s+")
RE_LARGE_NUM_SPACES = re.compile(r"\b(\d{1,3}(?:\s\d{3})+)\b")
RE_CENTURY_PL = re.compile(
    r"(\b(?:w|po)\s+)?\b(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII)\s+(wiek[a-z]*)\b",
    re.I,
)
_MONTHS_PATTERN = "|".join(PL_MONTHS)
RE_DATE_PL = re.compile(
    rf"\b([1-9]|[12]\d|3[01])\s+({_MONTHS_PATTERN})(?:\s+(\d{{4}}))?(?:\s*(?:roku|rok|r\.|r\b))?",
    re.I,
)
RE_YEAR_STANDALONE_PL = re.compile(
    r"(?:(\bod)\s+(\b\d{4})\s+do\s+)?(\b(?:w|we|po|o|przy|od|do|z|ze|około|ok\.|przed)\s+)?(\b\d{4})\s*(?:roku|rok|r\.|r\b)",
    re.I,
)
# Times with the preposition anchored to a word start (F3): the old unanchored "o\s+" only
# matched inside "po"/"około" by accident and przed/między were left nominative. o/od/do/po take
# the locative, "przed" the instrumental ("przed osiemnastą"), "około"/"ok." the genitive — the
# genitive of the feminine hour ordinals is identical to the locative, so POLISH_HOURS_LOC serves
# both. "za", "nad" and "pod" are left untouched on purpose.
RE_TIME_PL = re.compile(
    r"(?:(\bo|od|do|po|przed|około|ok\.)\s+(?:godz\.?\s*|godzinie\s*|godzina\s*)?|(\bgodz\.\s*|\bgodzina\s+|\bgodzinie\s+))?([01]?\d|2[0-3]):([0-5]\d)\b",
    re.I,
)

# "między 17:00 a 18:00": both bounds instrumental (F3); must run before RE_TIME_PL, whose bare
# fallback would read the second bound in the nominative.
RE_TIME_BETWEEN_PL = re.compile(
    r"\b(między)\s+([01]?\d|2[0-3]):([0-5]\d)\s+a\s+([01]?\d|2[0-3]):([0-5]\d)\b",
    re.I,
)

# Range "od X do Y [unit]" — the unit token (if any) is group 4 as a whole, so a range with
# "stopni"/"°C" is distinguishable from a bare unitless "od 5 do 10" (F1).
RE_RANGE_OD_DO = re.compile(
    r"\b(od)\s+(-?\d+(?:[.,]\d+)?)\s+do\s+(-?\d+(?:[.,]\d+)?)(?:\s*(°C|°|stopni(?:e|a|ów)?\s*Celsjusza|stopni(?:e|a|ów)?|[a-ząćęłńóśźż/]+))?",
    re.I,
)

# Fraction "X z Y [unit]"
RE_FRACTION_Z = re.compile(
    r"\b(-?\d+(?:[.,]\d+)?)\s+(z|ze)\s+(-?\d+(?:[.,]\d+)?)(?:\s*([a-ząćęłńóśźż/]+))?\b",
    re.I,
)

# Preposition before number: od, do, z, ze, około, ok., bez, u, dla, spośród
RE_PREP_NUM = re.compile(
    r"(?:(\bto\s+))?\b(od|do|z|ze|około|ok\.|bez|u|dla|spośród)\s+(-?\d+(?:[.,]\d+)?)(?:\s*(°C|°|stopni(?:e|a|ów)?\s*Celsjusza|stopni(?:e|a|ów)?|%|[a-ząćęłńóśźż/]+(?:\s+[a-ząćęłńóśźż]+)?))?\b",
    re.I,
)

RE_TEMP_PL = re.compile(r"(-)?(\d+(?:[.,]\d+)?)\s*(?:°\s*C|°C|stopni(?:e|a)?\s*Celsjusza)\b", re.I)
RE_DEG_PL = re.compile(r"(-)?(\d+(?:[.,]\d+)?)\s*°", re.I)
RE_PCT_PL = re.compile(r"(-)?(\d+(?:[.,]\d+)?)\s*%", re.I)
RE_CURRENCY_PLN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:zł|złotych|złote|złoty|PLN)\b", re.I)
RE_CURRENCY_SYM_PRE = re.compile(r"([€\$£])\s*(\d+(?:[.,]\d+)?)")
RE_CURRENCY_SYM_POST = re.compile(r"(\d+(?:[.,]\d+)?)\s*(EUR|USD|GBP|CHF|€|\$|£|euro|dolary?|dolarów|funty?|funtów|franki?|franków)\b", re.I)
RE_DECIMAL_PL = re.compile(r"\b(\d+)[.,](\d+)\b")
# Sign first so "-10,8" is caught whole before the decimal rule reads it without the minus (F2)
RE_NEGATIVE_NUM = re.compile(r"(^|\s)-(\d+(?:[.,]\d+)?)\b")
RE_INTEGER_PL = re.compile(r"\b(\d+)\b")

STREET_ABBR_FORMS: dict[str, dict[str, str]] = {
    "ul.": {"nom": "ulica", "loc": "ulicy", "gen": "ulicy"},
    "al.": {"nom": "aleja", "loc": "alei", "gen": "alei"},
    "pl.": {"nom": "plac", "loc": "placu", "gen": "placu"},
    "os.": {"nom": "osiedle", "loc": "osiedlu", "gen": "osiedla"},
}

RE_STREET_PREP = re.compile(
    r"\b(z\s+okolic|spod|przy|od|do|na|ze|z|we|w|koło|obok|blisko)\s+(ul\.|al\.|pl\.|os\.)",
    re.I,
)
RE_STREET_STANDALONE = re.compile(r"\b(ul\.|al\.|pl\.|os\.)", re.I)

TITLE_ABBR_FORMS: dict[str, dict[str, str]] = {
    "dr": {"nom": "doktor", "gen": "doktora", "inst": "doktorem", "loc": "doktorze"},
    "dr.": {"nom": "doktor", "gen": "doktora", "inst": "doktorem", "loc": "doktorze"},
    "prof.": {"nom": "profesor", "gen": "profesora", "inst": "profesorem", "loc": "profesorze"},
}

RE_TITLE_PREP = re.compile(r"\b(u|do|od|dla|ze|z|o)\s+(dr\b\.?|prof\.)", re.I)
RE_TITLE_STANDALONE = re.compile(r"\b(dr\b\.?|prof\.)", re.I)

RE_GODZ_PREP = re.compile(r"\b(o|od|do|po|przed)\s+godz\.", re.I)

PL_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnp\.", re.I), "na przykład"),
    (re.compile(r"\btzn\.", re.I), "to znaczy"),
    (re.compile(r"\bm\.\s*in\.", re.I), "między innymi"),
    (re.compile(r"\bok\.", re.I), "około"),
    (re.compile(r"\btzw\.", re.I), "tak zwany"),
    (re.compile(r"\bitd\.", re.I), "i tak dalej"),
    (re.compile(r"\bitp\.", re.I), "i tym podobne"),
    (re.compile(r"\bśw\.", re.I), "świętego"),
    (re.compile(r"\bgodz\.", re.I), "godzina"),
]

UNITS_MAP_PL: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*tys\.\s*(?:zł|złotych|złote|złoty|PLN)\b", re.I), "tysiąc", "złotych"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*mln\s*(?:zł|złotych|złote|złoty|PLN)\b", re.I), "milion", "złotych"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*mld\s*(?:zł|złotych|złote|złoty|PLN)\b", re.I), "miliard", "złotych"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*tys\.(?!\w)", re.I), "tysiąc", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*mln\b", re.I), "milion", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*mld\b", re.I), "miliard", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*gr\b", re.I), "grosz", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*km/h\b", re.I), "kilometr", "na godzinę"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*m/s\b", re.I), "metr", "na sekundę"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*km\b", re.I), "kilometr", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*kg\b", re.I), "kilogram", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*cm\b", re.I), "centymetr", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\b", re.I), "milimetr", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*m\b", re.I), "metr", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:GB|gigabajtów|gigabajty|gigabajta)\b", re.I), "gigabajt", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:MiB|MB|megabajtów|megabajty|megabajta)\b", re.I), "megabajt", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:godziny|godzin|godzina|godz)\b", re.I), "godzina", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:minuty|minut|minuta|min)\b", re.I), "minuta", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:mile|mil|mila)\b", re.I), "mila", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:dni|dzień|dnia)\b", re.I), "dzień", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:sekundy|sekund|sekunda|sek|s)\b", re.I), "sekunda", ""),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:litry|litrów|litra|litr|l)\b", re.I), "litr", ""),
]

EN_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\bapprox\.", re.I), "approximately"),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bvs\.", re.I), "versus"),
    (re.compile(r"\bDr\.", re.I), "Doctor"),
    (re.compile(r"\bMr\.", re.I), "Mister"),
    (re.compile(r"\bMrs\.", re.I), "Missus"),
    (re.compile(r"\bMs\.", re.I), "Miss"),
]

UNITS_MAP_EN: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*km/h\b", re.I), "kilometers per hour"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mph\b", re.I), "miles per hour"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*kg\b", re.I), "kilograms"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*km\b", re.I), "kilometers"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*cm\b", re.I), "centimeters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m\b", re.I), "meters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*GB\b", re.I), "gigabytes"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:MiB|MB)\b", re.I), "megabytes"),
]

# Word-level declension map: nominative words -> genitive words after prepositions
CARDINALS_WORD_NOM_TO_GEN: dict[str, str] = {
    "zero": "zera",
    "jeden": "jednego",
    "jedno": "jednego",
    "jedna": "jednej",
    "dwa": "dwóch",
    "dwie": "dwóch",
    "trzy": "trzech",
    "cztery": "czterech",
    "pięć": "pięciu",
    "sześć": "sześciu",
    "siedem": "siedmiu",
    "osiem": "ośmiu",
    "dziewięć": "dziewięciu",
    "dziesięć": "dziesięciu",
    "jedenaście": "jedenastu",
    "dwanaście": "dwunastu",
    "trzynaście": "trzynastu",
    "czternaście": "czternastu",
    "piętnaście": "piętnastu",
    "szesnaście": "szesnastu",
    "siedemnaście": "siedemnastu",
    "osiemnaście": "osiemnastu",
    "dziewiętnaście": "dziewiętnastu",
    "dwadzieścia": "dwudziestu",
    "trzydzieści": "trzydziestu",
    "czterdzieści": "czterdziestu",
    "pięćdziesiąt": "pięćdziesięciu",
    "sześćdziesiąt": "sześćdziesięciu",
    "siedemdziesiąt": "siedemdziesięciu",
    "osiemdziesiąt": "osiemdziesięciu",
    "dziewięćdziesiąt": "dziewięćdziesięciu",
    "sto": "stu",
    "dwieście": "dwustu",
    "trzysta": "trzystu",
    "czterysta": "czterystu",
    "pięćset": "pięciuset",
    "sześćset": "sześciuset",
    "siedemset": "siedmiuset",
    "osiemset": "ośmiuset",
    "dziewięćset": "dziewięciuset",
    "tysiąc": "tysiąca",
    "tysiące": "tysięcy",
}


def _sub_word_number_genitive(text: str) -> str:
    """Inflect number words following genitive prepositions in Polish."""
    words_pattern = "|".join(CARDINALS_WORD_NOM_TO_GEN.keys())

    def _repl_od_do(m: re.Match) -> str:
        orig_od = m.group(1)
        w1 = m.group(2).lower()
        w2 = m.group(3).lower()
        g1 = CARDINALS_WORD_NOM_TO_GEN.get(w1, w1)
        g2 = CARDINALS_WORD_NOM_TO_GEN.get(w2, w2)
        return f"{orig_od} {g1} do {g2}"

    res = re.sub(
        rf"\b(od)\s+({words_pattern})\s+do\s+({words_pattern})\b",
        _repl_od_do,
        text,
        flags=re.I,
    )

    def _repl_prep_word(m: re.Match) -> str:
        lead_to = m.group(1) or ""
        prep = m.group(2)
        w = m.group(3).lower()
        if lead_to and prep.lower() in ("około", "ok."):
            return m.group(0)
        # Protect year phrases like "od dwa tysiące dwudziestego roku" or "od tysiąc ... roku"
        rest = res[m.end():]
        if re.match(r"^\s+(?:tysiące|tysiąc|\w+set)\b.*?\broku\b", rest):
            return m.group(0)
        g = CARDINALS_WORD_NOM_TO_GEN.get(w, w)
        return f"{lead_to}{prep} {g}"

    res = re.sub(
        rf"(?:(\bto\s+))?\b(od|do|z|ze|około|bez|dla|u|spośród)\s+({words_pattern})\b",
        _repl_prep_word,
        res,
        flags=re.I,
    )

    # Decimals after genitive: "przecinek pięć" -> "przecinek pięciu"
    def _repl_dec_gen(m: re.Match) -> str:
        w1 = m.group(1)
        w2 = m.group(2).lower()
        g2 = CARDINALS_WORD_NOM_TO_GEN.get(w2, w2)
        return f"{w1} przecinek {g2}"

    res = re.sub(
        rf"\b(z\s+\w+stu|z\s+\w+tu|z\s+\w+ciu|z\s+dwóch|z\s+trzech|z\s+czterech)\s+przecinek\s+({words_pattern})\b",
        _repl_dec_gen,
        res,
        flags=re.I,
    )



    # Gender fixes: "jeden euro" -> "jedno euro", "dwa godziny" -> "dwie godziny"
    def _repl_jeden_euro(m: re.Match) -> str:
        prev = m.group(1) or ""
        if re.search(r"(?:dzieścia|dzieści|dziesiąt|set|sta|sto|tysiąc|tysiące|tysięcy)\s+$", prev, re.I):
            return m.group(0)
        return f"{prev}jedno euro"

    res = re.sub(r"(\b\w+\s+)?\bjeden\s+euro\b", _repl_jeden_euro, res, flags=re.I)
    res = re.sub(r"\bdwa\s+godziny\b", "dwie godziny", res, flags=re.I)
    res = re.sub(r"\bdwa\s+minuty\b", "dwie minuty", res, flags=re.I)
    res = re.sub(r"\bdwa\s+mile\b", "dwie mile", res, flags=re.I)
    res = re.sub(r"\bdwa\s+sekundy\b", "dwie sekundy", res, flags=re.I)

    return res


def _normalize_pl(text: str) -> str:
    s = text.replace("\xa0", " ")

    # 0. Phone numbers
    s = _sub_phone(s)

    # 1. Strip spaces in large formatted numbers early: e.g. "1 000 000" -> "1000000"
    s = RE_LARGE_NUM_SPACES.sub(lambda m: str(int(m.group(1).replace(" ", ""))), s)

    # 2. Roman numerals before "wiek/wieku"
    def _sub_century(m: re.Match) -> str:
        prep = m.group(1) or ""
        roman = m.group(2).upper()
        suffix = m.group(3).lower()
        if prep:
            word = POLISH_CENTURIES_LOC.get(roman, roman)
            return f"{prep}{word} {suffix}"
        if suffix.startswith("wieku"):
            word = POLISH_CENTURIES_GEN.get(roman, roman)
            return f"{word} {suffix}"
        if suffix.startswith("wiekiem"):
            word = POLISH_CENTURIES_LOC.get(roman, roman)
            return f"{word} {suffix}"
        word = POLISH_CENTURIES_NOM.get(roman, roman)
        return f"{word} {suffix}"

    s = RE_CENTURY_PL.sub(_sub_century, s)

    # 3. Dates: "9 października 2026 r." / "9 października 2026 roku" / "9 października"
    def _sub_date(m: re.Match) -> str:
        day_num = int(m.group(1))
        month_name = m.group(2).lower()
        year_str = m.group(3)

        day_word = POLISH_DAYS_ORD_GEN.get(day_num, str(day_num))
        if year_str:
            year_num = int(year_str)
            year_word = _pl_year_ord_gen(year_num)
            return f"{day_word} {month_name} {year_word} roku"
        return f"{day_word} {month_name}"

    s = RE_DATE_PL.sub(_sub_date, s)

    # Standalone year with r. or roku: "2026 r." / "2026 roku" / "w 1990 roku" / "w 1990 r."
    # plus the year range "od 2026 do 2027 roku": both ends ordinal genitive, never the mixed
    # cardinal/ordinal reading the range and year rules produced apart (F1).
    def _sub_year(m: re.Match) -> str:
        range_od = m.group(1)
        range_year = int(m.group(2)) if m.group(2) else None
        prefix = m.group(3) or ""
        year_num = int(m.group(4))
        prep_clean = prefix.strip().lower()
        if range_od:
            return f"{range_od} {_pl_year_ord_gen(range_year)} do {_pl_year_ord_gen(year_num)} roku"
        if prep_clean in ("w", "we", "po", "o", "przy"):
            case = "loc"
        else:
            case = "gen"
        year_word = _pl_year_ord(year_num, case=case)
        return f"{prefix}{year_word} roku"

    s = RE_YEAR_STANDALONE_PL.sub(_sub_year, s)

    # 4. Times: "o 17:59" / "od 12:00" / "po 22:00" / "przed 18:30" / "około 17:15" / "17:59"
    def _sub_time(m: re.Match) -> str:
        prep = m.group(1)
        godz_prefix = m.group(2)
        hour = int(m.group(3))
        minute = int(m.group(4))

        prep_clean = prep.strip().lower() if prep else ""
        # The surface text between the preposition and the hour: "o godzinie ", "o ", "przed ".
        lead = m.group(0)[: m.start(3) - m.start()]
        if prep_clean in ("o", "od", "do", "po"):
            h_word = POLISH_HOURS_LOC.get(hour, str(hour))
            # Keep a spelled-out "godzinie": "o godzinie 06:25" stays "o godzinie szóstej...",
            # as on main@7741180; the "godz." abbreviation still expands to a bare "o ".
            prefix = lead
            if "godz." in prefix.lower():
                prefix = prep
        elif prep_clean == "przed":
            h_word = POLISH_HOURS_INST.get(hour, str(hour))
            prefix = prep
        elif prep_clean in ("około", "ok."):
            # genitive == locative for the feminine hour ordinals
            h_word = POLISH_HOURS_LOC.get(hour, str(hour))
            prefix = "Około" if prep[0].isupper() else "około"
        elif godz_prefix and "godzinie" in godz_prefix.lower():
            h_word = POLISH_HOURS_LOC.get(hour, str(hour))
            prefix = godz_prefix
        elif godz_prefix:
            h_word = POLISH_HOURS_NOM.get(hour, str(hour))
            prefix = "godzina "
        else:
            h_word = POLISH_HOURS_NOM.get(hour, str(hour))
            prefix = ""

        if prefix and not prefix.endswith(" "):
            prefix = f"{prefix} "

        if minute == 0:
            return f"{prefix}{h_word}".strip()
        min_word = pl_cardinal(minute, case="nom")
        return f"{prefix}{h_word} {min_word}".strip()

    def _sub_time_between(m: re.Match) -> str:
        lead = m.group(1)

        def inst(h: int, minute: int) -> str:
            word = POLISH_HOURS_INST.get(h, str(h))
            return word if minute == 0 else f"{word} {pl_cardinal(minute, case='nom')}"

        return f"{lead} {inst(int(m.group(2)), int(m.group(3)))} a {inst(int(m.group(4)), int(m.group(5)))}"

    s = RE_TIME_BETWEEN_PL.sub(_sub_time_between, s)
    s = RE_TIME_PL.sub(_sub_time, s)

    # 5. Range: "od X do Y [unit]"
    def _sub_range(m: re.Match) -> str:
        od_word = m.group(1)
        x_str = m.group(2).replace(",", ".")
        y_str = m.group(3).replace(",", ".")
        unit_str = (m.group(4) or "").strip().lower()

        val_x = float(x_str) if "." in x_str else int(x_str)
        val_y = float(y_str) if "." in y_str else int(y_str)

        if not unit_str:
            # Bare range without a unit ("od 5 do 10"): plain genitive on both ends, no noun —
            # the old default noun read every unitless range as degrees ("stopni") (F1).
            return f"{od_word} {inflect_number(val_x, case='gen')} do {inflect_number(val_y, case='gen')}"

        noun = "stopień"
        suffix = ""
        full_match_text = m.group(0).lower()

        if "celsjusz" in full_match_text or "°c" in full_match_text:
            noun = "stopień"
            suffix = "Celsjusza"
        elif unit_str in UNIT_SYMBOL_TO_NOUN:
            noun, suffix = UNIT_SYMBOL_TO_NOUN[unit_str]
        elif unit_str in NOUN_DECLENSIONS:
            noun = unit_str
        elif unit_str:
            noun = unit_str

        decl_info = NOUN_DECLENSIONS.get(noun)
        gender = decl_info[4] if decl_info else "m"

        w_x = inflect_number(val_x, case="gen", gender=gender)
        w_y = inflect_number(val_y, case="gen", gender=gender)
        u_word = agree_noun(val_y, noun, case="gen")

        parts = [f"{od_word} {w_x} do {w_y} {u_word}"]
        if suffix:
            parts.append(suffix)
        return " ".join(parts).strip()

    s = RE_RANGE_OD_DO.sub(_sub_range, s)

    # 6. Fraction: "X z Y [unit]" (out of)
    def _sub_fraction(m: re.Match) -> str:
        x_str = m.group(1).replace(",", ".")
        z_word = m.group(2)
        y_str = m.group(3).replace(",", ".")
        unit_str = (m.group(4) or "").strip().lower()

        val_x = float(x_str) if "." in x_str else int(x_str)
        val_y = float(y_str) if "." in y_str else int(y_str)

        noun = "część"
        suffix = ""
        if unit_str in UNIT_SYMBOL_TO_NOUN:
            noun, suffix = UNIT_SYMBOL_TO_NOUN[unit_str]
        elif unit_str in NOUN_DECLENSIONS:
            noun = unit_str
        elif unit_str:
            noun = unit_str

        decl_info = NOUN_DECLENSIONS.get(noun)
        gender = decl_info[4] if decl_info else "m"

        w_x = inflect_number(val_x, case="nom", gender=gender)
        w_y = inflect_number(val_y, case="gen", gender=gender)

        if unit_str:
            u_word = agree_noun(val_y, noun, case="gen")
            parts = [f"{w_x} {z_word} {w_y} {u_word}"]
            if suffix:
                parts.append(suffix)
            return " ".join(parts).strip()
        return f"{w_x} {z_word} {w_y}"

    s = RE_FRACTION_Z.sub(_sub_fraction, s)

    # 7. Preposition + number + noun/unit (od, do, z, ze, około, ok., bez, u, dla, spośród)
    def _sub_prep_num(m: re.Match) -> str:
        lead_to = m.group(1) or ""
        prep = m.group(2)
        num_str = m.group(3).replace(",", ".")
        unit_str = (m.group(4) or "").strip().lower()

        val = float(num_str) if "." in num_str else int(num_str)
        prep_clean = "około" if prep.lower() == "ok." else prep
        is_to_okolo = bool(lead_to and prep.lower() in ("około", "ok."))
        case = "nom" if is_to_okolo else "gen"

        noun = ""
        suffix = ""
        if unit_str in ("°c", "stopni celsjusza", "stopnie celsjusza", "stopnia celsjusza"):
            noun = "stopień"
            suffix = "Celsjusza"
        elif unit_str in ("°", "stopni", "stopnie", "stopnia"):
            noun = "stopień"
        elif unit_str == "%":
            noun = "procent"
        elif unit_str in UNIT_SYMBOL_TO_NOUN:
            noun, suffix = UNIT_SYMBOL_TO_NOUN[unit_str]
        elif unit_str in NOUN_DECLENSIONS:
            noun = unit_str

        if noun:
            decl_info = NOUN_DECLENSIONS.get(noun)
            gender = decl_info[4] if decl_info else "m"
            w_num = inflect_number(val, case=case, gender=gender)
            u_word = agree_noun(val, noun, case=case)
            parts = [f"{lead_to}{prep_clean} {w_num} {u_word}"]
            if suffix:
                parts.append(suffix)
            return " ".join(parts).strip()

        w_num = inflect_number(val, case=case, gender="m")
        if unit_str:
            return f"{lead_to}{prep_clean} {w_num} {unit_str}"
        return f"{lead_to}{prep_clean} {w_num}"

    s = RE_PREP_NUM.sub(_sub_prep_num, s)

    # 8. Temperatures: "-5°C" / "12°C" / "12 °C" / "10.8°C"
    def _sub_temp(m: re.Match) -> str:
        sign = "minus " if m.group(1) == "-" else ""
        num_str = m.group(2).replace(",", ".")
        val = float(num_str) if "." in num_str else int(num_str)
        val_abs = abs(val)

        if isinstance(val_abs, float) and not val_abs.is_integer():
            deg_word = "stopnia"
            val_word = pl_cardinal_float(val_abs, case="nom")
        else:
            n_int = int(val_abs)
            deg_word = agree_noun(n_int, "stopień", case="nom")
            val_word = pl_cardinal(n_int, case="nom")
        return f"{sign}{val_word} {deg_word} Celsjusza"

    s = RE_TEMP_PL.sub(_sub_temp, s)

    # Standalone degrees: "12°"
    def _sub_deg(m: re.Match) -> str:
        sign = "minus " if m.group(1) == "-" else ""
        num_str = m.group(2).replace(",", ".")
        val = float(num_str) if "." in num_str else int(num_str)
        val_abs = abs(val)

        if isinstance(val_abs, float) and not val_abs.is_integer():
            deg_word = "stopnia"
            val_word = pl_cardinal_float(val_abs, case="nom")
        else:
            n_int = int(val_abs)
            deg_word = agree_noun(n_int, "stopień", case="nom")
            val_word = pl_cardinal(n_int, case="nom")
        return f"{sign}{val_word} {deg_word}"

    s = RE_DEG_PL.sub(_sub_deg, s)

    # 9. Percentages: "15%" / "12 %" / "2.5%" / "-2,5%"
    def _sub_pct(m: re.Match) -> str:
        sign = "minus " if m.group(1) == "-" else ""
        num_str = m.group(2).replace(",", ".")
        val = float(num_str) if "." in num_str else int(num_str)
        val_word = inflect_number(val, case="nom", gender="m")
        return f"{sign}{val_word} procent"

    s = RE_PCT_PL.sub(_sub_pct, s)

    # 10. Currency: "4,38 zł" / "4.38 PLN" / "10 zł" / "100 EUR" / "100 €" / "50 USD"
    def _sub_currency_pln(m: re.Match) -> str:
        num_str = m.group(1).replace(",", ".")
        if "." in num_str:
            parts = num_str.split(".")
            zlotych = int(parts[0])
            grosze = int(parts[1][:2].ljust(2, "0"))
            zl_phrase = inflect_number_noun(zlotych, "złoty", case="nom")
            if grosze > 0:
                gr_phrase = inflect_number_noun(grosze, "grosz", case="nom")
                return f"{zl_phrase} i {gr_phrase}"
            return zl_phrase
        val = int(num_str)
        return inflect_number_noun(val, "złoty", case="nom")

    s = RE_CURRENCY_PLN.sub(_sub_currency_pln, s)

    def _convert_foreign(num_str: str, curr: str) -> str:
        num_clean = num_str.replace(",", ".")
        val = float(num_clean) if "." in num_clean else int(num_clean)
        curr_upper = curr.upper()

        if curr_upper in ("EUR", "€", "EURO"):
            return inflect_number_noun(val, "euro", case="nom")
        if curr_upper in ("USD", "$", "DOLARÓW", "DOLAR", "DOLARY"):
            return inflect_number_noun(val, "dolar", case="nom")
        if curr_upper in ("GBP", "£", "FUNTÓW", "FUNT", "FUNTY"):
            return inflect_number_noun(val, "funt", case="nom")
        if curr_upper in ("CHF", "FRANKÓW", "FRANK", "FRANKI"):
            return inflect_number_noun(val, "frank", case="nom")
        w_val = inflect_number(val, case="nom")
        return f"{w_val} {curr}"

    s = RE_CURRENCY_SYM_PRE.sub(lambda m: _convert_foreign(m.group(2), m.group(1)), s)
    s = RE_CURRENCY_SYM_POST.sub(lambda m: _convert_foreign(m.group(1), m.group(2)), s)

    # 11. Units with numbers: km/h, m/s, km, kg, m, cm, mm, GB, MiB, MB, godziny, minuty, mile, dni
    for pat, noun_name, suffix in UNITS_MAP_PL:

        def _make_unit_sub(n_name=noun_name, suff=suffix):
            def _sub(m: re.Match) -> str:
                num_str = m.group(1).replace(",", ".")
                val = float(num_str) if "." in num_str else int(num_str)
                res = inflect_number_noun(val, n_name, case="nom")
                return f"{res} {suff}".strip() if suff else res

            return _sub

        s = pat.sub(_make_unit_sub(), s)

    # 12. Abbreviations
    def _sub_street_prep(m: re.Match) -> str:
        prep = m.group(1)
        abbr = m.group(2).lower()
        prep_lower = prep.lower()
        if prep_lower in ("na", "przy", "w", "we"):
            case = "loc"
        elif prep_lower in ("z okolic", "spod", "z", "ze", "do", "od", "koło", "obok", "blisko"):
            case = "gen"
        else:
            case = "nom"
        word = STREET_ABBR_FORMS[abbr][case]
        return f"{prep} {word}"

    def _sub_street_standalone(m: re.Match) -> str:
        abbr = m.group(1)
        word = STREET_ABBR_FORMS[abbr.lower()]["nom"]
        if abbr[0].isupper():
            word = word.capitalize()
        return word

    s = RE_STREET_PREP.sub(_sub_street_prep, s)
    s = RE_STREET_STANDALONE.sub(_sub_street_standalone, s)

    def _sub_title_prep(m: re.Match) -> str:
        prep = m.group(1)
        abbr = m.group(2).lower()
        prep_lower = prep.lower()
        if prep_lower in ("u", "do", "od", "dla"):
            case = "gen"
        elif prep_lower in ("z", "ze"):
            case = "inst"
        elif prep_lower in ("o",):
            case = "loc"
        else:
            case = "nom"
        word = TITLE_ABBR_FORMS[abbr][case]
        return f"{prep} {word}"

    def _sub_title_standalone(m: re.Match) -> str:
        abbr = m.group(1)
        word = TITLE_ABBR_FORMS[abbr.lower()]["nom"]
        if abbr[0].isupper():
            word = word.capitalize()
        return word

    s = RE_TITLE_PREP.sub(_sub_title_prep, s)
    s = RE_TITLE_STANDALONE.sub(_sub_title_standalone, s)

    def _sub_godz_prep(m: re.Match) -> str:
        prep = m.group(1)
        prep_lower = prep.lower()
        if prep_lower == "o":
            return f"{prep} godzinie"
        if prep_lower in ("od", "do"):
            return f"{prep} godziny"
        if prep_lower == "przed":
            return f"{prep} godziną"
        if prep_lower == "po":
            return f"{prep} godzinie"
        return f"{prep} godzina"

    s = RE_GODZ_PREP.sub(_sub_godz_prep, s)

    for pat, repl in PL_ABBREVIATIONS:
        s = pat.sub(repl, s)

    # 13. Negative numbers before decimals, so the sign is spoken for "-10,8" too (F2)
    def _sub_negative(m: re.Match) -> str:
        num_str = m.group(2)
        if "," in num_str or "." in num_str:
            int_part, _, dec_part = num_str.replace(".", ",").partition(",")
            words = (
                f"{pl_cardinal(int(int_part), case='nom')} przecinek "
                f"{pl_cardinal(int(dec_part), case='nom')}"
            )
        else:
            words = pl_cardinal(int(num_str), case="nom")
        return f"{m.group(1)}minus {words}"

    s = RE_NEGATIVE_NUM.sub(_sub_negative, s)

    # 14. Decimals: "10,8" or "10.8"
    def _sub_decimal(m: re.Match) -> str:
        int_part = int(m.group(1))
        dec_part = int(m.group(2))
        w_int = pl_cardinal(int_part, case="nom")
        w_dec = pl_cardinal(dec_part, case="nom")
        return f"{w_int} przecinek {w_dec}"

    s = RE_DECIMAL_PL.sub(_sub_decimal, s)

    # 15. Standalone integers
    def _sub_int(m: re.Match) -> str:
        n = int(m.group(1))
        return pl_cardinal(n, case="nom")

    s = RE_INTEGER_PL.sub(_sub_int, s)

    # 16. Word-level inflection cleanup (fixes words already rendered in nominative after prepositions)
    s = _sub_word_number_genitive(s)

    s = RE_CLEAN_WS.sub(" ", s).strip()
    return s


def _normalize_en(text: str) -> str:
    s = text.replace("\xa0", " ")

    # 1. Abbreviations
    for pat, repl in EN_ABBREVIATIONS:
        s = pat.sub(repl, s)

    # 2. Times: "17:59" / "06:54" / "at 17:59"
    def _sub_time_en(m: re.Match) -> str:
        h = int(m.group(1))
        minute = int(m.group(2))
        h_word = num2words.num2words(h, lang="en")
        if minute == 0:
            return f"{h_word} o'clock"
        if minute < 10:
            m_word = f"oh {num2words.num2words(minute, lang='en')}"
        else:
            m_word = num2words.num2words(minute, lang="en")
        return f"{h_word} {m_word}"

    s = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b").sub(_sub_time_en, s)

    # 3. Temperatures: "12°C", "72°F", "-5°C"
    def _sub_temp_en(m: re.Match) -> str:
        sign = "minus " if m.group(1) == "-" else ""
        val = float(m.group(2))
        scale = "Celsius" if "c" in m.group(3).lower() else "Fahrenheit"
        val_word = num2words.num2words(int(val) if val.is_integer() else val, lang="en")
        deg_word = "degree" if val == 1.0 else "degrees"
        return f"{sign}{val_word} {deg_word} {scale}"

    s = re.compile(r"(-)?(\d+(?:\.\d+)?)\s*°\s*([CF])\b", re.I).sub(_sub_temp_en, s)

    # 4. Percentages: "15%"
    def _sub_pct_en(m: re.Match) -> str:
        val = float(m.group(1))
        val_word = num2words.num2words(int(val) if val.is_integer() else val, lang="en")
        return f"{val_word} percent"

    s = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.I).sub(_sub_pct_en, s)

    # 5. Currency: "$50", "50 USD", "€100", "£20"
    def _convert_curr_en(val_num: float, sym: str) -> str:
        val_word = num2words.num2words(int(val_num) if val_num.is_integer() else val_num, lang="en")
        if sym in ("$", "USD"):
            c_word = "dollar" if val_num == 1.0 else "dollars"
        elif sym in ("€", "EUR"):
            c_word = "euro" if val_num == 1.0 else "euros"
        elif sym in ("£", "GBP"):
            c_word = "pound" if val_num == 1.0 else "pounds"
        else:
            c_word = sym
        return f"{val_word} {c_word}"

    s = re.compile(r"([\$€£])\s*(\d+(?:\.\d+)?)").sub(
        lambda m: _convert_curr_en(float(m.group(2)), m.group(1)), s
    )
    s = re.compile(r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP)\b", re.I).sub(
        lambda m: _convert_curr_en(float(m.group(1)), m.group(2)), s
    )

    # 6. Units: km/h, mph, kg, km, etc.
    for pat, u_name in UNITS_MAP_EN:

        def _sub_u(m: re.Match, unit_text=u_name) -> str:
            val = float(m.group(1))
            val_word = num2words.num2words(int(val) if val.is_integer() else val, lang="en")
            return f"{val_word} {unit_text}"

        s = pat.sub(_sub_u, s)

    # 7. Negative numbers before decimals, so "-2.5" keeps its minus (F2); num2words speaks
    # floats as "two point five" directly.
    s = RE_NEGATIVE_NUM.sub(
        lambda m: (
            f"{m.group(1)}minus "
            f"{num2words.num2words(float(m.group(2).replace(',', '.')), lang='en')}"
        ),
        s,
    )

    # 8. Decimals: "10.8"
    def _sub_dec_en(m: re.Match) -> str:
        w_int = num2words.num2words(int(m.group(1)), lang="en")
        w_dec = num2words.num2words(int(m.group(2)), lang="en")
        return f"{w_int} point {w_dec}"

    s = re.compile(r"\b(\d+)\.(\d+)\b").sub(_sub_dec_en, s)

    # 9. Standalone integers
    def _sub_int_en(m: re.Match) -> str:
        return num2words.num2words(int(m.group(1)), lang="en")

    s = re.compile(r"\b(\d+)\b").sub(_sub_int_en, s)

    s = RE_CLEAN_WS.sub(" ", s).strip()
    return s


def normalize_for_tts(text: str, lang: str = "pl") -> str:
    """Normalize text into phonetically speakable words for TTS (Piper/Paradee).

    Handles numbers, dates, times, currencies, units, percentages, abbreviations,
    and Roman numerals safely and with low latency (< 1 ms).
    """
    if not text or not text.strip():
        return ""
    try:
        if lang == "en":
            return _normalize_en(text)
        return _normalize_pl(text)
    except Exception:
        # Failsafe: never crash, return original text on unexpected error
        return text
