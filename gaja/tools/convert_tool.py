from __future__ import annotations

import unicodedata
from typing import Any

from gaja.text.normalize import (
    NOUN_DECLENSIONS,
    agree_noun,
    inflect_number_noun,
    normalize_for_tts,
    pl_cardinal,
)
from gaja.tools.base import ToolResult, format_number_pl

# Canonical unit definitions by category
# Base units:
#   length: meter (m)
#   mass: kilogram (kg)
#   volume: liter (l)
#   speed: meter per second (m/s)
#   time: second (s)
#   temperature: handled with custom functions

UNIT_CATEGORIES: dict[str, str] = {
    # Length
    "m": "length",
    "km": "length",
    "cm": "length",
    "mm": "length",
    "mi": "length",
    "yd": "length",
    "ft": "length",
    "in": "length",
    # Mass
    "kg": "mass",
    "g": "mass",
    "mg": "mass",
    "lb": "mass",
    "oz": "mass",
    "t": "mass",
    # Temperature
    "C": "temperature",
    "F": "temperature",
    "K": "temperature",
    # Volume
    "l": "volume",
    "ml": "volume",
    "gal": "volume",
    "pt": "volume",
    "m3": "volume",
    # Speed
    "km/h": "speed",
    "m/s": "speed",
    "mph": "speed",
    "knot": "speed",
    # Time
    "s": "time",
    "min": "time",
    "h": "time",
    "d": "time",
    "wk": "time",
}

# Conversion factor to category base unit
UNIT_TO_BASE: dict[str, float] = {
    # Length -> meters
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "yd": 0.9144,
    "ft": 0.3048,
    "in": 0.0254,
    # Mass -> kilograms
    "kg": 1.0,
    "g": 0.001,
    "mg": 1e-6,
    "lb": 0.45359237,
    "oz": 0.028349523125,
    "t": 1000.0,
    # Volume -> liters
    "l": 1.0,
    "ml": 0.001,
    "gal": 3.785411784,
    "pt": 0.473176473,
    "m3": 1000.0,
    # Speed -> m/s
    "m/s": 1.0,
    "km/h": 1.0 / 3.6,
    "mph": 0.44704,
    "knot": 0.5144444444444445,
    # Time -> seconds
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "wk": 604800.0,
}

# Polish noun mapping for inflection: canon_unit -> (noun_key, suffix)
CANON_UNIT_TO_NOUN: dict[str, tuple[str, str]] = {
    "m": ("metr", ""),
    "km": ("kilometr", ""),
    "cm": ("centymetr", ""),
    "mm": ("milimetr", ""),
    "mi": ("mila", ""),
    "yd": ("jard", ""),
    "ft": ("stopa", ""),
    "in": ("cal", ""),
    "kg": ("kilogram", ""),
    "g": ("gram", ""),
    "mg": ("miligram", ""),
    "lb": ("funt", ""),
    "oz": ("uncja", ""),
    "t": ("tona", ""),
    "C": ("stopień", "Celsjusza"),
    "F": ("stopień", "Fahrenheita"),
    "K": ("kelwin", ""),
    "l": ("litr", ""),
    "ml": ("mililitr", ""),
    "gal": ("galon", ""),
    "pt": ("pinta", ""),
    "m3": ("metr", "sześcienny"),
    "km/h": ("kilometr", "na godzinę"),
    "m/s": ("metr", "na sekundę"),
    "mph": ("mila", "na godzinę"),
    "knot": ("węzeł", ""),
    "s": ("sekunda", ""),
    "min": ("minuta", ""),
    "h": ("godzina", ""),
    "d": ("dzień", ""),
    "wk": ("tydzień", ""),
}

UNIT_NAMES_EN: dict[str, str] = {
    "m": "meters",
    "km": "kilometers",
    "cm": "centimeters",
    "mm": "millimeters",
    "mi": "miles",
    "yd": "yards",
    "ft": "feet",
    "in": "inches",
    "kg": "kilograms",
    "g": "grams",
    "mg": "milligrams",
    "lb": "pounds",
    "oz": "ounces",
    "t": "metric tons",
    "C": "degrees Celsius",
    "F": "degrees Fahrenheit",
    "K": "kelvins",
    "l": "liters",
    "ml": "milliliters",
    "gal": "gallons",
    "pt": "pints",
    "m3": "cubic meters",
    "km/h": "kilometers per hour",
    "m/s": "meters per second",
    "mph": "miles per hour",
    "knot": "knots",
    "s": "seconds",
    "min": "minutes",
    "h": "hours",
    "d": "days",
    "wk": "weeks",
}

UNIT_NAMES_EN_SG: dict[str, str] = {
    "m": "meter",
    "km": "kilometer",
    "cm": "centimeter",
    "mm": "millimeter",
    "mi": "mile",
    "yd": "yard",
    "ft": "foot",
    "in": "inch",
    "kg": "kilogram",
    "g": "gram",
    "mg": "milligram",
    "lb": "pound",
    "oz": "ounce",
    "t": "metric ton",
    "C": "degree Celsius",
    "F": "degree Fahrenheit",
    "K": "kelvin",
    "l": "liter",
    "ml": "milliliter",
    "gal": "gallon",
    "pt": "pint",
    "m3": "cubic meter",
    "km/h": "kilometer per hour",
    "m/s": "meter per second",
    "mph": "mile per hour",
    "knot": "knot",
    "s": "second",
    "min": "minute",
    "h": "hour",
    "d": "day",
    "wk": "week",
}


def _format_unit_en(number_val: float | int, canon_u: str) -> str:
    if number_val == 1:
        return UNIT_NAMES_EN_SG.get(canon_u, canon_u)
    return UNIT_NAMES_EN.get(canon_u, canon_u)

UNIT_ALIASES: dict[str, str] = {
    # Length
    "m": "m",
    "metr": "m",
    "metry": "m",
    "metrow": "m",
    "metrowych": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "km": "km",
    "kilometr": "km",
    "kilometry": "km",
    "kilometrow": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "cm": "cm",
    "centymetr": "cm",
    "centymetry": "cm",
    "centymetrow": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "milimetr": "mm",
    "milimetry": "mm",
    "milimetrow": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "mi": "mi",
    "mila": "mi",
    "mile": "mi",
    "mili": "mi",
    "mil": "mi",
    "miles": "mi",
    "yd": "yd",
    "jard": "yd",
    "jardy": "yd",
    "jardow": "yd",
    "yard": "yd",
    "yards": "yd",
    "ft": "ft",
    "stopa": "ft",
    "stopy": "ft",
    "stop": "ft",
    "foot": "ft",
    "feet": "ft",
    "in": "in",
    "cal": "in",
    "cale": "in",
    "cali": "in",
    "inch": "in",
    "inches": "in",
    # Mass
    "kg": "kg",
    "kilogram": "kg",
    "kilogramy": "kg",
    "kilogramow": "kg",
    "kilogramme": "kg",
    "kilograms": "kg",
    "g": "g",
    "gram": "g",
    "gramy": "g",
    "gramow": "g",
    "grams": "g",
    "mg": "mg",
    "miligram": "mg",
    "miligramy": "mg",
    "miligramow": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "lb": "lb",
    "lbs": "lb",
    "funt": "lb",
    "funty": "lb",
    "funtow": "lb",
    "pound": "lb",
    "pounds": "lb",
    "oz": "oz",
    "uncja": "oz",
    "uncje": "oz",
    "uncji": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "t": "t",
    "tona": "t",
    "tony": "t",
    "ton": "t",
    "tonne": "t",
    "tonnes": "t",
    "tons": "t",
    # Temperature
    "c": "C",
    "celsjusz": "C",
    "celsjusza": "C",
    "celsjusze": "C",
    "celsius": "C",
    "°c": "C",
    "f": "F",
    "fahrenheit": "F",
    "fahrenheita": "F",
    "fahrenheitow": "F",
    "°f": "F",
    "k": "K",
    "kelvin": "K",
    "kelvina": "K",
    "kelvins": "K",
    "kelwin": "K",
    "kelwina": "K",
    # Volume
    "l": "l",
    "litr": "l",
    "litry": "l",
    "litrow": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "ml": "ml",
    "mililitr": "ml",
    "mililitry": "ml",
    "mililitrow": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "gal": "gal",
    "galon": "gal",
    "galony": "gal",
    "galonow": "gal",
    "gallon": "gal",
    "gallons": "gal",
    "pt": "pt",
    "pinta": "pt",
    "pinty": "pt",
    "pint": "pt",
    "pints": "pt",
    "m3": "m3",
    "m^3": "m3",
    "metrszescienny": "m3",
    "metryszescienne": "m3",
    "metrowszesciennych": "m3",
    # Speed
    "km/h": "km/h",
    "kmh": "km/h",
    "kph": "km/h",
    "kilometrnanagodzine": "km/h",
    "kilometrynagodzine": "km/h",
    "kilometrownagodzine": "km/h",
    "m/s": "m/s",
    "ms": "m/s",
    "metrnanasekunde": "m/s",
    "metrynasekunde": "m/s",
    "metrownasekunde": "m/s",
    "mph": "mph",
    "mi/h": "mph",
    "milanagodzine": "mph",
    "milenagodzine": "mph",
    "milnagodzine": "mph",
    "milesperhour": "mph",
    "knot": "knot",
    "knots": "knot",
    "wezel": "knot",
    "wezly": "knot",
    "wezlow": "knot",
    # Time
    "s": "s",
    "sec": "s",
    "sek": "s",
    "sekunda": "s",
    "sekundy": "s",
    "sekund": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "minuta": "min",
    "minuty": "min",
    "minut": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "godz": "h",
    "godzina": "h",
    "godziny": "h",
    "godzin": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "dzien": "d",
    "dni": "d",
    "doba": "d",
    "doby": "d",
    "dob": "d",
    "day": "d",
    "days": "d",
    "wk": "wk",
    "tydzien": "wk",
    "tygodnie": "wk",
    "tygodni": "wk",
    "week": "wk",
    "weeks": "wk",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _resolve_unit(name: str) -> str | None:
    raw = name.strip()
    if not raw:
        return None
    # Exact check
    if raw in UNIT_ALIASES:
        return UNIT_ALIASES[raw]
    low = raw.lower().replace(" ", "")
    if low in UNIT_ALIASES:
        return UNIT_ALIASES[low]
    clean = _strip_accents(low)
    if clean in UNIT_ALIASES:
        return UNIT_ALIASES[clean]
    return None


def _convert_temperature(val: float, from_u: str, to_u: str) -> float:
    # Convert from_u to Celsius first
    if from_u == "C":
        c = val
    elif from_u == "F":
        c = (val - 32.0) * 5.0 / 9.0
    elif from_u == "K":
        c = val - 273.15
    else:
        raise ValueError(f"Nieznana jednostka temperatury: {from_u}")

    # Convert Celsius to to_u
    if to_u == "C":
        return c
    elif to_u == "F":
        return (c * 9.0 / 5.0) + 32.0
    elif to_u == "K":
        return c + 273.15
    else:
        raise ValueError(f"Nieznana jednostka temperatury: {to_u}")


async def convert_units(
    value: float | int,
    from_unit: str,
    to_unit: str,
    lang: str = "pl",
) -> ToolResult:
    """Converts units of measurement across length, mass, temperature, volume, speed, time."""
    canon_from = _resolve_unit(from_unit)
    canon_to = _resolve_unit(to_unit)

    if not canon_from:
        err = f"Nieznana jednostka źródłowa: {from_unit}"
        speak = f"Nie znam jednostki {from_unit}." if lang == "pl" else f"Unknown source unit {from_unit}."
        return ToolResult(data={"error": err}, speak=speak, error=err)

    if not canon_to:
        err = f"Nieznana jednostka docelowa: {to_unit}"
        speak = f"Nie znam jednostki {to_unit}." if lang == "pl" else f"Unknown target unit {to_unit}."
        return ToolResult(data={"error": err}, speak=speak, error=err)

    cat_from = UNIT_CATEGORIES.get(canon_from)
    cat_to = UNIT_CATEGORIES.get(canon_to)

    if not cat_from or not cat_to or cat_from != cat_to:
        err = f"Niekompatybilne kategorie jednostek: {cat_from or 'nieznana'} i {cat_to or 'nieznana'}."
        speak = (
            "Nie można przeliczyć jednostek z różnych kategorii."
            if lang == "pl"
            else "Cannot convert units belonging to different categories."
        )
        return ToolResult(data={"error": err}, speak=speak, error=err)

    try:
        val_num = float(value)
        if cat_from == "temperature":
            result = _convert_temperature(val_num, canon_from, canon_to)
        else:
            base_val = val_num * UNIT_TO_BASE[canon_from]
            result = base_val / UNIT_TO_BASE[canon_to]

        # Exact result in ToolResult.data
        exact_result: float | int = int(result) if result.is_integer() else result
        val_disp: float | int = int(val_num) if val_num.is_integer() else val_num

        data = {
            "value": val_disp,
            "from_unit": canon_from,
            "to_unit": canon_to,
            "result": exact_result,
            "category": cat_from,
        }

        # Round spoken results to max 1 decimal place, and to an integer when |x| >= 100 or when the decimal is .0
        if abs(result) >= 100:
            spoken_val: float | int = int(round(result))
        else:
            r1 = round(result, 1)
            spoken_val = int(r1) if r1.is_integer() else r1

        is_approx = not (result == spoken_val)

        if lang == "en":
            u_from_en = _format_unit_en(val_disp, canon_from)
            u_to_en = _format_unit_en(spoken_val, canon_to)
            approx_str = "about " if is_approx else ""
            raw_speak = f"{val_disp} {u_from_en} is {approx_str}{spoken_val} {u_to_en}."
        else:
            def _format_unit_phrase(number_val: float | int, canon_u: str, case: str = "nom") -> str:
                if canon_u in CANON_UNIT_TO_NOUN:
                    n_key, suff = CANON_UNIT_TO_NOUN[canon_u]
                    decl = NOUN_DECLENSIONS.get(n_key)
                    gender = decl[4] if decl else "m"
                    noun_form = agree_noun(number_val, n_key, case=case)

                    if canon_u == "m3":
                        if isinstance(number_val, float) and not number_val.is_integer():
                            adj = "sześciennego"
                        elif case == "gen":
                            adj = "sześciennego" if abs(int(number_val)) == 1 else "sześciennych"
                        else:
                            n_int = abs(int(number_val))
                            if n_int == 1:
                                adj = "sześcienny"
                            elif n_int % 10 in (2, 3, 4) and n_int % 100 not in (12, 13, 14):
                                adj = "sześcienne"
                            else:
                                adj = "sześciennych"
                        suff = adj

                    if isinstance(number_val, float) and not number_val.is_integer():
                        int_p = int(number_val)
                        dec_p = int(round((number_val - int_p) * 10))
                        tenths_map = {
                            1: "jedna dziesiąta",
                            2: "dwie dziesiąte",
                            3: "trzy dziesiąte",
                            4: "cztery dziesiąte",
                            5: "pięć dziesiątych",
                            6: "sześć dziesiątych",
                            7: "siedem dziesiątych",
                            8: "osiem dziesiątych",
                            9: "dziewięć dziesiątych",
                        }
                        if int_p == 0:
                            num_str = tenths_map.get(dec_p, f"zero przecinek {dec_p}")
                        else:
                            int_str = pl_cardinal(int_p, case=case, gender=gender)
                            tenth_str = tenths_map.get(dec_p, f"przecinek {dec_p}")
                            num_str = f"{int_str} i {tenth_str}"
                    else:
                        num_str = pl_cardinal(int(number_val), case=case, gender=gender)

                    if suff:
                        return f"{num_str} {noun_form} {suff}".strip()
                    return f"{num_str} {noun_form}".strip()
                return f"{format_number_pl(number_val)} {canon_u}"

            from_phrase = _format_unit_phrase(val_disp, canon_from, case="nom")
            to_phrase = _format_unit_phrase(spoken_val, canon_to, case="nom")
            approx_prefix = "około " if is_approx else ""
            raw_speak = f"{from_phrase} to {approx_prefix}{to_phrase}."

        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data=data, speak=speak)

    except Exception as exc:
        err = f"Błąd konwersji jednostek: {exc}"
        speak = "Nie udało się przeliczyć jednostek." if lang == "pl" else "Could not convert units."
        return ToolResult(data={"error": err}, speak=speak, error=err)
