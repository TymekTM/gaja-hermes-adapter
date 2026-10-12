from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo
from typing import Any

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult, polish_declension
from gaja.tools.time_tool import PL_MONTHS, EN_MONTHS


def calculate_easter(year: int) -> datetime.date:
    """Meeus/Jones/Butcher algorithm for Gregorian Easter."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def get_polish_holidays(year: int) -> list[tuple[datetime.date, str, str]]:
    """Return all statutory Polish public holidays for the given year (date, name_pl, name_en)."""
    easter = calculate_easter(year)
    holidays = [
        (datetime.date(year, 1, 1), "Nowy Rok", "New Year's Day"),
        (datetime.date(year, 1, 6), "Święto Trzech Króli", "Epiphany"),
        (easter, "Wielkanoc", "Easter Sunday"),
        (easter + datetime.timedelta(days=1), "Poniedziałek Wielkanocny", "Easter Monday"),
        (datetime.date(year, 5, 1), "Święto Pracy", "Labour Day"),
        (datetime.date(year, 5, 3), "Święto Narodowe Trzeciego Maja", "Constitution Day"),
        (easter + datetime.timedelta(days=49), "Zielone Świątki", "Pentecost"),
        (easter + datetime.timedelta(days=60), "Boże Ciało", "Corpus Christi"),
        (datetime.date(year, 8, 15), "Wniebowzięcie Najświętszej Maryi Panny", "Armed Forces Day"),
        (datetime.date(year, 11, 1), "Wszystkich Świętych", "All Saints' Day"),
        (datetime.date(year, 11, 11), "Narodowe Święto Niepodległości", "Independence Day"),
        (datetime.date(year, 12, 25), "Pierwszy dzień Bożego Narodzenia", "Christmas Day"),
        (datetime.date(year, 12, 26), "Drugi dzień Bożego Narodzenia", "Boxing Day"),
    ]
    return sorted(holidays, key=lambda x: x[0])


HOLIDAY_MAPPINGS: list[tuple[list[str], str, str]] = [
    (["boże", "boze", "narodzen", "gwiazdk", "wigil", "christmas"], "Pierwszy dzień Bożego Narodzenia", "Bożego Narodzenia"),
    (["wielkanoc", "easter"], "Wielkanoc", "Wielkanocy"),
    (["nowy rok", "sylwest", "nowego roku", "new year"], "Nowy Rok", "Nowego Roku"),
    (["trzech król", "trzech krol", "epiphany"], "Święto Trzech Króli", "Święta Trzech Króli"),
    (["prac", "1 maja", "labour"], "Święto Pracy", "Święta Pracy"),
    (["3 maja", "trzeci maja", "trzeciego maja", "konstytucj"], "Święto Narodowe Trzeciego Maja", "Święta Narodowego Trzeciego Maja"),
    (["zielon"], "Zielone Świątki", "Zielonych Świątek"),
    (["ciał", "cial"], "Boże Ciało", "Bożego Ciała"),
    (["wniebowzi", "maryi", "15 sierpnia"], "Wniebowzięcie Najświętszej Maryi Panny", "Wniebowzięcia Najświętszej Maryi Panny"),
    (["wszystkich święt", "wszystkich swiet", "1 listopada", "all saints"], "Wszystkich Świętych", "Wszystkich Świętych"),
    (["niepodległ", "niepodlegl", "11 listopada", "independence"], "Narodowe Święto Niepodległości", "Narodowego Święta Niepodległości"),
]


async def next_holiday(holiday: str | None = None, lang: str = "pl") -> ToolResult:
    """Find the next statutory Polish public holiday or days until a specific holiday."""
    tz = ZoneInfo("Europe/Warsaw")
    today = datetime.datetime.now(tz).date()

    candidates = get_polish_holidays(today.year) + get_polish_holidays(today.year + 1)
    upcoming = [h for h in candidates if h[0] >= today]

    target = None
    genitive_name = None
    if holiday and holiday.strip().lower() not in {"", "święta", "swieta", "święto", "swieto", "następne", "nastepne", "najbliższe", "najblizsze", "holiday", "next"}:
        norm = holiday.strip().lower()
        for stems, exact_name, genitive in HOLIDAY_MAPPINGS:
            if any(stem in norm for stem in stems):
                for h in upcoming:
                    if h[1] == exact_name:
                        target = h
                        genitive_name = genitive
                        break
                if target is not None:
                    break

    if target is None:
        if not upcoming:
            err = "Nie znaleziono nadchodzących świąt"
            return ToolResult(data={"error": err}, speak="Nie znaleziono nadchodzących świąt.", error=err)
        target = upcoming[0]

    target_date, name_pl, name_en = target
    days_until = (target_date - today).days

    name = name_en if lang == "en" else name_pl
    month_name = EN_MONTHS[target_date.month] if lang == "en" else PL_MONTHS[target_date.month]

    data = {
        "holiday": name_pl,
        "holiday_en": name_en,
        "date": target_date.isoformat(),
        "days_until": days_until,
    }

    if lang == "en":
        if days_until == 0:
            speak = f"Today is {name_en}!"
        elif days_until == 1:
            speak = f"Tomorrow is {name_en} ({month_name} {target_date.day})."
        elif genitive_name:
            speak = f"{name_en} is in {days_until} days, on {month_name} {target_date.day}."
        else:
            speak = f"The next public holiday is {name_en} on {month_name} {target_date.day}, in {days_until} days."
    else:
        days_word = polish_declension(days_until, "dzień", "dni", "dni")
        if days_until == 0:
            speak = f"Dzisiaj jest {name_pl}!"
        elif days_until == 1:
            speak = f"Jutro jest {name_pl} ({target_date.day} {month_name})."
        elif genitive_name:
            speak = f"Do {genitive_name} zostało {days_until} {days_word}, {target_date.day} {month_name}."
        else:
            speak = f"Najbliższe święto to {name_pl}, {target_date.day} {month_name}, za {days_until} {days_word}."

    speak = normalize_for_tts(speak, lang)
    return ToolResult(data=data, speak=speak)
