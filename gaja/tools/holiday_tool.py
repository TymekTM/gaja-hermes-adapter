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


async def next_holiday(lang: str = "pl") -> ToolResult:
    """Find the next statutory Polish public holiday relative to today."""
    tz = ZoneInfo("Europe/Warsaw")
    today = datetime.datetime.now(tz).date()

    candidates = get_polish_holidays(today.year) + get_polish_holidays(today.year + 1)
    upcoming = [h for h in candidates if h[0] >= today]

    if not upcoming:
        err = "Nie znaleziono nadchodzących świąt"
        return ToolResult(data={"error": err}, speak="Nie znaleziono nadchodzących świąt.", error=err)

    target_date, name_pl, name_en = upcoming[0]
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
        else:
            speak = f"The next public holiday is {name_en} on {month_name} {target_date.day}, in {days_until} days."
    else:
        days_word = polish_declension(days_until, "dzień", "dni", "dni")
        if days_until == 0:
            speak = f"Dzisiaj jest {name_pl}!"
        elif days_until == 1:
            speak = f"Jutro jest {name_pl} ({target_date.day} {month_name})."
        else:
            speak = f"Najbliższe święto to {name_pl}, {target_date.day} {month_name}, za {days_until} {days_word}."

    speak = normalize_for_tts(speak, lang)
    return ToolResult(data=data, speak=speak)
