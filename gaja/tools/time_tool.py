from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult

PL_MONTHS = [
    "",
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
]

PL_WEEKDAYS = [
    "poniedziałek",
    "wtorek",
    "środa",
    "czwartek",
    "piątek",
    "sobota",
    "niedziela",
]

EN_MONTHS = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

EN_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


async def get_time(timezone: str = "Europe/Warsaw", lang: str = "pl") -> ToolResult:
    """Return local date/time, weekday, and timezone (Europe/Warsaw)."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Europe/Warsaw")
        timezone = "Europe/Warsaw"

    now = datetime.datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    weekday_idx = now.weekday()
    month_idx = now.month

    if lang == "en":
        weekday_en = EN_WEEKDAYS[weekday_idx]
        month_en = EN_MONTHS[month_idx]
        raw_speak = f"It is {weekday_en}, {month_en} {now.day}, {now.year}, {time_str}."
        weekday = weekday_en
    else:
        weekday_pl = PL_WEEKDAYS[weekday_idx]
        month_pl = PL_MONTHS[month_idx]
        raw_speak = f"Jest {weekday_pl}, {now.day} {month_pl} {now.year} roku, godzina {time_str}."
        weekday = weekday_pl

    speak = normalize_for_tts(raw_speak, lang)
    data = {
        "date": date_str,
        "time": time_str,
        "weekday": weekday,
        "timezone": timezone,
        "iso": now.isoformat(),
    }
    return ToolResult(data=data, speak=speak)
