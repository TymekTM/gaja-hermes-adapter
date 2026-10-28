from __future__ import annotations

import calendar
import datetime
import re
from zoneinfo import ZoneInfo

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult, polish_declension

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

MONTHS_PL_MAP: dict[str, int] = {
    "stycz": 1,
    "lut": 2,
    "mar": 3,
    "kwiec": 4,
    "maj": 5,
    "czerw": 6,
    "lipc": 7,
    "sierp": 8,
    "wrzes": 9,
    "wrześ": 9,
    "paździer": 10,
    "pazdzier": 10,
    "listopad": 11,
    "grudn": 12,
}

ORDINALS_PL: dict[str, int] = {
    "pierwsz": 1,
    "drug": 2,
    "trzeci": 3,
    "czwart": 4,
    "piąt": 5,
    "piat": 5,
    "szóst": 6,
    "szost": 6,
    "siódm": 7,
    "siodm": 7,
    "ósm": 8,
    "osm": 8,
    "dziewiąt": 9,
    "dziewiat": 9,
    "dziesiąt": 10,
    "dziesiat": 10,
    "jedenast": 11,
    "dwunast": 12,
    "trzynast": 13,
    "czternast": 14,
    "piętnast": 15,
    "pietnast": 15,
    "szesnast": 16,
    "siedemnast": 17,
    "osiemnast": 18,
    "dziewiętnast": 19,
    "dziewietnast": 19,
    "dwudziest": 20,
    "trzydziest": 30,
}


_NUMBER_WORDS: dict[str, int] = {
    "jeden": 1,
    "jedną": 1,
    "jedna": 1,
    "dwa": 2,
    "dwie": 2,
    "trzy": 3,
    "cztery": 4,
    "pięć": 5,
    "piec": 5,
    "sześć": 6,
    "szesc": 6,
    "siedem": 7,
    "osiem": 8,
    "dziewięć": 9,
    "dziewiec": 9,
    "dziesięć": 10,
    "dziesiec": 10,
}
_COUNT_WORD = r"(?:\d+|jeden|jedną|jedna|dwa|dwie|trzy|cztery|pięć|piec|sześć|szesc|siedem|osiem|dziewięć|dziewiec|dziesięć|dziesiec)"


def _parse_count(token: str | None) -> int:
    if not token:
        return 1
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token.lower(), 1)


def _add_months(base_date: datetime.date, months: int) -> datetime.date:
    month_index = base_date.month - 1 + months
    year = base_date.year + month_index // 12
    month = month_index % 12 + 1
    day = min(base_date.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def parse_target_date(text: str, base_date: datetime.date) -> datetime.date | None:
    """Parse relative or absolute date expressions in Polish or ISO."""
    t = text.lower().strip()
    if not t or t in ("teraz", "dziś", "dzis", "today", "now"):
        return None
    if t in ("jutro", "tomorrow"):
        return base_date + datetime.timedelta(days=1)
    if t in ("pojutrze", "day after tomorrow"):
        return base_date + datetime.timedelta(days=2)
    if t in ("wczoraj", "yesterday"):
        return base_date - datetime.timedelta(days=1)

    m_rel = re.search(rf"za\s+({_COUNT_WORD})\s+dn(?:i|ia|ień)", t)
    if m_rel:
        return base_date + datetime.timedelta(days=_parse_count(m_rel.group(1)))

    m_rel_wk = re.search(rf"za\s+(?:({_COUNT_WORD})\s+)?(?:tygodni|tydzi)", t)
    if m_rel_wk:
        return base_date + datetime.timedelta(weeks=_parse_count(m_rel_wk.group(1)))

    m_rel_mo = re.search(rf"za\s+(?:({_COUNT_WORD})\s+)?miesi(?:ąc|ące|ęcy)", t)
    if m_rel_mo:
        return _add_months(base_date, _parse_count(m_rel_mo.group(1)))

    if re.search(r"w\s+(?:przyszłym|przyszlym|następnym|nastepnym)\s+tygodniu", t):
        return base_date + datetime.timedelta(weeks=1)

    if re.search(r"w\s+(?:przyszłym|przyszlym|następnym|nastepnym)\s+(?:miesiącu|miesiacu)", t):
        return _add_months(base_date, 1)

    m_iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m_iso:
        try:
            return datetime.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            pass

    m_dot = re.search(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{4}))?", t)
    if m_dot:
        try:
            day = int(m_dot.group(1))
            month = int(m_dot.group(2))
            year = int(m_dot.group(3)) if m_dot.group(3) else base_date.year
            d = datetime.date(year, month, day)
            if not m_dot.group(3) and d < base_date:
                d = datetime.date(year + 1, month, day)
            return d
        except ValueError:
            pass

    month = None
    for prefix, m_num in MONTHS_PL_MAP.items():
        if prefix in t:
            month = m_num
            break

    if month is not None:
        m_day = re.search(r"(\d{1,2})\s*(?:go\s+)?(?:\w+)?", t)
        day = None
        if m_day:
            day = int(m_day.group(1))
        else:
            d_val = 0
            if "dwudziest" in t:
                d_val += 20
            elif "trzydziest" in t:
                d_val += 30
            for ord_p, val in ORDINALS_PL.items():
                if val < 10 and ord_p in t:
                    d_val += val
                    break
                elif val >= 10 and ord_p in t and d_val == 0:
                    d_val = val
                    break
            if d_val > 0:
                day = d_val

        if day is not None:
            m_yr = re.search(r"(20\d\d)", t)
            year = int(m_yr.group(1)) if m_yr else base_date.year
            try:
                d = datetime.date(year, month, day)
                if not m_yr and d < base_date:
                    d = datetime.date(year + 1, month, day)
                return d
            except ValueError:
                pass

    return None


async def get_time(
    timezone: str = "Europe/Warsaw",
    target_date: str | None = None,
    kind: str = "all",
    lang: str = "pl",
) -> ToolResult:
    """Return local date/time or compute day of week and days for a target date."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Europe/Warsaw")
        timezone = "Europe/Warsaw"

    now = datetime.datetime.now(tz)
    today = now.date()

    parsed = parse_target_date(target_date, today) if target_date else None

    if parsed is not None:
        weekday_idx = parsed.weekday()
        days_until = (parsed - today).days

        if lang == "en":
            weekday_en = EN_WEEKDAYS[weekday_idx]
            month_en = EN_MONTHS[parsed.month]
            if days_until == 0:
                speak = f"{month_en} {parsed.day}, {parsed.year} is today, a {weekday_en}."
            elif days_until == 1:
                speak = f"{month_en} {parsed.day}, {parsed.year} is tomorrow, a {weekday_en}."
            elif days_until > 1:
                speak = f"{month_en} {parsed.day}, {parsed.year} is a {weekday_en}, in {days_until} days."
            else:
                speak = f"{month_en} {parsed.day}, {parsed.year} was a {weekday_en}, {abs(days_until)} days ago."
            weekday = weekday_en
        else:
            weekday_pl = PL_WEEKDAYS[weekday_idx]
            month_pl = PL_MONTHS[parsed.month]
            if days_until == 0:
                raw_speak = f"{parsed.day} {month_pl} to dzisiaj, {weekday_pl}."
            elif days_until == 1:
                raw_speak = f"{parsed.day} {month_pl} to jutro, {weekday_pl}."
            elif days_until > 1:
                days_word = polish_declension(days_until, "dzień", "dni", "dni")
                raw_speak = f"{parsed.day} {month_pl} {parsed.year} roku to {weekday_pl}, za {days_until} {days_word}."
            else:
                past_days = abs(days_until)
                days_word = polish_declension(past_days, "dzień", "dni", "dni")
                past_verb = "była" if weekday_pl in ("środa", "sobota", "niedziela") else "był"
                raw_speak = f"{parsed.day} {month_pl} {parsed.year} roku to {past_verb} {weekday_pl}, {past_days} {days_word} temu."
            weekday = weekday_pl
            speak = normalize_for_tts(raw_speak, lang)

        data = {
            "target_date": parsed.isoformat(),
            "weekday": weekday,
            "days_until": days_until,
            "year": parsed.year,
            "month": parsed.month,
            "day": parsed.day,
        }
        return ToolResult(data=data, speak=speak)

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    weekday_idx = now.weekday()
    month_idx = now.month

    if lang == "en":
        weekday_en = EN_WEEKDAYS[weekday_idx]
        month_en = EN_MONTHS[month_idx]
        if kind == "time":
            raw_speak = f"It is {time_str}."
        elif kind == "date":
            raw_speak = f"Today is {weekday_en}, {month_en} {now.day}."
        else:
            raw_speak = f"It is {weekday_en}, {month_en} {now.day}, {now.year}, {time_str}."
        weekday = weekday_en
    else:
        weekday_pl = PL_WEEKDAYS[weekday_idx]
        month_pl = PL_MONTHS[month_idx]
        if kind == "time":
            raw_speak = f"Jest {time_str}."
        elif kind == "date":
            raw_speak = f"Dziś jest {weekday_pl}, {now.day} {month_pl}."
        else:
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
    if kind in ("time", "date"):
        data["kind"] = kind
    return ToolResult(data=data, speak=speak)
