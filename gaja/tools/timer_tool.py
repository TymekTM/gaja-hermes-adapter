from __future__ import annotations

import datetime
import re
from zoneinfo import ZoneInfo
from typing import Any

from gaja.text.normalize import normalize_for_tts
from gaja.timers import TimerManager
from gaja.tools.base import ToolResult, polish_declension, current_session_context

WARSAW_TZ = ZoneInfo("Europe/Warsaw")

from gaja.timers import get_timer_manager, set_timer_manager


_HHMM_RE = re.compile(r"(\d{1,2})[:.](\d{2})")
# A word that turns a bare number into a time of day ("o 7 rano", "o 22 w nocy", "po 2 rano").
_TIME_OF_DAY_RE = (
    r"(?:rano|rankiem|po[lł]udni[ue]|popo[lł]udni[ue]|wiecz[oó]r\w*|noc\w*|dzi[sś]\w*|jutro"
    r"|pojutrze|codziennie)"
)
# Words that may legally follow a cued hour ("o 7 rano", "po 18 do mamy"); any other word
# right after the number ("po 15 minut", "o 2 tabletkach") makes the number a count, not an hour.
_TIME_QUALIFIER_RE = _TIME_OF_DAY_RE + r"|w|we|na|do|za|i|oraz"
_WORD_RE = r"[a-ząćęłńóśźż]+"
# The next word is a qualifier (or there is none).
_QUALIFIED_ENDING = r"(?!\s+(?!" + _TIME_QUALIFIER_RE + r"\b)" + _WORD_RE + r")"
# A following ",<digit>" is a decimal number ("po 1,5 tabletki"), never an hour.
_NOT_A_DECIMAL = r"(?!,\d)"
# "po <hour>" (not "o <hour>"). Hours 7-23 are unambiguous once no count noun follows
# ("po 18", "po 18 do mamy", "po 15 minut" is a duration). Hours 1-6 without :MM are dosage
# counts unless a time-of-day word follows: "po 2 rano", but "po 2 tabletki", "po 2 na raz",
# "po 2 w opakowaniu" ("2 na raz" = two at a time).
_PO_CUE_PATTERN = (
    r"\bpo\s+(?:\d{1,2}[:.]\d{2}" + _NOT_A_DECIMAL
    + r"|[1-6]" + _NOT_A_DECIMAL + r"(?=\s+" + _TIME_OF_DAY_RE + r"\b)"
    + r"|(?:[7-9]|1\d|2[0-3])" + _NOT_A_DECIMAL + _QUALIFIED_ENDING
    + r")"
)
_O_CUE_PATTERN = r"\b(?:o|godzinie|godz\.?)\s+\d{1,2}(?:[:.]\d{2})?" + _NOT_A_DECIMAL + _QUALIFIED_ENDING
_TIME_CUE_RE = re.compile(r"(?:" + _PO_CUE_PATTERN + r"|" + _O_CUE_PATTERN + r")", re.IGNORECASE)
_TIME_EXPRESSION_RE = re.compile(
    r"(?:" + _PO_CUE_PATTERN + r"|" + _O_CUE_PATTERN + r")"
    r"|\b\d{1,2}[:.]\d{2}\b",
    re.IGNORECASE,
)


def _parse_time_str(time_str: str, *, bare_ok: bool = False) -> tuple[int, int] | None:
    """Parse an explicit time: 18:30, 18.30, "o 18" or (for the tool's time arg) a bare "18"."""
    m = _HHMM_RE.search(time_str)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
    m_cue = _TIME_CUE_RE.search(time_str)
    if m_cue:
        # The cue pattern decides *whether* the number is a time; the digits inside the
        # matched span are then the hour (and minutes, when written with :MM or .MM).
        m_hm = _HHMM_RE.search(m_cue.group(0))
        if m_hm:
            h, mn = int(m_hm.group(1)), int(m_hm.group(2))
        else:
            h, mn = int(re.search(r"\d{1,2}", m_cue.group(0)).group(0)), 0
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
        return None
    if bare_ok:
        m_h = re.search(r"(\d{1,2})\b", time_str)
        if m_h:
            h = int(m_h.group(1))
            if 0 <= h <= 23:
                return h, 0
    return None


async def manage_timer(
    action: str = "set",
    duration_min: float | int | None = None,
    time: str | None = None,
    label: str | None = None,
    kind: str = "timer",
    lang: str = "pl",
) -> ToolResult:
    """Manage timers and reminders: set, check status, or cancel."""
    manager = get_timer_manager()
    now = datetime.datetime.now(WARSAW_TZ)
    act = (action or "set").lower().strip()
    k = (kind or "timer").lower().strip()

    if act in ("cancel", "stop", "anuluj", "usun", "usuń"):
        cancelled = manager.cancel_timers(label=label, kind=k)
        if cancelled:
            if lang == "en":
                raw_speak = "Timer cancelled." if k == "timer" else "Reminder cancelled."
            else:
                raw_speak = "Anulowałam minutnik." if k == "timer" else "Anulowałam przypomnienie."
        else:
            if lang == "en":
                raw_speak = "No active timers found."
            else:
                raw_speak = "Nie masz żadnego aktywnego minutnika do anulowania."
        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data={"cancelled": cancelled}, speak=speak)

    if act in ("status", "check", "ile", "sprawdź", "sprawdz"):
        statuses = manager.get_status()
        if not statuses:
            if lang == "en":
                raw_speak = "You have no active timers or reminders."
            else:
                raw_speak = "Nie masz żadnych aktywnych minutników ani przypomnień."
            speak = normalize_for_tts(raw_speak, lang)
            return ToolResult(data={"timers": []}, speak=speak)

        nearest = statuses[0]
        rem_s = nearest.get("remaining_seconds", 0)
        mins = rem_s // 60
        secs = rem_s % 60
        t_label = nearest.get("label", "minutnik")

        if nearest.get("overdue"):
            # Fired while nobody was connected; still pending and announced at the next hello.
            if lang == "en":
                raw_speak = f"{t_label} is overdue; it will be announced at the next connection."
            elif nearest.get("kind") == "reminder":
                raw_speak = f"Minęło przypomnienie {t_label}; ogłoszę je przy następnym połączeniu."
            else:
                raw_speak = f"Minął minutnik {t_label}; ogłoszę go przy następnym połączeniu."
        elif lang == "en":
            raw_speak = f"{t_label} has {mins} minutes and {secs} seconds remaining."
        else:
            m_word = polish_declension(mins, "minuta", "minuty", "minut")
            s_word = polish_declension(secs, "sekunda", "sekundy", "sekund")
            # The verb agrees with the first numeral: "została minuta", "zostały dwie", "zostało pięć".
            if mins > 0:
                left = polish_declension(mins, "została", "zostały", "zostało")
                raw_speak = f"Na minutniku {t_label} {left} {mins} {m_word} i {secs} {s_word}."
            else:
                left = polish_declension(secs, "została", "zostały", "zostało")
                raw_speak = f"Na minutniku {t_label} {left} {secs} {s_word}."

        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data={"timers": statuses}, speak=speak)

    ctx = current_session_context.get()
    s_id = ctx.get("session_id") if ctx else None
    d_id = ctx.get("device_id") if ctx else None

    # Action is "set"
    parsed_time = _parse_time_str(time, bare_ok=True) if time else None

    # Check if an explicit time was passed in the label (e.g. "przypomnij o 18:30").
    if not parsed_time and label:
        parsed_time = _parse_time_str(label)

    if parsed_time is not None or k == "reminder":
        k = "reminder"
        if parsed_time:
            h, mn = parsed_time
            due_dt = now.replace(hour=h, minute=mn, second=0, microsecond=0)
            if due_dt <= now:
                due_dt += datetime.timedelta(days=1)
        elif duration_min is not None and duration_min > 0:
            due_dt = (now.astimezone(datetime.timezone.utc) + datetime.timedelta(minutes=duration_min)).astimezone(
                WARSAW_TZ
            )
            h, mn = due_dt.hour, due_dt.minute
        else:
            due_dt = now + datetime.timedelta(minutes=30)
            h, mn = due_dt.hour, due_dt.minute

        clean_label = label or "przypomnienie"
        # Clean label of the explicit time expression that was parsed (collapse the leftover
        # double spaces so the spoken label does not pause mid-phrase).
        clean_label = " ".join(_TIME_EXPRESSION_RE.sub("", clean_label).split()) or "przypomnienie"

        created = manager.add_timer(clean_label, due_dt, kind="reminder", session_id=s_id, device_id=d_id)
        time_str = f"{h:02d}:{mn:02d}"

        if lang == "en":
            raw_speak = f"Reminder set: {clean_label} at {time_str}."
        else:
            raw_speak = f"Ustawiłam przypomnienie: {clean_label} o godzinie {time_str}."

        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data=created, speak=speak)

    # Timer with duration
    dur_min = duration_min
    if dur_min is None and label:
        m_dur = re.search(r"(\d+(?:[.,]\d+)?)\s*min", label)
        if m_dur:
            dur_min = float(m_dur.group(1).replace(",", "."))
        elif "pół godziny" in label or "pol godziny" in label:
            dur_min = 30.0
        elif "godzin" in label:
            dur_min = 60.0

    if dur_min is None or dur_min <= 0:
        dur_min = 10.0  # default 10 minutes

    int_min = int(dur_min)
    if label:
        clean_label = label.strip()
    else:
        m_word = polish_declension(int_min, "minuta", "minuty", "minut")
        clean_label = f"{int_min} {m_word}"

    due_dt = (now.astimezone(datetime.timezone.utc) + datetime.timedelta(minutes=dur_min)).astimezone(WARSAW_TZ)
    created = manager.add_timer(clean_label, due_dt, kind="timer", session_id=s_id, device_id=d_id)

    if lang == "en":
        m_word = "minute" if int_min == 1 else "minutes"
        raw_speak = f"Timer set for {int_min} {m_word}."
    else:
        if int_min == 1:
            raw_speak = "Ustawiłam minutnik na jedną minutę."
        else:
            m_word = polish_declension(int_min, "minuta", "minuty", "minut")
            raw_speak = f"Ustawiłam minutnik na {int_min} {m_word}."

    speak = normalize_for_tts(raw_speak, lang)
    return ToolResult(data=created, speak=speak)
