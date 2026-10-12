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


def _parse_time_str(time_str: str) -> tuple[int, int] | None:
    """Parse time string like 18:30, 18.30 or 18:00."""
    m = re.search(r"(\d{1,2})[:.](\d{2})", time_str)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
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
        cancelled = manager.cancel_timers(label=label)
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

        if lang == "en":
            raw_speak = f"{t_label} has {mins} minutes and {secs} seconds remaining."
        else:
            m_word = polish_declension(mins, "minuta", "minuty", "minut")
            s_word = polish_declension(secs, "sekunda", "sekundy", "sekund")
            if mins > 0:
                raw_speak = f"Na minutniku {t_label} zostało {mins} {m_word} i {secs} {s_word}."
            else:
                raw_speak = f"Na minutniku {t_label} zostało {secs} {s_word}."

        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data={"timers": statuses}, speak=speak)

    ctx = current_session_context.get()
    s_id = ctx.get("session_id") if ctx else None
    d_id = ctx.get("device_id") if ctx else None

    # Action is "set"
    parsed_time = _parse_time_str(time) if time else None

    # Check if time was passed in label (e.g. "przypomnij o 18:30")
    if not parsed_time and label:
        parsed_time = _parse_time_str(label)

    if parsed_time is not None or k == "reminder":
        k = "reminder"
        if parsed_time:
            h, mn = parsed_time
            due_dt = now.replace(hour=h, minute=mn, second=0, microsecond=0)
            if due_dt <= now:
                due_dt += datetime.timedelta(days=1)
        else:
            due_dt = now + datetime.timedelta(minutes=30)
            h, mn = due_dt.hour, due_dt.minute

        clean_label = label or "przypomnienie"
        # Clean label of time if included
        clean_label = re.sub(r"\b(?:o\s+)?\d{1,2}[:.]\d{2}\b", "", clean_label).strip() or "przypomnienie"

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

    due_dt = now + datetime.timedelta(minutes=dur_min)
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
