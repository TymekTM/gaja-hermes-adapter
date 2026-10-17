"""Quiet hours logic for voice-device speakers."""

from __future__ import annotations

import datetime
import re
from typing import Any

QUIET_HOURS_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


def validate_quiet_hours(v: str) -> str:
    """Validate and normalize 'HH:MM-HH:MM' quiet hours format. Empty string disables."""
    s = (v or "").strip()
    if not s:
        return ""
    m = QUIET_HOURS_RE.match(s)
    if not m:
        raise ValueError(
            f"quiet_hours must be formatted as 'HH:MM-HH:MM' (24-hour) or empty, got {v!r}"
        )
    h1, m1, h2, m2 = map(int, m.groups())
    if (h1, m1) == (h2, m2):
        raise ValueError("quiet_hours start and end time cannot be identical; use empty string to disable")
    return f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"


def parse_quiet_hours(quiet_hours: str) -> tuple[datetime.time, datetime.time] | None:
    """Parse a validated quiet hours window into (start_time, end_time) or None if disabled."""
    s = (quiet_hours or "").strip()
    if not s:
        return None
    m = QUIET_HOURS_RE.match(s)
    if not m:
        return None
    h1, m1, h2, m2 = map(int, m.groups())
    return datetime.time(h1, m1), datetime.time(h2, m2)


def is_quiet_hour(
    quiet_hours: str,
    now: datetime.time | datetime.datetime | None = None,
) -> bool:
    """Return True if ``now`` (or current local time of the host) falls inside ``quiet_hours``.

    Edges: start is inclusive, end is exclusive (e.g. 22:00 is quiet, 07:00 is not).
    Supports windows that wrap midnight (e.g. 22:00-07:00).
    Empty string disables quiet hours (always returns False).
    """
    parsed = parse_quiet_hours(quiet_hours)
    if parsed is None:
        return False
    start, end = parsed

    if now is None:
        t = datetime.datetime.now().time()
    elif isinstance(now, datetime.datetime):
        t = now.time()
    else:
        t = now

    if start < end:
        return start <= t < end
    else:
        return t >= start or t < end


def format_quiet_hours_error(raw_window: str) -> str:
    """Format a bilingual (PL/EN) error message for MCP tools when quiet hours block playback."""
    m = re.search(r"\(?(\d{2}:\d{2}-\d{2}:\d{2})\)?", str(raw_window))
    window = m.group(1) if m else str(raw_window).strip()
    return (
        f"Ciche godziny ({window}): głośnik jest wyciszony w nocy. "
        f"Jeśli użytkownik wyraźnie o to poprosił teraz, przekaż parametr force=true. / "
        f"Quiet hours active ({window}): speaker playback is muted at night. "
        f"If the user explicitly asked for this right now, pass force=true."
    )
