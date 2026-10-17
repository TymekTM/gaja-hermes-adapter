from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass
from typing import Any

from gaja.config import Settings
from gaja.text.normalize import (
    NOUN_DECLENSIONS,
    agree_noun,
    format_number_pl,
    inflect_number,
    inflect_number_noun,
    pl_cardinal,
    pl_cardinal_float,
    polish_declension,
)

# Values the tool selector copies from the tool signature instead of naming a real city.
PLACEHOLDER_LOCATIONS: frozenset[str] = frozenset(
    {
        "",
        "miasto",
        "city",
        "location",
        "nazwa miasta",
        "nazwa miejscowości",
        "miejscowość",
        "miejscowosc",
        "tutaj",
        "tu",
        "here",
        "domyślne",
        "domyslnie",
        "default",
        "none",
        "null",
    }
)


@dataclass
class ToolResult:
    data: dict[str, Any]
    speak: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        res = dict(self.data)
        if self.speak:
            res["speak"] = self.speak
        if self.error:
            res["error"] = self.error
        return res


def clean_speakable(text: str) -> str:
    """Remove special punctuation and markdown symbols so TTS reads smoothly."""
    text = re.sub(r"[*_#`~]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def inflect_noun(val: int | float, noun_key: str, case: str = "nom") -> str:
    """Helper alias for agree_noun."""
    return agree_noun(val, noun_key, case=case)


current_session_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "current_session_context", default=None
)


def get_default_location(cfg: Settings | None = None) -> str:
    """Return the configured default location from session context or Settings."""
    ctx = current_session_context.get()
    if ctx is not None:
        ctx_loc = str(ctx.get("default_location") or "").strip()
        if ctx_loc:
            return ctx_loc
    settings = cfg if cfg is not None else Settings()
    loc = str(settings.tools.default_location or "").strip()
    return loc or "Katowice"
