from __future__ import annotations
import contextvars

import re
from dataclasses import dataclass
from typing import Any

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
