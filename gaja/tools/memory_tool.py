from __future__ import annotations

from typing import Any

from gaja.memory import MemoryStore
from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult

from gaja.memory import get_memory_store, set_memory_store


async def remember(fact: str, lang: str = "pl") -> ToolResult:
    """Store a personal fact about the user or their preferences in local memory."""
    store = get_memory_store()
    text = fact.strip()
    if not text:
        err = "Brak treści do zapamiętania"
        return ToolResult(data={"error": err}, speak="Nie podałeś, co mam zapamiętać.", error=err)

    created = store.add_fact(text)
    if lang == "en":
        raw_speak = f"I remembered: {text}."
    else:
        raw_speak = f"Zapamiętałam: {text}."

    speak = normalize_for_tts(raw_speak, lang)
    return ToolResult(data=created, speak=speak)


async def forget(fact: str, lang: str = "pl") -> ToolResult:
    """Remove a personal fact from local memory."""
    store = get_memory_store()
    text = fact.strip()
    if not text:
        err = "Brak treści do usunięcia"
        return ToolResult(data={"error": err}, speak="Nie podałeś, co mam zapomnieć.", error=err)

    deleted = store.forget_by_query(text)
    if deleted:
        if lang == "en":
            raw_speak = "I forgot that."
        else:
            raw_speak = "Zapomniałam o tym."
    else:
        if lang == "en":
            raw_speak = "I could not find that fact in memory."
        else:
            raw_speak = "Nie znalazłam takiego faktu w pamięci."

    speak = normalize_for_tts(raw_speak, lang)
    return ToolResult(data={"deleted": [f["id"] for f in deleted]}, speak=speak)
