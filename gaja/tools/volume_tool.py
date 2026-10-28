"""Voice control of the speaker volume: the tool behind "Hej Gaja, ciszej".

Reached deterministically through ``gaja.llm.router.volume_fast_path`` (no LLM round trip) and
selectable by the tool model for other phrasings. Confirms with a short spoken sentence at the
NEW volume: the store's ``on_change`` (wired in ``gaja/server.py``) pushes the gain into the
live device session before this tool's reply is spoken.
"""

from __future__ import annotations

from typing import Any

from gaja.devices.volume import STEP_FLOOR, VolumeStore, get_volume_store
from gaja.text.normalize import inflect_number_noun, normalize_for_tts
from gaja.tools.base import ToolResult, current_session_context


def _device_id() -> str | None:
    ctx: dict[str, Any] | None = current_session_context.get()
    device_id = ctx.get("device_id") if ctx else None
    return str(device_id) if device_id else None


def _speak_level(level: int, lang: str) -> str:
    if lang == "en":
        return f"Volume {level} percent."
    word = inflect_number_noun(level, "procent")
    return f"Głośność {word}."


async def set_speaker_volume(
    action: str = "set",
    level: int | None = None,
    delta: int | None = None,
    lang: str = "pl",
) -> ToolResult:
    """Set the speaker volume of the device the user is talking to.

    ``action``: set (absolute ``level`` 0-100), step (relative ``delta`` points, negative quieter),
    mute (level 0) or max (level 100).
    """
    store: VolumeStore | None = get_volume_store()
    if store is None:
        speak = "Sterowanie głośnością jest niedostępne." if lang == "pl" else "Volume control is unavailable."
        return ToolResult(data={"error": "volume_unavailable"}, speak=speak, error="volume_unavailable")
    device_id = _device_id()
    if device_id is None:
        speak = (
            "Głośność można zmienić tylko na głośniku Gaji." if lang == "pl" else "Volume can only be changed on a Gaja voice device."
        )
        return ToolResult(data={"error": "no_device"}, speak=speak)

    act = (action or "set").strip().lower()
    if act == "mute":
        level = 0
        act = "set"
    elif act == "max":
        level = 100
        act = "set"

    if act == "set":
        if level is None:
            speak = "Podaj poziom głośności od zera do stu." if lang == "pl" else "Say a volume level from 0 to 100."
            return ToolResult(data={"error": "missing_level"}, speak=speak, error="missing_level")
        new_level = store.set_level(device_id, level)
        if new_level <= 0:
            speak = "Wyciszyłam." if lang == "pl" else "Muted."
        elif new_level >= 100:
            speak = "Głośność na maksimum." if lang == "pl" else "Maximum volume."
        else:
            speak = _speak_level(new_level, lang)
    elif act == "step":
        if delta is None:
            speak = "Powiedz tylko: ciszej albo głośniej." if lang == "pl" else "Just say: quieter or louder."
            return ToolResult(data={"error": "missing_delta"}, speak=speak, error="missing_delta")
        before = store.level(device_id)
        new_level = store.step_level(device_id, int(delta))
        if new_level == before and new_level <= STEP_FLOOR and delta < 0:
            speak = (
                f"Głośność jest już na minimum, {new_level} procent."
                if lang == "pl"
                else f"Volume is already at the minimum, {new_level} percent."
            )
        else:
            speak = _speak_level(new_level, lang)
    else:
        speak = f"Unknown volume action {act}." if lang == "en" else f"Nieznana akcja głośności: {act}."
        return ToolResult(data={"error": "unknown_action"}, speak=speak, error="unknown_action")

    return ToolResult(
        data={"device_id": device_id, "level": new_level, "gain": store.gain(device_id), "muted": new_level <= 0},
        speak=normalize_for_tts(speak, lang),
    )
