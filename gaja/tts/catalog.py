"""Selectable TTS voices as plain data, so nothing here imports an engine module (no model loads).

`id` is the engine name (the `hello.tts` value, `Person.tts` and `Household.general_voice`). The
`languages` sets mirror the `languages` frozensets declared by the engine classes in `gaja/tts/`;
tests/test_tts_catalog.py pins them together, and voices() checks at runtime which engines the
config actually builds (gaja/server.py TTS_ENGINES), so a voice the adapter cannot build is never
offered.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    id: str
    label: str
    languages: frozenset[str]


VOICES: tuple[Voice, ...] = (
    Voice("piper", "Piper", frozenset({"pl"})),
    Voice("piper-darkman", "Piper Darkman", frozenset({"pl"})),
    Voice("paradee", "Paradee", frozenset({"en"})),
    Voice("supertonic", "Supertonic", frozenset({"pl", "en"})),
)

VOICE_BY_ID: dict[str, Voice] = {v.id: v for v in VOICES}


def voice_languages(voice_id: str) -> frozenset[str] | None:
    """Languages a voice can speak; None = unknown id (no catalog entry: do not gate it here)."""
    voice = VOICE_BY_ID.get(voice_id)
    return voice.languages if voice is not None else None
