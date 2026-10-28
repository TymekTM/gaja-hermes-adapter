"""TTS voice catalog (gaja/tts/catalog.py) and the engine pool built from config."""

from __future__ import annotations

import pytest

from gaja.tts import paradee_tts, piper_tts, supertonic_tts  # noqa: F401  (imports, no model loads)
from gaja.tts.catalog import VOICES, VOICE_BY_ID, voice_languages


def test_catalog_covers_every_selectable_engine() -> None:
    # gaja/server.py builds the pool from TTS_ENGINES; every entry needs a catalog voice.
    from gaja.server import TTS_ENGINES

    assert VOICE_BY_ID.keys() == set(TTS_ENGINES)


def test_catalog_languages_match_the_engine_declarations() -> None:
    # The engine classes declare `languages`; the catalog mirrors them (no engine imports at runtime).
    # Piper gets its language from the constructor default ("pl" = the configured Polish model).
    piper_language = piper_tts.PiperTTS.__init__.__defaults__[0]
    assert voice_languages("piper") == frozenset({piper_language})
    assert voice_languages("piper-darkman") == frozenset({piper_language})
    assert voice_languages("paradee") == paradee_tts.ParadeeTTS.languages
    assert voice_languages("supertonic") == supertonic_tts.SupertonicTTS.languages
    assert voice_languages("nope") is None
    assert {v.id for v in VOICES} == {"piper", "piper-darkman", "paradee", "supertonic"}
