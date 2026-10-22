from __future__ import annotations

import asyncio

import numpy as np
import pytest

from gaja.tts.base import EnginePool


class CountingEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sample_rate = 22050
        self.languages = frozenset({"pl"})
        self.warmed = 0

    async def synthesize(self, text, style, language):
        return np.zeros(1, dtype=np.float32)

    async def warm_up(self) -> None:
        self.warmed += 1


def test_engine_is_built_and_warmed_once_on_first_use() -> None:
    built: list[CountingEngine] = []

    def factory() -> CountingEngine:
        engine = CountingEngine("piper")
        built.append(engine)
        return engine

    async def scenario() -> None:
        pool = EnginePool({"piper": factory, "other": lambda: CountingEngine("other")})
        assert pool.names() == ["piper", "other"]
        assert pool.loaded() == []
        first, second = await asyncio.gather(pool.get("piper"), pool.get("piper"))
        assert first is second
        assert await pool.get("piper") is first
        assert pool.loaded() == ["piper"]
        assert "other" in pool and "missing" not in pool

    asyncio.run(scenario())
    assert len(built) == 1
    assert built[0].warmed == 1


def test_preload_engines_set() -> None:
    from gaja.server import _preload_engines

    by_lang = {"pl": "piper", "en": "paradee"}
    assert _preload_engines(by_lang, None) == ["paradee", "piper"]  # today's behavior: all defaults
    assert _preload_engines(by_lang, ["pl"]) == ["piper"]  # paradee loads on first en use
    assert _preload_engines(by_lang, []) == []  # everything lazy
    with pytest.raises(ValueError, match="without an engine"):
        _preload_engines(by_lang, ["de"])
