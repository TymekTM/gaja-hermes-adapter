from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gaja.text.style import Style
from gaja.tts.base import register_tts
from gaja.tts.prosody import apply_post


@register_tts("paradee")
class ParadeeTTS:
    name = "paradee"
    languages = frozenset({"en"})
    WARM_UP_TEXT = "Hello."

    def __init__(self, threads: int = 1) -> None:
        from paradee import SAMPLE_RATE, Paradee

        self.sample_rate = SAMPLE_RATE
        self._model = Paradee(threads=threads)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paradee")

    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._render, text, style)

    async def warm_up(self) -> None:
        await self.synthesize(self.WARM_UP_TEXT, Style(), next(iter(self.languages)))

    def _render(self, text: str, style: Style) -> np.ndarray:
        audio = np.asarray(self._model(text, speed=style.rate), dtype=np.float32).reshape(-1)
        return apply_post(audio, self.sample_rate, style)
