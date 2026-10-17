from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from gaja.audio.vad import SAMPLE_RATE
from gaja.stt.base import Transcript


class WhistleSTT:
    """Whistle (cactus-needle) on one dedicated thread: the library keeps global state and is not thread-safe."""

    def __init__(self) -> None:
        from needle import Whistle

        self._model = Whistle()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whistle")

    async def warm_up(self) -> None:
        await self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), "pl")

    async def transcribe(
        self,
        pcm: np.ndarray,
        language: str | None = None,
        *,
        hotwords: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        started = time.perf_counter()
        decode_started: float | None = None

        def _run() -> dict[str, Any]:
            nonlocal decode_started
            decode_started = time.perf_counter()
            return self._model.transcribe(pcm.astype(np.float32, copy=False), language=language)

        result = await asyncio.get_running_loop().run_in_executor(self._executor, _run)
        wait_ms = (decode_started - started) * 1000 if decode_started is not None else 0.0
        return Transcript(
            text=result["text"].strip(),
            language=result["language"] or "",
            duration_ms=(time.perf_counter() - started) * 1000,
            audio_s=len(pcm) / SAMPLE_RATE,
            wait_ms=wait_ms,
        )
