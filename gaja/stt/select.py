from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import numpy as np

from gaja.stt.base import STT, Transcript
from gaja.stt.whistle import WhistleSTT

log = logging.getLogger(__name__)


class SwitchingSTT:
    """Preferred STT engine (Parakeet on the CPU or faster-whisper on the GPU) with Whistle (CPU) as the fallback.

    A failed inference is retried once on the preferred engine (CUDA OOM while another process
    spikes is usually transient). Only when the retry also fails is the utterance transcribed by
    Whistle and the preferred engine put on a cooldown (``cooldown_s``, doubling up to
    ``max_cooldown_s``). A background probe re-tries the preferred engine (decode of silence, or a
    fresh load if it never loaded) each time the cooldown ends; on success the switch reverts.
    ``degraded`` is True while utterances go to Whistle although another engine was requested.
    """

    def __init__(
        self,
        requested: str,
        primary_factory: Callable[[], STT] | None = None,
        *,
        cooldown_s: float = 60.0,
        max_cooldown_s: float = 600.0,
        retry_delay_s: float = 0.3,
    ) -> None:
        self.requested = requested
        self.fallback_reason: str | None = None
        self.failures = 0
        self._primary_factory = primary_factory
        self._primary: STT | None = None
        self._fallback: STT | None = None
        self._fallback_lock = asyncio.Lock()
        self._cooldown_s = cooldown_s
        self._max_cooldown_s = max_cooldown_s
        self._retry_delay_s = retry_delay_s
        self._degraded_until = 0.0
        self._degraded = False
        self._probe: asyncio.Task[None] | None = None

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def engine(self) -> str:
        return self.requested if self._primary is not None and not self._degraded else "whistle"

    @property
    def device(self) -> str:
        return "cuda" if self.engine == "faster-whisper" else "cpu"

    async def load(self) -> None:
        if self._primary_factory is not None:
            try:
                primary = await asyncio.to_thread(self._primary_factory)
                await primary.warm_up()
                self._primary = primary
            except Exception as exc:
                self._degrade(exc, "load")
        if self._primary is None:
            await self._whistle()

    async def transcribe(
        self,
        pcm: np.ndarray,
        language: str | None = None,
        *,
        hotwords: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        kwargs: dict[str, Any] = {}
        if hotwords is not None:
            kwargs["hotwords"] = hotwords
        if prompt is not None:
            kwargs["prompt"] = prompt

        if self._primary is not None and not self._degraded:
            for attempt in (1, 2):
                try:
                    try:
                        return await self._primary.transcribe(pcm, language, **kwargs)
                    except TypeError:
                        return await self._primary.transcribe(pcm, language)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt == 2:
                        self._degrade(exc, "inference")
                        break
                    log.warning("STT %s failed, retrying once: %s", self.requested, exc)
                    await asyncio.sleep(self._retry_delay_s)
        whistle_engine = await self._whistle()
        try:
            return await whistle_engine.transcribe(pcm, language, **kwargs)
        except TypeError:
            return await whistle_engine.transcribe(pcm, language)

    async def aclose(self) -> None:
        if self._probe is not None:
            self._probe.cancel()
            await asyncio.gather(self._probe, return_exceptions=True)

    def _degrade(self, exc: BaseException, stage: str) -> None:
        self.fallback_reason = f"{stage}: {type(exc).__name__}: {exc}"
        self.failures += 1
        self._degraded = self._primary_factory is not None
        delay = min(self._max_cooldown_s, self._cooldown_s * 2 ** (self.failures - 1))
        self._degraded_until = time.monotonic() + delay
        log.error(
            "STT engine %s failed at %s, using whistle for %.0fs: %s", self.requested, stage, delay, exc, exc_info=exc
        )
        if self._degraded and (self._probe is None or self._probe.done()):
            try:
                self._probe = asyncio.get_running_loop().create_task(self._probe_loop())
            except RuntimeError:
                self._probe = None

    async def _probe_loop(self) -> None:
        while self._degraded:
            await asyncio.sleep(max(0.0, self._degraded_until - time.monotonic()))
            try:
                if self._primary is None:
                    assert self._primary_factory is not None
                    primary = await asyncio.to_thread(self._primary_factory)
                    await primary.warm_up()
                    self._primary = primary
                else:
                    await self._primary.warm_up()
            except Exception as exc:
                self.failures += 1
                delay = min(self._max_cooldown_s, self._cooldown_s * 2 ** (self.failures - 1))
                self._degraded_until = time.monotonic() + delay
                log.warning("STT %s still failing, next probe in %.0fs: %s", self.requested, delay, exc)
                continue
            log.warning("STT %s recovered after %d failure(s)", self.requested, self.failures)
            self._degraded = False
            self.failures = 0
            self.fallback_reason = None

    async def _whistle(self) -> STT:
        async with self._fallback_lock:
            if self._fallback is None:
                engine = await asyncio.to_thread(WhistleSTT)
                await engine.warm_up()
                self._fallback = engine
        return self._fallback
