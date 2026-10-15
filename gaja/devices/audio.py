"""Device capabilities, output-rate negotiation and the resampling outbox."""

from __future__ import annotations

from fractions import Fraction
import asyncio
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.signal import resample_poly

from gaja.audio.pcm import float_to_pcm16, pcm16_to_float

MIN_RATE, MAX_RATE = 8000, 48000
MIC_RATE = 16000


class DeviceCaps(BaseModel):
    aec: bool = False
    out_rates: list[int] = Field(default_factory=lambda: [MIC_RATE], max_length=8)
    preroll_ms: int = Field(1500, ge=0, le=5000)
    # Protocol v1.10: the device can record its live mic on request (record_start/...; docs/DEVICE.md).
    record: bool = False

    @field_validator("out_rates")
    @classmethod
    def _rates(cls, rates: list[int]) -> list[int]:
        if not rates:
            raise ValueError("out_rates must not be empty")
        for rate in rates:
            if not MIN_RATE <= rate <= MAX_RATE:
                raise ValueError(f"unsupported output rate {rate}")
        return rates


def pick_out_rate(offered: list[int], preferred: int = MIC_RATE) -> int:
    """The device's rate nearest to what we like: preferred, else the lowest offered >= 16 kHz, else the highest."""
    if preferred in offered:
        return preferred
    above = sorted(r for r in offered if r >= MIC_RATE)
    return above[0] if above else max(offered)


def resample_pcm16(data: bytes, src: int, dst: int) -> bytes:
    if src == dst or not data:
        return data
    ratio = Fraction(dst, src)
    audio = resample_poly(pcm16_to_float(data), ratio.numerator, ratio.denominator)
    return float_to_pcm16(audio.astype(np.float32))


def beep_pcm16(rate: int, *, freqs: tuple[float, float] = (880.0, 1175.0), ms: int = 70, volume: float = 0.3) -> bytes:
    """Two short rising tones with a fade, the server-side 'earcon'."""
    half = int(rate * ms / 1000)
    t = np.arange(half) / rate
    fade = np.minimum(1.0, np.minimum(np.arange(half), np.arange(half)[::-1]) / max(1, int(rate * 0.005)))
    tones = [np.sin(2 * np.pi * f * t) * fade * volume for f in freqs]
    return float_to_pcm16(np.concatenate([tones[0], np.zeros(int(rate * 0.02)), tones[1]]).astype(np.float32))


class ResamplingOutbox:
    """Wraps a real outbox: every TTS utterance is resampled to the device's rate and cut into small frames.

    ``audio_start.sample_rate`` and ``ready.sample_rate`` are rewritten to the device rate, so the device only
    ever sees one rate. Until ``configure`` is called (before hello) messages pass through unchanged.
    """

    def __init__(self, inner: Any, chunk_ms: int = 100) -> None:
        self._inner = inner
        self._chunk_ms = chunk_ms
        self.rate: int | None = None
        self._src_rate: int | None = None

    def configure(self, rate: int) -> None:
        self.rate = rate

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.rate is not None and payload.get("type") in ("ready", "audio_start"):
            if payload["type"] == "audio_start":
                self._src_rate = int(payload["sample_rate"])
            payload = {**payload, "sample_rate": self.rate}
        await self._inner.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        if self.rate is None:
            await self._inner.send_bytes(data)
            return
        src_rate = self._src_rate or self.rate
        if len(data) > src_rate * 2 // 10:  # > 100 ms
            data = await asyncio.to_thread(resample_pcm16, data, src_rate, self.rate)
        else:
            data = resample_pcm16(data, src_rate, self.rate)
        step = max(2, self.rate * self._chunk_ms // 1000 * 2)
        for offset in range(0, len(data), step):
            await self._inner.send_bytes(data[offset : offset + step])
