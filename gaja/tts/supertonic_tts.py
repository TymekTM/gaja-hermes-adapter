from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from gaja.text.style import Style
from gaja.tts.base import register_tts
from gaja.tts.prosody import apply_post, engine_params

SPEED_RANGE = (0.7, 2.0)


@register_tts("supertonic")
class SupertonicTTS:
    name = "supertonic"
    languages = frozenset({"pl", "en"})
    WARM_UP_TEXT = "Dzień dobry."
    WARM_UP_LANGUAGE = "pl"

    def __init__(
        self,
        steps: int = 6,
        voice: str = "M4",
        speed: float = 1.05,
        threads: int = 2,
        output_sample_rate: int | None = 22050,
    ) -> None:
        from supertonic import TTS

        self._tts = TTS(auto_download=False, intra_op_num_threads=threads, inter_op_num_threads=1)
        self._voice_style = self._tts.get_voice_style(voice)
        self._steps = steps
        self._base_speed = speed
        self._native_sr = self._tts.sample_rate
        self.sample_rate = output_sample_rate or self._native_sr
        self._resample = None if self.sample_rate == self._native_sr else Fraction(self.sample_rate, self._native_sr)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="supertonic")

    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._render, text, style, language
        )

    async def warm_up(self) -> None:
        await self.synthesize(self.WARM_UP_TEXT, Style(), self.WARM_UP_LANGUAGE)

    def _render(self, text: str, style: Style, language: str) -> np.ndarray:
        speed = self._base_speed * engine_params("supertonic", style)["speed"]
        speed = min(max(speed, SPEED_RANGE[0]), SPEED_RANGE[1])
        wav, _ = self._tts.synthesize(
            text,
            voice_style=self._voice_style,
            total_steps=self._steps,
            speed=speed,
            lang=language,
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        if self._resample is not None:
            audio = resample_poly(audio, self._resample.numerator, self._resample.denominator).astype(np.float32)
        return apply_post(audio, self.sample_rate, style)
