from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime
from piper import PiperVoice, SynthesisConfig

from gaja.text.style import Style
from gaja.tts.base import register_tts
from gaja.tts.prosody import apply_post, engine_params


class PiperTTS:
    WARM_UP_TEXT = "Dzień dobry."

    def __init__(self, name: str, model_path: Path, language: str = "pl", threads: int = 2) -> None:
        self.name = name
        self._voice = PiperVoice.load(model_path)
        self._voice.session = _capped_session(model_path, threads)
        self.sample_rate = self._voice.config.sample_rate
        self.languages = frozenset({language})
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._render, text, style)

    async def warm_up(self) -> None:
        await self.synthesize(self.WARM_UP_TEXT, Style(), next(iter(self.languages)))

    def _render(self, text: str, style: Style) -> np.ndarray:
        config = SynthesisConfig(**engine_params("piper", style))
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text, config)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return apply_post(audio, self.sample_rate, style)


def _capped_session(model_path: Path, threads: int) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


@register_tts("piper")
def piper_gosia(model_path: Path, threads: int = 2) -> PiperTTS:
    return PiperTTS("piper", model_path, threads=threads)


@register_tts("piper-darkman")
def piper_darkman(model_path: Path, threads: int = 2) -> PiperTTS:
    return PiperTTS("piper-darkman", model_path, threads=threads)
