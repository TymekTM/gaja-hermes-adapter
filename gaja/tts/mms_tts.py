from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from gaja.text.style import Style
from gaja.tts.base import apply_energy, register_tts


@register_tts("mms")
class MMSTTS:
    name = "mms"
    languages = frozenset({"pl"})
    WARM_UP_TEXT = "Dzień dobry."

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._model = VitsModel.from_pretrained("facebook/mms-tts-pol")
        if self.device == "cuda":
            self._model = self._model.cuda()
        self._tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pol")
        self.sample_rate = self._model.config.sampling_rate
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mms")

    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._render, text, style
        )

    async def warm_up(self) -> None:
        await self.synthesize(self.WARM_UP_TEXT, Style(), next(iter(self.languages)))

    def _render(self, text: str, style: Style) -> np.ndarray:
        inputs = self._tokenizer(text, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, speaking_rate=style.rate).waveform
        if self.device == "cuda":
            torch.cuda.synchronize()
        audio = out.squeeze().float().cpu().numpy()
        return apply_energy(audio, style.energy)
