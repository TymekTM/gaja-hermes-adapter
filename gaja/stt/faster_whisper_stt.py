from __future__ import annotations

import asyncio
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from gaja.audio.vad import SAMPLE_RATE
from gaja.stt.base import Transcript

# Whisper's well-known hallucinations on silence/noise (subtitle credits), dropped as whole segments.
HALLUCINATIONS = re.compile(
    r"(napisy|tłumaczenie|transkrypcja)\s+(wykonał[a]?|stworzone|by|przez)|amara\.org|"
    r"dziękuję za (uwagę|obejrzenie)|subtitles by|thanks for watching|"
    r"^\W*(\.\.\.|…)?\W*$",
    re.IGNORECASE,
)
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0


class FasterWhisperSTT:
    """faster-whisper (CTranslate2) on one dedicated thread: CUDA contexts are per thread, and the model stays resident."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cuda",
        compute_type: str = "int8",
        beam_size: int = 1,
        condition_on_previous_text: bool = False,
        without_timestamps: bool = False,
        no_speech_threshold: float = 0.6,
        cpu_threads: int = 0,
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model, device=device, compute_type=compute_type, cpu_threads=cpu_threads, local_files_only=True
        )
        self._decode_kwargs = {
            "beam_size": beam_size,
            "vad_filter": False,
            "condition_on_previous_text": condition_on_previous_text,
            "without_timestamps": without_timestamps,
            "no_speech_threshold": no_speech_threshold,
        }
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faster-whisper")

    async def warm_up(self, seconds: float = 1.0) -> None:
        silence = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        await asyncio.get_running_loop().run_in_executor(self._executor, partial(self._decode, silence, "pl"))

    async def transcribe(
        self,
        pcm: np.ndarray,
        language: str | None = None,
        *,
        hotwords: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        started = time.perf_counter()
        cancel = threading.Event()
        call = partial(
            self._decode,
            pcm.astype(np.float32, copy=False),
            language,
            cancel,
            hotwords=hotwords,
            prompt=prompt,
        )
        future = asyncio.get_running_loop().run_in_executor(self._executor, call)
        try:
            text, detected = await asyncio.shield(future)
        except asyncio.CancelledError:
            # The decode thread stops after the current segment instead of blocking the next turn.
            cancel.set()
            raise
        return Transcript(
            text=text,
            language=detected,
            duration_ms=(time.perf_counter() - started) * 1000,
            audio_s=len(pcm) / SAMPLE_RATE,
        )

    def _decode(
        self,
        audio: np.ndarray,
        language: str | None,
        cancel: threading.Event | None = None,
        *,
        hotwords: str | None = None,
        prompt: str | None = None,
    ) -> tuple[str, str]:
        kwargs = dict(self._decode_kwargs)
        if hotwords is not None:
            kwargs["hotwords"] = hotwords
        if prompt is not None:
            kwargs["initial_prompt"] = prompt
        segments, info = self._model.transcribe(audio, language=language, **kwargs)
        kept: list[str] = []
        for segment in segments:
            if cancel is not None and cancel.is_set():
                break
            if keep_segment(segment.text, segment.no_speech_prob, segment.avg_logprob):
                kept.append(segment.text.strip())
        return " ".join(kept).strip(), info.language or ""


def keep_segment(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    if no_speech_prob > NO_SPEECH_PROB and avg_logprob < MIN_AVG_LOGPROB:
        return False
    return not HALLUCINATIONS.search(text.strip())
