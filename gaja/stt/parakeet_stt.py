from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from gaja.audio.vad import SAMPLE_RATE
from gaja.stt.base import Transcript

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "gaja-models"
    / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
)
# "Mówię po polsku." (Piper gosia, 16 kHz, 0.2 s trailing silence), see prefix_below_s.
PL_PREFIX_WAV = Path(__file__).resolve().parent / "assets" / "pl_prefix.wav"
_LEADING_PUNCT = " .,;:!?-–—…"


class ParakeetSTT:
    """NVIDIA Parakeet-TDT-0.6B-v3 (int8, sherpa-onnx) on the CPU, on one dedicated decode thread.

    Parakeet detects the language itself and cannot be told it. On a clip shorter than
    ``prefix_below_s`` it often picks English: the bare wake phrase "hej gaja" comes out as "Hey guy".
    So when Polish is requested, such clips get a short Polish phrase prepended. Tokens timed inside
    the prefix are then dropped from the result. Longer clips stay Polish without it.
    Hotwords use sherpa's contextual biasing (modified beam search) with one global ``hotwords_score``.
    """

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        num_threads: int = 4,
        hotwords_score: float = 3.0,
        max_active_paths: int = 4,
        prefix_below_s: float = 2.6,
    ) -> None:
        import sherpa_onnx

        d = Path(model_dir).expanduser()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=num_threads,
            model_type="nemo_transducer",
            decoding_method="modified_beam_search",
            max_active_paths=max_active_paths,
            hotwords_score=hotwords_score,
            modeling_unit="bpe",
            bpe_vocab=str(d / "bpe.vocab"),
        )
        prefix, rate = sf.read(PL_PREFIX_WAV, dtype="float32", always_2d=False)
        assert rate == SAMPLE_RATE, f"{PL_PREFIX_WAV}: expected {SAMPLE_RATE} Hz, got {rate}"
        self._prefix = prefix
        self._prefix_cut_s = self._calibrate_prefix_cut()
        self._prefix_below = int(prefix_below_s * SAMPLE_RATE)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet")

    def _calibrate_prefix_cut(self) -> float:
        """Midway between the prefix's last word token and its end.

        Token timestamps sit on 80 ms frames and the user's first token can land a frame before the
        prefix ends ("Odpal" at 1.20 s after a 1.31 s prefix), so a fixed cut near the end loses it.
        """
        end = len(self._prefix) / SAMPLE_RATE
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, self._prefix)
        self._recognizer.decode_stream(stream)
        words = [ts for tok, ts in zip(stream.result.tokens, stream.result.timestamps) if tok.strip(_LEADING_PUNCT)]
        return (words[-1] + end) / 2 if words else end - 0.15

    async def warm_up(self, seconds: float = 1.0) -> None:
        silence = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        await asyncio.get_running_loop().run_in_executor(self._executor, self._decode, silence, "pl", None)

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

        def _run() -> str:
            nonlocal decode_started
            decode_started = time.perf_counter()
            return self._decode(pcm.astype(np.float32, copy=False), language, hotwords)

        # One decode is a single native call that cannot be interrupted; a cancelled turn just drops the result.
        text = await asyncio.get_running_loop().run_in_executor(self._executor, _run)
        wait_ms = (decode_started - started) * 1000 if decode_started is not None else 0.0
        return Transcript(
            text=text,
            language=language or "",
            duration_ms=(time.perf_counter() - started) * 1000,
            audio_s=len(pcm) / SAMPLE_RATE,
            wait_ms=wait_ms,
        )

    def _decode(self, audio: np.ndarray, language: str | None, hotwords: str | None) -> str:
        prefixed = language == "pl" and 0 < len(audio) < self._prefix_below
        if prefixed:
            audio = np.concatenate([self._prefix, audio])
        stream = self._create_stream(sherpa_hotwords(hotwords or ""))
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        if not prefixed:
            return result.text.strip()
        return strip_prefix(result.tokens, result.timestamps, self._prefix_cut_s)

    def _create_stream(self, hotwords: str) -> Any:
        if not hotwords:
            return self._recognizer.create_stream()
        try:
            return self._recognizer.create_stream(hotwords)
        except Exception as exc:  # a term the BPE encoder rejects must not cost the whole utterance
            log.warning("Parakeet hotwords rejected, decoding without them: %s", exc)
            return self._recognizer.create_stream()


def strip_prefix(tokens: list[str], timestamps: list[float], cut_s: float) -> str:
    """Joins the tokens that start after ``cut_s`` (sentencepiece pieces carry their own spaces)."""
    text = "".join(tok for tok, ts in zip(tokens, timestamps) if ts >= cut_s)
    return text.strip().lstrip(_LEADING_PUNCT).strip()


@lru_cache(maxsize=32)
def sherpa_hotwords(hotwords: str) -> str:
    """faster-whisper style "Gaja, Kubernetes" -> sherpa "Gaja/gaja/Kubernetes/kubernetes".

    The BPE pieces are case-sensitive, so each term also goes in lower-cased. "/" separates entries
    and a ":" token would be read as a per-entry score, so both are dropped from the terms.
    """
    entries: list[str] = []
    for raw in hotwords.split(","):
        term = " ".join(w for w in raw.replace("/", " ").split() if not w.startswith(":"))
        for variant in (term, term.lower()):
            if variant and variant not in entries:
                entries.append(variant)
    return "/".join(entries)
