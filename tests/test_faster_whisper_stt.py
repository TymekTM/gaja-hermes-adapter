import asyncio
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CLIP_PL = ROOT / "bench" / "data" / "stt" / "pl" / "wer"
SR = 16000


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _model_cached(name: str) -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{name}"
    return cache.exists()


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="CUDA device not available")


@pytest.fixture(scope="module")
def stt():
    if not _model_cached("small"):
        pytest.skip("faster-whisper small not in local HF cache")
    from gaja.stt.faster_whisper_stt import FasterWhisperSTT

    try:
        engine = FasterWhisperSTT(model="small", device="cuda", compute_type="int8", beam_size=5)
        asyncio.run(engine.warm_up())
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            pytest.skip(f"CUDA memory is held by the live stack: {exc}")
        raise
    return engine


def _pl_clip() -> np.ndarray:
    clips = sorted(CLIP_PL.glob("*.wav"))
    if not clips:
        pytest.skip("bench/data/stt/pl/wer not present")
    audio, rate = sf.read(clips[0], dtype="float32")
    assert rate == SR
    return audio


def test_transcribe_polish_clip_returns_text_and_metadata(stt):
    audio = _pl_clip()
    result = asyncio.run(stt.transcribe(audio, "pl"))
    assert result.text.strip()
    assert result.language == "pl"
    assert result.audio_s == pytest.approx(len(audio) / SR)
    assert result.duration_ms > 0


def test_auto_language_detects_polish(stt):
    result = asyncio.run(stt.transcribe(_pl_clip(), None))
    assert result.language == "pl"


def test_concurrent_calls_are_serialized_and_all_return(stt):
    audio = _pl_clip()

    async def both():
        return await asyncio.gather(stt.transcribe(audio, "pl"), stt.transcribe(audio[: SR * 2], "pl"))

    full, short = asyncio.run(both())
    assert full.text and short.audio_s == pytest.approx(2.0)


def test_accepts_float64_input(stt):
    audio = _pl_clip().astype(np.float64)
    result = asyncio.run(stt.transcribe(audio, "pl"))
    assert result.text.strip()
