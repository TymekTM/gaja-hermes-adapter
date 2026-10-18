import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from gaja.stt.parakeet_stt import DEFAULT_MODEL_DIR, ParakeetSTT, sherpa_hotwords, strip_prefix

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
PREFIX_LEVEL = 0.1  # the fake prefix; user audio in these tests is 0.2


class FakeStream:
    def __init__(self, hotwords: str | None) -> None:
        self.hotwords = hotwords
        self.audio = np.zeros(0, dtype=np.float32)
        self.result = SimpleNamespace(text="", tokens=[], timestamps=[])

    def accept_waveform(self, rate: int, audio: np.ndarray) -> None:
        assert rate == SR
        self.audio = audio


class FakeRecognizer:
    """Hears the prefix as "Mówię po polsku." and the user's audio as " Hej Gaja.", timed by sample offset."""

    def __init__(self, prefix_len: int, *, reject_hotwords: bool = False) -> None:
        self.prefix_len = prefix_len
        self.reject_hotwords = reject_hotwords
        self.streams: list[FakeStream] = []

    def create_stream(self, hotwords: str | None = None) -> FakeStream:
        if hotwords is not None and self.reject_hotwords:
            raise RuntimeError("cannot encode hotwords")
        self.streams.append(FakeStream(hotwords))
        return self.streams[-1]

    def decode_stream(self, stream: FakeStream) -> None:
        has_prefix = bool(stream.audio.size) and stream.audio[0] == np.float32(PREFIX_LEVEL)
        user_at = self.prefix_len / SR if has_prefix else 0.0
        tokens = ([" Mó", "wię", " po", " polsku", "."] if has_prefix else []) + [" Hej", " Ga", "ja", "."]
        stamps = ([0.0, 0.2, 0.5, 0.7, 1.0] if has_prefix else []) + [user_at + t for t in (0.1, 0.4, 0.5, 0.7)]
        stream.result = SimpleNamespace(text="".join(tokens), tokens=tokens, timestamps=stamps)


def _engine(**kw) -> tuple[ParakeetSTT, FakeRecognizer]:
    stt = object.__new__(ParakeetSTT)  # skips loading sherpa-onnx and the model
    stt._prefix = np.full(int(1.3 * SR), PREFIX_LEVEL, dtype=np.float32)
    stt._prefix_cut_s = len(stt._prefix) / SR - 0.15
    stt._prefix_below = int(kw.get("prefix_below_s", 2.6) * SR)
    stt._executor = ThreadPoolExecutor(max_workers=1)
    rec = FakeRecognizer(len(stt._prefix), reject_hotwords=kw.get("reject_hotwords", False))
    stt._recognizer = rec
    return stt, rec


def test_hotwords_become_sherpa_entries_with_lowercase_variants() -> None:
    assert sherpa_hotwords("Hej Gaja, Gajo") == "Hej Gaja/hej gaja/Gajo/gajo"
    assert sherpa_hotwords("git, Docker, Docker") == "git/Docker/docker"


def test_hotwords_drop_separators_and_score_tokens() -> None:
    assert sherpa_hotwords("CI/CD, foo :5, , ") == "CI CD/ci cd/foo"
    assert sherpa_hotwords("") == ""


def test_strip_prefix_keeps_only_tokens_after_the_cut() -> None:
    tokens = [" Mó", "wię", " po", " polsku", ".", " Hej", ",", " Gajo", "."]
    stamps = [0.0, 0.2, 0.5, 0.7, 1.15, 1.3, 1.5, 1.6, 1.9]
    assert strip_prefix(tokens, stamps, 1.2) == "Hej, Gajo."
    assert strip_prefix([".", " Hej"], [1.25, 1.3], 1.2) == "Hej"


def test_short_polish_clip_is_decoded_behind_the_prefix() -> None:
    stt, rec = _engine()
    result = asyncio.run(stt.transcribe(np.full(2 * SR, 0.2, dtype=np.float32), "pl", hotwords="Hej Gaja"))
    assert result.text == "Hej Gaja."
    assert result.language == "pl"
    assert len(rec.streams[0].audio) == len(stt._prefix) + 2 * SR
    assert rec.streams[0].hotwords == "Hej Gaja/hej gaja"


def test_long_or_non_polish_clip_is_decoded_as_is() -> None:
    stt, rec = _engine()
    asyncio.run(stt.transcribe(np.full(3 * SR, 0.2, dtype=np.float32), "pl"))
    asyncio.run(stt.transcribe(np.full(2 * SR, 0.2, dtype=np.float32), "en"))
    asyncio.run(stt.transcribe(np.full(2 * SR, 0.2, dtype=np.float32), None))
    assert [len(s.audio) for s in rec.streams] == [3 * SR, 2 * SR, 2 * SR]
    assert [s.hotwords for s in rec.streams] == [None, None, None]


def test_rejected_hotwords_fall_back_to_a_plain_stream() -> None:
    stt, rec = _engine(reject_hotwords=True)
    result = asyncio.run(stt.transcribe(np.full(3 * SR, 0.2, dtype=np.float32), "pl", hotwords="Docker"))
    assert result.text == "Hej Gaja."
    assert rec.streams[0].hotwords is None


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "encoder.int8.onnx").exists(), reason="Parakeet model not downloaded")
def test_real_model_keeps_the_first_word_behind_the_prefix() -> None:
    clip = ROOT / "bench" / "data" / "stt" / "mixed" / "wer" / "mixed_000_gosia.wav"
    if not clip.exists():
        pytest.skip("mixed STT corpus not generated (bench/stt_prepare_mixed.py)")
    stt = ParakeetSTT()
    asyncio.run(stt.warm_up())
    audio, _ = sf.read(clip, dtype="float32")
    # 2.2 s: decoded behind the Polish prefix, whose cut must not eat the first word.
    text = asyncio.run(stt.transcribe(audio, "pl", hotwords="Kubernetes, deploy")).text
    assert text.lower().startswith("odpal mi deploy na kubernetes")
