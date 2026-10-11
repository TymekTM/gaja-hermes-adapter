from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gaja.text.style import Style

PROBE = Path(__file__).resolve().parent.parent / "bench" / "tts_probe_sentences_pl.txt"


class _FakeSupertonic:
    instances: list["_FakeSupertonic"] = []

    def __init__(self, auto_download, intra_op_num_threads, inter_op_num_threads):
        self.sample_rate = 44100
        self.threads = intra_op_num_threads
        self.calls: list[dict] = []
        _FakeSupertonic.instances.append(self)

    def get_voice_style(self, name):
        return f"style:{name}"

    def synthesize(self, text, voice_style, total_steps, speed, lang):
        self.calls.append({"text": text, "steps": total_steps, "speed": speed, "lang": lang, "voice": voice_style})
        return np.full((1, self.sample_rate), 0.1, dtype=np.float32), 1.0


@pytest.fixture
def fake_supertonic(monkeypatch):
    _FakeSupertonic.instances = []
    module = types.ModuleType("supertonic")
    module.TTS = _FakeSupertonic
    monkeypatch.setitem(sys.modules, "supertonic", module)
    return _FakeSupertonic


def _make(**kwargs):
    from gaja.tts.supertonic_tts import SupertonicTTS

    return SupertonicTTS(**kwargs)


def test_threads_and_steps_reach_the_engine(fake_supertonic):
    engine = _make(steps=6, threads=2, voice="F1")
    tts = fake_supertonic.instances[0]
    assert tts.threads == 2
    asyncio.run(engine.synthesize("Dzień dobry.", Style(), "pl"))
    assert tts.calls[0]["steps"] == 6
    assert tts.calls[0]["voice"] == "style:F1"


def test_speed_is_clamped_to_supported_range(fake_supertonic):
    fast = _make(speed=2.0)
    asyncio.run(fast.synthesize("Szybko.", Style(emotion="excited", rate=1.12), "pl"))
    assert fake_supertonic.instances[0].calls[-1]["speed"] == pytest.approx(2.0)

    slow = _make(speed=0.5)
    asyncio.run(slow.synthesize("Wolno.", Style(emotion="sad", rate=0.85), "pl"))
    assert fake_supertonic.instances[1].calls[-1]["speed"] == pytest.approx(0.7)


def test_style_rate_scales_speed_within_range(fake_supertonic):
    engine = _make(speed=1.0)
    asyncio.run(engine.synthesize("Tekst.", Style(rate=1.2), "pl"))
    assert fake_supertonic.instances[0].calls[-1]["speed"] == pytest.approx(1.2)


def test_warm_up_uses_polish_regardless_of_language_set(fake_supertonic):
    engine = _make()
    asyncio.run(engine.warm_up())
    assert fake_supertonic.instances[0].calls[-1]["lang"] == "pl"
    assert fake_supertonic.instances[0].calls[-1]["text"] == engine.WARM_UP_TEXT


def test_default_output_rate_is_22k_and_resamples_native_44k(fake_supertonic):
    engine = _make()
    assert engine.sample_rate == 22050
    audio = asyncio.run(engine.synthesize("Tekst.", Style(), "pl"))
    assert audio.dtype == np.float32
    assert abs(len(audio) - 22050) <= 2


def test_native_rate_when_output_rate_is_none(fake_supertonic):
    engine = _make(output_sample_rate=None)
    assert engine.sample_rate == 44100
    audio = asyncio.run(engine.synthesize("Tekst.", Style(), "pl"))
    assert len(audio) == 44100


def test_probe_sentences_have_short_and_medium_lengths():
    lines = [line.strip() for line in PROBE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 12
    counts = [len(line.split()) for line in lines]
    assert counts[:6] == [6] * 6
    assert all(11 <= n <= 13 for n in counts[6:])
