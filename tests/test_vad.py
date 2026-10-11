from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaja.audio.vad import SAMPLE_RATE, Endpointer, load_silero_prob
from gaja.config import VADConfig, resolve

CLIP = Path("bench/out/input_pl_16k.wav")


def energy_prob(window: np.ndarray) -> float:
    return 1.0 if float(np.abs(window).max()) > 0.05 else 0.0


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def feed_in_frames(endpointer: Endpointer, samples: np.ndarray, frame: int = 512) -> list:
    events = []
    for start in range(0, len(samples), frame):
        events.extend(endpointer.feed(samples[start : start + frame]))
    return events


def test_speech_then_silence_yields_start_and_end_with_audio() -> None:
    samples = np.concatenate([np.zeros(SAMPLE_RATE // 2, np.float32), tone(1.0), np.zeros(SAMPLE_RATE, np.float32)])
    events = feed_in_frames(Endpointer(energy_prob, VADConfig()), samples)
    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    assert events[1].audio is not None
    assert SAMPLE_RATE * 1.0 <= len(events[1].audio) <= SAMPLE_RATE * 1.9


def test_pre_roll_is_included_before_speech_start() -> None:
    samples = np.concatenate([tone(0.0), np.zeros(SAMPLE_RATE // 2, np.float32), tone(0.6), np.zeros(SAMPLE_RATE, np.float32)])
    events = feed_in_frames(Endpointer(energy_prob, VADConfig()), samples)
    audio = events[-1].audio
    assert audio is not None
    assert len(audio) >= int(0.6 * SAMPLE_RATE) + int(0.3 * SAMPLE_RATE) * 0.9


def test_too_short_blip_is_dropped_without_any_event() -> None:
    samples = np.concatenate([tone(0.05), np.zeros(SAMPLE_RATE, np.float32)])
    events = feed_in_frames(Endpointer(energy_prob, VADConfig()), samples)
    assert events == []


def test_single_loud_window_does_not_barge_in() -> None:
    samples = np.concatenate([tone(0.032), np.zeros(SAMPLE_RATE // 2, np.float32), tone(0.15), np.zeros(SAMPLE_RATE, np.float32)])
    assert feed_in_frames(Endpointer(energy_prob, VADConfig()), samples) == []


def test_speech_start_waits_for_barge_in_ms_of_speech() -> None:
    endpointer = Endpointer(energy_prob, VADConfig(barge_in_ms=250))
    assert feed_in_frames(endpointer, tone(0.2)) == []
    events = feed_in_frames(endpointer, tone(0.1))
    assert [e.kind for e in events] == ["speech_start"]
    assert endpointer.speaking


def test_strict_mode_needs_longer_and_louder_speech() -> None:
    cfg = VADConfig(barge_in_ms=250, barge_in_strict_ms=400, strict_threshold=0.7)
    probs = iter([0.6] * 20)
    endpointer = Endpointer(lambda _w: next(probs), cfg)
    endpointer.set_strict(True)
    # 0.6 is speech for the normal threshold but below the echo guard's 0.7: no barge-in.
    assert feed_in_frames(endpointer, tone(0.6)) == []
    strict = Endpointer(energy_prob, cfg)
    strict.set_strict(True)
    assert feed_in_frames(strict, tone(0.3)) == []
    assert [e.kind for e in feed_in_frames(strict, tone(0.15))] == ["speech_start"]


def test_flush_of_unconfirmed_segment_is_silent() -> None:
    endpointer = Endpointer(energy_prob, VADConfig())
    feed_in_frames(endpointer, tone(0.1))
    assert endpointer.flush() == []


def test_max_utterance_forces_end() -> None:
    cfg = VADConfig(max_utterance_s=2.0)
    events = feed_in_frames(Endpointer(energy_prob, cfg), tone(3.0))
    assert [e.kind for e in events[:2]] == ["speech_start", "speech_end"]
    assert events[1].audio is not None
    assert len(events[1].audio) >= SAMPLE_RATE * 1.9


def test_speech_continuing_past_max_utterance_does_not_barge_in_again() -> None:
    cfg = VADConfig(max_utterance_s=2.0)
    samples = np.concatenate([tone(3.0), np.zeros(SAMPLE_RATE, np.float32)])
    events = feed_in_frames(Endpointer(energy_prob, cfg), samples)
    assert [e.kind for e in events] == ["speech_start", "speech_end", "speech_end"]
    assert all(e.audio is not None for e in events[1:])


def test_speculation_fires_at_short_silence_before_real_end() -> None:
    cfg = VADConfig(spec_silence_ms=200, min_silence_ms=500)
    samples = np.concatenate([tone(1.0), np.zeros(SAMPLE_RATE, np.float32)])
    events = feed_in_frames(Endpointer(energy_prob, cfg), samples)
    assert [e.kind for e in events] == ["speech_start", "speculate", "speech_end"]
    assert events[1].audio is not None and events[2].audio is not None
    assert SAMPLE_RATE * 1.15 <= len(events[1].audio) <= SAMPLE_RATE * 1.3
    assert len(events[2].audio) > len(events[1].audio)


def test_resumed_speech_cancels_speculation_and_can_speculate_again() -> None:
    cfg = VADConfig(spec_silence_ms=200, min_silence_ms=500)
    samples = np.concatenate(
        [tone(0.5), np.zeros(int(0.3 * SAMPLE_RATE), np.float32), tone(0.5), np.zeros(SAMPLE_RATE, np.float32)]
    )
    events = feed_in_frames(Endpointer(energy_prob, cfg), samples)
    assert [e.kind for e in events] == [
        "speech_start",
        "speculate",
        "speculation_cancel",
        "speculate",
        "speech_end",
    ]


def test_speculation_disabled_when_not_before_real_end() -> None:
    samples = np.concatenate([tone(1.0), np.zeros(SAMPLE_RATE, np.float32)])
    for cfg in (VADConfig(), VADConfig(spec_silence_ms=500, min_silence_ms=500)):
        events = feed_in_frames(Endpointer(energy_prob, cfg), samples)
        assert [e.kind for e in events] == ["speech_start", "speech_end"]


def test_flush_ends_open_utterance() -> None:
    endpointer = Endpointer(energy_prob, VADConfig())
    events = feed_in_frames(endpointer, tone(0.8))
    flushed = endpointer.flush()
    assert [e.kind for e in events] == ["speech_start"]
    assert [e.kind for e in flushed] == ["speech_end"]


@pytest.mark.skipif(not resolve(CLIP).exists(), reason="reference clip not generated")
def test_silero_detects_speech_in_reference_clip() -> None:
    import soundfile as sf

    samples, rate = sf.read(resolve(CLIP), dtype="float32")
    assert rate == SAMPLE_RATE
    endpointer = Endpointer(load_silero_prob(), VADConfig())
    events = feed_in_frames(endpointer, samples) + endpointer.flush()
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["speech_start", "speech_end"]
    assert any(e.audio is not None and len(e.audio) > SAMPLE_RATE // 2 for e in events if e.kind == "speech_end")


def test_reset_forgets_buffered_speech_and_partial_windows() -> None:
    ep = Endpointer(lambda w: 1.0 if float(np.abs(w).max()) > 0.05 else 0.0, VADConfig())
    ep.feed(np.full(700, 0.5, dtype=np.float32))  # one full window + a partial one
    assert ep.in_speech
    ep.reset()
    assert not ep.in_speech and not ep.speaking
    events = ep.feed(np.zeros(16000, dtype=np.float32))
    assert events == []
