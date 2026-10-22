from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaja.audio.vad import SAMPLE_RATE, Endpointer, load_silero, load_silero_prob_onnx
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
    endpointer = Endpointer(load_silero("onnx"), VADConfig())
    events = feed_in_frames(endpointer, samples) + endpointer.flush()
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["speech_start", "speech_end"]
    assert any(e.audio is not None and len(e.audio) > SAMPLE_RATE // 2 for e in events if e.kind == "speech_end")


def test_silero_onnx_prob_runs_without_torch() -> None:
    # A fresh interpreter (suite order must not matter): loading and running the onnx VAD must not
    # pull torch, which costs ~414 MB RSS in the adapter (bench/results/ram/import-torch.txt).
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; sys.path.insert(0, '.');"
        "from gaja.audio.vad import load_silero;"
        "import numpy as np;"
        "p = load_silero('onnx');"
        "[p(np.zeros(512, dtype=np.float32)) for _ in range(3)];"
        "sys.exit(0 if 'torch' not in sys.modules else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    assert result.returncode == 0, f"onnx VAD backend imported torch:\n{result.stderr}"


def test_silero_onnx_state_is_per_instance_and_deterministic() -> None:
    t = tone(512 / SAMPLE_RATE, 0.1)[:512]
    a, b = load_silero_prob_onnx(), load_silero_prob_onnx()
    first = a(t)
    assert b(t) == first  # fresh instances agree on the first window
    for _ in range(5):
        a(t)  # advance a's private RNN state
    c = load_silero_prob_onnx()
    assert c(t) == first  # a's history did not leak into a new instance
    assert a(t) != pytest.approx(first, abs=1e-6)  # a's own state advanced


@pytest.mark.skipif(not __import__("os").environ.get("GAJA_RAM_BENCH"), reason="loads torch (~500 MB); run explicitly")
def test_silero_onnx_matches_torch_wrapper() -> None:
    from gaja.audio.vad import load_silero_prob

    torch_prob = load_silero_prob()
    onnx_prob = load_silero_prob_onnx()
    max_diff = 0.0
    for amplitude in (0.0, 0.05, 0.25, 0.9):
        for offset in range(4):
            window = tone(512 / SAMPLE_RATE, amplitude)[:512] if offset % 2 else np.full(512, -amplitude, np.float32)
            max_diff = max(max_diff, abs(torch_prob(window) - onnx_prob(window)))
    assert max_diff == 0.0, f"onnx VAD diverged from the torch wrapper by {max_diff}"


def test_reset_forgets_buffered_speech_and_partial_windows() -> None:
    ep = Endpointer(lambda w: 1.0 if float(np.abs(w).max()) > 0.05 else 0.0, VADConfig())
    ep.feed(np.full(700, 0.5, dtype=np.float32))  # one full window + a partial one
    assert ep.in_speech
    ep.reset()
    assert not ep.in_speech and not ep.speaking
    events = ep.feed(np.zeros(16000, dtype=np.float32))
    assert events == []


# ---------------------------------------------------------------------------
# Smart Turn v3 Endpointer tests (Step 4)
# ---------------------------------------------------------------------------

def test_smart_turn_disabled_preserves_old_behaviour() -> None:
    # Model returns 0.9 (complete), but smart_turn=False in config: must wait full 500 ms
    cfg = VADConfig(smart_turn=False, smart_turn_check_ms=250, min_silence_ms=500)
    fake_model = lambda _audio: 0.9
    endpointer = Endpointer(energy_prob, cfg, smart_turn=fake_model)

    # 1.0s tone + 300 ms silence (past check_ms=250, but before min_silence=500)
    samples = np.concatenate([tone(1.0), np.zeros(int(0.3 * SAMPLE_RATE), np.float32)])
    events = feed_in_frames(endpointer, samples)
    # Speech has not ended yet because smart_turn is disabled and silence < 500 ms
    assert [e.kind for e in events] == ["speech_start"]

    # Another 300 ms silence -> total 600 ms silence >= 500 ms -> speech_end emitted
    more_events = feed_in_frames(endpointer, np.zeros(int(0.3 * SAMPLE_RATE), np.float32))
    assert [e.kind for e in more_events] == ["speech_end"]


def test_smart_turn_complete_triggers_early_commit() -> None:
    cfg = VADConfig(smart_turn=True, smart_turn_check_ms=250, min_silence_ms=500)
    fake_complete = lambda _audio: 0.9
    endpointer = Endpointer(energy_prob, cfg, smart_turn=fake_complete)

    # 1.0s tone + 280 ms silence (just past check_ms=250)
    samples = np.concatenate([tone(1.0), np.zeros(int(0.28 * SAMPLE_RATE), np.float32)])
    events = feed_in_frames(endpointer, samples)
    # With smart turn saying complete, speech_end triggers early at ~250 ms!
    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    assert events[1].audio is not None
    # Endpointer is reset after early commit
    assert not endpointer.speaking


def test_smart_turn_incomplete_extends_wait_capped_by_max() -> None:
    cfg = VADConfig(
        smart_turn=True,
        smart_turn_check_ms=250,
        min_silence_ms=500,
        max_incomplete_silence_ms=1000,
    )
    fake_incomplete = lambda _audio: 0.1
    endpointer = Endpointer(energy_prob, cfg, smart_turn=fake_incomplete)

    # 1.0s tone + 600 ms silence: normally (500 ms) this would commit, but incomplete extends wait
    samples = np.concatenate([tone(1.0), np.zeros(int(0.6 * SAMPLE_RATE), np.float32)])
    events = feed_in_frames(endpointer, samples)
    assert [e.kind for e in events] == ["speech_start"]

    # Feed up to 1100 ms total silence: now exceeds max_incomplete_silence_ms (1000 ms)
    more_silence = np.zeros(int(0.5 * SAMPLE_RATE), np.float32)
    more_events = feed_in_frames(endpointer, more_silence)
    assert [e.kind for e in more_events] == ["speech_end"]


def test_smart_turn_incomplete_bridges_hesitation_pause() -> None:
    cfg = VADConfig(
        smart_turn=True,
        smart_turn_check_ms=250,
        min_silence_ms=500,
        max_incomplete_silence_ms=1200,
    )
    # First check (during pause) returns 0.1 (incomplete), second check returns 0.9 (complete)
    scores = iter([0.1, 0.9])
    model = lambda _audio: next(scores)
    endpointer = Endpointer(energy_prob, cfg, smart_turn=model)

    # Speech (0.8s) -> 600 ms hesitation pause -> speech (0.8s) -> 300 ms silence
    phrase = np.concatenate([
        tone(0.8),
        np.zeros(int(0.6 * SAMPLE_RATE), np.float32),  # 600 ms pause (incomplete bridges this)
        tone(0.8),
        np.zeros(int(0.3 * SAMPLE_RATE), np.float32),  # 300 ms silence (complete commits this)
    ])
    events = feed_in_frames(endpointer, phrase)
    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    # Total audio captured includes both parts of the utterance
    audio = events[1].audio
    assert audio is not None
    assert len(audio) >= int(1.6 * SAMPLE_RATE)


def test_smart_turn_strict_and_barge_in_unaffected() -> None:
    cfg = VADConfig(
        smart_turn=True,
        barge_in_ms=250,
        barge_in_strict_ms=400,
        strict_threshold=0.7,
        smart_turn_check_ms=250,
    )
    fake_complete = lambda _audio: 0.9
    endpointer = Endpointer(energy_prob, cfg, smart_turn=fake_complete)
    endpointer.set_strict(True)

    # Speech below strict threshold: no barge-in
    soft_tone = tone(0.6, amplitude=0.02)
    assert feed_in_frames(endpointer, soft_tone) == []

    # Strict mode needs barge_in_strict_ms (400 ms) of speech
    assert feed_in_frames(endpointer, tone(0.3)) == []
    events = feed_in_frames(endpointer, tone(0.15))
    assert [e.kind for e in events] == ["speech_start"]


def test_smart_turn_with_executor_async() -> None:
    import concurrent.futures
    import time

    cfg = VADConfig(smart_turn=True, smart_turn_check_ms=100, min_silence_ms=500)
    # Simulate a slow model (20 ms)
    def slow_model(audio: np.ndarray) -> float:
        time.sleep(0.02)
        return 0.95

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        endpointer = Endpointer(energy_prob, cfg, smart_turn=slow_model, executor=executor)
        # Feed tone + 120 ms silence (triggers async submission)
        events = feed_in_frames(endpointer, np.concatenate([tone(0.5), np.zeros(int(0.16 * SAMPLE_RATE), np.float32)]))
        assert [e.kind for e in events] == ["speech_start"]

        # Wait a short moment for executor to finish
        time.sleep(0.05)

        # Feed next silence window: picks up completed future and commits
        more_events = feed_in_frames(endpointer, np.zeros(512 * 2, np.float32))
        assert [e.kind for e in more_events] == ["speech_end"]
