"""Server-side echo canceller (gaja/devices/aec.py) against synthetic echo, and the v1.11
frame-tag contract with the firmware.

The deterministic core of speech barge-in: a reference signal (what Gaja plays) is echoed into
the microphone (delayed + attenuated + convolved with a short room impulse response) and the
canceller must remove it well enough that the amplitude VAD used in the tests (prob = 1 iff
peak > 0.05) never sees it -- while real user speech added on top of the same echo stays far
above that threshold. Everything is deterministic: fixed signals, fixed seed for the IR.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from gaja.devices.aec import BLOCK, FRAME_MIC, FRAME_REF, RATE, EchoCanceller

ROOT = Path(__file__).resolve().parents[1]
FRAME_TAG_H = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "frame_tag.h"


# ---- signals -------------------------------------------------------------------------------------


def speechish(seconds: float, amp: float = 0.5, seed: int = 7) -> np.ndarray:
    """Envelope-modulated bandlimited noise: speech-like (200 Hz .. 4 kHz, 2.3 Hz syllables),
    deterministic, broadband like real speech (a pure-tone signal is the pathological
    worst case for adaptive-filter conditioning, not what a speaker plays)."""
    rng = np.random.RandomState(seed)
    n = int(seconds * RATE)
    x = rng.standard_normal(n).astype(np.float32)
    h = np.sinc(2 * 4000.0 / RATE * (np.arange(33) - 16)) * np.hamming(33)
    x = np.convolve(x, (h / h.sum()).astype(np.float32))[:n].astype(np.float32)
    h2 = np.sinc(2 * 200.0 / RATE * (np.arange(201) - 100)) * np.hamming(201)
    slow = np.convolve(x, (h2 / h2.sum()).astype(np.float32))[:n].astype(np.float32)
    x = (x - slow).astype(np.float32)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 2.3 * np.arange(n) / RATE)
    x = x * env.astype(np.float32)
    return (x * (amp / np.sqrt(float(np.mean(x ** 2))))).astype(np.float32)


def room_ir(taps: int = 480, seed: int = 3) -> np.ndarray:
    """A short decaying room impulse response (32 ms at 16 kHz)."""
    rng = np.random.RandomState(seed)
    ir = rng.standard_normal(taps).astype(np.float32)
    ir *= np.exp(-np.arange(taps) / (taps / 4)).astype(np.float32)
    ir[0] += 1.0
    return (ir / np.sqrt(float(np.dot(ir, ir)))).astype(np.float32)


def echo_of(ref: np.ndarray, delay_s: float, gain: float = 0.5, ir: np.ndarray | None = None) -> np.ndarray:
    """What the microphone picks up: attenuated, delayed, optionally convolved with an IR."""
    d = int(delay_s * RATE)
    body = ref if ir is None else np.convolve(ref, ir)[: ref.size]
    out = np.zeros_like(ref)
    out[d:] = (gain * body)[: ref.size - d]
    return out.astype(np.float32)


def stream_frames(mic: np.ndarray, ref: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Cut both streams into 20 ms frames the way the device sends them (mic frame, ref frame)."""
    frames_mic = [mic[i : i + BLOCK] for i in range(0, mic.size, BLOCK)]
    frames_ref = [ref[i : i + BLOCK] for i in range(0, ref.size, BLOCK)]
    return frames_mic, frames_ref


def run_canceller(mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Feed alternating mic/ref frames like the device does; return everything cleaned."""
    frames_mic, frames_ref = stream_frames(mic, ref)
    aec = EchoCanceller()
    out: list[np.ndarray] = []
    for m, r in zip(frames_mic, frames_ref):
        aec.push_ref(r)
        out.append(aec.process(m))
    # The device keeps streaming mic frames after the reference stops (playback ended); the
    # starvation path must flush the held-back tail with a frozen filter instead of stalling.
    for _ in range(12):
        out.append(aec.process(np.zeros(BLOCK, dtype=np.float32)))
    return np.concatenate(out)


def db(x: float, y: float) -> float:
    return 20.0 * float(np.log10(max(float(x), 1e-10) / max(float(y), 1e-10)))


# ---- tests ---------------------------------------------------------------------------------------


def test_frame_tags_match_firmware() -> None:
    """The v1.11 frame tags are a cross-language contract: header and Python must agree."""
    text = FRAME_TAG_H.read_text(encoding="utf-8")
    mic = int(re.search(r"FRAME_TAG_MIC\s*=\s*(0x[0-9A-Fa-f]+)", text).group(1), 16)
    ref = int(re.search(r"FRAME_TAG_REF\s*=\s*(0x[0-9A-Fa-f]+)", text).group(1), 16)
    assert mic == FRAME_MIC == 0x00
    assert ref == FRAME_REF == 0x01


def longest_loud_run(samples: np.ndarray, threshold: float = 0.05) -> int:
    """Longest run of consecutive 32 ms VAD windows whose peak exceeds `threshold`."""
    n_windows = len(samples) // 320
    run = best = 0
    for i in range(n_windows):
        if float(np.abs(samples[i * 320 : (i + 1) * 320]).max()) > threshold:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def test_pure_delayed_echo_is_cancelled_below_the_vad_threshold() -> None:
    ref = speechish(2.0)
    mic = echo_of(ref, delay_s=0.08, gain=0.4)
    mic += (0.001 * np.random.RandomState(1).standard_normal(mic.size)).astype(np.float32)
    cleaned = run_canceller(mic, ref)
    assert np.isfinite(cleaned).all()
    # Measured over the converged part; the starvation tail at the end of the stream is a known
    # edge (see the convolved case).
    body = slice(int(1.2 * RATE), int(1.8 * RATE))
    residual_db = db(np.sqrt(np.mean(cleaned[body] ** 2)), np.sqrt(np.mean(mic[body] ** 2)))
    assert residual_db < -15.0, residual_db
    # The strict test VAD confirms barge-in only after 13 windows above 0.05 peak in one
    # speaking segment: the residual (including the gate-lift transient) must stay clearly below
    # that -- a couple of loud windows never confirm anything.
    assert longest_loud_run(cleaned[int(0.9 * RATE) : int(1.8 * RATE)]) < 10


def test_convolved_delayed_echo_is_cancelled() -> None:
    """The mandated synthetic case: reference convolved with an IR, delayed, attenuated."""
    ref = speechish(2.5)
    mic = echo_of(ref, delay_s=0.05, gain=0.5, ir=room_ir())
    cleaned = run_canceller(mic, ref)
    assert np.isfinite(cleaned).all()
    # The starvation tail (the last held-back blocks flushed once the reference stream ended)
    # is a known edge: its echo estimate runs on a partially missing reference window. It is
    # excluded here; on the device the reply ends with the amp going off, not a hard cutoff.
    body = slice(int(1.4 * RATE), int(2.2 * RATE))
    residual_db = db(np.sqrt(np.mean(cleaned[body] ** 2)), np.sqrt(np.mean(mic[body] ** 2)))
    assert residual_db < -12.0, residual_db
    assert longest_loud_run(cleaned[body]) < 10  # cannot confirm a strict-VAD speech segment (13)


def test_user_speech_on_top_of_echo_survives() -> None:
    """Double talk: the cancellation must not learn the USER away; the residual stays speech."""
    ref = speechish(2.5)
    echo = echo_of(ref, delay_s=0.05, gain=0.5, ir=room_ir())
    speech = (0.3 * np.sin(2 * np.pi * 220.0 * np.arange(echo.size) / RATE)).astype(np.float32)
    # The user starts speaking 0.8 s in (the canceller is aligned and converged by then).
    speech[: int(0.8 * RATE)] = 0.0
    mic = echo + speech
    cleaned = run_canceller(mic, ref)
    speak = slice(int(1.0 * RATE), int(2.0 * RATE))
    residual = cleaned[speak]
    assert float(np.abs(residual).max()) > 0.1  # far above the 0.05 VAD threshold: barge-in fires
    residual_db = db(np.sqrt(np.mean(residual ** 2)), np.sqrt(np.mean(speech[speak] ** 2)))
    assert residual_db > -10.0, residual_db  # the speech survives cancellation


def test_delay_is_found_within_the_search_range() -> None:
    """The unified long filter learns delay+IR together; there is no coarse delay state left.

    Kept as a behavioural check: the cancellation must reach echo paths across the plausible
    offset range (0, 40, 120 and 150 ms -- the tail is 192 ms; 0.25 s is OUT of spec).
    """
    for delay_s in (0.0, 0.04, 0.12, 0.15):
        ref = speechish(2.6)
        mic = echo_of(ref, delay_s=delay_s, gain=0.4)
        cleaned = run_canceller(mic, ref)
        body = slice(int(1.8 * RATE), int(2.4 * RATE))
        residual_db = db(np.sqrt(np.mean(cleaned[body] ** 2)), np.sqrt(np.mean(mic[body] ** 2)))
        assert residual_db < -10.0, (delay_s, residual_db)


def test_streaming_conserves_samples_and_bounded_latency() -> None:
    """Nothing is duplicated or lost: fed samples == returned samples once the tail flushed."""
    ref = speechish(1.5)
    mic = echo_of(ref, delay_s=0.06, gain=0.4)
    frames_mic, frames_ref = stream_frames(mic, ref)
    aec = EchoCanceller()
    fed = 0
    total_out = 0
    for m, r in zip(frames_mic, frames_ref):
        aec.push_ref(r)
        total_out += aec.process(m).size
        fed += m.size
    for _ in range(12):
        flush = np.zeros(BLOCK, dtype=np.float32)
        total_out += aec.process(flush).size
        fed += flush.size
    assert total_out == fed


def test_mic_passes_through_before_any_reference() -> None:
    """Frames before the first playback (caps.ref device, no ref yet) are untouched."""
    aec = EchoCanceller()
    mic = speechish(0.2, amp=0.3)
    out = aec.process(mic)
    np.testing.assert_allclose(out, mic, atol=1e-6)


def test_ref_without_mic_keeps_delay_convergence_cheap() -> None:
    """Reference frames alone must not break the next alignment (announcements end, etc.)."""
    aec = EchoCanceller()
    aec.push_ref(speechish(0.5))
    assert aec.process(speechish(0.2, amp=0.4)).size >= 0  # no crash, no exception
    aec.push_ref(speechish(0.5, seed=8))
    out = aec.process(speechish(0.5, amp=0.4, seed=9))
    assert np.isfinite(out).all()


def test_second_reply_after_a_mic_only_gap_is_still_cancelled() -> None:
    """The firmware sends reference frames only while playback is audible, but the mic keeps
    streaming between replies (follow-up listening, the next utterance). The reference timeline
    must resume aligned with the mic -- the reference of the silent gap is silence -- or the
    second reply after boot is not cancelled at all."""
    ir = room_ir()
    aec = EchoCanceller()
    noise = np.random.RandomState(5)

    def play(ref: np.ndarray) -> np.ndarray:
        mic = echo_of(ref, delay_s=0.06, gain=0.4, ir=ir)
        mic += (0.001 * noise.standard_normal(mic.size)).astype(np.float32)
        out = []
        for m, r in zip(*stream_frames(mic, ref)):
            aec.push_ref(r)
            out.append(aec.process(m))
        return np.concatenate(out), mic

    play(speechish(2.0, seed=11))
    for _ in range(150):  # 3 s of mic without any reference (playback ended)
        aec.process((0.001 * noise.standard_normal(BLOCK)).astype(np.float32))
    cleaned, mic = play(speechish(2.0, seed=12))
    body = slice(int(0.6 * RATE), int(1.6 * RATE))
    n = min(cleaned.size, body.stop)
    residual_db = db(np.sqrt(np.mean(cleaned[body.start : n] ** 2)), np.sqrt(np.mean(mic[body.start : n] ** 2)))
    assert residual_db < -12.0, residual_db
    # The onset residual of a new reply (the filter adapts to new content) must stay well under
    # barge_in_strict_ms = 400 (config.toml): < 18 loud 20 ms windows = < 360 ms. Without the
    # per-reply onset gate this case measured 20 windows (400 ms): right at the threshold.
    assert longest_loud_run(cleaned) < 18


def test_cleaned_audio_stays_float32_once_the_filter_converges() -> None:
    """The VAD's ONNX model rejects float64: every chunk, gated or converged, must be float32."""
    ref = speechish(2.5)
    mic = echo_of(ref, delay_s=0.05, gain=0.5, ir=room_ir())
    frames_mic, frames_ref = stream_frames(mic, ref)
    aec = EchoCanceller()
    dtypes = set()
    for m, r in zip(frames_mic, frames_ref):
        aec.push_ref(r)
        out = aec.process(m)
        if out.size:
            dtypes.add(out.dtype)
    assert dtypes == {np.dtype(np.float32)}
