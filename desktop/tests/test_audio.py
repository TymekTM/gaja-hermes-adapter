"""Resampler and PCM framing correctness (pure numpy, no audio hardware)."""

import time

import numpy as np

from gaja_desktop.audio import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    MultiOutput,
    SinkSpec,
    float_to_pcm16,
    pcm16_to_float,
    resample,
    resample_pcm16,
    split_frames,
)


def sine(freq_hz: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.sin(2 * np.pi * freq_hz * t)


def test_round_trip_identity() -> None:
    x = sine(440.0, 0.5, 16000)
    assert resample(x, 16000, 16000) is x or np.array_equal(resample(x, 16000, 16000), x)


def test_downsample_preserves_frequency() -> None:
    src = sine(440.0, 1.0, 48000)
    out = resample(src, 48000, 16000)
    assert len(out) == 16000
    # estimate the dominant frequency with an FFT
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / 16000)[int(np.argmax(spectrum))]
    assert abs(peak - 440.0) < 2.0


def test_upsample_preserves_frequency() -> None:
    src = sine(350.0, 1.0, 16000)
    out = resample(src, 16000, 48000)
    assert len(out) == 48000
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / 48000)[int(np.argmax(spectrum))]
    assert abs(peak - 350.0) < 2.0


def test_downsample_rejects_aliasing() -> None:
    """A 10 kHz tone at 48k aliases to 2 kHz at 16k; a proper lowpass keeps it tiny."""
    src = sine(10_000.0, 1.0, 48000)
    out = resample(src, 48000, 16000)
    assert np.sqrt(np.mean(out**2)) < 0.05  # nearly gone, not mirrored to 2 kHz


def test_low_frequency_amplitude_is_kept() -> None:
    src = sine(220.0, 1.0, 24000)
    out = resample(src, 24000, 16000)
    assert 0.9 < np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(src**2)) < 1.1


def test_pcm16_resample_bytes() -> None:
    pcm = float_to_pcm16(sine(440.0, 0.5, 48000))
    down = resample_pcm16(pcm, 48000, 16000)
    assert len(down) == len(pcm) // 3 and len(down) % 2 == 0
    assert resample_pcm16(b"", 48000, 16000) == b""
    assert resample_pcm16(pcm, 16000, 16000) is pcm


def test_float_pcm_helpers() -> None:
    x = np.array([0.5, -0.5, 2.0, -2.0, np.nan], dtype=np.float32)
    pcm = float_to_pcm16(x.reshape(-1, 1))  # 2-d input is mixed to mono
    vals = np.frombuffer(pcm, dtype="<i2")
    assert vals[0] == 16383 and vals[1] == -16383  # clipped to +-1
    assert np.allclose(pcm16_to_float(pcm)[:2], [16383 / 32767, -16383 / 32767], atol=1e-3)


def test_split_frames_pads_the_last_one() -> None:
    frames = split_frames(b"\x01\x00" * (FRAME_SAMPLES + 7))
    assert len(frames) == 2
    assert len(frames[0]) == FRAME_BYTES == 640
    assert len(frames[1]) == FRAME_BYTES and frames[1][14:] == bytes(FRAME_BYTES - 14)


class FakeSink:
    """Records everything written; stands in for a soundcard sink."""

    def __init__(self, name: str, rate: int) -> None:
        self.name = name
        self.rate = rate
        self.written: list[bytes] = []
        self.closed = False

    @property
    def buffered_s(self) -> float:
        return sum(len(c) for c in self.written) / 2 / self.rate

    def write(self, pcm: bytes) -> None:
        self.written.append(pcm)

    def close(self) -> None:
        self.closed = True


def test_multi_output_resamples_per_device() -> None:
    sinks: list[FakeSink] = []

    def factory(spec: SinkSpec, rate: int) -> FakeSink:
        sink = FakeSink(spec.needle, rate)
        sinks.append(sink)
        return sink

    out = MultiOutput(["CABLE Input", "Speakers@16000"], 48000, sink_factory=factory)
    out.start()
    assert [s.rate for s in sinks] == [48000, 16000]

    src = float_to_pcm16(sine(440.0, 0.2, 48000))
    out.play(src)
    # wait for the pump thread to drain (bounded; the queue is tiny)

    for _ in range(100):
        if sinks[0].written and sinks[1].written:
            break
        time.sleep(0.01)

    assert len(sinks[0].written[-1]) == len(src)  # same rate: passed through
    assert len(sinks[1].written[-1]) == len(src) // 3  # 48k -> 16k
    dominant = np.fft.rfftfreq(len(pcm16_to_float(sinks[1].written[-1])), 1 / 16000)[
        int(np.argmax(np.abs(np.fft.rfft(pcm16_to_float(sinks[1].written[-1])))))
    ]
    assert abs(dominant - 440.0) < 20.0  # 200 ms window: coarse FFT bin
    out.close()
    assert sinks[0].closed and sinks[1].closed


def test_multi_output_skips_broken_device() -> None:
    def factory(spec: SinkSpec, rate: int) -> FakeSink:
        if "Broken" in spec.needle:
            raise RuntimeError("no such device")
        return FakeSink(spec.needle, rate)

    out = MultiOutput(["Broken Thing", "Speakers"], 24000, sink_factory=factory)
    assert [s.name for s in out.sinks] == ["Speakers"]
    assert any("Broken Thing" in e for e in out.errors)
