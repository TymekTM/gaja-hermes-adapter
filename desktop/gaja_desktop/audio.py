"""Audio: capture (mic / VB-CABLE / output loopback) -> mono 16 kHz PCM16 20 ms frames, and playback
of the negotiated rate on one or more output devices (resampled per device).

`soundcard` is imported lazily inside functions so headless machines (tests, servers) can import and
test everything else. Nothing here ever blocks the asyncio loop: capture and playback run in their
own threads and hand frames over through plain callbacks / bounded queues.
"""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import numpy as np

MIC_RATE = 16000  # the protocol's mic rate
FRAME_MS = 20
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640

_LOOPBACK_PREFIX = "loopback:"


# ---- PCM helpers ------------------------------------------------------------------------


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 (-1..1), shape (n,) or (n, ch) -> mono PCM16 LE bytes."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)  # mono
    data = np.nan_to_num(data, nan=0.0)  # a NaN frame must not poison the cast below
    return (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    """PCM16 LE bytes -> float32 (-1..1)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0


def resample(samples: np.ndarray, src: int, dst: int, half_width: int = 16, beta: float = 8.0) -> np.ndarray:
    """Bandlimited resampling with a Kaiser-windowed sinc kernel (numpy only, no scipy).

    Good enough for 16/24/48 kHz voice; exact for DC and low frequencies, strong anti-aliasing
    when downsampling.
    """
    x = np.asarray(samples, dtype=np.float64)
    if src == dst or len(x) == 0:
        return x
    step = src / dst
    n_out = max(1, int(round(len(x) / step)))
    pos = np.arange(n_out) * step
    base = np.floor(pos).astype(np.int64)
    # A downsampler must lowpass at the *lower* Nyquist; the sinc's cutoff scales accordingly.
    cutoff = min(1.0, dst / src)
    taps = np.arange(-half_width, half_width + 1)
    window = np.kaiser(2 * half_width + 1, beta)
    acc = np.zeros(n_out)
    wsum = np.zeros(n_out)
    for k, off in enumerate(taps):
        idx = base + off
        valid = (idx >= 0) & (idx < len(x))
        if not valid.any():
            continue
        w = cutoff * np.sinc(cutoff * (pos - idx)) * window[k]
        acc[valid] += x[idx[valid]] * w[valid]
        wsum[valid] += w[valid]
    return acc / np.where(wsum == 0.0, 1.0, wsum)


def resample_pcm16(data: bytes, src: int, dst: int) -> bytes:
    """PCM16 LE bytes at `src` Hz -> PCM16 LE bytes at `dst` Hz."""
    if src == dst or not data:
        return data
    y = resample(np.frombuffer(data, dtype="<i2"), src, dst)
    return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


def split_frames(samples: bytes, frame_bytes: int = FRAME_BYTES) -> list[bytes]:
    """Split PCM16 bytes into fixed frames; the last one is zero padded."""
    out = []
    for i in range(0, len(samples), frame_bytes):
        chunk = samples[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + bytes(frame_bytes - len(chunk))
        out.append(chunk)
    return out


# ---- device listing ---------------------------------------------------------------------


@dataclass
class DeviceInfo:
    name: str
    id: str
    channels: int = 1
    loopback: bool = False


def _soundcard() -> Any:
    """Import soundcard lazily (it opens PulseAudio/CoreAudio/WASAPI libraries at import time)."""
    import soundcard  # noqa: PLC0415

    return soundcard


def list_input_devices() -> list[DeviceInfo]:
    """Microphones (and, marked loopback=True, capture-from-output entries)."""
    sc = _soundcard()
    out = []
    for mic in sc.all_microphones(include_loopback=True):
        out.append(DeviceInfo(name=mic.name, id=str(mic.id), channels=int(mic.channels or 1),
                              loopback=bool(getattr(mic, "is_loopback", False))))
    return out


def list_output_devices() -> list[DeviceInfo]:
    sc = _soundcard()
    return [DeviceInfo(name=sp.name, id=str(sp.id), channels=int(sp.channels or 2))
            for sp in sc.all_speakers()]


def find_device(devices: list[DeviceInfo], needle: str, *, loopback: bool) -> DeviceInfo | None:
    """First device whose name contains `needle` (case-insensitive). "" = first non-loopback (default)."""
    want = needle.strip().lower()
    for dev in devices:
        if loopback != dev.loopback:
            continue
        if not want or want in dev.name.lower():
            return dev
    return None


def resolve_capture(spec: str) -> tuple[DeviceInfo, bool]:
    """`""`/`"name"` -> microphone, `"loopback:<output name>"` -> that output's loopback capture.

    Returns (device, is_loopback). Raises RuntimeError with a readable message when nothing matches.
    """
    spec = spec.strip()
    loopback = spec.lower().startswith(_LOOPBACK_PREFIX)
    needle = spec[len(_LOOPBACK_PREFIX):] if loopback else spec
    mics = list_input_devices()
    dev = find_device(mics, needle, loopback=loopback)
    if dev is None:
        kind = "loopback capture of an output device" if loopback else "capture device"
        raise RuntimeError(f"no {kind} matching {needle or '<default>'!r}; see `gaja-desktop devices`")
    return dev, loopback


# ---- capture ---------------------------------------------------------------------------


class Capture:
    """Blocking capture thread: one device -> mono 16 kHz PCM16 20 ms frames via `callback`.

    The callback runs on the capture thread and must be fast (it hosts the wake-word detector);
    it must never touch the asyncio loop directly.
    """

    def __init__(self, spec: str, callback: Callable[[bytes], None], *, device: DeviceInfo | None = None) -> None:
        self.spec = spec
        self.callback = callback
        self._device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gaja-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        sc = _soundcard()
        try:
            if self._device is None:
                self._device, _ = resolve_capture(self.spec)
            mic = sc.get_microphone(self._device.id, include_loopback=True)
            leftover = b""
            with mic.recorder(samplerate=MIC_RATE, channels=1) as rec:
                while not self._stop.is_set():
                    raw = rec.record(FRAME_SAMPLES)
                    pcm = float_to_pcm16(raw)
                    if leftover:  # re-frame to exact 20 ms frames whatever the driver returns
                        pcm = leftover + pcm
                        leftover = b""
                    while len(pcm) >= FRAME_BYTES:
                        self.callback(pcm[:FRAME_BYTES])
                        pcm = pcm[FRAME_BYTES:]
                    leftover = pcm
        except Exception as exc:  # noqa: BLE001 - a dead device must not crash the process silently
            self.error = f"{type(exc).__name__}: {exc}"


# ---- playback -------------------------------------------------------------------------


@dataclass
class SinkSpec:
    """One playback target: `"name substring"` or `"name substring@48000"` (rate override)."""

    needle: str
    rate: int | None = None

    @staticmethod
    def parse(spec: str) -> "SinkSpec":
        m = re.fullmatch(r"(.+?)@(\d{4,5})", spec.strip())
        if m:
            return SinkSpec(needle=m.group(1), rate=int(m.group(2)))
        return SinkSpec(needle=spec.strip())


class Sink:
    """One output device. `write()` receives PCM16 mono at exactly `rate`."""

    def __init__(self, name: str, rate: int) -> None:
        self.name = name
        self.rate = rate
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def open(self, device_id: str | None = None) -> None:
        self._thread = threading.Thread(target=self._run, args=(device_id,), name="gaja-play", daemon=True)
        self._thread.start()

    def write(self, pcm: bytes) -> None:
        """Never blocks for long: on a stalled device the oldest audio is dropped instead."""
        while True:
            try:
                self._queue.put_nowait(pcm)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()  # drop the oldest chunk
                except queue.Empty:
                    pass

    def cancel(self) -> None:
        """Drop everything queued (barge-in): already-started audio keeps playing to its end."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def buffered_s(self) -> float:
        return sum(len(c) for c in list(self._queue.queue)) / 2 / self.rate

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self, device_id: str | None) -> None:
        sc = _soundcard()
        try:
            speaker = sc.get_speaker(device_id) if device_id else sc.default_speaker()
            with speaker.player(samplerate=self.rate, channels=1) as player:
                while not self._stop.is_set():
                    try:
                        pcm = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    player.play(pcm16_to_float(pcm).reshape(-1, 1))
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"


class MultiOutput:
    """Playback of one stream (the negotiated rate) on several devices at once.

    Real sinks are created through `sink_factory` (tests inject fakes). Audio handed to `play()` at
    `rate` is resampled to each sink's rate and queued; `busy()` approximates whether audio is still
    in flight (half duplex uses it to keep the mic off while Gaja speaks).
    """

    def __init__(self, specs: list[str], rate: int,
                 sink_factory: Callable[[SinkSpec, int], Sink] | None = None) -> None:
        self.rate = rate
        self._factory = sink_factory or self._open_sink
        self.sinks: list[Sink] = []
        self.errors: list[str] = []
        self._started = False
        for spec in specs:
            parsed = SinkSpec.parse(spec) if isinstance(spec, str) else spec
            try:
                sink = self._factory(parsed, parsed.rate or rate)
            except Exception as exc:  # noqa: BLE001 - one bad device must not kill playback
                self.errors.append(f"{parsed.needle or '<default>'}: {exc}")
                continue
            if sink is not None:
                self.sinks.append(sink)
        self._queue: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=100)
        self._tail_s = 0.0  # seconds of audio past the sinks' own queues
        self._lock = threading.Lock()

    @staticmethod
    def _open_sink(spec: SinkSpec, rate: int) -> Sink:
        """Find the output device and open it. The default sink ("") is the system default speaker."""
        sc = _soundcard()
        speakers = sc.all_speakers()
        want = spec.needle.lower()
        speaker = next((s for s in speakers if want in s.name.lower()), None) if want else sc.default_speaker()
        if speaker is None:
            raise RuntimeError(f"no playback device matching {spec.needle!r}; see `gaja-desktop devices`")
        sink = Sink(speaker.name, rate)
        sink.open(str(speaker.id))
        return sink

    def play(self, pcm: bytes, rate: int | None = None) -> None:
        """Queue one chunk (PCM16 mono at `rate`, default the negotiated rate) for every sink."""
        rate = rate or self.rate
        with self._lock:
            self._tail_s = min(5.0, self._tail_s + len(pcm) / 2 / rate)
        try:
            self._queue.put_nowait((pcm, rate))
        except queue.Full:
            pass  # the consumer is way behind: prefer dropping audio over growing latency

    def _pump(self) -> None:
        while True:
            pcm, rate = self._queue.get()
            if not pcm:
                return
            with self._lock:
                self._tail_s = max(0.0, self._tail_s - len(pcm) / 2 / rate)
            for sink in self.sinks:
                sink.write(resample_pcm16(pcm, rate, sink.rate))

    def start(self) -> None:
        if self.sinks and not self._started:
            self._started = True
            threading.Thread(target=self._pump, name="gaja-out", daemon=True).start()

    def busy(self) -> bool:
        """True while chunks are queued or sit in a sink's queue (coarse; ±one chunk)."""
        if not self._queue.empty():
            return True
        return any(sink.buffered_s > 0.02 for sink in self.sinks)

    def cancel(self) -> None:
        """Barge-in: drop everything not yet played (audio already handed to the OS keeps playing)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._tail_s = 0.0
        for sink in self.sinks:
            if hasattr(sink, "cancel"):
                sink.cancel()

    def close(self) -> None:
        self._queue.put((b"", 0))
        for sink in self.sinks:
            sink.close()
        self.sinks = []
