#!/usr/bin/env python3
"""
Simulator of an IoT voice device (ESP32 with on-device wake word) for Gaja protocol v1.8.

It connects to the device endpoint like the real firmware will: hello with device_id + token + caps, then
plays "the device detected the wake word" from a WAV: `wake` message, the pre-roll burst, and the rest of the
file as real-time-paced 20 ms PCM16 frames, followed by silence until the server closes the mic. It obeys
`listen_end` / `follow_up` / `wake_rejected`, "plays" the received audio with a playback model (so a
barge-in cuts what was not yet played), writes it to a WAV and prints events and timings.

    python clients/device_sim.py hej_gaja_pogoda.wav --token $TOKEN --out reply.wav
    python clients/device_sim.py cmd.wav --barge-in-at 1.5 --no-aec --follow-up-wav next.wav

Exit codes: 0 turn(s) completed, 1 timeout, 2 connection/auth error, 3 wake rejected by the server.
Protocol: docs/ARCHITECTURE.md (v1.8), docs/DEVICE.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

MIC_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000


def load_wav_16k(path: str) -> np.ndarray:
    """WAV -> mono int16 at 16 kHz (resampled with a polyphase filter when needed)."""
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != MIC_RATE:
        ratio = Fraction(MIC_RATE, rate)
        mono = resample_poly(mono, ratio.numerator, ratio.denominator).astype(np.float32)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")


def split_frames(samples: np.ndarray) -> list[bytes]:
    """20 ms PCM16 frames (the last one is padded with zeros)."""
    frames = []
    for i in range(0, len(samples), FRAME_SAMPLES):
        chunk = samples[i : i + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        frames.append(chunk.astype("<i2").tobytes())
    return frames


def split_preroll(samples: np.ndarray, wake_at_s: float, preroll_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """(pre-roll, live part): what the ring buffer holds at the wake instant, and what comes after it."""
    wake = min(len(samples), int(wake_at_s * MIC_RATE))
    start = max(0, wake - preroll_ms * MIC_RATE // 1000)
    return samples[start:wake], samples[wake:]


class Playback:
    """The speaker: received PCM is played back-to-back in real time; cancel() drops what was not played yet."""

    def __init__(self) -> None:
        self.rate = MIC_RATE
        self._chunks: list[tuple[float, bytes]] = []
        self.until = 0.0

    def feed(self, data: bytes, now: float) -> None:
        start = max(now, self.until)
        self._chunks.append((start, data))
        self.until = start + len(data) / 2 / self.rate

    def cancel(self, now: float) -> float:
        """Stop playing at `now`; returns the seconds of audio that were dropped."""
        dropped = max(0.0, self.until - now)
        kept: list[tuple[float, bytes]] = []
        for start, data in self._chunks:
            if start >= now:
                continue
            playable = round((now - start) * self.rate) * 2
            kept.append((start, data[:playable]))
        self._chunks = kept
        self.until = min(self.until, now)
        return dropped

    def playing(self, now: float) -> bool:
        return now < self.until

    def pcm(self) -> bytes:
        return b"".join(data for _, data in self._chunks)


@dataclass
class SimOptions:
    url: str = "ws://127.0.0.1:8792/device"
    device_id: str = "sim-1"
    token: str = ""
    wav: str = ""
    lang: str = "pl"
    wake_at: float = 1.2
    wake_score: float = 0.97
    preroll_ms: int = 1500
    aec: bool = True
    out_rates: tuple[int, ...] = (16000, 24000)
    barge_in_at: float | None = None
    barge_in_wav: str | None = None
    follow_up_wav: str | None = None
    wait_follow_up: bool = True
    out: str | None = None
    timeout: float = 45.0
    verbose: bool = True


@dataclass
class SimResult:
    exit_code: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    pcm: bytes = b""
    rate: int = MIC_RATE


class DeviceSim:
    def __init__(self, opts: SimOptions) -> None:
        self.o = opts
        self.result = SimResult()
        self.playback = Playback()
        self.mic_on = False
        self.pending: deque[bytes] = deque()
        self.done = asyncio.Event()
        self.t0 = time.monotonic()
        self.follow_ups = 0
        self._barge_sent = opts.barge_in_at is None
        self._follow_wav_sent = opts.follow_up_wav is None
        self._last_follow_ms = 0
        self._first_audio_after_wake = False
        self._ws: Any = None
        self._wake_t: float | None = None
        self._speech_end_t: float | None = None
        self._barge_task: asyncio.Task[None] | None = None
        self._follow_task: asyncio.Task[None] | None = None

    # ---- logging -----------------------------------------------------------------------------

    def log(self, name: str, detail: str = "") -> None:
        if self.o.verbose:
            print(f"[+{time.monotonic() - self.t0:7.3f}s] [{name.upper():<13}] {detail}", flush=True)

    def now(self) -> float:
        return time.monotonic()

    # ---- main --------------------------------------------------------------------------------

    async def run(self) -> SimResult:
        o = self.o
        samples = load_wav_16k(o.wav)
        try:
            async with websockets.connect(o.url, max_size=4 * 1024 * 1024) as ws:
                self._ws = ws
                self.log("connected", o.url)
                caps = {"aec": o.aec, "out_rates": list(o.out_rates), "preroll_ms": o.preroll_ms}
                await ws.send(json.dumps({"type": "hello", "client": "device", "device_id": o.device_id,
                                          "token": o.token, "fw": "device-sim/1.8", "lang": o.lang, "caps": caps}))
                self.log("send", f"hello device_id={o.device_id} aec={o.aec} out_rates={list(o.out_rates)}")
                first = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if first.get("type") != "ready":
                    self.log("error", f"expected ready, got {first}")
                    self.result.exit_code = 2
                    return self.result
                self.playback.rate = int(first["sample_rate"])
                self.result.rate = self.playback.rate
                self.log("ready", f"out_rate={first['sample_rate']} device={first.get('device')}")
                pump = asyncio.create_task(self._pump())
                receiver = asyncio.create_task(self._receive())
                try:
                    await self._wake(samples, o.wake_at, "wake")
                    if o.barge_in_at is not None:
                        self._barge_task = asyncio.create_task(self._barge_in(samples))
                    try:
                        await asyncio.wait_for(self.done.wait(), o.timeout)
                    except TimeoutError:
                        self.log("timeout", f"no completion after {o.timeout}s")
                        self.result.exit_code = 1
                finally:
                    for task in (pump, receiver, self._barge_task, self._follow_task):
                        if task is not None:
                            task.cancel()
                    await asyncio.gather(pump, receiver, *(t for t in (self._barge_task, self._follow_task) if t), return_exceptions=True)
        except (OSError, websockets.WebSocketException) as exc:
            self.log("fatal", f"{type(exc).__name__}: {exc}")
            self.result.exit_code = 2
        self._finish()
        return self.result

    def _finish(self) -> None:
        r = self.result
        r.pcm = self.playback.pcm()
        t = r.timings
        if self.o.out and r.pcm:
            sf.write(self.o.out, np.frombuffer(r.pcm, dtype="<i2"), r.rate, subtype="PCM_16")
            self.log("saved_audio", f"{self.o.out} ({len(r.pcm) // 2 / r.rate:.2f}s at {r.rate} Hz)")
        if "wake_to_first_audio_ms" in t:
            self.log("timing", f"wake -> first audio: {t['wake_to_first_audio_ms']:.0f} ms"
                     + (f"; end of speech sent -> first audio: {t['speech_end_to_first_audio_ms']:.0f} ms"
                        if "speech_end_to_first_audio_ms" in t else ""))
        if "wake_to_verdict_ms" in t:
            self.log("timing", f"wake -> verdict: {t['wake_to_verdict_ms']:.0f} ms")

    # ---- device side: wake, mic, speaker -----------------------------------------------------

    async def _wake(self, samples: np.ndarray, wake_at: float, word: str) -> None:
        """The wake word fired: tell the server, send the pre-roll in a burst, queue the live audio for the pump."""
        assert self._ws is not None
        preroll, live = split_preroll(samples, wake_at, self.o.preroll_ms)
        if self.playback.playing(self.now()):  # the device silences its own speaker on a wake
            self.playback.cancel(self.now())
        self.pending.clear()
        self.mic_on = True
        self._wake_t = self.now()
        self._speech_end_t = None
        self._first_audio_after_wake = False
        await self._ws.send(json.dumps({"type": "wake", "word": "hej_gaja", "score": self.o.wake_score,
                                        "preroll_ms": int(len(preroll) * 1000 / MIC_RATE)}))
        self.log("send", f"{word} score={self.o.wake_score} preroll={len(preroll) * 1000 // MIC_RATE} ms")
        for frame in split_frames(preroll):
            await self._ws.send(frame)
        self.pending.extend(split_frames(live))

    async def _pump(self) -> None:
        """20 ms mic frames: queued audio first, then silence while the mic is open (no frames while closed)."""
        next_t = self.now()
        while True:
            next_t += FRAME_MS / 1000
            delay = next_t - self.now()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.25:
                next_t = self.now()
            if not self.mic_on:
                continue
            if not self.o.aec and self.playback.playing(self.now()):
                continue  # no echo canceller: the device does not listen while it plays
            if self.pending:
                frame = self.pending.popleft()
                if not self.pending:
                    self._speech_end_t = self.now()
            else:
                frame = bytes(FRAME_SAMPLES * 2)
            await self._ws.send(frame)

    async def _barge_in(self, samples: np.ndarray) -> None:
        """A second on-device wake while the reply is playing."""
        while not self.playback.playing(self.now()):
            await asyncio.sleep(0.01)
        await asyncio.sleep(self.o.barge_in_at or 0)
        if self._follow_task is not None:
            self._follow_task.cancel()
        if not self.playback.playing(self.now()):
            self.log("barge-in", "nothing is playing any more; no second wake sent")
            self._barge_sent = True
            await self._follow_up_window(self._last_follow_ms)
            return
        again = load_wav_16k(self.o.barge_in_wav) if self.o.barge_in_wav else samples
        self.log("barge-in", f"second wake while speaking ({self.playback.until - self.now():.2f}s of audio left)")
        self._barge_sent = True
        await self._wake(again, self.o.wake_at, "wake (barge-in)")

    async def _follow_up_window(self, ms: int) -> None:
        """Device-side follow-up after the reply drained: send the follow-up WAV once, else wait `ms` and finish."""
        while self.playback.playing(self.now()):
            await asyncio.sleep(0.02)
        self.mic_on = True
        if not self._follow_wav_sent and self._barge_sent:
            self._follow_wav_sent = True
            self.pending.extend(split_frames(load_wav_16k(self.o.follow_up_wav)))
            self.log("follow-up", f"mic open for {ms} ms, sending {self.o.follow_up_wav}")
            return
        self.log("follow-up", f"mic open for {ms} ms")
        if self.o.wait_follow_up:
            await asyncio.sleep(ms / 1000)
            self.log("follow-up", "window over, nobody spoke")
        self.mic_on = False
        self.done.set()

    # ---- server events -----------------------------------------------------------------------

    async def _receive(self) -> None:
        async for message in self._ws:
            if isinstance(message, bytes):
                self._on_audio(message)
            else:
                self._on_event(json.loads(message))

    def _on_audio(self, data: bytes) -> None:
        now = self.now()
        if not self._first_audio_after_wake and self._wake_t is not None:
            self._first_audio_after_wake = True
            t = self.result.timings
            t["wake_to_first_audio_ms"] = (now - self._wake_t) * 1000
            if self._speech_end_t is not None:
                t["speech_end_to_first_audio_ms"] = (now - self._speech_end_t) * 1000
            self.log("first_audio", f"{t['wake_to_first_audio_ms']:.0f} ms after wake")
        self.playback.feed(data, now)

    def _on_event(self, evt: dict[str, Any]) -> None:
        self.result.events.append(evt)
        kind = evt.get("type")
        now = self.now()
        t = self.result.timings
        if kind in ("wake_accepted", "wake_rejected") and self._wake_t is not None:
            t.setdefault("wake_to_verdict_ms", (now - self._wake_t) * 1000)
        if kind == "wake_accepted":
            self.log("wake_accepted", "")
        elif kind == "wake_rejected":
            self.log("wake_rejected", f"reason={evt.get('reason')} -> idle")
            self.mic_on = False
            self.pending.clear()
            self.result.exit_code = 3
            self.done.set()
        elif kind == "listen_end":
            self.log("listen_end", "mic closed")
            self.mic_on = False
            self.pending.clear()
        elif kind == "follow_up":
            self.follow_ups += 1
            self._last_follow_ms = int(evt.get("ms") or 0)
            self.log("follow_up", f"ms={evt.get('ms')} (#{self.follow_ups})")
            if self._barge_sent:  # while a barge-in is still planned the barge task decides what happens next
                if self._follow_task is not None:
                    self._follow_task.cancel()
                self._follow_task = asyncio.create_task(self._follow_up_window(self._last_follow_ms))
        elif kind == "state":
            if evt.get("reason") == "barge_in":
                dropped = self.playback.cancel(now)
                self.log("state", f"{evt.get('state')} (barge_in: dropped {dropped:.2f}s of queued audio)")
            else:
                self.log("state", str(evt.get("state")))
        elif kind == "transcript":
            heard = f" (heard {evt['heard']!r})" if evt.get("heard") else ""
            self.log("transcript", f"[{evt.get('lang')}] {evt.get('text')!r}{heard}")
        elif kind == "utterance":
            self.log("utterance", f"id={evt.get('id')}: {evt.get('text')!r}")
        elif kind == "audio_start":
            self.log("audio_start", f"id={evt.get('id')} rate={evt.get('sample_rate')}")
            if int(evt.get("sample_rate", 0)) != self.playback.rate:
                self.log("warn", "sample rate differs from ready.sample_rate")
        elif kind == "metrics":
            self.log("metrics", f"route={evt.get('route')} stt={evt.get('stt_ms')} e2e_first_audio={evt.get('e2e_first_audio_ms')} ms")
        elif kind == "error":
            self.log("error", str(evt.get("message")))
            if evt.get("message") == "unauthorized":
                self.result.exit_code = 2
                self.done.set()
        elif kind in ("vad", "handoff", "tool", "notice", "audio_end"):
            self.log(str(kind), json.dumps({k: v for k, v in evt.items() if k != "type"}, ensure_ascii=False)[:160])
        else:
            self.log(str(kind), json.dumps(evt, ensure_ascii=False)[:160])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", help="audio with the wake phrase and a command (any rate; resampled to 16 kHz mono)")
    p.add_argument("--url", default="ws://127.0.0.1:8792/device")
    p.add_argument("--device-id", default="sim-1")
    p.add_argument("--token", default=os.environ.get("GAJA_DEVICE_TOKEN", ""), help="or env GAJA_DEVICE_TOKEN")
    p.add_argument("--lang", default="pl", choices=["pl", "en", "auto"])
    p.add_argument("--wake-at", type=float, default=1.2, help="second of the WAV where the on-device wake fires (end of the phrase)")
    p.add_argument("--wake-score", type=float, default=0.97)
    p.add_argument("--preroll-ms", type=int, default=1500)
    p.add_argument("--no-aec", action="store_true", help="device without echo cancellation: stops the mic while playing")
    p.add_argument("--out-rates", default="16000,24000", help="comma-separated output rates the speaker supports")
    p.add_argument("--barge-in-at", type=float, default=None, metavar="SECONDS", help="second wake this long after the reply started playing")
    p.add_argument("--barge-in-wav", default=None, help="audio for the second wake (default: the same WAV)")
    p.add_argument("--follow-up-wav", default=None, help="a command spoken in the follow-up window (no wake phrase)")
    p.add_argument("--no-wait-follow-up", action="store_true", help="finish as soon as the follow-up window opens")
    p.add_argument("--out", default=None, help="write the played reply audio to this WAV")
    p.add_argument("--timeout", type=float, default=45.0)
    a = p.parse_args()
    opts = SimOptions(
        url=a.url, device_id=a.device_id, token=a.token, wav=a.wav, lang=a.lang, wake_at=a.wake_at,
        wake_score=a.wake_score, preroll_ms=a.preroll_ms, aec=not a.no_aec,
        out_rates=tuple(int(x) for x in a.out_rates.split(",")), barge_in_at=a.barge_in_at,
        barge_in_wav=a.barge_in_wav, follow_up_wav=a.follow_up_wav, wait_follow_up=not a.no_wait_follow_up,
        out=a.out, timeout=a.timeout,
    )
    return asyncio.run(DeviceSim(opts).run()).exit_code


if __name__ == "__main__":
    sys.exit(main())
