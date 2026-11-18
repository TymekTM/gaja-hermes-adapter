#!/usr/bin/env python3
"""
Simulator of an IoT voice device (ESP32 with on-device wake word) for Gaja protocol v1.8.

It connects to the device endpoint like the real firmware will: hello with device_id + token + caps, then
plays "the device detected the wake word" from a WAV: `wake` message, the pre-roll burst, and the rest of the
file as real-time-paced 20 ms PCM16 frames, followed by silence until the server closes the mic. It obeys
`listen_end` / `follow_up` / `wake_rejected`, "plays" the received audio with a playback model (so a
barge-in cuts what was not yet played), writes it to a WAV and prints events and timings.

With `--record` it instead advertises `caps.record` (protocol v1.10) and waits for the server: on
`record_start` it answers `record_started`, streams the WAV as the live mic (then silence), reports a
`record_mark` after `--mark-at` seconds of audio, and ends with `record_end` on `record_stop` or at
`max_ms` (`reason: max`). This exercises the wake-word take recording flow end to end.

With `--pc` it advertises `caps.ptt` + `caps.pc` (protocol v1.14, docs/DESKTOP.md) and answers every
`pc_call` with a fake result (`shell` echoes the command, `screenshot` returns a tiny 1x1 PNG), so
the pc relay (`gaja/devices/pc.py`) and `/api/pc` can be exercised end to end. With `--pc` and no
WAV it just connects and serves tool calls until the timeout (exit 0).

    python clients/device_sim.py hej_gaja_pogoda.wav --token $TOKEN --out reply.wav
    python clients/device_sim.py cmd.wav --barge-in-at 1.5 --no-aec --follow-up-wav next.wav
    python clients/device_sim.py take.wav --record --mark-at 0.8 --token $TOKEN
    python clients/device_sim.py x.wav --pc --pc-host TYMEK-PC   # voice device + fake PC tools
    python clients/device_sim.py --pc --timeout 30               # PC only, no audio

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

# Protocol v1.14 (--pc): the fake PC tool executor advertises and answers these.
PC_TOOLS = ("shell", "open", "screenshot", "type", "keys", "mouse", "clipboard")
# A valid 1x1 transparent PNG (the fake `screenshot` result).
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8//8/AwAF/gL+py2ltQAAAABJRU5ErkJggg=="
)


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
    record: bool = False
    mark_at: float | None = None
    pc: bool = False
    pc_host: str = "SIM-PC"
    pc_os: str = "linux"
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
    record_log: list[dict[str, Any]] = field(default_factory=list)  # record mode: what the device sent
    pc_log: list[dict[str, Any]] = field(default_factory=list)  # --pc: tool calls answered


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
        self._record_task: asyncio.Task[None] | None = None
        self._record_stop = asyncio.Event()
        self._record_id = ""
        self._busy_reply: asyncio.Task[None] | None = None
        self._pc_tasks: set[asyncio.Task[None]] = set()

    # ---- logging -----------------------------------------------------------------------------

    def log(self, name: str, detail: str = "") -> None:
        if self.o.verbose:
            print(f"[+{time.monotonic() - self.t0:7.3f}s] [{name.upper():<13}] {detail}", flush=True)

    def now(self) -> float:
        return time.monotonic()

    # ---- main --------------------------------------------------------------------------------

    async def run(self) -> SimResult:
        o = self.o
        samples = load_wav_16k(o.wav) if o.wav else np.zeros(0, dtype="<i2")
        try:
            async with websockets.connect(o.url, max_size=4 * 1024 * 1024) as ws:
                self._ws = ws
                self.log("connected", o.url)
                caps = {"aec": o.aec, "out_rates": list(o.out_rates), "preroll_ms": o.preroll_ms}
                if o.record:
                    caps["record"] = True  # protocol v1.10: this device can record its mic on request
                if o.pc:
                    # Protocol v1.14: push-to-talk + a (fake) PC tool executor (docs/DESKTOP.md).
                    caps["ptt"] = True
                    caps["pc"] = {"os": o.pc_os, "host": o.pc_host, "tools": list(PC_TOOLS)}
                await ws.send(json.dumps({"type": "hello", "client": "device", "device_id": o.device_id,
                                          "token": o.token, "fw": "device-sim/1.14", "lang": o.lang, "caps": caps}))
                self.log("send", f"hello device_id={o.device_id} aec={o.aec} out_rates={list(o.out_rates)}"
                         + (" record=True" if o.record else "") + (f" pc host={o.pc_host}" if o.pc else ""))
                first = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if first.get("type") != "ready":
                    self.log("error", f"expected ready, got {first}")
                    self.result.exit_code = 2
                    return self.result
                self.playback.rate = int(first["sample_rate"])
                self.result.rate = self.playback.rate
                if o.pc and first.get("device", {}).get("pc") is not True:
                    self.log("warn", "ready.device.pc is not true: the server will not send pc_call")
                self.log("ready", f"out_rate={first['sample_rate']} device={first.get('device')}")
                pump = asyncio.create_task(self._pump())
                receiver = asyncio.create_task(self._receive())
                try:
                    if o.record:
                        self.log("record", f"waiting for record_start (streaming {o.wav}; Ctrl-C to quit)")
                    elif o.wav:
                        await self._wake(samples, o.wake_at, "wake")
                    else:
                        self.log("pc", "serving PC tool calls; no audio (Ctrl-C to quit)")
                    if o.barge_in_at is not None and not o.record and o.wav:
                        self._barge_task = asyncio.create_task(self._barge_in(samples))
                    try:
                        await asyncio.wait_for(self.done.wait(), o.timeout)
                    except TimeoutError:
                        self.log("timeout", f"no completion after {o.timeout}s")
                        # A PC-only run (no WAV) ends on its own terms, not with an error.
                        self.result.exit_code = 0 if (o.pc and not o.wav) else 1
                finally:
                    for task in (pump, receiver, self._barge_task, self._follow_task, self._record_task):
                        if task is not None:
                            task.cancel()
                    await asyncio.gather(pump, receiver, *(t for t in (self._barge_task, self._follow_task,
                                                                      self._record_task) if t),
                                         return_exceptions=True)
                    for task in self._pc_tasks:
                        task.cancel()
                    await asyncio.gather(*self._pc_tasks, return_exceptions=True)
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
            if self.pending:
                # The previous turn's follow_up can land a few ms after the barge-in wake was sent,
                # so this window opens while the barge-in utterance is still streaming -- and its
                # wake_accepted is seconds away (paced audio, then verify). A real device keeps its
                # mic open and waits for the server; ending the scenario here finished the run before
                # the second wake was accepted. The turn the utterance starts sends its own follow_up,
                # which opens a fresh window that ends the run.
                self.log("follow-up", "window over, but a barge-in utterance is still streaming")
                return
            self.log("follow-up", "window over, nobody spoke")
        self.mic_on = False
        self.done.set()

    # ---- server events -----------------------------------------------------------------------

    async def _record_take(self, evt: dict[str, Any]) -> None:
        """Protocol v1.10 record mode: stream the WAV as the live mic, then silence, until stop/max_ms."""
        assert self._ws is not None
        o = self.o
        take_id = str(evt.get("id"))
        max_ms = min(120_000, max(1000, int(evt.get("max_ms") or 60_000)))
        await self._ws.send(json.dumps({"type": "record_started", "id": take_id}))
        self.result.record_log.append({"type": "record_started", "id": take_id})
        self.log("record_started", f"id={take_id} max_ms={max_ms}")
        frames = split_frames(load_wav_16k(o.wav))
        sent_ms = 0
        marked = False
        next_t = self.now()
        while sent_ms < max_ms and not self._record_stop.is_set():
            next_t += FRAME_MS / 1000
            delay = next_t - self.now()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.25:
                next_t = self.now()
            await self._ws.send(frames[sent_ms // FRAME_MS] if sent_ms // FRAME_MS < len(frames) else bytes(FRAME_SAMPLES * 2))
            sent_ms += FRAME_MS
            if o.mark_at is not None and not marked and sent_ms >= o.mark_at * 1000:
                marked = True
                # t_ms = the audio sent so far when the detector fired (it lags the word end by ~0.7 s)
                await self._ws.send(json.dumps({"type": "record_mark", "id": take_id, "word": "hej_gaja",
                                                "t_ms": sent_ms}))
                self.result.record_log.append({"type": "record_mark", "word": "hej_gaja", "t_ms": sent_ms})
                self.log("record_mark", f"word=hej_gaja t_ms={sent_ms}")
        reason = "stop" if self._record_stop.is_set() else "max"
        await self._ws.send(json.dumps({"type": "record_end", "id": take_id, "ms": sent_ms, "reason": reason}))
        self.result.record_log.append({"type": "record_end", "ms": sent_ms, "reason": reason})
        self.log("record_end", f"ms={sent_ms} reason={reason}")
        self.done.set()

    # ---- fake PC tools (protocol v1.14, --pc) --------------------------------------------------

    def _pc_fake_result(self, evt: dict[str, Any]) -> dict[str, Any]:
        """Answer a pc_call with something shaped like the real client would send (docs/DESKTOP.md)."""
        tool = str(evt.get("tool"))
        args = evt.get("args") if isinstance(evt.get("args"), dict) else {}
        result: dict[str, Any] = {"type": "pc_result", "id": str(evt.get("id")), "ok": True, "text": "", "data": None}
        if tool not in PC_TOOLS:
            result.update(ok=False, error=f"bad_request: unknown tool {tool}")
        elif tool == "shell":
            command = str(args.get("command") or "")
            result.update(text=f"$ {command}\n{command}\n(sim shell echo)", data={"exit_code": 0})
        elif tool == "open":
            result.update(text=f"opened {args.get('target')} (sim)")
        elif tool == "screenshot":
            result.update(text="1x1 PNG (sim), monitor 0", image={
                "mime": "image/png", "b64": TINY_PNG_B64, "width": 1, "height": 1,
            })
        elif tool == "type":
            result.update(text=f"{len(str(args.get('text') or ''))} characters typed (sim)")
        elif tool == "keys":
            combo = args.get("combo") or " + ".join(args.get("sequence") or [])
            result.update(text=f"pressed {combo} (sim)")
        elif tool == "mouse":
            result.update(text=f"cursor at {args.get('x', 0)},{args.get('y', 0)} (sim)")
        elif tool == "clipboard":
            if args.get("action") == "set":
                result.update(text=f"clipboard set (sim): {args.get('text')}")
            else:
                result.update(text="(sim clipboard is empty)")
        return result

    async def _answer_pc_call(self, evt: dict[str, Any]) -> None:
        """Run one tool call off the receive loop; several may be in flight at once."""
        assert self._ws is not None
        await asyncio.sleep(0.05)  # a tiny bit of work, so concurrency is observable
        result = self._pc_fake_result(evt)
        self.result.pc_log.append({"tool": evt.get("tool"), "id": evt.get("id"),
                                   "ok": result.get("ok", True)})
        await self._ws.send(json.dumps(result))
        self.log("pc_result", f"tool={evt.get('tool')} id={evt.get('id')} ok={result.get('ok')}")

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
        elif kind == "record_start":
            if self._record_task is not None and not self._record_task.done():
                # Like the firmware: a record_start while recording (even with the same id) is busy.
                assert self._ws is not None
                self._busy_reply = asyncio.create_task(
                    self._ws.send(json.dumps({"type": "record_error", "id": str(evt.get("id")), "reason": "busy"}))
                )
                self.log("record_error", f"id={evt.get('id')} reason=busy")
                return
            self._record_id = str(evt.get("id"))
            self._record_stop = asyncio.Event()
            self._record_task = asyncio.create_task(self._record_take(evt))
        elif kind == "record_stop":
            # An id the device is not recording is ignored.
            if self._record_task is not None and not self._record_task.done() and str(evt.get("id")) == self._record_id:
                self._record_stop.set()
            else:
                self.log("record_stop", f"ignored (id={evt.get('id')})")
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
        elif kind == "pc_call":
            # Protocol v1.14: answer off the receive loop, concurrently with audio and other calls.
            if self._ws is not None:
                task = asyncio.create_task(self._answer_pc_call(evt))
                self._pc_tasks.add(task)
                task.add_done_callback(self._pc_tasks.discard)
                self.log("pc_call", f"tool={evt.get('tool')} id={evt.get('id')} timeout_s={evt.get('timeout_s')}")
        elif kind in ("vad", "handoff", "tool", "notice", "audio_end"):
            self.log(str(kind), json.dumps({k: v for k, v in evt.items() if k != "type"}, ensure_ascii=False)[:160])
        else:
            self.log(str(kind), json.dumps(evt, ensure_ascii=False)[:160])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", nargs="?", default="", help="audio with the wake phrase and a command (any rate; resampled to 16 kHz mono); optional with --pc")
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
    p.add_argument("--record", action="store_true",
                   help="recording mode (protocol v1.10): advertise caps.record, wait for record_start and stream the WAV as the live mic")
    p.add_argument("--mark-at", type=float, default=None, metavar="SECONDS",
                   help="record mode: send record_mark (hej_gaja) after this many seconds of the take were sent")
    p.add_argument("--pc", action="store_true",
                   help="advertise caps.ptt + caps.pc (protocol v1.14) and answer pc_call with fake results "
                        "(shell echoes the command, screenshot returns a tiny PNG)")
    p.add_argument("--pc-host", default="SIM-PC", help="--pc: caps.pc host name (default SIM-PC)")
    p.add_argument("--pc-os", default="linux", choices=["linux", "windows", "macos"], help="--pc: caps.pc os")
    p.add_argument("--out", default=None, help="write the played reply audio to this WAV")
    p.add_argument("--timeout", type=float, default=45.0)
    a = p.parse_args()
    if not a.wav and not a.record and not a.pc:
        p.error("a WAV is required unless --pc or --record is given")
    opts = SimOptions(
        url=a.url, device_id=a.device_id, token=a.token, wav=a.wav, lang=a.lang, wake_at=a.wake_at,
        wake_score=a.wake_score, preroll_ms=a.preroll_ms, aec=not a.no_aec,
        out_rates=tuple(int(x) for x in a.out_rates.split(",")), barge_in_at=a.barge_in_at,
        barge_in_wav=a.barge_in_wav, follow_up_wav=a.follow_up_wav, wait_follow_up=not a.no_wait_follow_up,
        record=a.record, mark_at=a.mark_at, pc=a.pc, pc_host=a.pc_host, pc_os=a.pc_os,
        out=a.out, timeout=a.timeout,
    )
    return asyncio.run(DeviceSim(opts).run()).exit_code


if __name__ == "__main__":
    sys.exit(main())
