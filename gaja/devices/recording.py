"""Server side of wake-word take recording through a device microphone (protocol v1.10).

The recorder page (web/record.html) can pick a connected recording-capable device as the audio source
instead of the browser mic. The server then drives the device over its WebSocket:

    POST /api/admin/record/takes          -> record_start {id, max_ms}   (one active recording per device)
    device                                -> record_started {id}, then binary PCM16 LE mono 16 kHz frames
    device                                -> record_mark {id, word, t_ms}  (its own wake detector fired)
    POST .../stop  or max_ms on the device -> record_stop / device stops itself
    device                                -> record_end {id, ms, reason: stop|max}

A disconnect mid-take keeps what was received and marks the take ``"partial": true`` in meta.
Frames that arrive while a recording is active belong to the recording and never reach the conversation
pipeline; a ``wake`` from the device during a recording is ignored (it should not happen).

Finished takes are written through the same `SampleStore` as browser uploads
(``<recordings_dir>/<session>/<label>/<prompt_id>/<take>.wav`` + a meta.jsonl line), with extra meta:
``source: "device"``, ``device_id``, ``fw``, ``marks``, ``device_detected``, ``partial``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import secrets
import time
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gaja.wakeword_samples import BYTES_PER_SECOND, MAX_BYTES, SAMPLE_RATE, SampleStore

if TYPE_CHECKING:
    from gaja.devices.controller import DeviceController

log = logging.getLogger(__name__)

MIN_MAX_MS = 1000
MAX_MAX_MS = 120_000
DEFAULT_MAX_MS = 60_000
# Frames after the device should have answered record_end on its own at max_ms; then we finalise anyway.
MAX_GRACE_S = 3.0
# PCM16 LE mono 16 kHz = 32 bytes per millisecond (the firmware derives record_end.ms with the same constant).
BYTES_PER_MS = BYTES_PER_SECOND // 1000
# How long the HTTP API waits for record_started / record_error after sending record_start.
START_ACK_S = 3.0
# How long the HTTP API waits for record_end after sending record_stop.
STOP_ACK_S = 5.0
# Live level of the last frame, as dBFS, when nothing arrived yet.
LEVEL_SILENT_DBFS = -100.0

RECORD_REASONS = ("stop", "max")


def clamp_max_ms(value: int | float | None) -> int:
    if value is None:
        return DEFAULT_MAX_MS
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"max_ms must be a number in {MIN_MAX_MS}..{MAX_MAX_MS}")
    return int(min(MAX_MAX_MS, max(MIN_MAX_MS, int(value))))


def make_take_id(device_id: str, prompt_id: str, take: int) -> str:
    """A take id for the protocol: [a-z0-9_-]{1,64}, readable, unique per attempt."""
    dev = re.sub(r"[^a-z0-9_-]", "", device_id.lower())[:16]
    raw = f"{dev}-{prompt_id}-{take:03d}"[:48].rstrip("-")
    return f"{raw}-{secrets.token_hex(3)}"


def wav_from_pcm16(pcm: bytes) -> bytes:
    """Wrap raw PCM16 LE mono 16 kHz samples in a RIFF/WAVE header (exactly what the browser uploads)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


@dataclass
class DeviceRecording:
    """One take in progress on one device; owned by the DeviceController of its connection."""

    id: str
    device_id: str
    fw: str
    session: str  # recordings session name (the storage tree), not a conversation
    label: str
    prompt_id: str
    take: int
    max_ms: int
    store: SampleStore
    started_at: float = field(default_factory=time.perf_counter)
    state: str = "starting"  # starting | recording | ending | storing | done | error
    buf: bytearray = field(default_factory=bytearray)
    marks: list[dict[str, Any]] = field(default_factory=list)
    end_reason: str = ""  # stop | max | partial
    error: str = ""
    truncated: bool = False
    ms: int = 0  # audio received, ms (from our own byte count until the device reports its total)
    level_dbfs: float = LEVEL_SILENT_DBFS
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    result: dict[str, Any] | None = None  # the SampleStore.write() response once stored
    _timer: asyncio.Task[None] | None = field(default=None, repr=False)

    # ---- live view ---------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "device_id": self.device_id,
            "session": self.session,
            "label": self.label,
            "prompt_id": self.prompt_id,
            "take": self.take,
            "max_ms": self.max_ms,
            "state": self.state,
            "elapsed_ms": int((time.perf_counter() - self.started_at) * 1000),
            "ms": self.ms,
            "level_dbfs": round(self.level_dbfs, 1),
            "marks": list(self.marks),
            "truncated": self.truncated,
            "device_detected": bool(self.marks),
        }
        if self.state in ("storing", "done", "error"):
            out["end_reason"] = self.end_reason
            out["error"] = self.error
            out["partial"] = self.end_reason == "partial"
            out["result"] = self.result
        return out

    @property
    def active(self) -> bool:
        return self.state in ("starting", "recording", "ending")

    @property
    def streaming(self) -> bool:
        """The device has accepted the take: binary frames now belong to it.

        Before ``record_started`` ("starting") frames still belong to the conversation: a device that was
        mid-wake when record_start reached it answers ``record_error: busy`` and its pre-roll must not be
        swallowed by a take that never began.
        """
        return self.state in ("recording", "ending")

    # ---- device events (called from the connection's receive loop) ---------------------------

    def on_started(self) -> None:
        if self.state == "starting":
            self.state = "recording"

    def on_frame(self, data: bytes) -> None:
        if not self.active or self.truncated:
            return
        room = MAX_BYTES - len(self.buf)
        if len(data) > room:
            # A device take longer than MAX_BYTES is cut and marked (max_ms <= 120 s normally never hits this).
            data, self.truncated = data[:room], True
            log.warning("device take %s cut at MAX_BYTES (%d B)", self.id, MAX_BYTES)
        self.buf += data
        self.ms = len(self.buf) * 1000 // BYTES_PER_SECOND
        if len(data) >= 2:
            samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            self.level_dbfs = float(20 * np.log10(max(rms, 1e-5)))

    def on_mark(self, msg: dict[str, Any]) -> None:
        word = str(msg.get("word") or "")[:32]
        t_ms = msg.get("t_ms")
        if not self.active or not word or isinstance(t_ms, bool) or not isinstance(t_ms, (int, float)):
            log.warning("ignoring record_mark %r on take %s", msg, self.id)
            return
        mark = {"word": word, "t_ms": max(0, int(t_ms))}
        self.marks.append(mark)
        log.info("device %s detected %r at %.2f s of take %s", self.device_id, word, mark["t_ms"] / 1000, self.id)

    def on_error(self, reason: str) -> None:
        if not self.active:
            return
        self.state, self.error = "error", str(reason)[:64] or "error"
        self.buf.clear()
        self._disarm()
        self.ended.set()

    # ---- lifecycle ---------------------------------------------------------------------------

    def arm_max_timer(self) -> None:
        self._disarm()
        self._timer = asyncio.create_task(self._expire())

    def _disarm(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()

    async def _expire(self) -> None:
        # The device should end by itself at max_ms (reason "max"); past a grace, keep what we have.
        await asyncio.sleep(self.max_ms / 1000 + MAX_GRACE_S)
        self._timer = None
        if self.active:
            log.warning("device %s did not end take %s at max_ms; finalising as partial", self.device_id, self.id)
            self.finalize("partial")

    def finalize(self, reason: str, reported_ms: int | None = None) -> dict[str, Any] | None:
        """Store the take (or record why it could not be stored); idempotent.

        Deliberately synchronous (like the browser upload's write): it must complete even when the
        connection is being torn down, where an await point could be cancelled. A take is at most
        MAX_BYTES, so the write is a few milliseconds.
        """
        if not self.active:
            return self.result
        self._disarm()
        if reported_ms is not None:
            # The device counts only real mic bytes. Whatever follows is not part of the take: the transport
            # padding of the last 640 B frame and, when record_end was late (the firmware retries it for up
            # to 2 s while already idle), audio of the next wake.
            keep = max(0, int(reported_ms)) * BYTES_PER_MS
            if len(self.buf) > keep:
                del self.buf[keep:]
            self.ms = len(self.buf) // BYTES_PER_MS
        else:
            self.ms = len(self.buf) // BYTES_PER_MS
        self.state = "storing"
        self.end_reason = reason if reason in RECORD_REASONS else "partial"
        self.level_dbfs = LEVEL_SILENT_DBFS
        try:
            wav = wav_from_pcm16(bytes(self.buf))
            self.buf.clear()
            self.result = self.store.write(
                self.session,
                self.label,
                self.prompt_id,
                self.take,
                wav,
                bump=True,
                user_agent=f"gaja-device/{self.fw or '?'}",
                extra={
                    "source": "device",
                    "device_id": self.device_id,
                    "fw": self.fw,
                    "marks": list(self.marks),
                    "device_detected": bool(self.marks),
                    "partial": self.end_reason == "partial",
                    **({"truncated": True} if self.truncated else {}),
                },
            )
            self.take = self.result["take"]
            self.state = "done"
        except Exception as exc:  # too short, wrong format, disk full: keep the take's marks, store nothing
            self.state, self.error = "error", str(getattr(exc, "detail", exc))[:200]
            log.warning("device take %s not stored: %s", self.id, self.error)
        self.ended.set()
        return self.result


async def begin(controller: DeviceController, *, store: SampleStore, session: str, label: str, prompt_id: str, take: int, max_ms: int) -> DeviceRecording:
    """Send record_start to an idle device and wait (briefly) for record_started / record_error."""
    recording = DeviceRecording(
        id=make_take_id(controller.info.id, prompt_id, take),
        device_id=controller.info.id,
        fw=controller.fw,
        session=session,
        label=label,
        prompt_id=prompt_id,
        take=take,
        max_ms=clamp_max_ms(max_ms),
        store=store,
    )
    controller.recording = recording
    try:
        await controller._outbox.send_json({"type": "record_start", "id": recording.id, "max_ms": recording.max_ms})
    except BaseException:
        # The socket is gone or the send timed out: nothing was started, leave no half-open take behind.
        recording.on_error("send")
        if controller.recording is recording:
            controller.recording = None
        raise
    recording.arm_max_timer()  # bounded even if the device never answers record_started/record_end
    try:
        await asyncio.wait_for(_ack(recording), START_ACK_S)
    except TimeoutError:
        # The firmware can fail to start a take without any record_error (record_started could not be sent).
        # Give up on it and tell the device to stop that id (it ignores an id it is not recording), so a
        # late start cannot leave it streaming into nothing.
        if recording.state == "starting":
            recording.on_error("not_started")
            try:
                await controller._outbox.send_json({"type": "record_stop", "id": recording.id})
            except ConnectionError:
                pass
    return recording


async def _ack(recording: DeviceRecording) -> None:
    """Resolve as soon as the device has answered record_started (or refused with record_error)."""
    while recording.state == "starting" and not recording.ended.is_set():
        await asyncio.sleep(0.02)
