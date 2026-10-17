"""Device-side conversation flow of a Session (protocol v1.8): wake, verification, listen window, follow-up.

The controller owns only what is specific to a device: the wake handshake, the mic window (``listen_end`` /
``follow_up``) and the wake-phrase check. Turns themselves (VAD -> STT -> router -> LLM -> TTS, barge-in,
Hermes) are the unchanged Session machinery; Session calls the hooks below.

Mic window, seen from the server:
- ``wake``      -> mic open (the device streams pre-roll + live audio), verification pending;
- ``wake_accepted`` / ``wake_rejected{reason}`` -> verdict; rejected closes the mic without ``listen_end``;
- a device WITHOUT AEC gets ``listen_end`` when the user's utterance ended (it stops streaming while the reply
  plays); a device WITH AEC keeps streaming so the normal VAD barge-in works;
- after every user turn ``follow_up{ms}`` (the mic is open again for a follow-up, counted from the end of the
  reply's playback); when nobody speaks within the window the server sends ``listen_end``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gaja.config import DevicesConfig
from gaja.devices.audio import MIC_RATE, DeviceCaps, ResamplingOutbox, beep_pcm16, pick_out_rate
from gaja.devices.registry import DeviceInfo
from gaja.devices.wake import WakeMatcher

if TYPE_CHECKING:
    from gaja.devices.recording import DeviceRecording
    from gaja.pipeline import Session, _Turn

log = logging.getLogger(__name__)
# Log lines the firmware forwards while idle (protocol `log`): the journal keeps the device's side of an
# outage even when nobody had its web UI open.
remote_log = logging.getLogger("gaja.devices.remote")

ECHO_TAIL_S = 0.4
REPLY_RECOVERY_NOTICE = "Przepraszam, zerwało połączenie."
RECOVERY_DELAY_S = 0.05
REMOTE_LOG_MAX_CHARS = 400
REMOTE_LOG_PER_MINUTE = 300  # a boot backlog is ~100 lines; beyond this the device is spamming
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SECRET = re.compile(
    r"token|password|passwd|\bpsk\b|wifi_psk|api[_-]?key|bearer|authorization|secret|key\s*[:=]",
    re.IGNORECASE,
)


def clean_remote_log(msg: str) -> str:
    """One forwarded line, safe for the journal: no colour codes or control bytes, bounded, secrets never."""
    text = _CONTROL.sub(" ", _ANSI.sub("", msg)).strip()[:REMOTE_LOG_MAX_CHARS]
    return "[line withheld: may contain a secret]" if _SECRET.search(text) else text


def remote_log_level(level: object) -> int:
    """ESPHome levels (1 error, 2 warn, 3 info, 4 config, 5+ debug) -> logging; debug lines are kept at INFO:
    the firmware only forwards its own and the wake word detector's, and they are why they are sent."""
    if level == 1:
        return logging.ERROR
    if level == 2:
        return logging.WARNING
    return logging.INFO


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class LostReply:
    text: str
    user_text: str = ""
    language: str = "pl"
    expires_at: float = 0.0


WAKE_LOG: deque[dict[str, Any]] = deque(maxlen=50)


def get_wake_log() -> list[dict[str, Any]]:
    return list(WAKE_LOG)


@dataclass
class _Wake:
    word: str
    score: float
    skip_samples: int  # leading samples to drop (pre-roll that holds the device's own playback)
    verify: bool
    started: float = field(default_factory=time.perf_counter)
    buf: list[np.ndarray] = field(default_factory=list)
    buf_samples: int = 0
    speech_seen: bool = False
    utterance_ended: bool = False
    verify_task: asyncio.Task[None] | None = None
    decision: asyncio.Future[Decision] = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    transcript: str = ""


class DeviceController:
    def __init__(self, session: Session, cfg: DevicesConfig, outbox: ResamplingOutbox, info: DeviceInfo) -> None:
        self._s = session
        self._cfg = cfg
        self._outbox = outbox
        self.info = info
        self.caps = DeviceCaps()
        self.out_rate = MIC_RATE
        self.fw = ""
        self.mic_open = False
        self.wake: _Wake | None = None
        self.recording: DeviceRecording | None = None  # protocol v1.10 (gaja/devices/recording.py)
        self.stats: Counter[str] = Counter()
        self._timer: asyncio.Task[None] | None = None
        self._recovery_task: asyncio.Task[None] | None = None
        self._log_window = (0.0, 0)  # (minute start, lines logged in it)
        self._matcher = WakeMatcher(
            accept_bare_name=cfg.accept_bare_name,
            extra_names=cfg.extra_wake_names,
            extra_greetings=cfg.extra_wake_greetings,
        )

    # ---- hello -----------------------------------------------------------------------------

    def configure(self, caps: DeviceCaps) -> dict[str, Any]:
        self.caps = caps
        self.out_rate = pick_out_rate(caps.out_rates, self._cfg.preferred_out_rate)
        self._outbox.configure(self.out_rate)
        hello = self._s._hello
        self.fw = (hello.fw if hello is not None else "") or ""
        return {
            "id": self.info.id,
            "aec": caps.aec,
            "out_rate": self.out_rate,
            "verify_wake": self._cfg.verify_wake,
            "verify_ms": self._cfg.verify_ms,
            "follow_up_ms": self._cfg.follow_up_ms,
            "ack": self._cfg.ack,
            "logs": True,  # `log` messages are accepted (firmware >= 0.2.4 forwards its log lines)
        }

    async def close(self) -> None:
        recording, self.recording = self.recording, None
        if recording is not None and recording.active:
            # The connection is gone: keep what was received, marked "partial" (protocol v1.10).
            recording.finalize("partial")
        self._disarm()
        recovery, self._recovery_task = self._recovery_task, None
        tasks = [t for t in (self.wake.verify_task if self.wake else None, recovery) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ---- lost reply recovery ---------------------------------------------------------------

    def save_lost_reply(self, text: str, *, user_text: str = "", language: str = "pl") -> None:
        """Remember an unfinished user-turn reply so it can be spoken once if the device reconnects soon."""
        ttl = self._cfg.reply_recovery_s
        text = text.strip()
        if ttl <= 0 or not text:
            return
        self._s._svc.lost_replies[self.info.id] = LostReply(
            text=text,
            user_text=user_text,
            language=language,
            expires_at=time.monotonic() + ttl,
        )
        log.info("device %s disconnected mid-reply; holding reply for %.0fs recovery window", self.info.id, ttl)

    def take_lost_reply(self) -> LostReply | None:
        """Consume a pending lost reply for this device if still within the recovery window."""
        lost = self._s._svc.lost_replies.pop(self.info.id, None)
        if lost is None:
            return None
        if self._cfg.reply_recovery_s <= 0 or time.monotonic() > lost.expires_at:
            return None
        return lost

    def clear_lost_reply(self) -> None:
        """Drop any pending lost reply (e.g. the user started a new turn first)."""
        self._s._svc.lost_replies.pop(self.info.id, None)

    def schedule_recovery(self) -> None:
        """If this device has an unexpired lost reply, schedule speaking it after hello."""
        lost = self._s._svc.lost_replies.get(self.info.id)
        if lost is None:
            return
        if self._cfg.reply_recovery_s <= 0 or time.monotonic() > lost.expires_at:
            self._s._svc.lost_replies.pop(self.info.id, None)
            return
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self._recover_reply())

    async def cancel_recovery(self) -> None:
        """Abort a pending reply recovery because the user started a new turn first."""
        self.clear_lost_reply()
        task, self._recovery_task = self._recovery_task, None
        if task is not None and task is not asyncio.current_task() and self._s._active is not task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _recover_reply(self) -> None:
        try:
            await asyncio.sleep(RECOVERY_DELAY_S)
            if self.busy or self.recording_active or self._s._gone:
                self.clear_lost_reply()
                return
            lost = self.take_lost_reply()
            if lost is None:
                return
            self._s._active = asyncio.current_task()
            await self._s._recover_turn(lost)
        except asyncio.CancelledError:
            raise
        finally:
            if self._recovery_task is asyncio.current_task():
                self._recovery_task = None

    # ---- device messages -------------------------------------------------------------------

    def on_log(self, msg: dict[str, Any]) -> None:
        text = msg.get("msg")
        if not isinstance(text, str) or not text.strip():
            return
        now = time.monotonic()
        start, count = self._log_window
        if now - start >= 60.0:
            if count > REMOTE_LOG_PER_MINUTE:
                remote_log.warning("[%s] %d log lines suppressed", self.info.id, count - REMOTE_LOG_PER_MINUTE)
            start, count = now, 0
        count += 1
        self._log_window = (start, count)
        if count <= REMOTE_LOG_PER_MINUTE:
            remote_log.log(remote_log_level(msg.get("level")), "[%s] %s", self.info.id, clean_remote_log(text))

    async def on_wake(self, msg: dict[str, Any]) -> None:
        s = self._s
        recording = self.recording
        if recording is not None and recording.streaming:
            # The firmware never sends `wake` while it records (a detection becomes record_mark), so a wake
            # means the take already ended on the device and its record_end is late or lost (it retries it for
            # up to 2 s). Close the take now, with exactly what arrived before this wake, and treat the wake
            # as a normal conversation wake. A later record_end for it is ignored. (v1.10)
            log.info("device %s woke during take %s: closing the take (record_end is late)", self.info.id, recording.id)
            recording.finalize("stop" if recording.state == "ending" else "partial")
        word = str(msg.get("word") or "hej_gaja")[:32]
        score = msg.get("score", 1.0)
        preroll_ms = msg.get("preroll_ms", self.caps.preroll_ms)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or isinstance(preroll_ms, bool) or not isinstance(preroll_ms, (int, float)):
            await s._error("wake: score and preroll_ms must be numbers")
            return
        await self.cancel_recovery()
        score = min(1.0, max(0.0, float(score)))
        preroll_ms = int(min(max(0, preroll_ms), self.caps.preroll_ms))
        busy = s._active is not None or s._spec is not None or time.perf_counter() < s._playback_until
        if busy and score < self._cfg.barge_in_min_score:
            self._count("wake_rejected", "low_score")
            await s._out.send_json({"type": "wake_rejected", "reason": "low_score"})
            return
        await self._drop_wake()
        if busy:
            # Barge-in by wake word: same path as VAD barge-in; Hermes tasks keep running.
            if not await s._interrupt_speech(reason="barge_in"):
                await s._set_state("listening", "barge_in")
        assert s._endpointer is not None
        s._endpointer.reset()
        self.mic_open = True
        own_voice = busy and not self.caps.aec  # the pre-roll holds the device's own playback: drop it, skip verification
        wake = _Wake(
            word=word,
            score=score,
            skip_samples=preroll_ms * MIC_RATE // 1000 if own_voice else 0,
            verify=self._cfg.verify_wake and not own_voice,
        )
        self.wake = wake
        self.stats["wake"] += 1
        log.info("device %s wake word=%s score=%.2f busy=%s verify=%s", self.info.id, word, score, busy, wake.verify)
        if wake.verify:
            self._arm(self._cfg.wake_timeout_s, "wake")
        else:
            await self._decide(wake, True, "unverified")

    def filter_audio(self, samples: np.ndarray) -> np.ndarray | None:
        """Mic samples from the device; None = not wanted now (mic closed or pre-roll to skip)."""
        if not self.mic_open:
            return None
        wake = self.wake
        if wake is None:
            return samples
        if wake.skip_samples:
            if samples.size <= wake.skip_samples:
                wake.skip_samples -= samples.size
                return None
            samples = samples[wake.skip_samples :]
            wake.skip_samples = 0
        if wake.verify and not wake.decision.done() and wake.verify_task is None and not wake.utterance_ended:
            wake.buf.append(samples)
            wake.buf_samples += samples.size
            if wake.buf_samples >= self._cfg.verify_ms * MIC_RATE // 1000:
                wake.verify_task = asyncio.create_task(self._verify(wake))
        return samples

    # ---- take recording (protocol v1.10) ----------------------------------------------------

    @property
    def recording_active(self) -> bool:
        return self.recording is not None and self.recording.active

    @property
    def recording_streaming(self) -> bool:
        """A take is accepted by the device and its mic frames are flowing (they are not conversation audio)."""
        return self.recording is not None and self.recording.streaming

    @property
    def busy(self) -> bool:
        """Not idle: a turn, a pending wake verification, or the echo tail of our own reply."""
        s = self._s
        return s._active is not None or s._spec is not None or self.wake is not None or time.perf_counter() < s._playback_until

    async def prepare_for_recording(self) -> None:
        """Close an open follow-up mic window (no turn in progress) so the device is idle."""
        await self.cancel_recovery()
        if not self.busy and self.mic_open:
            await self.listen_end()

    async def on_record_event(self, msg: dict[str, Any]) -> None:
        """record_started / record_mark / record_end / record_error from the device."""
        recording = self.recording
        take_id = msg.get("id")
        if recording is None or take_id != recording.id:
            log.warning("device %s record event for unknown take %r", self.info.id, take_id)
            return
        kind = msg.get("type")
        if kind == "record_started":
            recording.on_started()
        elif kind == "record_mark":
            recording.on_mark(msg)
        elif kind == "record_end":
            reason = msg.get("reason") if msg.get("reason") in ("stop", "max") else "partial"
            ms = msg.get("ms")
            reported = int(ms) if isinstance(ms, (int, float)) and not isinstance(ms, bool) else None
            recording.finalize(reason, reported)
        elif kind == "record_error":
            recording.on_error(str(msg.get("reason") or "error"))
        else:
            log.warning("device %s unknown record event %r", self.info.id, kind)

    def on_record_frame(self, data: bytes) -> None:
        if self.recording is not None:
            self.recording.on_frame(data)

    async def record_stop(self) -> None:
        """Ask the device to stop the active take (it answers record_end)."""
        recording = self.recording
        if recording is None or not recording.active:
            return
        if recording.state == "starting":
            # The device never confirmed the take, so it is not recording (the firmware can fail to start one
            # without a record_error): drop it now. "ending" here would make us swallow frames and wakes
            # until the max_ms watchdog. The stop is still sent; the firmware ignores an id it is not recording.
            recording.on_error("not_started")
        else:
            recording.state = "ending"
        await self._outbox.send_json({"type": "record_stop", "id": recording.id})

    # ---- VAD hooks -------------------------------------------------------------------------

    def on_speech_start(self) -> None:
        self.clear_lost_reply()
        task, self._recovery_task = self._recovery_task, None
        if task is not None and self._s._active is not task and not task.done():
            task.cancel()
        if self.wake is not None:
            self.wake.speech_seen = True
        self._disarm()

    async def on_speech_end(self) -> None:
        if self.wake is not None:
            self.wake.utterance_ended = True
            if self.wake.verify_task is not None and not self.wake.decision.done():
                self.wake.verify_task.cancel()
                self.wake.verify_task = None
        if not self.caps.aec:
            await self.listen_end()

    def blocks_speculation(self) -> bool:
        return self.wake is not None

    def wake_hotwords(self) -> str | None:
        """Hotwords for an utterance whose transcript still decides a pending wake, else None.

        A short "Hej Gaja ..." ends before verify_ms, so the turn's own transcript is the verdict. Decoded
        with the vocab hotwords Whisper heard "Daja"/"Jaja" and the live accept rate was 59 %; with the
        verify hotwords instead (not added to them) it was 96 % on the same takes.
        """
        wake = self.wake
        if wake is None or wake.decision.done():
            return None
        return self._cfg.verify_hotwords or None

    # ---- wake verdict ----------------------------------------------------------------------

    async def _verify(self, wake: _Wake) -> None:
        s = self._s
        try:
            await asyncio.sleep(0)
            audio = np.concatenate(wake.buf)[: self._cfg.verify_ms * MIC_RATE // 1000]
            wake.buf.clear()
            assert s._endpointer is not None and s._hello is not None
            if not (wake.speech_seen or s._endpointer.in_speech):
                await self._decide(wake, False, "no_speech")
                return
            transcript = await s._svc.stt.transcribe(audio, None if s._hello.lang == "auto" else s._hello.lang, hotwords=self._cfg.verify_hotwords)
            wake.transcript = transcript.text
            match = self._matcher.match(transcript.text)
            await self._decide(wake, match.matched, "ok" if match.matched else "no_phrase")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("wake verification failed")
            await self._decide(wake, False, "stt_error")

    async def _decide(self, wake: _Wake, accepted: bool, reason: str) -> None:
        if wake.decision.done() or self.wake is not wake:
            return
        wake.decision.set_result(Decision(accepted, reason))
        WAKE_LOG.append({
            "ts": time.time(),
            "device_id": self.info.id,
            "word": wake.word,
            "score": wake.score,
            "accepted": accepted,
            "reason": reason,
            "transcript": getattr(wake, "transcript", ""),
        })
        try:
            from gaja.api.devices import record_device_activity
            record_device_activity(self.info.id, is_wake=True)
        except Exception:
            pass
        out = self._s._out
        if accepted:
            self._count("wake_accepted")
            await out.send_json({"type": "wake_accepted"})
            if self.mic_open and not wake.speech_seen:
                self._arm(self._cfg.listen_timeout_s, "listen")
            elif wake.speech_seen:
                self._disarm()
            return
        self._count("wake_rejected", reason)
        self.mic_open = False
        self._disarm()
        assert self._s._endpointer is not None
        self._s._endpointer.reset()
        await out.send_json({"type": "wake_rejected", "reason": reason})

    async def resolve(self, text: str) -> tuple[str, bool] | None:
        """Wake handling for the transcript of the first utterance after a wake.

        Returns (text for the LLM, wake_only), or None when the wake was rejected (already reported).
        Without a pending wake (a follow-up window turn) the text passes unchanged.
        """
        wake = self.wake
        if wake is None:
            return text, False
        if not wake.decision.done():
            if wake.verify_task is not None:
                await asyncio.shield(wake.decision)
            else:
                wake.transcript = text
                match = self._matcher.match(text)
                await self._decide(wake, match.matched, "ok" if match.matched else "no_phrase")
        decision = wake.decision.result()
        if self.wake is wake:
            self.wake = None
        if not decision.accepted:
            return None
        match = self._matcher.match(text)
        if match.matched:
            return match.rest, not match.rest
        return text, not text.strip()

    # ---- replies and windows ---------------------------------------------------------------

    async def ack(self, turn: _Turn) -> None:
        """Answer a bare wake phrase."""
        s = self._s
        mode = self._cfg.ack
        if mode == "spoken":
            text = self._cfg.ack_text.get(turn.language) or self._cfg.ack_text.get("pl", "Słucham?")
            await s._say(turn, text)
        elif mode == "earcon":
            data = beep_pcm16(MIC_RATE)
            s._utterances += 1
            await turn.out.send_json({"type": "audio_start", "id": s._utterances, "sample_rate": MIC_RATE})
            send_started = time.perf_counter()  # before the send: backpressure must not stretch the playback window
            await turn.out.send_bytes(data)
            await turn.out.send_json({"type": "audio_end", "id": s._utterances})
            duration = len(data) / 2 / MIC_RATE
            s._playback_end = max(send_started, getattr(s, "_playback_end", 0.0)) + duration
            s._playback_until = s._playback_end + ECHO_TAIL_S

    async def turn_done(self) -> None:
        """A user turn finished: keep the mic open for a follow-up, counted from the end of playback."""
        s = self._s
        if self._cfg.follow_up_ms <= 0:
            await self.listen_end()
            return
        remaining = max(0.0, s._playback_until - time.perf_counter())
        if not self.mic_open and s._endpointer is not None:
            s._endpointer.reset()
        self.mic_open = True
        await s._out.send_json({"type": "follow_up", "ms": self._cfg.follow_up_ms})
        self._arm(self._cfg.follow_up_ms / 1000 + remaining + self._cfg.listen_grace_s, "listen")

    async def listen_end(self) -> None:
        self._disarm()
        if not self.mic_open:
            return
        self.mic_open = False
        await self._s._out.send_json({"type": "listen_end"})

    # ---- internals -------------------------------------------------------------------------

    async def _drop_wake(self) -> None:
        wake, self.wake = self.wake, None
        if wake is None:
            return
        if wake.verify_task is not None:
            wake.verify_task.cancel()
            await asyncio.gather(wake.verify_task, return_exceptions=True)
        if not wake.decision.done():
            wake.decision.set_result(Decision(False, "superseded"))

    def _count(self, name: str, reason: str | None = None) -> None:
        self.stats[name] += 1
        self._s._svc.metrics.count(name)
        if reason is not None:
            self.stats[f"{name}:{reason}"] += 1
            self._s._svc.metrics.count(f"{name}:{reason}")

    def _arm(self, seconds: float, kind: str) -> None:
        self._disarm()
        self._timer = asyncio.create_task(self._expire(seconds, kind))

    def _disarm(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()

    async def _expire(self, seconds: float, kind: str) -> None:
        await asyncio.sleep(seconds)
        s = self._s
        self._timer = None
        if kind == "wake":
            wake = self.wake
            if wake is not None and not wake.decision.done():
                await self._decide(wake, False, "timeout")
            return
        endpointer = s._endpointer
        if s._active is None and s._spec is None and not (endpointer and endpointer.in_speech):
            await self.listen_end()
