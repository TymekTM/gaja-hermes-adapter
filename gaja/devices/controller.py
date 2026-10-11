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
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gaja.config import DevicesConfig
from gaja.devices.audio import MIC_RATE, DeviceCaps, ResamplingOutbox, beep_pcm16, pick_out_rate
from gaja.devices.registry import DeviceInfo
from gaja.devices.wake import WakeMatcher

if TYPE_CHECKING:
    from gaja.pipeline import Session, _Turn

log = logging.getLogger(__name__)

ECHO_TAIL_S = 0.4


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str


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


class DeviceController:
    def __init__(self, session: Session, cfg: DevicesConfig, outbox: ResamplingOutbox, info: DeviceInfo) -> None:
        self._s = session
        self._cfg = cfg
        self._outbox = outbox
        self.info = info
        self.caps = DeviceCaps()
        self.out_rate = MIC_RATE
        self.mic_open = False
        self.wake: _Wake | None = None
        self.stats: Counter[str] = Counter()
        self._timer: asyncio.Task[None] | None = None
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
        return {
            "id": self.info.id,
            "aec": caps.aec,
            "out_rate": self.out_rate,
            "verify_wake": self._cfg.verify_wake,
            "verify_ms": self._cfg.verify_ms,
            "follow_up_ms": self._cfg.follow_up_ms,
            "ack": self._cfg.ack,
        }

    async def close(self) -> None:
        self._disarm()
        tasks = [t for t in (self.wake.verify_task if self.wake else None,) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ---- device messages -------------------------------------------------------------------

    async def on_wake(self, msg: dict[str, Any]) -> None:
        s = self._s
        word = str(msg.get("word") or "hej_gaja")[:32]
        score = msg.get("score", 1.0)
        preroll_ms = msg.get("preroll_ms", self.caps.preroll_ms)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or isinstance(preroll_ms, bool) or not isinstance(preroll_ms, (int, float)):
            await s._error("wake: score and preroll_ms must be numbers")
            return
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

    # ---- VAD hooks -------------------------------------------------------------------------

    def on_speech_start(self) -> None:
        if self.wake is not None:
            self.wake.speech_seen = True
        self._disarm()

    async def on_speech_end(self) -> None:
        if self.wake is not None:
            self.wake.utterance_ended = True
        if not self.caps.aec:
            await self.listen_end()

    def blocks_speculation(self) -> bool:
        return self.wake is not None

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
            transcript = await s._svc.stt.transcribe(audio, None if s._hello.lang == "auto" else s._hello.lang)
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
            await turn.out.send_bytes(data)
            await turn.out.send_json({"type": "audio_end", "id": s._utterances})
            s._playback_until = max(time.perf_counter(), s._playback_until) + len(data) / 2 / MIC_RATE + ECHO_TAIL_S

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

