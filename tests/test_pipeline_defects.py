"""Regression tests for the barge-in / turn-lifecycle defects fixed on t3/pipeline-devices-fixes."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
import pytest

from gaja.audio.vad import VadEvent
from gaja.devices.controller import REPLY_RECOVERY_NOTICE
from gaja.devices.registry import DeviceInfo
from gaja.llm.client import TextDelta
from gaja.llm.parse import ToolCall
from gaja.pipeline import Session
from tests.conftest import FakeLLM, FakeSTT, FakeTTS

HELLO = {
    "type": "hello", "client": "device", "device_id": "atom-1", "lang": "pl", "fw": "0.3.8",
    "caps": {"aec": False, "out_rates": [16000], "preroll_ms": 0},
}


class RecordingOutbox:
    """Collects every server message; ``block_bytes`` can stall a device's paced downlink."""

    def __init__(self, *, block_bytes: asyncio.Event | None = None) -> None:
        self.sent: list[dict] = []
        self.block_bytes = block_bytes
        self.dead = False

    async def send_json(self, payload: dict) -> None:
        if self.dead:
            raise RuntimeError("Cannot call \"send\" once a close message has been sent.")
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        if self.block_bytes is not None:
            await self.block_bytes.wait()
        if self.dead:
            raise RuntimeError("Cannot call \"send\" once a close message has been sent.")
        self.sent.append({"bytes": len(data)})

    def of_type(self, kind: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == kind]


class LongTTS(FakeTTS):
    """A few seconds of audio per sentence: the paced device outbox keeps a real playback tail."""

    def __init__(self, seconds: float = 3.0) -> None:
        self.seconds = seconds

    async def synthesize(self, text: str, style, language: str) -> np.ndarray:  # type: ignore[override]
        return np.full(int(24000 * self.seconds), 0.1, dtype=np.float32)


class BrokenTTS(FakeTTS):
    async def synthesize(self, text: str, style, language: str) -> np.ndarray:  # type: ignore[override]
        raise RuntimeError("tts backend down")


class FlakySTT(FakeSTT):
    """Raises on the first call (the turn fails), then transcribes ``text`` like FakeSTT."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.boom = True

    async def transcribe(self, *a, **k):
        if self.boom:
            self.boom = False
            raise RuntimeError("stt crashed")
        return await super().transcribe(*a, **k)


def make_device_session(make_services, *, llm=None, tts=None, cfg=None):
    services = make_services(llm=llm, cfg=cfg)
    if tts is not None:
        # The pool builds/warms engines on first use; pre-seed its cache with the test engine.
        services.tts._loaded["fake"] = tts  # noqa: SLF001
    out = RecordingOutbox()
    session = Session(services, out, device=DeviceInfo(id="atom-1"))
    return services, session, out


async def finish_reply_with_a_tail(session, out, text: str = "opowiedz coś") -> None:
    """Run a text turn to its end; the device is still draining the paced audio afterwards."""
    await session.handle_text(json.dumps({"type": "text", "text": text}))
    task = session._active
    assert task is not None
    await asyncio.wait_for(asyncio.gather(task), 20)
    assert session._active is None
    assert time.perf_counter() < session._playback_until, "no playback tail left: test would not exercise the bug"


@pytest.mark.asyncio
async def test_vad_barge_in_during_the_playback_tail_sends_playback_stop(make_services) -> None:
    """A VAD barge-in after the turn task ended still stops the audio the device holds buffered.

    The turn finishes when its last frame is *sent*; the device plays up to ``downlink_lead_s``
    later. Without ``playback_stop`` it keeps talking over the user for that whole tail.
    """
    _, session, out = make_device_session(
        make_services,
        llm=FakeLLM([[TextDelta("To jest długa odpowiedź na pytanie.")]]),
        tts=LongTTS(3.0),
    )
    await session.handle_text(json.dumps(HELLO))
    await finish_reply_with_a_tail(session, out)
    assert out.of_type("playback_stop") == []

    await session._on_vad(VadEvent("speech_start"))

    assert len(out.of_type("playback_stop")) == 1
    await session.close()


@pytest.mark.asyncio
async def test_wake_barge_in_during_the_playback_tail_sends_playback_stop(make_services) -> None:
    """The wake-word barge-in path (DeviceController.on_wake) stops the buffered tail as well."""
    _, session, out = make_device_session(
        make_services,
        llm=FakeLLM([[TextDelta("To jest długa odpowiedź na pytanie.")]]),
        tts=LongTTS(3.0),
    )
    await session.handle_text(json.dumps(HELLO))
    await finish_reply_with_a_tail(session, out)
    assert out.of_type("playback_stop") == []

    await session.handle_text(json.dumps({"type": "wake", "word": "hej_gaja", "score": 1.0, "preroll_ms": 0}))

    assert len(out.of_type("playback_stop")) == 1
    assert out.of_type("wake_accepted")
    await session.close()


@pytest.mark.asyncio
async def test_cancelled_speculation_commit_does_not_orphan_the_turn(make_services) -> None:
    """A commit cancelled while flushing the gate (device supersede) must not strand the turn.

    _commit_speculation clears self._spec before ``await spec.gate.open()``. A cancellation there
    used to leave the speculative turn waiting on spec.committed forever: close() could no longer
    reach it (self._spec is None) and the task lived on after the session was gone.
    """
    gate = asyncio.Event()  # holds the device outbox's first byte send
    out = RecordingOutbox(block_bytes=gate)
    handoff = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    services = make_services(llm=FakeLLM([[handoff]]))
    session = Session(services, out, device=DeviceInfo(id="atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session._start_speculation(np.zeros(16000, dtype=np.float32))
    spec = session._spec
    assert spec is not None and spec.task is not None
    await asyncio.sleep(0.05)  # the turn produced its held ack and waits for the commit

    commit = asyncio.create_task(session._commit_speculation())
    await asyncio.sleep(0.05)  # gate.open is now blocked in the paced byte send
    assert session._spec is None
    commit.cancel()
    await asyncio.gather(commit, return_exceptions=True)

    assert spec.task.done(), "the speculative turn was left waiting on spec.committed forever"
    await session.close()


@pytest.mark.asyncio
async def test_tts_failure_ends_the_turn_with_an_error_and_a_log_line(
    make_services, tmp_path, caplog
) -> None:
    """A TTS exception must not kill the turn silently.

    Before: the task ended with an unretrieved RuntimeError, no ``error`` event reached the
    client, and no turn was written to the metrics log -- the session just went quiet.
    """
    services = make_services(llm=FakeLLM([[TextDelta("To jest odpowiedź.")]]))
    services.tts._loaded["fake"] = BrokenTTS()  # noqa: SLF001
    out = RecordingOutbox()
    session = Session(services, out)
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    before = len(out.sent)

    with caplog.at_level(logging.ERROR):
        await session.handle_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        task = session._active
        assert task is not None
        done, pending = await asyncio.wait({task}, timeout=5.0)
        assert not pending, "the turn never finished"

    assert task.exception() is None, "the TTS exception escaped the turn task"
    sent = out.sent[before:]
    errors = [m for m in sent if m.get("type") == "error"]
    assert errors and errors[-1]["message"] == "turn failed (RuntimeError)"  # no backend detail leaks
    assert "turn 1 failed" in caplog.text
    assert any(record.exc_info for record in caplog.records)
    turn_log = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(turn_log) == 1, "the failed turn left no metrics/turn-log line"
    await session.close()


@pytest.mark.asyncio
async def test_cancelled_commit_reaches_the_reconnected_device_exactly_once(make_services) -> None:
    """A commit cancelled by supersede must still hold the reply for the reconnected device.

    The real supersede order (gaja/devices/listener.py): the old session is marked gone and its
    ``record_lost_reply()`` runs, the NEW session registers (a later ``_opened_at``), and only then
    does the old handler's cancellation land in the blocked commit. The reply must be held exactly
    once -- at supersede time, while the old session still owns the device -- and the reconnected
    device's recovery must speak it once, consuming it.
    """
    gate = asyncio.Event()
    out = RecordingOutbox(block_bytes=gate)
    services = make_services(llm=FakeLLM([[TextDelta("Dzisiaj jest słonecznie.")]]))
    session = Session(services, out, device=DeviceInfo(id="atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session._start_speculation(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.3)  # the turn finished; the held reply waits for the commit
    commit = asyncio.create_task(session._commit_speculation())
    await asyncio.sleep(0.1)  # the gate is blocked in the paced byte send

    # listener.supersede(), in its real order: gone + record_lost_reply, new Session, then cancel.
    session._gone = True
    session.record_lost_reply()
    newer_out = RecordingOutbox()
    newer = Session(services, newer_out, device=DeviceInfo(id="atom-1"))
    commit.cancel()
    await asyncio.gather(commit, return_exceptions=True)
    await session.close()  # the old handler's finally must not clear or double-record the reply

    lost = services.lost_replies.get("atom-1")
    assert lost is not None and lost.text == "Dzisiaj jest słonecznie."
    assert lost.user_text == "ile to dwa plus dwa"

    # The reconnected device hears the reply exactly once: its hello arms the recovery, which
    # consumes the held reply (nothing is left for a later reconnect).
    await newer.handle_text(json.dumps(HELLO))
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and (len(newer_out.of_type("utterance")) < 2 or newer._active is not None):
        await asyncio.sleep(0.05)
    spoken = [m["text"] for m in newer_out.of_type("utterance")]
    assert spoken == [REPLY_RECOVERY_NOTICE, "Dzisiaj jest słonecznie."]
    assert "atom-1" not in services.lost_replies
    await newer.close()


@pytest.mark.asyncio
async def test_a_recovered_cancelled_commit_reply_is_in_history_once(make_services) -> None:
    """F2: a reply held by a cancelled commit is already complete in the history.

    The cancelled commit wrote the full exchange to the history before the gate stalled. Recovery
    then spoke the notice and the reply again, and _recover_turn re-added both lines: the LLM saw
    the same exchange twice. Whichever path held the reply, after recovery the user line and the
    reply must each be in the history exactly once.
    """
    gate = asyncio.Event()
    out = RecordingOutbox(block_bytes=gate)
    services = make_services(llm=FakeLLM([[TextDelta("Dzisiaj jest słonecznie.")]]))
    session = Session(services, out, device=DeviceInfo(id="atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session._start_speculation(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.3)  # the held reply is complete and waits for the commit
    commit = asyncio.create_task(session._commit_speculation())
    await asyncio.sleep(0.1)  # the gate is blocked in the paced byte send
    commit.cancel()
    await asyncio.gather(commit, return_exceptions=True)
    await session.close()
    lost = session._device.take_lost_reply()
    assert lost is not None and lost.text == "Dzisiaj jest słonecznie."

    new = Session(services, RecordingOutbox(), device=DeviceInfo(id="atom-1"))
    try:
        await new.handle_text(json.dumps(HELLO))
        await new._recover_turn(lost)
        assert new._history.context() == [
            {"role": "user", "content": "ile to dwa plus dwa"},
            {"role": "assistant", "content": "Dzisiaj jest słonecznie."},
        ]
    finally:
        await new.close()


@pytest.mark.asyncio
async def test_a_failed_audio_turn_drops_its_pending_wake_and_leaves_the_device_idle(make_services) -> None:
    """F3: an STT/router/TTS exception must not leave a pending wake capturing the next utterance.

    The failure happened before resolve() consumed the wake, so the follow-up window that opened
    afterwards treated the next, unrelated utterance as the post-wake one: the garbage follow-up
    filter was skipped (a "PU." went to the LLM) and busy stayed True until the window expired.
    After the failed turn the device must be idle: no pending wake, busy False, and the reopened
    window is a plain follow-up window.
    """
    llm = FakeLLM([[TextDelta("ok")]])
    services = make_services(llm=llm, stt=FlakySTT("PU."), cfg={"devices": {"verify_wake": False}})
    out = RecordingOutbox()
    session = Session(services, out, device=DeviceInfo(id="atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session.handle_text(json.dumps({"type": "wake", "word": "hej_gaja", "score": 1.0, "preroll_ms": 0}))
    assert session._device is not None and session._device.wake is not None
    await session._start_user_turn(audio=np.zeros(16000, dtype=np.float32))
    task = session._active
    assert task is not None
    await asyncio.wait_for(asyncio.gather(task), 10)

    assert [m["message"] for m in out.of_type("error")] == ["turn failed (RuntimeError)"]
    assert session._device.wake is None, "the pending wake survived the failed turn"
    assert not session._device.busy
    assert out.of_type("follow_up"), "the follow-up window did not reopen after the failed turn"

    # The next utterance is a plain follow-up (the wake is gone): the low-content transcript is
    # filtered out instead of being routed to the LLM as the post-wake utterance.
    await session._start_user_turn(audio=np.zeros(16000, dtype=np.float32))
    task = session._active
    assert task is not None
    await asyncio.wait_for(asyncio.gather(task), 10)
    assert llm.calls == [], "a garbage follow-up was routed to the LLM as the post-wake utterance"
    assert len(out.of_type("error")) == 1
    assert not session._device.busy
    await session.close()


@pytest.mark.asyncio
async def test_cancelled_speculation_commit_holds_the_reply_for_recovery(make_services) -> None:
    """A commit cancelled mid-flush already wrote the reply to history: recovery must then speak it.

    The ConnectionError branch holds the reply; the cancel branch (device supersede, session closing)
    used to leave spec.committed unset, so record_lost_reply skipped the turn and the reconnected
    device never heard a reply its history claims was given.
    """
    gate = asyncio.Event()
    out = RecordingOutbox(block_bytes=gate)
    services = make_services(llm=FakeLLM([[TextDelta("Dzisiaj jest słonecznie.")]]))
    session = Session(services, out, device=DeviceInfo(id="atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session._start_speculation(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.3)  # the held reply is complete and waits for the commit
    commit = asyncio.create_task(session._commit_speculation())
    await asyncio.sleep(0.1)  # gate.open is blocked in the paced byte send
    commit.cancel()
    await asyncio.gather(commit, return_exceptions=True)
    assert any(m["role"] == "assistant" for m in session._history.messages())

    await session.close()
    lost = session._device.take_lost_reply()
    assert lost is not None and lost.text == "Dzisiaj jest słonecznie."
