"""close() must start nothing new: a turn it cancels still runs its finally, which primes."""

from __future__ import annotations

import asyncio
import json

import pytest

from gaja.llm.client import TextDelta
from gaja.pipeline import Session
from tests.conftest import FakeLLM


class RecordingOutbox:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.dead = False

    async def send_json(self, payload: dict) -> None:
        if self.dead:
            raise RuntimeError("Cannot call \"send\" once a close message has been sent.")
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        if self.dead:
            raise RuntimeError("Cannot call \"send\" once a close message has been sent.")
        self.sent.append({"bytes": len(data)})


class HangingPrimeLLM(FakeLLM):
    """prime_chat blocks; stream stays open after the first delta until released."""

    def __init__(self) -> None:
        super().__init__()
        self.prime_gate = asyncio.Event()
        self.prime_entered = asyncio.Event()
        self.stream_gate = asyncio.Event()

    async def prime_chat(self, messages) -> None:  # type: ignore[override]
        self.chat_primes.append(messages)
        self.prime_entered.set()
        await self.prime_gate.wait()

    async def stream(self, messages, *, allow_tools):  # type: ignore[override]
        yield TextDelta("Zaczynam odpowiadać")
        await self.stream_gate.wait()


@pytest.mark.asyncio
async def test_close_starts_nothing_from_a_cancelled_turns_finally(make_services) -> None:
    """close() cancels the turn, whose finally still calls _prime_next_turn(): no new task may start."""
    llm = HangingPrimeLLM()
    services = make_services(llm=llm, cfg={"llm": {"prime_after_turn": True}})
    out = RecordingOutbox()
    session = Session(services, out)
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    await asyncio.wait_for(llm.prime_entered.wait(), 2.0)  # the hello preload prime is now blocked
    llm.prime_entered.clear()

    await session.handle_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
    task = session._active
    assert task is not None
    await asyncio.sleep(0.05)  # the turn is mid-stream
    assert not task.done()

    await session.close()

    assert session._background == set(), "close() left a prime task started by the cancelled turn"
    primes_at_close = len(llm.chat_primes)
    llm.prime_gate.set()
    llm.stream_gate.set()
    await asyncio.sleep(0.1)
    assert len(llm.chat_primes) == primes_at_close


@pytest.mark.asyncio
async def test_announcements_return_early_after_close(make_services) -> None:
    """_announce, _speak_result and _sound_turn must not start a turn on a closed session.

    A caller blocked in _wait_exclusive_turn() while close() ran used to come back and start one
    more task, which wrote to the dead outbox (a timer announcement, a Hermes result or a sound).
    """
    import numpy as np

    services = make_services(llm=FakeLLM([[TextDelta("Wynik zadania.")]]))
    out = RecordingOutbox()
    session = Session(services, out)
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    await session.close()
    assert out.sent

    await session.announce("Test głośnika.")
    await session._speak_result("Wynik zadania.", "pl")
    await session.play_sound(np.zeros(320, dtype=np.int16).tobytes(), 16000)
    await asyncio.sleep(0.1)  # anything that slipped through would have landed by now

    assert not [m for m in out.sent if m.get("type") in ("utterance", "audio_start", "audio_end")], (
        "an announcement turn started after close()"
    )
    assert session._active is None and session._background == set()
