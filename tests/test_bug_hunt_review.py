"""Repro tests for bugs found in the read-only bug-hunt review (branch
t3/bug-hunt-async-device-review).  Each test asserts the CORRECT behaviour and is
marked ``xfail(strict=True)``: it fails on current ``main`` because of the bug and
will start failing as XPASS (strict) once the bug is fixed.

Run: flock /tmp/gaja-pytest.lock uv run pytest -q tests/test_bug_hunt_review.py
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
import websockets
from fastapi.testclient import TestClient

from gaja.devices import listener as listener_module
from gaja.devices.listener import DeviceListener, create_device_app
from gaja.devices.registry import DeviceRegistry
from gaja.llm.client import TextDelta
from gaja.pipeline import Session, _HermesJob
from tests.conftest import FakeLLM
from tests.test_ws_protocol import is_listening, read, read_until, silence_frames, speech_frames

TOKEN = "d" * 40
HELLO: dict[str, Any] = {
    "type": "hello", "client": "device", "device_id": "atom-1", "token": TOKEN, "lang": "pl",
    "fw": "0.1.0", "caps": {"aec": False, "out_rates": [16000], "preroll_ms": 1500},
}


@pytest.fixture(autouse=True)
def _no_auth_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 0.0)


@pytest.fixture
def registry(tmp_path) -> DeviceRegistry:
    path = tmp_path / "devices.toml"
    path.write_text(f'[devices.atom-1]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    path.chmod(0o600)
    return DeviceRegistry(path)


class SendAfterCloseOutbox:
    """Behaves like a starlette/uvicorn WebSocket after the peer vanished: sends raise
    RuntimeError (TestClient) / websockets.ConnectionClosedOK (uvicorn) -- neither is a
    ConnectionError or WebSocketDisconnect, which is what gaja.pipeline catches."""

    def __init__(self) -> None:
        self.count = 0

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.count += 1
        if self.count > 3:
            raise RuntimeError('Cannot call "send" once a close message has been sent.')

    async def send_bytes(self, data: bytes) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')


class DeadWhenToldOutbox:
    """A working outbox until ``dead`` is set -- then sends raise like a socket the peer closed."""

    def __init__(self) -> None:
        self.dead = False
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.dead:
            raise RuntimeError('Cannot call "send" once a close message has been sent.')
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        if self.dead:
            raise RuntimeError('Cannot call "send" once a close message has been sent.')
        self.sent.append({"bytes": len(data)})


@pytest.mark.asyncio
async def test_web_turn_survives_send_after_peer_left(make_services) -> None:
    """A web client that disconnects mid-turn must end the turn quietly (_gone set, no
    unhandled task exception), exactly as it does when the send raises ConnectionError."""
    llm = FakeLLM([[TextDelta("Dzień dobry. To jest odpowiedź.")]])
    services = make_services(llm=llm)
    session = Session(services, SendAfterCloseOutbox())
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    await session.handle_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
    task = session._active
    assert task is not None
    done, pending = await asyncio.wait({task}, timeout=2.0)
    assert not pending, "turn never finished"
    # The graceful path: the turn ends quietly instead of dying on the raw send error.
    assert task.exception() is None
    assert session._gone
    await session.close()


def test_device_mic_window_closes_after_unconfirmed_blip(make_services, registry) -> None:
    """A blip shorter than barge_in_ms starts an UNCONFIRMED VAD segment. When the listen
    timer expires inside that segment, _expire does nothing and nothing re-arms it: the
    server keeps the mic window open forever (no listen_end, wake word stays off on the
    device until its own 30-60 s watchdog). The window must eventually close."""
    devices = {
        "verify_wake": False, "listen_timeout_s": 0.4, "follow_up_ms": 50,
        "listen_grace_s": 0.05, "wake_timeout_s": 5.0,
    }
    services = make_services(cfg={"devices": devices})
    app = create_device_app(services, registry)
    client = TestClient(app, client=("127.0.0.1", 50000))
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps(HELLO))
        ready = read(ws)
        assert ready["type"] == "ready", ready
        read_until(ws, is_listening)
        # Unverified wake -> wake_accepted + a 0.4 s listen timer.
        ws.send_text(json.dumps({"type": "wake", "word": "hej_gaja", "score": 0.97, "preroll_ms": 0}))
        verdict = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") in ("wake_accepted", "wake_rejected"))
        assert verdict[-1] == {"type": "wake_accepted"}, verdict
        # At 0.25 s (timer fires at 0.4 s): 160 ms of speech = 5 windows < barge_in (8):
        # a segment opens, never confirms, and resets silently ~0.5 s after it ends.
        time.sleep(0.25)
        for frame in speech_frames(0.16):
            ws.send_bytes(frame)
        for frame in silence_frames(0.1):
            ws.send_bytes(frame)
        # Correct behaviour: the window closes on its own (listen_end / mic closed).
        deadline = time.time() + 2.5
        saw_listen_end = False
        while time.time() < deadline:
            try:
                msg = ws.receive(timeout=0.2)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "listen_end":
                saw_listen_end = True
                break
        session = next(s for s in services.sessions if s.is_device)
        assert saw_listen_end or not session._device.mic_open


@pytest.mark.asyncio
async def test_hermes_failure_on_a_dead_socket_ends_the_job_quietly(make_services) -> None:
    """A send failure inside _hermes_failed must not raise out of the handler: _run_hermes calls it
    from inside its own `except Exception`, so the raised ConnectionError would escape the handler,
    kill the job task silently and leave the job stuck in session._jobs."""
    from tests.conftest import FakeHermes

    services = make_services(hermes=FakeHermes(error="worker crashed"))
    out = DeadWhenToldOutbox()
    session = Session(services, out)
    out.dead = True  # the browser vanished before the job could report anything
    job = _HermesJob(id="j1", task="znajdź przepis na sernik", language="pl")
    runner = asyncio.create_task(session._run_hermes(job, [], "test"))
    done, pending = await asyncio.wait({runner}, timeout=2.0)
    assert not pending, "the Hermes job task never finished"
    assert runner.exception() is None, "the job task died on the second send failure"
    assert session._gone
    assert session._jobs == {}
    await session.close()


@pytest.mark.asyncio
async def test_auth_fail_delay_counts_against_the_pending_cap(make_services, registry, monkeypatch) -> None:
    """A socket sleeping in AUTH_FAIL_DELAY_S must still hold its pending slot: without that, a
    flood of bad-token handshakes rotates through the cap instead of being bounded by it."""
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 1.5)  # overrides this file's autouse 0.0
    services = make_services(
        cfg={"devices": {"listen_host": "127.0.0.1", "port": 0, "max_connections": 1, "hello_timeout_s": 2.0}}
    )
    listener = DeviceListener(services, registry, port=0)
    assert await listener.start()
    try:
        url = f"ws://127.0.0.1:{listener.port}/device"
        bad = await websockets.connect(url)
        await bad.send(json.dumps({**HELLO, "token": "w" * 40}))
        await asyncio.sleep(0.4)  # the server is now sleeping in the auth-fail delay
        with pytest.raises(websockets.exceptions.InvalidStatus):
            await websockets.connect(url)  # still counted: refused at once, not after the delay
        await bad.close()
        await asyncio.sleep(1.4)  # past the delay: the slot is free again
        good = await websockets.connect(url)
        await good.send(json.dumps(HELLO))
        ready = json.loads(await asyncio.wait_for(good.recv(), 5))
        assert ready["type"] == "ready", ready
        await good.close()
    finally:
        await listener.stop()


@pytest.mark.asyncio
async def test_typed_text_over_4000_chars_is_rejected(make_services) -> None:
    """Typed text turns share /say's 4000-character limit: a giant paste gets an error message and
    never starts a turn, so nothing of it reaches the router prompt or the LLM."""
    llm = FakeLLM([[TextDelta("Krótka odpowiedź.")]])
    services = make_services(llm=llm)
    out = DeadWhenToldOutbox()
    session = Session(services, out)
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    await session.handle_text(json.dumps({"type": "text", "text": "a" * 4001}))
    errors = [m for m in out.sent if m.get("type") == "error"]
    assert errors and "4000" in errors[-1]["message"]
    assert session._active is None  # rejected before a turn started
    assert llm.calls == []  # nothing reached the LLM
    # The boundary itself matches /say: exactly 4000 characters are still accepted.
    await session.handle_text(json.dumps({"type": "text", "text": "a" * 4000}))
    task = session._active
    assert task is not None
    await asyncio.wait_for(asyncio.gather(task), 5)
    await session.close()
