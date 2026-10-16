"""Device server hardening: playback accounting under backpressure, bounded sends, reconnects, handshake caps."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest
import websockets
from starlette.websockets import WebSocket

from gaja.devices import listener as listener_module
from gaja.devices.audio import beep_pcm16
from gaja.devices.listener import DeviceListener, _Outbox
from gaja.devices.registry import DeviceInfo
from gaja.llm.client import TextDelta
from gaja.pipeline import Session
from tests.conftest import FakeLLM
from tests.test_device_protocol import FAST, HELLO, TOKEN, registry  # noqa: F401  (registry fixture)


class SlowBytesOut:
    """An outbox under TCP backpressure: bytes take a long time to be accepted."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.sent: list[Any] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        await asyncio.sleep(self.delay_s)
        self.sent.append(data)


@pytest.mark.asyncio
async def test_playback_end_is_counted_from_before_the_send(make_services) -> None:
    services = make_services(cfg={"devices": {**FAST, "ack": "earcon"}})
    session = Session(services, SlowBytesOut(0), device=DeviceInfo("atom-1"))
    out = SlowBytesOut(0.5)
    before = time.perf_counter()
    await session._device.ack(SimpleNamespace(out=out, language="pl"))  # type: ignore[union-attr]
    beep_s = len(beep_pcm16(16000)) / 2 / 16000
    # The 0.5 s the send blocked must not be added on top of the beep's own duration.
    assert session._playback_end - before == pytest.approx(beep_s, abs=0.15)
    await session.close()


# ---- bounded sends ------------------------------------------------------------------------------


class HangingSocket:
    def __init__(self) -> None:
        self.calls = 0

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.calls += 1
        await asyncio.sleep(3600)

    send_bytes = send_json  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_outbox_send_timeout_marks_it_dead_once() -> None:
    dead: list[int] = []
    socket = HangingSocket()
    outbox = _Outbox(socket, timeout_s=0.05, on_dead=lambda: dead.append(1))  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        await outbox.send_json({"type": "x"})
    start = time.perf_counter()
    with pytest.raises(ConnectionError):
        await outbox.send_bytes(b"\0\0")  # fails at once, nothing queues behind the stuck socket
    assert time.perf_counter() - start < 0.04
    assert dead == [1] and outbox.dead and socket.calls == 1


def _listener(make_services, registry, **devices: Any) -> DeviceListener:  # noqa: F811
    services = make_services(cfg={"devices": {**FAST, "listen_host": "127.0.0.1", "port": 0, **devices}})
    return DeviceListener(services, registry, port=0)


async def _hello(url: str) -> tuple[Any, dict[str, Any]]:
    ws = await websockets.connect(url)
    await ws.send(json.dumps(HELLO))
    return ws, json.loads(await asyncio.wait_for(ws.recv(), 5))


@pytest.mark.asyncio
async def test_stuck_announcement_send_drops_the_connection_and_session(make_services, registry, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(listener_module, "SEND_TIMEOUT_S", 0.2)
    real = WebSocket.send_json

    async def hang_on_timer(self, data, mode="text"):
        if isinstance(data, dict) and data.get("type") == "timer":
            await asyncio.sleep(3600)
        await real(self, data, mode)

    monkeypatch.setattr(WebSocket, "send_json", hang_on_timer)
    listener = _listener(make_services, registry)
    assert await listener.start()
    try:
        ws, ready = await _hello(f"ws://127.0.0.1:{listener.port}/device")
        assert ready["type"] == "ready"
        services = listener.app.state.services
        (session,) = list(services.sessions)
        announce = asyncio.create_task(session.on_timer({"id": "t1", "label": "x", "kind": "timer"}, speak=False))
        await asyncio.wait_for(announce, 5)  # returns instead of wedging
        for _ in range(100):
            if not services.sessions and not listener.connected:
                break
            await asyncio.sleep(0.05)
        assert not services.sessions and listener.connected == []
        await ws.close()
    finally:
        await listener.stop()


@pytest.mark.asyncio
async def test_reconnect_does_not_wait_for_the_old_connections_close(make_services, registry, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(listener_module, "SUPERSEDE_CLOSE_S", 0.2)

    async def never_closes(self, code=1000, reason=None):
        await asyncio.sleep(3600)

    monkeypatch.setattr(WebSocket, "close", never_closes)
    listener = _listener(make_services, registry)
    assert await listener.start()
    try:
        url = f"ws://127.0.0.1:{listener.port}/device"
        first, _ = await _hello(url)
        start = time.perf_counter()
        second, ready = await _hello(url)
        assert ready["type"] == "ready" and time.perf_counter() - start < 1.0
        services = listener.app.state.services
        for _ in range(100):  # the old session is torn down, not leaked
            if len(services.sessions) == 1:
                break
            await asyncio.sleep(0.05)
        assert len(services.sessions) == 1 and listener.connected == ["atom-1"]
        await first.close()
        await second.close()
    finally:
        await listener.stop()


@pytest.mark.asyncio
async def test_pending_handshakes_are_capped_before_the_hello_read(make_services, registry) -> None:  # noqa: F811
    listener = _listener(make_services, registry, max_connections=2, hello_timeout_s=2.0)
    assert await listener.start()
    try:
        url = f"ws://127.0.0.1:{listener.port}/device"
        silent = [await websockets.connect(url), await websockets.connect(url)]  # never send a hello
        start = time.perf_counter()
        with pytest.raises(websockets.exceptions.InvalidStatus):
            await websockets.connect(url)  # refused at once, not after the 2 s hello timeout
        assert time.perf_counter() - start < 1.0
        for ws in silent:
            await ws.close()
        ws, ready = await _hello(url)  # the slots free up again
        assert ready["type"] == "ready"
        await ws.close()
    finally:
        await listener.stop()


# ---- a client that leaves mid-reply -------------------------------------------------------------


class DyingOut:
    """Sends fine until the first audio bytes, then the socket is gone."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        if getattr(self, "dead", False):
            raise ConnectionError("gone")
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.dead = True
        raise ConnectionError("gone")


@pytest.mark.asyncio
async def test_disconnect_mid_reply_is_a_normal_end(make_services, caplog) -> None:
    services = make_services(cfg={"devices": {**FAST}}, llm=FakeLLM([[TextDelta("Jest słonecznie.")]]))
    out = DyingOut()
    session = Session(services, out, device=DeviceInfo("atom-1"))
    await session.handle_text(json.dumps({**HELLO, "token": TOKEN}))
    caplog.set_level(logging.INFO)
    await session.handle_text(json.dumps({"type": "text", "text": "jaka pogoda"}))
    task = session._active
    assert task is not None
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=False), 10)  # raises if the turn died with an error
    kinds = [m.get("type") for m in out.sent]
    assert "follow_up" not in kinds and session._gone
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    await session.close()
