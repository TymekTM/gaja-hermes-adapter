"""PC relay (protocol v1.14, docs/DESKTOP.md): registry, pc_call/pc_result routing and failure rules.

End-to-end over a real device-listener socket (one event loop for the server, the fake PC and the
relay callers), plus pure relay-unit checks for the pending-future lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

import gaja.devices.pc as pc_module
from gaja.devices.audio import PcCaps
from gaja.devices.listener import DeviceListener
from gaja.devices.pc import PcRelay
from gaja.devices.registry import DeviceRegistry
from tests.conftest import FakeSTT
from tests.test_device_protocol import FAST, TOKEN

PC_CAPS = {"os": "windows", "host": "TYMEK-PC", "tools": ["shell", "open", "screenshot", "type", "keys", "mouse", "clipboard"]}
HELLO_PC = {
    "type": "hello",
    "client": "device",
    "device_id": "pc-tymek",
    "token": TOKEN,
    "fw": "desktop-0.1.0",
    "lang": "pl",
    "caps": {"aec": True, "out_rates": [48000, 16000], "preroll_ms": 1500, "ptt": True, "pc": PC_CAPS},
}


@pytest.fixture
def relay() -> PcRelay:
    return PcRelay()


def _register(relay: PcRelay, device_id: str = "pc-tymek", tools: list[str] | None = None, owner: Any = None) -> None:
    caps = PcCaps(os="windows", host="TYMEK-PC", tools=tools if tools is not None else list(PC_CAPS["tools"]))
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    owner = owner or object()
    relay.register(owner, device_id, caps, send)
    return None


# ---- unit: registry and pending-future lifecycle ---------------------------------------------------


def test_register_describes_devices(relay: PcRelay) -> None:
    _register(relay, "pc-a", tools=["shell"])
    _register(relay, "pc-b", tools=["clipboard"])
    devices = {d["id"]: d for d in relay.devices()}
    assert set(devices) == {"pc-a", "pc-b"}
    assert devices["pc-a"]["host"] == "TYMEK-PC"
    assert devices["pc-a"]["os"] == "windows"
    assert devices["pc-a"]["tools"] == ["shell"]
    assert devices["pc-a"]["connected_s"] >= 0.0
    assert relay.has_tool("pc-a", "shell") and not relay.has_tool("pc-a", "clipboard")
    assert relay.get("pc-c") is None


@pytest.mark.asyncio
async def test_call_unknown_device_answers_disconnected(relay: PcRelay) -> None:
    result = await relay.call("pc-nope", "shell", {"command": "dir"}, 1.0)
    assert result.ok is False and result.error == "disconnected"


@pytest.mark.asyncio
async def test_late_and_unknown_results_are_ignored(relay: PcRelay, caplog: pytest.LogCaptureFixture) -> None:
    owner = object()
    caps = PcCaps(os="linux", host="h", tools=["shell"])
    results: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        results.append(payload)

    relay.register(owner, "pc-1", caps, send)
    call_task = asyncio.create_task(relay.call("pc-1", "shell", {"command": "echo hi"}, 0.2))
    await asyncio.sleep(0.05)
    call_id = results[0]["id"]
    # A result for an id that was never sent.
    relay.on_result("pc-1", {"type": "pc_result", "id": "ffffffffffff", "ok": True, "text": "x"})
    # A malformed id.
    relay.on_result("pc-1", {"type": "pc_result", "id": "zz", "ok": True})
    # A result for another device.
    relay.on_result("pc-other", {"type": "pc_result", "id": call_id, "ok": True, "text": "x"})
    # The real answer.
    relay.on_result("pc-1", {"type": "pc_result", "id": call_id, "ok": True, "text": "hi", "data": {"exit_code": 0}})
    result = await call_task
    assert result.ok is True and result.text == "hi" and result.data == {"exit_code": 0}
    # A duplicate (already finished) result is ignored silently.
    with caplog.at_level("DEBUG", logger="gaja.devices.pc"):
        relay.on_result("pc-1", {"type": "pc_result", "id": call_id, "ok": True, "text": "again"})
    assert "ignored" in caplog.text


@pytest.mark.asyncio
async def test_server_timeout_is_timeout_s_plus_grace(monkeypatch: pytest.MonkeyPatch, relay: PcRelay) -> None:
    monkeypatch.setattr(pc_module, "GRACE_S", 0.2)
    owner = object()
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    relay.register(owner, "pc-1", PcCaps(os="linux", host="h", tools=["shell"]), send)
    t0 = time.monotonic()
    result = await relay.call("pc-1", "shell", {"command": "sleep 9"}, 0.3)
    assert result.ok is False and result.error == "timeout"
    assert 0.4 <= time.monotonic() - t0 < 2.0  # 0.3 s timeout + 0.2 s grace, not the full grace of 5 s
    # The timed-out call is no longer pending: a late result is ignored.
    call_id = sent[0]["id"]
    relay.on_result("pc-1", {"type": "pc_result", "id": call_id, "ok": True, "text": "late"})
    assert relay.get("pc-1") is not None and call_id not in relay.get("pc-1").pending


@pytest.mark.asyncio
async def test_send_failure_answers_disconnected(relay: PcRelay) -> None:
    owner = object()

    async def send(payload: dict[str, Any]) -> None:
        raise ConnectionError("device connection is closed")

    relay.register(owner, "pc-1", PcCaps(os="linux", host="h", tools=["shell"]), send)
    result = await relay.call("pc-1", "shell", {"command": "dir"}, 1.0)
    assert result.ok is False and result.error == "disconnected"


# ---- end to end over a real listener socket ---------------------------------------------------------


def _make_listener(make_services, tmp_path, *, pc_cfg: dict[str, Any] | None = None) -> tuple[DeviceListener, Any]:
    registry_path = tmp_path / "devices.toml"
    registry_path.write_text(f'[devices."pc-tymek"]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    registry_path.chmod(0o600)
    services = make_services(
        cfg={"devices": {**FAST, "verify_wake": False}, "pc": pc_cfg or {}}, stt=FakeSTT("hej gaja")
    )
    return DeviceListener(services, DeviceRegistry(registry_path), port=0), services


async def _connect_pc(url: str, hello: dict[str, Any] = HELLO_PC) -> Any:
    import websockets

    ws = await websockets.connect(url, max_size=4 * 1024 * 1024)
    await ws.send(json.dumps(hello))
    ready = json.loads(await ws.recv())
    assert ready["type"] == "ready", ready
    return ws, ready


async def _recv_call(ws: Any) -> dict[str, Any]:
    """The next pc_call frame, skipping state/transcript noise the session emits meanwhile."""
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if msg.get("type") == "pc_call":
            return msg


def test_relay_end_to_end_call_result_and_concurrency(make_services, tmp_path) -> None:
    async def scenario() -> None:
        listener, services = _make_listener(make_services, tmp_path)
        assert await listener.start()
        relay = services.pc_relay
        try:
            url = f"ws://127.0.0.1:{listener.port}/device"
            ws, ready = await _connect_pc(url)
            assert ready["device"]["pc"] is True
            assert [d["id"] for d in relay.devices()] == ["pc-tymek"]
            # Two concurrent calls; the answers may arrive out of order.
            first = asyncio.create_task(relay.call("pc-tymek", "shell", {"command": "echo one"}, 5))
            second = asyncio.create_task(relay.call("pc-tymek", "shell", {"command": "echo two"}, 5))
            got: dict[str, str] = {}
            for _ in range(2):
                msg = await _recv_call(ws)
                assert msg["tool"] == "shell"
                assert len(msg["id"]) == 12 and msg["id"] == msg["id"].lower()
                assert msg["timeout_s"] == 5
                await ws.send(json.dumps({"type": "pc_result", "id": msg["id"], "ok": True,
                                          "text": "one" if msg["args"]["command"].endswith("one") else "two",
                                          "data": {"exit_code": 0}}))
            r1, r2 = await asyncio.gather(first, second)
            got[r1.text], got[r2.text] = "r1", "r2"
            assert {r1.text, r2.text} == {"one", "two"}
            assert r1.ok and r2.ok and r1.data == {"exit_code": 0}
            await ws.close()
        finally:
            await listener.stop()

    asyncio.run(scenario())


def test_disconnect_fails_pending_calls_at_once(make_services, tmp_path) -> None:
    async def scenario() -> None:
        listener, services = _make_listener(make_services, tmp_path)
        assert await listener.start()
        relay = services.pc_relay
        try:
            url = f"ws://127.0.0.1:{listener.port}/device"
            ws, _ = await _connect_pc(url)
            call = asyncio.create_task(relay.call("pc-tymek", "shell", {"command": "sleep 60"}, 60))
            msg = await _recv_call(ws)  # the pc_call reached the device
            t0 = time.monotonic()
            await ws.close()  # the device vanishes without answering
            result = await call
            assert result.ok is False and result.error == "disconnected"
            assert time.monotonic() - t0 < 5.0  # at once, not after the 65 s server timeout
            assert relay.get("pc-tymek") is None
            assert relay.devices() == []
            # A late result from a reconnect-less socket has nowhere to go.
            relay.on_result("pc-tymek", {"type": "pc_result", "id": msg["id"], "ok": True, "text": "late"})
        finally:
            await listener.stop()

    asyncio.run(scenario())


def test_superseded_connection_fails_its_pending_and_the_new_one_registers(make_services, tmp_path) -> None:
    async def scenario() -> None:
        listener, services = _make_listener(make_services, tmp_path)
        assert await listener.start()
        relay = services.pc_relay
        try:
            url = f"ws://127.0.0.1:{listener.port}/device"
            old_ws, _ = await _connect_pc(url)
            call = asyncio.create_task(relay.call("pc-tymek", "shell", {"command": "sleep 60"}, 60))
            msg = await _recv_call(old_ws)
            # The device reconnects before the old socket is closed cleanly.
            new_ws, ready = await _connect_pc(url)
            assert ready["device"]["pc"] is True
            result = await call
            assert result.ok is False and result.error == "disconnected"
            assert [d["id"] for d in relay.devices()] == ["pc-tymek"]
            # The new connection serves calls.
            await new_ws.send(json.dumps({"type": "pc_result", "id": msg["id"], "ok": True, "text": "too late"}))
            new_call = asyncio.create_task(relay.call("pc-tymek", "clipboard", {"action": "get"}, 5))
            new_msg = await _recv_call(new_ws)
            assert new_msg["type"] == "pc_call" and new_msg["tool"] == "clipboard"
            await new_ws.send(json.dumps({"type": "pc_result", "id": new_msg["id"], "ok": True, "text": "clip"}))
            assert (await new_call).text == "clip"
            await old_ws.close()
            await new_ws.close()
        finally:
            await listener.stop()

    asyncio.run(scenario())


def test_pc_denied_or_disabled_leaves_ready_pc_false_and_no_relay_entry(make_services, tmp_path) -> None:
    async def scenario() -> None:
        listener, services = _make_listener(make_services, tmp_path, pc_cfg={"deny_devices": ["pc-tymek"]})
        assert await listener.start()
        try:
            ws, ready = await _connect_pc(f"ws://127.0.0.1:{listener.port}/device")
            assert ready["device"]["pc"] is False
            assert services.pc_relay.devices() == []
            await ws.close()
        finally:
            await listener.stop()

        listener2, services2 = _make_listener(make_services, tmp_path, pc_cfg={"enabled": False})
        assert await listener2.start()
        try:
            ws, ready = await _connect_pc(f"ws://127.0.0.1:{listener2.port}/device")
            assert ready["device"]["pc"] is False
            assert services2.pc_relay.devices() == []
            await ws.close()
        finally:
            await listener2.stop()

    asyncio.run(scenario())


def test_pc_result_from_a_web_session_is_not_an_error(make_services) -> None:
    """A plain /ws client sending pc_result falls through silently (it is device-only, never answered)."""
    from gaja.pipeline import Session
    from tests.test_ws_protocol import read_until

    services = make_services()
    app = FastAPI()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        session = Session(services, _SyncOutbox(websocket))
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text"):
                    await session.handle_text(message["text"])
        finally:
            await session.close()

    class _SyncOutbox:
        def __init__(self, websocket: Any) -> None:
            self._ws = websocket

        async def send_json(self, payload: dict[str, Any]) -> None:
            await self._ws.send_json(payload)

        async def send_bytes(self, data: bytes) -> None:
            await self._ws.send_bytes(data)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "client": "web", "lang": "pl"})
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "ready")
        ws.send_json({"type": "pc_result", "id": "a" * 12, "ok": True})
        # No error frame comes back for it (a reply would be noise); ping still works.
        ws.send_json({"type": "ping"})
        pong = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "pong")
        assert pong
