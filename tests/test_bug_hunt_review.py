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
from fastapi.testclient import TestClient

from gaja.devices import listener as listener_module
from gaja.devices.listener import create_device_app
from gaja.devices.registry import DeviceRegistry
from gaja.llm.client import TextDelta
from gaja.pipeline import Session
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
