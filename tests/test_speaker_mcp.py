"""Unit tests for device sound playback, speaker MCP tools, and ACP mcpServers wiring."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja import mcp_speaker
from gaja.api.devices import SOUNDS_DIR, available_sounds, create_devices_router, list_sounds, load_sound_wav
from gaja.devices.registry import DeviceInfo
from gaja.hermes.client import HermesClient, HermesClientConfig
from gaja.pipeline import Session
from tests.test_device_protocol import FAST, HELLO, TOKEN
from tests.test_hermes_client import FakeAcpTransport


class RecordingOutbox:
    """In-memory outbox recording JSON frames and PCM byte chunks."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any] | bytes] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(bytes(data))


class FakeListenerWithSession:
    """Minimal listener stub bridging HTTP /play and /say to a live Session."""

    def __init__(self, device_id: str, session: Session) -> None:
        self.device_id = device_id
        self.session = session
        self.connected = [device_id]

    async def play(self, device_id: str, pcm: bytes, sample_rate: int, *, name: str = "") -> bool:
        if device_id != self.device_id:
            return False
        await self.session.play_sound(pcm, sample_rate, name=name)
        return True

    async def say(self, device_id: str, text: str) -> bool:
        if device_id != self.device_id:
            return False
        await self.session.announce(text)
        return True


# ---- sound assets & loader -----------------------------------------------------------------------


def test_sound_assets_conform_to_spec() -> None:
    names = list_sounds()
    assert {"alert", "chime", "success"}.issubset(set(names))
    sounds_map = available_sounds()

    for name in ("alert", "chime", "success"):
        path = sounds_map[name]
        assert path.parent == SOUNDS_DIR
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getcomptype() == "NONE"
            rate = wf.getframerate()
            frames = wf.getnframes()
            duration_s = frames / rate
            assert 0.05 <= duration_s <= 2.0, f"{name}: duration {duration_s}s out of bounds"
            raw = wf.readframes(frames)

        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        peak = float(np.max(np.abs(samples))) / 32768.0
        peak_dbfs = 20.0 * np.log10(peak)
        assert peak_dbfs <= -6.0 + 1e-3, f"{name}: peak {peak_dbfs:.2f} dBFS exceeds -6 dBFS"

        loaded = load_sound_wav(name)
        assert loaded is not None
        pcm, loaded_rate = loaded
        assert pcm == raw
        assert loaded_rate == rate


@pytest.mark.parametrize(
    "bad_name",
    ["", "unknown_sound", "../etc/passwd", "/etc/passwd", "chime.wav", "chime/../../secret", ".."],
)
def test_load_sound_wav_rejects_unknown_or_traversal_without_touching_path(
    bad_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened_paths: list[str] = []
    real_wave_open = wave.open

    def spy_wave_open(f: Any, *args: Any, **kwargs: Any) -> Any:
        opened_paths.append(str(f))
        return real_wave_open(f, *args, **kwargs)

    monkeypatch.setattr(wave, "open", spy_wave_open)
    assert load_sound_wav(bad_name) is None
    assert opened_paths == []


# ---- /api/devices/sounds & /api/devices/{device_id}/play -----------------------------------------


def _make_devices_test_app(listener: Any = None) -> FastAPI:
    app = FastAPI()
    app.include_router(create_devices_router(lambda origin: origin is None))
    app.state.device_listener = listener
    return app


def test_get_sounds_endpoint() -> None:
    client = TestClient(_make_devices_test_app())
    res = client.get("/api/devices/sounds")
    assert res.status_code == 200
    body = res.json()
    assert "sounds" in body
    assert {"alert", "chime", "success"}.issubset(set(body["sounds"]))


@pytest.mark.parametrize(
    "bad_sound",
    ["nonexistent", "../etc/passwd", "../../config.toml", "chime.wav", "   "],
)
def test_play_endpoint_rejects_unknown_sound_with_4xx(bad_sound: str, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_wave_open = wave.open

    def spy_open(f: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(f))
        return real_wave_open(f, *args, **kwargs)

    monkeypatch.setattr(wave, "open", spy_open)
    client = TestClient(_make_devices_test_app(listener=None))
    res = client.post("/api/devices/atom-pokoj/play", json={"sound": bad_sound})
    assert 400 <= res.status_code < 500
    assert opened == []


def test_play_endpoint_returns_404_when_device_not_connected(make_services) -> None:
    # 1. No listener at all
    client = TestClient(_make_devices_test_app(listener=None))
    res = client.post("/api/devices/atom-pokoj/play", json={"sound": "chime"})
    assert res.status_code == 404

    # 2. Listener present, but requested device is not connected
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-other"))
    listener = FakeListenerWithSession("atom-other", session)
    client2 = TestClient(_make_devices_test_app(listener=listener))
    res2 = client2.post("/api/devices/atom-pokoj/play", json={"sound": "chime"})
    assert res2.status_code == 404


@pytest.mark.asyncio
async def test_play_endpoint_happy_path_streams_audio_start_bytes_audio_end(make_services) -> None:
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    listener = FakeListenerWithSession("atom-pokoj", session)
    app = _make_devices_test_app(listener=listener)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/devices/atom-pokoj/play", json={"sound": "chime"})

    assert res.status_code == 200
    assert res.json() == {"ok": True, "sound": "chime"}

    json_msgs = [m for m in raw_out.sent if isinstance(m, dict)]
    byte_chunks = [m for m in raw_out.sent if isinstance(m, bytes)]
    msg_types = [m["type"] for m in json_msgs]

    assert "audio_start" in msg_types
    assert "audio_end" in msg_types
    assert "metrics" in msg_types
    assert "follow_up" in msg_types
    assert byte_chunks, "Expected PCM bytes between audio_start and audio_end"

    first_start_idx = raw_out.sent.index(next(m for m in json_msgs if m["type"] == "audio_start"))
    first_end_idx = raw_out.sent.index(next(m for m in json_msgs if m["type"] == "audio_end"))
    first_bytes_idx = next(i for i, m in enumerate(raw_out.sent) if isinstance(m, bytes))
    last_bytes_idx = max(i for i, m in enumerate(raw_out.sent) if isinstance(m, bytes))

    assert first_start_idx < first_bytes_idx <= last_bytes_idx < first_end_idx

    loaded = load_sound_wav("chime")
    assert loaded is not None
    expected_pcm, expected_rate = loaded
    audio_start_msg = next(m for m in json_msgs if m["type"] == "audio_start")
    assert audio_start_msg["sample_rate"] == expected_rate
    assert b"".join(byte_chunks) == expected_pcm

    states = [m["state"] for m in json_msgs if m["type"] == "state"]
    assert states == ["speaking", "listening"]

    metrics_msg = next(m for m in json_msgs if m["type"] == "metrics")
    assert metrics_msg["kind"] == "notification"
    assert metrics_msg["route"] == "sound"

    await session.close()


# ---- MCP speaker tool functions ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_speaker_tools_with_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    async def fake_request(method: str, path: str, *, json_body: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        calls.append((method, path, json_body))
        if method == "GET" and path == "/api/devices":
            return {
                "devices": [
                    {"id": "atom-pokoj", "name": "Pokój", "online": True, "enabled": True},
                ]
            }
        if method == "POST" and path == "/api/devices/atom-pokoj/say":
            return {"ok": True}
        if method == "POST" and path == "/api/devices/atom-pokoj/play":
            return {"ok": True, "sound": json_body["sound"] if json_body else ""}
        return {"ok": False, "error": "not found", "status": 404}

    monkeypatch.setattr(mcp_speaker, "_request", fake_request)

    speakers_res = await mcp_speaker.list_speakers()
    assert speakers_res["devices"][0]["id"] == "atom-pokoj"
    assert speakers_res["devices"][0]["online"] is True

    sounds_res = await mcp_speaker.list_sounds()
    assert {"alert", "chime", "success"}.issubset(set(sounds_res["sounds"]))

    say_res = await mcp_speaker.speaker_say("atom-pokoj", "Obiad gotowy!")
    assert say_res == {"ok": True, "device_id": "atom-pokoj", "text": "Obiad gotowy!"}

    play_res = await mcp_speaker.speaker_play_sound("atom-pokoj", "chime")
    assert play_res == {"ok": True, "device_id": "atom-pokoj", "sound": "chime"}

    assert calls == [
        ("GET", "/api/devices", None),
        ("POST", "/api/devices/atom-pokoj/say", {"text": "Obiad gotowy!"}),
        ("POST", "/api/devices/atom-pokoj/play", {"sound": "chime"}),
    ]


@pytest.mark.asyncio
async def test_mcp_speaker_input_validation_and_http_errors() -> None:
    # Input validation fails fast without touching HTTP
    assert (await mcp_speaker.speaker_say("../bad", "test"))["ok"] is False
    assert (await mcp_speaker.speaker_say("atom-pokoj", "   "))["ok"] is False
    assert (await mcp_speaker.speaker_say("atom-pokoj", "x" * 501))["ok"] is False
    assert (await mcp_speaker.speaker_play_sound("../bad", "chime"))["ok"] is False
    assert (await mcp_speaker.speaker_play_sound("atom-pokoj", "   "))["ok"] is False

    # Verify _request against an in-process httpx MockTransport
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/devices/atom-pokoj/play":
            return httpx.Response(404, json={"detail": "device not connected"})
        return httpx.Response(200, json={"devices": []})

    mock_transport = httpx.MockTransport(handler)
    orig_AsyncClient = httpx.AsyncClient

    def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = mock_transport
        return orig_AsyncClient(*args, **kwargs)

    with patch("gaja.mcp_speaker.httpx.AsyncClient", side_effect=patched_client):
        res_404 = await mcp_speaker.speaker_play_sound("atom-pokoj", "chime")
        assert res_404 == {
            "ok": False,
            "status": 404,
            "error": "device not connected",
        }
        res_ok = await mcp_speaker.list_speakers()
        assert res_ok == {"devices": []}


# ---- ACP mcpServers wiring -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hermes_client_wires_speaker_mcp_server_by_default() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(
        HermesClientConfig(speaker_mcp=True, api_url="http://127.0.0.1:8790"),
        transport=transport,
    )

    events = [e async for e in client.submit("Zagraj gong w pokoju", context=[], session_id="s-mcp-on")]
    assert events[0].kind == "accepted"

    new_sess_calls = [params for method, params in transport.requests if method == "session/new"]
    assert len(new_sess_calls) == 1
    mcp_servers = new_sess_calls[0]["mcpServers"]
    assert len(mcp_servers) == 1

    spec = mcp_servers[0]
    assert spec["name"] == "gaja_speaker"
    assert spec["command"] == sys.executable
    assert spec["args"] == ["-m", "gaja.mcp_speaker"]
    env_dict = {item["name"]: item["value"] for item in spec["env"]}
    assert env_dict["GAJA_API_URL"] == "http://127.0.0.1:8790"
    assert Path(env_dict["PYTHONPATH"]).joinpath("gaja", "mcp_speaker.py").is_file()

    await client.aclose()


@pytest.mark.asyncio
async def test_hermes_client_omits_speaker_mcp_when_disabled() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(
        HermesClientConfig(speaker_mcp=False),
        transport=transport,
    )

    events = [e async for e in client.submit("Cześć", context=[], session_id="s-mcp-off")]
    assert events[0].kind == "accepted"

    new_sess_calls = [params for method, params in transport.requests if method == "session/new"]
    assert len(new_sess_calls) == 1
    assert new_sess_calls[0]["mcpServers"] == []

    await client.aclose()
