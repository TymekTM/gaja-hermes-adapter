"""Protocol v1.10: wake-word takes recorded through a device microphone, driven by the admin API.

A fake device speaks the WebSocket side (record_start/started/frames/mark/stop/end) while the HTTP API is
called on the main app; finished takes must land in the same tree as browser uploads, and frames that arrive
while a take records must never reach the conversation pipeline.

The device app is mounted inside the main app so that both run on one event loop, as in production (the
device listener is an embedded uvicorn server on the adapter's own loop).
"""

from __future__ import annotations

import json
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount

from gaja.admin.auth import write_credentials
from gaja.devices import listener as listener_module
from gaja.devices.listener import create_device_app
from gaja.devices.registry import DeviceRegistry
from gaja.server import create_app
from tests.conftest import FakeLLM
from tests.test_device_protocol import FAST, HELLO, TOKEN, ScriptedSTT
from tests.test_ws_protocol import is_type, read_until, speech_frames

PASSWORD = "correct horse battery"
CAPS_RECORD = {"aec": False, "out_rates": [16000], "preroll_ms": 1500, "record": True}
DEVICE_WS = "/__device/device"


@pytest.fixture(autouse=True)
def _no_auth_delay(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 0.0)


@dataclass
class Apps:
    api: TestClient
    services: Any
    rec_dir: Path


@pytest.fixture
def apps(make_services, tmp_path) -> Any:
    registry_path = tmp_path / "devices.toml"
    registry_path.write_text(f'[devices.atom-1]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    registry_path.chmod(0o600)
    creds = tmp_path / "admin.toml"
    write_credentials(creds, PASSWORD)
    rec_dir = tmp_path / "rec"
    services = make_services(
        cfg={
            "devices": {**FAST},
            "wakeword": {"recordings_dir": str(rec_dir), "max_total_mb": 64},
            "admin": {"credentials_file": str(creds)},
        },
    )
    app = create_app(services.cfg, services)
    app.router.routes.insert(0, Mount("/__device", app=create_device_app(services, DeviceRegistry(registry_path))))
    with TestClient(app, client=("127.0.0.1", 50000)) as api:
        assert api.post("/api/admin/login", json={"password": PASSWORD}).status_code == 200
        yield Apps(api, services, rec_dir)


def connect(ws: Any, caps: dict[str, Any] = CAPS_RECORD) -> None:
    ws.send_text(json.dumps({**HELLO, "caps": caps}))
    ready = read_until(ws, is_type("ready"))
    assert ready[-1]["type"] == "ready"


def start_take(api: TestClient, pool: ThreadPoolExecutor, **overrides: Any) -> Any:
    body = {"device_id": "atom-1", "session": "s1", "label": "wake", "prompt_id": "wake_normal", "max_ms": 10000}
    return pool.submit(api.post, "/api/admin/record/takes", json={**body, **overrides})


def meta_lines(rec_dir: Path, session: str = "s1") -> list[dict[str, Any]]:
    return [json.loads(line) for line in (rec_dir / session / "meta.jsonl").read_text().splitlines()]


def wait_for_take(api: TestClient, device_id: str = "atom-1", timeout_s: float = 10.0) -> dict[str, Any]:
    """Poll the status endpoint until the take finished (HTTP may race the WS message handling)."""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        take = api.get(f"/api/admin/record/takes/{device_id}").json()["take"]
        if take is not None and take["state"] in ("done", "error"):
            return take
        time.sleep(0.05)
    pytest.fail("the take did not finish in time")


def no_pipeline_messages(ws: Any, forbid: tuple[str, ...] = ("record_start", "vad", "transcript", "utterance", "wake_accepted")) -> None:
    """The device answers a ping; none of the forbidden events may arrive before the pong."""
    ws.send_text(json.dumps({"type": "ping"}))
    messages = read_until(ws, is_type("pong"))
    kinds = [m.get("type") for m in messages if isinstance(m, dict)]
    assert not any(k in forbid for k in kinds), (kinds, forbid)


# ---- happy path ----------------------------------------------------------------------------------


def test_record_start_frames_mark_stop_end_writes_wav_and_meta(apps: Apps) -> None:
    stt = ScriptedSTT("Hej Gaja")
    apps.services.stt = stt
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            assert start["id"] and start["max_ms"] == 10000
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            response = future.result(timeout=10)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True and body["state"] == "recording"
        assert body["take"] == 0 and body["id"] == start["id"]

        for frame in speech_frames(0.6):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "record_mark", "id": start["id"], "word": "hej_gaja", "t_ms": 500}))
        for frame in speech_frames(0.6):
            ws.send_bytes(frame)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(apps.api.post, "/api/admin/record/takes/atom-1/stop")
            stop = read_until(ws, is_type("record_stop"))[-1]
            assert stop["id"] == start["id"]
            ws.send_text(json.dumps({"type": "record_end", "id": start["id"], "ms": 1200, "reason": "stop"}))
            done = future.result(timeout=10).json()
        no_pipeline_messages(ws)
    assert done["state"] == "done" and done["end_reason"] == "stop"
    assert done["device_detected"] is True and done["marks"] == [{"word": "hej_gaja", "t_ms": 500}]
    assert done["partial"] is False and done["result"]["take"] == 0
    assert done["result"]["duration_s"] == pytest.approx(1.2, abs=0.05)

    wav = apps.rec_dir / "s1/wake/wake_normal/000.wav"
    assert wav.is_file()
    with wave.open(str(wav), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000
        assert w.getnframes() == pytest.approx(16000 * 1.2, abs=800)
    meta = meta_lines(apps.rec_dir)[-1]
    assert meta["source"] == "device" and meta["device_id"] == "atom-1" and meta["fw"] == "0.1.0"
    assert meta["marks"] == [{"word": "hej_gaja", "t_ms": 500}] and meta["device_detected"] is True
    assert meta["partial"] is False and meta["label"] == "wake" and meta["take"] == 0

    assert stt.calls == []  # frames during a recording never reach the pipeline
    progress = apps.api.get("/api/wakeword/progress", params={"session": "s1"}).json()
    assert progress["total"] == 1
    assert progress["device"]["wake"] == {"takes": 1, "detected": 1, "marks": 1, "recall": 1.0}


def test_max_ms_auto_end_stores_the_take(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool, max_ms=1000)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            assert future.result(timeout=10).json()["state"] == "recording"
        for frame in speech_frames(0.8):
            ws.send_bytes(frame)
        # the device stops on its own at max_ms
        ws.send_text(json.dumps({"type": "record_end", "id": start["id"], "ms": 1000, "reason": "max"}))
    status = wait_for_take(apps.api)
    assert status["state"] == "done" and status["end_reason"] == "max"
    assert status["device_detected"] is False and status["result"]["take"] == 0
    assert (apps.rec_dir / "s1/wake/wake_normal/000.wav").is_file()
    assert meta_lines(apps.rec_dir)[-1]["device_detected"] is False


# ---- refusals ------------------------------------------------------------------------------------


def test_busy_device_gets_409_without_being_contacted(apps: Apps) -> None:
    apps.services.llm = FakeLLM(hang=True)
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "state" and m.get("state") == "thinking")
        response = apps.api.post(
            "/api/admin/record/takes",
            json={"device_id": "atom-1", "session": "s1", "label": "wake", "prompt_id": "wake_normal"},
        )
        assert response.status_code == 409
        no_pipeline_messages(ws, forbid=("record_start",))  # the device was never contacted


def test_unsupported_device_is_refused(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws, caps={"aec": False, "out_rates": [16000], "preroll_ms": 1500})  # no caps.record
        response = apps.api.post(
            "/api/admin/record/takes",
            json={"device_id": "atom-1", "session": "s1", "label": "wake", "prompt_id": "wake_normal"},
        )
        assert response.status_code == 409
        assert "does not support recording" in response.json()["detail"]
        no_pipeline_messages(ws)


def test_one_active_recording_per_device(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            assert future.result(timeout=10).json()["state"] == "recording"
            second = pool.submit(
                apps.api.post,
                "/api/admin/record/takes",
                json={"device_id": "atom-1", "session": "s1", "label": "wake", "prompt_id": "wake_normal"},
            ).result(timeout=10)
        assert second.status_code == 409
        assert "already in progress" in second.json()["detail"]


def test_device_record_error_is_reported(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_error", "id": start["id"], "reason": "mic"}))
            response = future.result(timeout=10)
        assert response.status_code == 503 and "mic" in response.json()["detail"]
        status = apps.api.get("/api/admin/record/takes/atom-1").json()["take"]
        assert status["state"] == "error"
        assert not list(apps.rec_dir.rglob("*.wav"))


def test_wake_during_a_recording_is_ignored(apps: Apps) -> None:
    stt = ScriptedSTT("Hej Gaja")
    apps.services.stt = stt
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            assert future.result(timeout=10).json()["state"] == "recording"
        ws.send_text(json.dumps({"type": "wake", "word": "hej_gaja", "score": 0.99, "preroll_ms": 0}))
        for frame in speech_frames(0.6):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "record_end", "id": start["id"], "ms": 600, "reason": "stop"}))
        no_pipeline_messages(ws)  # no wake_accepted, no vad events: the wake was dropped, the take goes on
    assert stt.calls == []
    assert (apps.rec_dir / "s1/wake/wake_normal/000.wav").is_file()


# ---- disconnect ----------------------------------------------------------------------------------


def test_disconnect_mid_take_keeps_a_partial_take(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            assert future.result(timeout=10).json()["state"] == "recording"
        for frame in speech_frames(0.6):
            ws.send_bytes(frame)
        # the socket dies mid-take; the server keeps what it received
    status = wait_for_take(apps.api)
    assert status["state"] == "done" and status["end_reason"] == "partial"
    assert status["partial"] is True
    assert status["device_detected"] is False
    wav = apps.rec_dir / "s1/wake/wake_normal/000.wav"
    assert wav.is_file()
    with wave.open(str(wav), "rb") as w:
        assert w.getnframes() == pytest.approx(16000 * 0.6, abs=800)
    meta = meta_lines(apps.rec_dir)[-1]
    assert meta["partial"] is True and meta["source"] == "device"
    # a fresh connection can record again, and the take number continues
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            body = future.result(timeout=10).json()
            ws.send_text(json.dumps({"type": "record_end", "id": start["id"], "ms": 500, "reason": "stop"}))
        assert body["take"] == 1


# ---- auth ----------------------------------------------------------------------------------------


def test_admin_auth_and_origin_required(make_services, tmp_path) -> None:
    creds = tmp_path / "admin.toml"
    write_credentials(creds, PASSWORD)
    services = make_services(cfg={"admin": {"credentials_file": str(creds)}})
    with TestClient(create_app(services.cfg, services)) as api:
        for method, path, kwargs in (
            ("get", "/api/admin/record/devices", {}),
            ("post", "/api/admin/record/takes", {"json": {"device_id": "x", "session": "s", "label": "wake", "prompt_id": "p"}}),
            ("get", "/api/admin/record/takes/atom-1", {}),
            ("post", "/api/admin/record/takes/atom-1/stop", {}),
        ):
            assert getattr(api, method)(path, **kwargs).status_code == 401, path
            evil = getattr(api, method)(path, headers={"Origin": "https://evil.example"}, **kwargs)
            assert evil.status_code == 403, path
        api.post("/api/admin/login", json={"password": PASSWORD})
        assert api.get("/api/admin/record/devices").status_code == 200


def test_devices_list_reports_caps_busy_and_recording(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        listed = apps.api.get("/api/admin/record/devices").json()["devices"]
        assert listed == [{
            "id": "atom-1", "name": "atom-1", "fw": "0.1.0", "caps_record": True,
            "busy": False, "recording": False, "take": None,
        }]
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = start_take(apps.api, pool)
            start = read_until(ws, is_type("record_start"))[-1]
            ws.send_text(json.dumps({"type": "record_started", "id": start["id"]}))
            assert future.result(timeout=10).json()["state"] == "recording"
        listed = apps.api.get("/api/admin/record/devices").json()["devices"]
        assert listed[0]["recording"] is True and listed[0]["take"]["state"] == "recording"
        ws.send_text(json.dumps({"type": "record_end", "id": start["id"], "ms": 300, "reason": "stop"}))


def test_bad_session_and_label_rejected(apps: Apps) -> None:
    with apps.api.websocket_connect(DEVICE_WS) as ws:
        connect(ws)
        for overrides in ({"session": "../evil"}, {"label": "other"}, {"prompt_id": "a b"}):
            response = apps.api.post("/api/admin/record/takes", json={
                "device_id": "atom-1", "session": "s1", "label": "wake", "prompt_id": "p", **overrides})
            assert response.status_code in (400, 422), overrides
        assert apps.api.post(
            "/api/admin/record/takes",
            json={"device_id": "nope", "session": "s1", "label": "wake", "prompt_id": "p"},
        ).status_code == 404

