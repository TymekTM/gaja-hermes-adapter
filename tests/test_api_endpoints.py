"""Unit tests for Gaja HTTP API endpoints: history, devices, diagnostics."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja.api.devices import create_devices_router, record_device_activity
from gaja.api.diagnostics import create_diagnostics_router
from gaja.api.history import TurnItem, append_turn, create_history_router, read_turns
from gaja.config import Settings
from gaja.devices.controller import WAKE_LOG, get_wake_log
from gaja.metrics import MetricsRecorder, TurnMetrics


@pytest.fixture
def history_path(tmp_path: Path) -> Path:
    return tmp_path / "history.jsonl"


@pytest.fixture
def client(history_path: Path) -> TestClient:
    app = FastAPI()

    allowed = ["http://localhost:8790", "https://gaja.local"]

    def origin_ok(origin: str | None) -> bool:
        if origin is None:
            return True
        return origin.rstrip("/") in allowed

    app.include_router(create_history_router(history_path, origin_ok, max_turns=5))
    app.include_router(create_devices_router(origin_ok))
    app.include_router(create_diagnostics_router(origin_ok))

    # Add mock services to app.state
    class DummyServices:
        def __init__(self, turns_path: Path) -> None:
            self.cfg = Settings()
            self.metrics = MetricsRecorder(turns_path, window=20)

    app.state.services = DummyServices(history_path.parent / "turns.jsonl")

    return TestClient(app)


def test_history_crud_and_limit(client: TestClient, history_path: Path) -> None:
    # 1. Empty history
    res = client.get("/api/history")
    assert res.status_code == 200
    assert res.json() == {"turns": []}

    # 2. Add turns
    res1 = client.post("/api/history", json={"role": "user", "text": "Cześć Gaja", "lang": "pl"})
    assert res1.status_code == 200
    t1 = res1.json()["turn"]
    assert t1["role"] == "user"
    assert t1["text"] == "Cześć Gaja"
    assert "id" in t1 and "ts" in t1

    res2 = client.post("/api/history", json={"role": "assistant", "text": "Dzień dobry!", "lang": "pl"})
    assert res2.status_code == 200

    # 3. Read back
    res = client.get("/api/history?limit=1")
    assert res.status_code == 200
    turns = res.json()["turns"]
    assert len(turns) == 1
    assert turns[0]["text"] == "Dzień dobry!"

    res = client.get("/api/history?limit=10")
    assert len(res.json()["turns"]) == 2

    # 4. Delete history
    res = client.delete("/api/history")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    res = client.get("/api/history")
    assert res.json() == {"turns": []}


def test_history_post_keeps_the_client_turn_id(client: TestClient) -> None:
    turn_id = "t_1750000000000_abc12"
    res = client.post("/api/history", json={"id": turn_id, "role": "user", "text": "Cześć Gaja", "lang": "pl"})
    assert res.status_code == 200, res.text
    assert res.json()["turn"]["id"] == turn_id
    turns = client.get("/api/history").json()["turns"]
    assert [t["id"] for t in turns] == [turn_id]


@pytest.mark.parametrize("bad_id", ["../x", "a" * 65, "with space", ""])
def test_history_post_rejects_an_unusable_client_id(client: TestClient, bad_id: str) -> None:
    res = client.post("/api/history", json={"id": bad_id, "role": "user", "text": "x"})
    assert res.status_code == 422


def test_history_rotation(history_path: Path) -> None:
    # Max turns is 5 in this test
    for i in range(10):
        append_turn(history_path, TurnItem(role="user", text=f"Turn {i}"), max_stored=5)

    turns = read_turns(history_path, limit=20)
    assert len(turns) == 5
    assert turns[0]["text"] == "Turn 5"
    assert turns[-1]["text"] == "Turn 9"


def test_history_origin_check(client: TestClient) -> None:
    # Allowed origin
    res = client.get("/api/history", headers={"Origin": "http://localhost:8790"})
    assert res.status_code == 200

    # Forbidden origin
    res = client.get("/api/history", headers={"Origin": "http://evil.example"})
    assert res.status_code == 403


def test_devices_endpoint(client: TestClient) -> None:
    record_device_activity("atom-salon", is_wake=True)
    res = client.get("/api/devices")
    assert res.status_code == 200
    body = res.json()
    assert "devices" in body
    assert isinstance(body["devices"], list)
    assert "quiet_now" in body
    assert "quiet_hours" in body

    # Origin check
    res = client.get("/api/devices", headers={"Origin": "http://evil.example"})
    assert res.status_code == 403


def test_devices_endpoint_quiet_hours_refusal_and_force(client: TestClient) -> None:
    client.app.state.quiet_hours = "22:00-07:00"
    client.app.state.quiet_hours_now = datetime.time(23, 0)

    res = client.get("/api/devices")
    assert res.status_code == 200
    assert res.json()["quiet_now"] is True
    assert res.json()["quiet_hours"] == "22:00-07:00"

    # POST /say without force returns 409 naming the window
    res_say = client.post("/api/devices/atom-salon/say", json={"text": "Halo"})
    assert res_say.status_code == 409
    assert "22:00-07:00" in res_say.json()["detail"]

    # POST /play without force returns 409 naming the window
    res_play = client.post("/api/devices/atom-salon/play", json={"sound": "chime"})
    assert res_play.status_code == 409
    assert "22:00-07:00" in res_play.json()["detail"]

    # POST /say with force bypasses quiet hours (proceeds to device check -> 404 because not connected in this test)
    res_say_forced = client.post("/api/devices/atom-salon/say", json={"text": "Halo", "force": True})
    assert res_say_forced.status_code == 404

    # POST /play with force bypasses quiet hours (proceeds to sound load & device check -> 404 because not connected)
    res_play_forced = client.post("/api/devices/atom-salon/play", json={"sound": "chime", "force": True})
    assert res_play_forced.status_code == 404


def test_diagnostics_endpoint(client: TestClient) -> None:
    WAKE_LOG.clear()
    WAKE_LOG.append({
        "ts": 1234567.8,
        "device_id": "test-dev",
        "word": "hej gaja",
        "score": 0.95,
        "accepted": True,
        "reason": "ok",
        "transcript": "hej gaja pogoda",
    })

    # Record some metrics
    metrics: MetricsRecorder = client.app.state.services.metrics
    metrics.record("s1", TurnMetrics(turn=1, kind="audio", stt_ms=120.0, e2e_first_audio_ms=850.0))

    res = client.get("/api/diagnostics")
    assert res.status_code == 200
    body = res.json()
    assert "gpu" in body
    assert "stages" in body
    assert "wakes" in body
    assert len(body["wakes"]) == 1
    assert body["wakes"][0]["device_id"] == "test-dev"
    assert body["stages"]["turns"] >= 1

    # Origin check
    res = client.get("/api/diagnostics", headers={"Origin": "http://evil.example"})
    assert res.status_code == 403
