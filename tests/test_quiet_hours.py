"""Unit tests for speaker quiet hours: window parsing, edge cases, 409 refusal, force bypass, and timers."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja.api.devices import create_devices_router
from gaja.config import Settings
from gaja.devices.quiet_hours import (
    format_quiet_hours_error,
    is_quiet_hour,
    parse_quiet_hours,
    validate_quiet_hours,
)
from gaja.devices.registry import DeviceInfo
from gaja.pipeline import Session
from tests.test_device_protocol import FAST, HELLO, TOKEN
from tests.test_speaker_mcp import FakeListenerWithSession, RecordingOutbox


# ---- 1. Window parsing, validation, and edges ----------------------------------------------------


def test_validate_quiet_hours_valid() -> None:
    assert validate_quiet_hours("22:00-07:00") == "22:00-07:00"
    assert validate_quiet_hours("00:00-23:59") == "00:00-23:59"
    assert validate_quiet_hours("08:30-17:45") == "08:30-17:45"
    assert validate_quiet_hours("") == ""
    assert validate_quiet_hours("   ") == ""


@pytest.mark.parametrize(
    "bad_value",
    [
        "24:00-07:00",
        "22:60-07:00",
        "22:00-24:00",
        "22:00",
        "22-07",
        "random-string",
        "22:00-22:00",  # equal start and end
        "10:00-10:00",
    ],
)
def test_validate_quiet_hours_invalid(bad_value: str) -> None:
    with pytest.raises(ValueError):
        validate_quiet_hours(bad_value)


def test_is_quiet_hour_wrap_around_and_edges() -> None:
    window = "22:00-07:00"

    # Edges: 22:00 is quiet, 07:00 is not
    assert is_quiet_hour(window, datetime.time(21, 59)) is False
    assert is_quiet_hour(window, datetime.time(22, 0)) is True
    assert is_quiet_hour(window, datetime.time(22, 1)) is True
    assert is_quiet_hour(window, datetime.time(23, 59, 59)) is True
    assert is_quiet_hour(window, datetime.time(0, 0)) is True
    assert is_quiet_hour(window, datetime.time(3, 30)) is True
    assert is_quiet_hour(window, datetime.time(6, 59)) is True
    assert is_quiet_hour(window, datetime.time(7, 0)) is False
    assert is_quiet_hour(window, datetime.time(7, 1)) is False
    assert is_quiet_hour(window, datetime.time(12, 0)) is False


def test_is_quiet_hour_non_wrapping() -> None:
    window = "08:00-16:00"
    assert is_quiet_hour(window, datetime.time(7, 59)) is False
    assert is_quiet_hour(window, datetime.time(8, 0)) is True
    assert is_quiet_hour(window, datetime.time(12, 0)) is True
    assert is_quiet_hour(window, datetime.time(15, 59)) is True
    assert is_quiet_hour(window, datetime.time(16, 0)) is False
    assert is_quiet_hour(window, datetime.time(16, 1)) is False


def test_is_quiet_hour_disabled_when_empty() -> None:
    assert is_quiet_hour("", datetime.time(23, 0)) is False
    assert is_quiet_hour("", datetime.time(3, 0)) is False
    assert is_quiet_hour("", datetime.time(12, 0)) is False


def test_is_quiet_hour_with_datetime() -> None:
    window = "22:00-07:00"
    dt_quiet = datetime.datetime(2026, 10, 17, 23, 15)
    dt_day = datetime.datetime(2026, 10, 17, 14, 0)
    assert is_quiet_hour(window, dt_quiet) is True
    assert is_quiet_hour(window, dt_day) is False


# ---- 2. HTTP 409 on /say and /play, force bypass, status reporting -------------------------------


def _make_app(listener: Any = None, quiet_hours: str = "22:00-07:00", current_time: datetime.time | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(create_devices_router(lambda origin: origin is None, quiet_hours=quiet_hours))
    app.state.device_listener = listener
    if current_time is not None:
        app.state.quiet_hours_now = current_time
    return app


def test_get_devices_reports_quiet_now_and_window() -> None:
    # 1. At 23:00 -> quiet_now: True
    app_night = _make_app(quiet_hours="22:00-07:00", current_time=datetime.time(23, 0))
    client_night = TestClient(app_night)
    res = client_night.get("/api/devices")
    assert res.status_code == 200
    data = res.json()
    assert data["quiet_now"] is True
    assert data["quiet_hours"] == "22:00-07:00"

    # 2. At 14:00 -> quiet_now: False
    app_day = _make_app(quiet_hours="22:00-07:00", current_time=datetime.time(14, 0))
    client_day = TestClient(app_day)
    res_day = client_day.get("/api/devices")
    assert res_day.status_code == 200
    data_day = res_day.json()
    assert data_day["quiet_now"] is False
    assert data_day["quiet_hours"] == "22:00-07:00"


def test_say_and_play_refuse_during_quiet_hours_with_409() -> None:
    app = _make_app(quiet_hours="22:00-07:00", current_time=datetime.time(23, 30))
    client = TestClient(app)

    # /say refused with 409 and JSON error naming the window
    res_say = client.post("/api/devices/atom-pokoj/say", json={"text": "Dobry wieczór"})
    assert res_say.status_code == 409
    body_say = res_say.json()
    assert "detail" in body_say
    assert "22:00-07:00" in body_say["detail"]

    # /play refused with 409 and JSON error naming the window
    res_play = client.post("/api/devices/atom-pokoj/play", json={"sound": "chime"})
    assert res_play.status_code == 409
    body_play = res_play.json()
    assert "detail" in body_play
    assert "22:00-07:00" in body_play["detail"]


@pytest.mark.asyncio
async def test_say_and_play_force_bypasses_quiet_hours(make_services) -> None:
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    listener = FakeListenerWithSession("atom-pokoj", session)
    # Night clock: 23:00
    app = _make_app(listener=listener, quiet_hours="22:00-07:00", current_time=datetime.time(23, 0))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Unforced -> 409
        unforced = await ac.post("/api/devices/atom-pokoj/play", json={"sound": "chime"})
        assert unforced.status_code == 409

        # 2. Forced -> 200 (bypasses quiet hours)
        forced = await ac.post("/api/devices/atom-pokoj/play", json={"sound": "chime", "force": True})
        assert forced.status_code == 200
        assert forced.json() == {"ok": True, "sound": "chime"}

        # 3. Say forced -> 200
        forced_say = await ac.post("/api/devices/atom-pokoj/say", json={"text": "Ważna wiadomość", "force": True})
        assert forced_say.status_code == 200
        assert forced_say.json() == {"ok": True}

    await session.close()


# ---- 3. Timers unaffected during quiet hours -----------------------------------------------------


@pytest.mark.asyncio
async def test_timers_unaffected_during_quiet_hours(make_services) -> None:
    """House rule: timers and reminders set by the user ring as before, even during quiet hours."""
    services = make_services(cfg={"devices": {**FAST, "quiet_hours": "22:00-07:00"}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    # Simulate quiet hour (e.g. 23:30)
    fake_clock = datetime.time(23, 30)
    with patch("gaja.devices.quiet_hours.datetime") as mock_dt:
        mock_dt.time = datetime.time
        mock_dt.datetime.now.return_value = datetime.datetime(2026, 10, 17, 23, 30)

        # Trigger timer directly on the session as notify_sessions does
        timer_payload = {"id": "t_egg1", "label": "jajka", "kind": "timer"}
        await session.on_timer(timer_payload, note=None, speak=True)

    json_msgs = [m for m in raw_out.sent if isinstance(m, dict)]
    byte_chunks = [m for m in raw_out.sent if isinstance(m, bytes)]
    msg_types = [m["type"] for m in json_msgs]

    assert "timer" in msg_types
    assert "audio_start" in msg_types
    assert "audio_end" in msg_types
    assert byte_chunks, "Audio stream must be produced for timer during quiet hours"

    timer_msg = next(m for m in json_msgs if m["type"] == "timer")
    assert timer_msg["id"] == "t_egg1"
    assert timer_msg["label"] == "jajka"

    metrics_msg = next(m for m in json_msgs if m["type"] == "metrics")
    assert metrics_msg["kind"] == "notification"
    assert metrics_msg["route"] == "timer"

    await session.close()
