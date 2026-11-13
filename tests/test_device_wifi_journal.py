"""Server-side Wi-Fi event journal (`wifi_events`, protocol v1.13): logging, diagnostics, malformed tolerance."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja.api.diagnostics import create_diagnostics_router
from gaja.config import Settings
from gaja.devices import wifi_journal as wifi_journal_module
from gaja.metrics import MetricsRecorder
from tests.test_device_protocol import HELLO, TOKEN, connect, device_client, registry  # noqa: F401
from tests.test_ws_protocol import is_type, read_until


def _ping(ws: Any) -> None:
    ws.send_text(json.dumps({"type": "ping"}))
    read_until(ws, is_type("pong"))


def test_wifi_events_accepted_logged_with_wall_clock_and_previous_boot(device_client, caplog, monkeypatch) -> None:  # noqa: F401
    wifi_journal_module.WIFI_EVENTS.clear()
    fixed_now = 1_700_000_100.500
    monkeypatch.setattr(wifi_journal_module.time, "time", lambda: fixed_now)

    client, _ = device_client()
    with caplog.at_level("INFO"), client, client.websocket_connect("/device") as ws:
        connect(ws, caps={**HELLO["caps"], "wifi_journal": True})
        ws.send_text(json.dumps({
            "type": "wifi_events",
            "boot": 3,
            "uptime_ms": 60_000,
            "reset_reason": "panic",
            "dropped": 2,
            "events": [
                {
                    "ms": 95_000,
                    "boot": 2,
                    "kind": "disconnected",
                    "reason": 2,
                    "bssid": "d8:32:14:47:e7:e2",
                    "rssi": -68,
                },
                {
                    "ms": 12_500,
                    "boot": 3,
                    "kind": "scan_done",
                    "reason": 4,
                },
                {
                    "ms": 15_000,
                    "boot": 3,
                    "kind": "connected",
                    "bssid": "d8:32:14:47:e7:e2",
                    "channel": 8,
                },
                {
                    "ms": 16_000,
                    "boot": 3,
                    "kind": "got_ip",
                    "reason": 1,
                },
                {
                    "ms": 50_000,
                    "boot": 3,
                    "kind": "disconnected",
                    "reason": 201,
                },
            ],
        }))
        _ping(ws)

    info_lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "gaja.devices.wifi_journal" and r.levelno == logging.INFO
    ]
    assert len(info_lines) == 5

    # Previous boot event uses "previous boot N" and maps reason code 2 -> AUTH_EXPIRE.
    assert "atom-1" in info_lines[0]
    assert "previous boot 2" in info_lines[0]
    assert "AUTH_EXPIRE" in info_lines[0]
    assert "d8:32:14:47:e7:e2" in info_lines[0]
    assert "-68" in info_lines[0]

    # Current boot events compute wall-clock estimate: fixed_now - (60_000 - 15_000) / 1000 = fixed_now - 45.0
    expected_ts = fixed_now - 45.0
    expected_clock = datetime.fromtimestamp(expected_ts).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert expected_clock in info_lines[2]
    assert "connected" in info_lines[2]
    assert "ch=8" in info_lines[2] or "channel=8" in info_lines[2]

    # Reason code 201 -> NO_AP_FOUND
    assert "NO_AP_FOUND" in info_lines[4]


def test_wifi_events_kept_in_diagnostics_per_device(device_client, tmp_path) -> None:  # noqa: F401
    wifi_journal_module.WIFI_EVENTS.clear()
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        connect(ws, caps={**HELLO["caps"], "wifi_journal": True})
        ws.send_text(json.dumps({
            "type": "wifi_events",
            "boot": 2,
            "uptime_ms": 20_000,
            "reset_reason": "software",
            "dropped": 1,
            "events": [
                {"ms": 5_000, "boot": 2, "kind": "disconnected", "reason": 200, "bssid": "aa:bb:cc:dd:ee:ff", "rssi": -71},
                {"ms": 10_000, "boot": 2, "kind": "got_ip", "reason": 0},
            ],
        }))
        _ping(ws)

    app = FastAPI()
    app.include_router(create_diagnostics_router(lambda _o: True))

    class DummyServices:
        def __init__(self) -> None:
            self.cfg = Settings()
            self.metrics = MetricsRecorder(tmp_path / "turns.jsonl", window=10)

    app.state.services = DummyServices()
    diag = TestClient(app).get("/api/diagnostics")
    assert diag.status_code == 200
    body = diag.json()
    assert "wifi_events" in body
    dev = body["wifi_events"]["atom-1"]
    assert dev["boot"] == 2
    assert dev["reset_reason"] == "software"
    assert dev["dropped"] == 1
    assert len(dev["events"]) == 2
    assert dev["events"][0]["reason_name"] == "BEACON_TIMEOUT"
    assert dev["events"][0]["bssid"] == "aa:bb:cc:dd:ee:ff"
    assert dev["events"][1]["kind"] == "got_ip"


def test_malformed_wifi_events_are_ignored_gracefully(device_client) -> None:  # noqa: F401
    wifi_journal_module.WIFI_EVENTS.clear()
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        for bad in (
            {"type": "wifi_events"},
            {"type": "wifi_events", "boot": "not-an-int", "uptime_ms": 1000, "events": []},
            {"type": "wifi_events", "boot": True, "uptime_ms": 1000, "events": []},
            {"type": "wifi_events", "boot": 1, "uptime_ms": -5, "events": []},
            {"type": "wifi_events", "boot": 1, "uptime_ms": 1000, "events": "not-a-list"},
            # Out-of-range counters (devices send uint32) once crashed the wall-clock estimate.
            {"type": "wifi_events", "boot": 1, "uptime_ms": 10**14, "events": [{"ms": 1, "boot": 1, "kind": "connected"}]},
            {"type": "wifi_events", "boot": 1, "uptime_ms": 10**400, "events": [{"ms": 1, "boot": 1, "kind": "connected"}]},
            {"type": "wifi_events", "boot": 1, "uptime_ms": 1000, "events": [{"ms": 2**32, "boot": 1, "kind": "connected"}]},
            {"type": "wifi_events", "boot": 2**40, "uptime_ms": 1000, "events": []},
            {
                "type": "wifi_events",
                "boot": 1,
                "uptime_ms": 1000,
                "events": ["bad-item", {"ms": "nope", "kind": "disconnected"}, {"ms": 10, "kind": 123}],
            },
        ):
            ws.send_text(json.dumps(bad))
        # No error message was sent back; ping gets pong directly.
        ws.send_text(json.dumps({"type": "ping"}))
        msgs = read_until(ws, is_type("pong"))
        assert not [m for m in msgs if m.get("type") == "error"]
        assert msgs[-1]["type"] == "pong"
    stored = wifi_journal_module.get_wifi_events_log().get("atom-1", {"events": []})["events"]
    assert all(ev["ms"] <= wifi_journal_module.U32_MAX for ev in stored)


def test_wifi_events_batches_are_rate_limited(device_client, caplog) -> None:  # noqa: F401
    from gaja.devices import controller as controller_module

    wifi_journal_module.WIFI_EVENTS.clear()
    client, _ = device_client()
    limit = controller_module.WIFI_EVENTS_PER_MINUTE
    with client, client.websocket_connect("/device") as ws:
        connect(ws, caps={**HELLO["caps"], "wifi_journal": True})
        for i in range(limit + 10):
            ws.send_text(json.dumps({
                "type": "wifi_events",
                "boot": 1,
                "uptime_ms": 5000,
                "events": [{"ms": 1000 + i, "boot": 1, "kind": "scan_done", "reason": 0}],
            }))
        _ping(ws)
    stored = wifi_journal_module.get_wifi_events_log()["atom-1"]["events"]
    assert len(stored) == limit


def test_absent_wifi_journal_capability_changes_nothing(device_client) -> None:  # noqa: F401
    wifi_journal_module.WIFI_EVENTS.clear()
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ready = connect(ws)  # HELLO has no caps.wifi_journal
        assert ready["type"] == "ready"
        _ping(ws)
    assert wifi_journal_module.get_wifi_events_log() == {}


def test_wifi_events_diagnostics_buffer_is_bounded(device_client) -> None:  # noqa: F401
    wifi_journal_module.WIFI_EVENTS.clear()
    client, _ = device_client()
    total = wifi_journal_module.MAX_EVENTS_PER_DEVICE + 25
    with client, client.websocket_connect("/device") as ws:
        connect(ws, caps={**HELLO["caps"], "wifi_journal": True})
        for offset in range(0, total, 25):
            batch = [
                {"ms": 1000 + i, "boot": 1, "kind": "scan_done", "reason": i}
                for i in range(offset, min(total, offset + 25))
            ]
            ws.send_text(json.dumps({
                "type": "wifi_events",
                "boot": 1,
                "uptime_ms": 5000,
                "reset_reason": "power-on",
                "dropped": 0,
                "events": batch,
            }))
        _ping(ws)
    stored = wifi_journal_module.get_wifi_events_log()["atom-1"]["events"]
    assert len(stored) == wifi_journal_module.MAX_EVENTS_PER_DEVICE
    assert stored[0]["reason"] == 25
    assert stored[-1]["reason"] == total - 1

