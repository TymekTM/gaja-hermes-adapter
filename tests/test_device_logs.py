"""Firmware log forwarding (`log` messages, firmware >= 0.2.4) and Hermes stderr levels."""

from __future__ import annotations

import json
import logging

from gaja.devices import controller as controller_module
from gaja.devices.controller import clean_remote_log, remote_log_level
from gaja.hermes.client import stderr_level
from tests.test_device_protocol import TOKEN, connect, device_client, registry  # noqa: F401  (fixtures)
from tests.test_ws_protocol import is_type, read_until


def test_clean_remote_log_strips_colour_and_control_bytes_and_bounds_the_line() -> None:
    assert clean_remote_log("\x1b[0;32m[I][gaja_device:12]: link up\x1b[0m\r\n") == "[I][gaja_device:12]: link up"
    assert clean_remote_log("a\x00b\tc") == "a b c"
    assert len(clean_remote_log("x" * 5000)) == controller_module.REMOTE_LOG_MAX_CHARS


def test_clean_remote_log_withholds_lines_that_may_carry_a_secret() -> None:
    for line in ("wifi password: hunter2", "Token=abc", "PSK set", "api secret"):
        assert clean_remote_log(line) == "[line withheld: may contain a secret]"


def test_remote_log_level_maps_esphome_levels() -> None:
    assert remote_log_level(1) == logging.ERROR
    assert remote_log_level(2) == logging.WARNING
    for level in (3, 4, 5, 6, None, "x"):
        assert remote_log_level(level) == logging.INFO


def test_stderr_level_follows_hermes_own_level_but_never_below_info() -> None:
    assert stderr_level("2026-10-16 20:51:12 [INFO] acp: session ready") == logging.INFO
    assert stderr_level("[DEBUG] noisy") == logging.INFO
    assert stderr_level("[ERROR] tool failed") == logging.ERROR
    assert stderr_level("Traceback (most recent call last):") == logging.WARNING


def _ping(ws) -> None:
    ws.send_text(json.dumps({"type": "ping"}))
    read_until(ws, is_type("pong"))


def test_device_log_lines_reach_the_journal_cleaned_and_unanswered(device_client, caplog) -> None:  # noqa: F811
    client, _ = device_client()
    with caplog.at_level("INFO", logger="gaja.devices.remote"), client, client.websocket_connect("/device") as ws:
        ready = connect(ws)
        assert ready["device"]["logs"] is True
        ws.send_text(json.dumps({"type": "log", "level": 2, "msg": "\x1b[33mwifi: disconnected\x1b[0m"}))
        ws.send_text(json.dumps({"type": "log", "level": 3, "msg": "token " + TOKEN}))
        ws.send_text(json.dumps({"type": "log", "level": 3, "msg": 42}))
        _ping(ws)
    remote = [r for r in caplog.records if r.name == "gaja.devices.remote"]
    assert [(r.levelno, r.getMessage()) for r in remote] == [
        (logging.WARNING, "[atom-1] wifi: disconnected"),
        (logging.INFO, "[atom-1] [line withheld: may contain a secret]"),
    ]
    assert TOKEN not in caplog.text


def test_device_log_flood_is_capped_per_minute(device_client, caplog, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(controller_module, "REMOTE_LOG_PER_MINUTE", 3)
    client, _ = device_client()
    with caplog.at_level("INFO", logger="gaja.devices.remote"), client, client.websocket_connect("/device") as ws:
        connect(ws)
        for i in range(10):
            ws.send_text(json.dumps({"type": "log", "level": 3, "msg": f"line {i}"}))
        _ping(ws)
    remote = [r.getMessage() for r in caplog.records if r.name == "gaja.devices.remote"]
    assert remote == ["[atom-1] line 0", "[atom-1] line 1", "[atom-1] line 2"]
