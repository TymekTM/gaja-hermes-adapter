"""Tests for STT timing, wait duration, page faults, and slow STT logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from starlette.testclient import TestClient

from gaja.llm.client import TextDelta
from gaja.server import create_app
from tests.conftest import FakeLLM, FakeSTT
from tests.test_ws_protocol import is_listening, is_type, read_until, silence_frames, speech_frames


def test_audio_turn_records_stt_timing_fields(make_services, tmp_path: Path) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")]])
    stt = FakeSTT("ile to dwa plus dwa", duration_ms=250.0, wait_ms=15.0)
    app = create_app(services=make_services(llm=llm, stt=stt))

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        for frame in speech_frames(0.8):
            ws.send_bytes(frame)
        for frame in silence_frames(0.8):
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))

    metrics_msg = next(m for m in messages if isinstance(m, dict) and m.get("type") == "metrics")
    assert metrics_msg["stt_ms"] == 250.0
    assert metrics_msg["stt_wait_ms"] == 15.0
    assert isinstance(metrics_msg["stt_majflt"], int)
    assert metrics_msg["stt_majflt"] >= 0
    assert isinstance(metrics_msg["stt_audio_s"], float)
    assert metrics_msg["stt_audio_s"] > 0.5

    # Check written to logs/turns.jsonl
    turns_file = tmp_path / "turns.jsonl"
    lines = turns_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged_turn = json.loads(lines[0])
    assert logged_turn["stt_ms"] == 250.0
    assert logged_turn["stt_wait_ms"] == 15.0
    assert logged_turn["stt_majflt"] == metrics_msg["stt_majflt"]
    assert logged_turn["stt_audio_s"] == metrics_msg["stt_audio_s"]


def test_text_turn_leaves_stt_timing_fields_none(make_services) -> None:
    llm = FakeLLM([[TextDelta("Witaj.")]])
    app = create_app(services=make_services(llm=llm))

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "cześć"}))
        messages = read_until(ws, is_type("metrics"))

    metrics_msg = next(m for m in messages if isinstance(m, dict) and m.get("type") == "metrics")
    assert metrics_msg["stt_ms"] is None
    assert metrics_msg["stt_wait_ms"] is None
    assert metrics_msg["stt_majflt"] is None
    assert metrics_msg["stt_audio_s"] is None


def test_slow_stt_logs_info_line_when_over_1500ms(make_services, caplog) -> None:
    llm = FakeLLM([[TextDelta("Odpowiedź.")]])
    stt = FakeSTT("test wolnego stt", duration_ms=1850.5, wait_ms=420.0)
    app = create_app(services=make_services(llm=llm, stt=stt))

    with caplog.at_level(logging.INFO, logger="gaja.pipeline"):
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
            read_until(ws, is_listening)
            for frame in speech_frames(0.6):
                ws.send_bytes(frame)
            for frame in silence_frames(0.8):
                ws.send_bytes(frame)
            read_until(ws, is_type("metrics"))

    slow_stt_logs = [r for r in caplog.records if r.name == "gaja.pipeline" and "slow STT:" in r.getMessage()]
    assert len(slow_stt_logs) == 1
    msg = slow_stt_logs[0].getMessage()
    assert "stt_ms=1850.5" in msg
    assert "stt_wait_ms=420.0" in msg
    assert "stt_majflt=" in msg
    assert "stt_audio_s=" in msg


def test_fast_stt_does_not_log_slow_stt_info(make_services, caplog) -> None:
    llm = FakeLLM([[TextDelta("Szybka odpowiedź.")]])
    stt = FakeSTT("szybkie stt", duration_ms=300.0, wait_ms=2.0)
    app = create_app(services=make_services(llm=llm, stt=stt))

    with caplog.at_level(logging.INFO, logger="gaja.pipeline"):
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
            read_until(ws, is_listening)
            for frame in speech_frames(0.6):
                ws.send_bytes(frame)
            for frame in silence_frames(0.8):
                ws.send_bytes(frame)
            read_until(ws, is_type("metrics"))

    slow_stt_logs = [r for r in caplog.records if r.name == "gaja.pipeline" and "slow STT:" in r.getMessage()]
    assert len(slow_stt_logs) == 0
