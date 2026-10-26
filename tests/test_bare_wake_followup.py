"""The bare-wake follow-up flow as the live device runs it (2026-10-22 incident).

Production serves a full-duplex firmware as half duplex (speech_barge_in off -> ready aec=false)
and answers a bare "Hej Gaja" with ack="none": no "Słucham?", no second beep -- the follow-up
window opens silently right after the wake turn. The user starts the question immediately, and on
the device part of it lands in the half-duplex gap before the window opens. Whatever fragment the
server ends up endpointing must never disappear without a trace: a follow-up turn that ends
without a reply has to say WHY in the journal (no transcript text -- privacy).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from gaja.devices import listener as listener_module
from gaja.devices.listener import create_device_app
from gaja.devices.registry import DeviceRegistry
from gaja.llm.client import TextDelta
from gaja.pipeline import is_garbage_followup
from tests.conftest import FakeLLM
from tests.test_device_protocol import TOKEN, ScriptedSTT
from tests.test_ws_protocol import is_type, read, read_until, silence_frames, speech_frames

# Production-like: ack none, speculation on (config.toml spec_silence_ms=250), the device served
# as half duplex (speech_barge_in off), verify on with the default window.
LIVE = {
    "follow_up_ms": 300,
    "listen_grace_s": 0.05,
    "wake_timeout_s": 5.0,
    "listen_timeout_s": 5.0,
    "speech_barge_in": False,
    "ack": "none",
    "verify_wake": True,
    "verify_ms": 2500,
}


@pytest.fixture(autouse=True)
def _no_auth_delay(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 0.0)


@pytest.fixture
def registry(tmp_path) -> DeviceRegistry:
    path = tmp_path / "devices.toml"
    path.write_text(f'[devices.atom-1]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    path.chmod(0o600)
    return DeviceRegistry(path)


def _client(make_services, registry, *, stt: ScriptedSTT, llm: FakeLLM, cfg: dict[str, Any] | None = None):
    services = make_services(
        cfg={"devices": {**LIVE, **(cfg or {})}, "vad": {"spec_silence_ms": 250}},
        stt=stt,
        llm=llm,
    )
    from fastapi.testclient import TestClient

    client = TestClient(create_device_app(services, registry), client=("127.0.0.1", 50000))
    return client, services


def _hello() -> dict[str, Any]:
    return {
        "type": "hello",
        "client": "device",
        "device_id": "atom-1",
        "token": TOKEN,
        "lang": "pl",
        "fw": "gaja-esphome/0.3.3",
        "caps": {"aec": True, "out_rates": [16000], "preroll_ms": 1500},
    }


def _bare_wake(ws: Any) -> list[Any]:
    """The live choreography: wake, the phrase in the pre-roll burst + live tail, then silence."""
    ws.send_text(json.dumps({**_hello(), "caps": {"aec": True, "out_rates": [16000], "preroll_ms": 1500}}))
    ready = read(ws)
    assert ready["device"]["aec"] is False  # served as half duplex, like the live device
    ws.send_text(json.dumps({"type": "wake", "word": "hej_gaja", "score": 0.97, "preroll_ms": 300}))
    for frame in speech_frames(0.3) + silence_frames(0.8):
        ws.send_bytes(frame)
    messages = read_until(ws, is_type("follow_up"))
    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    assert "wake_accepted" in kinds and "listen_end" in kinds
    metrics = next(m for m in messages if isinstance(m, dict) and m["type"] == "metrics")
    assert metrics["route"] == "wake"  # the bare phrase: acknowledged, nothing asked yet
    return messages


def test_empty_follow_up_transcript_is_logged_and_the_window_reopens(
    make_services, registry, tmp_path, caplog
) -> None:
    """A follow-up utterance whose transcript comes back empty (a torn half-duplex fragment on the
    live device) ends without a reply: the journal must say so, and the window must reopen so the
    user can simply repeat the question."""
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("Hej Gaja!", "", "jaka jest pogoda")
    client, services = _client(make_services, registry, stt=stt, llm=llm)
    with client, client.websocket_connect("/device") as ws:
        _bare_wake(ws)
        with caplog.at_level(logging.INFO, logger="gaja.pipeline"):
            # The question, endpointed with a fragment the STT cannot decode (scripted "").
            for frame in speech_frames(0.4) + silence_frames(0.9):
                ws.send_bytes(frame)
            messages = read_until(ws, is_type("follow_up"))
    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    # The turn itself stays invisible to the device (no transcript, no reply), as before...
    assert "transcript" not in kinds and "utterance" not in kinds
    assert "listen_end" in kinds and kinds[-1] == "follow_up"
    # ...but the journal must now say why the follow-up ended without a reply, without the text.
    why = [r for r in caplog.records if r.levelno == logging.INFO and "without a reply" in r.getMessage()]
    assert why, "no journal line explains the follow-up turn that ended without a reply"
    assert "empty transcript" in why[0].getMessage()
    assert "jaka" not in why[0].getMessage().lower()  # the transcript text never reaches the journal
    assert len(llm.calls) == 0  # nothing went to the LLM while the transcript was empty


def test_follow_up_after_an_empty_fragment_is_still_answered(make_services, registry, tmp_path) -> None:
    """The window reopens after the dropped fragment, and the repeated question gets its reply."""
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("Hej Gaja!", "", "jaka jest pogoda")
    client, services = _client(make_services, registry, stt=stt, llm=llm)
    with client, client.websocket_connect("/device") as ws:
        _bare_wake(ws)
        for frame in speech_frames(0.4) + silence_frames(0.9):  # the fragment: dropped silently
            ws.send_bytes(frame)
        dropped = read_until(ws, is_type("follow_up"))
        assert "transcript" not in [m["type"] for m in dropped if isinstance(m, dict)]
        for frame in speech_frames(0.5) + silence_frames(0.9):  # the user repeats the question
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))
    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    transcript = next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "jaka jest pogoda"
    assert "utterance" in kinds and "audio_start" in kinds
    assert len(llm.calls) == 1
    metrics = next(m for m in messages if isinstance(m, dict) and m["type"] == "metrics")
    assert metrics["route"] == "local"


def test_cancelled_speculation_on_a_follow_up_is_logged(make_services, registry, tmp_path, caplog) -> None:
    """A mid-utterance pause speculates, the user keeps talking, the speculation is cancelled:
    the reply that was dropped must be explained in the journal (no transcript text)."""
    llm = FakeLLM([[TextDelta("Jest słonecznie.")], [TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("Hej Gaja!", "jaka jest", "jaka jest pogoda")
    client, services = _client(make_services, registry, stt=stt, llm=llm)
    with client, client.websocket_connect("/device") as ws:
        _bare_wake(ws)
        with caplog.at_level(logging.INFO, logger="gaja.pipeline"):
            # Speech, a 250 ms+ pause (speculation fires), speech again (it is cancelled), silence.
            for frame in speech_frames(0.3) + silence_frames(0.35) + speech_frames(0.3) + silence_frames(0.9):
                ws.send_bytes(frame)
            messages = read_until(ws, is_type("metrics"))
    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    assert "utterance" in kinds  # the completed question was answered
    assert len(llm.calls) >= 1
    cancelled = [r for r in caplog.records if r.levelno == logging.INFO and "cancelled" in r.getMessage()]
    assert cancelled, "no journal line explains the cancelled speculative turn"
    assert "jaka" not in cancelled[0].getMessage().lower()


# ---- garbage follow-up transcripts (device log 2026-10-26: a stray "PU." got answered) --------------


@pytest.mark.parametrize("text", ["PU.", "pu", "a", "2", "...", "hm", "a a"])
def test_garbage_followup_detection(text: str) -> None:
    assert is_garbage_followup(text)


@pytest.mark.parametrize("text", ["tak", "Nie!", "dzięki", "dziękuję bardzo", "ok", "pa", "yo", "no", "hmm", "głośniej"])
def test_garbage_followup_keeps_real_short_answers(text: str) -> None:
    assert not is_garbage_followup(text)


def test_garbage_follow_up_transcript_ends_quietly_and_the_window_reopens(
    make_services, registry, tmp_path, caplog
) -> None:
    """STT heard "PU." in the follow-up window (noise or an echo tail): no reply, no LLM call, no
    transcript event — the window reopens and the journal says why, without the transcript text."""
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("Hej Gaja!", "PU.")
    client, services = _client(make_services, registry, stt=stt, llm=llm)
    with client, client.websocket_connect("/device") as ws:
        _bare_wake(ws)
        with caplog.at_level(logging.INFO, logger="gaja.pipeline"):
            for frame in speech_frames(0.3) + silence_frames(0.9):
                ws.send_bytes(frame)
            messages = read_until(ws, is_type("follow_up"))
    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    assert "transcript" not in kinds and "utterance" not in kinds
    assert kinds[-1] == "follow_up"  # the window reopened, quietly
    why = [r for r in caplog.records if r.levelno == logging.INFO and "without a reply" in r.getMessage()]
    assert any("low-content follow-up transcript" in r.getMessage() for r in why)
    assert "pu" not in why[0].getMessage().lower()  # the transcript text never reaches the journal
    assert len(llm.calls) == 0


def test_short_real_answer_in_the_follow_up_is_still_answered(make_services, registry, tmp_path) -> None:
    """The allow-list protects real short answers: a follow-up "tak" reaches the LLM and is answered."""
    llm = FakeLLM([[TextDelta("Dobrze, robimy to.")]])
    stt = ScriptedSTT("Hej Gaja!", "tak")
    client, services = _client(make_services, registry, stt=stt, llm=llm)
    with client, client.websocket_connect("/device") as ws:
        _bare_wake(ws)
        for frame in speech_frames(0.3) + silence_frames(0.9):
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))
    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    transcript = next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "tak"
    assert "utterance" in kinds
    assert len(llm.calls) == 1
    metrics = next(m for m in messages if isinstance(m, dict) and m["type"] == "metrics")
    assert metrics["route"] == "local"
