"""Protocol v1.8: device endpoint, auth, hello negotiation, wake flow (stubbed STT / LLM / TTS)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gaja.devices import listener as listener_module
from gaja.devices.listener import DeviceListener, create_device_app
from gaja.devices.registry import DeviceRegistry
from gaja.llm.client import TextDelta
from gaja.server import create_app
from gaja.stt.base import Transcript
from tests.conftest import FakeLLM, FakeSTT
from tests.test_ws_protocol import is_listening, is_type, read, read_until, silence_frames, speech_frames

TOKEN = "d" * 40
HELLO: dict[str, Any] = {"type": "hello", "client": "device", "device_id": "atom-1", "token": TOKEN, "lang": "pl",
                         "fw": "0.1.0", "caps": {"aec": False, "out_rates": [16000, 24000], "preroll_ms": 1500}}
FAST = {"follow_up_ms": 50, "listen_grace_s": 0.05, "wake_timeout_s": 5.0, "listen_timeout_s": 5.0, "speech_barge_in": True}


class ScriptedSTT(FakeSTT):
    """Returns the scripted texts in order (the last one repeats) and records what it was given."""

    def __init__(self, *texts: str) -> None:
        super().__init__(texts[0])
        self.texts = list(texts)
        self.calls: list[float] = []

    async def transcribe(
        self,
        pcm: np.ndarray,
        language: str | None = None,
        *,
        hotwords: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        self.last_hotwords = hotwords
        self.last_prompt = prompt
        if not hasattr(self, "hotwords_calls"):
            self.hotwords_calls = []
        self.hotwords_calls.append(hotwords)
        self.calls.append(len(pcm) / 16000)
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return Transcript(text=text, language="pl", duration_ms=3.0, audio_s=len(pcm) / 16000)


@pytest.fixture(autouse=True)
def _no_auth_delay(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 0.0)


@pytest.fixture
def registry(tmp_path) -> DeviceRegistry:
    path = tmp_path / "devices.toml"
    path.write_text(f'[devices.atom-1]\ntoken = "{TOKEN}"\n[devices.atom-2]\ntoken = "{"e" * 40}"\n', encoding="utf-8")
    path.chmod(0o600)
    return DeviceRegistry(path)


@pytest.fixture
def device_client(make_services, registry):
    def factory(*, devices: dict[str, Any] | None = None, **kwargs: Any) -> tuple[TestClient, Any]:
        services = make_services(cfg={"devices": {**FAST, **(devices or {})}}, **kwargs)
        return TestClient(create_device_app(services, registry), client=("127.0.0.1", 50000)), services

    return factory


def assert_closed(ws: Any) -> None:
    assert ws.receive()["type"] == "websocket.close"


def msg_types(messages: list[Any]) -> list[str]:
    return [m["type"] if isinstance(m, dict) else "bytes" for m in messages]


def connect(ws: Any, **overrides: Any) -> dict[str, Any]:
    ws.send_text(json.dumps({**HELLO, **overrides}))
    ready = read(ws)
    assert ready["type"] == "ready", ready
    read_until(ws, is_listening)
    return ready


def send_wake(ws: Any, preroll_s: float = 0.0, score: float = 0.97) -> None:
    ws.send_text(json.dumps({"type": "wake", "word": "hej_gaja", "score": score, "preroll_ms": int(preroll_s * 1000)}))


def utterance(ws: Any, speech_s: float = 1.0, silence_s: float = 0.8) -> None:
    for frame in speech_frames(speech_s) + silence_frames(silence_s):
        ws.send_bytes(frame)


def until(kind: str) -> Any:
    return is_type(kind)


# ---- auth ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [{"token": None}, {"token": "wrong" * 8}, {"token": ""}, {"device_id": "atom-9"}, {"device_id": None}, {"token": 12345}],
)
def test_bad_credentials_are_refused_with_one_message(device_client, override) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, **override}))
        assert read(ws) == {"type": "error", "message": "unauthorized"}
        assert_closed(ws)


def test_token_of_another_device_is_refused(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "device_id": "atom-2"}))
        assert read(ws)["message"] == "unauthorized"


def test_first_message_must_be_a_hello(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        assert read(ws)["message"] == "unauthorized"
    with client, client.websocket_connect("/device") as ws:
        ws.send_bytes(b"\x00\x01")
        assert read(ws)["message"] == "unauthorized"


def test_token_never_reaches_the_logs(device_client, caplog) -> None:
    client, _ = device_client()
    with caplog.at_level("DEBUG"), client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "token": "SECRET-" * 6}))
        read(ws)
    assert "SECRET-" not in caplog.text and TOKEN not in caplog.text


def test_peer_outside_allowed_networks_is_refused(make_services, registry) -> None:
    services = make_services(cfg={"devices": FAST})
    client = TestClient(create_device_app(services, registry), client=("8.8.8.8", 50000))
    with client, pytest.raises(WebSocketDisconnect), client.websocket_connect("/device"):
        pass


def test_device_app_serves_nothing_but_the_device_socket(device_client) -> None:
    client, _ = device_client()
    with client:
        for path in ("/", "/health", "/metrics", "/ws", "/docs", "/openapi.json", "/index.html"):
            assert client.get(path).status_code == 404, path


def test_device_hello_on_the_main_ws_is_refused(make_services) -> None:
    with TestClient(create_app(services=make_services())) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps(HELLO))
        assert read(ws)["message"] == "invalid hello: device clients must use the device endpoint"
        ws.send_text(json.dumps({"type": "hello", "client": "web"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "wake"}))
        assert read(ws) == {"type": "error", "message": "wake is only for device clients"}


def test_web_hello_on_the_device_endpoint_is_refused_after_auth(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "client": "web"}))
        assert read(ws)["message"] == "invalid hello: the device endpoint needs client=device"


def test_reconnect_replaces_the_previous_connection(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as first:
        connect(first)
        with client.websocket_connect("/device") as second:
            connect(second)
            assert_closed(first)
            second.send_text(json.dumps({"type": "ping"}))
            assert read(second) == {"type": "pong"}


# ---- hello / negotiation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offered", "expected"),
    [([16000, 24000], 16000), ([24000, 44100], 24000), ([8000, 22050, 44100], 22050), ([8000], 8000), ([48000], 48000)],
)
def test_hello_negotiates_the_output_rate(device_client, offered, expected) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "caps": {"out_rates": offered}}))
        ready = read(ws)
    assert ready["sample_rate"] == expected
    assert ready["device"]["out_rate"] == expected
    assert ready["device"]["aec"] is False and ready["device"]["verify_wake"] is True


@pytest.mark.parametrize("caps", [{"out_rates": []}, {"out_rates": [96000]}, {"out_rates": ["x"]}, {"preroll_ms": 99999}])
def test_invalid_caps_are_rejected(device_client, caps) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "caps": caps}))
        assert read(ws)["message"].startswith("invalid hello")


def test_second_hello_is_refused(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps(HELLO))
        assert read(ws) == {"type": "error", "message": "hello already sent"}


def test_tts_is_resampled_to_the_device_rate_and_chunked(device_client) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")]])
    client, _ = device_client(devices={"out_chunk_ms": 50}, llm=llm)
    with client, client.websocket_connect("/device") as ws:
        ready = connect(ws, caps={"out_rates": [16000]})
        assert ready["sample_rate"] == 16000
        ws.send_text(json.dumps({"type": "text", "text": "ile to dwa plus dwa"}))
        messages = read_until(ws, until("follow_up"))
    starts = [m for m in messages if isinstance(m, dict) and m["type"] == "audio_start"]
    assert [s["sample_rate"] for s in starts] == [16000]
    audio = [m for m in messages if isinstance(m, bytes)]
    assert all(len(a) <= 1600 for a in audio)  # 50 ms at 16 kHz, 16 bit
    assert sum(len(a) for a in audio) == 3200  # FakeTTS: 2400 samples at 24 kHz = 1600 samples at 16 kHz


def test_audio_at_the_native_rate_is_not_touched(device_client) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")]])
    client, _ = device_client(llm=llm)
    with client, client.websocket_connect("/device") as ws:
        ready = connect(ws, caps={"out_rates": [24000]})
        assert ready["sample_rate"] == 24000
        ws.send_text(json.dumps({"type": "text", "text": "ile to dwa plus dwa"}))
        messages = read_until(ws, until("follow_up"))
    assert sum(len(m) for m in messages if isinstance(m, bytes)) == 4800


# ---- wake flow ----------------------------------------------------------------------------------


def test_wake_with_command_runs_the_turn_and_strips_the_phrase(device_client) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda?")
    client, services = device_client(llm=llm, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=0.5)
        utterance(ws, 1.5)
        messages = read_until(ws, until("follow_up"))
    kinds = msg_types(messages)
    for expected in ("wake_accepted", "listen_end", "transcript", "utterance", "audio_start", "bytes", "audio_end", "metrics"):
        assert expected in kinds, (expected, kinds)
    assert kinds.index("wake_accepted") < kinds.index("transcript") < kinds.index("utterance")
    # non-AEC: the mic is closed when the utterance ended, before the reply is spoken; follow_up comes last
    assert kinds.index("listen_end") < kinds.index("utterance")
    assert kinds[-1] == "follow_up" and messages[-1]["ms"] == 50
    transcript = next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "jaka jest pogoda?" and transcript["heard"] == "Hej Gaja, jaka jest pogoda?"
    assert llm.calls[0][0][-1] == {"role": "user", "content": "jaka jest pogoda?"}
    assert services.metrics.counters["wake_accepted"] == 1
    assert len(stt.calls) == 1  # short utterance: the turn's own transcript is the verification


def test_bare_wake_phrase_is_acknowledged_and_keeps_listening(device_client) -> None:
    llm = FakeLLM()
    client, _ = device_client(devices={"ack": "spoken"}, llm=llm, stt=ScriptedSTT("Hej Gaja."))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        utterance(ws, 0.8)
        messages = read_until(ws, until("follow_up"))
    spoken = [m["text"] for m in messages if isinstance(m, dict) and m["type"] == "utterance"]
    assert spoken == ["Słucham?"]
    assert not llm.calls
    assert next(m for m in messages if isinstance(m, dict) and m["type"] == "metrics")["route"] == "wake"


def test_bare_wake_phrase_without_ack_only_reopens_the_mic(device_client) -> None:
    # ack = "none": the firmware's own beep before the follow-up window is the only cue.
    client, _ = device_client(devices={"ack": "none"}, stt=ScriptedSTT("Hej Gaja."))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        utterance(ws, 0.8)
        messages = read_until(ws, until("follow_up"))
    assert "audio_start" not in msg_types(messages) and "utterance" not in msg_types(messages)
    assert next(m for m in messages if isinstance(m, dict) and m["type"] == "metrics")["route"] == "wake"


def test_follow_up_window_needs_no_wake_phrase(device_client) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")]])
    client, _ = device_client(devices={"follow_up_ms": 2000}, llm=llm, stt=ScriptedSTT("Hej Gaja.", "a ile to dwa plus dwa"))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        utterance(ws, 0.8)
        read_until(ws, until("follow_up"))
        utterance(ws, 0.8)  # no wake in between: the follow-up window is open
        messages = read_until(ws, until("metrics"))
        messages += read_until(ws, until("follow_up"))
    assert [m["text"] for m in messages if isinstance(m, dict) and m["type"] == "transcript"] == ["a ile to dwa plus dwa"]
    assert len(llm.calls) == 1
    assert "wake_accepted" not in msg_types(messages)


def test_wrong_phrase_is_rejected_without_a_turn(device_client) -> None:
    llm = FakeLLM()
    client, services = device_client(llm=llm, stt=ScriptedSTT("Hej Maja, jaka pogoda"))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        utterance(ws, 1.0)
        messages = read_until(ws, until("wake_rejected"))
        assert messages[-1] == {"type": "wake_rejected", "reason": "no_phrase"}
        utterance(ws, 0.5)  # the device went idle; stray frames are ignored
        ws.send_text(json.dumps({"type": "ping"}))
        rest = read_until(ws, until("pong"))
    assert not llm.calls
    assert "follow_up" not in msg_types(messages + rest) and "transcript" not in msg_types(messages + rest)
    assert services.metrics.counters["wake_rejected"] == 1
    assert services.metrics.counters["wake_rejected:no_phrase"] == 1


def test_verification_runs_on_the_first_seconds_while_the_user_still_talks(device_client) -> None:
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda w", "Hej Gaja, jaka jest pogoda w Krakowie?")
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    client, _ = device_client(llm=llm, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=1.5)
        for frame in speech_frames(4.0):
            ws.send_bytes(frame)
        messages = read_until(ws, until("wake_accepted"))
        # accepted while the user is still speaking: no end of speech yet
        assert {"type": "vad", "state": "speech_end"} not in messages
        for frame in silence_frames(0.8):
            ws.send_bytes(frame)
        rest = read_until(ws, until("follow_up"))
    assert stt.calls[0] == pytest.approx(2.5, abs=0.05)  # the first 2.5 s only
    assert len(stt.calls) == 2  # verification + the turn
    transcript = next(m for m in rest if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "jaka jest pogoda w Krakowie?"


def test_early_rejection_stops_a_long_false_wake(device_client) -> None:
    stt = ScriptedSTT("dziękuję za oglądanie")
    llm = FakeLLM()
    client, services = device_client(llm=llm, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=1.5)
        for frame in speech_frames(4.0):
            ws.send_bytes(frame)
        messages = read_until(ws, until("wake_rejected"))
        for frame in speech_frames(1.0) + silence_frames(0.8):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "ping"}))
        rest = read_until(ws, until("pong"))
    assert not llm.calls and len(stt.calls) == 1
    assert "transcript" not in msg_types(messages + rest)
    assert services.metrics.counters["wake_rejected:no_phrase"] == 1


def test_silence_after_a_wake_is_rejected_without_calling_stt(device_client) -> None:
    stt = ScriptedSTT("Hej Gaja")
    client, _ = device_client(stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        for frame in silence_frames(3.0):
            ws.send_bytes(frame)
        messages = read_until(ws, until("wake_rejected"))
    assert messages[-1]["reason"] == "no_speech"
    assert stt.calls == []


def test_a_wake_that_never_gets_audio_times_out(device_client) -> None:
    client, _ = device_client(devices={"wake_timeout_s": 0.2})
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        assert read_until(ws, until("wake_rejected"))[-1]["reason"] == "timeout"


def test_verification_can_be_turned_off(device_client) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("jaka jest pogoda")
    client, _ = device_client(devices={"verify_wake": False}, llm=llm, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        ready = connect(ws)
        assert ready["device"]["verify_wake"] is False
        send_wake(ws)
        assert read(ws) == {"type": "wake_accepted"}
        utterance(ws, 1.0)
        messages = read_until(ws, until("follow_up"))
    assert next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")["text"] == "jaka jest pogoda"
    assert len(llm.calls) == 1


@pytest.mark.parametrize(("mode", "expect_audio"), [("earcon", True), ("none", False)])
def test_ack_modes(device_client, mode, expect_audio) -> None:
    client, _ = device_client(devices={"ack": mode}, stt=ScriptedSTT("Hej Gaja"))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        utterance(ws, 0.8)
        messages = read_until(ws, until("follow_up"))
    kinds = msg_types(messages)
    assert ("audio_start" in kinds) is expect_audio
    assert "utterance" not in kinds


def test_aec_device_keeps_streaming_and_gets_listen_end_when_the_window_expires(device_client) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    client, _ = device_client(llm=llm, stt=ScriptedSTT("Hej Gaja, jaka pogoda"))
    with client, client.websocket_connect("/device") as ws:
        ready = connect(ws, caps={"aec": True, "out_rates": [16000]})
        assert ready["device"]["aec"] is True
        send_wake(ws)
        utterance(ws, 1.0)
        messages = read_until(ws, until("follow_up"))
        assert "listen_end" not in msg_types(messages)
        tail = read_until(ws, until("listen_end"))
    assert "utterance" not in msg_types(tail)


def test_without_speech_barge_in_a_full_duplex_device_is_served_as_half_duplex(device_client) -> None:
    """speech_barge_in off (the default): no echo canceller, the device gets listen_end after the
    utterance like a half-duplex one, and tagged frames still parse (caps.ref)."""
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    client, _ = device_client(llm=llm, stt=ScriptedSTT("Hej Gaja, jaka pogoda"), devices={"speech_barge_in": False})
    with client, client.websocket_connect("/device") as ws:
        hello = {**HELLO, "caps": {"aec": True, "ref": True, "out_rates": [16000], "preroll_ms": 1500}}
        ws.send_text(json.dumps(hello))
        assert read(ws)["device"]["aec"] is False
        read_until(ws, is_listening)
        send_wake(ws, preroll_s=0.3)
        for frame in speech_frames(1.0) + silence_frames(0.8):
            ws.send_bytes(b"\x00" + frame)
        read_until(ws, until("listen_end"))  # the utterance ended: stop streaming
        read_until(ws, until("transcript"))  # ... and it still became the turn


def test_follow_up_can_be_disabled(device_client) -> None:
    client, _ = device_client(devices={"follow_up_ms": 0}, llm=FakeLLM([[TextDelta("Dobrze.")]]), stt=ScriptedSTT("Hej Gaja, cześć"))
    with client, client.websocket_connect("/device") as ws:
        connect(ws, caps={"aec": True, "out_rates": [16000]})
        send_wake(ws)
        utterance(ws, 1.0)
        messages = read_until(ws, until("listen_end"))
    assert "follow_up" not in msg_types(messages)


# ---- barge-in -----------------------------------------------------------------------------------


def long_reply_services(device_client, **kwargs: Any):
    llm = FakeLLM([[TextDelta("Dzień dobry, to jest bardzo długa odpowiedź. ")]], hang=True)
    return device_client(llm=llm, **kwargs), llm


def test_wake_during_playback_cancels_the_reply(device_client) -> None:
    (client, services), llm = long_reply_services(device_client, stt=ScriptedSTT("jaka jest pogoda"))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        send_wake(ws, preroll_s=0.3, score=0.95)
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "wake_accepted")
    states = [m for m in messages if isinstance(m, dict) and m["type"] == "state"]
    assert {"type": "state", "state": "listening", "reason": "barge_in"} in states
    assert llm.closed
    assert services.metrics.counters["wake_accepted"] == 1


def test_wake_barge_in_sends_playback_stop(device_client) -> None:
    """v1.11: a busy wake interrupts the reply and tells the device to drop its buffered audio.

    The device may hold up to ~44 s of reply in its playback ring; unlike a browser it cannot
    notice the cancelled downstream by itself.
    """
    (client, _), llm = long_reply_services(device_client, stt=ScriptedSTT("jaka jest pogoda"))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        send_wake(ws, preroll_s=0.3, score=0.95)
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "wake_accepted")
        dicts = [m for m in messages if isinstance(m, dict)]
        assert {"type": "playback_stop"} in dicts
        # The stop reaches the device before the wake verdict: it must flush BEFORE new pre-roll
        # audio for the fresh turn arrives.
        assert dicts.index({"type": "playback_stop"}) < dicts.index({"type": "wake_accepted"})
    assert llm.closed


def test_playback_stop_is_not_sent_without_playback(device_client) -> None:
    """A turn start with no reply in flight must not spam the device with playback_stop."""
    llm = FakeLLM([[TextDelta("Krótko.")]])
    client, _ = device_client(stt=ScriptedSTT("Hej Gaja, jaka jest pogoda?"), llm=llm)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=0.3)
        utterance(ws, 1.0)
        messages = read_until(ws, until("audio_end"))
    assert not any(isinstance(m, dict) and m.get("type") == "playback_stop" for m in messages)


def test_weak_wake_during_playback_is_ignored(device_client) -> None:
    (client, _), llm = long_reply_services(device_client)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        send_wake(ws, score=0.6)
        assert read(ws) == {"type": "wake_rejected", "reason": "low_score"}
        ws.send_text(json.dumps({"type": "ping"}))
        read_until(ws, until("pong"))
        assert not llm.closed


def test_wake_during_playback_without_aec_skips_the_polluted_preroll(device_client) -> None:
    stt = ScriptedSTT("jaka jest pogoda")
    (client, _), _ = long_reply_services(device_client, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        send_wake(ws, preroll_s=1.0, score=0.97)
        # 1 s of pre-roll (the device's own voice) then the user's real command
        for frame in speech_frames(1.0) + speech_frames(0.8) + silence_frames(0.8):
            ws.send_bytes(frame)
        read_until(ws, until("transcript"))
    # speech_frames(1.0) was dropped: the turn heard 0.8 s of speech (+ silence), not 1.8 s
    assert stt.calls and stt.calls[0] < 1.6


def _send_scene(ws: Any, mic: np.ndarray, ref: np.ndarray | None) -> None:
    """Stream a synthetic scene as the firmware does: 20 ms tagged frames, mic (0x00) then ref (0x01).

    `ref` is what the device feeds its speaker; `mic` is what its microphone picks up. `ref=None`
    withholds the reference stream (the control case: the server-side canceller never sees it).
    Samples are clipped to [-1, 1] exactly like the device's int16 wire format clips.
    """
    from tests.test_device_aec import BLOCK

    blocks = mic.size // BLOCK
    for i in range(blocks):
        block = np.clip(mic[i * BLOCK : (i + 1) * BLOCK], -1.0, 1.0)
        ws.send_bytes(b"\x00" + (block * 32767).astype("<i2").tobytes())
        if ref is not None:
            block = np.clip(ref[i * BLOCK : (i + 1) * BLOCK], -1.0, 1.0)
            ws.send_bytes(b"\x01" + (block * 32767).astype("<i2").tobytes())


def _long_reply_llm() -> FakeLLM:
    """One reply of 40 one-sentence deltas = 40 utterances x 0.1 s (FakeTTS) = a 4.0 s playback
    window on the session clock (+ the 0.4 s echo tail of Session._playback_until). hang=True
    keeps the turn alive after the deltas, so the reply is still "playing" for the whole scene
    and the barge-in has something to cancel (playback_stop)."""
    return FakeLLM([[TextDelta("Ala ma kota. ")] * 40], hang=True)


def _wake_and_start_long_reply(ws: Any) -> None:
    """caps.ref hello, a verified wake while idle, then the first audio_start of the long reply.

    The wake opens the mic window and nothing closes it for a caps.aec device: mic frames keep
    flowing (and matter) while the reply plays -- that is the full-duplex premise.
    """
    hello = {**HELLO, "caps": {"aec": True, "ref": True, "out_rates": [16000], "preroll_ms": 1500}}
    ws.send_text(json.dumps(hello))
    ready = read(ws)
    assert ready["device"]["aec"] is True and ready["device"]["verify_wake"] is True
    read_until(ws, is_listening)
    send_wake(ws, preroll_s=0.3)  # idle: nothing is playing yet, the wake starts turn 1
    for frame in speech_frames(1.0) + silence_frames(0.8):
        ws.send_bytes(b"\x00" + frame)  # a caps.ref device tags its mic frames
    read_until(ws, until("transcript"))  # the wake utterance becomes the first turn
    read_until(ws, until("audio_start"), limit=400)  # the reply is streaming: the window is open


def _echo_scene(speech_from: float = 2.0, total: float = 2.6):
    """The full-duplex scene: Gaja speaks (ref), her echo fills the mic, the user speaks over it.

    Deterministic: a speechish reference (broadband, like real TTS -- a pure tone would be
    learnable by the adaptive filter, which is not what a speaker plays), the echo = delayed +
    attenuated + convolved with a room IR, and the user speaking speechish over the echo from
    `speech_from` to `total` (far above the 0.05 test-VAD threshold, and broadband so the
    canceller's double-talk guard keeps it intact). The scene ends at `total`: that is where the
    barge-in fires, and a real device that received playback_stop stops its speaker -- the ref
    stream ends with it (the caller then sends plain mic silence).
    """
    from tests.test_device_aec import RATE, echo_of, room_ir, speechish

    ref = speechish(total, amp=0.25)
    echo = echo_of(ref, delay_s=0.05, gain=0.5, ir=room_ir())
    user = np.zeros_like(echo)
    user[int(speech_from * RATE) :] = speechish(total - speech_from, amp=0.4, seed=11)
    return ref, echo + user


def test_full_duplex_speech_barge_in_with_synthetic_echo(device_client) -> None:
    """THE v1.11 test: Gaja's own echo never barges in; the user speaking over it does.

    The scene runs exactly the frames the hardware sends, WHILE the reply is playing (the
    session's playback window is open): reference + echo as alternating tagged frames.
    1. 2 s of pure echo: the canceller removes it; no VAD speech_start, no playback_stop.
    2. The user speaks on top of the echo: strict VAD confirms, the reply is cancelled
       (playback_stop, state listening/barge_in). The device stops playing (the ref stream ends
       with it) and the mic hears room silence, which closes the interrupted utterance -- it
       becomes a real turn (transcript).
    The control case below proves this test can fail (same echo, canceller starved of ref).
    """
    stt = ScriptedSTT("Hej Gaja, opowiedz coś", "zamknij się")
    llm = _long_reply_llm()
    client, _ = device_client(stt=stt, llm=llm, devices={"follow_up_ms": 60000})
    with client, client.websocket_connect("/device") as ws:
        _wake_and_start_long_reply(ws)
        ref, mic = _echo_scene()
        # Phase 1: pure echo during playback must never barge in.
        cut = int(2.0 * 16000)
        _send_scene(ws, mic[:cut], ref[:cut])
        ws.send_text(json.dumps({"type": "ping"}))
        echo_window = read_until(ws, until("pong"), limit=400)
        starts = [
            m
            for m in echo_window
            if isinstance(m, dict) and m.get("type") == "vad" and m.get("state") == "speech_start"
        ]
        assert not starts, starts
        assert not any(isinstance(m, dict) and m.get("type") == "playback_stop" for m in echo_window)

        # Phase 2: the user speaks over the echo (Gaja is still talking -- that is the point),
        # the barge-in fires and the device stops playing: ref frames end, the mic hears silence.
        _send_scene(ws, mic[cut:], ref[cut:])
        _send_scene(ws, np.zeros(int(0.8 * 16000), dtype=np.float32), None)
        messages = read_until(ws, until("transcript"), limit=400)
        dicts = [m for m in messages if isinstance(m, dict)]
        assert {"type": "vad", "state": "speech_start"} in dicts
        assert {"type": "playback_stop"} in dicts
        assert {"type": "state", "state": "listening", "reason": "barge_in"} in dicts
        transcript = {"type": "transcript", "text": "zamknij się", "lang": "pl", "final": True}
        # playback_stop reaches the device before the interrupted utterance becomes a new turn:
        assert dicts.index({"type": "playback_stop"}) < dicts.index(transcript)
        assert dicts[-1] == transcript
    assert llm.closed  # the interrupted reply's LLM stream was cancelled


def test_full_duplex_echo_without_the_canceller_barges_in(device_client) -> None:
    """CONTROL for the test above: the SAME echo with the reference frames withheld barges in.

    caps.ref is still advertised, so the frame format is identical -- but the canceller never
    receives a reference (mic frames pass through untouched), which is what a broken firmware
    ref stream would look like. The raw echo hits the (strict) VAD and barges in during
    playback: speech_start + playback_stop, with no user speech anywhere in the scene.
    """
    from tests.test_device_aec import echo_of, room_ir, speechish

    stt = ScriptedSTT("Hej Gaja, opowiedz coś")
    llm = _long_reply_llm()
    client, _ = device_client(stt=stt, llm=llm, devices={"follow_up_ms": 60000})
    with client, client.websocket_connect("/device") as ws:
        _wake_and_start_long_reply(ws)
        ref = speechish(1.5, amp=0.25)
        echo = echo_of(ref, delay_s=0.05, gain=0.5, ir=room_ir())
        _send_scene(ws, echo, None)  # mic only: the reference stream is withheld
        # _interrupt_speech sends playback_stop and THEN the state message: read up to the state.
        messages = read_until(
            ws,
            lambda m: isinstance(m, dict) and m.get("type") == "state" and m.get("reason") == "barge_in",
            limit=400,
        )
        dicts = [m for m in messages if isinstance(m, dict)]
        assert {"type": "vad", "state": "speech_start"} in dicts
        assert {"type": "state", "state": "listening", "reason": "barge_in"} in dicts
    assert llm.closed  # the echo barged in, so the reply was cancelled


def test_full_duplex_malformed_tagged_frames_are_ignored(device_client) -> None:
    """A bad tagged frame never kills a caps.ref session (docs/DEVICE.md): unknown tags and torn
    reference frames are dropped, and the mic stream keeps working."""
    llm = FakeLLM([[TextDelta("Krótko.")]])
    client, _ = device_client(stt=ScriptedSTT("Hej Gaja, jaka jest pogoda?"), llm=llm)
    with client, client.websocket_connect("/device") as ws:
        hello = {**HELLO, "caps": {"aec": True, "ref": True, "out_rates": [16000], "preroll_ms": 1500}}
        ws.send_text(json.dumps(hello))
        read(ws)
        read_until(ws, is_listening)
        send_wake(ws, preroll_s=0.3)
        ws.send_bytes(b"\x07" + b"\x00" * 64)  # unknown tag
        ws.send_bytes(b"\x01" + b"\x00" * 63)  # reference frame with an odd byte count
        for frame in speech_frames(1.0) + silence_frames(0.8):
            ws.send_bytes(b"\x00" + frame)
        read_until(ws, until("transcript"))  # the wake utterance still became a turn
        read_until(ws, until("audio_end"))
        ws.send_text(json.dumps({"type": "ping"}))
        assert read_until(ws, until("pong"))[-1] == {"type": "pong"}  # session alive


def _probe_frames(seconds: float) -> list[bytes]:
    """Mic frames that keep an UNCONFIRMED VAD segment open without ever confirming it.

    One loud 32 ms window (peak 0.4 > 0.05) followed by 14 quiet ones (peak 0.02): the silence
    never reaches the 16-window reset, so the endpointer stays in_speech (the wake verifier's
    speech check passes), but the loud count never reaches the 13 strict windows, so the VAD
    barge-in never fires. Anything the test sees cancel the reply now came from the wake path.
    """
    t = np.arange(512) / 16000
    loud = (0.4 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()
    quiet = (0.02 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()
    frames: list[bytes] = []
    while len(frames) * 512 < seconds * 16000:
        frames.extend([loud] + [quiet] * 14)
    return frames


def test_full_duplex_wake_during_playback_cancels_only_when_verified(device_client) -> None:
    """A caps.aec busy wake is VERIFIED before anything is cancelled (deferred barge-in).

    The verifier needs 2.5 s of mic audio (verify_ms) and transcribes it: a false wake is
    rejected and the reply KEEPS playing (a false alarm costs nothing -- the point of the
    echo-cleaned pre-roll), a real one is accepted and only then is the reply cancelled with
    playback_stop. The probe signal cannot confirm the VAD (see _probe_frames), so a
    playback_stop here can only come from the accepted wake.
    """
    stt = ScriptedSTT("Hej Gaja, opowiedz coś", "nic ciekawego", "hej gaja poczekaj")
    llm = _long_reply_llm()
    client, _ = device_client(stt=stt, llm=llm, devices={"follow_up_ms": 60000})
    with client, client.websocket_connect("/device") as ws:
        _wake_and_start_long_reply(ws)
        frames = _probe_frames(2.9)

        # A false wake during playback: verified, rejected -- the reply keeps playing.
        send_wake(ws, preroll_s=0.3)
        for frame in frames[:45]:
            ws.send_bytes(b"\x00" + frame)
        ws.send_text(json.dumps({"type": "ping"}))
        # Mid-verify (1.5 s of 2.5 s buffered): nothing may cancel the reply yet.
        while_verifying = read_until(ws, until("pong"), limit=400)
        assert not any(isinstance(m, dict) and m.get("type") == "playback_stop" for m in while_verifying)
        for frame in frames[45:]:
            ws.send_bytes(b"\x00" + frame)
        rejected = read_until(ws, until("wake_rejected"), limit=400)
        assert rejected[-1] == {"type": "wake_rejected", "reason": "no_phrase"}
        ws.send_text(json.dumps({"type": "ping"}))
        after_reject = read_until(ws, until("pong"), limit=400)
        assert not any(isinstance(m, dict) and m.get("type") == "playback_stop" for m in after_reject)
        assert not llm.closed

        # A real wake during playback: verified, accepted -- NOW the deferred barge-in fires.
        send_wake(ws, preroll_s=0.3)
        for frame in frames:
            ws.send_bytes(b"\x00" + frame)
        messages = read_until(
            ws,
            lambda m: isinstance(m, dict) and m.get("type") == "state" and m.get("reason") == "barge_in",
            limit=400,
        )
        dicts = [m for m in messages if isinstance(m, dict)]
        assert {"type": "wake_accepted"} in dicts
        assert {"type": "playback_stop"} in dicts
        assert dicts.index({"type": "wake_accepted"}) < dicts.index({"type": "playback_stop"})
        # The VAD never confirmed anything in this test: no speech_start anywhere.
        for chunk in (while_verifying, after_reject, messages):
            assert not any(
                isinstance(m, dict) and m.get("type") == "vad" and m.get("state") == "speech_start"
                for m in chunk
            )
    assert llm.closed  # the accepted wake cancelled the reply


def test_wake_cancel_leaves_hermes_tasks_running(make_services, registry) -> None:
    from gaja.llm.parse import ToolCall
    from tests.conftest import FakeHermes

    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "dane"})
    hermes = FakeHermes(hang=True)
    services = make_services(cfg={"devices": FAST}, llm=FakeLLM([[call]]), hermes=hermes)
    client = TestClient(create_device_app(services, registry), client=("127.0.0.1", 1))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        read_until(ws, until("metrics"))
        send_wake(ws)
        ws.send_text(json.dumps({"type": "ping"}))
        read_until(ws, until("pong"))
        assert not hermes.cancelled


def test_hermes_result_on_device_arms_follow_up_window(make_services, registry) -> None:
    from gaja.llm.parse import ToolCall
    from tests.conftest import FakeHermes

    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "dane"})
    hermes = FakeHermes(result="Pogoda: dwanaście stopni.")
    services = make_services(cfg={"devices": FAST}, llm=FakeLLM([[call]]), hermes=hermes)
    client = TestClient(create_device_app(services, registry), client=("127.0.0.1", 1))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        # The handoff turn answers with an ack; the Hermes result arrives later as its own turn.
        read_until(ws, until("audio_end"))  # the ack utterance
        # After the result has been spoken the mic must open again, without the wake phrase.
        messages = read_until(ws, until("audio_end"))  # the result utterance
        messages += read_until(ws, until("follow_up"))
    assert messages[-1] == {"type": "follow_up", "ms": 50}


def test_tool_answer_on_device_arms_follow_up_window(device_client) -> None:
    from tests.test_conversation import FakeTools

    tools = FakeTools(speak="W Katowicach jest dwanaście stopni.")
    llm = FakeLLM(p_tool=0.9, tool_call=' get_weather(when="now")')
    client, _ = device_client(llm=llm, router=True, tools=tools)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        messages = read_until(ws, until("follow_up"))
    assert "tool" in msg_types(messages)
    assert messages[-1] == {"type": "follow_up", "ms": 50}


def test_wake_message_validation(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "wake", "score": "high"}))
        assert read(ws)["message"] == "wake: score and preroll_ms must be numbers"


# ---- reply recovery after a dropped socket -------------------------------------------------------


def test_lost_reply_is_recorded_once_per_turn_so_a_late_close_cannot_re_enable_it(make_services) -> None:
    """A reconnect records the interrupted reply in supersede() *and* again when the old session closes.

    If the old session's close lands after the reconnected device already spoke the recovered reply, a
    second record would arm the recovery window again and a third connection would replay the reply.
    """
    from gaja.devices.registry import DeviceInfo
    from gaja.metrics import TurnMetrics
    from gaja.pipeline import Session, _Turn

    class NullOut:
        async def send_json(self, payload: dict[str, Any]) -> None:
            return None

        async def send_bytes(self, data: bytes) -> None:
            return None

    services = make_services(cfg={"devices": {**FAST}})
    session = Session(services, NullOut(), device=DeviceInfo("atom-1"))
    session._current_turn = _Turn(
        t0=0.0,
        language="pl",
        metrics=TurnMetrics(turn=1, kind="audio"),
        out=NullOut(),
        user_text="jaka pogoda",
        reply_text="Jest słonecznie.",
    )

    session.record_lost_reply()
    assert "atom-1" in services.lost_replies

    # The reconnected device consumed the reply and spoke it.
    assert session._device is not None
    assert session._device.take_lost_reply() is not None

    # The old connection finishing its close must not put the reply back.
    session.record_lost_reply()
    assert "atom-1" not in services.lost_replies


@pytest.mark.asyncio
async def test_a_reconnected_device_keeps_its_conversation(make_services) -> None:
    """A Wi-Fi drop opens a new session for the same device: the local LLM must still see the earlier turns."""
    from gaja.devices.registry import DeviceInfo
    from gaja.pipeline import Session

    class NullOut:
        async def send_json(self, payload: dict[str, Any]) -> None:
            return None

        async def send_bytes(self, data: bytes) -> None:
            return None

    services = make_services(cfg={"devices": {**FAST}})
    old = Session(services, NullOut(), device=DeviceInfo("atom-1"))
    await old.handle_text(json.dumps(HELLO))
    old._history.add("user", "Mój kot ma na imię Filemon.")
    old._history.add("assistant", "Ładne imię.")
    await old.close()

    new = Session(services, NullOut(), device=DeviceInfo("atom-1"))
    other = Session(services, NullOut(), device=DeviceInfo("atom-2"))
    try:
        await new.handle_text(json.dumps(HELLO))
        assert [t["content"] for t in new._history.context()] == ["Mój kot ma na imię Filemon.", "Ładne imię."]
        assert other._history.context() == []
    finally:
        await new.close()
        await other.close()


@pytest.mark.asyncio
async def test_a_recovered_reply_replaces_the_one_the_drop_cut_short(make_services) -> None:
    """The old session stored the reply as said ("…"); the recovered full reply takes its place, once."""
    from gaja.devices.controller import LostReply
    from gaja.devices.registry import DeviceInfo
    from gaja.pipeline import INTERRUPTED_MARK, Session

    class NullOut:
        async def send_json(self, payload: dict[str, Any]) -> None:
            return None

        async def send_bytes(self, data: bytes) -> None:
            return None

    services = make_services(cfg={"devices": {**FAST}})
    old = Session(services, NullOut(), device=DeviceInfo("atom-1"))
    old._history.add("user", "jaka pogoda")
    old._history.add("assistant", "Jest sło" + INTERRUPTED_MARK)
    await old.close()

    new = Session(services, NullOut(), device=DeviceInfo("atom-1"))
    try:
        await new._recover_turn(LostReply("Jest słonecznie.", user_text="jaka pogoda"))
        assert new._history.context() == [
            {"role": "user", "content": "jaka pogoda"},
            {"role": "assistant", "content": "Jest słonecznie."},
        ]
    finally:
        await new.close()


@pytest.mark.asyncio
async def test_speculation_commit_on_a_dropped_socket_never_leaves_the_turn_waiting(make_services) -> None:
    """The Wi-Fi can die while the speculative gate flushes at commit: the turn must unwind, not hang."""
    from gaja.devices.registry import DeviceInfo
    from gaja.pipeline import Session
    from tests.conftest import FakeHermes

    class FlakyOut:
        def __init__(self) -> None:
            self.dead = False
            self.sent: list[Any] = []

        async def send_json(self, payload: dict[str, Any]) -> None:
            if self.dead:
                raise ConnectionError("socket gone")
            await asyncio.sleep(0)
            self.sent.append(payload)

        async def send_bytes(self, data: bytes) -> None:
            if self.dead:
                raise ConnectionError("socket gone")
            await asyncio.sleep(0)
            self.sent.append(data)

    services = make_services(cfg={"devices": {**FAST}}, llm=FakeLLM(p_hermes=0.99), hermes=FakeHermes(hang=True), router=True)
    out = FlakyOut()
    session = Session(services, out, device=DeviceInfo("atom-1"))
    await session.handle_text(json.dumps(HELLO))  # the token is only checked by the device endpoint
    out.sent.clear()

    await session._start_speculation(np.zeros(1600, dtype=np.float32))
    await asyncio.sleep(0.05)  # the speculative turn queues its ack in the gate, then waits on spec.committed
    spec = session._spec
    assert spec is not None and spec.task is not None

    out.dead = True  # the device drops off before the commit can flush the gate
    await session._commit_speculation()
    assert spec.committed.is_set(), "a speculative turn was left waiting on a commit that never happened"
    assert spec.task.done(), "the speculative turn was left running for a device that is gone"
    # The turn never reached Hermes for a dead device, and its only spoken text was the ack (not a reply).
    assert session._jobs == {}
    assert "atom-1" not in services.lost_replies
    await session.close()


@pytest.mark.asyncio
async def test_a_session_that_lost_the_device_never_arms_recovery(make_services) -> None:
    """A reconnect makes a newer session the owner: the old one must not arm a reply that would
    then be spoken at the next reconnect, i.e. after the newer session's turns."""
    from gaja.devices.registry import DeviceInfo
    from gaja.metrics import TurnMetrics
    from gaja.pipeline import Session, _Turn

    class NullOut:
        async def send_json(self, payload: dict[str, Any]) -> None:
            return None

        async def send_bytes(self, data: bytes) -> None:
            return None

    def turn(reply: str) -> _Turn:
        return _Turn(t0=0.0, language="pl", metrics=TurnMetrics(turn=1, kind="audio"), out=NullOut(), reply_text=reply)

    services = make_services(cfg={"devices": {**FAST}})
    old = Session(services, NullOut(), device=DeviceInfo("atom-1"))
    old._current_turn = turn("Stara odpowiedź.")
    new = Session(services, NullOut(), device=DeviceInfo("atom-1"))  # the device reconnected
    try:
        old.record_lost_reply()
        assert "atom-1" not in services.lost_replies

        # The live session still holds its own interrupted reply.
        new._current_turn = turn("Nowa odpowiedź.")
        new.record_lost_reply()
        lost = services.lost_replies.get("atom-1")
        assert lost is not None and lost.text == "Nowa odpowiedź."
    finally:
        await old.close()
        await new.close()


@pytest.mark.asyncio
async def test_a_hermes_handoff_ack_is_not_held_as_a_lost_reply(make_services) -> None:
    """Only a user-turn reply is recovered: the "I am checking" ack said before a Hermes handoff is not."""
    from gaja.devices.registry import DeviceInfo
    from gaja.pipeline import Session
    from tests.conftest import FakeHermes

    class AckThenDeadOut:
        """The utterance goes out, then the socket dies while its audio is being sent."""

        def __init__(self) -> None:
            self.sent: list[Any] = []

        async def send_json(self, payload: dict[str, Any]) -> None:
            if payload.get("type") == "audio_start":
                raise ConnectionError("socket gone")
            self.sent.append(payload)

        async def send_bytes(self, data: bytes) -> None:
            raise ConnectionError("socket gone")

    services = make_services(cfg={"devices": {**FAST}}, llm=FakeLLM(p_hermes=0.99), hermes=FakeHermes(hang=True), router=True)
    out = AckThenDeadOut()
    session = Session(services, out, device=DeviceInfo("atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session._start_user_turn(audio=np.zeros(1600, dtype=np.float32))
    assert session._active is not None
    await asyncio.wait_for(session._active, 5)

    assert any(isinstance(m, dict) and m.get("type") == "utterance" for m in out.sent), "no ack was spoken"
    assert "atom-1" not in services.lost_replies
    await session.close()


@pytest.mark.asyncio
async def test_speculation_finished_before_a_failed_commit_still_holds_its_reply(make_services) -> None:
    """A short reply can be fully generated (and gated) before the commit: a drop at flush still recovers it."""
    from gaja.devices.registry import DeviceInfo
    from gaja.pipeline import Session

    class FlakyOut:
        def __init__(self) -> None:
            self.dead = False

        async def send_json(self, payload: dict[str, Any]) -> None:
            if self.dead:
                raise ConnectionError("socket gone")

        async def send_bytes(self, data: bytes) -> None:
            if self.dead:
                raise ConnectionError("socket gone")

    services = make_services(cfg={"devices": {**FAST}}, llm=FakeLLM([[TextDelta("Jest słonecznie i ciepło.")]]))
    out = FlakyOut()
    session = Session(services, out, device=DeviceInfo("atom-1"))
    await session.handle_text(json.dumps(HELLO))

    await session._start_speculation(np.zeros(1600, dtype=np.float32))
    spec = session._spec
    assert spec is not None and spec.task is not None
    await asyncio.wait_for(spec.task, 5)  # the whole reply is synthesized into the gate before the commit

    out.dead = True
    await session._commit_speculation()
    lost = services.lost_replies.get("atom-1")
    assert lost is not None and lost.text == "Jest słonecznie i ciepło."
    await session.close()


def test_reply_interrupted_by_disconnect_is_recovered_once_on_reconnect(device_client) -> None:
    from gaja.devices.controller import REPLY_RECOVERY_NOTICE

    llm = FakeLLM([[TextDelta("Pierwsze zdanie odpowiedzi. Drugie zdanie odpowiedzi. ")]], hang=True)
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda")
    client, services = device_client(llm=llm, stt=stt)
    with client:
        with client.websocket_connect("/device") as ws:
            connect(ws)
            send_wake(ws)
            utterance(ws, 1.0)
            # Receive the first sentence's audio_end, then drop the socket mid-reply.
            read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")

        assert "atom-1" in services.lost_replies
        assert len(llm.calls) == 1

        with client.websocket_connect("/device") as ws:
            connect(ws)
            recovered = read_until(ws, until("follow_up"))

        utterances = [m["text"] for m in recovered if isinstance(m, dict) and m.get("type") == "utterance"]
        assert utterances[0] == REPLY_RECOVERY_NOTICE
        assert " ".join(utterances[1:]) == "Pierwsze zdanie odpowiedzi. Drugie zdanie odpowiedzi."
        # The LLM was not re-queried: the recovered text came from the interrupted turn.
        assert len(llm.calls) == 1
        assert "atom-1" not in services.lost_replies

        # Only once: a second reconnect does not repeat the reply.
        with client.websocket_connect("/device") as ws:
            connect(ws)
            ws.send_text(json.dumps({"type": "ping"}))
            after = read_until(ws, until("pong"))
        assert "utterance" not in msg_types(after)


def test_reply_delivered_before_disconnect_is_not_repeated_on_reconnect(device_client) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie.")]])
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda")
    client, services = device_client(llm=llm, stt=stt)
    with client:
        with client.websocket_connect("/device") as ws:
            connect(ws)
            send_wake(ws)
            utterance(ws, 1.0)
            read_until(ws, until("follow_up"))

        assert "atom-1" not in services.lost_replies

        with client.websocket_connect("/device") as ws:
            connect(ws)
            ws.send_text(json.dumps({"type": "ping"}))
            messages = read_until(ws, until("pong"))
    assert "utterance" not in msg_types(messages)


@pytest.mark.parametrize("window_s", [0.0, 0.02])
def test_reconnect_after_recovery_window_does_not_replay_reply(device_client, window_s) -> None:
    import time

    llm = FakeLLM([[TextDelta("Długa odpowiedź do powtórzenia. ")]], hang=True)
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda")
    client, services = device_client(devices={"reply_recovery_s": window_s}, llm=llm, stt=stt)
    with client:
        with client.websocket_connect("/device") as ws:
            connect(ws)
            send_wake(ws)
            utterance(ws, 1.0)
            read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")

        if window_s > 0:
            time.sleep(0.05)

        with client.websocket_connect("/device") as ws:
            connect(ws)
            time.sleep(0.08)
            ws.send_text(json.dumps({"type": "ping"}))
            messages = read_until(ws, until("pong"))
    assert "utterance" not in msg_types(messages)
    assert "atom-1" not in services.lost_replies


def test_user_speaking_first_after_reconnect_cancels_reply_recovery(device_client) -> None:
    from gaja.devices.controller import REPLY_RECOVERY_NOTICE

    calls = 0

    class HangingFirstLLM(FakeLLM):
        async def stream(self, messages, *, allow_tools: bool):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield TextDelta("Stara niedokończona odpowiedź. ")
                await asyncio.sleep(3600)
            else:
                yield TextDelta("Nowa odpowiedź.")

    stt = ScriptedSTT("Hej Gaja, pierwsze pytanie", "Hej Gaja, drugie pytanie")
    client, services = device_client(llm=HangingFirstLLM(), stt=stt)
    with client:
        with client.websocket_connect("/device") as ws:
            connect(ws)
            send_wake(ws)
            utterance(ws, 1.0)
            read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")

        assert "atom-1" in services.lost_replies

        with client.websocket_connect("/device") as ws:
            connect(ws)
            # The user immediately wakes the device for a new turn before recovery fires.
            send_wake(ws)
            utterance(ws, 1.0)
            messages = read_until(ws, until("follow_up"))

    utterances = [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert REPLY_RECOVERY_NOTICE not in utterances
    assert "Stara niedokończona odpowiedź." not in utterances
    assert utterances == ["Nowa odpowiedź."]
    assert "atom-1" not in services.lost_replies


# ---- embedded listener over a real socket --------------------------------------------------------


def test_embedded_listener_serves_the_device_socket_and_stops(make_services, registry) -> None:
    import websockets

    services = make_services(cfg={"devices": {**FAST, "listen_host": "127.0.0.1", "port": 0}})

    async def scenario() -> None:
        listener = DeviceListener(services, registry, port=0)
        assert await listener.start()
        try:
            url = f"ws://127.0.0.1:{listener.port}/device"
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps(HELLO))
                ready = json.loads(await ws.recv())
                assert ready["type"] == "ready" and ready["sample_rate"] == 16000
                assert listener.connected == ["atom-1"]
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({**HELLO, "token": "x" * 40}))
                assert json.loads(await ws.recv()) == {"type": "error", "message": "unauthorized"}
            with pytest.raises(Exception):
                async with websockets.connect(f"ws://127.0.0.1:{listener.port}/ws"):
                    pass
        finally:
            await listener.stop()
        with pytest.raises(OSError):
            await websockets.connect(url)

    asyncio.run(scenario())


def test_main_app_starts_the_listener_only_when_enabled(make_services) -> None:
    off = make_services()
    with TestClient(create_app(services=off)) as client:
        assert client.get("/health").json()["devices"] == {"enabled": False, "listening": False, "connected": []}
    on = make_services(cfg={"devices": {"enabled": True, "listen_host": "127.0.0.1", "port": 0, "tokens_file": "/nonexistent/devices.toml"}})
    with TestClient(create_app(services=on)) as client:
        health = client.get("/health").json()["devices"]
        assert health["enabled"] and health["listening"]


def test_verify_passes_configured_hotwords_to_stt(device_client) -> None:
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda")
    client, _ = device_client(devices={"verify_hotwords": "Hej Gaja, Gajo"}, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=1.5)
        for frame in speech_frames(3.0):
            ws.send_bytes(frame)
        read_until(ws, until("wake_accepted"))
    assert "Hej Gaja, Gajo" in stt.hotwords_calls


def test_short_wake_utterance_is_decoded_with_the_verify_hotwords(device_client) -> None:
    # Ends before verify_ms, so the turn's own transcript decides the wake: it must use the verify hotwords.
    stt = ScriptedSTT("Hej Gaja, jaka jest pogoda")
    client, _ = device_client(devices={"verify_hotwords": "Hej Gaja, Gajo"}, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=0.5)
        for frame in speech_frames(1.0) + silence_frames(0.8):
            ws.send_bytes(frame)
        read_until(ws, until("wake_accepted"))
    assert len(stt.calls) == 1
    assert stt.hotwords_calls == ["Hej Gaja, Gajo"]

def test_wake_engine_decodes_the_verify_window_and_the_main_engine_the_turn(device_client) -> None:
    stt, wake_stt = ScriptedSTT("Jaka jest pogoda w Krakowie"), ScriptedSTT("Hej Gaja, jaka jest")
    client, services = device_client(devices={"verify_hotwords": "Hej Gaja, Gajo"}, stt=stt)
    services.wake_stt = wake_stt
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=1.5)
        utterance(ws, 3.0)
        messages = read_until(ws, until("follow_up"))
    transcript = next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "Jaka jest pogoda w Krakowie"
    assert wake_stt.hotwords_calls == ["Hej Gaja, Gajo"]
    assert len(stt.calls) == 1 and stt.hotwords_calls != ["Hej Gaja, Gajo"]


def test_short_wake_utterance_is_decided_by_the_wake_engine(device_client) -> None:
    stt, wake_stt = ScriptedSTT("Hey guy, jaka jest pogoda"), ScriptedSTT("Hej Gaja, jaka jest pogoda")
    client, services = device_client(devices={"verify_hotwords": "Hej Gaja, Gajo"}, stt=stt)
    services.wake_stt = wake_stt
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=0.5)
        for frame in speech_frames(1.0) + silence_frames(0.8):
            ws.send_bytes(frame)
        read_until(ws, until("wake_accepted"))
    assert stt.calls == []
    assert wake_stt.hotwords_calls == ["Hej Gaja, Gajo"]


def test_wake_verify_cancelled_when_speech_ends_before_verify_finishes(device_client) -> None:
    """Finding #2 Path A: Utterance ends while verify_task is in flight -> cancel verify, single decode resolves."""
    class SlowVerifySTT(FakeSTT):
        def __init__(self) -> None:
            super().__init__("Hej Gaja, włącz światło")
            self.calls: list[float] = []
            self.verify_cancelled = False

        async def transcribe(self, pcm: np.ndarray, language: str | None = None, **kwargs) -> Transcript:
            self.calls.append(len(pcm) / 16000)
            if len(pcm) / 16000 <= 2.6:
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    self.verify_cancelled = True
                    raise
            return Transcript(text="Hej Gaja, włącz światło", language="pl", duration_ms=5.0, audio_s=len(pcm) / 16000)

    stt = SlowVerifySTT()
    llm = FakeLLM([[TextDelta("Włączono światło.")]])
    client, services = device_client(llm=llm, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=0.5)
        # Send 2.7s of utterance: crosses 2.5s verify threshold, spawning verify_task, but ends at 2.7s
        utterance(ws, 2.7)
        messages = read_until(ws, until("follow_up"))

    assert stt.verify_cancelled is True
    kinds = msg_types(messages)
    assert "wake_accepted" in kinds
    transcript = next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "włącz światło"
    assert llm.calls[0][0][-1] == {"role": "user", "content": "włącz światło"}


def test_wake_verify_finishes_before_speech_ends(device_client) -> None:
    """Finding #2 Path B: Verification completes mid-speech before speech ends."""
    stt = ScriptedSTT("Hej Gaja", "Hej Gaja, wyłącz światło")
    llm = FakeLLM([[TextDelta("Wyłączono światło.")]])
    client, services = device_client(llm=llm, stt=stt)
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws, preroll_s=0.5)
        # 3.0s utterance: verify_task completes fast at 2.5s, speech ends at 3.0s
        utterance(ws, 3.0)
        messages = read_until(ws, until("follow_up"))

    kinds = msg_types(messages)
    assert "wake_accepted" in kinds
    assert len(stt.calls) == 2  # call 1 for verify (2.5s), call 2 for full utterance (3.0s)
    transcript = next(m for m in messages if isinstance(m, dict) and m["type"] == "transcript")
    assert transcript["text"] == "wyłącz światło"


@pytest.mark.asyncio
async def test_resampling_outbox_chunk_size_dispatch() -> None:
    """Finding #8: ResamplingOutbox offloads >100ms resampling to thread while keeping tiny chunks on loop."""
    from unittest.mock import AsyncMock
    from gaja.devices.audio import ResamplingOutbox
    import numpy as np

    mock_inner = AsyncMock()
    outbox = ResamplingOutbox(mock_inner, chunk_ms=100)
    outbox.configure(16000)
    await outbox.send_json({"type": "audio_start", "sample_rate": 24000})

    # Tiny chunk (50 ms at 24 kHz = 1200 samples = 2400 bytes)
    tiny_pcm = np.zeros(1200, dtype="<i2").tobytes()
    await outbox.send_bytes(tiny_pcm)

    # Large chunk (500 ms at 24 kHz = 12000 samples = 24000 bytes)
    large_pcm = np.zeros(12000, dtype="<i2").tobytes()
    await outbox.send_bytes(large_pcm)

    assert mock_inner.send_bytes.call_count >= 2


def test_pre_parsed_allowed_networks() -> None:
    """Finding #19: Pre-parse allowed networks once and verify peer_allowed efficiency."""
    import ipaddress
    from gaja.devices.listener import parse_networks, peer_allowed

    raw_nets = ["127.0.0.0/8", "192.168.1.0/24", "::1/128"]
    parsed = parse_networks(raw_nets)
    assert len(parsed) == 3
    assert all(isinstance(n, (ipaddress.IPv4Network, ipaddress.IPv6Network)) for n in parsed)

    # Allowed cases
    assert peer_allowed("127.0.0.1", parsed)
    assert peer_allowed("192.168.1.42", parsed)
    assert peer_allowed("::1", parsed)
    # IPv4-mapped IPv6
    assert peer_allowed("::ffff:192.168.1.50", parsed)

    # Refused cases
    assert not peer_allowed("10.0.0.1", parsed)
    assert not peer_allowed("8.8.8.8", parsed)
    assert not peer_allowed(None, parsed)
    assert not peer_allowed("invalid-ip", parsed)


@pytest.mark.asyncio
async def test_resampling_outbox_paces_long_replies_instead_of_bursting_them() -> None:
    """A whole reply dumped into the socket at once overflows the Wi-Fi link (lost segments, TCP backoff
    for seconds, the device drained mid-reply). Past a short burst the outbox sends at ~2x real time and
    never runs more than ``lead_s`` ahead of playback; the first audio still leaves at once."""
    from unittest.mock import AsyncMock
    from gaja.devices.audio import ResamplingOutbox

    inner = AsyncMock()
    sent_at: list[float] = []
    inner.send_bytes.side_effect = lambda data: sent_at.append(time.perf_counter())
    outbox = ResamplingOutbox(inner, chunk_ms=100, lead_s=1.0, burst_s=0.3, speedup=2.0)
    outbox.configure(16000)
    await outbox.send_json({"type": "audio_start", "sample_rate": 16000})
    t0 = time.perf_counter()
    await outbox.send_bytes(np.zeros(16000 * 3, dtype="<i2").tobytes())  # 3 s of audio
    elapsed = time.perf_counter() - t0
    assert len(sent_at) == 30
    assert sent_at[2] - t0 < 0.05  # the burst goes out immediately
    # 3 s of audio may run at most 1 s ahead of playback: >= ~2 s to send, but well below real time + lead.
    assert 1.8 < elapsed < 2.6, elapsed

    # playback_stop (barge-in) drops what the device had queued: the next reply bursts again.
    await outbox.send_json({"type": "playback_stop"})
    t1 = time.perf_counter()
    await outbox.send_bytes(np.zeros(16000 // 5, dtype="<i2").tobytes())
    assert time.perf_counter() - t1 < 0.05
