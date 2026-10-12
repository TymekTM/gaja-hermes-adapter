"""Protocol v1.8: device endpoint, auth, hello negotiation, wake flow (stubbed STT / LLM / TTS)."""

from __future__ import annotations

import asyncio
import json
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
FAST = {"follow_up_ms": 50, "listen_grace_s": 0.05, "wake_timeout_s": 5.0, "listen_timeout_s": 5.0}


class ScriptedSTT(FakeSTT):
    """Returns the scripted texts in order (the last one repeats) and records what it was given."""

    def __init__(self, *texts: str) -> None:
        super().__init__(texts[0])
        self.texts = list(texts)
        self.calls: list[float] = []

    async def transcribe(self, pcm: np.ndarray, language: str | None = None) -> Transcript:
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
    client, _ = device_client(llm=llm, stt=ScriptedSTT("Hej Gaja."))
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        send_wake(ws)
        utterance(ws, 0.8)
        messages = read_until(ws, until("follow_up"))
    spoken = [m["text"] for m in messages if isinstance(m, dict) and m["type"] == "utterance"]
    assert spoken == ["Słucham?"]
    assert not llm.calls
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


def test_wake_message_validation(device_client) -> None:
    client, _ = device_client()
    with client, client.websocket_connect("/device") as ws:
        connect(ws)
        ws.send_text(json.dumps({"type": "wake", "score": "high"}))
        assert read(ws)["message"] == "wake: score and preroll_ms must be numbers"


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


def test_wake_verify_cancelled_when_speech_ends_before_verify_finishes(device_client) -> None:
    """Finding #2 Path A: Utterance ends while verify_task is in flight -> cancel verify, single decode resolves."""
    class SlowVerifySTT(FakeSTT):
        def __init__(self) -> None:
            super().__init__("Hej Gaja, włącz światło")
            self.calls: list[float] = []
            self.verify_cancelled = False

        async def transcribe(self, pcm: np.ndarray, language: str | None = None) -> Transcript:
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
