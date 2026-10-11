from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from gaja.llm.client import TextDelta
from gaja.llm.parse import ToolCall
from gaja.pipeline import ACKS
from gaja.server import create_app
from tests.conftest import FakeHermes, FakeLLM, FakeSTT


def read(ws: Any) -> Any:
    message = ws.receive()
    if message.get("text") is not None:
        return json.loads(message["text"])
    return message["bytes"]


def read_until(ws: Any, done: Callable[[Any], bool], limit: int = 200) -> list[Any]:
    seen: list[Any] = []
    for _ in range(limit):
        message = read(ws)
        seen.append(message)
        if done(message):
            return seen
    raise AssertionError(f"no matching message; saw {seen}")


def is_type(kind: str) -> Callable[[Any], bool]:
    return lambda m: isinstance(m, dict) and m.get("type") == kind


def is_listening(message: Any) -> bool:
    return isinstance(message, dict) and message.get("type") == "state" and message.get("state") == "listening"


def speech_frames(seconds: float, amplitude: float = 0.5) -> list[bytes]:
    t = np.arange(int(seconds * 16000)) / 16000
    samples = (amplitude * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    return [samples[i : i + 512].tobytes() for i in range(0, len(samples), 512)]


def silence_frames(seconds: float) -> list[bytes]:
    return [np.zeros(512, dtype="<i2").tobytes() for _ in range(int(seconds * 16000 / 512))]


def test_health_and_ping(make_services) -> None:
    app = create_app(services=make_services())
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert read(ws) == {"type": "pong"}


def test_hello_is_required_first(make_services) -> None:
    app = create_app(services=make_services())
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "hej"}))
        assert read(ws) == {"type": "error", "message": "send hello first"}


def test_hello_returns_ready_and_listening(make_services) -> None:
    app = create_app(services=make_services())
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl", "client": "cli"}))
        ready = read(ws)
        assert ready["type"] == "ready"
        assert ready["tts"] == "fake"
        assert ready["sample_rate"] == 24000
        assert set(ready["langs"]) == {"pl", "en"}
        assert ready["engines"] == ["fake"]
        assert ready["engine_by_language"] == {"pl": "fake", "en": "fake"}
        assert read(ws) == {"type": "state", "state": "listening"}


def test_unknown_tts_override_is_rejected(make_services) -> None:
    app = create_app(services=make_services())
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl", "tts": "nope"}))
        assert read(ws) == {"type": "error", "message": "unknown tts engine 'nope'"}


def test_text_turn_streams_utterances_audio_and_metrics(make_services) -> None:
    llm = FakeLLM([[TextDelta("[happy] Dwa plus dwa to cztery. "), TextDelta("Miłego dnia.")]])
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "ile to dwa plus dwa"}))
        messages = read_until(ws, is_type("metrics"))

    kinds = [m["type"] if isinstance(m, dict) else "bytes" for m in messages]
    assert kinds[0] == "state"
    assert kinds.count("bytes") == 2
    utterances = [m for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert [u["text"] for u in utterances] == ["Dwa plus dwa to cztery.", "Miłego dnia."]
    assert utterances[0]["style"]["emotion"] == "happy"
    audio_starts = [m for m in messages if isinstance(m, dict) and m.get("type") == "audio_start"]
    assert [a["sample_rate"] for a in audio_starts] == [24000, 24000]
    metrics = messages[-1]
    assert metrics["turn"] == 1
    assert metrics["e2e_first_audio_ms"] is not None
    assert metrics["llm_ttft_ms"] is not None


def test_audio_turn_uses_stt_and_logs_metrics(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("Dobrze, to proste.")]])
    app = create_app(services=make_services(llm=llm, stt=FakeSTT("ile to dwa plus dwa")))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        for frame in speech_frames(0.8):
            ws.send_bytes(frame)
        for frame in silence_frames(0.8):
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))
    transcripts = [m for m in messages if isinstance(m, dict) and m.get("type") == "transcript"]
    assert transcripts == [{"type": "transcript", "text": "ile to dwa plus dwa", "lang": "pl", "final": True}]
    assert (tmp_path / "turns.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_speculative_turn_is_released_only_after_real_endpoint(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("Dobrze, to proste.")]])
    services = make_services(llm=llm, stt=FakeSTT("ile to dwa plus dwa"), cfg={"vad": {"spec_silence_ms": 200}})
    with TestClient(create_app(services=services)) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        for frame in speech_frames(0.6):
            ws.send_bytes(frame)
        for frame in silence_frames(0.8):
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))
    speech_end = messages.index({"type": "vad", "state": "speech_end"})
    transcript = next(i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("type") == "transcript")
    assert transcript > speech_end
    assert messages[-1]["committed_speculative"] is True
    assert messages[-1]["spec_ms"] is not None
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["committed_speculative"], r["spec_ms"] is not None) for r in records] == [(True, True)]


def test_resumed_speech_drops_speculation_without_sending_anything(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("Dobrze, to proste.")], [TextDelta("Ile to? Cztery.")]])
    services = make_services(llm=llm, stt=FakeSTT("ile to dwa plus dwa"), cfg={"vad": {"spec_silence_ms": 200}})
    with TestClient(create_app(services=services)) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        for frame in speech_frames(0.6) + silence_frames(0.3) + speech_frames(0.6) + silence_frames(0.8):
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))
    transcripts = [m for m in messages if isinstance(m, dict) and m.get("type") == "transcript"]
    utterances = [m for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert len(transcripts) == 1
    assert [u["text"] for u in utterances] == ["Ile to?", "Cztery."]
    assert messages[-1]["committed_speculative"] is True
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["committed_speculative"], r["spec_ms"] is not None) for r in records] == [(False, True), (True, True)]


def test_llm_outage_is_spoken_not_fatal(make_services) -> None:
    app = create_app(services=make_services(llm=FakeLLM(down=True)))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "cześć"}))
        messages = read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "ping"}))
        assert read(ws) == {"type": "pong"}
    errors = [m for m in messages if isinstance(m, dict) and m.get("type") == "error"]
    assert errors == [{"type": "error", "message": "language model unavailable"}]
    utterances = [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances == ["Model językowy jest chwilowo niedostępny."]


def test_handoff_speaks_filler_then_result(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę w Krakowie", "reason": "aktualne dane"})
    hermes = FakeHermes(result="Pogoda: dwanaście stopni.")
    app = create_app(services=make_services(llm=FakeLLM([[call]]), hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        spoken = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end" and m["id"] == 2)
    utterances = [m["text"] for m in messages + spoken if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances[0] in ACKS["pl"]
    assert utterances[-1] == "Pogoda: dwanaście stopni."
    assert hermes.tasks == ["sprawdź pogodę w Krakowie"]
    states = [m["state"] for m in messages if isinstance(m, dict) and m.get("type") == "handoff"]
    assert states == ["started", "done"]


def test_hermes_error_is_spoken_as_notice(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    hermes = FakeHermes(error="connection refused")
    app = create_app(services=make_services(llm=FakeLLM([[call]]), hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "error")
        spoken = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "metrics" and m["kind"] == "result")
    handoffs = [m for m in messages if isinstance(m, dict) and m.get("type") == "handoff"]
    assert [h["state"] for h in handoffs] == ["started", "error"]
    assert handoffs[-1]["text"] == "connection refused"
    utterances = [m["text"] for m in spoken if isinstance(m, dict) and m.get("type") == "utterance"]
    assert " ".join(utterances) == "Hermes jest teraz niedostępny, nie udało się wykonać zadania."


def test_barge_in_cancels_reply_and_closes_llm_stream(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dzień dobry, to jest bardzo długa odpowiedź. ")]], hang=True)
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        for frame in speech_frames(0.5):
            ws.send_bytes(frame)
        messages = read_until(ws, is_listening)
    assert any(isinstance(m, dict) and m.get("type") == "vad" and m["state"] == "speech_start" for m in messages)
    assert llm.closed


def test_barge_in_state_carries_reason(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dzień dobry, to jest bardzo długa odpowiedź. ")]], hang=True)
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        for frame in speech_frames(0.5):
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("vad"))
        messages += read_until(ws, is_listening)
    assert messages[-1] == {"type": "state", "state": "listening", "reason": "barge_in"}


def test_disconnect_forgets_hermes_session(make_services) -> None:
    hermes = FakeHermes()
    services = make_services(hermes=hermes)
    app = create_app(services=services)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
            session_id = read(ws)["session_id"]
        assert hermes.forgotten == [session_id]


def test_barge_in_during_hermes_task_does_not_cancel_it(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    hermes = FakeHermes(hang=True, result="Jest słonecznie.")
    llm = FakeLLM([[call], [TextDelta("Dzień dobry, to jest bardzo długa odpowiedź. ")]], hang=True)
    app = create_app(services=make_services(llm=llm, hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        started = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        job_id = started[-1]["id"]
        read_until(ws, is_type("metrics"))
        # A second, local turn is talking when the user barges in.
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        for frame in speech_frames(0.5):
            ws.send_bytes(frame)
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "state" and m.get("reason") == "barge_in")
        for frame in silence_frames(0.8):
            ws.send_bytes(frame)
        read_until(ws, is_type("transcript"))
        # The barge-in turn's reply hangs in FakeLLM; stop it so the result can be announced.
        ws.send_text(json.dumps({"type": "interrupt"}))
        ws.send_text(json.dumps({"type": "ping"}))
        read_until(ws, is_type("pong"))
        assert not hermes.cancelled
        assert hermes.release is not None
        client.portal.call(hermes.release.set)
        done = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        spoken = read_until(ws, is_type("utterance"))
    assert done[-1]["id"] == job_id
    assert spoken[-1]["text"] == "Jest słonecznie."
    assert not hermes.cancelled


def test_cancel_task_message_cancels_the_hermes_task(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    hermes = FakeHermes(hang=True)
    app = create_app(services=make_services(llm=FakeLLM([[call]]), hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        started = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        ws.send_text(json.dumps({"type": "cancel_task", "id": "nope"}))
        assert read_until(ws, is_type("error"))[-1]["message"] == "no running Hermes task 'nope'"
        ws.send_text(json.dumps({"type": "cancel_task", "id": started[-1]["id"]}))
        cancelled = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "cancelled")
    assert cancelled[-1]["id"] == started[-1]["id"]
    assert hermes.cancelled


def test_interrupt_scope_all_cancels_hermes_too(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    hermes = FakeHermes(hang=True)
    app = create_app(services=make_services(llm=FakeLLM([[call]]), hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        ws.send_text(json.dumps({"type": "interrupt", "scope": "all"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "cancelled")
    assert hermes.cancelled


def test_voice_cancel_intent_stops_hermes_and_confirms(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    hermes = FakeHermes(hang=True)
    llm = FakeLLM([[call]])
    app = create_app(services=make_services(llm=llm, hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        read_until(ws, is_type("metrics"))
        ws.send_text(json.dumps({"type": "text", "text": "Anuluj zadanie."}))
        messages = read_until(ws, is_type("metrics"))
    assert hermes.cancelled
    assert any(m.get("type") == "handoff" and m["state"] == "cancelled" for m in messages if isinstance(m, dict))
    assert [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"] == [
        "Dobrze, anulowałam zadanie Hermesa."
    ]
    assert messages[-1]["route"] == "cancel"
    assert len(llm.calls) == 1


def test_short_noise_while_speaking_does_not_barge_in(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dzień dobry, to jest bardzo długa odpowiedź. ")]], hang=True)
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        for frame in speech_frames(0.3) + silence_frames(0.8):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "ping"}))
        messages = read_until(ws, is_type("pong"))
        assert not llm.closed
    assert not any(isinstance(m, dict) and m.get("type") == "vad" for m in messages)
