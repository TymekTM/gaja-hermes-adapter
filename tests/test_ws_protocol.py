from __future__ import annotations

import asyncio
import pytest

import json
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from gaja.llm.client import TextDelta
from gaja.llm.parse import ToolCall
from gaja.pipeline import ACKS, is_cancel_intent
from gaja.server import create_app
from tests.conftest import FakeHermes, FakeLLM, FakeSTT

READ_TIMEOUT_S = 10.0


def _decode(message: Any) -> Any:
    if message.get("text") is not None:
        return json.loads(message["text"])
    return message["bytes"]


def read(ws: Any) -> Any:
    return _decode(ws.receive())


def _receive_bounded(ws: Any, timeout_s: float) -> Any:
    """One ``ws.receive()`` with a deadline.

    The TestClient websocket offers no receive timeout (starlette's ``WebSocketTestSession.receive``
    blocks on its portal forever), so the receive runs in a daemon thread whose result is awaited
    with a timeout; past the deadline the test fails instead of hanging the shared, flock-protected
    suite (r4e). The abandoned thread unblocks when the ``with`` block closes the session.
    """
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result.put((True, ws.receive()))
        except BaseException as exc:  # WebSocketDisconnect & co: re-raised in the waiter below
            result.put((False, exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        ok, payload = result.get(timeout=timeout_s)
    except queue.Empty:
        raise AssertionError(f"read_until: no message within {timeout_s:.0f}s") from None
    if ok:
        return payload
    raise payload


def read_until(
    ws: Any, done: Callable[[Any], bool], limit: int = 200, timeout_s: float = READ_TIMEOUT_S
) -> list[Any]:
    seen: list[Any] = []
    for _ in range(limit):
        message = _decode(_receive_bounded(ws, timeout_s))
        seen.append(message)
        if done(message):
            return seen
    raise AssertionError(f"no matching message within {timeout_s:.0f}s; last messages: {seen[-5:]}")


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
    assert messages[-1]["first_sentence_ms"] is not None
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        "stop", "Stop!", "stop.", "przestań", "Przestań!",
        "przestań już", "przestań teraz", "przestań proszę", "przestań to", "Stop, proszę.", "stop now",
        "stop it",
    ],
)
def test_a_stop_without_a_task_is_barge_in_not_a_cancel_intent(text: str) -> None:
    """docs/WEB_CLIENT.md: interrupting Gaja does not cancel tasks; stop cancels only when it names the task."""
    assert not is_cancel_intent(text)


@pytest.mark.parametrize(
    "text",
    [
        "anuluj zadanie", "Anuluj zadanie.", "hermes, stop", "Hermes stop", "przerwij to", "przerwij to zadanie",
        "stop the task", "przestań to zadanie", "Stop, zadanie.", "stop hermes", "stop everything",
    ],
)
def test_explicit_cancel_phrases_are_cancel_intents(text: str) -> None:
    assert is_cancel_intent(text)


def test_bare_stop_while_hermes_runs_is_barge_in_only(make_services) -> None:
    """A spoken bare "stop" must not cancel the running Hermes task (it only stops the speech).

    The turn still goes through normal routing; the task keeps running and announces its result.
    """
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    hermes = FakeHermes(hang=True)
    llm = FakeLLM([[call], [TextDelta("Nie ma sprawy.")]])
    app = create_app(services=make_services(llm=llm, hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        read_until(ws, is_type("metrics"))
        ws.send_text(json.dumps({"type": "text", "text": "stop"}))
        messages = read_until(ws, is_type("metrics"))
        # Inside the connection: close() at the end of the `with` block would cancel the job.
        assert not hermes.cancelled
    assert not any(m.get("type") == "handoff" and m.get("state") == "cancelled" for m in messages if isinstance(m, dict))
    assert messages[-1]["route"] == "local"
    assert [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"] == ["Nie ma sprawy."]


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


@pytest.mark.asyncio
async def test_websocket_outbox_serialization_and_cancellation() -> None:
    """Finding #3: _WebSocketOutbox serializes sends and does not deadlock on cancel."""
    import asyncio
    from unittest.mock import AsyncMock
    from gaja.server import _WebSocketOutbox

    mock_ws = AsyncMock()
    order = []

    async def slow_send_json(payload):
        order.append(('start', payload['id']))
        await asyncio.sleep(0.05)
        order.append(('end', payload['id']))

    mock_ws.send_json.side_effect = slow_send_json
    outbox = _WebSocketOutbox(mock_ws)

    # 1. Concurrent sends are serialized
    t1 = asyncio.create_task(outbox.send_json({'id': 1}))
    t2 = asyncio.create_task(outbox.send_json({'id': 2}))
    await asyncio.gather(t1, t2)
    assert order == [('start', 1), ('end', 1), ('start', 2), ('end', 2)]

    # 2. Cancelled send releases lock cleanly without deadlock
    order.clear()
    t3 = asyncio.create_task(outbox.send_json({'id': 3}))
    await asyncio.sleep(0.01)  # t3 starts and holds lock
    t3.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t3

    # Subsequent send acquires lock immediately without deadlock
    await outbox.send_json({'id': 4})
    assert ('start', 4) in order and ('end', 4) in order


def test_cancel_task_during_result_announcement(make_services) -> None:
    """Finding #4: Hermes job remains in self._jobs during _speak_result so cancel_task works."""
    from gaja.text.style import Style
    from gaja.tts.base import EnginePool
    import numpy as np

    from tests.conftest import FakeTTS
    class SlowResultTTS(FakeTTS):
        async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
            if "wyniki" in text:
                await asyncio.sleep(0.5)
            return np.full(2400, 0.1, dtype=np.float32)

    call = ToolCall("handoff_to_hermes", {"task": "długie zadanie", "reason": "test"})
    hermes = FakeHermes(result="Oto wyniki zadania Hermesa.")
    tts = SlowResultTTS()
    services = make_services(llm=FakeLLM([[call]]), hermes=hermes)
    services.tts = EnginePool({"fake": lambda: tts})
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "uruchom zadanie"}))
        started = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")
        job_id = started[-1]["id"]
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        # While the result is being synthesized in _speak_result, cancel it
        ws.send_text(json.dumps({"type": "cancel_task", "id": job_id}))
        cancelled = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "cancelled")
        assert cancelled[-1]["id"] == job_id


@pytest.mark.asyncio
async def test_gate_open_ordering_with_slow_outbox() -> None:
    """Finding #9: _Gate.open delivers held items in order even if new items arrive while draining."""
    from gaja.pipeline import _Gate

    delivered: list[tuple[str, Any]] = []

    class SlowOutbox:
        async def send_json(self, payload: dict[str, Any]) -> None:
            delivered.append(("json", payload["id"]))
            await asyncio.sleep(0.02)

        async def send_bytes(self, data: bytes) -> None:
            delivered.append(("bytes", data))
            await asyncio.sleep(0.02)

    gate = _Gate(SlowOutbox())
    # Send initial items while gate is closed
    await gate.send_json({"id": 1})
    await gate.send_bytes(b"frame2")

    # Start opening gate in background
    open_task = asyncio.create_task(gate.open())

    # Concurrently send items while open() is draining
    await asyncio.sleep(0.005)  # Let open() start sending item 1
    t3 = asyncio.create_task(gate.send_json({"id": 3}))
    t4 = asyncio.create_task(gate.send_bytes(b"frame4"))

    await asyncio.gather(open_task, t3, t4)
    assert delivered == [("json", 1), ("bytes", b"frame2"), ("json", 3), ("bytes", b"frame4")]


@pytest.mark.asyncio
async def test_barge_in_during_result_announcement_delays_next_result(make_services, monkeypatch) -> None:
    """Finding #10: Barge-in during a Hermes result announcement enforces quiet window before next result."""
    import gaja.pipeline as pipe
    from unittest.mock import AsyncMock
    from gaja.pipeline import Session

    monkeypatch.setattr(pipe, "RESULT_QUIET_S", 0.15)
    services = make_services()
    mock_out = AsyncMock()
    session = Session(services, mock_out)

    started_times = []
    orig_result_turn = session._result_turn

    async def tracking_result_turn(result: str, language: str, *, direct: bool = False, identity: Any = None):
        started_times.append((result, asyncio.get_event_loop().time()))
        if result == "result 1":
            await asyncio.sleep(1.0)  # simulate long speech
        else:
            await orig_result_turn(result, language, direct=direct, identity=identity)

    session._result_turn = tracking_result_turn

    # Launch result 1
    t1 = asyncio.create_task(session._speak_result("result 1", "pl", direct=True))
    await asyncio.sleep(0.02)  # allow result 1 to acquire lock and start speaking
    assert session._active is not None
    assert len(started_times) == 1 and started_times[0][0] == "result 1"

    # Queue result 2 (blocks on _result_lock)
    t2 = asyncio.create_task(session._speak_result("result 2", "pl", direct=True))
    await asyncio.sleep(0.01)

    # User interrupts via barge-in
    interrupt_time = asyncio.get_event_loop().time()
    await session._interrupt_speech(reason="barge_in")
    assert session._active is None

    # Immediately after interrupt (e.g. 0.05s later), result 2 MUST NOT have started yet
    await asyncio.sleep(0.05)
    assert len(started_times) == 1, "Result 2 started too early during quiet window"

    # After quiet window (0.15s total) has elapsed, result 2 should start
    await asyncio.sleep(0.15)
    assert len(started_times) == 2, "Result 2 did not start after quiet window elapsed"
    assert started_times[1][0] == "result 2"
    assert started_times[1][1] - interrupt_time >= 0.14

    await t1
    await t2


@pytest.mark.asyncio
async def test_speculative_handoff_flushes_gate_before_post_commit_events(make_services) -> None:
    from gaja.pipeline import Session

    class YieldingOutbox:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any] | bytes] = []

        async def send_json(self, payload: dict[str, Any]) -> None:
            await asyncio.sleep(0)
            self.sent.append(payload)

        async def send_bytes(self, data: bytes) -> None:
            await asyncio.sleep(0)
            self.sent.append(data)

    llm = FakeLLM(p_hermes=0.99)
    hermes = FakeHermes(hang=True)
    services = make_services(llm=llm, hermes=hermes, router=True)
    out = YieldingOutbox()
    session = Session(services, out)
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    out.sent.clear()

    await session._start_speculation(np.zeros(1600, dtype=np.float32))
    await asyncio.sleep(0.05)  # Speculative turn synthesizes ack into gate and waits on spec.committed
    await session._commit_speculation()
    if session._active is not None:
        await asyncio.wait_for(session._active, 5)

    json_msgs = [m for m in out.sent if isinstance(m, dict)]
    kinds = [m["type"] for m in json_msgs]
    assert kinds.index("transcript") < kinds.index("utterance") < kinds.index("audio_end") < kinds.index("metrics")
    states = [m["state"] for m in json_msgs if m["type"] == "state"]
    assert states == ["thinking", "speaking", "listening"]
    await session.close()


@pytest.mark.asyncio
async def test_committed_speculation_updates_playback_window(make_services) -> None:
    import time
    from gaja.pipeline import Session, ECHO_TAIL_S

    class CollectOutbox:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any] | bytes] = []

        async def send_json(self, payload: dict[str, Any]) -> None:
            self.sent.append(payload)

        async def send_bytes(self, data: bytes) -> None:
            self.sent.append(data)

    services = make_services(llm=FakeLLM([[TextDelta("Jest dwunasta.")]]))
    out = CollectOutbox()
    session = Session(services, out)
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))

    await session._start_speculation(np.zeros(1600, dtype=np.float32))
    assert session._spec is not None and session._spec.task is not None
    await asyncio.wait_for(session._spec.task, 5)
    assert session._playback_end == 0.0
    assert session._playback_until == 0.0

    before = time.perf_counter()
    await session._commit_speculation()
    assert session._playback_end > before
    assert session._playback_until == pytest.approx(session._playback_end + ECHO_TAIL_S)
    await session.close()


def test_new_topic_forgets_session_scoped_hermes_and_resets_person(make_services) -> None:
    hermes = FakeHermes()
    services = make_services(hermes=hermes)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        ready = read(ws)
        session_id = ready["session_id"]
        assert read(ws) == {"type": "state", "state": "listening"}
        ws.send_text(json.dumps({"type": "text", "text": "Gaja, nowy temat"}))
        read_until(ws, is_type("metrics"))
        assert hermes.resets == ["guest"]
        assert hermes.new_topics == [session_id] and hermes.forgotten == []


class _ScriptedRouterLLM(FakeLLM):
    """FakeLLM whose TOOL mass and generated call are scripted per router complete() call."""

    def __init__(self, script: list[tuple[float, str]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._script = list(script)

    async def complete(self, prompt: str, *, timeout_s: float, slot: int | None = None) -> dict[str, Any]:
        self._p_tool, self._tool_call = self._script.pop(0)
        return await super().complete(prompt, timeout_s=timeout_s, slot=slot)


def test_tool_turn_picking_the_unready_spotify_tool_hands_off_to_hermes(make_services) -> None:
    """End-to-end guard for pipeline._tool_turn's `except ToolUnavailable` (pipeline.py:1366): the
    router model can still pick the registered-but-not-ready spotify tool — nothing in
    any_plausible vetoes spotify, and another tool stays plausible via history — so the pipeline
    MUST catch the raised ToolUnavailable and hand the turn to Hermes with a spoken ack. Without
    that except (or with it narrowed away) the turn dies silently between router and reply."""
    import json as _json
    from gaja.llm.router import Router
    from gaja.tools.registry import default_registry

    llm = _ScriptedRouterLLM(
        replies=[[TextDelta("Serwer działa i ma spokojny RAM.")]],
        script=[(0.0, ""), (0.9, ' spotify(action="pause")')],
    )
    hermes = FakeHermes(result="Hermes pauzuje muzykę.")
    reg = default_registry
    services = make_services(llm=llm, hermes=hermes)
    services.tools = reg
    services.router = Router(
        llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=reg.schemas(), tool_threshold=0.5, spotify_ready=lambda: False,
    )
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(_json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        # Turn 1 is a LOCAL server-status answer; its history keeps another tool plausible for the
        # bare "pauza" of turn 2 (which matches no cue by itself).
        ws.send_text(_json.dumps({"type": "text", "text": "jaki jest uptime serwera"}))
        read_until(ws, is_listening)
        ws.send_text(_json.dumps({"type": "text", "text": "pauza"}))
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        spoken = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "utterance" and m.get("text") in ACKS["pl"] or isinstance(m, dict) and m.get("type") == "metrics")
    assert hermes.tasks == ["pauza"], messages
    utterances = [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances and utterances[0] in ACKS["pl"], messages
