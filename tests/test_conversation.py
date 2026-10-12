"""Conversation history, local tool turns and the WebSocket Origin check (round 6 fixes)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gaja.llm.client import TextDelta
from gaja.llm.parse import ToolCall
from gaja.pipeline import RESULT_PROMPT
from gaja.server import create_app, origin_allowed
from gaja.tools.base import ToolResult
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until, speech_frames


class FakeTools:
    def __init__(self, speak: str = "W Katowicach jest dwanaście stopni.", error: str | None = None) -> None:
        self.speak = speak
        self.error = error
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def schemas(self, lang: str = "pl") -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "when": {"type": "string"}}},
                },
            },
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
        ]

    async def execute(self, name: str, arguments: dict[str, Any], lang: str = "pl", timeout_s: float = 2.5) -> ToolResult:
        self.calls.append((name, arguments, lang))
        return ToolResult(data={}, speak=self.speak, error=self.error)


def hello(ws) -> None:
    ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
    read_until(ws, is_listening)


def say(ws, text: str) -> list[Any]:
    ws.send_text(json.dumps({"type": "text", "text": text}))
    return read_until(ws, is_type("metrics"))


def utterances(messages: list[Any]) -> list[str]:
    return [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]


def history_of(llm: FakeLLM, call: int) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in llm.calls[call][0] if m["role"] != "system"]


# --- history fidelity ---


def test_second_turn_sees_the_first_exchange(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")], [TextDelta("Osiem.")]])
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        say(ws, "ile to dwa plus dwa")
        say(ws, "a razy dwa?")
    assert history_of(llm, 1) == [
        ("user", "ile to dwa plus dwa"),
        ("assistant", "Dwa plus dwa to cztery."),
        ("user", "a razy dwa?"),
    ]


def test_barge_in_keeps_only_what_was_spoken(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dzień dobry, to jest bardzo długa odpowiedź. ")]], hang=True)
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_text(json.dumps({"type": "text", "text": "opowiedz coś"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end")
        for frame in speech_frames(0.6):
            ws.send_bytes(frame)
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "state" and m.get("reason") == "barge_in")
        ws.send_text(json.dumps({"type": "interrupt"}))
        ws.send_text(json.dumps({"type": "text", "text": "dobra, wystarczy"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "transcript" and m.get("text") == "dobra, wystarczy")
        ws.send_text(json.dumps({"type": "interrupt"}))
        ws.send_text(json.dumps({"type": "ping"}))
        read_until(ws, is_type("pong"))
    turns = history_of(llm, 1)
    assert turns[0] == ("user", "opowiedz coś")
    role, content = turns[1]
    assert role == "assistant" and content.endswith("…")
    assert content.startswith("Dzień dobry")
    assert turns[-1] == ("user", "dobra, wystarczy")


def test_handoff_result_is_in_history(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "aktualne dane"})
    llm = FakeLLM([[call], [TextDelta("Proszę bardzo.")]])
    hermes = FakeHermes(result="Jest słonecznie.")
    app = create_app(services=make_services(llm=llm, hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_text(json.dumps({"type": "text", "text": "jaka jest pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "metrics" and m.get("kind") == "result")
        say(ws, "dzięki")
    turns = history_of(llm, 1)
    assert turns[0] == ("user", "jaka jest pogoda")
    assert turns[1][0] == "assistant"  # the spoken acknowledgement
    assert ("user", RESULT_PROMPT.format(result="Jest słonecznie.")) in turns
    assert ("assistant", "Jest słonecznie.") in turns
    assert turns[-1] == ("user", "dzięki")
    assert [r for r, _ in turns] == ["user", "assistant", "user", "assistant", "user"]


def test_hermes_gets_the_conversation_context(make_services) -> None:
    call = ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę tam", "reason": "aktualne dane"})
    llm = FakeLLM([[TextDelta("Gdańsk to piękne miasto.")], [call]])
    hermes = FakeHermes(result="Jest słonecznie.")
    app = create_app(services=make_services(llm=llm, hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        say(ws, "opowiedz o Gdańsku")
        ws.send_text(json.dumps({"type": "text", "text": "a jaka jest tam pogoda"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
    assert hermes.languages == ["pl"]
    assert [m["content"] for m in hermes.contexts[0]] == ["opowiedz o Gdańsku", "Gdańsk to piękne miasto."]


# --- local tools (variant C) ---


def test_tool_turn_speaks_the_tool_text_and_records_it(make_services) -> None:
    tools = FakeTools()
    llm = FakeLLM([[TextDelta("Nie ma za co.")]], p_tool=0.9, tool_call=' get_weather(when="now")')
    app = create_app(services=make_services(llm=llm, router=True, tools=tools))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        messages = say(ws, "jaka jest pogoda")
        llm._p_tool = 0.0
        say(ws, "dzięki")
    event = next(m for m in messages if isinstance(m, dict) and m.get("type") == "tool")
    assert event == {
        "type": "tool",
        "name": "get_weather",
        "arguments": {"when": "now", "location": "Katowice"},
        "ok": True,
        "text": "W Katowicach jest dwanaście stopni.",
    }
    assert tools.calls == [("get_weather", {"when": "now", "location": "Katowice"}, "pl")]
    assert utterances(messages) == ["W Katowicach jest dwanaście stopni."]
    assert messages[-1]["route"] == "tool" and messages[-1]["tool"] == "get_weather"
    assert messages[-1]["tool_ms"] is not None
    assert len(llm.calls) == 1  # the tool turn itself used no chat completion
    assert history_of(llm, 0)[:2] == [("user", "jaka jest pogoda"), ("assistant", "W Katowicach jest dwanaście stopni.")]


def test_tool_error_is_spoken_and_flagged(make_services) -> None:
    tools = FakeTools(speak="Nie udało się sprawdzić pogody.", error="timeout")
    llm = FakeLLM(p_tool=0.9, tool_call=' get_weather(location="Kraków")')
    app = create_app(services=make_services(llm=llm, router=True, tools=tools))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        messages = say(ws, "pogoda w Krakowie")
    event = next(m for m in messages if isinstance(m, dict) and m.get("type") == "tool")
    assert event["ok"] is False
    assert utterances(messages) == ["Nie udało się sprawdzić pogody."]


def test_unparseable_tool_call_falls_back_to_a_local_reply(make_services) -> None:
    tools = FakeTools()
    llm = FakeLLM([[TextDelta("Nie wiem.")]], p_tool=0.9, tool_call="nonsense")
    app = create_app(services=make_services(llm=llm, router=True, tools=tools))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        messages = say(ws, "coś dziwnego")
    assert tools.calls == []
    assert utterances(messages) == ["Nie wiem."]
    assert messages[-1]["route"] == "local"


# --- Origin check ---


def test_origin_allowed() -> None:
    allowed = ["http://localhost:8790", "https://example-node.tailscale.example.net:8443"]
    assert origin_allowed(None, allowed)
    assert origin_allowed("http://localhost:8790", allowed)
    assert origin_allowed("HTTPS://example-node.tailscale.example.net:8443/", allowed)
    assert not origin_allowed("http://evil.example", allowed)
    assert not origin_allowed("http://localhost:8791", allowed)
    assert not origin_allowed("null", allowed)


def test_foreign_origin_is_rejected(make_services) -> None:
    app = create_app(services=make_services())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
                ws.receive_text()
        assert exc.value.code == 1008
        with client.websocket_connect("/ws", headers={"origin": "http://localhost:8790"}) as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert read_until(ws, is_type("pong"))
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert read_until(ws, is_type("pong"))


def test_static_files_do_not_escape_the_web_dir(make_services) -> None:
    app = create_app(services=make_services())
    with TestClient(app) as client:
        for path in ("/../config.toml", "/%2e%2e/config.toml", "/..%2fconfig.toml", "/%2e%2e%2fpyproject.toml"):
            response = client.get(path)
            assert response.status_code == 404 or "[server]" not in response.text, path
            assert "[project]" not in response.text, path


# --- speculative endpointing has no side effects until commit ---

SPEC = {"vad": {"spec_silence_ms": 200}}


def resumed_speech() -> list[bytes]:
    from tests.test_ws_protocol import silence_frames

    return speech_frames(0.6) + silence_frames(0.3) + speech_frames(0.6) + silence_frames(0.8)


def test_dropped_speculation_starts_no_hermes_task(make_services) -> None:
    from tests.conftest import FakeSTT

    llm = FakeLLM(p_hermes=0.99)
    hermes = FakeHermes(result="Gotowe.")
    services = make_services(llm=llm, hermes=hermes, router=True, stt=FakeSTT("wyślij maila do Kasi"), cfg=SPEC)
    with TestClient(create_app(services=services)) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        for frame in resumed_speech():
            ws.send_bytes(frame)
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
    assert hermes.tasks == ["wyślij maila do Kasi"]


def test_dropped_speculation_runs_no_tool_and_writes_no_history(make_services) -> None:
    from tests.conftest import FakeSTT

    tools = FakeTools()
    llm = FakeLLM([[TextDelta("Proszę.")]], p_tool=0.9, tool_call=' get_time()')
    services = make_services(llm=llm, router=True, tools=tools, stt=FakeSTT("która godzina"), cfg=SPEC)
    with TestClient(create_app(services=services)) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        for frame in resumed_speech():
            ws.send_bytes(frame)
        messages = read_until(ws, is_type("metrics"))
        llm._p_tool = 0.0
        say(ws, "dzięki")
    assert messages[-1]["committed_speculative"] is True
    assert tools.calls in ([("get_time", {}, "pl")], [("get_time", {"kind": "time"}, "pl")])
    assert history_of(llm, 0) == [
        ("user", "która godzina"),
        ("assistant", "W Katowicach jest dwanaście stopni."),
        ("user", "dzięki"),
    ]


# --- TTS normalizer hook ---


def test_normalizer_runs_after_tag_parsing_and_only_feeds_tts() -> None:
    from gaja.pipeline import _to_sentences
    from gaja.text.style import Style

    seen: list[tuple[str, str]] = []

    def normalize(text: str, lang: str) -> str:
        seen.append((text, lang))
        return text.replace("15%", "piętnaście procent")

    sentences, _ = _to_sentences(["[happy] To jest 15% więcej."], Style(), auto_emotion=False, language="pl", normalize=normalize)
    assert seen == [("To jest 15% więcej.", "pl")]
    assert sentences[0].text == "To jest 15% więcej."
    assert sentences[0].speech == "To jest piętnaście procent więcej."


def test_normalizer_failure_speaks_the_raw_text() -> None:
    from gaja.pipeline import _to_sentences
    from gaja.text.style import Style

    def broken(text: str, lang: str) -> str:
        raise ValueError("boom")

    sentences, _ = _to_sentences(["Dzień dobry."], Style(), auto_emotion=False, normalize=broken)
    assert sentences[0].text == "Dzień dobry." and sentences[0].speech is None


def test_real_normalizer_is_importable() -> None:
    from gaja.server import _load_normalizer

    normalize = _load_normalizer()
    assert normalize is not None
    assert isinstance(normalize("Mam 3 koty.", "pl"), str)


# --- session preload ---


def test_hello_preloads_both_slots_once(make_services) -> None:
    llm = FakeLLM([[TextDelta("Cześć.")]])
    app = create_app(services=make_services(llm=llm, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        say(ws, "cześć")
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))  # a re-hello does not preload again
        read_until(ws, is_listening)
    session_start = [m for m in llm.chat_primes if len(m) == 1]
    assert len(session_start) == 1 and session_start[0][0]["role"] == "system"
    assert llm.primed and llm.primed[0].startswith("<|im_start|>system\nrouter-sys")


def test_router_prime_after_a_turn_covers_the_next_classify_prompt(make_services) -> None:
    llm = FakeLLM([[TextDelta("Pierwsza odpowiedź.")], [TextDelta("Druga odpowiedź.")]])
    app = create_app(services=make_services(llm=llm, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        say(ws, "pierwsze pytanie")
        say(ws, "drugie pytanie")
    second = next(p for p in llm.prompts if "Zapytanie: drugie pytanie" in p)
    with_history = [p for p in llm.primed if "Zapytanie: pierwsze pytanie" in p]
    assert with_history, llm.primed
    assert all(second.startswith(p) for p in with_history)


def test_chat_prime_after_a_turn_is_the_next_requests_history(make_services) -> None:
    llm = FakeLLM([[TextDelta("Pierwsza odpowiedź.")], [TextDelta("Druga odpowiedź.")]])
    app = create_app(services=make_services(llm=llm))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        say(ws, "pierwsze pytanie")
        say(ws, "drugie pytanie")
    second = llm.calls[1][0]
    after_first = [m for m in llm.chat_primes if any(x.get("content") == "Pierwsza odpowiedź." for x in m)]
    assert after_first, llm.chat_primes
    assert second[: len(after_first[0])] == after_first[0]
    assert second[len(after_first[0]) :] == [{"role": "user", "content": "drugie pytanie"}]


def test_chat_prime_after_turn_can_be_disabled(make_services) -> None:
    llm = FakeLLM([[TextDelta("Odpowiedź.")]])
    app = create_app(services=make_services(llm=llm, cfg={"llm": {"prime_after_turn": False}}))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        say(ws, "pytanie")
    assert all(len(m) == 1 for m in llm.chat_primes)


def test_preload_with_llm_down_still_serves_the_session(make_services) -> None:
    llm = FakeLLM(down=True)
    app = create_app(services=make_services(llm=llm, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        messages = say(ws, "cześć")
    assert messages[-1]["type"] == "metrics"


@pytest.mark.asyncio
async def test_playback_three_sentences_only_adds_one_echo_tail(make_services) -> None:
    """Finding #1: 3 synthesized sentences only add ECHO_TAIL_S once, not 3 times."""
    import time
    import numpy as np
    from unittest.mock import AsyncMock
    from gaja.metrics import TurnMetrics
    from gaja.pipeline import Session, _Turn, ECHO_TAIL_S
    from gaja.pipeline import Sentence
    from gaja.text.style import Style
    from tests.conftest import FakeTTS
    from gaja.tts.base import EnginePool

    class CustomDurationTTS(FakeTTS):
        async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
            return np.full(12000, 0.1, dtype=np.float32)  # 0.5 s at 24000 Hz

    services = make_services()
    tts = CustomDurationTTS()
    services.tts = EnginePool({"fake": lambda: tts})
    outbox = AsyncMock()
    session = Session(services, outbox)
    turn = _Turn(t0=time.perf_counter(), language="pl", metrics=TurnMetrics(turn=1), out=outbox)

    t0 = time.perf_counter()
    for text in ["Pierwsze zdanie.", "Drugie zdanie.", "Trzecie zdanie."]:
        await session._play(turn, Sentence(text, Style()))

    # Each sentence is 0.5 s, total 1.5 s
    expected_audio_duration = 3 * (12000 / 24000)
    assert session._playback_end == pytest.approx(t0 + expected_audio_duration, abs=0.1)
    # The echo tail must only be added ONCE to the final playback end
    assert session._playback_until == pytest.approx(session._playback_end + ECHO_TAIL_S, abs=0.01)
    assert session._playback_until - t0 == pytest.approx(expected_audio_duration + ECHO_TAIL_S, abs=0.1)
